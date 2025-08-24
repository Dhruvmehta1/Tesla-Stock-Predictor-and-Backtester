
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit

# Suppress Optuna info/warning output globally
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelTrainer:
    def __init__(self):
        self.models = {}

    def train_with_time_cv(self, X_train, y_train, X_val, y_val, close_prices_val):
        """Train with time series cross-validation to prevent look-ahead bias"""
        # This method is a placeholder that is not currently used.
        # We've simplified to use standard validation in all model objective functions.
        print("Note: Using simplified validation instead of TimeSeriesSplit")
        pass

    def train_models(self, X_train, y_train, X_val, y_val, close_prices_val):
        """
        Train all models with Optuna hyperparameter search, caching best params for each model in best_params.json.
        If best params exist for a model, use them and skip Optuna.
        Implements safeguards against overfitting.
        """

        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        import os
        import json
        from sklearn.model_selection import StratifiedKFold

        # Check for class imbalance and apply appropriate measures
        class_counts = np.bincount(y_train)
        class_ratio = min(class_counts) / max(class_counts) if max(class_counts) > 0 else 0

        # Create paths if needed
        os.makedirs("tesla_stock_predictor/models", exist_ok=True)
        best_params_path = "tesla_stock_predictor/models/best_params.json"
        if os.path.exists(best_params_path):
            with open(best_params_path, "r") as f:
                try:
                    best_params = json.load(f)
                except json.JSONDecodeError:
                    print("Error loading best params, creating new")
                    best_params = {}
        else:
            best_params = {}

        # Helper for Sharpe ratio
        def sharpe_ratio(preds, y_true, close_prices):
            returns = []
            capital = 10000
            position = 0
            portfolio = [capital]

            # Safety check for length
            if len(preds) <= 1:
                return 0.0

            # Safely handle different types of close_prices
            for i in range(len(preds)-1):
                if preds[i] == 1:
                    position = 1
                else:
                    position = 0

                # Handle different types of close_prices
                try:
                    if hasattr(close_prices, 'iloc') and hasattr(close_prices, 'index'):
                        # DataFrame or Series with Index
                        if i >= len(close_prices) or i+1 >= len(close_prices):
                            # Skip if indices are out of bounds
                            continue
                        price_today = close_prices.iloc[i]
                        price_tomorrow = close_prices.iloc[i+1]
                    elif isinstance(close_prices, np.ndarray):
                        # Numpy array
                        price_today = close_prices[i]
                        price_tomorrow = close_prices[i+1]
                    else:
                        # Unknown, try direct indexing
                        price_today = close_prices[i]
                        price_tomorrow = close_prices[i+1]

                    daily_return = position * (price_tomorrow / price_today - 1)
                    capital *= (1 + daily_return)
                    returns.append(daily_return)
                    portfolio.append(capital)
                except Exception as e:
                    print(f"Error in sharpe calculation: {e}")
                    continue

            avg_return = np.mean(returns) if returns else 0
            std_return = np.std(returns) if returns else 0
            sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            return sharpe

        # --- Random Forest: Use Optuna only if not cached ---
        if 'rf' in best_params:
            rf_best_params = best_params['rf']
        else:
            def rf_objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                max_depth = trial.suggest_int('max_depth', 3, 15)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                class_weight = trial.suggest_categorical('class_weight', ['balanced', None])

                # Use cross-validation for more robust evaluation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_sharpes = []
                cv_f1s = []

                for train_idx, test_idx in cv.split(X_train, y_train):
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

                    # For CV, we need to use price data from the training dataset
                    # We'll create a price Series with the same index as X_cv_val
                    train_prices = pd.Series(index=X_train.index)

                    # Copy the original close prices to the train_prices Series
                    if 'Close' in X_train.columns:
                        train_prices = X_train['Close'].copy()
                    elif isinstance(close_prices_val, pd.Series):
                        # Try to get prices from the validation prices that match training indices
                        for idx in X_train.index:
                            if idx in close_prices_val.index:
                                train_prices[idx] = close_prices_val[idx]

                    # Get just the prices for the CV test fold
                    cv_prices = train_prices.iloc[test_idx]

                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        class_weight=class_weight,
                        random_state=42,
                        n_jobs=1,
                        oob_score=True  # Use out-of-bag score to reduce overfitting
                    )
                    model.fit(X_cv_train, y_cv_train)
                    preds = model.predict(X_cv_val)

                    # Calculate metrics
                    try:
                        fold_sharpe = sharpe_ratio(preds, y_cv_val, cv_prices)
                    except Exception as e:
                        print(f"Warning: Sharpe calculation failed: {e}")
                        fold_sharpe = 0  # Default value if calculation fails

                    from sklearn.metrics import f1_score
                    fold_f1 = f1_score(y_cv_val, preds, average='macro')

                    cv_sharpes.append(fold_sharpe)
                    cv_f1s.append(fold_f1)

                # Combine financial and ML metrics for more robust model selection
                avg_sharpe = np.mean(cv_sharpes)
                avg_f1 = np.mean(cv_f1s)

                # Balance between financial performance and ML metrics
                # This helps avoid models that overfit to financial metrics
                combined_score = (0.7 * avg_sharpe) + (0.3 * avg_f1 * 2)

                return combined_score
            rf_sampler = optuna.samplers.TPESampler(seed=42)
            rf_study = optuna.create_study(direction="maximize", sampler=rf_sampler)
            rf_study.optimize(rf_objective, n_trials=20, show_progress_bar=True)
            rf_best_params = rf_study.best_params
            best_params['rf'] = rf_best_params
            with open(best_params_path, "w") as f:
                json.dump(best_params, f)
        # Add class_weight if not already set
        if 'class_weight' not in rf_best_params:
            rf_best_params['class_weight'] = 'balanced'

        # Add bootstrap and oob_score for better generalization
        rf_best_params['bootstrap'] = True
        rf_best_params['oob_score'] = True

        rf_best = RandomForestClassifier(**rf_best_params, random_state=42, n_jobs=1)
        rf_best.fit(X_train, y_train)
        self.models['rf'] = rf_best

        # Print out-of-bag score
        # Feature importance is stored in the model but not printed

        # --- Gradient Boosting: Use Optuna only if not cached ---
        if 'gb' in best_params:
            gb_best_params = best_params['gb']
        else:
            def gb_objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 100, 300)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
                max_depth = trial.suggest_int('max_depth', 3, 8)
                subsample = trial.suggest_float('subsample', 0.7, 1.0)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
                max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

                # Use cross-validation for more robust evaluation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_sharpes = []
                cv_f1s = []

                for train_idx, test_idx in cv.split(X_train, y_train):
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

                    # For CV, we need to use price data from the training dataset
                    # We'll create a price Series with the same index as X_cv_val
                    train_prices = pd.Series(index=X_train.index)

                    # Copy the original close prices to the train_prices Series
                    if 'Close' in X_train.columns:
                        train_prices = X_train['Close'].copy()
                    elif isinstance(close_prices_val, pd.Series):
                        # Try to get prices from the validation prices that match training indices
                        for idx in X_train.index:
                            if idx in close_prices_val.index:
                                train_prices[idx] = close_prices_val[idx]

                    # Get just the prices for the CV test fold
                    cv_prices = train_prices.iloc[test_idx]

                    # Early stopping to prevent overfitting
                    from sklearn.model_selection import train_test_split
                    X_t, X_v, y_t, y_v = train_test_split(X_cv_train, y_cv_train, test_size=0.2, random_state=42)

                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        subsample=subsample,  # Use subsample to reduce overfitting
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        validation_fraction=0.2,
                        n_iter_no_change=10,  # Early stopping
                        tol=0.001,
                        random_state=42
                    )
                    model.fit(X_t, y_t)
                    preds = model.predict(X_cv_val)

                    # Calculate metrics
                    try:
                        fold_sharpe = sharpe_ratio(preds, y_cv_val, cv_prices)
                    except Exception as e:
                        print(f"Warning: Sharpe calculation failed: {e}")
                        fold_sharpe = 0  # Default value if calculation fails

                    from sklearn.metrics import f1_score
                    fold_f1 = f1_score(y_cv_val, preds, average='macro')

                    cv_sharpes.append(fold_sharpe)
                    cv_f1s.append(fold_f1)

                # Balance between financial performance and ML metrics
                avg_sharpe = np.mean(cv_sharpes)
                avg_f1 = np.mean(cv_f1s)
                combined_score = (0.7 * avg_sharpe) + (0.3 * avg_f1 * 2)

                return combined_score
            gb_sampler = optuna.samplers.TPESampler(seed=42)
            gb_study = optuna.create_study(direction="maximize", sampler=gb_sampler)
            gb_study.optimize(gb_objective, n_trials=20, show_progress_bar=False)
            gb_best_params = gb_study.best_params
            best_params['gb'] = gb_best_params
            with open(best_params_path, "w") as f:
                json.dump(best_params, f)
        # Add validation set for early stopping
        from sklearn.model_selection import train_test_split
        X_train_gb, X_val_gb, y_train_gb, y_val_gb = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Add validation parameters for early stopping
        # These parameters need to be included during model creation, not during fit
        if 'validation_fraction' not in gb_best_params:
            gb_best_params['validation_fraction'] = 0.2
            gb_best_params['n_iter_no_change'] = 10
            gb_best_params['tol'] = 0.001

        # Make sure we're not passing eval_set which is not supported
        if 'eval_set' in gb_best_params:
            del gb_best_params['eval_set']

        # Remove parameters not supported by sklearn's GradientBoostingClassifier
        # and ensure validation_fraction is included in the params, not as a separate fit argument
        gb_params = {k: v for k, v in gb_best_params.items() if k not in ['eval_set']}

        # Create and fit multiple GB models with different configurations for better discrimination

        # Base model with balanced parameters
        gb_params.update({
            'subsample': 0.8,
            'max_features': 'sqrt',
            'min_samples_leaf': 10,
            'min_samples_split': 20
        })

        gb_models = []

        # Train base model
        gb_best = GradientBoostingClassifier(**gb_params, random_state=42)
        gb_best.fit(X_train_gb, y_train_gb)
        gb_models.append(("Standard", gb_best))

        # Train a model optimized for discrimination (more complex)
        try:
            gb_params_complex = gb_params.copy()
            gb_params_complex.update({
                'min_samples_leaf': 5,
                'min_samples_split': 10,
                'max_depth': min(10, gb_params.get('max_depth', 6))
            })
            gb_complex = GradientBoostingClassifier(**gb_params_complex, random_state=42)
            gb_complex.fit(X_train_gb, y_train_gb)
            gb_models.append(("Complex", gb_complex))
        except Exception as e:
            pass  # Failed to train complex GB model

        # Train a model with quantile loss for different probability distribution
        try:
            gb_params_quantile = gb_params.copy()
            gb_params_quantile.update({
                'loss': 'quantile',
                'alpha': 0.5  # median quantile
            })
            gb_quantile = GradientBoostingClassifier(**gb_params_quantile, random_state=42)
            gb_quantile.fit(X_train_gb, y_train_gb)
            gb_models.append(("Quantile", gb_quantile))
        except Exception as e:
            pass  # Failed to train quantile GB model

        # Calibrate and evaluate each model
        from sklearn.metrics import brier_score_loss
        calibrated_models = []

        for model_name, model in gb_models:
            for method in ['isotonic', 'sigmoid']:
                try:
                    # Calibrate the model
                    gb_calibrated = CalibratedClassifierCV(model, method=method, cv=3)
                    gb_calibrated.fit(X_val, y_val)

                    # Get probabilities
                    val_probs = gb_calibrated.predict_proba(X_val)[:, 1]

                    # Evaluate calibration
                    brier_score = brier_score_loss(y_val, val_probs)

                    # Evaluate discrimination
                    unique_probs = np.unique(val_probs)
                    unique_count = len(unique_probs)

                    # Combined score (emphasize discrimination)
                    combined_score = unique_count - (5 * brier_score)

                    calibrated_models.append((gb_calibrated, combined_score, method, model_name))
                except Exception as e:
                    pass  # GB calibration failed

        # Select the best model
        if calibrated_models:
            calibrated_models.sort(key=lambda x: x[1], reverse=True)
            best_model, best_score, best_method, model_name = calibrated_models[0]
            self.models['gb'] = best_model
        else:
            # Fallback to basic model with sigmoid calibration
            try:
                gb_calibrated = CalibratedClassifierCV(gb_best, method='sigmoid', cv=3)
                gb_calibrated.fit(X_val, y_val)
                self.models['gb'] = gb_calibrated
            except Exception as e:
                self.models['gb'] = gb_best

        # --- Logistic Regression: Use Optuna only if not cached ---
        if 'lr' in best_params:
            lr_best_params = best_params['lr']
        else:
            def lr_objective(trial):
                C = trial.suggest_float('C', 0.01, 10.0, log=True)
                penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
                solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
                class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
                max_iter = trial.suggest_int('max_iter', 500, 2000)

                # Only allow valid solver/penalty combos
                valid = True
                if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                    valid = False
                if penalty == 'l2' and solver not in ['liblinear', 'saga']:
                    valid = False
                if not valid:
                    return -9999  # Skip invalid combos

                try:
                    # Use cross-validation for more robust evaluation
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_sharpes = []
                    cv_f1s = []

                    for train_idx, test_idx in cv.split(X_train, y_train):
                        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
                        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

                        # For CV, we need to use price data from the training dataset
                        # We'll create a price Series with the same index as X_cv_val
                        train_prices = pd.Series(index=X_train.index)

                        # Copy the original close prices to the train_prices Series
                        if 'Close' in X_train.columns:
                            train_prices = X_train['Close'].copy()
                        elif isinstance(close_prices_val, pd.Series):
                            # Try to get prices from the validation prices that match training indices
                            for idx in X_train.index:
                                if idx in close_prices_val.index:
                                    train_prices[idx] = close_prices_val[idx]

                        # Get just the prices for the CV test fold
                        cv_prices = train_prices.iloc[test_idx]

                        # Use L1/L2 regularization to prevent overfitting
                        model = LogisticRegression(
                            C=C,  # Smaller C means stronger regularization
                            penalty=penalty,
                            solver=solver,
                            class_weight=class_weight,
                            max_iter=max_iter,
                            random_state=42,
                            n_jobs=1
                        )
                        model.fit(X_cv_train, y_cv_train)
                        preds = model.predict(X_cv_val)

                        # If all preds are the same, penalize this fold
                        if np.all(preds == preds[0]):
                            cv_sharpes.append(-9999)
                            cv_f1s.append(0)
                            continue

                        # Calculate metrics
                        try:
                            fold_sharpe = sharpe_ratio(preds, y_cv_val, cv_prices)
                        except Exception as e:
                            print(f"Warning: Sharpe calculation failed: {e}")
                            fold_sharpe = 0  # Default value if calculation fails

                        from sklearn.metrics import f1_score
                        fold_f1 = f1_score(y_cv_val, preds, average='macro')

                        cv_sharpes.append(fold_sharpe)
                        cv_f1s.append(fold_f1)

                    # Balance between financial performance and ML metrics
                    avg_sharpe = np.mean(cv_sharpes)
                    avg_f1 = np.mean(cv_f1s)

                    # If all folds failed, return a penalty
                    if avg_sharpe < -1000:
                        return -9999

                    combined_score = (0.7 * avg_sharpe) + (0.3 * avg_f1 * 2)
                    return combined_score
                except Exception:
                    return -9999
            lr_sampler = optuna.samplers.TPESampler(seed=42)
            lr_study = optuna.create_study(direction="maximize", sampler=lr_sampler)
            lr_study.optimize(lr_objective, n_trials=20, show_progress_bar=False)
            lr_best_params = lr_study.best_params
            best_params['lr'] = lr_best_params
            with open(best_params_path, "w") as f:
                json.dump(best_params, f)
        # Add L1 and L2 regularization if not already specified
        if 'penalty' not in lr_best_params:
            lr_best_params['penalty'] = 'elasticnet'
            lr_best_params['solver'] = 'saga'  # Only saga supports elasticnet
            lr_best_params['l1_ratio'] = 0.5  # Mix of L1 and L2

        lr_best = LogisticRegression(**lr_best_params, random_state=42, n_jobs=1)
        lr_best.fit(X_train, y_train)
        self.models['lr'] = lr_best

        # Print coefficient information
        # LogisticRegression - Feature importance is stored in the model but not printed

        # --- Decision Tree: Use Optuna only if not cached ---
        if 'dt' in best_params:
            dt_best_params = best_params['dt']
        else:
            def dt_objective(trial):
                max_depth = trial.suggest_int('max_depth', 3, 20)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.01)
                class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
                splitter = trial.suggest_categorical('splitter', ['random'])

                try:
                    # Use cross-validation for more robust evaluation
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_sharpes = []
                    cv_f1s = []

                    for train_idx, test_idx in cv.split(X_train, y_train):
                        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
                        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

                        # For CV, we need to use price data from the training dataset
                        # We'll create a price Series with the same index as X_cv_val
                        train_prices = pd.Series(index=X_train.index)

                        # Copy the original close prices to the train_prices Series
                        if 'Close' in X_train.columns:
                            train_prices = X_train['Close'].copy()
                        elif isinstance(close_prices_val, pd.Series):
                            # Try to get prices from the validation prices that match training indices
                            for idx in X_train.index:
                                if idx in close_prices_val.index:
                                    train_prices[idx] = close_prices_val[idx]

                        # Get just the prices for the CV test fold
                        cv_prices = train_prices.iloc[test_idx]

                        # Cost-complexity pruning to prevent overfitting
                        model = DecisionTreeClassifier(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            ccp_alpha=ccp_alpha,  # Cost complexity pruning
                            class_weight=class_weight,
                            splitter=splitter,
                            random_state=42
                        )
                        model.fit(X_cv_train, y_cv_train)
                        preds = model.predict(X_cv_val)

                        # If all preds are the same, penalize this fold
                        if np.all(preds == preds[0]):
                            cv_sharpes.append(-9999)
                            cv_f1s.append(0)
                            continue

                        # Calculate metrics
                        try:
                            fold_sharpe = sharpe_ratio(preds, y_cv_val, cv_prices)
                        except Exception as e:
                            print(f"Warning: Sharpe calculation failed: {e}")
                            fold_sharpe = 0  # Default value if calculation fails

                        from sklearn.metrics import f1_score
                        fold_f1 = f1_score(y_cv_val, preds, average='macro')

                        cv_sharpes.append(fold_sharpe)
                        cv_f1s.append(fold_f1)

                    # Balance between financial performance and ML metrics
                    avg_sharpe = np.mean(cv_sharpes) if cv_sharpes else -9999
                    avg_f1 = np.mean(cv_f1s) if cv_f1s else 0

                    # Penalize if all folds failed
                    if avg_sharpe < -1000:
                        return -9999

                    combined_score = (0.7 * avg_sharpe) + (0.3 * avg_f1 * 2)
                    return combined_score
                except Exception:
                    return -9999
            dt_sampler = optuna.samplers.TPESampler(seed=42)
            dt_study = optuna.create_study(direction="maximize", sampler=dt_sampler)
            dt_study.optimize(dt_objective, n_trials=20, show_progress_bar=False)
            dt_best_params = dt_study.best_params
            best_params['dt'] = dt_best_params
            with open(best_params_path, "w") as f:
                json.dump(best_params, f)
        # Add cross-validation to find optimal ccp_alpha for pruning
        from sklearn.model_selection import GridSearchCV

        # Only if ccp_alpha isn't already set or is too small
        if 'ccp_alpha' not in dt_best_params or dt_best_params['ccp_alpha'] < 0.001:
            # Determine cost-complexity pruning parameter
            param_grid = {'ccp_alpha': np.linspace(0.001, 0.03, 10)}
            grid_search = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='f1_macro'
            )
            grid_search.fit(X_train, y_train)
            dt_best_params['ccp_alpha'] = grid_search.best_params_['ccp_alpha']
            print(f"Selected optimal ccp_alpha: {dt_best_params['ccp_alpha']:.4f}")

        # Check if class_weight is already in params
        if 'class_weight' not in dt_best_params:
            dt_best_params['class_weight'] = 'balanced'

        # Create and train base decision tree model with good depth for discrimination
        dt_best = DecisionTreeClassifier(**dt_best_params, random_state=42)
        dt_best.fit(X_train, y_train)

        # Create a calibrated tree ensemble that maintains discriminative power
        try:
            # Create multiple trees with different depths for better discrimination
            base_models = []
            depths = [8, 10, 12]  # Use deeper trees for better discrimination
            min_samples = [5, 10, 15]  # Different sample sizes for diversity

            # Create an ensemble of different trees
            for depth, min_sample in zip(depths, min_samples):
                tree = DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_leaf=min_sample,
                    class_weight='balanced',
                    random_state=42
                )
                base_models.append(tree)

            # Add the original tree to maintain its discriminative features
            base_models.append(dt_best)

            # Create calibration for each tree and test discrimination
            from sklearn.metrics import brier_score_loss
            from scipy.stats import entropy

            calibrated_models = []

            for i, base_model in enumerate(base_models):
                base_model.fit(X_train, y_train)

                # Try both calibration methods
                for method in ['isotonic', 'sigmoid']:
                    try:
                        dt_calibrated = CalibratedClassifierCV(
                            base_model,
                            method=method,
                            cv=3,             # Use fewer folds to prevent overfitting
                            ensemble=True     # Use ensemble of calibrated classifiers
                        )
                        dt_calibrated.fit(X_val, y_val)

                        # Get probabilities
                        val_probs = dt_calibrated.predict_proba(X_val)[:, 1]

                        # Calculate calibration score
                        brier_score = brier_score_loss(y_val, val_probs)

                        # Calculate discrimination score (using probability entropy)
                        unique_probs = np.unique(val_probs)
                        prob_entropy = len(unique_probs)  # Higher means more unique probabilities

                        # Calculate combined score (balance calibration and discrimination)
                        combined_score = prob_entropy - (5 * brier_score)  # Favor discrimination

                        calibrated_models.append((dt_calibrated, combined_score, method, i))
                    except Exception as e:
                        pass  # Tree calibration failed

            # Select the model with the best combined score
            if calibrated_models:
                # Sort by combined score (higher is better)
                calibrated_models.sort(key=lambda x: x[1], reverse=True)
                best_model, best_score, best_method, best_idx = calibrated_models[0]
                self.models['dt'] = best_model
            else:
                # Use original tree with basic calibration
                dt_calibrated = CalibratedClassifierCV(dt_best, method='sigmoid', cv=3)
                dt_calibrated.fit(X_val, y_val)
                self.models['dt'] = dt_calibrated
        except Exception as e:
            # Use uncalibrated model as fallback
            self.models['dt'] = dt_best

        # --- LightGBM: Use Optuna only if not cached ---
        if 'lgb' in best_params:
            lgb_best_params = best_params['lgb']
        else:
            def lgb_objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 100, 500)
                max_depth = trial.suggest_int('max_depth', 3, 15)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                subsample = trial.suggest_float('subsample', 0.5, 1.0)
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
                reg_alpha = trial.suggest_float('reg_alpha', 0, 5)
                reg_lambda = trial.suggest_float('reg_lambda', 0, 5)
                min_child_samples = trial.suggest_int('min_child_samples', 5, 30)
                scale_pos_weight = trial.suggest_float('scale_pos_weight', 0.5, 5)

                # Use cross-validation for more robust evaluation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_sharpes = []
                cv_f1s = []

                for train_idx, test_idx in cv.split(X_train, y_train):
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

                    # For CV, we need to use price data from the training dataset
                    # We'll create a price Series with the same index as X_cv_val
                    train_prices = pd.Series(index=X_train.index)

                    # Copy the original close prices to the train_prices Series
                    if 'Close' in X_train.columns:
                        train_prices = X_train['Close'].copy()
                    elif isinstance(close_prices_val, pd.Series):
                        # Try to get prices from the validation prices that match training indices
                        for idx in X_train.index:
                            if idx in close_prices_val.index:
                                train_prices[idx] = close_prices_val[idx]

                    # Get just the prices for the CV test fold
                    cv_prices = train_prices.iloc[test_idx]

                    # Create a validation set for early stopping
                    from sklearn.model_selection import train_test_split
                    X_t, X_v, y_t, y_v = train_test_split(X_cv_train, y_cv_train, test_size=0.2, random_state=42)
                    eval_set = [(X_v, y_v)]

                    model = lgb.LGBMClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        reg_alpha=reg_alpha,  # L1 regularization
                        reg_lambda=reg_lambda,  # L2 regularization
                        min_child_samples=min_child_samples,
                        scale_pos_weight=scale_pos_weight,  # Already handles class imbalance
                        random_state=42,
                        n_jobs=1,
                        verbose=-1,
                        early_stopping_rounds=50,  # Early stopping to prevent overfitting
                        feature_name='auto'  # Ensure proper feature naming
                    )
                    model.fit(
                        X_t, y_t,
                        eval_set=eval_set,
                        eval_metric='logloss',
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                    )
                    preds = model.predict(X_cv_val)

                    # Calculate metrics
                    try:
                        fold_sharpe = sharpe_ratio(preds, y_cv_val, cv_prices)
                    except Exception as e:
                        print(f"Warning: Sharpe calculation failed: {e}")
                        fold_sharpe = 0  # Default value if calculation fails

                    from sklearn.metrics import f1_score
                    fold_f1 = f1_score(y_cv_val, preds, average='macro')

                    cv_sharpes.append(fold_sharpe)
                    cv_f1s.append(fold_f1)

                # Balance between financial performance and ML metrics
                avg_sharpe = np.mean(cv_sharpes)
                avg_f1 = np.mean(cv_f1s)
                combined_score = (0.7 * avg_sharpe) + (0.3 * avg_f1 * 2)

                return combined_score
            lgb_sampler = optuna.samplers.TPESampler(seed=42)
            lgb_study = optuna.create_study(direction="maximize", sampler=lgb_sampler)
            lgb_study.optimize(lgb_objective, n_trials=20, show_progress_bar=False)
            lgb_best_params = lgb_study.best_params
            best_params['lgb'] = lgb_best_params
            with open(best_params_path, "w") as f:
                json.dump(best_params, f)
        # Add early stopping for LightGBM
        from sklearn.model_selection import train_test_split
        X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        eval_set = [(X_val_lgb, y_val_lgb)]

        # Enhance LightGBM parameters for better discrimination
        lgb_params = {
            **lgb_best_params,
            'random_state': 42,
            'n_jobs': 1,
            'verbose': -1,
            'feature_name': 'auto',
            'scale_pos_weight': 2.0,  # Weight positive class more heavily (don't use is_unbalance with this)
            'num_leaves': 63,  # Increase leaves for better discrimination
            'max_depth': 8,  # Deeper tree for more detailed patterns
            'min_data_in_leaf': 10,  # Smaller leaf size for more detailed patterns
            'feature_fraction': 0.8,  # Use subset of features for each tree
            'bagging_fraction': 0.8,  # Use subset of data for each tree
            'bagging_freq': 1  # Perform bagging at every iteration
        }

        # Create a diverse set of LightGBM models with different configurations
        lgb_models = []

        # Train the main model
        lgb_best = lgb.LGBMClassifier(**lgb_params)
        lgb_best.fit(
            X_train_lgb, y_train_lgb,
            eval_set=eval_set,
            eval_metric='logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        lgb_models.append(lgb_best)

        # Train a more aggressive model that might overfit but has better discrimination
        try:
            # Create a more aggressive config (make a copy to avoid modifying the original)
            aggressive_params = lgb_params.copy()
            aggressive_params.update({
                'num_leaves': 127,
                'max_depth': 12,
                'min_data_in_leaf': 5
            })
            lgb_aggressive = lgb.LGBMClassifier(**aggressive_params)
            lgb_aggressive.fit(
                X_train_lgb, y_train_lgb,
                eval_set=eval_set,
                eval_metric='logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            lgb_models.append(lgb_aggressive)
        except Exception as e:
            print(f"  Failed to train aggressive LightGBM: {e}")

        # Train a more conservative model for better generalization
        try:
            # Create a more conservative config (make a copy to avoid modifying the original)
            conservative_params = lgb_params.copy()
            conservative_params.update({
                'num_leaves': 31,
                'max_depth': 6,
                'min_data_in_leaf': 20
            })
            lgb_conservative = lgb.LGBMClassifier(**conservative_params)
            lgb_conservative.fit(
                X_train_lgb, y_train_lgb,
                eval_set=eval_set,
                eval_metric='logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            lgb_models.append(lgb_conservative)
        except Exception as e:
            print(f"  Failed to train conservative LightGBM: {e}")

        # Calibrate each model and evaluate discrimination
        calibrated_models = []
        from sklearn.metrics import brier_score_loss

        for i, model in enumerate(lgb_models):
            for method in ['isotonic', 'sigmoid']:
                try:
                    lgb_calibrated = CalibratedClassifierCV(model, method=method, cv=3)
                    lgb_calibrated.fit(X_val, y_val)

                    # Get probabilities
                    val_probs = lgb_calibrated.predict_proba(X_val)[:, 1]

                    # Calculate calibration score
                    brier_score = brier_score_loss(y_val, val_probs)

                    # Calculate discrimination score (using probability uniqueness)
                    unique_probs = np.unique(val_probs)
                    prob_entropy = len(unique_probs)

                    # Combined score (emphasize discrimination)
                    combined_score = prob_entropy - (5 * brier_score)

                    model_type = ["Best", "Aggressive", "Conservative"][i] if i < 3 else f"Model {i}"

                    calibrated_models.append((lgb_calibrated, combined_score, method, model_type))
                except Exception as e:
                    pass  # LightGBM calibration failed

        # Select best model based on combined score
        if calibrated_models:
            calibrated_models.sort(key=lambda x: x[1], reverse=True)
            best_model, best_score, best_method, model_type = calibrated_models[0]
            self.models['lgb'] = best_model
        else:
            # Fallback to basic calibration
            lgb_calibrated = CalibratedClassifierCV(lgb_best, method='sigmoid', cv=3)
            lgb_calibrated.fit(X_val, y_val)
            self.models['lgb'] = lgb_calibrated
