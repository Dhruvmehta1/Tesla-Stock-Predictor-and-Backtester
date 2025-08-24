import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
import os
import json
import warnings

class DeterministicModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self._set_all_seeds()

    def _set_all_seeds(self):
        """Set all possible random seeds for complete determinism"""
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        os.environ['PYTHONHASHSEED'] = str(self.random_state)

        # Set environment variables for deterministic behavior
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        # Configure Optuna for deterministic behavior
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _create_deterministic_sampler(self):
        """Create a deterministic Optuna sampler"""
        return optuna.samplers.TPESampler(
            seed=self.random_state,
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,
            group=True,
            warn_independent_sampling=False,
            constant_liar=True
        )

    def _get_deterministic_cv(self, n_splits=5):
        """Create deterministic cross-validation splits"""
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state
        )

    def _sharpe_ratio(self, preds, y_true, close_prices):
        """Calculate Sharpe ratio with proper error handling"""
        if len(preds) <= 1 or len(set(preds)) <= 1:
            return 0.0

        returns = []
        capital = 10000

        try:
            for i in range(len(preds)-1):
                position = 1 if preds[i] == 1 else 0

                if hasattr(close_prices, 'iloc'):
                    if i >= len(close_prices) or i+1 >= len(close_prices):
                        continue
                    price_today = close_prices.iloc[i]
                    price_tomorrow = close_prices.iloc[i+1]
                else:
                    price_today = close_prices[i]
                    price_tomorrow = close_prices[i+1]

                if price_today > 0:
                    daily_return = position * (price_tomorrow / price_today - 1)
                    returns.append(daily_return)

        except Exception:
            return 0.0

        if not returns or np.std(returns) <= 0:
            return 0.0

        return (np.mean(returns) / np.std(returns)) * np.sqrt(252)

    def _load_or_create_params(self, model_name, best_params_path):
        """Load cached parameters or return None to trigger optimization"""
        if os.path.exists(best_params_path):
            try:
                with open(best_params_path, "r") as f:
                    best_params = json.load(f)
                return best_params.get(model_name)
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def _save_params(self, model_name, params, best_params_path):
        """Save optimized parameters"""
        os.makedirs(os.path.dirname(best_params_path), exist_ok=True)

        if os.path.exists(best_params_path):
            try:
                with open(best_params_path, "r") as f:
                    best_params = json.load(f)
            except (json.JSONDecodeError, KeyError):
                best_params = {}
        else:
            best_params = {}

        best_params[model_name] = params

        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)

    def _optimize_random_forest(self, X_train, y_train, close_prices_val):
        """Optimize Random Forest hyperparameters"""
        def objective(trial):
            self._set_all_seeds()  # Reset seeds for each trial

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'bootstrap': True,
                'oob_score': True,
                'random_state': self.random_state,
                'n_jobs': 1
            }

            cv = self._get_deterministic_cv()
            cv_scores = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Create price data for CV fold
                cv_prices = pd.Series(index=X_cv_val.index, dtype=float)
                if 'Close' in X_train.columns:
                    train_prices = X_train['Close'].copy()
                    cv_prices = train_prices.iloc[val_idx]
                elif isinstance(close_prices_val, pd.Series):
                    for idx in X_cv_val.index:
                        if idx in close_prices_val.index:
                            cv_prices[idx] = close_prices_val[idx]

                model = RandomForestClassifier(**params)
                model.fit(X_cv_train, y_cv_train)
                preds = model.predict(X_cv_val)

                # Calculate combined score
                f1 = f1_score(y_cv_val, preds, average='macro')
                sharpe = self._sharpe_ratio(preds, y_cv_val, cv_prices)
                combined_score = 0.3 * f1 * 2 + 0.7 * sharpe
                cv_scores.append(combined_score)

            return np.mean(cv_scores)

        sampler = self._create_deterministic_sampler()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"rf_optimization_{self.random_state}"
        )
        study.optimize(objective, n_trials=25, show_progress_bar=True)
        return study.best_params

    def _optimize_gradient_boosting(self, X_train, y_train, close_prices_val):
        """Optimize Gradient Boosting hyperparameters"""
        def objective(trial):
            self._set_all_seeds()

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'validation_fraction': 0.2,
                'n_iter_no_change': 10,
                'tol': 0.001,
                'random_state': self.random_state
            }

            cv = self._get_deterministic_cv()
            cv_scores = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                cv_prices = pd.Series(index=X_cv_val.index, dtype=float)
                if 'Close' in X_train.columns:
                    train_prices = X_train['Close'].copy()
                    cv_prices = train_prices.iloc[val_idx]
                elif isinstance(close_prices_val, pd.Series):
                    for idx in X_cv_val.index:
                        if idx in close_prices_val.index:
                            cv_prices[idx] = close_prices_val[idx]

                model = GradientBoostingClassifier(**params)
                model.fit(X_cv_train, y_cv_train)
                preds = model.predict(X_cv_val)

                f1 = f1_score(y_cv_val, preds, average='macro')
                sharpe = self._sharpe_ratio(preds, y_cv_val, cv_prices)
                combined_score = 0.3 * f1 * 2 + 0.7 * sharpe
                cv_scores.append(combined_score)

            return np.mean(cv_scores)

        sampler = self._create_deterministic_sampler()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"gb_optimization_{self.random_state}"
        )
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        return study.best_params

    def _optimize_logistic_regression(self, X_train, y_train, close_prices_val):
        """Optimize Logistic Regression hyperparameters"""
        def objective(trial):
            self._set_all_seeds()

            C = trial.suggest_float('C', 0.01, 10.0, log=True)
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])

            if penalty == 'elasticnet':
                solver = 'saga'
                l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
            else:
                solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
                l1_ratio = None

            # Validate solver-penalty combinations
            if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                return -9999
            if penalty == 'elasticnet' and solver != 'saga':
                return -9999

            params = {
                'C': C,
                'penalty': penalty,
                'solver': solver,
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'max_iter': trial.suggest_int('max_iter', 500, 2000),
                'random_state': self.random_state,
                'n_jobs': 1
            }

            if l1_ratio is not None:
                params['l1_ratio'] = l1_ratio

            try:
                cv = self._get_deterministic_cv()
                cv_scores = []

                for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    cv_prices = pd.Series(index=X_cv_val.index, dtype=float)
                    if 'Close' in X_train.columns:
                        train_prices = X_train['Close'].copy()
                        cv_prices = train_prices.iloc[val_idx]
                    elif isinstance(close_prices_val, pd.Series):
                        for idx in X_cv_val.index:
                            if idx in close_prices_val.index:
                                cv_prices[idx] = close_prices_val[idx]

                    model = LogisticRegression(**params)
                    model.fit(X_cv_train, y_cv_train)
                    preds = model.predict(X_cv_val)

                    # Skip if predictions are all the same
                    if len(set(preds)) <= 1:
                        cv_scores.append(-9999)
                        continue

                    f1 = f1_score(y_cv_val, preds, average='macro')
                    sharpe = self._sharpe_ratio(preds, y_cv_val, cv_prices)
                    combined_score = 0.3 * f1 * 2 + 0.7 * sharpe
                    cv_scores.append(combined_score)

                return np.mean(cv_scores) if cv_scores and np.mean(cv_scores) > -1000 else -9999

            except Exception:
                return -9999

        sampler = self._create_deterministic_sampler()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"lr_optimization_{self.random_state}"
        )
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        return study.best_params

    def _optimize_decision_tree(self, X_train, y_train, close_prices_val):
        """Optimize Decision Tree hyperparameters"""
        def objective(trial):
            self._set_all_seeds()

            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.01),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
                'random_state': self.random_state
            }

            try:
                cv = self._get_deterministic_cv()
                cv_scores = []

                for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    cv_prices = pd.Series(index=X_cv_val.index, dtype=float)
                    if 'Close' in X_train.columns:
                        train_prices = X_train['Close'].copy()
                        cv_prices = train_prices.iloc[val_idx]
                    elif isinstance(close_prices_val, pd.Series):
                        for idx in X_cv_val.index:
                            if idx in close_prices_val.index:
                                cv_prices[idx] = close_prices_val[idx]

                    model = DecisionTreeClassifier(**params)
                    model.fit(X_cv_train, y_cv_train)
                    preds = model.predict(X_cv_val)

                    if len(set(preds)) <= 1:
                        cv_scores.append(-9999)
                        continue

                    f1 = f1_score(y_cv_val, preds, average='macro')
                    sharpe = self._sharpe_ratio(preds, y_cv_val, cv_prices)
                    combined_score = 0.3 * f1 * 2 + 0.7 * sharpe
                    cv_scores.append(combined_score)

                return np.mean(cv_scores) if cv_scores and np.mean(cv_scores) > -1000 else -9999

            except Exception:
                return -9999

        sampler = self._create_deterministic_sampler()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"dt_optimization_{self.random_state}"
        )
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        return study.best_params

    def _optimize_lightgbm(self, X_train, y_train, close_prices_val):
        """Optimize LightGBM hyperparameters"""
        def objective(trial):
            self._set_all_seeds()

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'random_state': self.random_state,
                'n_jobs': 1,
                'verbose': -1,
                'deterministic': True,
                'force_col_wise': True,
                'feature_name': 'auto'
            }

            try:
                cv = self._get_deterministic_cv()
                cv_scores = []

                for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    cv_prices = pd.Series(index=X_cv_val.index, dtype=float)
                    if 'Close' in X_train.columns:
                        train_prices = X_train['Close'].copy()
                        cv_prices = train_prices.iloc[val_idx]
                    elif isinstance(close_prices_val, pd.Series):
                        for idx in X_cv_val.index:
                            if idx in close_prices_val.index:
                                cv_prices[idx] = close_prices_val[idx]

                    # Split training data for early stopping
                    from sklearn.model_selection import train_test_split
                    X_t, X_v, y_t, y_v = train_test_split(
                        X_cv_train, y_cv_train, test_size=0.2, random_state=self.random_state
                    )

                    model = lgb.LGBMClassifier(**params)
                    model.fit(
                        X_t, y_t,
                        eval_set=[(X_v, y_v)],
                        eval_metric='logloss',
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                    )
                    preds = model.predict(X_cv_val)

                    f1 = f1_score(y_cv_val, preds, average='macro')
                    sharpe = self._sharpe_ratio(preds, y_cv_val, cv_prices)
                    combined_score = 0.3 * f1 * 2 + 0.7 * sharpe
                    cv_scores.append(combined_score)

                return np.mean(cv_scores)

            except Exception:
                return -9999

        sampler = self._create_deterministic_sampler()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"lgb_optimization_{self.random_state}"
        )
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        return study.best_params

    def train_models(self, X_train, y_train, X_val, y_val, close_prices_val):
        """Train all models with deterministic Optuna optimization"""
        self._set_all_seeds()

        warnings.filterwarnings("ignore", category=UserWarning)

        # Create paths
        os.makedirs("tesla_stock_predictor/models", exist_ok=True)
        best_params_path = "tesla_stock_predictor/models/best_params.json"

        print("Training models with deterministic Optuna optimization...")

        # --- Random Forest ---
        print("Training Random Forest...")
        rf_params = self._load_or_create_params('rf', best_params_path)
        if rf_params is None:
            print("  Optimizing Random Forest hyperparameters...")
            rf_params = self._optimize_random_forest(X_train, y_train, close_prices_val)
            self._save_params('rf', rf_params, best_params_path)
        else:
            print("  Using cached Random Forest parameters")

        rf_params.update({
            'bootstrap': True,
            'oob_score': True,
            'random_state': self.random_state,
            'n_jobs': 1
        })

        self.models['rf'] = RandomForestClassifier(**rf_params)
        self.models['rf'].fit(X_train, y_train)
        print(f"  OOB Score: {self.models['rf'].oob_score_:.4f}")

        # --- Gradient Boosting ---
        print("Training Gradient Boosting...")
        gb_params = self._load_or_create_params('gb', best_params_path)
        if gb_params is None:
            print("  Optimizing Gradient Boosting hyperparameters...")
            gb_params = self._optimize_gradient_boosting(X_train, y_train, close_prices_val)
            self._save_params('gb', gb_params, best_params_path)
        else:
            print("  Using cached Gradient Boosting parameters")

        gb_params.update({
            'random_state': self.random_state
        })

        gb_model = GradientBoostingClassifier(**gb_params)
        gb_model.fit(X_train, y_train)

        # Apply calibration to GB
        gb_calibrated = CalibratedClassifierCV(gb_model, method='sigmoid', cv=3)
        gb_calibrated.fit(X_val, y_val)
        self.models['gb'] = gb_calibrated
        print("  Gradient Boosting calibrated")

        # --- Logistic Regression ---
        print("Training Logistic Regression...")
        lr_params = self._load_or_create_params('lr', best_params_path)
        if lr_params is None:
            print("  Optimizing Logistic Regression hyperparameters...")
            lr_params = self._optimize_logistic_regression(X_train, y_train, close_prices_val)
            self._save_params('lr', lr_params, best_params_path)
        else:
            print("  Using cached Logistic Regression parameters")

        lr_params.update({
            'random_state': self.random_state,
            'n_jobs': 1
        })

        self.models['lr'] = LogisticRegression(**lr_params)
        self.models['lr'].fit(X_train, y_train)
        print("  Logistic Regression trained")

        # --- Decision Tree ---
        print("Training Decision Tree...")
        dt_params = self._load_or_create_params('dt', best_params_path)
        if dt_params is None:
            print("  Optimizing Decision Tree hyperparameters...")
            dt_params = self._optimize_decision_tree(X_train, y_train, close_prices_val)
            self._save_params('dt', dt_params, best_params_path)
        else:
            print("  Using cached Decision Tree parameters")

        dt_params.update({
            'random_state': self.random_state
        })

        dt_model = DecisionTreeClassifier(**dt_params)
        dt_model.fit(X_train, y_train)

        # Apply calibration to DT
        dt_calibrated = CalibratedClassifierCV(dt_model, method='isotonic', cv=3)
        dt_calibrated.fit(X_val, y_val)
        self.models['dt'] = dt_calibrated
        print("  Decision Tree calibrated")

        # --- LightGBM ---
        print("Training LightGBM...")
        lgb_params = self._load_or_create_params('lgb', best_params_path)
        if lgb_params is None:
            print("  Optimizing LightGBM hyperparameters...")
            lgb_params = self._optimize_lightgbm(X_train, y_train, close_prices_val)
            self._save_params('lgb', lgb_params, best_params_path)
        else:
            print("  Using cached LightGBM parameters")

        lgb_params.update({
            'random_state': self.random_state,
            'n_jobs': 1,
            'verbose': -1,
            'deterministic': True,
            'force_col_wise': True,
            'feature_name': 'auto'
        })

        # Split data for LightGBM early stopping
        from sklearn.model_selection import train_test_split
        X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )

        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_train_lgb, y_train_lgb,
            eval_set=[(X_val_lgb, y_val_lgb)],
            eval_metric='logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        # Apply calibration to LightGBM
        lgb_calibrated = CalibratedClassifierCV(lgb_model, method='sigmoid', cv=3)
        lgb_calibrated.fit(X_val, y_val)
        self.models['lgb'] = lgb_calibrated
        print("  LightGBM calibrated")

        print("All models trained successfully!")

        # Print final parameter summary
        print("\nFinal optimized parameters saved to:", best_params_path)
        return self.models
