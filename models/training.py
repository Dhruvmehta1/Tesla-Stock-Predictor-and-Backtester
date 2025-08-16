
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb
import optuna

# Suppress Optuna info/warning output globally
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelTrainer:
    def __init__(self):
        self.models = {}

    def train_models(self, X_train, y_train, X_val, y_val, close_prices_val):
        """
        Train all models with Optuna hyperparameter search, caching best params for each model in best_params.json.
        If best params exist for a model, use them and skip Optuna.
        """

        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        import os
        import json
        best_params_path = "tesla_stock_predictor/models/best_params.json"
        if os.path.exists(best_params_path):
            with open(best_params_path, "r") as f:
                best_params = json.load(f)
        else:
            best_params = {}

        # Helper for Sharpe ratio
        def sharpe_ratio(preds, y_true, close_prices):
            returns = []
            capital = 10000
            position = 0
            portfolio = [capital]
            for i in range(len(preds)-1):
                if preds[i] == 1:
                    position = 1
                else:
                    position = 0
                daily_return = position * (close_prices.iloc[i+1] / close_prices.iloc[i] - 1)
                capital *= (1 + daily_return)
                returns.append(daily_return)
                portfolio.append(capital)
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
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=1
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                return sharpe_ratio(preds, y_val, close_prices_val)
            rf_sampler = optuna.samplers.TPESampler(seed=42)
            rf_study = optuna.create_study(direction="maximize", sampler=rf_sampler)
            rf_study.optimize(rf_objective, n_trials=20, show_progress_bar=True)
            rf_best_params = rf_study.best_params
            best_params['rf'] = rf_best_params
            with open(best_params_path, "w") as f:
                json.dump(best_params, f)
        rf_best = RandomForestClassifier(**rf_best_params, random_state=42, n_jobs=1)
        rf_best.fit(X_train, y_train)
        self.models['rf'] = rf_best

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
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                return sharpe_ratio(preds, y_val, close_prices_val)
            gb_sampler = optuna.samplers.TPESampler(seed=42)
            gb_study = optuna.create_study(direction="maximize", sampler=gb_sampler)
            gb_study.optimize(gb_objective, n_trials=20, show_progress_bar=False)
            gb_best_params = gb_study.best_params
            best_params['gb'] = gb_best_params
            with open(best_params_path, "w") as f:
                json.dump(best_params, f)
        gb_best = GradientBoostingClassifier(**gb_best_params, random_state=42)
        gb_best.fit(X_train, y_train)
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
                    model = LogisticRegression(
                        C=C,
                        penalty=penalty,
                        solver=solver,
                        class_weight=class_weight,
                        max_iter=max_iter,
                        random_state=42,
                        n_jobs=1
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    # If all preds are the same, penalize
                    if np.all(preds == preds[0]):
                        return -9999
                    return sharpe_ratio(preds, y_val, close_prices_val)
                except Exception:
                    return -9999
            lr_sampler = optuna.samplers.TPESampler(seed=42)
            lr_study = optuna.create_study(direction="maximize", sampler=lr_sampler)
            lr_study.optimize(lr_objective, n_trials=20, show_progress_bar=False)
            lr_best_params = lr_study.best_params
            best_params['lr'] = lr_best_params
            with open(best_params_path, "w") as f:
                json.dump(best_params, f)
        lr_best = LogisticRegression(**lr_best_params, random_state=42, n_jobs=1)
        lr_best.fit(X_train, y_train)
        self.models['lr'] = lr_best

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
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        ccp_alpha=ccp_alpha,
                        class_weight=class_weight,
                        splitter=splitter,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    # If all preds are the same, penalize
                    if np.all(preds == preds[0]):
                        return -9999
                    return sharpe_ratio(preds, y_val, close_prices_val)
                except Exception:
                    return -9999
            dt_sampler = optuna.samplers.TPESampler(seed=42)
            dt_study = optuna.create_study(direction="maximize", sampler=dt_sampler)
            dt_study.optimize(dt_objective, n_trials=20, show_progress_bar=False)
            dt_best_params = dt_study.best_params
            best_params['dt'] = dt_best_params
            with open(best_params_path, "w") as f:
                json.dump(best_params, f)
        dt_best = DecisionTreeClassifier(**dt_best_params, random_state=42)
        dt_best.fit(X_train, y_train)
        # Only calibrate if tree predicts more than one class
        val_preds = dt_best.predict(X_val)
        if len(np.unique(val_preds)) > 1:
            try:
                dt_calibrated = CalibratedClassifierCV(dt_best, method='isotonic', cv=3)
                dt_calibrated.fit(X_train, y_train)
                self.models['dt'] = dt_calibrated
            except Exception:
                self.models['dt'] = dt_best
        else:
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
                model = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    min_child_samples=min_child_samples,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=1,
                    verbose=-1
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                return sharpe_ratio(preds, y_val, close_prices_val)
            lgb_sampler = optuna.samplers.TPESampler(seed=42)
            lgb_study = optuna.create_study(direction="maximize", sampler=lgb_sampler)
            lgb_study.optimize(lgb_objective, n_trials=20, show_progress_bar=False)
            lgb_best_params = lgb_study.best_params
            best_params['lgb'] = lgb_best_params
            with open(best_params_path, "w") as f:
                json.dump(best_params, f)
        lgb_best = lgb.LGBMClassifier(**lgb_best_params, random_state=42, n_jobs=1, verbose=-1)
        lgb_best.fit(X_train, y_train)
        # Calibrate LightGBM probabilities
        lgb_calibrated = CalibratedClassifierCV(lgb_best, method='sigmoid', cv=3)
        lgb_calibrated.fit(X_train, y_train)
        self.models['lgb'] = lgb_calibrated
