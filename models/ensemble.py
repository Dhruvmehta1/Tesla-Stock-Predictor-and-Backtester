import numpy as np
import pandas as pd
from datetime import timedelta
import warnings

def ensemble_predict(
    self,
    X,
    model_list=None,
    weights=None,
    threshold=None,
    model_thresholds=None
):
    """
    Unified ensemble logic: model selection, weights, and thresholds are all handled here.

    Args:
        X: Feature matrix.
        model_list: List of model names to include in the ensemble. If None, use all in self.models.
        weights: Dict of model weights (need not sum to 1.0; will be normalized). If None, equal weights.
        threshold: Float, threshold for ensemble prediction. If None, use 0.5.
        model_thresholds: Dict of per-model thresholds. If None, defaults to 0.5 for all.

    Returns:
        ensemble_pred: Final ensemble predictions.
        ensemble_prob: Ensemble probabilities.
        confidence: Confidence scores.
        predictions: Individual model predictions.
        probabilities: Individual model probabilities.
    """
    # Determine which models to use
    if model_list is None:
        model_list = list(self.models.keys())

    # Default per-model thresholds
    if model_thresholds is None:
        model_thresholds = {k: 0.5 for k in model_list}

    # Default to equal weights if not provided
    if weights is None:
        weights = {k: 1.0 for k in model_list}
    # Normalize weights
    total_weight = sum(weights.get(k, 0) for k in model_list)
    if total_weight == 0:
        raise ValueError("At least one model must have nonzero weight.")
    weights = {k: weights.get(k, 0) / total_weight for k in model_list}

    # Default ensemble threshold
    if threshold is None:
        threshold = 0.5

    predictions = {}
    probabilities = {}

    for name in model_list:
        if name not in self.models:
            continue

        # Add defensive checks for feature count mismatch
        try:
            model = self.models[name]

            # Check for feature count consistency
            if hasattr(model, 'n_features_in_'):
                if model.n_features_in_ != X.shape[1]:
                    print(f"WARNING: Model {name} expects {model.n_features_in_} features but got {X.shape[1]}. Skipping.")
                    continue

            # Check for feature names consistency
            if hasattr(model, 'feature_names_in_') and hasattr(X, 'columns'):
                if not all(feat in X.columns for feat in model.feature_names_in_):
                    print(f"WARNING: Model {name} has feature name mismatch. Skipping.")
                    # Show some details about the mismatch
                    missing_features = [f for f in model.feature_names_in_ if f not in X.columns]
                    if missing_features:
                        print(f"  Missing features (first 5): {missing_features[:5]}")
                    continue

            # If X is a numpy array (not DataFrame), we can't check feature names
            # but we can still try prediction as long as feature count matches
            if isinstance(X, np.ndarray) and hasattr(model, 'n_features_in_'):
                if model.n_features_in_ != X.shape[1]:
                    print(f"WARNING: Model {name} expects {model.n_features_in_} features but numpy array has {X.shape[1]}. Skipping.")
                    continue

            prob = model.predict_proba(X)[:, 1]
            probabilities[name] = prob
            pred_thresh = model_thresholds.get(name, 0.5)
            predictions[name] = (prob > pred_thresh).astype(int)
        except Exception as e:
            print(f"Error using model {name}: {e}. Skipping.")
            continue

    # Defensive check: ensure at least one model produced probabilities
    if not probabilities:
        raise ValueError("No models produced probabilities. Check that models are trained and present in self.models.")

    # Weighted ensemble probability (dynamic, only selected models)
    ensemble_prob = np.zeros_like(next(iter(probabilities.values())))
    for name in model_list:
        if name not in probabilities:
            continue
        ensemble_prob += probabilities[name] * weights.get(name, 0)

    # Ensure we have at least one valid model before proceeding
    if np.all(ensemble_prob == 0) and len(probabilities) == 0:
        print("WARNING: All models were skipped due to errors or feature mismatches. Returning default prediction.")
        return np.zeros(len(X), dtype=int), np.zeros(len(X), dtype=float), np.zeros(len(X), dtype=float), {}, {}

    ensemble_pred = (ensemble_prob > threshold).astype(int)

    # Confidence based on agreement between models
    if predictions:
        agreement = np.mean(list(predictions.values()), axis=0)
        confidence = np.abs(agreement - 0.5) * 2  # Scale to 0-1
    else:
        confidence = np.zeros_like(ensemble_pred, dtype=float)

    return ensemble_pred, ensemble_prob, confidence, predictions, probabilities

import itertools

def grid_search_model_thresholds(
    predictor,
    X_val_scaled,
    y_val,
    df_clean,
    X_val,  # <-- Add this parameter
    model_list=None,
    weights=None,
    threshold_grid=None,
    ensemble_threshold_grid=None
):
    """
    Grid search with cross-validation to avoid overfitting
    Args:
        predictor: TSLAPredictor instance
        X_val_scaled: Validation features (scaled)
        y_val: Validation targets
        df_clean: Cleaned dataframe with 'Close' prices
        model_list: List of model names to include in the ensemble
        weights: Dict of model weights
        threshold_grid: List of threshold values to try for individual models (e.g., [0.4, 0.5, 0.6])
        ensemble_threshold_grid: List of ensemble threshold values to try (e.g., [0.4, 0.5, 0.6])
    """
    from sklearn.model_selection import KFold
    import numpy as np

    if model_list is None:
        model_list = ['rf', 'lr', 'dt', 'lgb', 'gb']
    if weights is None:
        weights = {k: 1 for k in model_list}
    if threshold_grid is None:
        threshold_grid = [0.4, 0.5, 0.6]
    if ensemble_threshold_grid is None:
        ensemble_threshold_grid = [0.4, 0.5, 0.6]

    # Use KFold to create multiple validation sets
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    threshold_combinations = list(itertools.product(threshold_grid, repeat=len(model_list)))
    ensemble_results = []

    print(f"Running cross-validated grid search for {len(threshold_combinations) * len(ensemble_threshold_grid)} combinations...")

    for ensemble_thresh in ensemble_threshold_grid:
        for combo in threshold_combinations:
            model_thresholds = dict(zip(model_list, combo))
            cv_sharpes = []
            cv_returns = []
            cv_drawdowns = []

            try:
                # Convert X_val_scaled to numpy array if it's a DataFrame
                X_val_scaled_arr = X_val_scaled.values if hasattr(X_val_scaled, 'values') else X_val_scaled

                # Use cross-validation to evaluate this configuration
                for train_idx, val_idx in kf.split(X_val_scaled_arr):
                    try:
                        X_cv_val = X_val_scaled_arr[val_idx]

                        # Get predictions for this fold
                        val_pred, _, _, _, _ = predictor.ensemble_predict(
                            X_cv_val,
                            model_list=model_list,
                            weights=weights,
                            threshold=ensemble_thresh,
                            model_thresholds=model_thresholds
                        )

                        # Calculate sharpe ratio for this fold
                        from tesla_stock_predictor.main import backtest_financial_metrics
                        # Get the corresponding index for this validation fold
                        val_dates = X_val.index[val_idx]
                        sharpe, total_return, max_drawdown, _ = backtest_financial_metrics(val_pred, val_dates, df_clean)
                        cv_sharpes.append(sharpe)
                        cv_returns.append(total_return)
                        cv_drawdowns.append(max_drawdown)
                    except Exception as e:
                        print(f"Error in fold: {e}")
                        continue
            except Exception as e:
                print(f"Error in cross-validation: {e}")
                # Fallback: use direct evaluation
                val_pred, _, _, _, _ = predictor.ensemble_predict(
                    X_val_scaled,
                    model_list=model_list,
                    weights=weights,
                    threshold=ensemble_thresh,
                    model_thresholds=model_thresholds
                )
                from tesla_stock_predictor.main import backtest_financial_metrics
                sharpe, total_return, max_drawdown, _ = backtest_financial_metrics(val_pred, X_val.index, df_clean)
                cv_sharpes = [sharpe]
                cv_returns = [total_return]
                cv_drawdowns = [max_drawdown]

            # Average metrics across folds
            mean_sharpe = np.mean(cv_sharpes) if cv_sharpes else 0.0
            mean_return = np.mean(cv_returns) if cv_returns else 0.0
            mean_drawdown = np.mean(cv_drawdowns) if cv_drawdowns else 1.0

            ensemble_results.append({
                "model_thresholds": model_thresholds,
                "ensemble_threshold": ensemble_thresh,
                "sharpe": mean_sharpe,
                "total_return": mean_return,
                "max_drawdown": mean_drawdown,
                "weights": weights
            })

    # Defensive patch: always return a config dict, even if all Sharpe ratios are zero or results is empty
    if not ensemble_results:
        print("Warning: No valid ensemble configurations found. Using default settings.")
        default_config = {
            "model_thresholds": {name: threshold_grid[0] for name in model_list},
            "ensemble_threshold": ensemble_threshold_grid[0],
            "sharpe": 0,
            "total_return": 0,
            "max_drawdown": 0,
            "weights": weights
        }
        return default_config

    # Sort results by Sharpe ratio (descending)
    ensemble_results.sort(key=lambda x: x["sharpe"], reverse=True)

    # Return best configuration
    best = ensemble_results[0]
    print(f"Best ensemble config found with Sharpe: {best['sharpe']:.4f}, Return: {best['total_return']:.4f}")
    return best

from tesla_stock_predictor.features.engineering import generate_features_for_next_day

def predict_tomorrow(self, df):
    """Generate tomorrow's prediction with detailed analysis"""
    # Find the next business day (or trading day)
    last_date = df.index[-1]
    next_day = last_date + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    # Normalize both for robust comparison
    df_dates = pd.to_datetime(df.index).normalize()
    next_day_norm = pd.to_datetime(next_day).normalize()
    # Diagnostics for tomorrow's prediction date check
    print("Next day to predict:", next_day_norm)
    print("All dates in df:", df_dates)
    rows_for_next_day = df.loc[df_dates == next_day_norm]
    print("Rows for next day:", rows_for_next_day)
    # Only skip if next day exists AND has a valid Close price
    if not rows_for_next_day.empty and rows_for_next_day['Close'].notna().any():
        print(f"Data for {next_day.date()} already exists. No prediction needed.")
        return None
    else:
        print("No data for next day, proceeding with prediction.")

        # Ensure we have models available
        if not self.models or len(self.models) == 0:
            print("No models available for prediction. Please train models first.")
            return None
        # Use the same feature selection as training, ensuring consistent features
        try:
            # First engineer features to get all potential features
            features_df = self.select_features(df)
            latest_features = features_df.iloc[-1:].copy()

            # Check if we have a scaler and ensure feature names match
            if hasattr(self, 'scaler') and hasattr(self.scaler, 'feature_names_in_'):
                # Only keep features that were present during scaler fitting
                expected_features = self.scaler.feature_names_in_
                common_features = [f for f in expected_features if f in latest_features.columns]
                if len(common_features) < len(expected_features):
                    print(f"Warning: Only {len(common_features)}/{len(expected_features)} scaler features available")
                    missing_features = set(expected_features) - set(latest_features.columns)
                    print(f"Missing features (first 5): {list(missing_features)[:5]}")

                if len(common_features) == 0:
                    raise ValueError("No matching features found between scaler and current data")

                # Reorder columns to match the order used during training
                latest_features = latest_features[common_features]

                # Warn about missing features that might affect prediction
                if len(common_features) < len(expected_features):
                    print("Feature mismatch may affect prediction quality. Consider retraining.")
            else:
                print("WARNING: Scaler has no feature_names_in_ attribute. Feature alignment may fail.")

            # Now transform the data with matching features
            latest_scaled = self.scaler.transform(latest_features)
        except Exception as e:
            print(f"Error preparing features: {e}")
            # Get a list of expected vs. actual features for debugging
            if hasattr(self, 'scaler') and hasattr(self.scaler, 'feature_names_in_'):
                expected = set(self.scaler.feature_names_in_)
                actual = set(features_df.columns)
                missing = expected - actual
                extra = actual - expected
                print(f"Missing features: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
                print(f"Extra features: {list(extra)[:5]}{'...' if len(extra) > 5 else ''}")
            raise
        pred, prob, conf, indiv_preds, indiv_probs = self.ensemble_predict(latest_scaled)

        # Calculate next trading day
        last_date = df.index[-1]
        next_day = last_date + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)

        print(f"Prediction for {next_day.date()} based on {len(self.models)} models")
        if len(indiv_probs) > 0:
            print("Individual model predictions:")
            for model_name, prob_val in indiv_probs.items():
                print(f"  - {model_name}: {prob_val[0]:.4f}")

        # Generate signal
        if prob[0] > 0.65:
            signal = "STRONG BUY"
        elif prob[0] > 0.4:
            signal = "BUY"
        elif prob[0] < 0.35:
            signal = "STRONG SELL"
        elif prob[0] < 0.4:
            signal = "SELL"
        else:
            signal = "HOLD"

        print("About to return tomorrow's prediction result.")
        return {
            'date': next_day,
            'signal': signal,
            'probability': prob[0],
            'confidence': conf[0],
            'individual_models': {name: indiv_probs[name][0] for name in self.models.keys()}
        }
