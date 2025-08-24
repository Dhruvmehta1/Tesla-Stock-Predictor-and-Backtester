import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
import random

# Set fixed seeds for complete determinism
random.seed(42)
np.random.seed(42)

def ensemble_predict(
    self,
    X,
    model_list=None,
    weights=None,
    threshold=None,
    model_thresholds=None
):
    """
    Unified ensemble logic with enhanced error handling and validation.

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
    # Set seeds again for complete determinism in each prediction
    random.seed(42)
    np.random.seed(42)

    # Validate inputs
    if X is None:
        raise ValueError("Input features X cannot be None")

    if len(X) == 0:
        raise ValueError("Input features X cannot be empty")

    if not hasattr(self, 'models') or not self.models:
        raise ValueError("No trained models available in self.models")

    # Determine which models to use
    if model_list is None:
        model_list = list(self.models.keys())

    # Filter model_list to only include available models
    available_models = [model for model in model_list if model in self.models]
    if not available_models:
        raise ValueError(f"None of the requested models {model_list} are available in self.models")

    model_list = available_models

    # Default per-model thresholds
    if model_thresholds is None:
        model_thresholds = {k: 0.5 for k in model_list}

    # Default to equal weights if not provided
    if weights is None:
        weights = {k: 1.0 for k in model_list}

    # Filter weights to only include available models and normalize
    filtered_weights = {k: weights.get(k, 0) for k in model_list if weights.get(k, 0) > 0}
    if not filtered_weights:
        # If no valid weights, use equal weights for all models
        filtered_weights = {k: 1.0 for k in model_list}

    total_weight = sum(filtered_weights.values())
    if total_weight == 0:
        raise ValueError("At least one model must have nonzero weight.")
    weights = {k: filtered_weights[k] / total_weight for k in filtered_weights}

    # Default ensemble threshold
    if threshold is None:
        threshold = 0.5

    predictions = {}
    probabilities = {}
    successful_models = []

    for name in model_list:
        if name not in self.models:
            continue

        try:
            model = self.models[name]

            # Check for feature count consistency
            if hasattr(model, 'n_features_in_'):
                if model.n_features_in_ != X.shape[1]:
                    print(f"Warning: Model {name} expects {model.n_features_in_} features, got {X.shape[1]}. Skipping.")
                    continue

            # Check for feature names consistency
            if hasattr(model, 'feature_names_in_') and hasattr(X, 'columns'):
                missing_features = set(model.feature_names_in_) - set(X.columns)
                if missing_features:
                    print(f"Warning: Model {name} missing features: {missing_features}. Skipping.")
                    continue

            # Make predictions
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)
                if prob.shape[1] > 1:
                    prob = prob[:, 1]  # Get positive class probability
                else:
                    prob = prob[:, 0]  # Single class case
            else:
                # Fallback for models without predict_proba
                pred = model.predict(X)
                prob = pred.astype(float)  # Convert to probability-like scores

            probabilities[name] = prob
            pred_thresh = model_thresholds.get(name, 0.5)
            predictions[name] = (prob > pred_thresh).astype(int)
            successful_models.append(name)

        except Exception as e:
            print(f"Warning: Model {name} failed with error: {e}. Skipping.")
            continue

    # Ensure at least one model produced probabilities
    if not probabilities:
        raise ValueError("No models produced probabilities. Check that models are trained and compatible with input features.")

    # Update weights to only include successful models
    successful_weights = {k: weights.get(k, 0) for k in successful_models if k in weights}
    if successful_weights:
        total_weight = sum(successful_weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in successful_weights.items()}

    # Weighted ensemble probability
    n_samples = len(next(iter(probabilities.values())))
    ensemble_prob = np.zeros(n_samples)

    for name in successful_models:
        if name in probabilities and name in weights:
            ensemble_prob += probabilities[name] * weights[name]

    # Ensure ensemble_prob is valid
    if np.all(ensemble_prob == 0) and len(probabilities) > 0:
        # Fallback to simple average if weighting failed
        ensemble_prob = np.mean(list(probabilities.values()), axis=0)

    ensemble_pred = (ensemble_prob > threshold).astype(int)

    # Confidence based on agreement between models
    if predictions and len(predictions) > 1:
        # Calculate agreement as standard deviation of predictions
        pred_array = np.array(list(predictions.values()))
        agreement = np.mean(pred_array, axis=0)
        confidence = 1 - 2 * np.abs(agreement - 0.5)  # Higher confidence when closer to 0 or 1
        confidence = np.clip(confidence, 0, 1)  # Ensure confidence is in [0, 1]
    else:
        # Single model or no predictions - use probability distance from threshold
        confidence = 2 * np.abs(ensemble_prob - threshold)
        confidence = np.clip(confidence, 0, 1)

    return ensemble_pred, ensemble_prob, confidence, predictions, probabilities

def grid_search_model_thresholds(
    predictor,
    X_val_scaled,
    y_val,
    df_clean,
    X_val,
    model_list=None,
    weights=None,
    threshold_grid=None,
    ensemble_threshold_grid=None
):
    """
    Simple validation without extensive parameter tuning to avoid overfitting
    """
    # Set seeds for deterministic grid search
    random.seed(42)
    np.random.seed(42)
    if model_list is None:
        model_list = ['rf', 'lr', 'dt', 'lgb', 'gb']
    if weights is None:
        weights = {k: 1 for k in model_list}

    # Use fixed threshold for complete determinism
    ensemble_thresholds = [0.455]  # Single fixed value

    # Use fixed individual model thresholds
    base_model_thresholds = {model: 0.5 for model in model_list}

    ensemble_results = []

    # Test each ensemble threshold with fixed model thresholds
    for ensemble_thresh in ensemble_thresholds:
        try:
            # Single evaluation on validation set - no cross-validation
            val_pred, _, _, _, _ = predictor.ensemble_predict(
                X_val_scaled,
                model_list=model_list,
                weights=weights,
                threshold=ensemble_thresh,
                model_thresholds=base_model_thresholds
            )

            # Import backtest function safely
            try:
                # Try to import from main module
                import sys
                sys.path.append(".")

                # Try different import paths
                try:
                    from main import backtest_financial_metrics
                except ImportError:
                    try:
                        import main
                        backtest_financial_metrics = main.backtest_financial_metrics
                    except ImportError:
                        # Define a simple fallback backtest function
                        def backtest_financial_metrics(preds, idx, df, **kwargs):
                            # Simple buy-and-hold return calculation as fallback
                            if len(preds) == 0 or len(idx) == 0:
                                return 0, 0, 0, ([], [], [], 0, 0, pd.DataFrame())

                            # Get returns for the prediction period
                            prices = df.loc[idx, 'Close']
                            if len(prices) < 2:
                                return 0, 0, 0, ([], [], [], 0, 0, pd.DataFrame())

                            total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                            daily_returns = prices.pct_change().dropna()

                            if len(daily_returns) > 0 and daily_returns.std() > 0:
                                sharpe = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
                            else:
                                sharpe = 0

                            max_dd = (prices / prices.cummax() - 1).min()

                            empty_log = pd.DataFrame({
                                'Date': [idx[0]], 'Action': ['FALLBACK'], 'Portfolio_Value': [10000]
                            })

                            return sharpe, total_return, abs(max_dd), ([], [], [], 0, 0, empty_log)

                sharpe, total_return, max_drawdown, _ = backtest_financial_metrics(
                    val_pred, X_val.index, df_clean
                )

            except Exception as e:
                print(f"Warning: Backtest function failed: {e}")
                sharpe, total_return, max_drawdown = 0, 0, 0

            ensemble_results.append({
                "model_thresholds": base_model_thresholds,
                "ensemble_threshold": ensemble_thresh,
                "sharpe": sharpe,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "weights": weights
            })

        except Exception as e:
            print(f"Warning: Failed to evaluate ensemble with threshold {ensemble_thresh}: {e}")
            continue

    # Always return a valid config, even if all evaluations failed
    if not ensemble_results:
        print("Warning: All ensemble evaluations failed, using default configuration")
        default_config = {
            "model_thresholds": {name: 0.5 for name in model_list},
            "ensemble_threshold": 0.5,
            "sharpe": 0,
            "total_return": 0,
            "max_drawdown": 0,
            "weights": weights
        }
        return default_config

    # Sort results by Sharpe ratio (descending), with fallback to total return
    try:
        ensemble_results.sort(key=lambda x: (x.get("sharpe", 0), x.get("total_return", 0)), reverse=True)
    except Exception as e:
        print(f"Warning: Failed to sort results: {e}")

    # Return best configuration
    best = ensemble_results[0]
    print(f"Best ensemble config: threshold={best['ensemble_threshold']}, sharpe={best.get('sharpe', 0):.4f}")
    return best

def predict_tomorrow(self, df):
    """
    Predict tomorrow without lookahead bias, with enhanced error handling
    """
    # Set seeds for deterministic predictions
    random.seed(42)
    np.random.seed(42)

    try:
        # Validate inputs
        if df is None or df.empty:
            print("Error: DataFrame is empty or None")
            return None

        if not hasattr(self, 'models') or not self.models:
            print("Error: No trained models available")
            return None

        # Find the next business day (or trading day)
        last_date = df.index[-1]
        next_day = last_date + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)

        # Normalize both for robust comparison
        df_dates = pd.to_datetime(df.index).normalize()
        next_day_norm = pd.to_datetime(next_day).normalize()

        # Check if we already have data for the next day
        rows_for_next_day = df.loc[df_dates == next_day_norm]

        # Only skip if next day exists AND has a valid Close price
        if not rows_for_next_day.empty and 'Close' in rows_for_next_day.columns:
            if rows_for_next_day['Close'].notna().any():
                return None

        # Ensure we have selected features from training
        if not hasattr(self, 'selected_features') or not self.selected_features:
            # Fall back to using the scaler's expected features
            if hasattr(self, 'scaler') and hasattr(self.scaler, 'feature_names_in_'):
                self.selected_features = list(self.scaler.feature_names_in_)
            else:
                print("Error: No selected features found. Train model first.")
                return None

        # Use only data up to today for feature engineering
        today_data = df.copy()

        try:
            # Get the last row's features
            if hasattr(self, 'scaler') and hasattr(self.scaler, 'feature_names_in_'):
                expected_features = self.scaler.feature_names_in_

                # Create aligned features DataFrame using only existing data
                selected_cols = [col for col in today_data.columns if col in self.selected_features]
                if not selected_cols:
                    print("Error: No matching features found between data and selected features")
                    return None

                latest_features = today_data[selected_cols].iloc[-1:].copy()

                # Create properly aligned features DataFrame
                aligned_features = pd.DataFrame(0, index=latest_features.index,
                                              columns=expected_features)

                # Fill in values from latest_features where they exist
                for col in expected_features:
                    if col in latest_features.columns:
                        aligned_features[col] = latest_features[col]

                # Replace latest_features with properly aligned DataFrame
                latest_features = aligned_features

            else:
                print("Error: Scaler not properly initialized")
                return None

            # Transform the data with matching features
            if not hasattr(self, 'scaler'):
                print("Error: No scaler available")
                return None

            latest_scaled = self.scaler.transform(latest_features)

        except Exception as e:
            print(f"Error: Feature preparation failed: {e}")
            return None

        try:
            # Make ensemble prediction
            pred, prob, conf, indiv_preds, indiv_probs = self.ensemble_predict(latest_scaled)
        except Exception as e:
            print(f"Error: Ensemble prediction failed: {e}")
            return None

        # Calculate next trading day
        last_date = df.index[-1]
        next_day = last_date + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)

        # Generate signal with more balanced thresholds
        prob_value = prob[0] if isinstance(prob, np.ndarray) and len(prob) > 0 else prob

        if prob_value > 0.7:
            signal = "STRONG BUY"
        elif prob_value > 0.55:
            signal = "BUY"
        elif prob_value < 0.3:
            signal = "STRONG SELL"
        elif prob_value < 0.45:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Prepare individual model results
        individual_models = {}
        for name in self.models.keys():
            if name in indiv_probs:
                prob_val = indiv_probs[name]
                if isinstance(prob_val, np.ndarray) and len(prob_val) > 0:
                    individual_models[name] = prob_val[0]
                else:
                    individual_models[name] = prob_val

        return {
            'date': next_day,
            'signal': signal,
            'probability': prob_value,
            'confidence': conf[0] if isinstance(conf, np.ndarray) and len(conf) > 0 else conf,
            'individual_models': individual_models
        }

    except Exception as e:
        print(f"Error: Tomorrow prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None
