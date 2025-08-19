import numpy as np
import pandas as pd
from datetime import timedelta

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
        prob = self.models[name].predict_proba(X)[:, 1]
        probabilities[name] = prob
        pred_thresh = model_thresholds.get(name, 0.5)
        predictions[name] = (prob > pred_thresh).astype(int)

    # Defensive check: ensure at least one model produced probabilities
    if not probabilities:
        raise ValueError("No models produced probabilities. Check that models are trained and present in self.models.")

    # Weighted ensemble probability (dynamic, only selected models)
    ensemble_prob = np.zeros_like(next(iter(probabilities.values())))
    for name in model_list:
        if name not in probabilities:
            continue
        ensemble_prob += probabilities[name] * weights.get(name, 0)

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
    Grid search for per-model probability thresholds and ensemble threshold to optimize Sharpe ratio.
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
    if model_list is None:
        model_list = ['rf', 'lr', 'dt', 'lgb', 'gb']
    if weights is None:
        weights = {k: 1 for k in model_list}
    if threshold_grid is None:
        threshold_grid = [0.4, 0.5, 0.6]
    if ensemble_threshold_grid is None:
        ensemble_threshold_grid = [0.4, 0.5, 0.6]

    threshold_combinations = list(itertools.product(threshold_grid, repeat=len(model_list)))
    results = []

    print(f"Running grid search for {len(threshold_combinations) * len(ensemble_threshold_grid)} combinations (model thresholds x ensemble threshold)...")

    for idx, (combo, ens_thresh) in enumerate(itertools.product(threshold_combinations, ensemble_threshold_grid)):
        model_thresholds = dict(zip(model_list, combo))
        val_pred, _, _, _, _ = predictor.ensemble_predict(
            X_val_scaled,
            model_list=model_list,
            weights=weights,
            threshold=ens_thresh,
            model_thresholds=model_thresholds
        )
        # Debug print removed to reduce output and speed up grid search
        # Import backtest_financial_metrics from main.py locally to avoid circular import
        from tesla_stock_predictor.main import backtest_financial_metrics
        sharpe, total_return, max_drawdown, _ = backtest_financial_metrics(val_pred, X_val.index, df_clean)
        results.append({
            "model_thresholds": model_thresholds,
            "ensemble_threshold": ens_thresh,
            "sharpe": sharpe,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "weights": weights  # Pass through weights for compatibility with main.py usage
        })

    # Defensive patch: always return a config dict, even if all Sharpe ratios are zero or results is empty
    if not results:
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
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    # Suppress grid search output: do not print validation/grid search metrics
    best = results[0]
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
    if next_day_norm in df_dates.values:
        print(f"Data for {next_day.date()} already exists. No prediction needed.")
        return None
    # Generate features for the next trading day using only data up to the last available date
    next_features = generate_features_for_next_day(df)
    latest_scaled = self.scaler.transform(next_features)
    pred, prob, conf, indiv_preds, indiv_probs = self.ensemble_predict(latest_scaled)

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

    return {
        'date': next_day,
        'signal': signal,
        'probability': prob[0],
        'confidence': conf[0],
        'individual_models': {name: indiv_probs[name][0] for name in self.models.keys()}
    }
