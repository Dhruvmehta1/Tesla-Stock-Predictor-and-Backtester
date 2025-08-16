import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from joblib import dump, load
import pandas as pd
import numpy as np
import json
import random
import hashlib
import datetime
from datetime import datetime as dt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import logging
import sys
import traceback

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



# --- Move these functions to top-level for proper np scope ---
def backtest_financial_metrics(preds, X_idx, df_clean):
    test_close = df_clean.loc[X_idx, 'Close']
    returns = []
    capital = 10000
    position = 0
    portfolio = [capital]
    for i in range(len(preds)-1):
        if preds[i] == 1:
            position = 1
        else:
            position = 0
        daily_return = position * (test_close.iloc[i+1] / test_close.iloc[i] - 1)
        capital *= (1 + daily_return)
        returns.append(daily_return)
        portfolio.append(capital)
    total_return = (portfolio[-1] / portfolio[0]) - 1
    avg_return = np.mean(returns) if returns else 0
    std_return = np.std(returns) if returns else 0
    sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
    max_drawdown = 0
    peak = portfolio[0]
    for value in portfolio:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_drawdown:
            max_drawdown = dd
    return sharpe, total_return, max_drawdown, portfolio

def hash_df(df):
    """Compute a hash of the DataFrame for data integrity checking"""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

from tesla_stock_predictor.data.polygon_data import get_polygon_data
from tesla_stock_predictor.features.engineering import engineer_features, select_features, create_targets
from tesla_stock_predictor.models.training import ModelTrainer
from tesla_stock_predictor.models.ensemble import ensemble_predict, predict_tomorrow
from tesla_stock_predictor.debug.debug_tools import print_selected_feature_names

class TSLAPredictor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []

    def get_polygon_data(self, ticker, start_date, end_date):
        return get_polygon_data(ticker, start_date, end_date, self.api_key)

    def engineer_features(self, df):
        return engineer_features(df)

    def select_features(self, df):
        features = select_features(df)
        self.feature_names = features.columns.tolist()
        return features

    def create_targets(self, df):
        return create_targets(df)

    def train_models(self, X_train, y_train, X_val, y_val, close_prices_val):
        trainer = ModelTrainer()
        trainer.train_models(X_train, y_train, X_val, y_val, close_prices_val)
        self.models = trainer.models

    def ensemble_predict(self, X, model_list=None, weights=None, threshold=None, model_thresholds=None):
        # Forward all arguments to the unified ensemble logic
        return ensemble_predict(self, X, model_list, weights, threshold, model_thresholds)

    def predict_tomorrow(self, df):
        return predict_tomorrow(self, df)

def main():
    # --- Logging setup ---
    logging.basicConfig(
        level=logging.ERROR,
        format="%(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler("run.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Parameters
    ticker = "TSLA"
    start = "2020-01-01"
    end = dt.now().strftime("%Y-%m-%d")

    # Load API key from environment variable
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")

    predictor = TSLAPredictor(api_key)

    # Fetch and process data
    df = predictor.get_polygon_data(ticker, start, end)

    # === DATA SOURCE DEBUGGING: Save raw data only on new day OR when data changes ===
    import os

    # Compute hash first
    tsla_hash = hash_df(df)
    today = dt.now().strftime("%Y%m%d")

    # Check if we have a previous hash log
    hash_log_file = "tesla_stock_predictor/debug/data_hash_log.txt"
    should_save = False
    save_reason = ""

    if os.path.exists(hash_log_file):
        with open(hash_log_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1].strip() if lines else ""

        if last_line:
            last_date, last_hash = last_line.split(",", 1)
            if last_date != today:
                should_save = True
                save_reason = "new day"
            elif last_hash != tsla_hash:
                should_save = True
                save_reason = "data changed"
    else:
        should_save = True
        save_reason = "first run"

    print(f"=== DATA SOURCE DEBUG ===")
    print(f"TSLA data hash: {tsla_hash}")
    print(f"TSLA data shape: {df.shape}")
    print(f"TSLA data date range: {df.index.min()} to {df.index.max()}")

    if should_save:
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")

        # Save raw TSLA data snapshot
        df.to_csv(f"tesla_stock_predictor/debug/raw_tsla_data_{timestamp}.csv")
        df.head().to_csv(f"tesla_stock_predictor/debug/raw_tsla_head_{timestamp}.csv")
        df.tail().to_csv(f"tesla_stock_predictor/debug/raw_tsla_tail_{timestamp}.csv")

        # Update hash log
        with open(hash_log_file, 'a') as f:
            f.write(f"{today},{tsla_hash}\n")

        print(f"Raw data saved with timestamp: {timestamp} (reason: {save_reason})")
    else:
        print("Data unchanged since last run today - no files saved")

    print("=========================")

    # Fetch sector ETF (XLY) and SPY data from Polygon
    sector_ticker = "XLY"
    spy_ticker = "SPY"
    sector_df = predictor.get_polygon_data(sector_ticker, start, end)
    spy_df = predictor.get_polygon_data(spy_ticker, start, end)

    # Rename columns for clarity before merging
    sector_df = sector_df.rename(columns={"Close": "Sector_Close"})
    spy_df = spy_df.rename(columns={"Close": "SPY_Close"})

    # Merge sector and SPY close prices into TSLA dataframe
    df = df.merge(sector_df[["Sector_Close"]], left_index=True, right_index=True, how="left")
    df = df.merge(spy_df[["SPY_Close"]], left_index=True, right_index=True, how="left")

    # Use incremental feature engineering with cache
    from tesla_stock_predictor.features.engineering import engineer_features_incremental
    df = engineer_features_incremental(df)
    # Save engineered features for debug inspection
    df.to_csv("tesla_stock_predictor/debug/features_latest.csv")
    df = predictor.create_targets(df)

    # Remove rows with NaN targets (last few rows)
    df_clean = df.dropna(subset=['Target_1d']).copy()

    # Use all available data for feature selection
    predictor.select_features(df_clean)

    X = df_clean[predictor.feature_names].copy()
    X = X.fillna(method='ffill').fillna(0)
    y = df_clean['Target_1d']

    # Ensure X and y are aligned
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    df_clean = df_clean.sort_index()

    # Fixed date splits
    TRAIN_START_DATE = "2023-08-03 04:00:00"
    TRAIN_END_DATE = "2024-12-20 05:00:00"
    VAL_START_DATE = "2024-12-23 05:00:00"
    VAL_END_DATE = "2025-04-11 04:00:00"
    TEST_START_DATE = "2025-04-14 04:00:00"

    train_mask = (X.index >= pd.to_datetime(TRAIN_START_DATE)) & (X.index <= pd.to_datetime(TRAIN_END_DATE))
    val_mask = (X.index >= pd.to_datetime(VAL_START_DATE)) & (X.index <= pd.to_datetime(VAL_END_DATE))
    test_mask = (X.index >= pd.to_datetime(TEST_START_DATE))

    X_train_full = X.loc[train_mask]
    y_train_full = y.loc[train_mask]
    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    # Fit scaler on full training data and transform validation set before loop
    scaler = StandardScaler()
    scaler.fit(X_train_full)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    predictor.scaler = scaler  # Ensure scaler is available for tomorrow's prediction

    print(f"Fixed date split:")
    print(f"Train: {X_train_full.index.min()} to {X_train_full.index.max()} ({len(X_train_full)} samples)")
    print(f"Val: {X_val.index.min()} to {X_val.index.max()} ({len(X_val)} samples)")
    print(f"Test: {X_test.index.min()} to {X_test.index.max()} ({len(X_test)} samples)")

    # Print class distributions for debugging
    print("Train class distribution:", np.bincount(y_train_full))
    print("Validation class distribution:", np.bincount(y_val))
    print("Test class distribution:", np.bincount(y_test))

    # Prepare test dates for incremental walk-forward
    test_dates = X_test.index.sort_values()

    # --- Incremental walk-forward backtest ---
    trade_log = []
    portfolio = [10000]
    capital = 10000
    cash = capital
    position = 0
    shares = 0
    last_buy_price = None
    sl = None
    tp = None

    # Strictly causal, efficient walk-forward backtesting:
    # For each test day, train models only on data up to the previous day, predict for the current day, and log the trade.
    # Never re-predict or re-trade previous days when new data is added.

    trade_log_path = "tesla_stock_predictor/debug/trade_log.csv"
    import pandas.errors
    if os.path.exists(trade_log_path) and os.path.getsize(trade_log_path) > 0:
        try:
            trade_log_df = pd.read_csv(trade_log_path)
            if not trade_log_df.empty and "Date" in trade_log_df.columns:
                trade_log_df["Date"] = pd.to_datetime(trade_log_df["Date"])
                last_logged_date = trade_log_df["Date"].max()
            else:
                last_logged_date = None
        except pandas.errors.EmptyDataError:
            trade_log_df = pd.DataFrame()
            last_logged_date = None
    else:
        trade_log_df = pd.DataFrame()
        last_logged_date = None



    model_cache_dir = "tesla_stock_predictor/models/daily_models"
    os.makedirs(model_cache_dir, exist_ok=True)

    # --- Train/load all ensemble models for validation period before ensemble grid search ---
    # This ensures predictor.models is populated for grid_search_model_thresholds
    predictor.train_models(X_train_full, y_train_full, X_val, y_val, df_clean.loc[X_val.index, 'Close'])

    # --- Grid Search for Ensemble Weights/Thresholds with Caching (MOVED UP) ---
    model_list = ['rf', 'lr', 'dt', 'lgb', 'gb']
    weights = {'rf': 1, 'lr': 1, 'dt': 1, 'lgb': 1, 'gb': 1}
    threshold = None
    model_thresholds = None
    from tesla_stock_predictor.models.ensemble import grid_search_model_thresholds

    val_start_str = str(X_val.index.min()).replace(" ", "_").replace(":", "-")
    val_end_str = str(X_val.index.max()).replace(" ", "_").replace(":", "-")
    ensemble_config_path = f"tesla_stock_predictor/models/ensemble_config_{val_start_str}_{val_end_str}.json"

    # Grid search: 6 threshold values for each model, 4 ensemble threshold values (~31k combinations)
    fast_threshold_grid = [0.2,0.25,0.3,0.35,0.4, 0.5]  # 6 values
    fast_ensemble_threshold_grid = [0.3,0.35,0.4,0.45,0.5]  # 4 values

    best_ensemble_config = None
    if os.path.exists(ensemble_config_path):
        with open(ensemble_config_path, "r") as f:
            try:
                best_ensemble_config = json.load(f)
            except Exception:
                best_ensemble_config = None
        # Validate loaded config
        if not best_ensemble_config or not isinstance(best_ensemble_config, dict) or "model_thresholds" not in best_ensemble_config or "ensemble_threshold" not in best_ensemble_config:
            print(f"Cached ensemble config is missing or invalid. Running grid search for {ensemble_config_path}...")
            best_ensemble_config = grid_search_model_thresholds(
                predictor,
                X_val_scaled,
                y_val,
                df_clean,
                X_val,  # Pass X_val for correct index
                model_list=model_list,
                weights=weights,
                threshold_grid=fast_threshold_grid,
                ensemble_threshold_grid=fast_ensemble_threshold_grid
            )
            # Save only relevant keys to cache
            config_to_save = {
                "model_thresholds": best_ensemble_config["model_thresholds"],
                "ensemble_threshold": best_ensemble_config["ensemble_threshold"],
                "weights": best_ensemble_config.get("weights", None)
            }
            with open(ensemble_config_path, "w") as f:
                json.dump(config_to_save, f)
            print(f"Saved new ensemble config to {ensemble_config_path}")
        else:
            print(f"Loaded cached ensemble config from {ensemble_config_path}")
    else:
        best_ensemble_config = grid_search_model_thresholds(
            predictor,
            X_val_scaled,
            y_val,
            df_clean,
            X_val,  # Pass X_val for correct index
            model_list=model_list,
            weights=weights,
            threshold_grid=fast_threshold_grid,
            ensemble_threshold_grid=fast_ensemble_threshold_grid
        )
        # Save only relevant keys to cache
        config_to_save = {
            "model_thresholds": best_ensemble_config["model_thresholds"],
            "ensemble_threshold": best_ensemble_config["ensemble_threshold"],
            "weights": best_ensemble_config.get("weights", None)
        }
        with open(ensemble_config_path, "w") as f:
            json.dump(config_to_save, f)
        print(f"Saved new ensemble config to {ensemble_config_path}")

    for i, date in enumerate(test_dates):
        # Only process new dates (not already logged)
        if last_logged_date is not None and pd.to_datetime(date) <= last_logged_date:
            continue

        # Use all train data up to yesterday
        train_mask = (X_train_full.index < date)
        X_train = X_train_full.loc[train_mask]
        y_train = y_train_full.loc[train_mask]
        if len(X_train) < 50:
            continue

        # Apply SMOTE only to training data, then scale
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Scaler caching: save/load scaler per day
        scaler_cache_dir = "tesla_stock_predictor/models/daily_scalers"
        os.makedirs(scaler_cache_dir, exist_ok=True)
        scaler_path = os.path.join(scaler_cache_dir, f"scaler_{date}.joblib")
        if os.path.exists(scaler_path):
            scaler = load(scaler_path)
        else:
            scaler = StandardScaler()
            scaler.fit(X_train_balanced)
            dump(scaler, scaler_path)
        X_train_scaled = scaler.transform(X_train_balanced)
        X_today = X.loc[[date]]
        X_today_scaled = scaler.transform(X_today)

        # Model caching: train and save/load all models for this day
        predictor.models = {}
        for model_name in ['rf', 'lr', 'dt', 'lgb', 'gb']:
            model_path = os.path.join(model_cache_dir, f"{model_name}_model_{date}.joblib")
            if os.path.exists(model_path):
                predictor.models[model_name] = load(model_path)
            else:
                # Train and save model if not cached
                if model_name in predictor.models:
                    model = predictor.models[model_name]
                else:
                    # Use best params from cache
                    from joblib import dump
                    best_params_path = "tesla_stock_predictor/models/best_params.json"
                    with open(best_params_path, "r") as f:
                        best_params = json.load(f)
                    params = best_params.get(model_name, {})
                    if model_name == 'rf':
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(**params)
                    elif model_name == 'gb':
                        from sklearn.ensemble import GradientBoostingClassifier
                        model = GradientBoostingClassifier(**params)
                    elif model_name == 'lr':
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(**params)
                    elif model_name == 'dt':
                        from sklearn.tree import DecisionTreeClassifier
                        model = DecisionTreeClassifier(**params)
                    elif model_name == 'lgb':
                        import lightgbm as lgb
                        model = lgb.LGBMClassifier(**params)
                    else:
                        continue
                    model.fit(X_train_scaled, y_train_balanced)
                dump(model, model_path)
                predictor.models[model_name] = model

        # Use ensemble to predict today's signal
        available_model_list = [m for m in predictor.models.keys()]
        if available_model_list:
            signal, *_ = predictor.ensemble_predict(
                X_today_scaled,
                model_list=available_model_list,
                weights=best_ensemble_config.get("weights", weights),
                threshold=best_ensemble_config.get("ensemble_threshold", threshold),
                model_thresholds=best_ensemble_config.get("model_thresholds", model_thresholds)
            )
            signal = signal[0]  # Get scalar prediction for today
        else:
            # Fallback: no models available, hold
            signal = 0

        # Execute today's trade logic (simplified, you can expand as needed)
        open_price = df_clean.loc[date, 'Open']
        close_price = df_clean.loc[date, 'Close']
        action = None
        trade_shares = 0
        trade_price = None
        trade_pnl = None
        transaction_cost = 0.001
        cost = 0
        borrow_cost = 0

        # Only allow one trade per day, strictly causal
        if signal == 1 and position == 0:
            trade_shares = int(cash // (open_price * (1 + transaction_cost)))
            if trade_shares > 0:
                trade_price = open_price
                total_cost = trade_shares * trade_price * (1 + transaction_cost)
                cash -= total_cost
                cost = trade_shares * trade_price * transaction_cost
                shares = trade_shares
                position = 1
                action = "BUY"
                last_buy_price = trade_price
        elif signal == 0 and position == 1:
            trade_price = open_price
            cash += shares * trade_price
            cost = shares * trade_price * transaction_cost
            cash -= cost
            trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
            action = "SELL"
            shares = 0
            position = 0
            last_buy_price = None

        equity = shares * close_price
        portfolio_value = cash + equity
        portfolio.append(portfolio_value)

        # Append new trade to trade log DataFrame
        new_trade = pd.DataFrame([{
            "Date": date,
            "Action": action if action else "HOLD",
            "Signal": signal,
            "Open_Price": open_price,
            "Close_Price": close_price,
            "Trade_Price": trade_price,
            "Shares": trade_shares,
            "Cash": cash,
            "Equity": equity,
            "Portfolio_Value": portfolio_value,
            "Position": position,
            "Trade_PnL": trade_pnl,
            "Transaction_Cost": cost,
            "Borrow_Cost": borrow_cost,
        }])
        trade_log_df = pd.concat([trade_log_df, new_trade], ignore_index=True)

    # Show last 10 trading days in the trade log
    trade_log_df = pd.DataFrame(trade_log)
    if not trade_log_df.empty and "Date" in trade_log_df.columns:
        trade_log_df["Date"] = pd.to_datetime(trade_log_df["Date"])
        daily_log = trade_log_df.groupby("Date", as_index=False).last()
        print("\n--- Trade Log for Last 10 Trading Days ---")
        print(daily_log.tail(10).to_string(index=False))
    else:
        print("\n--- Trade Log for Last 10 Trading Days ---")
        print("No trades logged yet.")

    # --- Save trade log to CSV in debug folder ---
    trade_log_df.to_csv(trade_log_path, index=False)
    print(f"Trade log saved to {trade_log_path}")

    # Use best ensemble config for validation predictions
    val_pred, _, _, _, _ = predictor.ensemble_predict(
        X_val_scaled,
        model_list=model_list,
        weights=best_ensemble_config.get("weights", weights),
        threshold=best_ensemble_config.get("ensemble_threshold", threshold),
        model_thresholds=best_ensemble_config.get("model_thresholds", model_thresholds)
    )
    f1 = f1_score(y_val, val_pred, average='macro')
    sharpe, total_return, max_drawdown, _ = backtest_financial_metrics(val_pred, X_val.index, df_clean)
    print(f"Validation F1: {f1:.4f}, Sharpe: {sharpe:.4f}, Max Drawdown: {max_drawdown:.4f}")

    # Use the same logic for test set
    # Use best ensemble config for test predictions
    test_pred, test_prob, test_conf, test_indiv_preds, test_indiv_probs = predictor.ensemble_predict(
        X_test_scaled,
        model_list=model_list,
        weights=best_ensemble_config.get("weights", weights),
        threshold=best_ensemble_config.get("ensemble_threshold", threshold),
        model_thresholds=best_ensemble_config.get("model_thresholds", model_thresholds)
    )

    test_f1 = f1_score(y_test, test_pred, average='macro')
    test_accuracy = accuracy_score(y_test, test_pred)

    # --- Print Model Performance Metrics ---
    print(f"\nEnsemble F1: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}")
    print("\nDetailed Classification Report:\n" + classification_report(y_test, test_pred))

    # Define test_close for use in unified simulation
    test_close = df_clean.loc[X_test.index, 'Close']

    # --- Unified Portfolio Simulation & Trade Log ---
    trade_log = []
    capital = 10000
    cash = capital
    shares = 0
    position = 0  # 0 = out, 1 = in
    portfolio = [capital]
    last_buy_price = None

    # Initialize trade state variables to avoid unbound errors
    sl = None
    tp = None
    risk_per_share = None
    sell_next_open = False  # Ensure always defined

    test_dates = test_close.index
    returns = []
    max_drawdown = 0
    peak = capital

    for i in range(len(test_pred)-1):
        date = test_dates[i]
        open_price = df_clean.loc[date, 'Open']
        close_price = df_clean.loc[date, 'Close']
        high_price = df_clean.loc[date, 'High']
        signal = test_pred[i]
        action = None
        trade_shares = 0
        trade_price = None
        trade_pnl = None
        transaction_cost = 0.001  # 0.1% per trade
        cost = 0  # Transaction cost for this step
        borrow_cost = 0  # Borrow fee for this step

        # --- Model-based exit at next open ---
        if sell_next_open and position == 1:
            trade_price = open_price
            cash += shares * trade_price
            cost = shares * trade_price * transaction_cost
            cash -= cost
            trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
            action = "SELL_NEXT_OPEN"
            shares = 0
            position = 0
            last_buy_price = None
            sl = None
            tp = None
            sell_next_open = False
            equity = 0
            # On SELL, use trade_price for portfolio value
            portfolio_value = cash + (shares * trade_price if trade_price is not None else 0)
            portfolio.append(portfolio_value)
            trade_log.append({
                "Date": date,
                "Action": action,
                "Signal": signal,
                "Open_Price": open_price,
                "Close_Price": close_price,
                "Trade_Price": trade_price,
                "Shares": 0,
                "Cash": cash,
                "Equity": equity,
                "Portfolio_Value": portfolio_value,
                "Position": position,
                "Trade_PnL": trade_pnl,
                "Transaction_Cost": cost,
                "Borrow_Cost": borrow_cost,
            })
            continue  # Skip the rest of the loop for this day



        # --- Shorting logic commented out ---
        # position: 0 = flat, 1 = long, -1 = short (shorting disabled below)

        # --- Corrected Trading Logic: SL, Partial Exit at TP, Trailing Stop, Model-based Exit ---

        if signal == 1 and position == 0:
            trade_shares = int(cash // (open_price * (1 + transaction_cost)))
            if trade_shares > 0:
                trade_price = open_price
                total_cost = trade_shares * trade_price * (1 + transaction_cost)
                cash -= total_cost
                cost = trade_shares * trade_price * transaction_cost
                shares = trade_shares
                position = 1
                action = "BUY"
                last_buy_price = trade_price
                risk_per_share = 0.01 * trade_price  # 0.9% of entry price per share
                sl = trade_price - risk_per_share
                tp = trade_price + 3.2 * risk_per_share
                sell_next_open = False

                # --- Immediately check for SL/TP hit on the same day as BUY ---
                # SL check (gap down or intraday cross)
                if sl is not None:
                    if open_price <= sl:
                        # Gap down: sell at open
                        trade_price = open_price
                        cash += shares * trade_price
                        cost = shares * trade_price * transaction_cost
                        cash -= cost
                        trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
                        action = "SELL_SL_OPEN"
                        shares = 0
                        position = 0
                        last_buy_price = None
                        sl = None
                        tp = None
                        sell_next_open = False
                    elif min(open_price, close_price) <= sl <= max(open_price, close_price):
                        # Crossed SL intraday: sell at SL
                        trade_price = sl
                        cash += shares * trade_price
                        cost = shares * trade_price * transaction_cost
                        cash -= cost
                        trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
                        action = "SELL_SL_INTRADAY"
                        shares = 0
                        position = 0
                        last_buy_price = None
                        sl = None
                        tp = None
                        sell_next_open = False
                        equity = 0
                        # On SELL, use trade_price for portfolio value
                        portfolio_value = cash + (shares * trade_price if trade_price is not None else 0)
                        trade_log.append({
                            "Date": date,
                            "Action": action,
                            "Signal": signal,
                            "Open_Price": open_price,
                            "Close_Price": close_price,
                            "Trade_Price": trade_price,
                            "Shares": shares,
                            "Cash": cash,
                            "Equity": equity,
                            "Portfolio_Value": portfolio_value,
                            "Position": position,
                            "Trade_PnL": trade_pnl,
                            "Transaction_Cost": cost,
                            "Borrow_Cost": borrow_cost,
                        })
                # TP check (gap up, intraday cross, or intraday high)
                if tp is not None and shares > 0:  # Only check TP if still in position
                    if open_price >= tp:
                        # Gap up: sell at open
                        trade_price = open_price
                        cash += shares * trade_price
                        cost = shares * trade_price * transaction_cost
                        cash -= cost
                        trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
                        action = "SELL_TP_OPEN"
                        shares = 0
                        position = 0
                        last_buy_price = None
                        sl = None
                        tp = None
                        sell_next_open = False
                    elif min(open_price, close_price) <= tp <= max(open_price, close_price):
                        # Crossed TP intraday: sell at TP
                        trade_price = tp
                        cash += shares * trade_price
                        cost = shares * trade_price * transaction_cost
                        cash -= cost
                        trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
                        action = "SELL_TP_INTRADAY"
                        shares = 0
                        position = 0
                        last_buy_price = None
                        sl = None
                        tp = None
                        sell_next_open = False
                    elif min(open_price, high_price) <= tp <= max(open_price, high_price):
                        # Crossed TP intraday (between open and high): sell at TP
                        trade_price = tp
                        cash += shares * trade_price
                        cost = shares * trade_price * transaction_cost
                        cash -= cost
                        trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
                        action = "SELL_TP_INTRADAY_HIGH"
                        shares = 0
                        position = 0
                        last_buy_price = None
                        sl = None
                        tp = None
                        sell_next_open = False
            else:
                action = "HOLD"

        elif position == 1 and sl is not None:
            if open_price <= sl:
                # Gap down: sell at open
                trade_price = open_price
                cash += shares * trade_price
                cost = shares * trade_price * transaction_cost
                cash -= cost
                trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
                action = "SELL_SL_OPEN"
                shares = 0
                position = 0
                last_buy_price = None
                sl = None
                tp = None
                sell_next_open = False
            elif min(open_price, close_price) <= sl <= max(open_price, close_price):
                # Crossed SL intraday: sell at SL
                trade_price = sl
                cash += shares * trade_price
                cost = shares * trade_price * transaction_cost
                cash -= cost
                trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
                action = "SELL_SL_INTRADAY"
                shares = 0
                position = 0
                last_buy_price = None
                sl = None
                tp = None
                sell_next_open = False
                equity = 0
                # On SELL, use trade_price for portfolio value
                portfolio_value = cash + (shares * trade_price if trade_price is not None else 0)
                trade_log.append({
                    "Date": date,
                    "Action": action,
                    "Signal": signal,
                    "Open_Price": open_price,
                    "Close_Price": close_price,
                    "Trade_Price": trade_price,
                    "Shares": shares,
                    "Cash": cash,
                    "Equity": equity,
                    "Portfolio_Value": portfolio_value,
                    "Position": position,
                    "Trade_PnL": trade_pnl,
                    "Transaction_Cost": cost,
                    "Borrow_Cost": borrow_cost,
                })
            elif tp is not None and open_price >= tp:
                # Gap up: sell at open
                trade_price = open_price
                cash += shares * trade_price
                cost = shares * trade_price * transaction_cost
                cash -= cost
                trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
                shares = 0
                position = 0
                last_buy_price = None
                sl = None
                tp = None
                sell_next_open = False
                action = "SELL_TP_OPEN"
            elif tp is not None and (min(open_price, close_price) <= tp <= max(open_price, close_price)):
                # Crossed TP intraday: sell at TP
                trade_price = tp
                cash += shares * trade_price
                cost = shares * trade_price * transaction_cost
                cash -= cost
                trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
                shares = 0
                position = 0
                last_buy_price = None
                sl = None
                tp = None
                sell_next_open = False
                action = "SELL_TP_INTRADAY"
            elif tp is not None and (min(open_price, high_price) <= tp <= max(open_price, high_price)):
                # Crossed TP intraday (between open and high): sell at TP
                trade_price = tp
                cash += shares * trade_price
                cost = shares * trade_price * transaction_cost
                cash -= cost
                trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
                shares = 0
                position = 0
                last_buy_price = None
                sl = None
                tp = None
                sell_next_open = False
                action = "SELL_TP_INTRADAY_HIGH"
            elif sell_next_open:
                trade_price = open_price
                cash += shares * trade_price
                cost = shares * trade_price * transaction_cost
                cash -= cost
                trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
                action = "SELL_NEXT_OPEN"
                shares = 0
                position = 0
                last_buy_price = None
                sl = None
                tp = None
                sell_next_open = False
            else:
                action = "HOLD"
        else:
            action = "HOLD"

        # Print daily summary with action for every day
        # print(f"Date: {date}, SL: {sl}, TP: {tp}, Open: {open_price}, Close: {close_price}, Shares: {shares}, Action: {action}")

        # Generalized equity calculation:
        # equity = shares x close_price on HOLD/BUY, equity = shares x trade_price on SELL
        if action in ["SELL_SL_OPEN", "SELL_SL_INTRADAY", "SELL_TP_OPEN", "SELL_TP_INTRADAY", "SELL_NEXT_OPEN"]:
            equity = shares * trade_price if trade_price is not None else 0
        else:
            equity = shares * close_price
        portfolio_value = cash + equity
        portfolio.append(portfolio_value)

        # Calculate daily return for Sharpe
        if i < len(test_pred)-2:
            next_close = df_clean.loc[test_dates[i+1], 'Close']
            if position == 1:
                daily_return = (next_close / close_price - 1)
            elif position == -1:
                daily_return = (close_price / next_close - 1)
            else:
                daily_return = 0
            returns.append(daily_return)

        # Track max drawdown
        if portfolio_value > peak:
            peak = portfolio_value
        dd = (peak - portfolio_value) / peak
        if dd > max_drawdown:
            max_drawdown = dd

        trade_log.append({
            "Date": date,
            "Action": action,
            "Signal": signal,
            "Open_Price": open_price,
            "Close_Price": close_price,
            "Trade_Price": trade_price,
            "Shares": trade_shares,
            "Cash": cash,
            "Equity": equity,
            "Portfolio_Value": portfolio_value,
            "Position": position,
            "Trade_PnL": trade_pnl,
            "Transaction_Cost": cost,
            "Borrow_Cost": borrow_cost,
        })

    trade_log_df = pd.DataFrame(trade_log)
    if not trade_log_df.empty and "Date" in trade_log_df.columns:
        trade_log_df["Date"] = pd.to_datetime(trade_log_df["Date"])
        # Show last 10 trading days in the trade log (one entry per day, final state)
        daily_log = trade_log_df.groupby("Date", as_index=False).last()

    else:
        print("No trades logged yet.")
    print("\n--- Trade Log for Last 10 Trading Days ---")
    print(daily_log.tail(10).to_string(index=False))

    # --- Save all trades to CSV in debug folder ---
    trade_log_df.to_csv(trade_log_path, index=False)
    print(f"Trade log saved to {trade_log_path}")

    # --- Financial Metrics from Unified Simulation (Portfolio-based returns) ---
    # Calculate daily returns from actual portfolio values
    portfolio_returns = []
    for i in range(1, len(portfolio)):
        daily_return = (portfolio[i] / portfolio[i-1]) - 1
        portfolio_returns.append(daily_return)
    total_return = (portfolio[-1] / portfolio[0]) - 1
    avg_return = np.mean(portfolio_returns) if portfolio_returns else 0
    std_return = np.std(portfolio_returns) if portfolio_returns else 0
    sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0

    print(f"\n--- Financial Backtest Metrics (Test Set) ---")
    print(f"Cumulative Return: {total_return:.4f}")
    print(f"Annualized Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")
    print(f"Final Portfolio Value: ${portfolio[-1]:.2f}")

    # --- Plot Equity Curve ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates[:len(portfolio)], portfolio, label="Equity Curve", color="blue")
        plt.title("Equity Curve (Test Set)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("equity_curve.png")
        plt.show()
    except ImportError:
        print("matplotlib is not installed. Skipping equity curve plot.")

    # Add Actual_Movement column for correct day-over-day movement
    df_clean['Actual_Movement'] = (df_clean['Close'] > df_clean['Close'].shift(1)).astype(int)
    # Dynamically build recent_df to only include present model probabilities
    # Always include all models in the output table, regardless of their ensemble weight
    model_prob_cols = [
        ('RF_Prob', 'rf'),
        ('GB_Prob', 'gb'),
        ('LR_Prob', 'lr'),
        ('DT_Prob', 'dt'),
        ('LGB_Prob', 'lgb'),
    ]

    recent_df_dict = {
        'Date': X_test.index[-10:],
        'Actual': df_clean.loc[X_test.index[-10:], 'Actual_Movement'].values,
        'Ensemble_Pred': test_pred[-10:],
        'Ensemble_Prob': test_prob[-10:],
        'Confidence': test_conf[-10:],
        'Close_Price': df_clean.loc[X_test.index[-10:], 'Close'].values,
    }
    for col_name, model_key in model_prob_cols:
        # If the model is present in test_indiv_probs, use its probabilities; else fill with NaN
        if model_key in test_indiv_probs:
            recent_df_dict[col_name] = test_indiv_probs[model_key][-10:]
        else:
            recent_df_dict[col_name] = [float('nan')] * 10

    recent_df = pd.DataFrame(recent_df_dict)
    prob_cols = ['Ensemble_Prob', 'Confidence'] + [col_name for col_name, _ in model_prob_cols]
    for col in prob_cols:
        recent_df[col] = recent_df[col].round(3)

    # --- Print Last 10 Test Predictions Table ---
    print("\n📊 Last 10 Test Predictions vs Actual (Actual_Movement = today's close vs previous day's close):")
    print(recent_df.to_string(index=False))

    try:
        prediction = predictor.predict_tomorrow(df)
        # --- Print Tomorrow's Prediction ---
        print("\n🔮 Tomorrow's Prediction:")
        print(f"Date: {prediction['date'].strftime('%Y-%m-%d')}")
        print(f"Signal: {prediction['signal']}")
        print(f"Probability: {prediction['probability']:.4f}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        print("Individual model probabilities:")
        for model, prob in prediction['individual_models'].items():
            print(f"  {model.upper()}: {prob:.4f}")
    except Exception as e:
        logging.error(f"Error during tomorrow's prediction: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
