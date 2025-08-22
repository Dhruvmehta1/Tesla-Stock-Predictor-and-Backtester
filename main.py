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

# Import cost-sensitive threshold optimization
from tesla_stock_predictor.analysis.cost_threshold import backtest_with_cost_sensitive_threshold



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
        features_df = select_features(df)
        # If we have stored selected features, filter to only those features
        if hasattr(self, 'selected_features') and self.selected_features:
            available_features = [f for f in self.selected_features if f in features_df.columns]
            if len(available_features) < len(self.selected_features):
                print(f"WARNING: Only {len(available_features)}/{len(self.selected_features)} selected features available")
                missing = set(self.selected_features) - set(available_features)
                print(f"Missing features (first 5): {list(missing)[:5]}")
            if available_features:
                print(f"Using {len(available_features)} selected features for consistency")
                return features_df[available_features]
        print("No stored feature selection found or no matching features. Using all features.")
        return features_df

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
        # Check if we have all required components for prediction
        if not hasattr(self, 'selected_features') or not self.selected_features:
            print("WARNING: No selected features stored. Using all available features.")
        if not hasattr(self, 'scaler'):
            print("WARNING: No scaler found in predictor. Prediction may fail.")
            # Create a default scaler as fallback
            self.scaler = StandardScaler()
        # Verify models exist
        if not hasattr(self, 'models') or not self.models:
            print("WARNING: No trained models found in predictor.")
        return predict_tomorrow(self, df)

def main():
    import os  # Import os at function scope
    # --- Ensure directories exist ---
    required_dirs = [
        "tesla_stock_predictor/debug",
        "tesla_stock_predictor/models",
        "tesla_stock_predictor/models/daily_models",
        "tesla_stock_predictor/data",
        "tesla_stock_predictor/features"
    ]
    debug_dir = "tesla_stock_predictor/debug"
    # Enable/disable visualization features
    visualize_equity_curve = True
    for directory in required_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create directory {directory}: {e}")

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
    features = engineer_features_incremental(df)
    # Defensive check: print engineered features shape and columns
    if features.empty or len(features.columns) == 0:
        raise ValueError("Feature engineering failed: no features generated. Check raw data and feature engineering logic.")

    # Add price columns to features if not already present
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns and col not in features.columns:
            features[col] = df[col]
    df = features

    # Save engineered features for debug inspection
    df.to_csv("tesla_stock_predictor/debug/features_latest.csv")

    # Target creation and cleaning
    df = predictor.create_targets(df)
    df_clean = df.dropna(subset=['Target_1d']).copy()

    # Define date splits with fixed dates
    # Fixed date splits - THESE DATES MUST NOT BE CHANGED TO ENSURE CONSISTENT BACKTESTING
    TRAIN_START_DATE = "2023-08-22 04:00:00"
    TRAIN_END_DATE   = "2024-08-20 04:00:00"
    VAL_START_DATE   = "2024-08-21 04:00:00"
    VAL_END_DATE     = "2025-02-20 04:00:00"
    TEST_START_DATE  = "2025-02-21 04:00:00"
    # No TEST_END_DATE - include all data after TEST_START_DATE

    print(f"\n--- Using Fixed Date Splits ---")
    print(f"Train: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    print(f"Validation: {VAL_START_DATE} to {VAL_END_DATE}")
    print(f"Test: {TEST_START_DATE} and later")

    train_mask = (df_clean.index >= pd.to_datetime(TRAIN_START_DATE)) & (df_clean.index <= pd.to_datetime(TRAIN_END_DATE))
    val_mask = (df_clean.index >= pd.to_datetime(VAL_START_DATE)) & (df_clean.index <= pd.to_datetime(VAL_END_DATE))
    # Ensure test set includes all data after TEST_START_DATE (no upper limit)
    test_mask = (df_clean.index >= pd.to_datetime(TEST_START_DATE))

    # Now perform feature selection only on training data
    df_train_clean = df_clean.loc[train_mask].copy()  # Use copy to avoid SettingWithCopyWarning
    feature_df_train = predictor.select_features(df_train_clean)
    if feature_df_train.shape[1] == 0:
        raise ValueError("Feature selection failed: no features selected.")

    # Filter out problematic features early in the pipeline
    problematic_features = [
        'Close_to_High', 'Close_to_Low',
        'Body_size', 'Upper_shadow', 'Lower_shadow',
        'BB_position', 'BB_squeeze', 'BB_Width',
        'High_Low_ratio', 'OBV',
        'Volume_ratio', 'Volume_Shadow', 'Volume_trend',
        'Trend_strength', 'ATR_vs_BodySize'
    ]

    cols_to_drop = [col for col in feature_df_train.columns
                   if any(feat in col for feat in problematic_features)]

    if cols_to_drop:
        print(f"\n=== REMOVING {len(cols_to_drop)} PROBLEMATIC FEATURES DURING SELECTION ===")
        for col in cols_to_drop[:10]:
            print(f"  - {col}")
        if len(cols_to_drop) > 10:
            print(f"  - ... and {len(cols_to_drop) - 10} more")

        feature_df_train = feature_df_train.drop(columns=cols_to_drop)
        print(f"Remaining features after removal: {feature_df_train.shape[1]}")

    # Use the same selected features for the entire dataset
    # These features have already had problematic ones removed
    selected_features = feature_df_train.columns
    print(f"Using {len(selected_features)} selected features (after removing problematic ones)")
    feature_df = df_clean[selected_features]

    # Create feature matrix and target
    X = feature_df.copy()
    X = X.fillna(method='ffill').fillna(0)
    y = df_clean['Target_1d']

    # Ensure X and y are aligned
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    df_clean = df_clean.sort_index()

    # Split the data using our masks
    X_train_full = X.loc[train_mask]
    y_train_full = y.loc[train_mask]
    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    # Print class distributions and date ranges for debugging
    print("Train class distribution:", np.bincount(y_train_full))
    print("Train date range:", X_train_full.index.min(), "to", X_train_full.index.max())
    print("Validation class distribution:", np.bincount(y_val))
    print("Validation date range:", X_val.index.min(), "to", X_val.index.max())
    print("Test class distribution:", np.bincount(y_test))
    print("Test date range:", X_test.index.min(), "to", X_test.index.max())

    # Diagnostics for data/feature leakage and scaling/alignment
    print("\n--- Data/Feature Leakage & Alignment Diagnostics ---")
    # Check for index overlap
    train_idx = set(X_train_full.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    print("Train/Val overlap:", len(train_idx & val_idx))
    print("Train/Test overlap:", len(train_idx & test_idx))
    print("Val/Test overlap:", len(val_idx & test_idx))

    # Check for NaNs or all-zero columns
    for split_name, split in [("Train", X_train_full), ("Val", X_val), ("Test", X_test)]:
        print(f"{split_name} NaNs:", split.isnull().any().any())
        print(f"{split_name} all-zero columns:", (split == 0).all().any())
        print(f"{split_name} duplicate columns:", split.columns.duplicated().any())
        print(f"{split_name} duplicate rows:", split.duplicated().any())

    # Check scaling: mean/std of each split
    print("Train feature mean/std (first 5):", np.round(X_train_full.mean(axis=0).values[:5], 4), np.round(X_train_full.std(axis=0).values[:5], 4))
    print("Val feature mean/std (first 5):", np.round(X_val.mean(axis=0).values[:5], 4), np.round(X_val.std(axis=0).values[:5], 4))
    print("Test feature mean/std (first 5):", np.round(X_test.mean(axis=0).values[:5], 4), np.round(X_test.std(axis=0).values[:5], 4))

    # Check for high correlation between features and target in test set (potential leakage)
    if hasattr(y_test, "values"):
        y_test_arr = y_test.values
    else:
        y_test_arr = y_test
    corr_with_target = []
    high_corr_features = []
    suspicious_features = []

    # Problematic features to exclude - these might be causing unrealistic performance
    problematic_features = [
        'Close_to_High', 'Close_to_Low',
        'Body_size', 'Upper_shadow', 'Lower_shadow',
        'BB_position', 'BB_squeeze', 'BB_Width',
        'High_Low_ratio', 'OBV',
        # Add more potentially problematic features
        'Volume_ratio', 'Volume_Shadow', 'Volume_trend',
        'Trend_strength', 'ATR_vs_BodySize'
    ]

    print(f"\n=== REMOVING POTENTIALLY PROBLEMATIC FEATURES ===")
    print(f"Excluding these features that might cause unrealistic performance: {problematic_features}")

    for col in X_test.columns:
        # Skip problematic features
        if any(feat in col for feat in problematic_features):
            print(f"Excluding feature: {col}")
            suspicious_features.append(col)
            continue

        try:
            # Use Spearman correlation which is more robust
            from scipy.stats import spearmanr
            corr, p_value = spearmanr(X_test[col], y_test_arr)
            if p_value < 0.05 and abs(corr) > 0.3:
                print(f"WARNING: Possible data leakage! Feature {col} has correlation {corr:.4f} with target (p={p_value:.4f})")
                high_corr_features.append((col, abs(corr), p_value))
                suspicious_features.append(col)
            corr_with_target.append(corr)
        except Exception as e:
            print(f"Error calculating correlation for {col}: {e}")
            corr_with_target.append(np.nan)
    max_corr = np.nanmax(np.abs(corr_with_target))
    print(f"Max abs corr(feature, target) in test set: {max_corr:.4f}")
    if max_corr > 0.3:
        print("ALERT: High feature-target correlation detected. Investigating for data leakage.")
        high_corr_features.sort(key=lambda x: x[1], reverse=True)
        print("Top correlated features:")
        for feature, corr, p_val in high_corr_features[:5]:
            print(f"  {feature}: corr={corr:.4f}, p={p_val:.4f}")

        # Filter out suspicious features with high correlation
        if suspicious_features:
            print(f"\nRemoving {len(suspicious_features)} suspicious features with high target correlation:")
            for feat in suspicious_features[:10]:  # Show first 10 only if many
                print(f"  - {feat}")
            if len(suspicious_features) > 10:
                print(f"  - ... and {len(suspicious_features) - 10} more")

            # Save removed features to a file for reference
            try:
                removed_features_file = os.path.join(debug_dir, "removed_features.txt")
                with open(removed_features_file, "w") as f:
                    f.write("# Features removed due to high correlation with target\n")
                    for feat in suspicious_features:
                        f.write(f"{feat}\n")
                print(f"Saved list of removed features to {removed_features_file}")
            except Exception as e:
                print(f"Error saving removed features: {e}")

            # Remove suspicious features from all splits
            X_train_full = X_train_full.drop(columns=suspicious_features, errors='ignore')
            X_val = X_val.drop(columns=suspicious_features, errors='ignore')
            X_test = X_test.drop(columns=suspicious_features, errors='ignore')
            print(f"Remaining features: {X_train_full.shape[1]}")
    print("--- End Diagnostics ---\n")



    # Fit scaler on full training data and transform validation set before loop
    scaler = StandardScaler()
    scaler.fit(X_train_full)
    X_val_scaled = scaler.transform(X_val)
    # === FILTER OUT CONSTANT FEATURES AND FEATURES WITH LEAKAGE ===
    def filter_constant_features(X_train, X_val, X_test):
        keep = []
        for col in X_train.columns:
            if X_train[col].nunique() > 1 and X_val[col].nunique() > 1 and X_test[col].nunique() > 1:
                keep.append(col)
        return keep

    # Define filters for potentially leaky feature names
    def has_leaky_pattern(feature_name):
        # Patterns that might indicate data leakage when NOT properly shifted
        leaky_patterns = [
            # Target info leakage
            'target', 'Target', 'label', 'Label',
            # Future data patterns
            'future', 'next', 'tomorrow', 'predict',
            # Cross-validation leakage
            'fold', 'split', 'test_', 'val_',
        ]
        return any(pattern in feature_name.lower() for pattern in leaky_patterns)

    # Additional filtering for suspicious feature names
    selected_features = list(X_test.columns)
    leaky_named_features = [col for col in selected_features if has_leaky_pattern(col)]
    if leaky_named_features:
        print(f"\nRemoving {len(leaky_named_features)} features with suspicious names:")
        for feat in leaky_named_features:
            print(f"  - {feat}")
        selected_features = [col for col in selected_features if col not in leaky_named_features]

    keep = filter_constant_features(X_train_full[selected_features], X_val[selected_features], X_test[selected_features])
    X_train_full = X_train_full[keep]
    X_val = X_val[keep]
    X_test = X_test[keep]

    # Drop all-zero columns and problematic features in any split
    def drop_all_zero_columns(*dfs):
        cols_to_drop = set()

        # Problematic features to exclude - these might be causing unrealistic performance
        problematic_features = [
            'Close_to_High', 'Close_to_Low',
            'Body_size', 'Upper_shadow', 'Lower_shadow',
            'BB_position', 'BB_squeeze', 'BB_Width',
            'High_Low_ratio', 'OBV',
            'Volume_ratio', 'Volume_Shadow', 'Volume_trend',
            'Trend_strength', 'ATR_vs_BodySize'
        ]

        # Add problematic features to drop list
        for df in dfs:
            for col in df.columns:
                if any(feat in col for feat in problematic_features):
                    cols_to_drop.add(col)

        # Add all-zero columns to drop list
        for df in dfs:
            cols_to_drop |= set(df.columns[(df == 0).all()])

        # Print actual dropped feature names for verification
        problematic_dropped = [col for col in cols_to_drop if any(feat in col for feat in problematic_features)]
        other_dropped = [col for col in cols_to_drop if not any(feat in col for feat in problematic_features)]

        print(f"=== FEATURE REMOVAL VERIFICATION ===")
        print(f"Dropping {len(problematic_dropped)} problematic features:")
        for feat in problematic_dropped:
            print(f"  - {feat}")
        print(f"Dropping {len(other_dropped)} other columns (all zeros, etc.)")
        for df in dfs:
            df.drop(columns=list(cols_to_drop), inplace=True, errors='ignore')
        return dfs

    X_train_full, X_val, X_test = drop_all_zero_columns(X_train_full, X_val, X_test)

    # Save indices before scaling
    train_index = X_train_full.index
    val_index = X_val.index
    test_index = X_test.index

    # Now fit scaler and proceed as before
    scaler = StandardScaler()
    # Store feature names before fitting
    feature_names = X_train_full.columns.tolist()
    scaler.fit(X_train_full)
    # Store the scaler in the predictor for use during prediction
    predictor.scaler = scaler
    # Explicitly attach feature names to the scaler for later consistency
    if hasattr(scaler, 'feature_names_in_'):
        scaler.feature_names_in_ = np.array(feature_names)
    # Also store selected features in predictor
    predictor.selected_features = feature_names

    # Save the final feature list for reference and consistency checks
    try:
        features_file = os.path.join(debug_dir, "selected_features.txt")
        with open(features_file, "w") as f:
            f.write("# Final selected features after filtering\n")
            for feat in feature_names:
                f.write(f"{feat}\n")
        print(f"Saved list of {len(feature_names)} selected features to {features_file}")

        # Verify problematic features were actually removed
        problematic_features = [
            'Close_to_High', 'Close_to_Low',
            'Body_size', 'Upper_shadow', 'Lower_shadow',
            'BB_position', 'BB_squeeze', 'BB_Width',
            'High_Low_ratio', 'OBV',
            'Volume_ratio', 'Volume_Shadow', 'Volume_trend',
            'Trend_strength', 'ATR_vs_BodySize'
        ]
        remaining_problematic = [feat for feat in feature_names if any(problem in feat for problem in problematic_features)]
        if remaining_problematic:
            print(f"WARNING: Some problematic features are still present after filtering:")
            for feat in remaining_problematic:
                print(f"  - {feat}")
        else:
            print(f"SUCCESS: All problematic features have been removed")
    except Exception as e:
        print(f"Error saving feature list: {e}")
    X_train_full = scaler.transform(X_train_full)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    predictor.scaler = scaler  # Ensure scaler is available for tomorrow's prediction

    # Create robust column names for our scaled data
    print(f"Handling DataFrame conversion with proper column alignment...")
    print(f"X_train_full shape: {X_train_full.shape}, keep length: {len(keep)}")

    # After scaling, we need to ensure column names are correctly aligned
    # Create generic column names based on actual column count
    generic_columns = [f"feature_{i}" for i in range(X_train_full.shape[1])]

    # Convert arrays to DataFrames with indices
    X_train_full = pd.DataFrame(X_train_full, index=train_index, columns=generic_columns)
    X_val = pd.DataFrame(X_val, index=val_index, columns=generic_columns[:X_val.shape[1]])
    X_test = pd.DataFrame(X_test, index=test_index, columns=generic_columns[:X_test.shape[1]])

    print(f"Successfully converted arrays to DataFrames with proper column counts")
    print(f"Final shapes - X_train: {X_train_full.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # === DIAGNOSTICS: Print test feature statistics before prediction ===



    # Prepare test dates for incremental walk-forward - ensure chronological order
    test_dates = X_test.index.sort_values()
    print(f"\n--- Walk-Forward Backtest Period ---")
    print(f"Backtesting from {test_dates.min()} to {test_dates.max()}")
    print(f"Total test days: {len(test_dates)}")

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
    # Use only the globally filtered feature set for all splits
    predictor.train_models(X_train_full, y_train_full, X_val, y_val, df_clean.loc[X_val.index, 'Close'])

    # --- Grid Search for Ensemble Weights/Thresholds with Caching (MOVED UP) ---
    model_list = ['rf', 'lr', 'dt', 'lgb', 'gb']
    # Adjust these weights to control model influence (higher = more influence)
    weights = {'rf': 1, 'lr': 1, 'dt': 1, 'lgb': 1, 'gb': 1}
    # *** COST-SENSITIVE THRESHOLD CONFIGURATION ***
    # Risk aversion factor: higher = more risk-averse, lower = more risk-seeking
    # 1.0 = neutral, 1.5 = moderately risk-averse, 0.8 = moderately risk-seeking
    risk_aversion = 1.0

    # Average market statistics (these will be calculated from data)
    avg_up_move = 0.018  # Default 1.8% average up move for Tesla
    avg_down_move = -0.015  # Default -1.5% average down move for Tesla
    transaction_cost = 0.001  # Default 0.1% transaction cost

    # Initial threshold (will be optimized using cost-sensitive approach)
    ensemble_threshold = 0.37
    model_thresholds = None
    # Always override cached weights with the ones we compute
    override_cached_weights = True
    try:
        from tesla_stock_predictor.models.ensemble import grid_search_model_thresholds
    except ImportError as e:
        print(f"Error importing grid_search_model_thresholds: {e}")
        print("Using default ensemble settings")
        best_ensemble_config = {
            "model_thresholds": {m: 0.5 for m in model_list},
            "ensemble_threshold": 0.5,
            "weights": weights
        }

    try:
        val_start_str = str(X_val.index.min()).replace(" ", "_").replace(":", "-")
        val_end_str = str(X_val.index.max()).replace(" ", "_").replace(":", "-")
        ensemble_config_path = f"tesla_stock_predictor/models/ensemble_config_{val_start_str}_{val_end_str}.json"
    except Exception as e:
        print(f"Error creating ensemble config path: {e}")
        ensemble_config_path = "tesla_stock_predictor/models/ensemble_config_default.json"

    # Grid search: more threshold values for each model, more ensemble threshold values
    # Use lower thresholds to address bias toward class 0
    fast_threshold_grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  # 8 values, adding lower options
    fast_ensemble_threshold_grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  # 8 values, adding lower options

    best_ensemble_config = None
    if os.path.exists(ensemble_config_path):
        with open(ensemble_config_path, "r") as f:
            try:
                best_ensemble_config = json.load(f)
                # Override with our custom weights if flag is set
                if override_cached_weights and best_ensemble_config:
                    print(f"\n--- Using custom weights: {weights} ---")
                    print(f"--- Using master threshold: {ensemble_threshold} ---")
                    best_ensemble_config["weights"] = weights
                    best_ensemble_config["ensemble_threshold"] = ensemble_threshold
            except Exception:
                best_ensemble_config = None
        # Validate loaded config
        if not best_ensemble_config or not isinstance(best_ensemble_config, dict) or "model_thresholds" not in best_ensemble_config or "ensemble_threshold" not in best_ensemble_config:
            print(f"Cached ensemble config is missing or invalid. Running grid search for {ensemble_config_path}...")
            best_ensemble_config = grid_search_model_thresholds(
                predictor,
                X_val,
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
            X_val,
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

    # Normalize all dates to date only (no time) for robust comparison
    trade_log_dates = set(pd.to_datetime(trade_log_df["Date"]).dt.normalize()) if not trade_log_df.empty else set()

    # Function to reset model cache when needed
    def reset_model_cache():
        """Delete all cached models to force retraining with the current feature set"""
        cleaned_models = 0
        print("\nRESETTING MODEL CACHE due to feature set changes...")
        for model_file in os.listdir(model_cache_dir):
            if model_file.endswith(".joblib"):
                try:
                    model_path = os.path.join(model_cache_dir, model_file)
                    os.remove(model_path)
                    cleaned_models += 1
                except Exception as e:
                    print(f"Error removing model {model_file}: {e}")
        print(f"Removed {cleaned_models} cached models. Will retrain with current feature set.")

    # Check if we need to reset the model cache (e.g., if feature count changed)
    # Create a feature count tracking file if it doesn't exist
    feature_count_file = os.path.join(model_cache_dir, "feature_count.txt")
    current_feature_count = X_train_full.shape[1]
    reset_cache = False

    if os.path.exists(feature_count_file):
        try:
            with open(feature_count_file, "r") as f:
                stored_count = int(f.read().strip())
            if stored_count != current_feature_count:
                print(f"Feature count changed: {stored_count} -> {current_feature_count}")
                reset_cache = True
        except Exception as e:
            print(f"Error reading feature count file: {e}")
            reset_cache = True
    else:
        # First run, create the file
        reset_cache = True

    if reset_cache:
        reset_model_cache()
        # Update the feature count file
        try:
            with open(feature_count_file, "w") as f:
                f.write(str(current_feature_count))
        except Exception as e:
            print(f"Error writing feature count file: {e}")

    # Clean up incompatible cached models
    print("\nChecking for incompatible cached models...")
    cleaned_models = 0
    for model_file in os.listdir(model_cache_dir):
        if model_file.endswith(".joblib"):
            try:
                model_path = os.path.join(model_cache_dir, model_file)
                model = load(model_path)
                if hasattr(model, 'n_features_in_'):
                    if model.n_features_in_ != X_train_full.shape[1]:
                        print(f"Removing incompatible model {model_file}: expects {model.n_features_in_} features, current data has {X_train_full.shape[1]}")
                        os.remove(model_path)
                        cleaned_models += 1
            except Exception as e:
                print(f"Error checking model {model_file}, removing: {e}")
                try:
                    os.remove(os.path.join(model_cache_dir, model_file))
                    cleaned_models += 1
                except:
                    pass
    print(f"Removed {cleaned_models} incompatible cached models.")

    # Record the number of features being used to detect model-data mismatches
    feature_count = X_train_full.shape[1]
    print(f"Current feature count: {feature_count}. Will be used to verify model compatibility.")

    # Set a maximum number of iterations for the main loop to prevent infinite execution
    max_date_iterations = 50
    processed_dates = 0

    for i, date in enumerate(test_dates):
        # Check if we've processed too many dates (prevents infinite loops)
        processed_dates += 1
        if processed_dates > max_date_iterations:
            print(f"WARNING: Maximum date iterations ({max_date_iterations}) reached. Breaking loop.")
            break
        date_norm = pd.to_datetime(date).normalize()
        if date_norm in trade_log_dates:
            continue  # Already processed

        # Use all train data up to yesterday
        train_mask = (X_train_full.index < date)
        X_train = X_train_full.loc[train_mask]
        y_train = y_train_full.loc[train_mask]

        # Define X_train_final with all columns initially
        X_train_final = X_train.copy()

        # Apply feature filtering if keep list exists
        if 'keep' in locals() and keep:
            # Get intersection of available columns and keep list
            available_keep = [col for col in keep if col in X_train.columns]
            if available_keep:
                X_train_final = X_train[available_keep]
                print(f"Applied keep list: {len(available_keep)}/{len(keep)} features available")
            else:
                print(f"WARNING: No features from keep list available in data")

        # Store the current set of features in the predictor
        if X_train_final.shape[1] > 0:
            predictor.selected_features = X_train_final.columns.tolist()

            # Also update the scaler's feature names for this training window
            if hasattr(scaler, 'feature_names_in_'):
                scaler.feature_names_in_ = np.array(X_train_final.columns)

        # Ensure X_train and y_train have the same indices
        common_indices = X_train.index.intersection(y_train.index)
        X_train = X_train.loc[common_indices]
        y_train = y_train.loc[common_indices]
        print(f"After alignment: X_train={X_train.shape}, y_train={y_train.shape}")

        # Verify alignment
        if not X_train.index.equals(y_train.index):
            print("WARNING: X_train and y_train indices are not aligned even after intersection")
            # Force exact alignment by index order
            y_train = y_train.loc[X_train.index]

        if len(X_train) < 50:
            continue

        # Initialize keep list if it doesn't exist yet
        if 'keep' not in locals():
            # Use all columns as default
            keep = X_train.columns.tolist()
            print(f"Initializing keep list with all {len(keep)} features")

        # Run a quick correlation check on the training data to catch any remaining leakage
        # This is specific to each walk-forward window
        if i == 0:  # Only check on first iteration to avoid repeated warnings
            leaky_cols = []
            for col in X_train.columns:
                try:
                    from scipy.stats import spearmanr
                    corr, p_value = spearmanr(X_train[col], y_train)
                    if p_value < 0.01 and abs(corr) > 0.5:  # Stricter threshold for training
                        print(f"HIGH ALERT: Strong correlation in walk-forward window for {col}: {corr:.4f} (p={p_value:.6f})")
                        leaky_cols.append(col)
                except Exception:
                    continue

            if leaky_cols:
                print(f"Removing {len(leaky_cols)} strongly correlated features from walk-forward training")
                X_train = X_train.drop(columns=leaky_cols, errors='ignore')
                # Also update global variables to maintain consistency
                X_train_full = X_train_full.drop(columns=leaky_cols, errors='ignore')
                X_val = X_val.drop(columns=leaky_cols, errors='ignore')
                X_test = X_test.drop(columns=leaky_cols, errors='ignore')
                # Update keep list
                keep = [col for col in keep if col not in leaky_cols]

        # Use global selected features and filtering for all walk-forward days
        X_today = X_test.loc[[date]]

        # Ensure keep list exists
        if 'keep' not in locals() or not keep:
            print(f"WARNING: keep list not defined or empty. Using all available features.")
            keep = X_train.columns.tolist()

        # Get intersection of available columns and keep list
        available_keep = [col for col in keep if col in X_train.columns]
        if not available_keep:
            print(f"CRITICAL: No features from keep list available in data. Using all columns.")
            X_train_final = X_train.copy()
            X_today_final = X_today.copy()
        else:
            X_train_final = X_train[available_keep]
            X_today_final = X_today[available_keep]

        # Print shape for debugging
        print(f"Shape before scaling: X_train_final={X_train_final.shape}, y_train={len(y_train)}")

        # Skip this day if no features left after filtering (should not happen, but safe)
        if X_train_final.shape[1] == 0:
            continue

        # Store the original indices to ensure alignment
        train_indices = X_train_final.index

        # Verify we have features to work with
        if X_train_final.shape[1] == 0:
            print("ERROR: No features available for scaling.")
            continue

        # First scale the data
        scaler = StandardScaler()
        try:
            # Save feature names for consistency
            feature_names = X_train_final.columns.tolist()
            print(f"Scaling {len(feature_names)} features")
            X_train_scaled = scaler.fit_transform(X_train_final)
            # Set feature names in scaler
            if hasattr(scaler, 'feature_names_in_'):
                scaler.feature_names_in_ = np.array(feature_names)
            X_today_scaled = scaler.transform(X_today_final)
        except Exception as e:
            print(f"ERROR during scaling: {e}")
            continue

        # Verify dimensions after scaling
        print(f"Shape after scaling: X_train_scaled={X_train_scaled.shape}, y_train={len(y_train)}")

        # Ensure y_train matches X_train_scaled exactly by reordering
        y_train = y_train.loc[train_indices]  # Use the same indices as X_train_final

        # Verify lengths match before SMOTE
        if X_train_scaled.shape[0] != len(y_train):
            print(f"CRITICAL ERROR: Length mismatch before SMOTE: X={X_train_scaled.shape[0]}, y={len(y_train)}")
            # Create consistent subset to avoid errors
            min_len = min(X_train_scaled.shape[0], len(y_train))
            X_train_scaled = X_train_scaled[:min_len]
            y_train = y_train.iloc[:min_len]

        # Final verification before SMOTE
        print(f"Final verification: X_train_scaled rows={X_train_scaled.shape[0]}, y_train length={len(y_train)}")

        # Then apply SMOTE to the scaled training data
        try:
            smote = SMOTE(random_state=42)
            X_train_for_model, y_train_for_model = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE: X_train_for_model={X_train_for_model.shape}, y_train_for_model={len(y_train_for_model)}")

        except ValueError as e:
            print(f"SMOTE error: {e}")
            print(f"X_train_scaled shape: {X_train_scaled.shape}, y_train shape: {y_train.shape}")
            # If SMOTE fails, continue without balancing
            X_train_for_model = X_train_scaled
            y_train_for_model = y_train.values if isinstance(y_train, pd.Series) else y_train

        # No need to scale again since we already scaled before SMOTE

        # Model caching: train and save/load all models for this day
        predictor.models = {}

        # Set a maximum number of iterations to prevent infinite loops
        max_iterations = 30
        current_iteration = 0
        for model_name in ['rf', 'lr', 'dt', 'lgb', 'gb']:
            # Check if we've exceeded maximum iterations to prevent infinite loops
            current_iteration += 1
            if current_iteration > max_iterations:
                print(f"WARNING: Maximum iterations ({max_iterations}) reached. Breaking loop.")
                break

            model_path = os.path.join(model_cache_dir, f"{model_name}_model_{date}.joblib")
            if os.path.exists(model_path):
                try:
                    # Load the model and check if feature count matches
                    cached_model = load(model_path)
                    # For RandomForestClassifier and similar models that have n_features_in_ attribute
                    if hasattr(cached_model, 'n_features_in_'):
                        if cached_model.n_features_in_ != X_train_for_model.shape[1]:
                            print(f"Feature count mismatch for {model_name}: model expects {cached_model.n_features_in_} features, but data has {X_train_for_model.shape[1]}. Removing and recomputing.")
                            os.remove(model_path)
                            raise ValueError("Feature count mismatch")
                    predictor.models[model_name] = cached_model
                except Exception as e:
                    print(f"Model cache for {model_name} on {date} is corrupted or incompatible. Recomputing. Error: {e}")
                    # Train and save model if not cached or cache is bad
                    best_params_path = "tesla_stock_predictor/models/best_params.json"
                    try:
                        import os  # Import os at local scope
                        os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
                        if os.path.exists(best_params_path):
                            with open(best_params_path, "r") as f:
                                best_params = json.load(f)
                            params = best_params.get(model_name, {})
                        else:
                            print(f"Best params file not found. Creating new one.")
                            best_params = {}
                            params = {}
                            with open(best_params_path, "w") as f:
                                json.dump(best_params, f)
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        print(f"Could not load best params: {e}. Using default parameters.")
                        params = {}
                    if model_name == 'rf':
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(**params)
                    elif model_name == 'gb':
                        from sklearn.ensemble import GradientBoostingClassifier
                    else:
                        continue
                    try:
                        # Always use the consistently defined training data
                        model.fit(X_train_for_model, y_train_for_model)
                        dump(model, model_path)
                        predictor.models[model_name] = model
                    except Exception as e:
                        print(f"Error training {model_name} model: {e}")
                        continue
            else:
                # Train and save model if not cached
                best_params_path = "tesla_stock_predictor/models/best_params.json"
                try:
                    import os  # Import os at local scope
                    os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
                    if os.path.exists(best_params_path):
                        with open(best_params_path, "r") as f:
                            best_params = json.load(f)
                        params = best_params.get(model_name, {})
                    else:
                        print(f"Best params file not found. Creating new one.")
                        best_params = {}
                        params = {}
                        with open(best_params_path, "w") as f:
                            json.dump(best_params, f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Could not load best params: {e}. Using default parameters.")
                    params = {}
                if model_name == 'rf':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(**params)
                elif model_name == 'gb':
                    from sklearn.ensemble import GradientBoostingClassifier
                elif model_name == 'dt':
                    from sklearn.tree import DecisionTreeClassifier
                    model = DecisionTreeClassifier(**params)
                elif model_name == 'lr':
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(**params)
                elif model_name == 'lgb':
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(**params)
                else:
                    continue
                try:
                    # Always use the consistently defined training data
                    model.fit(X_train_for_model, y_train_for_model)
                    dump(model, model_path)
                    predictor.models[model_name] = model
                except Exception as e:
                    print(f"Error training {model_name} model: {e}")
                    continue

        # Use ensemble to predict today's signal
        available_model_list = [m for m in predictor.models.keys()]
        if available_model_list:
            try:
                # Ensure X_today_scaled has the same number of features as expected by models
                if any(hasattr(predictor.models[m], 'n_features_in_') for m in available_model_list):
                    model_feature_counts = [predictor.models[m].n_features_in_ for m in available_model_list if hasattr(predictor.models[m], 'n_features_in_')]
                    if model_feature_counts and model_feature_counts[0] != X_today_scaled.shape[1]:
                        print(f"Feature mismatch during prediction: models expect {model_feature_counts[0]} features, but data has {X_today_scaled.shape[1]} features.")
                        # Call the reset function to clear all cached models
                        reset_model_cache()
                        # Also clear the current predictor models
                        predictor.models = {}
                        # Fallback to holding position for this iteration
                        signal = 0
                        continue

                # Use the cost-optimized threshold for predictions
                adjusted_ensemble_threshold = ensemble_threshold
                print(f"Using cost-optimized threshold for prediction: {adjusted_ensemble_threshold:.4f}")

                signal, *_ = predictor.ensemble_predict(
                    X_today_scaled,
                    model_list=available_model_list,
                    weights=best_ensemble_config.get("weights", weights),
                    threshold=adjusted_ensemble_threshold,
                    model_thresholds=best_ensemble_config.get("model_thresholds", model_thresholds)
                )
                signal = signal[0]  # Get scalar prediction for today
            except Exception as e:
                print(f"Error during prediction: {e}")
                # Fallback: prediction error, hold
                signal = 0
        else:
            # Fallback: no models available, hold
            signal = 0

        # Execute today's trade logic (simplified, you can expand as needed)
        open_price = df_clean.loc[date, 'Open']
        close_price = df_clean.loc[date, 'Close']
        high_price = df_clean.loc[date, 'High']
        low_price = df_clean.loc[date, 'Low']
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
    try:
        # Calculate average market statistics from the validation data
        val_returns = df_clean.loc[X_val.index, 'Close'].pct_change().dropna()
        up_days = val_returns > 0
        down_days = val_returns <= 0

        calculated_avg_up_move = val_returns[up_days].mean() if up_days.any() else 0.018
        calculated_avg_down_move = val_returns[down_days].mean() if down_days.any() else -0.015

        # Use calculated values if they're reasonable, otherwise use defaults
        avg_up_move = calculated_avg_up_move if 0.005 < calculated_avg_up_move < 0.05 else 0.018
        avg_down_move = calculated_avg_down_move if -0.05 < calculated_avg_down_move < -0.005 else -0.015

        print(f"\n--- Market Statistics for Cost-Sensitive Optimization ---")
        print(f"Average UP move: {avg_up_move:.4f} ({avg_up_move*100:.2f}%)")
        print(f"Average DOWN move: {avg_down_move:.4f} ({avg_down_move*100:.2f}%)")
        print(f"Risk aversion factor: {risk_aversion}")

        # Run cost-sensitive threshold optimization
        cost_result = backtest_with_cost_sensitive_threshold(
            predictor=predictor,
            X_val=X_val,
            y_val=y_val,
            df_clean=df_clean,
            weights=best_ensemble_config.get("weights", weights),
            avg_up_move=avg_up_move,
            avg_down_move=avg_down_move,
            transaction_cost=transaction_cost,
            risk_aversion=risk_aversion,
            save_plots=True
        )

        # Update the ensemble threshold with the optimized value
        ensemble_threshold = cost_result["ensemble_threshold"]
        adjusted_ensemble_threshold = ensemble_threshold

        print(f"\n--- Using cost-optimized threshold for validation: {adjusted_ensemble_threshold:.4f} ---")

        # First try standard threshold for validation metrics
        val_pred_standard, val_prob_standard, _, _, _ = predictor.ensemble_predict(
            X_val,
            model_list=model_list,
            weights=best_ensemble_config.get("weights", weights),
            threshold=best_ensemble_config.get("ensemble_threshold", threshold),
            model_thresholds=best_ensemble_config.get("model_thresholds", model_thresholds)
        )
        f1_standard = f1_score(y_val, val_pred_standard, average='macro')
        print(f"Validation F1 with standard threshold: {f1_standard:.4f} (higher is better, max=1.0)")

        # Then use adjusted threshold
        if adjusted_ensemble_threshold > 0.3:
            adjusted_ensemble_threshold = adjusted_ensemble_threshold * 0.8  # Reduce threshold by 20%

        val_pred, val_prob, val_conf, val_indiv_preds, val_indiv_probs = predictor.ensemble_predict(
            X_val,
            model_list=model_list,
            weights=best_ensemble_config.get("weights", weights),
            threshold=adjusted_ensemble_threshold,
            model_thresholds=best_ensemble_config.get("model_thresholds", model_thresholds)
        )
        f1 = f1_score(y_val, val_pred, average='macro')
        sharpe, total_return, max_drawdown, _ = backtest_financial_metrics(val_pred, X_val.index, df_clean)
        print(f"Validation F1: {f1:.4f}, Sharpe: {sharpe:.4f}, Max Drawdown: {max_drawdown:.4f}")
    except Exception as e:
        print(f"Error in validation predictions: {e}")
        print("Continuing with default model weights...")

    # Use the same logic for test set
    # Use best ensemble config for test predictions
    try:
        # Use the cost-optimized threshold for testing
        adjusted_ensemble_threshold = ensemble_threshold
        print(f"\n--- Using cost-optimized threshold for testing: {adjusted_ensemble_threshold:.4f} ---")

        # We're now using the cost-sensitive approach to find the optimal threshold
        # based on actual trading economics and risk preferences
        try:
            from tesla_stock_predictor.analysis.calibration import analyze_confidence_vs_correctness, find_optimal_threshold

            # We still calculate the optimal threshold for reference, but don't use it
            optimal_threshold, scores = find_optimal_threshold(
                y_val,
                val_prob,
                metric='f1',
                plot_graphs=False,
                save_path=f"{debug_dir}/threshold_optimization"
            )

            # Save optimal thresholds (both F1 and cost-sensitive) for reference
            with open(f"{debug_dir}/threshold_comparison.json", 'w') as f:
                json.dump({
                    'f1_optimal_threshold': float(optimal_threshold),
                    'f1_score': float(scores[np.argmax(scores)]),
                    'cost_sensitive_threshold': float(ensemble_threshold),
                    'risk_aversion': float(risk_aversion),
                    'avg_up_move': float(avg_up_move),
                    'avg_down_move': float(avg_down_move),
                    'transaction_cost': float(transaction_cost)
                }, f)

            print(f"Threshold comparison:")
            print(f"  - F1 optimal threshold: {optimal_threshold:.4f}")
            print(f"  - Cost-sensitive optimal threshold: {adjusted_ensemble_threshold:.4f}")
            print(f"Using cost-sensitive threshold that aligns with trading economics")

        except Exception as e:
            print(f"Error in threshold calculation: {e}")

        # Simple confidence analysis without plots
        try:
            from tesla_stock_predictor.analysis.calibration import analyze_confidence_vs_correctness

            # Run analysis on validation set
            val_metrics = analyze_confidence_vs_correctness(
                y_val,
                val_pred,
                val_conf,
                model_names=model_list,
                probs=val_indiv_probs,
                plot_graphs=False
            )
            print("Validation confidence analysis results:")
            print(f"  Mean confidence: {val_metrics['mean_confidence']:.4f}")
            print(f"  Mean confidence when correct: {val_metrics['mean_confidence_correct']:.4f}")
            print(f"  Mean confidence when incorrect: {val_metrics['mean_confidence_incorrect']:.4f}")
            print(f"  Correlation between confidence and correctness: {val_metrics['correlation']:.4f}")
        except Exception as e:
            print(f"Error in confidence analysis: {e}")

        test_pred, test_prob, test_conf, test_indiv_preds, test_indiv_probs = predictor.ensemble_predict(
            X_test,
            model_list=model_list,
            weights=best_ensemble_config.get("weights", weights),
            threshold=adjusted_ensemble_threshold,
            model_thresholds=best_ensemble_config.get("model_thresholds", model_thresholds)
        )

        # Simple test confidence analysis without plots
        try:
            from tesla_stock_predictor.analysis.calibration import analyze_confidence_vs_correctness

            # Run analysis on test set
            test_metrics = analyze_confidence_vs_correctness(
                y_test,
                test_pred,
                test_conf,
                model_names=model_list,
                probs=test_indiv_probs,
                plot_graphs=False
            )
            print("\nModel Reliability Analysis:")
            print(f"  Overall accuracy: {test_metrics['accuracy']:.4f} (higher is better)")
            print(f"  Average confidence: {test_metrics['mean_confidence']:.4f}")
            print(f"  Confidence when prediction is correct: {test_metrics['mean_confidence_correct']:.4f}")
            print(f"  Confidence when prediction is wrong: {test_metrics['mean_confidence_incorrect']:.4f}")
            print(f"  Interpretation: {'Good calibration' if test_metrics['mean_confidence_correct'] > test_metrics['mean_confidence_incorrect'] else 'Poor calibration'}")
        except Exception as e:
            print(f"Error in test confidence analysis: {e}")

        test_f1 = f1_score(y_test, test_pred, average='macro')
        test_accuracy = accuracy_score(y_test, test_pred)
    except Exception as e:
        print(f"Error in test predictions: {e}")
        # Create fallback predictions
        test_pred = np.zeros(len(y_test))
        test_prob = np.zeros(len(y_test))
        test_conf = np.zeros(len(y_test))
        test_indiv_preds = {}
        test_indiv_probs = {}
        test_f1 = 0
        test_accuracy = 0
        print("Using fallback predictions (all zeros).")

    # --- Print Model Performance Metrics ---
    print(f"\nEnsemble F1: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}")
    print("\nDetailed Classification Report:\n")
    print(classification_report(y_test, test_pred))

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

    for i in range(len(test_pred)):
        date = test_dates[i]
        open_price = df_clean.loc[date, 'Open']
        close_price = df_clean.loc[date, 'Close']
        high_price = df_clean.loc[date, 'High']
        low_price = df_clean.loc[date, 'Low']
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
                risk_per_share = 0.012 * trade_price  # 0.9% of entry price per share
                sl = trade_price - risk_per_share
                tp = trade_price + 2.76 * risk_per_share
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
                    elif min(open_price, high_price) <= tp <= max(open_price, high_price):
                        # Crossed TP intraday (between open and high): sell at TP
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
            else:
                action = "HOLD"

        elif position == 1 and sl is not None:
            # Check if we need to close the position based on stop-loss
            if low_price <= sl:  # Only execute SL if the price actually reaches/crosses it
                trade_price = max(low_price, sl)  # Realistic fill price
                cash += shares * trade_price
                cost = shares * trade_price * transaction_cost
                cash -= cost
                trade_pnl = (trade_price - last_buy_price) * shares if last_buy_price is not None else None
                action = "SELL_SL"
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
            elif high_price >= tp:  # Only execute TP if the price actually reaches/crosses it
                trade_price = min(high_price, tp)  # Realistic fill price
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
                action = "SELL_TP"
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

    # Calculate win/loss ratio from trade log
    win_trades = trade_log_df["Trade_PnL"].dropna() > 0
    loss_trades = trade_log_df["Trade_PnL"].dropna() < 0
    num_wins = win_trades.sum()
    num_losses = loss_trades.sum()
    win_loss_ratio = num_wins / num_losses if num_losses > 0 else float('inf')

    print(f"\n--- Financial Backtest Metrics (Test Set) ---")
    print(f"Cumulative Return: {total_return:.4f}")
    print(f"Annualized Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")
    print(f"Final Portfolio Value: ${portfolio[-1]:.2f}")
    print(f"Win/Loss Ratio: {win_loss_ratio:.2f} ({num_wins} wins / {num_losses} losses)")

    # --- Plot Equity Curve ---
    try:
        if 'visualize_equity_curve' in locals() and visualize_equity_curve:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(test_dates, portfolio[1:], label="Equity Curve", color="blue")
            plt.title(f"Equity Curve (Test Set: {test_dates.min().date()} to {test_dates.max().date()})")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"equity_curve_{test_dates.min().date()}_to_{test_dates.max().date()}.png")
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

    # Always run tomorrow's prediction block
    print("Attempting tomorrow's prediction...")
    try:
        # Check if we have models before trying to predict
        if not hasattr(predictor, 'models') or not predictor.models:
            print("No models available for prediction. Skipping tomorrow's prediction.")
        else:
            # Pass the full original df to predictor.predict_tomorrow
            prediction = predictor.predict_tomorrow(df)
            if prediction is not None:
                # --- Print Tomorrow's Prediction ---
                print("\n🔮 Tomorrow's Prediction:")
                print(f"Date: {prediction['date'].strftime('%Y-%m-%d')}")
                print(f"Signal: {prediction['signal']}")
                print(f"Probability: {prediction['probability']:.4f}")
                print(f"Confidence: {prediction['confidence']:.4f}")
                print("Individual model probabilities:")
                for model, prob in prediction['individual_models'].items():
                    print(f"  {model.upper()}: {prob:.4f}")
            else:
                print("\n🔮 Tomorrow's Prediction: No prediction needed (data for next day already exists).")
    except Exception as e:
        print("Exception during tomorrow's prediction:", e)
        logging.error(f"Error during tomorrow's prediction: {e}")
        logging.error(traceback.format_exc())
        print("Consider retraining models or checking feature engineering.")

if __name__ == "__main__":
    main()
