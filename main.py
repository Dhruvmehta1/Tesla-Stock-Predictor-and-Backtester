import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import json
import datetime
from datetime import datetime as dt
from sklearn.metrics import f1_score, accuracy_score, classification_report
from dotenv import load_dotenv
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from pathlib import Path

# Disable warnings for cleaner output
warnings.filterwarnings('ignore')

def plot_portfolio_value(trade_log):
    if trade_log.empty or 'Portfolio_Value' not in trade_log.columns:
        print("No portfolio data available to plot")
        return

    # Ensure the plots directory exists
    plots_dir = Path("debug/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create plot
    plt.figure(figsize=(12, 6))

    # Convert Date to datetime if it's not already
    if trade_log['Date'].dtype != 'datetime64[ns]':
        trade_log['Date'] = pd.to_datetime(trade_log['Date'])

    # Plot portfolio value
    plt.plot(trade_log['Date'], trade_log['Portfolio_Value'], linewidth=2)

    # Add detailed buy/sell markers
    buys = trade_log[trade_log['Action'].str.contains('BUY', na=False)]
    tp_sells = trade_log[trade_log['Action'].str.contains('TP', na=False)]
    sl_sells = trade_log[(trade_log['Action'].str.contains('SL', na=False)) &
                        (~trade_log['Action'].str.contains('TP', na=False))]
    other_sells = trade_log[(~trade_log['Action'].str.contains('BUY', na=False)) &
                           (~trade_log['Action'].str.contains('TP', na=False)) &
                           (~trade_log['Action'].str.contains('SL', na=False)) &
                           (trade_log['Action'] != 'HOLD')]

    if not buys.empty:
        plt.scatter(buys['Date'], buys['Portfolio_Value'], color='green', marker='^',
                   label='Buy', alpha=0.7, s=60)
    if not tp_sells.empty:
        plt.scatter(tp_sells['Date'], tp_sells['Portfolio_Value'], color='blue', marker='v',
                   label='Take Profit', alpha=0.7, s=60)
    if not sl_sells.empty:
        plt.scatter(sl_sells['Date'], sl_sells['Portfolio_Value'], color='red', marker='v',
                   label='Stop Loss', alpha=0.7, s=60)
    if not other_sells.empty:
        plt.scatter(other_sells['Date'], other_sells['Portfolio_Value'], color='orange', marker='v',
                   label='Other Sell', alpha=0.7, s=60)

    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    # Add labels and title
    plt.title('Portfolio Value Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add shaded areas for profit/loss days
    if len(trade_log) > 1:
        for i in range(1, len(trade_log)):
            prev_val = trade_log['Portfolio_Value'].iloc[i-1]
            curr_val = trade_log['Portfolio_Value'].iloc[i]
            if curr_val > prev_val:
                plt.axvspan(trade_log['Date'].iloc[i-1], trade_log['Date'].iloc[i],
                           alpha=0.1, color='green')
            elif curr_val < prev_val:
                plt.axvspan(trade_log['Date'].iloc[i-1], trade_log['Date'].iloc[i],
                           alpha=0.1, color='red')

    plt.legend(loc='upper left')

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Save plot, overwriting any existing file
    plot_path = plots_dir / "portfolio_value.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Portfolio value plot saved to {plot_path}")

    # Close the plot to free memory
    plt.close()


def backtest_financial_metrics(preds, X_idx, df_clean, sl_pct=0.01, tp_sl_ratio=3.0):
    """
    Fixed backtesting function with improved error handling and validation.
    """
    # Validate inputs
    if len(preds) == 0 or len(X_idx) == 0:
        print("WARNING: Empty predictions or index provided to backtesting")
        empty_log = pd.DataFrame({
            'Date': [df_clean.index[0] if not df_clean.empty else pd.Timestamp.now()],
            'Action': ['NO_DATA'], 'Signal': [0], 'Open_Price': [0], 'Close_Price': [0],
            'Trade_Price': [np.nan], 'Shares': [0], 'Cash': [10000], 'Equity': [0],
            'Portfolio_Value': [10000], 'Position': [0], 'Trade_PnL': [np.nan],
            'Transaction_Cost': [0], 'Borrow_Cost': [0]
        })
        return 0, 0, 0, ([], [], [], 0, 0, empty_log)

    # Ensure we have the required columns
    required_cols = ['Open', 'Close', 'High', 'Low']
    missing_cols = [col for col in required_cols if col not in df_clean.columns]
    if missing_cols:
        print(f"WARNING: Missing required columns: {missing_cols}")
        empty_log = pd.DataFrame({
            'Date': X_idx[:1], 'Action': ['NO_DATA'], 'Signal': [0], 'Open_Price': [0], 'Close_Price': [0],
            'Trade_Price': [np.nan], 'Shares': [0], 'Cash': [10000], 'Equity': [0],
            'Portfolio_Value': [10000], 'Position': [0], 'Trade_PnL': [np.nan],
            'Transaction_Cost': [0], 'Borrow_Cost': [0]
        })
        return 0, 0, 0, ([], [], [], 0, 0, empty_log)

    # Create a trade log DataFrame
    trade_log_data = []

    try:
        # Get price data with proper error handling - align indices
        aligned_data = df_clean.reindex(X_idx).dropna(subset=['Open', 'Close', 'High', 'Low'])

        if aligned_data.empty:
            print("WARNING: No valid price data after alignment")
            empty_log = pd.DataFrame({
                'Date': X_idx[:1], 'Action': ['NO_DATA'], 'Signal': [0], 'Open_Price': [0], 'Close_Price': [0],
                'Trade_Price': [np.nan], 'Shares': [0], 'Cash': [10000], 'Equity': [0],
                'Portfolio_Value': [10000], 'Position': [0], 'Trade_PnL': [np.nan],
                'Transaction_Cost': [0], 'Borrow_Cost': [0]
            })
            return 0, 0, 0, ([], [], [], 0, 0, empty_log)

        # Align predictions with available data
        valid_indices = aligned_data.index
        preds_series = pd.Series(preds, index=X_idx)
        aligned_preds = preds_series.reindex(valid_indices).dropna()

        if len(aligned_preds) == 0:
            print("WARNING: No predictions align with valid price data")
            empty_log = pd.DataFrame({
                'Date': X_idx[:1], 'Action': ['NO_DATA'], 'Signal': [0], 'Open_Price': [0], 'Close_Price': [0],
                'Trade_Price': [np.nan], 'Shares': [0], 'Cash': [10000], 'Equity': [0],
                'Portfolio_Value': [10000], 'Position': [0], 'Trade_PnL': [np.nan],
                'Transaction_Cost': [0], 'Borrow_Cost': [0]
            })
            return 0, 0, 0, ([], [], [], 0, 0, empty_log)

    except Exception as e:
        print(f"WARNING: Error accessing price data: {e}")
        empty_log = pd.DataFrame({
            'Date': X_idx[:1], 'Action': ['NO_DATA'], 'Signal': [0], 'Open_Price': [0], 'Close_Price': [0],
            'Trade_Price': [np.nan], 'Shares': [0], 'Cash': [10000], 'Equity': [0],
            'Portfolio_Value': [10000], 'Position': [0], 'Trade_PnL': [np.nan],
            'Transaction_Cost': [0], 'Borrow_Cost': [0]
        })
        return 0, 0, 0, ([], [], [], 0, 0, empty_log)

    # Initialize tracking variables
    returns = []
    cash = 10000.0  # Start with $10k
    shares = 0
    positions = []
    trades = []
    drawdowns = []
    equity_curve = []
    max_equity = cash

    # Track entry price for proper P&L calculation
    entry_price = 0.0
    entry_date = None

    # For tracking win/loss
    win_count = 0
    loss_count = 0

    # Position state tracking
    position_state = 'CASH'  # 'CASH', 'LONG', 'PENDING_BUY'

    # Add slippage and more realistic costs
    slippage_bps = 5  # 5 basis points slippage

    transaction_cost_pct = 0.0015  # 0.15% transaction cost

    # Iterate through each day with aligned data
    for i, (date, row) in enumerate(aligned_data.iterrows()):
        if date not in aligned_preds.index:
            continue

        open_price = row['Open']
        close_price = row['Close']
        high_price = row['High']
        low_price = row['Low']
        pred = int(aligned_preds.loc[date])

        # Skip if any price is missing or invalid
        if pd.isna(open_price) or pd.isna(close_price) or open_price <= 0 or close_price <= 0:
            continue

        # Initialize daily variables
        trade_pnl = np.nan
        trade_price = np.nan
        transaction_cost = 0.0
        action = 'HOLD'

        # Check if we need to execute a pending buy from previous day's signal
        if position_state == 'PENDING_BUY' and shares == 0:
            # Execute buy at today's open price (realistic timing)
            available_cash = cash * 0.995  # Leave 0.5% buffer for costs
            execution_price = open_price * (1 + slippage_bps / 10000)  # Add slippage
            max_shares = int(available_cash / execution_price) if execution_price > 0 else 0

            if max_shares > 0 and execution_price > 0:
                shares = max_shares
                cost = shares * execution_price
                transaction_cost = cost * transaction_cost_pct
                total_cost = cost + transaction_cost

                if cash >= total_cost:
                    cash -= total_cost
                    entry_price = execution_price
                    entry_date = date
                    position_state = 'LONG'

                    trade_price = execution_price
                    action = 'BUY_EXECUTED'

                    trades.append((date, action, shares, execution_price, cost))
                    positions.append((date, 'LONG', shares, execution_price, cash))
                else:
                    # Not enough cash, cancel buy
                    position_state = 'CASH'
                    action = 'BUY_CANCELLED'
            else:
                position_state = 'CASH'
                action = 'BUY_CANCELLED'

        # Check for TP/SL exits if we have a position
        if shares > 0 and position_state == 'LONG' and entry_price > 0:
            # Calculate TP/SL levels based on entry price
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + sl_pct * tp_sl_ratio)

            # Check for gap openings
            gap_tp_hit = open_price >= tp_price
            gap_sl_hit = open_price <= sl_price

            # Check for intraday TP/SL hits
            intraday_tp_hit = high_price >= tp_price
            intraday_sl_hit = low_price <= sl_price

            # Combine gap and intraday hits
            tp_hit = gap_tp_hit or intraday_tp_hit
            sl_hit = gap_sl_hit or intraday_sl_hit

            # Always prioritize TP over SL if both hit
            if tp_hit and sl_hit:
                sl_hit = False

            # Execute TP
            if tp_hit:
                execution_price = tp_price * (1 - slippage_bps / 10000)
                proceeds = shares * execution_price
                transaction_cost = proceeds * transaction_cost_pct
                net_proceeds = proceeds - transaction_cost

                trade_pnl = (execution_price - entry_price) * shares - transaction_cost
                cash += net_proceeds

                trades.append((date, 'SELL_TP', shares, execution_price, proceeds))
                positions.append((date, 'CASH', 0, 0, cash))

                shares = 0
                entry_price = 0.0
                entry_date = None
                position_state = 'CASH'
                trade_price = execution_price
                action = 'SELL_TP'
                win_count += 1

            # Execute SL
            elif sl_hit:
                execution_price = sl_price * (1 + slippage_bps / 10000)
                proceeds = shares * execution_price
                transaction_cost = proceeds * transaction_cost_pct
                net_proceeds = proceeds - transaction_cost

                trade_pnl = (execution_price - entry_price) * shares - transaction_cost
                cash += net_proceeds

                trades.append((date, 'SELL_SL', shares, execution_price, proceeds))
                positions.append((date, 'CASH', 0, 0, cash))

                shares = 0
                entry_price = 0.0
                entry_date = None
                position_state = 'CASH'
                trade_price = execution_price
                action = 'SELL_SL'
                loss_count += 1

        # Check for new buy signals (only if not in position)
        if position_state == 'CASH' and pred == 1 and shares == 0:
            position_state = 'PENDING_BUY'
            action = 'BUY_SIGNAL'

        # Calculate current equity
        if shares > 0:
            current_value = shares * close_price
            equity = cash + current_value
        else:
            equity = cash

        equity_curve.append((date, equity))

        # Track maximum equity for drawdown calculation
        max_equity = max(max_equity, equity)
        current_drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
        drawdowns.append(current_drawdown)

        # Calculate daily returns
        if i > 0 and len(equity_curve) > 1:
            daily_return = (equity - equity_curve[i-1][1]) / equity_curve[i-1][1]
        else:
            daily_return = 0
        returns.append(daily_return)

        # Add to trade log data
        position_value = shares * close_price if shares > 0 else 0

        trade_log_data.append({
            'Date': date, 'Action': action, 'Signal': pred, 'Open_Price': open_price,
            'Close_Price': close_price, 'Trade_Price': trade_price, 'Shares': shares,
            'Cash': cash, 'Equity': position_value, 'Portfolio_Value': equity,
            'Position': 1 if shares > 0 else 0, 'Trade_PnL': trade_pnl,
            'Transaction_Cost': transaction_cost, 'Borrow_Cost': 0.0
        })

        # Safety checks
        if cash < 0:
            print(f"WARNING: Cash went negative on {date}: ${cash:.2f}")
            cash = max(cash, 0)

    # Create trade log DataFrame
    trade_log = pd.DataFrame(trade_log_data)

    # Calculate final metrics
    if equity_curve and len(equity_curve) > 1:
        final_value = equity_curve[-1][1]
        total_return = (final_value / 10000.0) - 1

        # Calculate Sharpe ratio from equity curve returns
        equity_values = [point[1] for point in equity_curve]
        if len(equity_values) > 1:
            portfolio_returns = np.diff(equity_values) / equity_values[:-1]
            # Remove any infinite or NaN returns
            portfolio_returns = portfolio_returns[np.isfinite(portfolio_returns)]

            if len(portfolio_returns) > 0 and np.std(portfolio_returns) > 0:
                sharpe = np.sqrt(252) * (np.mean(portfolio_returns) / np.std(portfolio_returns))
            else:
                sharpe = 0
        else:
            sharpe = 0
    else:
        total_return = 0
        sharpe = 0

    max_drawdown = max(drawdowns) if drawdowns else 0

    # Save trade log to CSV
    os.makedirs("debug", exist_ok=True)
    if not trade_log.empty:
        trade_log.to_csv("debug/trade_log.csv", index=False)

    # Return summary metrics
    return sharpe, total_return, max_drawdown, (equity_curve, trades, positions, win_count, loss_count, trade_log)


class TSLAPredictor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.scaler = None
        self.models = {}
        self.feature_names = []
        self.selected_features = []

    def get_polygon_data(self, ticker, start_date, end_date):
        try:
            import sys
            sys.path.append(".")  # Add current directory to path
            from data.polygon_data import get_polygon_data
            return get_polygon_data(ticker, start_date, end_date, self.api_key)
        except ImportError as e:
            print(f"ERROR: Could not import polygon_data module: {e}")
            raise
        except Exception as e:
            print(f"ERROR: Failed to get data for {ticker}: {e}")
            raise

    def engineer_features(self, df):
        try:
            import sys
            sys.path.append(".")  # Add current directory to path
            from features.engineering import engineer_features
            return engineer_features(df)
        except ImportError as e:
            print(f"ERROR: Could not import engineering module: {e}")
            raise
        except Exception as e:
            print(f"ERROR: Failed to engineer features: {e}")
            raise

    def select_features(self, df):
        try:
            import sys
            sys.path.append(".")  # Add current directory to path
            from features.engineering import select_features
            return select_features(df)
        except ImportError as e:
            print(f"ERROR: Could not import feature selection: {e}")
            raise

    def train_models(self, X_train, y_train, X_val, y_val, close_prices_val):
        try:
            import sys
            sys.path.append(".")  # Add current directory to path
            from models.training import ModelTrainer
            trainer = ModelTrainer()
            trainer.train_models(X_train, y_train, X_val, y_val, close_prices_val)
            self.models = trainer.models
        except ImportError as e:
            print(f"ERROR: Could not import training module: {e}")
            raise
        except Exception as e:
            print(f"ERROR: Failed to train models: {e}")
            raise

    def ensemble_predict(self, X, model_list=None, weights=None, threshold=None, model_thresholds=None):
        try:
            import sys
            sys.path.append(".")  # Add current directory to path
            from models.ensemble import ensemble_predict
            return ensemble_predict(self, X, model_list, weights, threshold, model_thresholds)
        except ImportError as e:
            print(f"ERROR: Could not import ensemble module: {e}")
            raise
        except Exception as e:
            print(f"ERROR: Ensemble prediction failed: {e}")
            raise

    def predict_tomorrow(self, df):
        try:
            import sys
            sys.path.append(".")  # Add current directory to path
            from models.ensemble import predict_tomorrow
            return predict_tomorrow(self, df)
        except ImportError as e:
            print(f"ERROR: Could not import prediction module: {e}")
            return None
        except Exception as e:
            print(f"ERROR: Tomorrow prediction failed: {e}")
            return None


def main():
    print("Starting Tesla Stock Predictor...")

    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("ERROR: POLYGON_API_KEY not found in environment variables or .env file")
        print("Make sure you have a .env file with POLYGON_API_KEY=your_api_key")
        return None

    try:
        # Initialize predictor
        predictor = TSLAPredictor(api_key)

        # Define fixed date splits for consistent backtesting
        TRAIN_START_DATE = "2023-08-22 04:00:00"
        TRAIN_END_DATE = "2024-08-20 04:00:00"
        VAL_START_DATE = "2024-08-21 04:00:00"
        VAL_END_DATE = "2025-02-20 04:00:00"
        TEST_START_DATE = "2025-02-21 04:00:00"

        # Format dates properly for API calls
        start_date_str = pd.to_datetime(TRAIN_START_DATE).strftime("%Y-%m-%d")
        end_date = dt.now().strftime("%Y-%m-%d")

        print(f"Fetching data from {start_date_str} to {end_date}")

        # Get Tesla data
        df = predictor.get_polygon_data("TSLA", start_date_str, end_date)
        if df.empty:
            print("ERROR: No Tesla data retrieved")
            return None

        # Get sector ETF data for relative features
        try:
            sector_df = predictor.get_polygon_data("XLY", start_date_str, end_date)
            spy_df = predictor.get_polygon_data("SPY", start_date_str, end_date)

            # Rename columns for clarity before merging
            sector_df = sector_df.rename(columns={"Close": "Sector_Close"})
            spy_df = spy_df.rename(columns={"Close": "SPY_Close"})

            # Merge sector and SPY close prices into TSLA dataframe
            df = df.merge(sector_df[["Sector_Close"]], left_index=True, right_index=True, how="left")
            df = df.merge(spy_df[["SPY_Close"]], left_index=True, right_index=True, how="left")
        except Exception as e:
            print(f"WARNING: Could not fetch sector/SPY data: {e}")
            # Continue without sector data

        print(f"Retrieved {len(df)} data points")

        # Suppress pandas warnings for cleaner output
        warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

        # Split RAW data FIRST (before any feature engineering)
        train_mask = (df.index >= pd.to_datetime(TRAIN_START_DATE)) & (df.index <= pd.to_datetime(TRAIN_END_DATE))
        val_mask = (df.index >= pd.to_datetime(VAL_START_DATE)) & (df.index <= pd.to_datetime(VAL_END_DATE))
        test_mask = (df.index > pd.to_datetime(TEST_START_DATE))

        # Create clean raw data splits
        df_train_raw = df.loc[train_mask].copy()
        df_val_raw = df.loc[val_mask].copy()
        df_test_raw = df.loc[test_mask].copy()

        print(f"Data splits: Train={len(df_train_raw)}, Val={len(df_val_raw)}, Test={len(df_test_raw)}")

        if len(df_train_raw) == 0:
            print("ERROR: No training data available")
            return None

        # Process features following the Sacred Order
        try:
            import sys
            sys.path.append(".")  # Add current directory to path
            from features.processing import process_features

            # Process all features (steps 1-5 of the Sacred Order)
            results = process_features(df_train_raw, df_val_raw, df_test_raw)
        except ImportError as e:
            print(f"ERROR: Could not import processing module: {e}")
            return None
        except Exception as e:
            print(f"ERROR: Feature processing failed: {e}")
            return None

        # Extract processed data
        X_train = results['X_train']
        y_train = results['y_train']
        X_val = results.get('X_val')
        y_val = results.get('y_val')
        X_val_scaled = results.get('X_val_scaled')
        X_test = results.get('X_test')
        y_test = results.get('y_test')
        X_test_scaled = results.get('X_test_scaled')

        # Store scaler in predictor
        predictor.scaler = results['scaler']
        predictor.selected_features = results['feature_names']

        # Combine splits for backtesting purposes
        df_clean = pd.concat([df_train_raw, df_val_raw, df_test_raw])

        print(f"Features processed: {len(results['feature_names'])} features selected")

        # Model Training
        if X_val is not None and y_val is not None:
            print("Training models...")
            # Safely get close prices for validation data
            try:
                val_close_prices = df_clean.loc[X_val.index, 'Close']
            except KeyError:
                # Handle case where some validation indices might not exist in df_clean
                val_close_prices = df_clean.reindex(X_val.index)['Close'].dropna()
                if val_close_prices.empty:
                    print("WARNING: No close prices available for validation data")
                    val_close_prices = pd.Series([100.0] * len(X_val), index=X_val.index)  # Dummy values

            predictor.train_models(X_train, y_train, X_val, y_val, val_close_prices)
            print("Models trained successfully")
        else:
            print("ERROR: No validation data available for training")
            return None

        # Ensemble configuration
        model_list = ['rf', 'lr', 'dt', 'lgb', 'gb']
        weights = {
            'rf': 2.0,   # Random Forest
            'lr': 0,     # Logistic Regression: excluded
            'dt': 0,     # Decision Tree: excluded
            'lgb': 1.5,  # LightGBM
            'gb': 0.5    # Gradient Boosting
        }

        # Import grid search function
        try:
            from models.ensemble import grid_search_model_thresholds

            # Get ensemble configuration
            val_start_str = pd.to_datetime(VAL_START_DATE).strftime("%Y%m%d")
            val_end_str = pd.to_datetime(VAL_END_DATE).strftime("%Y%m%d")
            ensemble_config_path = f"models/ensemble_config_{val_start_str}_{val_end_str}.json"

            # Force reconfiguration
            if os.path.exists(ensemble_config_path):
                try:
                    os.remove(ensemble_config_path)
                    print(f"Deleted cached ensemble config: {ensemble_config_path}")
                except Exception as e:
                    print(f"Failed to delete cached config: {e}")

            # Get optimized ensemble configuration
            if X_val_scaled is not None:
                print("Optimizing ensemble configuration...")
                best_ensemble_config = grid_search_model_thresholds(
                    predictor, X_val_scaled, y_val, df_clean, X_val,
                    model_list=model_list, weights=weights
                )

                # Save config
                models_dir = os.path.dirname(ensemble_config_path)
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir, exist_ok=True)

                with open(ensemble_config_path, "w") as f:
                    json.dump({
                        "model_thresholds": best_ensemble_config["model_thresholds"],
                        "ensemble_threshold": 0.48,
                        "weights": best_ensemble_config.get("weights", None)
                    }, f)

                # Use optimized parameters
                model_thresholds = best_ensemble_config.get("model_thresholds", {model: 0.5 for model in model_list})
                ensemble_threshold = 0.46
            else:
                print("WARNING: No validation data for ensemble optimization, using defaults")
                model_thresholds = {model: 0.5 for model in model_list}
                ensemble_threshold = 0.5

        except ImportError as e:
            print(f"WARNING: Could not import ensemble grid search: {e}")
            model_thresholds = {model: 0.5 for model in model_list}
            ensemble_threshold = 0.5

        # Make predictions with optimized ensemble
        if X_val_scaled is not None:
            val_pred, val_prob, val_conf, val_indiv_preds, val_indiv_probs = predictor.ensemble_predict(
                X_val_scaled, model_list=model_list, weights=weights,
                threshold=ensemble_threshold, model_thresholds=model_thresholds
            )

        # Get predictions on test data
        if X_test_scaled is not None and y_test is not None:
            test_pred, test_prob, test_conf, test_indiv_preds, test_indiv_probs = predictor.ensemble_predict(
                X_test_scaled, model_list=model_list, weights=weights,
                threshold=ensemble_threshold, model_thresholds=model_thresholds
            )

            # Evaluate model performance
            test_f1 = f1_score(y_test, test_pred, average='macro')
            test_accuracy = accuracy_score(y_test, test_pred)

            # Print classification results
            print(f"Ensemble F1: {test_f1:.4f}, Accuracy: {test_accuracy:.4f} (Threshold: 0.48)")
            print("\nDetailed Classification Report:")
            print(classification_report(y_test, test_pred))

            # Run backtesting on test data with realistic parameters
            test_sharpe, test_return, test_drawdown, test_details = backtest_financial_metrics(
                test_pred, X_test.index, df_clean, sl_pct=0.015, tp_sl_ratio=2.7
            )

            # Unpack the details
            equity_curve, trades, positions, win_count, loss_count, trade_log = test_details

            if not trade_log.empty:
                # Display trade log for last 10 trading days
                last_10_trades = trade_log.tail(10)
                print("\n--- Trade Log for Last 10 Trading Days ---")
                print(last_10_trades.to_string(index=False))
                print("Trade log saved to debug/trade_log.csv")

                # Print financial metrics
                print("\n--- Financial Backtest Metrics (Test Set) ---")
                print(f"Cumulative Return: {test_return:.4f}")
                print(f"Annualized Sharpe Ratio: {test_sharpe:.4f}")
                print(f"Max Drawdown: {test_drawdown:.4f}")
                print(f"Final Portfolio Value: ${trade_log['Portfolio_Value'].iloc[-1]:.2f}")
                win_loss_ratio = f"{win_count / (win_count + loss_count):.2f}" if (win_count + loss_count) > 0 else "N/A"
                print(f"Win/Loss Ratio: {win_loss_ratio} ({win_count} wins / {loss_count} losses)")

                # Print additional stats
                tp_sells = len(trade_log[trade_log['Action'].str.contains('TP', na=False)])
                sl_sells = len(trade_log[(trade_log['Action'].str.contains('SL', na=False)) &
                              (~trade_log['Action'].str.contains('TP', na=False))])
                print(f"Take-Profit Exits: {tp_sells}, Stop-Loss Exits: {sl_sells}")

                # Plot portfolio value over time
                plot_portfolio_value(trade_log)

                # Create prediction results dataframe for last 10 days
                if len(test_pred) >= 10:
                    test_results = pd.DataFrame({
                        'Date': X_test.index[-10:],
                        'Actual': y_test[-10:],
                        'Ensemble_Pred': test_pred[-10:],
                        'Ensemble_Prob': test_prob[-10:],
                        'Confidence': test_conf[-10:],
                        'Close_Price': df_clean.loc[X_test.index[-10:], 'Close']
                    })

                    # Add individual model probabilities
                    for model_name in model_list:
                        if model_name in test_indiv_probs and len(test_indiv_probs[model_name]) >= 10:
                            test_results[f'{model_name.upper()}_Prob'] = test_indiv_probs[model_name][-10:]

                    # Print last 10 test predictions
                    print("\nLast 10 Test Predictions vs Actual:")
                    print("(1=UP movement, 0=DOWN movement compared to previous day)")
                    print("(Ensemble_Prob = predicted probability of UP movement)")
                    print("(Confidence = agreement between models)")
                    print(test_results.to_string(index=False))
            else:
                print("WARNING: No trade log data available for analysis")
        else:
            print("WARNING: No test data available for evaluation")

        # Get next day prediction
        try:
            tomorrow_result = predictor.predict_tomorrow(df_clean)

            if tomorrow_result is not None:
                print("\nNext Trading Day Prediction:")
                print(f"Date: {tomorrow_result['date'].strftime('%Y-%m-%d')}")
                print(f"Signal: {tomorrow_result['signal']}")
                print(f"Probability: {tomorrow_result['probability']:.4f}")
                print(f"Confidence: {tomorrow_result['confidence']:.4f}")
                print("\nIndividual Model Probabilities:")
                for model_name, prob in tomorrow_result['individual_models'].items():
                    print(f"  {model_name.upper()}: {prob:.4f}")

                signal_strength = "Strong" if abs(tomorrow_result['probability'] - 0.48) > 0.1 else "Moderate" if abs(tomorrow_result['probability'] - 0.48) > 0.05 else "Weak"
                print(f"\nSignal Strength: {signal_strength}")
                print(f"Latest Close Price: ${df_clean['Close'].iloc[-1]:.2f}")
            else:
                print("\nNext Trading Day Prediction:")
                print("Unable to generate prediction for the next trading day.")
        except Exception as e:
            print(f"WARNING: Failed to generate tomorrow's prediction: {e}")

    except Exception as e:
        print(f"ERROR: Main execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("Tesla Stock Predictor completed successfully!")


if __name__ == "__main__":
    # Run the main program
    main()
