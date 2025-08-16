# Tesla Stock Predictor

A machine learning project for predicting TSLA stock movements, developed entirely through AI pair programming.

## Project Status & Philosophy

This project is a living record of my journey building a machine learning pipeline for Tesla stock prediction.  
All results, features, and methods are subject to change as I learn, experiment, and improve the system.

- **Metrics shown in this repository are for illustration only and may not reflect future or out-of-sample performance.**
- **Some features described in the documentation are planned or experimental, not guaranteed to be present in every version.**
- **I welcome suggestions, feedback, and collaboration as the project evolves.**
- **Important: Results may change between runs if fresh data is downloaded from Polygon.io, as their historical data can be updated or corrected. For fully reproducible backtests, always cache and use a local copy of the data.**
- **Data Source Debugging: The project includes built-in measures to track and identify when external data sources change, helping distinguish between code issues and data source variability.**
- **Historical trade reproducibility:** The pipeline now guarantees that historical trades and ensemble predictions remain unchanged when only a new trading day is added, unless the underlying historical data itself changes. This was achieved by fixing all sources of non-comparable results, including feature engineering, model training, and trade logging. Data fetching from Polygon was not the root cause—internal pipeline logic and caching were the main factors.

## Core Features (Current State)
- Random Forest, Decision Tree, LightGBM ensemble
- Optuna-optimized ensemble weights
- Transaction costs included in backtesting
- Extensive feature engineering (technical indicators, lags, volatility, etc.)
- No data leakage (all features use only past/current data)
- Robust SL/TP logic: Exits are triggered if TP/SL is hit at the open, between open and close, or between open and high (for TP), simulating realistic intraday execution with daily data.
- All trades (including "HOLD" days) are logged and saved to `debug/trade_log.csv` for full transparency and auditability.
- Only the last 10 trades are printed in the terminal; the CSV contains the full trade history.
- Defensive programming: checks for `None` and empty DataFrames prevent runtime errors in feature engineering and caching.
- The unified backtest is the only source of official metrics and trade logs; validation metrics are for reference only.
- Data hash tracking and snapshotting ensure you know when historical data changes, supporting reproducibility.

---

## Latest Discovery: Simulating Intraday Trades with Only Daily Data

- **Workaround for No Intraday Data:**  
  - Our biggest breakthrough was finding a way around the limitation of not having access to real intraday data. By using just the open and close prices, we cleverly simulated intraday price movement: if the SL or TP was between the open and close, we assumed it was crossed during the day and executed the sell at that level. This let us realistically apply and test stop-loss and take-profit strategies, even with only daily data.
  - **Limitation:** If both the stop-loss (SL) and take-profit (TP) are hit on the same day, we always sell at the TP. In reality, the SL could be hit before the TP, but with only daily data, we can't know which came first. We acknowledge this limitation in our backtest results.
  - **Realistic Results:** After enforcing one trade per day and robust SL/TP logic, the equity curve became much smoother and more realistic, and the Sharpe ratio dropped to a plausible level (around 2). This demonstrates the importance of matching trading logic to data frequency and realistic execution constraints.

## Recent Debugging Journey: Trading Logic, Issues, and Fixes

### What We Set Out to Do
- Implement a robust swing trading strategy with clear stop-loss (SL) and take-profit (TP) rules.
- Ensure all exits (SL/TP) are handled correctly, including gap-up/gap-down scenarios and intraday crosses.
- Avoid data leakage and ensure honest, reproducible backtesting.

---

## Data Split & Feature Engineering Debugging Journey

### Un-comparable Results & Chronological Split
At one stage, we switched to a chronological split (oldest 70% for train, next 15% for validation, newest 15% for test) to avoid lookahead bias. This led to very different trades and a sharp drop in the equity curve, as the model struggled with market regime changes between train and test periods.

### Fixed Date Splits
We then tried fixed date boundaries for train, validation, and test sets to further control for data leakage. While this made splits consistent, the equity curve remained poor due to out-of-sample regime shifts and feature changes.

### Feature Set Restoration
To recover performance, we reverted to our classic feature set—restoring missing moving averages, Bollinger Band squeeze features, and Supertrend × time/support interactions, while removing experimental features. This brought the equity curve and results back in line with previous successful runs.

### Lessons Learned
- Consistent data splits and feature sets are critical for reproducible backtesting.
- Market regime changes can dramatically affect out-of-sample performance; percentage-based splits may be preferable for strategy development.
- Version control and careful documentation of feature engineering changes help prevent future comparability issues.

### Issues We Faced
- **SL/TP Not Triggering:** The original logic only triggered a sell if SL/TP was strictly between open and close. This missed cases where the open or close gapped through SL/TP.
- **Partial Exits & Trailing Stops:** Early versions included partial exits and trailing stops, which complicated state management and sometimes left positions open when they should have been closed.
- **Uninitialized Variables:** Variables like `risk_per_share` could be `None`, causing runtime errors when used in arithmetic.
- **State Not Reset:** After a sell, trade state variables were not always reset, leading to incorrect position tracking.

### How We Thought Through the Problems
- Carefully traced the logic for every possible price path (gap up, gap down, intraday cross, no cross).
- Used concrete trade log examples to identify where the logic failed.
- Discussed and clarified the intended trading rules: always sell at open if gapped through SL/TP, otherwise sell at SL/TP if crossed intraday.

### The Fixes We Implemented
- **SL/TP Logic:**  
  - Sell at open if `open_price >= tp` (for TP) or `open_price <= sl` (for SL).
  - Sell at TP/SL if crossed intraday: `min(open_price, close_price) <= tp/sl <= max(open_price, close_price)`.
- **Removed Trailing/Partial Logic:**  
  - Simplified to full exits only, for clarity and reliability.
- **State Management:**  
  - After any sell, reset all trade state variables (`shares`, `position`, `last_buy_price`, `sl`, `tp`, etc.).
- **Guard Clauses:**  
  - Checked for `None` before using variables in arithmetic to prevent runtime errors.
- **Trade Log Verification:**  
  - Used the trade log to confirm that every exit was recorded and shares were reset to zero after a sell.

### What We Learned
- Even simple trading rules can be tricky to implement correctly due to edge cases (gaps, intraday moves).
- Concrete trade logs are essential for debugging trading systems.
- Honest documentation of issues and fixes is as important as the code itself.

---


## Example Metrics (Subject to Change)
- Example Sharpe ratios and drawdowns are reported in the documentation, but these are subject to change as the model, features, and validation methods evolve.

## Known Limitations & Next Steps
- Some features described in the documentation are planned but not yet implemented.
- Validation strategy and feature set are evolving; see DOCUMENTATION.md for details.
- **Data Source Variability:** If you always download fresh data from Polygon.io, your results may change between runs—even for the same date range—if Polygon updates or corrects their historical data. This can affect all downstream results, including features, splits, trades, and metrics. For reproducibility, cache your data locally and use the same file for all backtests.
- **Debugging Tools:** The project includes automatic data hash tracking and smart snapshot saving to help identify when data source changes are causing non-reproducible results.
- Suggestions and contributions are welcome!

## Requirements
- Python environment
- Polygon.io API key

## Project Structure
```
tesla_stock_predictor/
├── data/              # Data handling
├── features/          # Feature engineering
├── models/            # Model implementations
├── utils/             # Utility functions
└── main.py            # Main execution script
```

## Setup
1. Set your Polygon.io API key:
```bash
export POLYGON_API_KEY='your_api_key_here'
```

2. Run the predictor:
```bash
python -m tesla_stock_predictor.main
```

## Development Process
- Started: April 2025
- Method: 100% AI pair programming (ChatGPT, Claude, GitHub Copilot)
- Evolution: 
  - Initially included LSTM and MLP (removed)
  - XGBoost tested and removed
  - Final architecture: RF, DT, LightGBM ensemble

---

## Debugging & Problem-Solving Log

See [STORY.md](STORY.md) for a narrative of the development journey, and [DOCUMENTATION.md](DOCUMENTATION.md) for technical details and a changelog of major fixes and design decisions.

### Recent Major Fixes

- **Non-comparable results and historical trade changes:**  
  All sources of non-comparable results were identified and fixed. This included ensuring that feature engineering, model training, and trade logging only process new/unseen data and never recalculate historical trades unless the underlying data changes.  
  - The ensemble prediction and trade log are now strictly causal and append-only for new days.
  - Data fetching from Polygon.io was not the root cause; the main issues were internal pipeline logic and cache handling.
  - Data hash tracking and snapshotting now provide clear evidence if the underlying data changes, so you can distinguish between data and code issues.
- **Trade log improvements:**  
  The unified backtest now saves all trades (including "HOLD" days) to `debug/trade_log.csv`, while only the last 10 trades are printed in the terminal for clarity.
- **Reproducibility:**  
  If you use cached data, your results are fully reproducible. Historical trades only change if the underlying data itself changes.

### Retraining Instructions

- **To retrain the grid search for ensemble thresholds:**  
  Empty the relevant `ensemble_config_*.json` file(s) in the `models/` directory. This will force the pipeline to rerun the grid search and generate new ensemble configurations.

- **To fully retrain the models and regenerate all features:**  
  Delete the feature cache (`features/features_cache.csv`), model cache files in `models/daily_models/`, and any other relevant cache files. This will force the pipeline to recompute all features and retrain all models from scratch.

## Technical Documentation
- See [DOCUMENTATION.md](DOCUMENTATION.md) for technical details
- See [STORY.md](STORY.md) for development journey

## License

This project is licensed under a custom **Educational Non-Commercial License** (see [LICENSE](LICENSE)).  
- **You may use, modify, and contribute to this project for educational and non-commercial purposes only.**
- **Commercial use, selling, sublicensing, or rebranding is strictly prohibited without explicit permission.**
- **Attribution to the original author (Dhruv Mehta) is required.**

## Disclaimer

This project is provided for educational purposes only.  
**No warranty is given. The author is not liable for any damages, including financial losses incurred from trading or investment decisions made using this project. Use at your own risk.**

## Acknowledgments
- ChatGPT, Claude AI, and GitHub Copilot for development
- Polygon.io for market data

This README reflects only the implemented and verified features of the project. All results and claims are subject to change as the project evolves.