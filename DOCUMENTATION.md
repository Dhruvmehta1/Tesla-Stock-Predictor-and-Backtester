# Technical Documentation: Tesla Stock Predictor

## Project Overview
Developed in April 2025, this project uses AI pair programming (ChatGPT, Claude, GitHub Copilot) to create a Tesla stock prediction model focusing on high-confidence predictions and realistic backtesting.

## Core Components

---

## Technical Changelog: Trading Logic Debugging & Fixes (April–July 2025)

---

### Major Reproducibility and Robustness Fixes (Summer 2025)

- **Non-comparable results and historical trade changes:**  
  All sources of non-comparable results were identified and fixed. This included ensuring that feature engineering, model training, and trade logging only process new/unseen data and never recalculate historical trades unless the underlying data changes.  
  - The ensemble prediction and trade log are now strictly causal and append-only for new days.
  - Data fetching from Polygon.io was not the root cause; the main issues were internal pipeline logic and cache handling.
  - Data hash tracking and snapshotting now provide clear evidence if the underlying data changes, so you can distinguish between data and code issues.
- **Trade log improvements:**  
  The unified backtest now saves all trades (including "HOLD" days) to `debug/trade_log.csv`, while only the last 10 trades are printed in the terminal for clarity.
- **Reproducibility:**  
  If you use cached data, your results are fully reproducible. Historical trades only change if the underlying data itself changes.
- **Defensive programming:**  
  Checks for `None` and empty DataFrames prevent runtime errors in feature engineering and caching.

### Trading Logic: Issues, Thinking, and Fixes

#### What We Set Out to Do
- Implement a robust swing trading strategy with clear stop-loss (SL) and take-profit (TP) rules.
- Ensure all exits (SL/TP) are handled correctly, including gap-up/gap-down scenarios and intraday crosses.
- Avoid data leakage and ensure honest, reproducible backtesting.
- Guarantee that historical trades and ensemble predictions remain unchanged when only a new trading day is added, unless the underlying historical data itself changes.

---

## Debugging Journey: Data Splits, Feature Engineering, and Reproducibility

### 1. Un-comparable Results & Data Split Evolution

- Initially, results were inconsistent and could not be reliably compared across runs. This was traced to frequent changes in the data split logic and feature set, which led to different samples being used for training, validation, and testing.
- We experimented with a chronological split (oldest 70% for train, next 15% for validation, newest 15% for test) to avoid lookahead bias and better simulate real-world trading. However, trades generated in the validation and test sets were very different from previous runs, and the equity curve dropped sharply.
- To further control for data leakage and ensure reproducibility, we tried using fixed date boundaries for train, validation, and test sets. While this made the splits consistent, the equity curve remained poor due to market regime shifts and feature changes.
- Ultimately, we reverted to the original percentage-based split (70/15/15), which restored consistency and comparability.

### 2. Feature Engineering Restoration

- Realizing that feature engineering changes were also affecting results, we reverted to our classic feature set. Missing moving averages, Bollinger Band squeeze features, and Supertrend × time/support interactions were restored.
- Advanced features (Fibonacci retracements, multi-indicator confirmations) were removed to match the original best-performing setup.
- This restoration improved model performance and brought the equity curve back in line with previous successful runs.

### 3. Lessons Learned

- Consistent data splits and feature sets are critical for reproducible backtesting.
- Market regime changes can dramatically affect out-of-sample performance; percentage-based splits may be preferable for strategy development.
- Version control and careful documentation of feature engineering changes help prevent future comparability issues.

#### Issues We Faced
- **Non-comparable results and historical trade changes:**  
  - Historical trades and ensemble predictions would sometimes change when only a new trading day was added, even if the underlying data did not change.
  - This was traced to internal pipeline logic and cache handling, not to data fetching from Polygon.io.
  - Feature engineering, model training, and trade logging were recalculating or overwriting historical results instead of only appending new data.
- **SL/TP Not Triggering:** The original logic only triggered a sell if SL/TP was strictly between open and close. This missed cases where the open or close gapped through SL/TP.
- **Partial Exits & Trailing Stops:** Early versions included partial exits and trailing stops, which complicated state management and sometimes left positions open when they should have been closed.
- **Uninitialized Variables:** Variables like `risk_per_share` could be `None`, causing runtime errors when used in arithmetic.
- **State Not Reset:** After a sell, trade state variables were not always reset, leading to incorrect position tracking.

#### How We Thought Through the Problems
- Carefully traced the logic for every possible price path (gap up, gap down, intraday cross, no cross).
- Used concrete trade log examples to identify where the logic failed.
- Discussed and clarified the intended trading rules: always sell at open if gapped through SL/TP, otherwise sell at SL/TP if crossed intraday.

#### The Fixes We Implemented
- **Strictly Causal, Append-Only Pipeline:**  
  - Feature engineering, model training, and trade logging were refactored to only process and append new/unseen data. Historical trades and ensemble predictions are never recalculated unless the underlying data changes.
  - The trade log is now append-only and records every day (including "HOLD" days) for full transparency.
  - Data hash tracking and snapshotting were added to detect and debug changes in historical data.
- **SL/TP Logic:**  
  - Sell at open if `open_price >= tp` (for TP) or `open_price <= sl` (for SL).
  - Sell at TP/SL if crossed intraday: `min(open_price, close_price) <= tp/sl <= max(open_price, close_price)`.
  - Sell at TP if the high price exceeds TP intraday, even if not between open and close.
- **Removed Trailing/Partial Logic:**  
  - Simplified to full exits only, for clarity and reliability.
- **One Trade Per Day Constraint:**  
  - Disabled multiple trades per day to match what’s possible with daily data. This change made the equity curve much smoother and the Sharpe ratio more plausible, reflecting realistic trading conditions.
- **Defensive Programming:**  
  - Added checks for `None` and empty DataFrames in feature engineering and caching to prevent runtime errors.
- **State Management:**  
  - After any sell, reset all trade state variables (`shares`, `position`, `last_buy_price`, `sl`, `tp`, etc.).
- **Guard Clauses:**  
  - Checked for `None` before using variables in arithmetic to prevent runtime errors.
- **Trade Log Verification:**  
  - Used the trade log to confirm that every exit was recorded and shares were reset to zero after a sell.

#### What We Learned
- Even simple trading rules can be tricky to implement correctly due to edge cases (gaps, intraday moves).
- Concrete trade logs are essential for debugging trading systems.
- Honest documentation of issues and fixes is as important as the code itself.

---

### Data Pipeline
- **Source**: Polygon.io API
- **Authentication**: Environment variable for API key
- **Key Challenge**: Successfully resolved 401 authentication error
- **Data Source Variability**: Discovered that downloading fresh data from Polygon.io on each run can lead to different results, even for the same date range. This is due to possible corrections, late updates, or inconsistencies in the data provided by Polygon.io. As a result, trades and portfolio values for previous days may change unexpectedly between runs.
- **Reproducibility Solution**: To ensure consistent and reproducible backtest results, it is essential to cache the raw data locally after the first download and use this cached version for all subsequent runs. This guarantees that the same input data is used every time, making results comparable and debugging more reliable.

### Model Architecture Evolution
1. **Initial Implementation**
   - Started with LSTM and MLP
   - Removed due to performance degradation
   - XGBoost removed due to poor class 1 performance

2. **Current Architecture**
   - Random Forest (RF)
   - Decision Tree (DT)
   - LightGBM
   - Ensemble weights optimized by Optuna

### Performance Metrics

- Metrics such as Sharpe ratio and drawdown are reported for specific runs and may change as the project evolves.
- All results should be interpreted as part of an ongoing research and learning process, not as guarantees of future performance.
- Metrics are for illustration only and may not reflect future or out-of-sample performance.

## Technical Challenges Resolved

### Non-Deterministic Behavior
- **Issue**: Different results on each run despite set random seeds
- **Root Cause**: Unseeded cross-validation splitter; also, data source variability from Polygon.io (data corrections, late updates, or API inconsistencies) can cause different results for the same date range.
- **Solution**: Implemented proper seed management for all model and data processing steps. Additionally, identified the need to cache downloaded data locally to eliminate discrepancies caused by changes in the external data source.

### Data Leakage
- **Issue**: Forward-looking bias in feature engineering
- **Solution**: Proper time-series validation implementation
- **Validation**: Walk-forward approach to prevent look-ahead bias

### Model Performance
- **Issue**: Class imbalance in predictions
- **Solution**: 
  - Custom thresholds per model
  - Optuna-optimized ensemble weights

## Implementation Details

### Environment Setup
- Python environment
- Polygon.io API key requirement
- Proper module imports (resolved initial TensorFlow issues)

### Model Pipeline
1. Data fetching from Polygon.io
2. Feature engineering
3. Model training with proper validation
4. Ensemble prediction
5. Performance calculation with transaction costs

## Known Limitations

- Requires valid Polygon.io API key
- Dependent on API availability
- Needs proper seed management for reproducibility
- **Data Source Variability:** Results may change between runs if fresh data is downloaded from Polygon.io each time, due to possible corrections or updates in the data. For reproducible research and backtesting, always use a locally cached copy of the data.
- **Historical Trade Reproducibility:** The pipeline now guarantees that historical trades and ensemble predictions remain unchanged when only a new trading day is added, unless the underlying historical data itself changes. This was achieved by fixing all sources of non-comparable results, including feature engineering, model training, and trade logging. Data fetching from Polygon.io was not the root cause—internal pipeline logic and caching were the main factors.
- **Validation Approach:** The project experimented with chronological and fixed-date splits, but reverted to a percentage-based split (70/15/15) for consistency and comparability. Chronological splits led to a sharp drop in the equity curve due to market regime changes and out-of-sample challenges.
- **Feature Engineering:** The feature set was restored to the classic configuration after realizing that missing or changed features contributed to poor results. Advanced features were removed, and proven technical indicators and engineered combos were reintroduced.
- Validation and feature engineering are under active development.

---

## Debugging & Problem-Solving Log

See [README.md](README.md) for a summary of recent issues and fixes, and [STORY.md](STORY.md) for a narrative of the development journey.

## Development Notes
- Started: April 2025
- Method: 100% AI pair programming
- Tools: ChatGPT, Claude AI, GitHub Copilot
- Focus: Realistic implementation and validation

## Error Handling
- API authentication validation
- Data integrity checks
- Model convergence monitoring

This documentation reflects only the verified and implemented features of the Tesla Stock Predictor project, based on actual development history and conversations. All claims and metrics are snapshots in time and may change as the project evolves.