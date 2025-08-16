# My Journey Building the Tesla Stock Predictor

## Introduction

This project chronicles my journey in building a machine learning-based stock prediction system for Tesla. While I leveraged AI tools (ChatGPT, Claude, GitHub Copilot) for code implementation, my role focused on architecture decisions, problem-solving, and ensuring practical value.

---

## Fixing Non-Comparable Results and Ensuring Reproducibility

### The Problem

As the project matured, I discovered that historical trades and ensemble predictions would sometimes change when only a new trading day was added, even if the underlying data did not change. This made results non-comparable and undermined confidence in the backtest. Initially, I suspected the data source (Polygon.io), but after thorough investigation, I realized the root cause was internal: feature engineering, model training, and trade logging were recalculating or overwriting historical results instead of only appending new data.

### The Solution

- **Strictly Causal, Append-Only Pipeline:**  
  I refactored the pipeline so that feature engineering, model training, and trade logging only process and append new/unseen data. Historical trades and ensemble predictions are never recalculated unless the underlying data itself changes.
- **Trade Log Improvements:**  
  The trade log is now append-only and records every day (including "HOLD" days) for full transparency. Only the last 10 trades are printed in the terminal, but the CSV contains the full trade history.
- **Data Hash Tracking and Snapshotting:**  
  I implemented data hash tracking and smart snapshotting to detect and debug changes in historical data. This provides clear evidence if the underlying data changes, so I can distinguish between data and code issues.
- **Defensive Programming:**  
  Added checks for `None` and empty DataFrames in feature engineering and caching to prevent runtime errors.
- **Reproducibility:**  
  If cached data is used, results are fully reproducible. Historical trades only change if the underlying data itself changes.

---


## Debugging the Trading Logic: A Narrative

### The Challenge

As the project matured, I shifted focus from pure model performance to the robustness and realism of the trading logic. The goal was to ensure that the backtest reflected how trades would actually execute in live markets, especially around stop-loss (SL) and take-profit (TP) events.

### Issues We Faced

- **Non-comparable results and historical trade changes:**  
  Historical trades and ensemble predictions would sometimes change when only a new trading day was added, even if the underlying data did not change. This was traced to internal pipeline logic and cache handling, not to data fetching from Polygon.io. Feature engineering, model training, and trade logging were recalculating or overwriting historical results instead of only appending new data.
- **SL/TP Not Triggering:**  
  The original logic only triggered a sell if SL/TP was strictly between open and close. This missed cases where the open or close gapped through SL/TP, leaving positions open when they should have been closed.
- **Partial Exits & Trailing Stops:**  
  Early versions included partial exits and trailing stops, which complicated state management and sometimes left positions open incorrectly.
- **Uninitialized Variables:**  
  Variables like `risk_per_share` could be `None`, causing runtime errors when used in arithmetic.
- **State Not Reset:**  
  After a sell, trade state variables were not always reset, leading to incorrect position tracking.

### How We Thought Through the Problems

- Carefully traced the logic for every possible price path (gap up, gap down, intraday cross, no cross).
- Used concrete trade log examples to identify where the logic failed.
- Discussed and clarified the intended trading rules: always sell at open if gapped through SL/TP, otherwise sell at SL/TP if crossed intraday.
- Investigated the causes of non-comparable results and discovered that the main issues were internal pipeline logic and cache handling, not the data source.

### The Fixes We Implemented

- **Strictly Causal, Append-Only Pipeline:**  
  Feature engineering, model training, and trade logging were refactored to only process and append new/unseen data. Historical trades and ensemble predictions are never recalculated unless the underlying data changes.
- **SL/TP Logic:**  
  - If the market opens past your target or stop, we sell right at the open.
  - If the price crosses your target or stop during the day, we sell at that exact level.
  - If the high price exceeds TP intraday (even if not between open and close), we exit at TP.
  - This matches how real stop-loss and take-profit orders work.
- **Removed Trailing/Partial Logic:**  
  - Simplified to full exits only, for clarity and reliability.
- **State Management:**  
  - After any sell, reset all trade state variables (`shares`, `position`, `last_buy_price`, `sl`, `tp`, etc.).
- **Guard Clauses:**  
  - Checked for `None` before using variables in arithmetic to prevent runtime errors.
- **Trade Log Verification:**  
  - Used the trade log to confirm that every exit was recorded and shares were reset to zero after a sell.
  - The trade log is now append-only and records every day (including "HOLD" days) for full transparency. Only the last 10 trades are printed in the terminal, but the CSV contains the full trade history.
- **Data Hash Tracking and Snapshotting:**  
  - Implemented data hash tracking and smart snapshotting to detect and debug changes in historical data. This provides clear evidence if the underlying data changes, so I can distinguish between data and code issues.
- **Reproducibility:**  
  - If cached data is used, results are fully reproducible. Historical trades only change if the underlying data itself changes.

### What We Learned

- Even simple trading rules can be tricky to implement correctly due to edge cases (gaps, intraday moves).
- Concrete trade logs are essential for debugging trading systems.
- Honest documentation of issues and fixes is as important as the code itself.
- For true reproducibility, all sources of non-comparable results must be addressed internally; the data source is rarely the root cause.

---

## Reflections on Realism, Results, Data Splitting, and Feature Restoration

### Fixing Un-Comparable Results: Data Splitting and Feature Engineering Journey

A major challenge in this project was the lack of comparability between backtest results across different runs. This was traced to frequent changes in the data split logic and feature set, which led to different samples being used for training, validation, and testing. To address this, I experimented with several approaches:

- **Chronological Split:**  
  I switched to a chronological split (oldest 70% for train, next 15% for validation, newest 15% for test) to avoid lookahead bias and better simulate real-world trading. However, this led to trades in the validation and test sets that were very different from previous runs, and the equity curve dropped sharply. The model struggled to adapt to market regime changes, as the validation/test periods were out-of-sample and potentially very different from the training period.

- **Fixed Date Splits:**  
  To further control for data leakage and ensure reproducibility, I tried using fixed date boundaries for train, validation, and test sets. The splits were set to match the exact dates from a previous successful run. While this made the splits consistent, the equity curve remained poor, likely due to the market regime shift between train and test periods, and possibly due to missing or changed features.

- **Feature Set Restoration:**  
  Realizing that feature engineering changes were also affecting results, I reverted to my classic feature set, restoring missing moving averages, Bollinger Band squeeze features, and Supertrend × time/support interactions. Advanced features (Fibonacci retracements, multi-indicator confirmations) were removed to match the original best-performing setup. This restoration helped the model's performance and equity curve recover, making results more comparable to earlier runs.

**Lesson:**  
Consistent data splits and feature sets are critical for reproducible backtesting. Market regime changes can dramatically affect out-of-sample performance, and percentage-based splits may be preferable for strategy development. Careful documentation and version control of feature engineering changes help prevent future comparability issues.

Implementing robust SL/TP logic and limiting to one trade per day transformed the equity curve from erratic, unrealistic swings to a smooth, upward trend. This was a key lesson: realistic trading constraints not only make the backtest more honest, but also produce believable metrics (like a Sharpe ratio around 2 instead of 4+). The journey from “too good to be true” to “robust and plausible” was one of the most valuable outcomes of this project.

### Data Source Discrepancy: The Hidden Variable

A major discovery late in the project was that **non-reproducible results were caused by changes in the data source itself**. Despite deterministic code, fixed seeds, and robust feature engineering, I observed that backtest results (including trade logs and validation metrics) would change between runs, even with no code changes. After a thorough investigation, I traced this to the fact that the pipeline always downloaded fresh data from Polygon.io, and Polygon's historical data can change over time due to corrections, late updates, or API inconsistencies.

This meant that even if only a new trading day was added, the trades and portfolio values for previous days could also change if Polygon updated any historical prices or volumes. The lesson: **for true reproducibility in financial research, you must cache your raw data locally and use the same snapshot for all backtests**. Only then can you guarantee that results are comparable across runs and over time.

### Debugging Measures: Proving the Data Source Theory

To definitively confirm that data source variability was the root cause, I implemented comprehensive debugging measures in the pipeline:

- **Data Hash Tracking:** Added functionality to compute and compare MD5 hashes of the raw data downloaded from Polygon.io on each run.
- **Smart Data Saving:** Implemented logic to save raw data snapshots only when running on a new day OR when the data actually changes, preventing unnecessary file clutter.
- **Hash Log Maintenance:** Created a persistent log (`data_hash_log.txt`) to track data hashes over time, making it easy to spot when Polygon updates their historical data.
- **Visual Inspection Tools:** Automatically save the first and last few rows of each dataset for manual comparison between runs.

These measures provide concrete evidence of whether data source changes are causing non-reproducible results, eliminating guesswork and enabling focused debugging. The implementation proves that even with perfectly deterministic code and fixed random seeds, external data source variability can be the hidden culprit behind inconsistent results.

This experience reinforced the importance of not just code and model determinism, but also data versioning and integrity in any serious quantitative research.

---

## My Key Contributions

### Strategic Decision Making

- **Model Selection:** I critically evaluated each model's performance and made decisive calls:
  - Removed LSTM and MLP when they degraded ensemble performance
  - Cut XGBoost despite its popularity due to poor class 1 prediction
  - Focused on a Random Forest–Decision Tree hybrid strategy that showed better real-world results
  - Added LightGBM to enhance ensemble capabilities

### Problem Solving & Quality Control

- **Data Quality:**
  - Identified and addressed data leakage issues
  - Switched from yfinance to Polygon.io for more reliable data
  - Insisted on proper time-series validation

- **Performance Optimization:**
  - Spotted the critical issue of two persistent false positives
  - Focused on high-confidence predictions for practical usage

- **Code Quality:**
  - Required deterministic outputs for reproducibility
  - Ensured proper seed management across all models
  - Maintained modular code structure

### Project Management

- **Development Approach:**
  - Pioneered a 100% AI-assisted development methodology
  - Coordinated between different AI tools for optimal results
  - Maintained project coherence across multiple iterations

- **Quality Standards:**
  - Insisted on realistic backtesting with transaction costs
  - Required clear performance metrics (e.g., Sharpe ratio, drawdown)
  - Demanded honest reporting of capabilities and limitations

### Innovation in Development

- **AI Tool Integration:**
  - Effectively used ChatGPT for code generation and debugging
  - Leveraged Claude for architecture decisions
  - Utilized GitHub Copilot for implementation details
  - Successfully coordinated between tools to maintain consistency

## Key Decisions That Shaped the Project

### Model Architecture

1. **Initial Phase** (April 2025)
   - Started ambitious with LSTM, MLP, RF, XGBoost
   - Quickly identified performance issues
   - Made tough but necessary cuts

2. **Refinement Phase**
   - Focused on RF-DT hybrid strategy
   - Added LightGBM strategically
   - Implemented custom ensemble weighting

3. **Optimization Phase**
   - Developed improved ensemble logic
   - Integrated Optuna for weight optimization
   - Added transaction cost considerations

### Technical Improvements

- **Data Pipeline:**
  - Chose Polygon.io for reliable data
  - Implemented proper API authentication
  - Ensured clean data handling

- **Feature Engineering:**
  - Focused on proven technical indicators
  - Prevented look-ahead bias
  - Maintained feature quality

- **Validation Strategy:**
  - Insisted on walk-forward validation
  - Included transaction costs
  - Required robust testing

## Learning Outcomes

### Technical Growth

- Deepened understanding of machine learning in finance
- Mastered ensemble model optimization
- Learned effective AI tool utilization

### Project Management

- Developed skills in directing AI-assisted development
- Learned to balance ambition with practicality
- Gained experience in quality control

## Reflections

This project is as much about the process as the outcome.  
I have tried, failed, learned, and iterated—sometimes achieving strong results, sometimes not.  
All claims and metrics are snapshots in time, not promises.  
The journey continues, and I invite others to join, critique, and contribute.

## Open Challenges & Next Steps

- Improving validation strategy for non-stationary data.
- Exploring additional ensemble logic and mechanisms.
- Suggestions and feedback are welcome!

## Conclusion

While this project used AI tools for code implementation, my contribution lay in the strategic decisions, problem identification, solution validation, and maintaining high standards for practical application. The project demonstrates how human insight and decision-making can effectively guide AI-assisted development to create robust, practical solutions.

A key challenge that delayed the project’s submission from July 17th to the final publication date was the lack of robust validation-based tuning for ensemble weights and thresholds. This issue highlighted the importance of thorough validation and honest reporting in financial machine learning projects.

Started: April 2025  
Solo project (individual effort)  
Tools: ChatGPT, Claude AI, GitHub Copilot

## Impact and Reflection

This project is a testament to how modern AI tools can empower a solo developer to build sophisticated systems without writing code manually. My role was not passive—I was the architect, tester, and quality controller. I set the direction, defined the standards, and made the critical decisions that shaped the project’s success.

### What I’m Proud Of

- **Vision and Direction:** I kept the project focused on realistic, actionable results, not just theoretical metrics.
- **Persistence:** I didn’t settle for “good enough”—I pushed for robust validation, reproducibility, and honest reporting.
- **Innovation:** I pioneered a workflow where AI tools handled the code, but I drove the process, ensuring the final product was practical and reliable.
- **Learning:** I turned every challenge—be it data leakage, non-determinism, or model selection—into an opportunity to deepen my understanding of both finance and machine learning.

## Final Thoughts

This journey proved that with the right questions, critical thinking, and a relentless focus on quality, you can leverage AI to build something truly valuable—even if you never write a single line of code yourself. The Tesla Stock Predictor stands as a unique blend of human insight and AI capability, and I’m excited to see where this approach can lead in future projects.