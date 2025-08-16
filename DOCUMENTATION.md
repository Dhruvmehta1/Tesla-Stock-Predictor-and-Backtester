tesla_stock_predictor/DOCUMENTATION.md
# Technical Documentation: Tesla Stock Predictor

## 1. System Overview

Tesla Stock Predictor is a machine learning pipeline for predicting TSLA stock movements using an ensemble of Random Forest, Decision Tree, and LightGBM models. The system emphasizes robust backtesting, realistic trading logic, and reproducibility.

---

## 2. Directory & Module Structure

- **data/**: Data fetching and preprocessing scripts.
- **features/**: Feature engineering modules and cache.
- **models/**: Model definitions, training, ensemble logic, and model caches.
- **utils/**: Utility functions (logging, config, etc.).
- **debug/**: Trade logs and debugging outputs.
- **main.py**: Main entry point for running the pipeline.

---

## 3. Data Pipeline

### Data Source
- **Provider:** Polygon.io (daily TSLA OHLCV data).
- **Authentication:** Requires `POLYGON_API_KEY` as an environment variable.
- **Format:** Data is fetched as pandas DataFrames and stored locally for caching.

### Data Preprocessing
- Handles missing values and ensures chronological order.
- Only uses data available up to the current day to prevent lookahead bias.

---

## 4. Feature Engineering

All features are engineered to avoid data leakage by using only information available up to the current day. Features include:

- **Moving Averages:** Simple and exponential (5, 10, 20, 50 days).
- **Bollinger Bands:** Upper/lower bands, squeeze indicators.
- **Supertrend:** Trend-following indicator, combined with time/support levels.
- **Lag Features:** Previous day’s close, volume, returns.
- **Volatility:** Rolling standard deviation of returns.
- **Custom Interactions:** e.g., Supertrend × time/support.

Feature engineering is implemented in `features/` and cached in `features/features_cache.csv`. All rolling calculations use `shift()` to exclude the current day’s value.

---

## 5. Model Architecture

- **Models Used:** Random Forest, Decision Tree, LightGBM (all from scikit-learn or lightgbm).
- **Ensemble:** Model outputs are combined using Optuna-optimized weights.
- **Training/Validation/Test Split:** Default is 70% train, 15% validation, 15% test (chronological split to avoid lookahead bias).
- **Class Imbalance:** Custom thresholds and ensemble weights are used to address imbalance.

---

## 6. Backtesting & Trading Logic

- **Backtesting:** Simulates trades using only daily data.
- **Stop-Loss/Take-Profit:**  
  - If SL/TP is hit at open, trade is exited at open.
  - If crossed intraday, trade is exited at the SL/TP level.
  - If both are hit, TP takes precedence (due to daily data limitation).
- **Transaction Costs:** Included in all backtests.
- **Trade Logging:** All trades (including HOLD days) are logged to `debug/trade_log.csv`.

---

## 7. Reproducibility & Caching

- **Data Hashing:** Hashes of raw data are stored to detect changes.
- **Snapshotting:** Snapshots of key pipeline outputs are saved for reproducibility.
- **Cache Files:**  
  - `features/features_cache.csv`: Feature cache.
  - `models/daily_models/`: Model caches.
  - `models/ensemble_config_*.json`: Ensemble configuration.

To ensure reproducibility, always use cached data and models. If you want to retrain or regenerate features, delete the relevant cache files.

---

## 8. Usage & Advanced Operations

### Running the Pipeline
```bash
python -m tesla_stock_predictor.main
```

### Retraining Models & Features
- To retrain ensemble thresholds: Delete `models/ensemble_config_*.json`.
- To fully retrain and regenerate features: Delete `features/features_cache.csv` and model caches in `models/daily_models/`.

### Adding New Features or Models
- Add new feature scripts to `features/` and update the feature engineering pipeline.
- Add new model classes to `models/` and update the ensemble logic.

---

## 9. Troubleshooting & FAQ

- **API Errors:** Ensure `POLYGON_API_KEY` is set and valid.
- **Data Issues:** Check for missing or corrupted data files in `data/`.
- **Model Errors:** Ensure all dependencies are installed and cache files are not corrupted.
- **Reproducibility:** Use only cached data/models for consistent results.

---

## 10. Known Limitations

- **Data Source Variability:** Results may change if fresh data is downloaded from Polygon.io due to corrections/updates.
- **Granularity:** Only daily data is used; intraday price movement is simulated.
- **External Dependencies:** Requires Polygon.io API and internet access for data fetching.

---

## 11. Changelog

- See git history or release notes for major technical changes.

---

## 12. References & Links

- [README.md](README.md): Quickstart and project overview
- [STORY.md](STORY.md): Development journey and debugging narrative
- [Polygon.io](https://polygon.io/): Data provider
- [scikit-learn](https://scikit-learn.org/), [lightgbm](https://lightgbm.readthedocs.io/): ML libraries
