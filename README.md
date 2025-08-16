tesla_stock_predictor/README.md
# Tesla Stock Predictor

A machine learning project for predicting TSLA stock movements using an ensemble of Random Forest, Decision Tree, and LightGBM models. Designed for robust, realistic backtesting and reproducibility.

## Features
- Ensemble of Random Forest, Decision Tree, and LightGBM models
- Optuna-optimized ensemble weights
- Transaction costs included in backtesting
- Extensive feature engineering (technical indicators, lags, volatility, etc.)
- No data leakage (all features use only past/current data)
- Realistic stop-loss/take-profit logic with daily data
- Full trade logging to `debug/trade_log.csv`
- Defensive programming for error prevention
- Data hash tracking and snapshotting for reproducibility

## Requirements
- Python 3.x
- Polygon.io API key

## Installation & Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Polygon.io API key:
   ```bash
   export POLYGON_API_KEY='your_api_key_here'
   ```

## Usage

Run the predictor:
```bash
python -m tesla_stock_predictor.main
```

## Project Structure
```
tesla_stock_predictor/
├── data/              # Data handling
├── features/          # Feature engineering
├── models/            # Model implementations
├── utils/             # Utility functions
└── main.py            # Main execution script
```

## Documentation

- See [DOCUMENTATION.md](DOCUMENTATION.md) for technical details and retraining instructions.
- See [STORY.md](STORY.md) for the development journey and debugging narrative.

## License

This project is licensed under a custom **Educational Non-Commercial License** (see [LICENSE](LICENSE)).
Attribution to the original author (Dhruv Mehta) is required.

## Disclaimer

This project is provided for educational purposes only. No warranty is given. Use at your own risk.

## Acknowledgments

- GitHub Copilot for development assistance
- Polygon.io for market data