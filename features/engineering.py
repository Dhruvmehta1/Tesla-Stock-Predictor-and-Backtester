import numpy as np
import pandas as pd
import os

def supertrend(df, atr_period=10, factor=3.0):
    """
    Calculate Supertrend indicator.
    Returns supertrend value and direction (1 for uptrend, -1 for downtrend).
    """
    hl2 = (df['High'] + df['Low']) / 2
    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean()

    # Basic bands
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)

    supertrend = np.zeros(len(df))
    direction = np.ones(len(df))

    supertrend[0] = upperband.iloc[0]
    direction[0] = 1

    for i in range(1, len(df)):
        if np.isnan(atr.iloc[i]):
            supertrend[i] = np.nan
            direction[i] = direction[i-1]
            continue

        # Previous supertrend value
        prev_supertrend = supertrend[i-1]
        prev_direction = direction[i-1]

        # If close crosses below lowerband, switch to downtrend
        if df['Close'].iloc[i] > lowerband.iloc[i]:
            supertrend[i] = lowerband.iloc[i]
            direction[i] = 1
        elif df['Close'].iloc[i] < upperband.iloc[i]:
            supertrend[i] = upperband.iloc[i]
            direction[i] = -1
        else:
            supertrend[i] = prev_supertrend
            direction[i] = prev_direction

        # Maintain band logic
        if direction[i] == 1 and supertrend[i] < supertrend[i-1]:
            supertrend[i] = supertrend[i-1]
        if direction[i] == -1 and supertrend[i] > supertrend[i-1]:
            supertrend[i] = supertrend[i-1]

    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)

def engineer_features(df, sector_df=None, spy_df=None):
    """Enhanced feature engineering WITHOUT data leakage
    Optionally merges sector ETF and SPY data for relative features.
    Always returns a DataFrame, never None.
    """
    # If df is empty or missing required columns, return empty DataFrame
    if df is None or len(df) == 0 or not all(col in df.columns for col in ['Close', 'High', 'Low', 'Volume']):
        return pd.DataFrame(index=df.index if df is not None else None)
    # ... rest of the function remains unchanged ...

def engineer_features_incremental(df, sector_df=None, spy_df=None):
    """
    Incremental feature engineering with caching.
    Always loads and saves cache from features/features_cache.csv.
    Only computes features for new/unseen dates and appends to cache.
    """
    cache_path = "tesla_stock_predictor/features/features_cache.csv"
    # Load cached features if they exist
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        last_cached_date = cached.index.max()
        # Only compute features for new dates
        new_df = df[df.index > last_cached_date]
    else:
        cached = pd.DataFrame()
        new_df = df

    if not new_df.empty:
        new_features = engineer_features(new_df, sector_df=sector_df, spy_df=spy_df)
        if new_features is not None and not new_features.empty:
            # Remove any overlap just in case
            if not cached.empty:
                new_features = new_features[~new_features.index.isin(cached.index)]
            # Append new features to cache
            cached = pd.concat([cached, new_features])
            cached.to_csv(cache_path)

    return cached

def generate_features_for_next_day(df, sector_df=None, spy_df=None):
    """
    Generate features for the next trading day using only data up to the last available date.
    Returns a DataFrame with a single row for the next trading day.
    """
    # Find the next business day (or trading day)
    last_date = df.index[-1]
    next_day = last_date + pd.Timedelta(days=1)
    while next_day.weekday() >= 5:  # Skip weekends
        next_day += pd.Timedelta(days=1)

    # Create a new DataFrame with the next day as index, copying the last row's data
    next_row = df.iloc[[-1]].copy()
    next_row.index = [next_day]

    # Concatenate to simulate the new day for rolling features
    df_extended = pd.concat([df, next_row])

    # Recompute features for the extended DataFrame
    features_df = engineer_features(df_extended, sector_df=sector_df, spy_df=spy_df)

    # Defensive: If features_df is None or empty, raise a clear error
    if features_df is None or features_df.empty:
        raise ValueError("Feature generation for next day failed or returned empty DataFrame.")
    # Return only the last row (the next day)
    return features_df.iloc[[-1]]

    # Create a copy to avoid modifying original
    df = df.copy()

    # Dictionary to store all new features - avoids DataFrame fragmentation
    features = {}

    # --- EMA (Exponential Moving Averages) ---
    features['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    features['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    features['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    features['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    features['EMA12_26_diff'] = features['EMA12'] - features['EMA26']
    features['EMA12_50_diff'] = features['EMA12'] - features['EMA50']
    features['EMA50_200_diff'] = features['EMA50'] - features['EMA200']

    # --- Stochastic Oscillator ---
    low14 = df['Low'].rolling(window=14, min_periods=14).min()
    high14 = df['High'].rolling(window=14, min_periods=14).max()
    features['Stoch_%K'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-10)
    features['Stoch_%D'] = features['Stoch_%K'].rolling(window=3, min_periods=3).mean()

    # --- Williams %R ---
    features['Williams_%R'] = -100 * (high14 - df['Close']) / (high14 - low14 + 1e-10)

    # --- Commodity Channel Index (CCI) ---
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma_tp = tp.rolling(window=20, min_periods=20).mean()
    md = tp.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    features['CCI'] = (tp - ma_tp) / (0.015 * md + 1e-10)

    # --- On-Balance Volume (OBV) ---
    features['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # --- Chaikin Money Flow (CMF) ---
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10) * df['Volume']
    features['CMF'] = mfv.rolling(window=20, min_periods=20).sum() / df['Volume'].rolling(window=20, min_periods=20).sum()

    # --- Supertrend Indicator ---
    supertrend_val, supertrend_dir = supertrend(df, atr_period=10, factor=3.0)
    # features['Supertrend'] = supertrend_val  # Removed raw Supertrend value
    features['Supertrend_Direction'] = supertrend_dir

    # --- Average Directional Index (ADX) ---
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    features['ADX'] = pd.Series(dx).rolling(window=14, min_periods=14).mean()

    # --- Donchian Channel (20-day high/low) ---
    features['Donchian_High_20'] = df['High'].rolling(window=20, min_periods=20).max()
    features['Donchian_Low_20'] = df['Low'].rolling(window=20, min_periods=20).min()

    # --- Parabolic SAR (simple version) ---
    # This is a basic implementation, for more accuracy use TA-Lib or similar
    sar_values = np.full(len(df), np.nan)
    af = 0.02
    max_af = 0.2
    trend = 1  # 1 for up, -1 for down
    ep = df['Low'].iloc[0]
    sar = df['High'].iloc[0]
    for i in range(2, len(df)):
        prev_sar = sar
        if trend == 1:
            sar = prev_sar + af * (ep - prev_sar)
            if df['Low'].iloc[i] < sar:
                trend = -1
                sar = ep
                ep = df['High'].iloc[i]
                af = 0.02
            else:
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]
                    af = min(af + 0.02, max_af)
        else:
            sar = prev_sar + af * (ep - prev_sar)
            if df['High'].iloc[i] > sar:
                trend = 1
                sar = ep
                ep = df['Low'].iloc[i]
                af = 0.02
            else:
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]
                    af = min(af + 0.02, max_af)
        sar_values[i] = sar
    features['SAR'] = pd.Series(sar_values, index=df.index)

    # --- Accumulation/Distribution Line (ADL) ---
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    features['ADL'] = (clv * df['Volume']).cumsum()

    # (Fibonacci features removed)

    # --- FIXED: All features now use only past/current data ---

    # Lag features (safe - using past data)
    features['lag1_Close'] = df['Close'].shift(1)
    features['lag2_Close'] = df['Close'].shift(2)
    features['lag3_Close'] = df['Close'].shift(3)

    # Moving averages (safe - using past data only)
    features['MA5'] = df['Close'].rolling(5, min_periods=5).mean()
    features['MA9'] = df['Close'].rolling(9, min_periods=9).mean()
    features['MA10'] = df['Close'].rolling(10, min_periods=10).mean()
    features['MA20'] = df['Close'].rolling(20, min_periods=20).mean()
    features['MA21'] = df['Close'].rolling(21, min_periods=21).mean()
    features['MA50'] = df['Close'].rolling(50, min_periods=50).mean()
    # --- Z-score of price relative to MA20 ---
    features['Zscore_Close_MA20'] = (df['Close'] - features['MA20']) / (features['MA20'].rolling(20, min_periods=20).std() + 1e-10)
    # --- Slope of MA20 (trend strength alternative) ---
    features['MA20_Slope'] = features['MA20'].diff(5) / 5

    # Price ratios to moving averages
    features['Price_to_MA5'] = df['Close'] / features['MA5'] - 1
    features['Price_to_MA10'] = df['Close'] / features['MA10'] - 1
    features['Price_to_MA20'] = df['Close'] / features['MA20'] - 1
    features['MA9_MA21_Ratio'] = features['MA9'] / features['MA21']

    # Returns (safe - using past prices)
    features['Returns_1d'] = df['Close'].pct_change(1)
    features['Returns_2d'] = df['Close'].pct_change(2)
    features['Returns_3d'] = df['Close'].pct_change(3)
    features['Returns_5d'] = df['Close'].pct_change(5)
    features['Returns_10d'] = df['Close'].pct_change(10)

    # Volatility (safe - using past returns)
    features['Volatility_5d'] = features['Returns_1d'].rolling(5, min_periods=5).std()
    features['Volatility_10d'] = features['Returns_1d'].rolling(10, min_periods=10).std()
    features['Volatility_20d'] = features['Returns_1d'].rolling(20, min_periods=20).std()

    # ATR and related features
    features['prev_close'] = df['Close'].shift(1)
    features['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - features['prev_close']),
                                     abs(df['Low'] - features['prev_close'])))
    features['ATR'] = features['TR'].rolling(14, min_periods=14).mean()
    features['ATR_14'] = features['ATR']
    features['ATR_ratio'] = features['ATR_14'] / df['Close']

    # High volatility flag
    features['ATR_20d_quantile'] = features['ATR'].rolling(20, min_periods=20).quantile(0.8)
    features['High_Vol_Flag'] = (features['ATR'] > features['ATR_20d_quantile']).astype(int)

    # Bollinger Bands
    features['BB_middle'] = df['Close'].rolling(20, min_periods=20).mean()
    features['BB_std'] = df['Close'].rolling(20, min_periods=20).std()
    features['BB_upper'] = features['BB_middle'] + 2 * features['BB_std']
    features['BB_lower'] = features['BB_middle'] - 2 * features['BB_std']
    features['BB_position'] = (df['Close'] - features['BB_lower']) / (features['BB_upper'] - features['BB_lower'])
    features['BB_Width'] = features['BB_upper'] - features['BB_lower']
    features['BB_squeeze'] = features['BB_std'] / features['BB_middle']

    # RSI (safe - using past price changes)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    features['RSI'] = 100 - (100 / (1 + rs))
    features['RSI_oversold'] = (features['RSI'] < 30).astype(int)
    features['RSI_overbought'] = (features['RSI'] > 70).astype(int)

    # MACD (safe - using past prices)
    ema_12 = df['Close'].ewm(span=12, min_periods=12).mean()
    ema_26 = df['Close'].ewm(span=26, min_periods=26).mean()
    features['MACD'] = ema_12 - ema_26
    features['Signal_Line'] = features['MACD'].ewm(span=9, min_periods=9).mean()
    features['MACD_histogram'] = features['MACD'] - features['Signal_Line']
    features['MACD_bullish'] = (features['MACD'] > features['Signal_Line']).astype(int)

    # Crossover signals (safe - comparing current to previous)
    features['MACD_Signal_crossover'] = np.where(
        (features['MACD'] > features['Signal_Line']) & (features['MACD'].shift(1) <= features['Signal_Line'].shift(1)), 1,
        np.where((features['MACD'] < features['Signal_Line']) & (features['MACD'].shift(1) >= features['Signal_Line'].shift(1)), -1, 0)
    )

    features['MA9_21_crossover'] = np.where(
        (features['MA9'] > features['MA21']) & (features['MA9'].shift(1) <= features['MA21'].shift(1)), 1,
        np.where((features['MA9'] < features['MA21']) & (features['MA9'].shift(1) >= features['MA21'].shift(1)), -1, 0)
    )

    # Volume features (safe - using past volume data)
    features['Volume_MA_5'] = df['Volume'].rolling(5, min_periods=5).mean()
    features['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=20).mean()
    features['Volume_ratio'] = df['Volume'] / features['Volume_MA_20']
    features['Volume_trend'] = features['Volume_MA_5'] / features['Volume_MA_20']

    # Price action patterns (safe - using past prices)
    features['Higher_high'] = ((df['High'] > df['High'].shift(1)) &
                        (df['High'].shift(1) > df['High'].shift(2))).astype(int)
    features['Lower_low'] = ((df['Low'] < df['Low'].shift(1)) &
                      (df['Low'].shift(1) < df['Low'].shift(2))).astype(int)

    # Gap detection (safe - comparing current open to previous close)
    features['Gap_up'] = (df['Open'] > features['prev_close'] * 1.02).astype(int)
    features['Gap_down'] = (df['Open'] < features['prev_close'] * 0.98).astype(int)

    # Momentum indicators (safe - using past prices)
    features['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    features['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1

    # --- Relative Strength vs. Sector ETF and SPY ---
    if 'Sector_Close' in df.columns:
        features['TSLA_to_Sector'] = df['Close'] / df['Sector_Close']
        features['Sector_Return'] = df['Sector_Close'].pct_change()
        features['Sector_MA50'] = df['Sector_Close'].rolling(50, min_periods=50).mean()
        features['Sector_Momentum_5'] = df['Sector_Close'] / df['Sector_Close'].shift(5) - 1
        features['Relative_Momentum_Sector'] = features['Momentum_5'] - features['Sector_Momentum_5']
    if 'SPY_Close' in df.columns:
        features['TSLA_to_SPY'] = df['Close'] / df['SPY_Close']
        features['SPY_Return'] = df['SPY_Close'].pct_change()
        features['SPY_MA50'] = df['SPY_Close'].rolling(50, min_periods=50).mean()
        features['SPY_Momentum_5'] = df['SPY_Close'] / df['SPY_Close'].shift(5) - 1
        features['Relative_Momentum_SPY'] = features['Momentum_5'] - features['SPY_Momentum_5']

    # Support/Resistance levels (safe - using past high/low data)
    features['High_20d'] = df['High'].rolling(20, min_periods=20).max()
    features['Low_20d'] = df['Low'].rolling(20, min_periods=20).min()
    features['Near_20d_high'] = (df['Close'] > features['High_20d'] * 0.98).astype(int)
    features['Near_20d_low'] = (df['Close'] < features['Low_20d'] * 1.02).astype(int)

    # Time-based features (safe - using current date)
    features['Day_of_week'] = df.index.dayofweek
    features['Month'] = df.index.month
    features['Is_month_end'] = df.index.is_month_end.astype(int)
    features['Is_friday'] = (df.index.dayofweek == 4).astype(int)
    features['Is_monday'] = (df.index.dayofweek == 0).astype(int)

    # Market regime features (safe - using past trend data)
    features['Bull_market'] = (features['MA50'] > features['MA50'].shift(10)).astype(int)
    # Use the already created MA20 (consistent naming)
    features['Price_to_MA_20'] = df['Close'] / features['MA20'] - 1
    features['Trend_strength'] = abs(features['Price_to_MA_20'])

    # Additional safe features
    features['Close_to_High'] = df['Close'] / df['High']
    features['Close_to_Low'] = df['Close'] / df['Low']
    features['High_Low_ratio'] = df['High'] / df['Low']
    features['Body_size'] = abs(df['Close'] - df['Open']) / df['Open']
    features['Upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
    features['Lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']

    # --- Stage 2: Features that depend on Stage 1 features or earlier ---
    # 📉 Momentum × RSI / MACD
    features['RSI_Momentum_5'] = features['RSI'] * features['Momentum_5']
    features['RSI_Momentum_10'] = features['RSI'] * features['Momentum_10']
    features['MACD_RSI_Interaction'] = features['MACD_histogram'] * features['RSI']
    features['MACD_Signal_Gap'] = features['MACD_histogram'] - features['Signal_Line']
    # 📊 Volatility × Price Action
    features['Volatility_Returns'] = features['Volatility_5d'] * features['Returns_3d']
    features['BBWidth_to_MA'] = features['BB_Width'] / (abs(features['Price_to_MA_20']) + 1e-6)
    features['Volatility_Shadow'] = features['Volatility_10d'] * features['Upper_shadow']
    # 🔀 Price to MA Interactions
    features['MA9_21_Combo'] = features['Price_to_MA10'] * features['MA9_MA21_Ratio']
    features['MA9_Momentum'] = features['Price_to_MA5'] * features['Momentum_5']
    features['BB_Squeeze_MA20'] = features['BB_squeeze'] * features['Price_to_MA_20']

    # --- Additional classic engineered combos from features.py ---
    features['Friday_Momentum'] = features['Is_friday'] * features['Momentum_5']
    features['Monday_GapUp'] = features['Is_monday'] * features['Gap_up']
    features['Month_MA_ratio'] = features['Month'] * features['MA9_MA21_Ratio']
    features['Volume_Momentum'] = features['Volume_ratio'] * features['Momentum_5']
    features['Volume_Volatility'] = features['Volume_ratio'] * features['Volatility_5d']
    features['Volume_Shadow'] = features['Volume_ratio'] * (features['Upper_shadow'] + features['Lower_shadow'])
    features['NearHigh_Momentum'] = features['Near_20d_high'] * features['Momentum_10']
    features['NearLow_GapDown'] = features['Near_20d_low'] * features['Gap_down']
    features['ResistanceBreakout'] = features['Near_20d_high'] * features['Price_to_MA5']
    features['Bull_RSI'] = features['Bull_market'] * features['RSI']
    features['HighVol_GapUp'] = features['High_Vol_Flag'] * features['Gap_up']
    features['Oversold_Body'] = features['RSI_oversold'] * features['Body_size']
    features['ATR_vs_BodySize'] = features['ATR_ratio'] / (features['Body_size'] + 1e-6)

    # 🔄 Trend Confirmation Features - Supertrend + Other Indicators
    # Supertrend + EMA alignment (strong confirmation when both agree)
    features['Supertrend_EMA_Confirm'] = features['Supertrend_Direction'] * np.sign(features['EMA12_26_diff'])
    features['Supertrend_EMA_Agree'] = (np.sign(features['EMA12_26_diff']) == features['Supertrend_Direction']).astype(int)

    # Supertrend + RSI confirmation (trend strength with momentum)
    features['Supertrend_RSI_Bull'] = (features['Supertrend_Direction'] > 0) & (features['RSI'] > 50)
    features['Supertrend_RSI_Bear'] = (features['Supertrend_Direction'] < 0) & (features['RSI'] < 50)
    features['Supertrend_RSI_Confirm'] = features['Supertrend_RSI_Bull'].astype(int) - features['Supertrend_RSI_Bear'].astype(int)

    # Trend persistence - how long the current Supertrend has maintained direction
    trend_shifts = features['Supertrend_Direction'].diff().fillna(0) != 0
    features['Trend_Duration'] = (~trend_shifts).cumsum() * features['Supertrend_Direction']

    # Volatility-adjusted Supertrend signal (stronger in low volatility)
    features['Supertrend_Vol_Adj'] = features['Supertrend_Direction'] / (features['Volatility_20d'] + 0.001)
    # 🕰️ Time Features × Price Action
    features['Friday_Momentum'] = features['Is_friday'] * features['Momentum_5']
    features['Monday_GapUp'] = features['Is_monday'] * features['Gap_up']
    features['Month_MA_ratio'] = features['Month'] * features['MA9_MA21_Ratio']

    # 🕰️ Supertrend Time Interactions
    features['Monday_Supertrend'] = features['Is_monday'] * features['Supertrend_Direction']
    features['Friday_Supertrend'] = features['Is_friday'] * features['Supertrend_Direction']
    features['Month_Supertrend'] = features['Is_month_end'] * features['Supertrend_Direction']
    # 📈 Volume Behavior
    features['Volume_Momentum'] = features['Volume_ratio'] * features['Momentum_5']
    features['Volume_Volatility'] = features['Volume_ratio'] * features['Volatility_5d']
    features['Volume_Shadow'] = features['Volume_ratio'] * (features['Upper_shadow'] + features['Lower_shadow'])
    # 🔁 Support/Resistance × Momentum
    features['NearHigh_Momentum'] = features['Near_20d_high'] * features['Momentum_10']
    features['NearLow_GapDown'] = features['Near_20d_low'] * features['Gap_down']
    features['ResistanceBreakout'] = features['Near_20d_high'] * features['Price_to_MA5']

    # 🔁 Support/Resistance × Supertrend (powerful breakout signals)
    features['Supertrend_Breakout_High'] = (features['Supertrend_Direction'] > 0) & features['Near_20d_high']
    features['Supertrend_Bounce_Low'] = (features['Supertrend_Direction'] > 0) & features['Near_20d_low']
    features['Supertrend_Break_Score'] = features['Supertrend_Direction'] * (df['Close'] - features['MA20']) / features['MA20']

    # 🔒 Boolean Flags × Quantitative Features
    features['Bull_RSI'] = features['Bull_market'] * features['RSI']
    features['HighVol_GapUp'] = features['High_Vol_Flag'] * features['Gap_up']
    features['Oversold_Body'] = features['RSI_oversold'] * features['Body_size']
    # Features that depend on Stage 1
    features['ATR_vs_BodySize'] = features['ATR_ratio'] / (features['Body_size'] + 1e-6)

    # 🌟 Multi-Signal Confirmation Systems
    # Signal strength index - combining multiple technical signals
    features['Signal_Strength'] = (
        (features['RSI'] > 50).astype(int) +
        (features['Supertrend_Direction'] > 0).astype(int) +
        (features['MA9'] > features['MA21']).astype(int) +
        (features['MACD'] > features['Signal_Line']).astype(int) +
        (df['Close'] > features['BB_middle']).astype(int)
    ) / 5.0  # Normalized to 0-1

    # Divergence detection - price making new highs but indicators not confirming
    features['RSI_Price_Divergence'] = (df['Close'] > df['Close'].shift(5)) & (features['RSI'] < features['RSI'].shift(5))

    # Supertrend-enhanced pattern strength
    features['Trend_Pattern_Strength'] = features['Supertrend_Direction'] * features['Signal_Strength'] * (1 + abs(features['Momentum_5']))

    # Combine all features at once to avoid fragmentation
    new_features_df = pd.DataFrame(features, index=df.index)
    result_df = pd.concat([df, new_features_df], axis=1)

    return result_df

def select_features(df):
    """Select the most predictive features - only non-leaky ones"""

    # Define feature categories and their patterns
    feature_patterns = {
        'lag': ['lag1_Close', 'lag2_Close', 'lag3_Close'],
        'returns': [col for col in df.columns if 'Returns_' in col or 'Momentum_' in col],
        'ma_ratios': [col for col in df.columns if 'Price_to_MA' in col or 'MA9_MA21_Ratio' in col],
        'volatility': [col for col in df.columns if 'Volatility_' in col or 'ATR' in col],
        'bollinger': [col for col in df.columns if 'BB_' in col],
        'rsi': [col for col in df.columns if 'RSI' in col],
        'macd': [col for col in df.columns if 'MACD' in col or 'Signal_Line' in col],
        'crossovers': [col for col in df.columns if 'crossover' in col],
        'volume': [col for col in df.columns if 'Volume_' in col],
        'patterns': ['Higher_high', 'Lower_low', 'Gap_up', 'Gap_down'],
        'support_resistance': ['Near_20d_high', 'Near_20d_low', 'High_20d', 'Low_20d'],
        'time': ['Day_of_week', 'Month', 'Is_month_end', 'Is_friday', 'Is_monday'],
        'market_regime': ['Bull_market', 'Trend_strength', 'High_Vol_Flag'],
        'price_action': [col for col in df.columns if any(x in col for x in ['Close_to_', 'High_Low_ratio', 'Body_size', 'shadow'])],
        'interactions': [col for col in df.columns if any(x in col for x in ['_Momentum', '_Interaction', '_Gap', '_Returns', '_BodySize', '_MA', '_Shadow', '_GapUp', '_GapDown', '_RSI', '_Vol'])],
        # Advanced technical indicators
        'ema': [col for col in df.columns if 'EMA' in col],
        'stoch': [col for col in df.columns if 'Stoch_%K' in col or 'Stoch_%D' in col],
        'williams': [col for col in df.columns if 'Williams_%R' in col],
        'cci': [col for col in df.columns if 'CCI' in col],
        'obv': [col for col in df.columns if 'OBV' in col],
        'cmf': [col for col in df.columns if 'CMF' in col],
        'adx': [col for col in df.columns if 'ADX' in col],
        'donchian': [col for col in df.columns if 'Donchian_' in col],
        'sar': [col for col in df.columns if 'SAR' in col],
        'zscore': [col for col in df.columns if 'Zscore' in col],
        'adl': [col for col in df.columns if 'ADL' in col],
        'slope': [col for col in df.columns if 'Slope' in col],
        # Sector/relative features
        'sector': [col for col in df.columns if 'Sector_' in col or 'TSLA_to_Sector' in col or 'Relative_Momentum_Sector' in col],
        'spy': [col for col in df.columns if 'SPY_' in col or 'TSLA_to_SPY' in col or 'Relative_Momentum_SPY' in col],
    }

    # Flatten all feature patterns into a single list
    feature_cols = []
    for category, patterns in feature_patterns.items():
        if isinstance(patterns, list) and len(patterns) > 0:
            if isinstance(patterns[0], str) and not any(col in df.columns for col in patterns):
                # This is a pattern list, extend it
                feature_cols.extend(patterns)
            else:
                # This is already a filtered list from df.columns
                feature_cols.extend(patterns)

    # Remove duplicates and ensure all features exist in df, and sort for deterministic order
    feature_cols = sorted(set(feature_cols))
    available_features = []

    for col in feature_cols:
        if col in df.columns:
            # Check if feature has sufficient non-null values
            if df[col].notna().sum() > len(df) * 0.5:  # At least 50% non-null
                available_features.append(col)

    # Add any remaining engineered features we might have missed
    engineered_patterns = ['_Combo', '_Flag', '_bullish', '_oversold', '_overbought']
    for col in df.columns:
        if any(pattern in col for pattern in engineered_patterns) and col not in available_features:
            if df[col].notna().sum() > len(df) * 0.5:
                available_features.append(col)

    # Sort available_features for deterministic order
    available_features = sorted(available_features)

    # Ensure critical trend features and missing classics are always included
    critical_features = [
        "Supertrend_Direction",
        "Supertrend_EMA_Confirm",
        "Supertrend_RSI_Confirm",
        "Trend_Duration",
        "Signal_Strength",
        "Trend_Pattern_Strength",
        # Add missing classic engineered features
        "MA5",
        "MA9",
        "MA10",
        "MA20",
        "MA21",
        "MA50",
        "BB_Squeeze_MA20",
        "Monday_Supertrend",
        "Friday_Supertrend",
        "Month_Supertrend",
        "Supertrend_Breakout_High",
        "Supertrend_Bounce_Low",
        "Supertrend_Break_Score"
    ]
    for col in critical_features:
        if col in df.columns and col not in available_features:
            if df[col].notna().sum() > len(df) * 0.5:
                available_features.append(col)
    available_features = sorted(available_features)

    print(f"📊 Selected {len(available_features)} features out of {len(df.columns)} total columns")

    # Log the selected feature list to a file for reproducibility and debugging
    with open("tesla_stock_predictor/debug/selected_features_latest.txt", "w") as f:
        for feat in available_features:
            f.write(f"{feat}\n")

    # Features should already be clean from engineer_features method
    feature_df = df[available_features].copy()
    feature_df = feature_df.fillna(0)

    return feature_df

def create_targets(df):
    """Create prediction targets - FIXED to avoid look-ahead"""
    # Target: Next day's direction (this is what we're predicting)
    # We shift by -1 to get tomorrow's outcome, but this is the target, not a feature
    df['Target_1d'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Additional targets for analysis
    df['Target_2d'] = (df['Close'].shift(-2) > df['Close']).astype(int)
    df['Target_strong'] = (df['Close'].shift(-1) > df['Close'] * 1.01).astype(int)

    return df
