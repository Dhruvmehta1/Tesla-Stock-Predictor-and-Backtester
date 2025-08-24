import numpy as np
import pandas as pd
import os
from scipy import stats

def supertrend(df, atr_period=10, factor=3.0):
    """
    Calculate Supertrend indicator without look-ahead bias.
    Returns supertrend value and direction (1 for uptrend, -1 for downtrend).
    """
    # Safety check - ensure we have required columns
    required_cols = ['High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Missing required column '{col}' for supertrend calculation.")
            return pd.Series(0, index=df.index), pd.Series(0, index=df.index)

    df = df.copy()
    high = df['High']
    low = df['Low']
    close = df['Close']

    # Calculate true range (using previous close to avoid lookahead)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean()

    # Calculate bands using HL2
    hl2 = (high + low) / 2
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)

    # Initialize Supertrend - shift everything by 1 to prevent lookahead
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)

    # Calculate Supertrend using strictly causal logic
    for i in range(1, len(df)):
        if i == 1:
            supertrend.iloc[i] = lowerband.iloc[i-1] if not pd.isna(lowerband.iloc[i-1]) else 0
            continue

        prev_close_val = close.iloc[i-1]
        prev_supertrend = supertrend.iloc[i-1]

        # Use previous day's bands to avoid lookahead
        curr_upper = upperband.iloc[i-1] if not pd.isna(upperband.iloc[i-1]) else prev_supertrend
        curr_lower = lowerband.iloc[i-1] if not pd.isna(lowerband.iloc[i-1]) else prev_supertrend

        if prev_supertrend <= prev_close_val:
            # Was in uptrend - use lower band
            supertrend.iloc[i] = max(curr_lower, prev_supertrend)
        else:
            # Was in downtrend - use upper band
            supertrend.iloc[i] = min(curr_upper, prev_supertrend)

        # Determine direction based on previous close vs current supertrend
        if prev_close_val <= supertrend.iloc[i]:
            direction.iloc[i] = -1  # Downtrend
        else:
            direction.iloc[i] = 1   # Uptrend

    return supertrend, direction

def create_advanced_pattern_features(df):
    """
    Create advanced pattern recognition features that combine multiple indicators
    to detect complex market patterns that standalone features cannot capture.
    """
    features = {}

    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Missing {col} for advanced pattern features")
            return features

    # Use lagged prices to prevent lookahead bias
    prev_close = df['Close'].shift(1)
    prev_high = df['High'].shift(1)
    prev_low = df['Low'].shift(1)
    prev_open = df['Open'].shift(1)
    prev_volume = df['Volume'].shift(1)

    # === MOMENTUM CONVERGENCE/DIVERGENCE PATTERNS ===

    # Multi-timeframe momentum alignment
    short_momentum = prev_close / prev_close.shift(3) - 1  # 3-day
    med_momentum = prev_close / prev_close.shift(7) - 1    # 1-week
    long_momentum = prev_close / prev_close.shift(21) - 1  # 1-month

    # Momentum alignment score (-3 to +3)
    momentum_signs = np.sign(short_momentum) + np.sign(med_momentum) + np.sign(long_momentum)
    features['Momentum_Alignment'] = momentum_signs

    # Momentum acceleration pattern
    momentum_accel = short_momentum - short_momentum.shift(1)
    features['Momentum_Acceleration'] = momentum_accel

    # Divergence between price and volume momentum
    volume_momentum = prev_volume / prev_volume.shift(5) - 1
    price_volume_divergence = np.sign(med_momentum) - np.sign(volume_momentum)
    features['Price_Volume_Divergence'] = price_volume_divergence

    # === VOLATILITY REGIME PATTERNS ===

    # Multi-period volatility pattern
    vol_short = prev_close.rolling(5).std()
    vol_med = prev_close.rolling(20).std()
    vol_long = prev_close.rolling(60).std()

    # Volatility regime classification
    vol_regime = np.where(vol_short > vol_med * 1.5, 2,  # High vol
                 np.where(vol_short < vol_med * 0.5, 0, 1))  # Low/Normal vol
    features['Volatility_Regime'] = vol_regime

    # Volatility breakout pattern - when volatility expands rapidly
    vol_expansion = (vol_short / (vol_med + 1e-10)) > 2.0
    features['Volatility_Breakout'] = vol_expansion.astype(int)

    # Volatility compression pattern - potential breakout setup
    vol_compression = (vol_short / (vol_long + 1e-10)) < 0.3
    features['Volatility_Compression'] = vol_compression.astype(int)

    # === PRICE ACTION PATTERNS ===

    # Complex candlestick patterns
    body_size = abs(prev_close - prev_open) / (prev_open + 1e-10)
    upper_wick = (prev_high - np.maximum(prev_open, prev_close)) / (prev_close + 1e-10)
    lower_wick = (np.minimum(prev_open, prev_close) - prev_low) / (prev_close + 1e-10)

    # Doji pattern (small body, long wicks)
    doji_pattern = ((body_size < 0.01) & ((upper_wick + lower_wick) > 0.02)).astype(int)
    features['Doji_Pattern'] = doji_pattern

    # Hammer/Hanging man pattern
    hammer_pattern = ((lower_wick > 2 * body_size) & (upper_wick < body_size)).astype(int)
    features['Hammer_Pattern'] = hammer_pattern

    # Shooting star pattern
    star_pattern = ((upper_wick > 2 * body_size) & (lower_wick < body_size)).astype(int)
    features['Shooting_Star_Pattern'] = star_pattern

    # === TREND STRENGTH & QUALITY PATTERNS ===

    # Calculate EMAs for trend analysis
    ema_fast = prev_close.ewm(span=8, min_periods=8).mean()
    ema_med = prev_close.ewm(span=21, min_periods=21).mean()
    ema_slow = prev_close.ewm(span=50, min_periods=50).mean()

    # Trend quality - how consistently price stays above/below EMAs
    above_emas = ((prev_close > ema_fast) & (ema_fast > ema_med) & (ema_med > ema_slow)).astype(int)
    below_emas = ((prev_close < ema_fast) & (ema_fast < ema_med) & (ema_med < ema_slow)).astype(int)
    features['Strong_Uptrend'] = above_emas
    features['Strong_Downtrend'] = below_emas

    # Trend strength score based on EMA alignment
    ema_alignment = np.where(above_emas, 2, np.where(below_emas, -2, 0))
    features['Trend_Strength_Score'] = ema_alignment

    # EMA convergence/divergence pattern
    ema_spread = (ema_fast - ema_slow) / (ema_slow + 1e-10)
    ema_convergence = (abs(ema_spread) < abs(ema_spread.shift(5))).astype(int)
    features['EMA_Convergence'] = ema_convergence

    # === SUPPORT/RESISTANCE PATTERNS ===

    # Dynamic support/resistance levels
    high_20 = prev_high.rolling(20, min_periods=20).max()
    low_20 = prev_low.rolling(20, min_periods=20).min()
    high_50 = prev_high.rolling(50, min_periods=50).max()
    low_50 = prev_low.rolling(50, min_periods=50).min()

    # Distance from key levels
    dist_from_high = (high_20 - prev_close) / (prev_close + 1e-10)
    dist_from_low = (prev_close - low_20) / (prev_close + 1e-10)

    features['Distance_From_20D_High'] = dist_from_high
    features['Distance_From_20D_Low'] = dist_from_low

    # Support/resistance test pattern
    resistance_test = ((prev_high >= high_20 * 0.99) & (prev_close < high_20 * 0.98)).astype(int)
    support_test = ((prev_low <= low_20 * 1.01) & (prev_close > low_20 * 1.02)).astype(int)

    features['Resistance_Test'] = resistance_test
    features['Support_Test'] = support_test

    # === VOLUME PATTERNS ===

    # Volume-price relationship patterns
    vol_ma = prev_volume.rolling(20, min_periods=20).mean()
    price_up = (prev_close > prev_close.shift(1)).astype(int)
    volume_above_avg = (prev_volume > vol_ma * 1.2).astype(int)

    # Bullish/bearish volume patterns
    features['Bullish_Volume'] = (price_up & volume_above_avg).astype(int)
    features['Bearish_Volume'] = ((1 - price_up) & volume_above_avg).astype(int)

    # Volume accumulation/distribution pattern
    vol_trend = prev_volume.rolling(10).mean() / prev_volume.rolling(30).mean()
    features['Volume_Accumulation'] = (vol_trend > 1.1).astype(int)
    features['Volume_Distribution'] = (vol_trend < 0.9).astype(int)

    # === MEAN REVERSION PATTERNS ===

    # Calculate Bollinger Bands
    bb_period = 20
    bb_ma = prev_close.rolling(bb_period, min_periods=bb_period).mean()
    bb_std = prev_close.rolling(bb_period, min_periods=bb_period).std()
    bb_upper = bb_ma + (2 * bb_std)
    bb_lower = bb_ma - (2 * bb_std)

    # Bollinger Band squeeze pattern
    bb_squeeze = ((bb_upper - bb_lower) / (bb_ma + 1e-10)) < 0.1
    features['BB_Squeeze'] = bb_squeeze.astype(int)

    # Mean reversion setup - price at extremes with low volatility
    at_bb_upper = (prev_close > bb_upper * 0.98).astype(int)
    at_bb_lower = (prev_close < bb_lower * 1.02).astype(int)
    low_vol = (vol_short < vol_med).astype(int)

    features['Mean_Reversion_Up_Setup'] = (at_bb_lower & low_vol).astype(int)
    features['Mean_Reversion_Down_Setup'] = (at_bb_upper & low_vol).astype(int)

    # === MOMENTUM OSCILLATOR PATTERNS ===

    # RSI-based patterns (using proper lagged calculation)
    rsi_period = 14
    delta = prev_close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # RSI divergence pattern
    rsi_higher_low = ((rsi > rsi.shift(5)) & (prev_close < prev_close.shift(5))).astype(int)
    rsi_lower_high = ((rsi < rsi.shift(5)) & (prev_close > prev_close.shift(5))).astype(int)

    features['RSI_Bullish_Divergence'] = rsi_higher_low
    features['RSI_Bearish_Divergence'] = rsi_lower_high

    # RSI momentum pattern
    rsi_momentum = rsi - rsi.shift(3)
    features['RSI_Momentum'] = rsi_momentum

    # === BREAKOUT PATTERNS ===

    # Price breakout from consolidation
    range_20 = high_20 - low_20
    consolidation = (range_20 / (prev_close + 1e-10)) < 0.05  # Tight range

    upward_breakout = (prev_close > high_20.shift(1)).astype(int)
    downward_breakout = (prev_close < low_20.shift(1)).astype(int)

    features['Consolidation_Pattern'] = consolidation.astype(int)
    features['Upward_Breakout'] = (upward_breakout & consolidation.shift(1)).astype(int)
    features['Downward_Breakout'] = (downward_breakout & consolidation.shift(1)).astype(int)

    # === MULTI-INDICATOR CONFLUENCE ===

    # Bullish confluence score
    bullish_signals = (
        (momentum_signs > 0).astype(int) +
        above_emas +
        (rsi > 50).astype(int) +
        (prev_close > bb_ma).astype(int) +
        volume_above_avg * price_up
    )
    features['Bullish_Confluence'] = bullish_signals

    # Bearish confluence score
    bearish_signals = (
        (momentum_signs < 0).astype(int) +
        below_emas +
        (rsi < 50).astype(int) +
        (prev_close < bb_ma).astype(int) +
        volume_above_avg * (1 - price_up)
    )
    features['Bearish_Confluence'] = bearish_signals

    # === VOLATILITY-ADJUSTED PATTERNS ===

    # Normalize patterns by current volatility regime
    vol_adj_factor = vol_short / (vol_med + 1e-10)

    # Volatility-adjusted momentum
    features['Vol_Adj_Momentum'] = med_momentum / (vol_adj_factor + 1e-10)

    # Volatility-adjusted support/resistance distances
    features['Vol_Adj_Resistance_Dist'] = dist_from_high / (vol_adj_factor + 1e-10)
    features['Vol_Adj_Support_Dist'] = dist_from_low / (vol_adj_factor + 1e-10)

    # === MARKET MICROSTRUCTURE PATTERNS ===

    # Intraday strength pattern
    intraday_strength = (prev_close - prev_low) / (prev_high - prev_low + 1e-10)
    features['Intraday_Strength'] = intraday_strength

    # Gap patterns with context
    gap_up = prev_open > prev_close.shift(1) * 1.005
    gap_down = prev_open < prev_close.shift(1) * 0.995

    # Gap fill patterns
    gap_fill_up = (gap_down & (prev_high > prev_close.shift(1))).astype(int)
    gap_fill_down = (gap_up & (prev_low < prev_close.shift(1))).astype(int)

    features['Gap_Fill_Up'] = gap_fill_up
    features['Gap_Fill_Down'] = gap_fill_down

    # === STATISTICAL PATTERNS ===

    # Price rank within recent range
    price_rank_20 = prev_close.rolling(20).rank(pct=True)
    features['Price_Rank_20D'] = price_rank_20

    # Volume rank within recent range
    vol_rank_20 = prev_volume.rolling(20).rank(pct=True)
    features['Volume_Rank_20D'] = vol_rank_20

    # Unusual volume with price movement
    unusual_volume = (vol_rank_20 > 0.8) & (abs(med_momentum) > vol_med * 2)
    features['Unusual_Volume_Move'] = unusual_volume.astype(int)

    return features

def engineer_features(df, sector_df=None, spy_df=None):
    """Enhanced feature engineering WITHOUT data leakage - completely rewritten"""

    if df is None or df.empty:
        print("ERROR: Empty or None dataframe passed to engineer_features")
        return pd.DataFrame()

    df = df.copy()
    features = {}

    # Verify required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return df

    # CRITICAL: All calculations use shifted (lagged) data to prevent lookahead bias
    prev_close = df['Close'].shift(1)
    prev_high = df['High'].shift(1)
    prev_low = df['Low'].shift(1)
    prev_open = df['Open'].shift(1)
    prev_volume = df['Volume'].shift(1)

    # === BASIC TECHNICAL INDICATORS (All using lagged data) ===

    # Moving Averages - using previous day's data only
    for period in [5, 9, 10, 20, 21, 50, 200]:
        features[f'MA{period}'] = prev_close.rolling(period, min_periods=period).mean()

    # Exponential Moving Averages
    for period in [12, 26, 50, 200]:
        features[f'EMA{period}'] = prev_close.ewm(span=period, min_periods=period).mean()

    # EMA differences (trend indicators)
    if 'EMA12' in features and 'EMA26' in features:
        features['EMA12_26_diff'] = features['EMA12'] - features['EMA26']
    if 'EMA12' in features and 'EMA50' in features:
        features['EMA12_50_diff'] = features['EMA12'] - features['EMA50']
    if 'EMA50' in features and 'EMA200' in features:
        features['EMA50_200_diff'] = features['EMA50'] - features['EMA200']

    # === PRICE-TO-MA RATIOS (Using lagged data) ===
    for period in [5, 10, 20, 50]:
        ma_col = f'MA{period}'
        if ma_col in features:
            features[f'Price_to_{ma_col}'] = (prev_close / (features[ma_col] + 1e-10)) - 1

    # Cross-MA ratios
    if 'MA9' in features and 'MA21' in features:
        features['MA9_MA21_Ratio'] = features['MA9'] / (features['MA21'] + 1e-10)

    # === RETURNS & LAG FEATURES ===
    for lag in [1, 2, 3, 5, 10, 20]:
        features[f'Returns_{lag}d'] = df['Close'].pct_change(lag)
        features[f'lag{lag}_Close'] = df['Close'].shift(lag)

    # === VOLATILITY MEASURES ===
    returns_1d = features['Returns_1d']
    for window in [5, 10, 20, 60]:
        features[f'Volatility_{window}d'] = returns_1d.rolling(window, min_periods=window).std()

    # === TRUE RANGE & ATR ===
    # Calculate True Range using properly lagged previous close
    features['prev_close'] = prev_close
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - prev_close)
    tr3 = abs(df['Low'] - prev_close)
    features['TR'] = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

    for period in [7, 14, 21]:
        features[f'ATR_{period}'] = features['TR'].rolling(period, min_periods=period).mean()

    # ATR-based features
    if 'ATR_14' in features:
        features['ATR_ratio'] = features['ATR_14'] / (prev_close + 1e-10)
        atr_quantile = features['ATR_14'].rolling(20, min_periods=20).quantile(0.8)
        features['High_Vol_Flag'] = (features['ATR_14'] > atr_quantile).astype(int)

    # === RSI (Properly calculated with lagged data) ===
    rsi_period = 14
    delta = prev_close.diff()  # Change from previous to previous-previous
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    features['RSI'] = 100 - (100 / (1 + rs))

    # RSI-based flags
    features['RSI_oversold'] = (features['RSI'] < 30).astype(int)
    features['RSI_overbought'] = (features['RSI'] > 70).astype(int)

    # === BOLLINGER BANDS (Properly lagged) ===
    bb_period = 20
    features['BB_middle'] = prev_close.rolling(bb_period, min_periods=bb_period).mean()
    bb_std = prev_close.rolling(bb_period, min_periods=bb_period).std()
    features['BB_std'] = bb_std
    features['BB_upper'] = features['BB_middle'] + 2 * bb_std
    features['BB_lower'] = features['BB_middle'] - 2 * bb_std

    # BB position and width
    bb_range = features['BB_upper'] - features['BB_lower']
    features['BB_position'] = (prev_close - features['BB_lower']) / (bb_range + 1e-10)
    features['BB_Width'] = bb_range
    features['BB_squeeze'] = bb_std / (features['BB_middle'] + 1e-10)

    # === MACD (Properly lagged) ===
    if 'EMA12' in features and 'EMA26' in features:
        features['MACD'] = features['EMA12'] - features['EMA26']
        features['Signal_Line'] = features['MACD'].ewm(span=9, min_periods=9).mean()
        features['MACD_histogram'] = features['MACD'] - features['Signal_Line']
        features['MACD_bullish'] = (features['MACD'] > features['Signal_Line']).astype(int)

        # MACD crossover signals
        macd_cross_up = ((features['MACD'] > features['Signal_Line']) &
                        (features['MACD'].shift(1) <= features['Signal_Line'].shift(1))).astype(int)
        macd_cross_down = ((features['MACD'] < features['Signal_Line']) &
                          (features['MACD'].shift(1) >= features['Signal_Line'].shift(1))).astype(int)
        features['MACD_Signal_crossover'] = macd_cross_up - macd_cross_down

    # === STOCHASTIC OSCILLATOR (Fixed to use lagged data) ===
    stoch_period = 14
    # Use lagged high/low for the lookback period
    lowest_low = prev_low.rolling(stoch_period, min_periods=stoch_period).min()
    highest_high = prev_high.rolling(stoch_period, min_periods=stoch_period).max()

    features['Stoch_K'] = 100 * ((prev_close - lowest_low) / (highest_high - lowest_low + 1e-10))
    features['Stoch_D'] = features['Stoch_K'].rolling(3, min_periods=3).mean()

    # === WILLIAMS %R (Fixed) ===
    features['Williams_R'] = -100 * ((highest_high - prev_close) / (highest_high - lowest_low + 1e-10))

    # === CCI (Fixed to use lagged data) ===
    cci_period = 20
    typical_price = (prev_high + prev_low + prev_close) / 3
    tp_ma = typical_price.rolling(cci_period, min_periods=cci_period).mean()
    mean_deviation = typical_price.rolling(cci_period, min_periods=cci_period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    features['CCI'] = (typical_price - tp_ma) / (0.015 * mean_deviation + 1e-10)

    # === VOLUME-BASED INDICATORS (All lagged) ===

    # Volume moving averages
    for period in [5, 20, 50]:
        features[f'Volume_MA_{period}'] = prev_volume.rolling(period, min_periods=period).mean()

    # Volume ratios
    if 'Volume_MA_20' in features:
        features['Volume_ratio'] = prev_volume / (features['Volume_MA_20'] + 1e-10)
    if 'Volume_MA_5' in features and 'Volume_MA_20' in features:
        features['Volume_trend'] = features['Volume_MA_5'] / (features['Volume_MA_20'] + 1e-10)

    # On-Balance Volume (OBV)
    price_direction = np.sign(prev_close.diff())
    features['OBV'] = (price_direction * prev_volume).fillna(0).cumsum()

    # Chaikin Money Flow (CMF)
    money_flow_multiplier = ((prev_close - prev_low) - (prev_high - prev_close)) / (prev_high - prev_low + 1e-10)
    money_flow_volume = money_flow_multiplier * prev_volume
    cmf_period = 21
    features['CMF'] = (money_flow_volume.rolling(cmf_period, min_periods=cmf_period).sum() /
                      prev_volume.rolling(cmf_period, min_periods=cmf_period).sum())

    # === MOMENTUM INDICATORS ===
    for period in [5, 10, 20]:
        features[f'Momentum_{period}'] = (prev_close / prev_close.shift(period)) - 1

    # === SUPERTREND ===
    supertrend_val, supertrend_dir = supertrend(df, atr_period=10, factor=3.0)
    features['Supertrend_Direction'] = supertrend_dir.shift(1)  # Lag by 1 to prevent lookahead

    # === ADVANCED PATTERN FEATURES ===
    pattern_features = create_advanced_pattern_features(df)
    features.update(pattern_features)

    # === SUPPORT/RESISTANCE LEVELS (Lagged) ===
    for period in [10, 20, 50]:
        features[f'High_{period}d'] = prev_high.rolling(period, min_periods=period).max()
        features[f'Low_{period}d'] = prev_low.rolling(period, min_periods=period).min()

    # Distance from support/resistance
    if 'High_20d' in features and 'Low_20d' in features:
        features['Near_20d_high'] = (prev_close > features['High_20d'] * 0.98).astype(int)
        features['Near_20d_low'] = (prev_close < features['Low_20d'] * 1.02).astype(int)

    # === PRICE ACTION PATTERNS (All lagged) ===
    features['Higher_high'] = ((prev_high > prev_high.shift(1)) &
                              (prev_high.shift(1) > prev_high.shift(2))).astype(int)
    features['Lower_low'] = ((prev_low < prev_low.shift(1)) &
                            (prev_low.shift(1) < prev_low.shift(2))).astype(int)

    # Gap detection (comparing current open to previous close - this is valid)
    features['Gap_up'] = (df['Open'] > prev_close * 1.02).astype(int)
    features['Gap_down'] = (df['Open'] < prev_close * 0.98).astype(int)

    # Intraday price action (using lagged OHLC)
    features['Close_to_High'] = prev_close / (prev_high + 1e-10)
    features['Close_to_Low'] = prev_close / (prev_low + 1e-10)
    features['High_Low_ratio'] = prev_high / (prev_low + 1e-10)

    # Candlestick body and shadow analysis
    features['Body_size'] = abs(prev_close - prev_open) / (prev_open + 1e-10)
    features['Upper_shadow'] = (prev_high - np.maximum(prev_open, prev_close)) / (prev_close + 1e-10)
    features['Lower_shadow'] = (np.minimum(prev_open, prev_close) - prev_low) / (prev_close + 1e-10)

    # === TIME-BASED FEATURES ===
    features['Day_of_week'] = df.index.dayofweek
    features['Month'] = df.index.month
    features['Is_month_end'] = df.index.is_month_end.astype(int)
    features['Is_friday'] = (df.index.dayofweek == 4).astype(int)
    features['Is_monday'] = (df.index.dayofweek == 0).astype(int)
    features['Is_quarter_end'] = df.index.is_quarter_end.astype(int)

    # === MARKET REGIME FEATURES ===
    if 'MA50' in features:
        features['Bull_market'] = (features['MA50'] > features['MA50'].shift(10)).astype(int)
        if 'MA20' in features:
            features['Trend_strength'] = abs(prev_close / (features['MA20'] + 1e-10) - 1)

    # === Z-SCORE FEATURES (Properly calculated) ===
    if 'MA20' in features:
        rolling_std = prev_close.rolling(20, min_periods=20).std()
        features['Zscore_Close_MA20'] = (prev_close - features['MA20']) / (rolling_std + 1e-10)

    # MA slope (trend direction)
    if 'MA20' in features:
        features['MA20_Slope'] = features['MA20'].diff(5) / 5

    # === SECTOR AND SPY RELATIVE STRENGTH (If available) ===
    if 'Sector_Close' in df.columns:
        sector_prev = df['Sector_Close'].shift(1)
        features['TSLA_to_Sector'] = prev_close / (sector_prev + 1e-10)
        features['Sector_Return'] = df['Sector_Close'].pct_change()
        features['Sector_MA50'] = sector_prev.rolling(50, min_periods=50).mean()

        # Relative momentum
        sector_momentum_5 = (sector_prev / sector_prev.shift(5)) - 1
        if 'Momentum_5' in features:
            features['Relative_Momentum_Sector'] = features['Momentum_5'] - sector_momentum_5

    if 'SPY_Close' in df.columns:
        spy_prev = df['SPY_Close'].shift(1)
        features['TSLA_to_SPY'] = prev_close / (spy_prev + 1e-10)
        features['SPY_Return'] = df['SPY_Close'].pct_change()
        features['SPY_MA50'] = spy_prev.rolling(50, min_periods=50).mean()

        # Relative momentum
        spy_momentum_5 = (spy_prev / spy_prev.shift(5)) - 1
        if 'Momentum_5' in features:
            features['Relative_Momentum_SPY'] = features['Momentum_5'] - spy_momentum_5

    # === ADX (Average Directional Index) ===
    # Calculate using lagged data to prevent lookahead
    up_move = prev_high.diff()
    down_move = -prev_low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    if 'TR' in features:
        atr_14 = features['TR'].rolling(14, min_periods=14).mean()
        plus_di = 100 * plus_dm.rolling(14, min_periods=14).sum() / (atr_14 + 1e-10)
        minus_di = 100 * minus_dm.rolling(14, min_periods=14).sum() / (atr_14 + 1e-10)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        features['ADX'] = dx.rolling(14, min_periods=14).mean()

    # === DONCHIAN CHANNELS ===
    for period in [20, 50]:
        features[f'Donchian_High_{period}'] = prev_high.rolling(period, min_periods=period).max()
        features[f'Donchian_Low_{period}'] = prev_low.rolling(period, min_periods=period).min()
        features[f'Donchian_Mid_{period}'] = (features[f'Donchian_High_{period}'] + features[f'Donchian_Low_{period}']) / 2

    # === PARABOLIC SAR (Simple implementation) ===
    sar_values = pd.Series(np.full(len(df), np.nan), index=df.index)
    if len(df) >= 3:
        af = 0.02
        max_af = 0.2
        trend = 1
        ep = prev_low.iloc[0] if not pd.isna(prev_low.iloc[0]) else df['Low'].iloc[0]
        sar = prev_high.iloc[0] if not pd.isna(prev_high.iloc[0]) else df['High'].iloc[0]

        for i in range(2, len(df)):
            prev_sar = sar
            curr_high = prev_high.iloc[i] if not pd.isna(prev_high.iloc[i]) else df['High'].iloc[i-1]
            curr_low = prev_low.iloc[i] if not pd.isna(prev_low.iloc[i]) else df['Low'].iloc[i-1]

            if trend == 1:
                sar = prev_sar + af * (ep - prev_sar)
                if curr_low < sar:
                    trend = -1
                    sar = ep
                    ep = curr_high
                    af = 0.02
                else:
                    if curr_high > ep:
                        ep = curr_high
                        af = min(af + 0.02, max_af)
            else:
                sar = prev_sar + af * (ep - prev_sar)
                if curr_high > sar:
                    trend = 1
                    sar = ep
                    ep = curr_low
                    af = 0.02
                else:
                    if curr_low < ep:
                        ep = curr_low
                        af = min(af + 0.02, max_af)
            sar_values.iloc[i] = sar

    features['SAR'] = sar_values.shift(1)  # Lag SAR by 1 to prevent lookahead

    # === ACCUMULATION/DISTRIBUTION LINE ===
    clv = ((prev_close - prev_low) - (prev_high - prev_close)) / (prev_high - prev_low + 1e-10)
    features['ADL'] = (clv * prev_volume).cumsum()

    # === CROSSOVER SIGNALS (Safe - comparing lagged values) ===
    if 'MA9' in features and 'MA21' in features:
        ma9_cross_up = ((features['MA9'] > features['MA21']) &
                       (features['MA9'].shift(1) <= features['MA21'].shift(1))).astype(int)
        ma9_cross_down = ((features['MA9'] < features['MA21']) &
                         (features['MA9'].shift(1) >= features['MA21'].shift(1))).astype(int)
        features['MA9_21_crossover'] = ma9_cross_up - ma9_cross_down

    # === STATISTICAL FEATURES ===

    # Price percentile ranks
    for period in [10, 20, 50]:
        features[f'Price_Percentile_{period}d'] = prev_close.rolling(period, min_periods=period).rank(pct=True)
        features[f'Volume_Percentile_{period}d'] = prev_volume.rolling(period, min_periods=period).rank(pct=True)

    # Price velocity (rate of change acceleration)
    if 'Returns_1d' in features:
        features['Price_Velocity'] = features['Returns_1d'].diff()
        features['Price_Acceleration'] = features['Price_Velocity'].diff()

    # === REGIME CHANGE DETECTION ===

    # Volatility regime changes
    if 'Volatility_20d' in features:
        vol_ma = features['Volatility_20d'].rolling(10, min_periods=10).mean()
        features['Vol_Regime_Change'] = (features['Volatility_20d'] > vol_ma * 1.5).astype(int)

    # Trend regime changes
    if 'MA20_Slope' in features:
        trend_change = abs(features['MA20_Slope'].diff()) > features['MA20_Slope'].rolling(20).std()
        features['Trend_Regime_Change'] = trend_change.astype(int)

    # === MARKET MICROSTRUCTURE FEATURES ===

    # Price efficiency (how much price moves relative to volume)
    if 'Returns_1d' in features:
        features['Price_Efficiency'] = abs(features['Returns_1d']) / (np.log(prev_volume + 1) + 1e-10)

    # Volume-weighted features
    if 'Volume_MA_20' in features:
        vwap_num = (prev_close * prev_volume).rolling(20, min_periods=20).sum()
        vwap_den = prev_volume.rolling(20, min_periods=20).sum()
        features['VWAP_20'] = vwap_num / (vwap_den + 1e-10)
        features['Price_to_VWAP'] = prev_close / (features['VWAP_20'] + 1e-10) - 1

    # === INTERACTION FEATURES ===

    # Volume-price interaction
    if 'Returns_1d' in features and 'Volume_ratio' in features:
        features['Volume_Price_Interaction'] = features['Returns_1d'] * features['Volume_ratio']

    # Volatility-momentum interaction
    if 'Volatility_10d' in features and 'Momentum_5' in features:
        features['Vol_Momentum_Interaction'] = features['Volatility_10d'] * abs(features['Momentum_5'])

    # RSI-price interaction
    if 'RSI' in features:
        features['RSI_Price_Interaction'] = features['RSI'] * (prev_close / features.get('MA20', prev_close))

    # === FEATURE QUALITY CONTROLS ===

    # Replace infinite values
    for key in features:
        if isinstance(features[key], pd.Series):
            features[key] = features[key].replace([np.inf, -np.inf], np.nan)

    # Combine all features with original DataFrame
    new_features_df = pd.DataFrame(features, index=df.index)
    result_df = pd.concat([df, new_features_df], axis=1)

    print(f"Feature engineering complete. Added {len(features)} new features.")

    return result_df

def select_features(df):
    """Enhanced feature selection focusing on non-leaky, predictive features"""

    if df is None or len(df) == 0:
        return pd.DataFrame()

    try:
        os.makedirs("debug", exist_ok=True)
    except:
        pass

    # Define feature categories with enhanced patterns
    feature_patterns = {
        'basic_technical': [
            # Moving averages and ratios
            'MA5', 'MA9', 'MA10', 'MA20', 'MA21', 'MA50', 'MA200',
            'EMA12', 'EMA26', 'EMA50', 'EMA200',
            'EMA12_26_diff', 'EMA12_50_diff', 'EMA50_200_diff',
            'MA9_MA21_Ratio', 'MA20_Slope'
        ],
        'price_action': [
            # Price relationships and patterns
            col for col in df.columns if any(x in col for x in [
                'Price_to_MA', 'Close_to_', 'Body_size', 'shadow',
                'Higher_high', 'Lower_low', 'Gap_'
            ])
        ],
        'momentum': [
            # Returns and momentum
            col for col in df.columns if any(x in col for x in [
                'Returns_', 'Momentum_', 'Price_Velocity', 'Price_Acceleration'
            ])
        ],
        'volatility': [
            # Volatility measures
            col for col in df.columns if any(x in col for x in [
                'Volatility_', 'ATR', 'BB_', 'Vol_'
            ])
        ],
        'oscillators': [
            # Technical oscillators
            'RSI', 'RSI_oversold', 'RSI_overbought', 'RSI_Momentum',
            'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI'
        ],
        'volume': [
            # Volume-based features
            col for col in df.columns if any(x in col for x in [
                'Volume_', 'OBV', 'CMF', 'ADL', 'VWAP'
            ])
        ],
        'advanced_patterns': [
            # Advanced pattern recognition features
            col for col in df.columns if any(x in col for x in [
                'Confluence', 'Momentum_Alignment', 'Divergence', 'Regime',
                'Pattern', 'Breakout', 'Support_Test', 'Resistance_Test',
                'Unusual_', 'Bullish_Volume', 'Bearish_Volume'
            ])
        ],
        'statistical': [
            # Statistical features
            col for col in df.columns if any(x in col for x in [
                'Percentile', 'Rank', 'Zscore', 'Efficiency'
            ])
        ],
        'support_resistance': [
            # Support/resistance levels
            col for col in df.columns if any(x in col for x in [
                'High_', 'Low_', 'Near_', 'Distance_From', 'Donchian'
            ])
        ],
        'crossovers': [
            # Crossover signals
            col for col in df.columns if 'crossover' in col
        ],
        'macd': [
            # MACD family
            'MACD', 'Signal_Line', 'MACD_histogram', 'MACD_bullish'
        ],
        'supertrend': [
            'Supertrend_Direction'
        ],
        'time_features': [
            'Day_of_week', 'Month', 'Is_month_end', 'Is_friday', 'Is_monday', 'Is_quarter_end'
        ],
        'relative_strength': [
            # Relative to market/sector
            col for col in df.columns if any(x in col for x in [
                'Sector_', 'SPY_', 'Relative_Momentum'
            ])
        ],
        'interactions': [
            # Feature interactions
            col for col in df.columns if 'Interaction' in col
        ],
        'microstructure': [
            # Market microstructure
            col for col in df.columns if any(x in col for x in [
                'Intraday_Strength', 'Price_Efficiency', 'Gap_Fill'
            ])
        ]
    }

    # Collect all features
    all_feature_cols = []
    for category, patterns in feature_patterns.items():
        if isinstance(patterns, list) and len(patterns) > 0:
            if isinstance(patterns[0], str) and not patterns[0].startswith('col for col'):
                # Direct feature names
                all_feature_cols.extend(patterns)
            else:
                # Pattern-based features (already collected)
                all_feature_cols.extend(patterns)

    # Remove duplicates and filter by availability
    available_features = []
    seen = set()

    for col in all_feature_cols:
        if col not in seen and col in df.columns:
            # Check if feature has enough non-null values
            non_null_ratio = df[col].notna().sum() / len(df)
            if non_null_ratio > 0.3:  # At least 30% non-null
                available_features.append(col)
                seen.add(col)

    # Add any lag features that weren't caught
    lag_features = [col for col in df.columns if col.startswith('lag') and col not in seen]
    for col in lag_features:
        if df[col].notna().sum() / len(df) > 0.3:
            available_features.append(col)

    print(f"Selected {len(available_features)} features from {len(df.columns)} total columns")

    # Create feature DataFrame
    if available_features:
        try:
            feature_df = df[available_features].copy()
            # Handle any remaining NaN values
            feature_df = feature_df.fillna(method='ffill').fillna(0) if hasattr(feature_df, 'fillna') else feature_df.fillna(0)

            # Save feature list for debugging
            try:
                with open("debug/selected_features.txt", "w") as f:
                    f.write("# Selected features by category\n")
                    f.write(f"# Total: {len(available_features)} features\n\n")

                    for category, patterns in feature_patterns.items():
                        category_features = [f for f in available_features
                                           if any(p in f for p in patterns if isinstance(p, str))]
                        if category_features:
                            f.write(f"## {category.upper()} ({len(category_features)} features)\n")
                            for feat in sorted(category_features):
                                f.write(f"{feat}\n")
                            f.write("\n")
            except Exception as e:
                print(f"Warning: Could not save feature list: {e}")

            return feature_df
        except Exception as e:
            print(f"Error creating feature DataFrame: {e}")
            return pd.DataFrame()
    else:
        print("Warning: No features selected")
        return pd.DataFrame()

def create_targets(df):
    """Create prediction targets with strict future-looking approach"""

    if df is None or len(df) == 0:
        print("WARNING: Empty dataframe passed to create_targets")
        return pd.DataFrame()

    if 'Close' not in df.columns or 'Open' not in df.columns:
        print("ERROR: Required columns (Close, Open) not found")
        return df

    df = df.copy()

    # FIXED: Proper future-looking targets
    # Target: Will tomorrow's close be higher than tomorrow's open?
    df['Target_1d'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)

    # Target: Will close in 2 days be higher than open in 2 days?
    df['Target_2d'] = (df['Close'].shift(-2) > df['Open'].shift(-2)).astype(int)

    # Strong move target: Will tomorrow's close be >1% higher than tomorrow's open?
    df['Target_strong'] = (df['Close'].shift(-1) > df['Open'].shift(-1) * 1.01).astype(int)

    # Additional targets for different prediction horizons
    # Next day direction (close-to-close)
    df['Target_direction_1d'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Multi-day targets
    df['Target_direction_3d'] = (df['Close'].shift(-3) > df['Close']).astype(int)
    df['Target_direction_5d'] = (df['Close'].shift(-5) > df['Close']).astype(int)

    # Volatility-adjusted targets
    returns_std = df['Close'].pct_change().rolling(20).std()
    threshold = returns_std * 0.5  # Half of recent volatility as threshold

    df['Target_significant_up'] = (df['Close'].shift(-1) > df['Close'] * (1 + threshold)).astype(int)
    df['Target_significant_down'] = (df['Close'].shift(-1) < df['Close'] * (1 - threshold)).astype(int)

    print(f"Created targets. Target distribution (Target_1d): {df['Target_1d'].value_counts().to_dict()}")

    return df
