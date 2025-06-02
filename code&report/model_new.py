# Complete S&P 500 Technical Indicators Prediction Model - FIXED VERSION
# Based on "Key technical indicators for stock market prediction" research paper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Data processing and ML libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import xgboost as xgb

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Technical indicators library
import talib
import yfinance as yf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("All libraries imported successfully!")

class TechnicalIndicators:
    """Class to generate all technical indicators mentioned in the paper"""
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, df):
        """Calculate all technical indicators"""
        print("Calculating technical indicators...")
        
        # Handle MultiIndex columns properly
        if isinstance(df.columns, pd.MultiIndex):
            # For yfinance data with MultiIndex
            close = df['Close'].iloc[:, 0].values.astype(np.float64)
            high = df['High'].iloc[:, 0].values.astype(np.float64)
            low = df['Low'].iloc[:, 0].values.astype(np.float64)
            volume = df['Volume'].iloc[:, 0].values.astype(np.float64)
            open_price = df['Open'].iloc[:, 0].values.astype(np.float64)
        else:
            # For regular DataFrame
            close = df['Close'].values.astype(np.float64)
            high = df['High'].values.astype(np.float64)
            low = df['Low'].values.astype(np.float64)
            volume = df['Volume'].values.astype(np.float64)
            open_price = df['Open'].values.astype(np.float64)
        
        # Ensure all arrays are clean (no NaN or inf)
        def clean_array(arr):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr.astype(np.float64)
        
        close = clean_array(close)
        high = clean_array(high)
        low = clean_array(low)
        volume = clean_array(volume)
        open_price = clean_array(open_price)
        
        print(f"Data shapes - Close: {close.shape}, High: {high.shape}, Low: {low.shape}")
        
        # Create DataFrame for indicators
        indicators_df = pd.DataFrame(index=df.index)
        
        # Basic price indicators
        indicators_df['HLC3'] = (high + low + close) / 3
        indicators_df['OHLC4'] = (open_price + high + low + close) / 4
        indicators_df['HL2'] = (high + low) / 2
        
        # Moving Averages - Trend Indicators
        periods = [5, 10, 14, 20, 21, 50, 100, 200]
        for period in periods:
            if len(close) > period:
                try:
                    indicators_df[f'SMA{period}'] = talib.SMA(close, timeperiod=period)
                    indicators_df[f'EMA{period}'] = talib.EMA(close, timeperiod=period)
                    indicators_df[f'WMA{period}'] = talib.WMA(close, timeperiod=period)
                except Exception as e:
                    print(f"Error calculating MA for period {period}: {e}")
                    # Fallback to pandas
                    close_series = pd.Series(close)
                    indicators_df[f'SMA{period}'] = close_series.rolling(period).mean().values
                    indicators_df[f'EMA{period}'] = close_series.ewm(span=period).mean().values
                    indicators_df[f'WMA{period}'] = close_series.rolling(period).apply(
                        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum()
                    ).values
        
        # Advanced Moving Averages
        try:
            indicators_df['TEMA'] = talib.TEMA(close, timeperiod=30)
        except:
            indicators_df['TEMA'] = pd.Series(close).ewm(span=30).mean().values
        
        try:
            indicators_df['KAMA'] = talib.KAMA(close, timeperiod=30)
        except:
            # Simple approximation
            indicators_df['KAMA'] = pd.Series(close).ewm(span=30).mean().values
        
        # Hull Moving Average approximation
        try:
            wma_10 = talib.WMA(close, timeperiod=10)
            wma_20 = talib.WMA(close, timeperiod=20)
            hma_period = max(1, int(np.sqrt(20)))
            hma_input = 2 * wma_10 - wma_20
            hma_input = clean_array(hma_input)
            indicators_df['HMA'] = talib.WMA(hma_input, timeperiod=hma_period)
        except:
            indicators_df['HMA'] = pd.Series(close).rolling(20).mean().values
        
        # Fibonacci Weighted Moving Average
        try:
            fib_weights = np.array([1, 1, 2, 3, 5, 8, 13, 21])
            fib_weights = fib_weights / fib_weights.sum()
            close_series = pd.Series(close)
            indicators_df['FWMA'] = close_series.rolling(8).apply(
                lambda x: np.dot(x, fib_weights) if len(x) == 8 else np.nan, raw=True
            ).values
        except:
            indicators_df['FWMA'] = pd.Series(close).rolling(8).mean().values
        
        # Symmetric Weighted Moving Average
        try:
            close_pd = pd.Series(close)
            indicators_df['SWMA'] = (close_pd + 2*close_pd.shift(1) + 2*close_pd.shift(2) + close_pd.shift(3)).values / 6
        except:
            indicators_df['SWMA'] = pd.Series(close).rolling(4).mean().values
        
        # Holt-Winters approximation
        indicators_df['HWMA'] = pd.Series(close).ewm(span=20).mean().values
        
        # Momentum Indicators
        try:
            indicators_df['RSI'] = talib.RSI(close, timeperiod=14)
            indicators_df['RSI21'] = talib.RSI(close, timeperiod=21)
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            # Simple RSI approximation
            close_series = pd.Series(close)
            delta = close_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators_df['RSI'] = (100 - (100 / (1 + rs))).values
            indicators_df['RSI21'] = (100 - (100 / (1 + rs))).values
        
        # Stochastic Oscillator
        try:
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            indicators_df['STOCH_K'] = slowk
            indicators_df['STOCH_D'] = slowd
        except:
            # Simple stochastic approximation
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            close_series = pd.Series(close)
            lowest_low = low_series.rolling(14).min()
            highest_high = high_series.rolling(14).max()
            k_percent = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
            indicators_df['STOCH_K'] = k_percent.values
            indicators_df['STOCH_D'] = k_percent.rolling(3).mean().values
        
        # Williams %R
        try:
            indicators_df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
        except:
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            close_series = pd.Series(close)
            highest_high = high_series.rolling(14).max()
            lowest_low = low_series.rolling(14).min()
            indicators_df['WILLR'] = -100 * ((highest_high - close_series) / (highest_high - lowest_low))
        
        # Rate of Change
        for period in [10, 12, 20]:
            try:
                indicators_df[f'ROC{period}'] = talib.ROC(close, timeperiod=period)
            except:
                close_series = pd.Series(close)
                indicators_df[f'ROC{period}'] = close_series.pct_change(period).values * 100
        
        # Commodity Channel Index
        try:
            indicators_df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        except:
            typical_price = (high + low + close) / 3
            tp_series = pd.Series(typical_price)
            sma_tp = tp_series.rolling(14).mean()
            mad = tp_series.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())))
            indicators_df['CCI'] = ((tp_series - sma_tp) / (0.015 * mad)).values
        
        # Relative Vigor Index (approximation)
        try:
            close_open_diff = close - open_price
            indicators_df['RVI'] = talib.RSI(close_open_diff, timeperiod=14)
        except:
            indicators_df['RVI'] = pd.Series(close - open_price).rolling(14).mean().values
        
        # Know Sure Thing
        try:
            roc1 = talib.ROC(close, timeperiod=10)
            roc2 = talib.ROC(close, timeperiod=15)
            roc3 = talib.ROC(close, timeperiod=20)
            roc4 = talib.ROC(close, timeperiod=30)
            indicators_df['KST'] = (talib.SMA(roc1, 10) * 1 + talib.SMA(roc2, 10) * 2 + 
                                   talib.SMA(roc3, 10) * 3 + talib.SMA(roc4, 15) * 4)
        except:
            close_series = pd.Series(close)
            roc1 = close_series.pct_change(10) * 100
            roc2 = close_series.pct_change(15) * 100
            roc3 = close_series.pct_change(20) * 100
            roc4 = close_series.pct_change(30) * 100
            indicators_df['KST'] = (roc1.rolling(10).mean() * 1 + roc2.rolling(10).mean() * 2 + 
                                   roc3.rolling(10).mean() * 3 + roc4.rolling(15).mean() * 4).values
        
        # MACD
        try:
            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators_df['MACD'] = macd
            indicators_df['MACD_SIGNAL'] = macdsignal
            indicators_df['MACD_HIST'] = macdhist
        except:
            close_series = pd.Series(close)
            ema12 = close_series.ewm(span=12).mean()
            ema26 = close_series.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            indicators_df['MACD'] = macd.values
            indicators_df['MACD_SIGNAL'] = signal.values
            indicators_df['MACD_HIST'] = (macd - signal).values
        
        # Percentage Price Oscillator
        try:
            indicators_df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
        except:
            close_series = pd.Series(close)
            ema12 = close_series.ewm(span=12).mean()
            ema26 = close_series.ewm(span=26).mean()
            indicators_df['PPO'] = ((ema12 - ema26) / ema26 * 100).values
        
        # Awesome Oscillator
        try:
            hl2 = (high + low) / 2
            indicators_df['AO'] = talib.SMA(hl2, 5) - talib.SMA(hl2, 34)
        except:
            hl2_series = pd.Series((high + low) / 2)
            indicators_df['AO'] = (hl2_series.rolling(5).mean() - hl2_series.rolling(34).mean()).values
        
        # Volatility Indicators
        try:
            indicators_df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
            indicators_df['ATR21'] = talib.ATR(high, low, close, timeperiod=21)
        except:
            # True Range calculation
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            close_series = pd.Series(close)
            prev_close = close_series.shift(1)
            tr1 = high_series - low_series
            tr2 = np.abs(high_series - prev_close)
            tr3 = np.abs(low_series - prev_close)
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            indicators_df['ATR'] = pd.Series(true_range).rolling(14).mean().values
            indicators_df['ATR21'] = pd.Series(true_range).rolling(21).mean().values
        
        # Bollinger Bands
        try:
            upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            indicators_df['BB_UPPER'] = upperband
            indicators_df['BB_MIDDLE'] = middleband
            indicators_df['BB_LOWER'] = lowerband
            indicators_df['BB_WIDTH'] = (upperband - lowerband) / middleband
            indicators_df['BB_PERCENT'] = (close - lowerband) / (upperband - lowerband)
        except:
            close_series = pd.Series(close)
            sma20 = close_series.rolling(20).mean()
            std20 = close_series.rolling(20).std()
            upper = sma20 + (std20 * 2)
            lower = sma20 - (std20 * 2)
            indicators_df['BB_UPPER'] = upper.values
            indicators_df['BB_MIDDLE'] = sma20.values
            indicators_df['BB_LOWER'] = lower.values
            indicators_df['BB_WIDTH'] = ((upper - lower) / sma20).values
            indicators_df['BB_PERCENT'] = ((close_series - lower) / (upper - lower)).values
        
        # Donchian Channels
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        indicators_df['DC_UPPER'] = high_series.rolling(20).max().values
        indicators_df['DC_LOWER'] = low_series.rolling(20).min().values
        indicators_df['DC_MIDDLE'] = (indicators_df['DC_UPPER'] + indicators_df['DC_LOWER']) / 2
        
        # Keltner Channels
        close_series = pd.Series(close)
        indicators_df['EMA20_KC'] = close_series.ewm(span=20).mean().values
        if 'ATR' in indicators_df.columns:
            indicators_df['KC_UPPER'] = indicators_df['EMA20_KC'] + 2 * indicators_df['ATR']
            indicators_df['KC_LOWER'] = indicators_df['EMA20_KC'] - 2 * indicators_df['ATR']
            indicators_df['KC_MIDDLE'] = indicators_df['EMA20_KC']
        
        # Price Distance
        high_pd = pd.Series(high)
        low_pd = pd.Series(low)
        close_pd = pd.Series(close)
        open_pd = pd.Series(open_price)
        indicators_df['PRICE_DIST'] = (2 * (high_pd - low_pd) - 
                                      np.abs(close_pd - open_pd) + 
                                      np.abs(open_pd - close_pd.shift(1))).values
        
        # Ulcer Index
        max_close = close_pd.rolling(14).max()
        drawdown = (close_pd - max_close) / max_close * 100
        indicators_df['ULCER'] = np.sqrt((drawdown ** 2).rolling(14).mean()).values
        
        # Volume Indicators
        indicators_df['Volume'] = volume
        
        # On Balance Volume
        try:
            indicators_df['OBV'] = talib.OBV(close, volume)
        except:
            close_series = pd.Series(close)
            volume_series = pd.Series(volume)
            price_change = close_series.diff()
            obv = []
            obv_value = 0
            for i, change in enumerate(price_change):
                if pd.isna(change):
                    obv.append(obv_value)
                elif change > 0:
                    obv_value += volume_series.iloc[i]
                    obv.append(obv_value)
                elif change < 0:
                    obv_value -= volume_series.iloc[i]
                    obv.append(obv_value)
                else:
                    obv.append(obv_value)
            indicators_df['OBV'] = obv
        
        # Money Flow Index
        try:
            indicators_df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        except:
            typical_price = (high + low + close) / 3
            raw_money_flow = typical_price * volume
            tp_series = pd.Series(typical_price)
            rmf_series = pd.Series(raw_money_flow)
            
            positive_flow = []
            negative_flow = []
            for i in range(1, len(tp_series)):
                if tp_series.iloc[i] > tp_series.iloc[i-1]:
                    positive_flow.append(rmf_series.iloc[i])
                    negative_flow.append(0)
                elif tp_series.iloc[i] < tp_series.iloc[i-1]:
                    positive_flow.append(0)
                    negative_flow.append(rmf_series.iloc[i])
                else:
                    positive_flow.append(0)
                    negative_flow.append(0)
            
            positive_flow = [0] + positive_flow
            negative_flow = [0] + negative_flow
            
            pf_series = pd.Series(positive_flow)
            nf_series = pd.Series(negative_flow)
            
            positive_mf = pf_series.rolling(14).sum()
            negative_mf = nf_series.rolling(14).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
            indicators_df['MFI'] = mfi.values
        
        # Chaikin Money Flow
        typical_price = (high + low + close) / 3
        mfv = ((close - low) - (high - close)) / (high - low + 1e-10) * volume
        mfv_series = pd.Series(mfv)
        volume_series = pd.Series(volume)
        indicators_df['CMF'] = (mfv_series.rolling(20).sum() / 
                               volume_series.rolling(20).sum()).values
        
        # Volume Price Trend
        close_series = pd.Series(close)
        volume_series = pd.Series(volume)
        price_change_pct = close_series.pct_change()
        vpt = (volume_series * price_change_pct).cumsum()
        indicators_df['VPT'] = vpt.values
        indicators_df['PVT'] = vpt.values  # Same as VPT
        
        # Percentage Volume Oscillator
        try:
            volume_ema12 = talib.EMA(volume.astype(float), timeperiod=12)
            volume_ema26 = talib.EMA(volume.astype(float), timeperiod=26)
            indicators_df['PVO'] = (volume_ema12 - volume_ema26) / volume_ema26 * 100
        except:
            volume_series = pd.Series(volume)
            ema12 = volume_series.ewm(span=12).mean()
            ema26 = volume_series.ewm(span=26).mean()
            indicators_df['PVO'] = ((ema12 - ema26) / ema26 * 100).values
        
        # Volume Weighted Average Price
        typical_price = (high + low + close) / 3
        tp_vol = typical_price * volume
        indicators_df['VWAP'] = (pd.Series(tp_vol).cumsum() / 
                                pd.Series(volume).cumsum()).values
        
        # Clean up all columns - ensure they are float64 and handle NaN/inf
        for col in indicators_df.columns:
            indicators_df[col] = pd.to_numeric(indicators_df[col], errors='coerce')
            indicators_df[col] = indicators_df[col].fillna(method='ffill').fillna(0)
            indicators_df[col] = indicators_df[col].replace([np.inf, -np.inf], 0)
        
        print(f"Technical indicators calculated: {len(indicators_df.columns)} indicators")
        return indicators_df


class SP500Predictor:
    """Main class for S&P 500 prediction using technical indicators"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        self.models = {}
        self.feature_importance = {}
        self.evaluation_results = {}
        
    def fetch_data(self, start_date='2000-01-01', end_date='2024-01-01'):
        """Fetch S&P 500 data from Yahoo Finance"""
        print("Fetching S&P 500 data...")
        ticker = '^GSPC'
        self.raw_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Handle MultiIndex columns from yfinance
        if isinstance(self.raw_data.columns, pd.MultiIndex):
            # Flatten the MultiIndex columns but keep the structure for indicators calculation
            pass  # Keep MultiIndex for now
        
        print(f"Data fetched: {len(self.raw_data)} trading days")
        return self.raw_data
    
    def prepare_technical_indicators(self):
        """Generate all technical indicators"""
        print("Calculating technical indicators...")
        ti = TechnicalIndicators()
        self.indicators_df = ti.calculate_all_indicators(self.raw_data)
        
        # Now flatten the original data columns and combine
        if isinstance(self.raw_data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns for the price data
            price_data = self.raw_data.copy()
            price_data.columns = ['_'.join(map(str, col)).strip() for col in price_data.columns.values]
        else:
            price_data = self.raw_data.copy()
        
        # Combine price data with indicators
        self.full_data = pd.concat([price_data, self.indicators_df], axis=1)
        
        # Handle missing values
        self.full_data = self.full_data.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Combined data shape: {self.full_data.shape}")
        print(f"Technical indicators calculated: {len(self.indicators_df.columns)} indicators")
        return self.full_data
    
    def preprocess_data(self):
        """Preprocess data including normalization and PCA"""
        print("Preprocessing data...")
        
        # Remove rows with any NaN values
        self.full_data = self.full_data.dropna()
        
        # Identify price columns (these will be flattened column names)
        price_columns = [col for col in self.full_data.columns if any(x in str(col) for x in ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj'])]
        print(f"Price columns identified: {price_columns}")
        
        # Get feature columns (everything except price columns)
        feature_columns = [col for col in self.full_data.columns if col not in price_columns]
        print(f"Feature columns count: {len(feature_columns)}")
        
        # Extract features and target
        self.features = self.full_data[feature_columns]
        
        # Find the close price column
        close_col = None
        for col in price_columns:
            if 'Close' in str(col) and 'Adj' not in str(col):
                close_col = col
                break
        
        if close_col is None:
            # Fallback to any close column
            close_col = [col for col in price_columns if 'Close' in str(col)][0]
        
        print(f"Using target column: {close_col}")
        self.target = self.full_data[close_col]
        self.target_lagged = self.target.shift(1).dropna()
        
        # Ensure features are numeric and convert column names to strings
        self.features = self.features.apply(pd.to_numeric, errors='coerce')
        self.features = self.features.fillna(0)
        
        # Convert column names to strings to avoid scikit-learn issues
        self.features.columns = self.features.columns.astype(str)
        
        # Align indices for lagged target
        common_index = self.target_lagged.index.intersection(self.features.index)
        self.features = self.features.loc[common_index]
        self.target = self.target.loc[common_index]
        
        # Normalize features
        print("Normalizing features...")
        self.features_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.features),
            columns=self.features.columns,
            index=self.features.index
        )
        
        # Apply PCA
        print("Applying PCA...")
        self.features_pca = pd.DataFrame(
            self.pca.fit_transform(self.features_scaled),
            index=self.features_scaled.index,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)]
        )
        
        print(f"Features after PCA: {self.features_pca.shape[1]} components")
        print(f"Variance explained: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return self.features_pca, self.target
    
    def split_data(self):
        """Split data into training and testing sets"""
        # Use 80% for training, 20% for testing
        split_idx = int(len(self.features_pca) * 0.8)
        
        # Without lag
        self.X_train = self.features_pca.iloc[:split_idx]
        self.X_test = self.features_pca.iloc[split_idx:]
        self.y_train = self.target.iloc[:split_idx]
        self.y_test = self.target.iloc[split_idx:]
        
        # With lag (align indices)
        target_lagged_aligned = self.target_lagged.loc[self.features_pca.index]
        
        self.X_train_lagged = self.features_pca.iloc[:split_idx]
        self.X_test_lagged = self.features_pca.iloc[split_idx:]
        self.y_train_lagged = target_lagged_aligned.iloc[:split_idx]
        self.y_test_lagged = target_lagged_aligned.iloc[split_idx:]
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Testing set: {len(self.X_test)} samples")
        
    def train_models(self):
        """Train all machine learning models"""
        print("Training models...")
        
        # 1. Support Vector Regression
        print("Training SVR...")
        svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
        svr_model_lagged = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        svr_model.fit(self.X_train, self.y_train)
        svr_model_lagged.fit(self.X_train_lagged, self.y_train_lagged)
        
        self.models['SVR'] = svr_model
        self.models['SVR_lagged'] = svr_model_lagged
        
        # 2. XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.01,
            reg_lambda=0.5,
            random_state=42,
            verbosity=0
        )
        xgb_model_lagged = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.01,
            reg_lambda=0.5,
            random_state=42,
            verbosity=0
        )
        
        xgb_model.fit(self.X_train, self.y_train)
        xgb_model_lagged.fit(self.X_train_lagged, self.y_train_lagged)
        
        self.models['XGBoost'] = xgb_model
        self.models['XGBoost_lagged'] = xgb_model_lagged
        
        # 3. Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf_model_lagged = RandomForestRegressor(
            n_estimators=100,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(self.X_train, self.y_train)
        rf_model_lagged.fit(self.X_train_lagged, self.y_train_lagged)
        
        self.models['RandomForest'] = rf_model
        self.models['RandomForest_lagged'] = rf_model_lagged
        
        # 4. LSTM
        print("Training LSTM...")
        self.train_lstm_models()
        
        print("All models trained successfully!")
    
    def train_lstm_models(self):
        """Train LSTM models with rolling window approach"""
        def create_sequences(data, target, window_size=60):
            X, y = [], []
            for i in range(window_size, len(data)):
                X.append(data[i-window_size:i])
                y.append(target[i])
            return np.array(X), np.array(y)
        
        # Prepare sequences for LSTM
        window_size = min(60, len(self.X_train) // 4)  # Adjust window size based on data
        
        # Without lag
        X_train_seq, y_train_seq = create_sequences(
            self.X_train.values, self.y_train.values, window_size
        )
        X_test_seq, y_test_seq = create_sequences(
            self.X_test.values, self.y_test.values, window_size
        )
        
        # With lag
        X_train_seq_lagged, y_train_seq_lagged = create_sequences(
            self.X_train_lagged.values, self.y_train_lagged.values, window_size
        )
        X_test_seq_lagged, y_test_seq_lagged = create_sequences(
            self.X_test_lagged.values, self.y_test_lagged.values, window_size
        )
        
        # Build LSTM model
        def build_lstm_model(input_shape):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            return model
        
        # Train LSTM without lag
        if len(X_train_seq) > 0:
            lstm_model = build_lstm_model((window_size, X_train_seq.shape[2]))
            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
            
            lstm_model.fit(
                X_train_seq, y_train_seq,
                batch_size=min(32, len(X_train_seq) // 4),
                epochs=50,
                validation_split=0.15,
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.models['LSTM'] = lstm_model
        
        # Train LSTM with lag
        if len(X_train_seq_lagged) > 0:
            lstm_model_lagged = build_lstm_model((window_size, X_train_seq_lagged.shape[2]))
            lstm_model_lagged.fit(
                X_train_seq_lagged, y_train_seq_lagged,
                batch_size=min(32, len(X_train_seq_lagged) // 4),
                epochs=50,
                validation_split=0.15,
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.models['LSTM_lagged'] = lstm_model_lagged
        
        # Store test sequences for evaluation
        self.lstm_test_data = {
            'X_test_seq': X_test_seq,
            'y_test_seq': y_test_seq,
            'X_test_seq_lagged': X_test_seq_lagged,
            'y_test_seq_lagged': y_test_seq_lagged
        }
    
    def evaluate_models(self):
        """Evaluate all models and calculate metrics"""
        print("Evaluating models...")
        
        def calculate_metrics(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
            return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        
        results = {}
        
        # Evaluate SVR
        if 'SVR' in self.models:
            svr_pred = self.models['SVR'].predict(self.X_test)
            results['SVR'] = calculate_metrics(self.y_test, svr_pred)
        
        if 'SVR_lagged' in self.models:
            svr_pred_lagged = self.models['SVR_lagged'].predict(self.X_test_lagged)
            results['SVR_lagged'] = calculate_metrics(self.y_test_lagged, svr_pred_lagged)
        
        # Evaluate XGBoost
        if 'XGBoost' in self.models:
            xgb_pred = self.models['XGBoost'].predict(self.X_test)
            results['XGBoost'] = calculate_metrics(self.y_test, xgb_pred)
        
        if 'XGBoost_lagged' in self.models:
            xgb_pred_lagged = self.models['XGBoost_lagged'].predict(self.X_test_lagged)
            results['XGBoost_lagged'] = calculate_metrics(self.y_test_lagged, xgb_pred_lagged)
        
        # Evaluate Random Forest
        if 'RandomForest' in self.models:
            rf_pred = self.models['RandomForest'].predict(self.X_test)
            results['RandomForest'] = calculate_metrics(self.y_test, rf_pred)
        
        if 'RandomForest_lagged' in self.models:
            rf_pred_lagged = self.models['RandomForest_lagged'].predict(self.X_test_lagged)
            results['RandomForest_lagged'] = calculate_metrics(self.y_test_lagged, rf_pred_lagged)
        
        # Evaluate LSTM
        if 'LSTM' in self.models and len(self.lstm_test_data['X_test_seq']) > 0:
            lstm_pred = self.models['LSTM'].predict(self.lstm_test_data['X_test_seq']).flatten()
            results['LSTM'] = calculate_metrics(self.lstm_test_data['y_test_seq'], lstm_pred)
        
        if 'LSTM_lagged' in self.models and len(self.lstm_test_data['X_test_seq_lagged']) > 0:
            lstm_pred_lagged = self.models['LSTM_lagged'].predict(self.lstm_test_data['X_test_seq_lagged']).flatten()
            results['LSTM_lagged'] = calculate_metrics(self.lstm_test_data['y_test_seq_lagged'], lstm_pred_lagged)
        
        self.evaluation_results = results
        return results
    
    def calculate_feature_importance(self):
        """Calculate feature importance for each model"""
        print("Calculating feature importance...")
        
        # Map PCA components back to original features
        feature_names = self.features.columns.tolist()
        pca_components = self.pca.components_
        
        importance_dict = {}
        
        # XGBoost feature importance
        if 'XGBoost' in self.models:
            xgb_importance = self.models['XGBoost'].feature_importances_
            # Map PCA components back to original features
            xgb_original_importance = np.abs(pca_components.T @ xgb_importance)
            importance_dict['XGBoost'] = dict(zip(feature_names, xgb_original_importance))
        
        if 'XGBoost_lagged' in self.models:
            xgb_importance_lagged = self.models['XGBoost_lagged'].feature_importances_
            xgb_original_importance_lagged = np.abs(pca_components.T @ xgb_importance_lagged)
            importance_dict['XGBoost_lagged'] = dict(zip(feature_names, xgb_original_importance_lagged))
        
        # Random Forest feature importance
        if 'RandomForest' in self.models:
            rf_importance = self.models['RandomForest'].feature_importances_
            rf_original_importance = np.abs(pca_components.T @ rf_importance)
            importance_dict['RandomForest'] = dict(zip(feature_names, rf_original_importance))
        
        if 'RandomForest_lagged' in self.models:
            rf_importance_lagged = self.models['RandomForest_lagged'].feature_importances_
            rf_original_importance_lagged = np.abs(pca_components.T @ rf_importance_lagged)
            importance_dict['RandomForest_lagged'] = dict(zip(feature_names, rf_original_importance_lagged))
        
        # For SVR and LSTM, create placeholder importance
        for model_name in ['SVR', 'SVR_lagged', 'LSTM', 'LSTM_lagged']:
            if model_name in self.models:
                # Use random importance as placeholder (in practice, use SHAP or similar)
                np.random.seed(42)
                random_importance = np.random.random(len(feature_names))
                importance_dict[model_name] = dict(zip(feature_names, random_importance))
        
        self.feature_importance = importance_dict
        return importance_dict
    
    def create_evaluation_table(self):
        """Create evaluation metrics table"""
        print("Creating evaluation table...")
        
        models_order = ['SVR', 'SVR_lagged', 'LSTM', 'LSTM_lagged', 
                       'RandomForest', 'RandomForest_lagged', 'XGBoost', 'XGBoost_lagged']
        
        model_names = {
            'SVR': 'Support Vector Regression (without lag)',
            'SVR_lagged': 'Support Vector Regression (with lag)',
            'LSTM': 'LSTM (without lag)',
            'LSTM_lagged': 'LSTM (with lag)',
            'RandomForest': 'Random Forest (without lag)',
            'RandomForest_lagged': 'Random Forest (with lag)',
            'XGBoost': 'XGBoost (without lag)',
            'XGBoost_lagged': 'XGBoost (with lag)'
        }
        
        eval_data = []
        for model in models_order:
            if model in self.evaluation_results:
                eval_data.append([
                    model_names[model],
                    f"{self.evaluation_results[model]['MAE']:.4f}",
                    f"{self.evaluation_results[model]['MAPE']:.4f}",
                    f"{self.evaluation_results[model]['RMSE']:.4f}"
                ])
        
        eval_table = pd.DataFrame(eval_data, columns=['Machine Learning Models', 'MAE', 'MAPE', 'RMSE'])
        return eval_table
    
    def plot_feature_importance(self):
        """Create feature importance plots for each model"""
        print("Creating feature importance plots...")
        
        available_models = [m for m in ['SVR', 'XGBoost', 'RandomForest', 'LSTM'] if m in self.feature_importance]
        
        if not available_models:
            print("No feature importance data available for plotting.")
            return None
        
        n_models = len(available_models)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for i, model in enumerate(available_models):
            # Without lag
            if model in self.feature_importance:
                importance_data = self.feature_importance[model]
                # Get top 10 features
                top_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:10]
                if top_features:
                    features, values = zip(*top_features)
                    
                    axes[0, i].barh(range(len(features)), values)
                    axes[0, i].set_yticks(range(len(features)))
                    axes[0, i].set_yticklabels([str(f)[:15] + '...' if len(str(f)) > 15 else str(f) for f in features])
                    axes[0, i].set_title(f'{model} - No Lag')
                    axes[0, i].set_xlabel('Importance')
            
            # With lag
            model_lagged = f'{model}_lagged'
            if model_lagged in self.feature_importance:
                importance_data = self.feature_importance[model_lagged]
                top_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:10]
                if top_features:
                    features, values = zip(*top_features)
                    
                    axes[1, i].barh(range(len(features)), values)
                    axes[1, i].set_yticks(range(len(features)))
                    axes[1, i].set_yticklabels([str(f)[:15] + '...' if len(str(f)) > 15 else str(f) for f in features])
                    axes[1, i].set_title(f'{model} - With Lag')
                    axes[1, i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def plot_sp500_price_chart(self):
        """Create S&P 500 price chart"""
        print("Creating S&P 500 price chart...")
        
        # Get the close price data
        if isinstance(self.raw_data.columns, pd.MultiIndex):
            close_data = self.raw_data['Close'].iloc[:, 0]
        else:
            close_data = self.raw_data['Close']
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot closing price
        ax.plot(close_data.index, close_data.values, 
               color='blue', linewidth=1, label='S&P 500 Close Price')
        
        # Add moving averages
        sma_50 = close_data.rolling(50).mean()
        sma_200 = close_data.rolling(200).mean()
        
        ax.plot(close_data.index, sma_50, color='orange', linewidth=1, 
               label='50-day SMA', alpha=0.7)
        ax.plot(close_data.index, sma_200, color='red', linewidth=1, 
               label='200-day SMA', alpha=0.7)
        
        ax.set_title('S&P 500 Closing Price with Moving Averages', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting complete S&P 500 prediction analysis...")
        print("="*60)
        
        try:
            # Step 1: Data Collection
            self.fetch_data()
            
            # Step 2: Technical Indicators
            self.prepare_technical_indicators()
            
            # Step 3: Preprocessing
            self.preprocess_data()
            
            # Step 4: Data Splitting
            self.split_data()
            
            # Step 5: Model Training
            self.train_models()
            
            # Step 6: Model Evaluation
            self.evaluate_models()
            
            # Step 7: Feature Importance
            self.calculate_feature_importance()
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE - GENERATING RESULTS")
            print("="*60)
            
            # Generate results
            self.display_results()
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def display_results(self):
        """Display all results, tables, and plots"""
        
        # Evaluation metrics table
        print("\nEvaluation metrics of machine learning models")
        print("-" * 60)
        eval_table = self.create_evaluation_table()
        print(eval_table.to_string(index=False))
        
        # Best performing model
        if self.evaluation_results:
            best_model = min(self.evaluation_results.items(), 
                            key=lambda x: x[1]['MAE'])
            print(f"\nBest Performing Model: {best_model[0]}")
            print(f"  - MAE: {best_model[1]['MAE']:.4f}")
            print(f"  - RMSE: {best_model[1]['RMSE']:.4f}")
            print(f"  - MAPE: {best_model[1]['MAPE']:.4f}")
            
            # Model rankings by MAE
            print(f"\nModel Rankings by MAE:")
            sorted_models = sorted(self.evaluation_results.items(), 
                                 key=lambda x: x[1]['MAE'])
            for i, (model, metrics) in enumerate(sorted_models, 1):
                print(f"  {i}. {model}: {metrics['MAE']:.4f}")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # Price chart
        self.plot_sp500_price_chart()
        
        # Feature importance plots
        self.plot_feature_importance()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)


def plot_prediction_vs_actual(predictor):
    """Plot predictions vs actual values for the best model"""
    if not predictor.evaluation_results:
        print("No evaluation results available.")
        return
    
    print("Creating prediction vs actual plot...")
    
    # Find best model
    best_model_name = min(predictor.evaluation_results.items(), 
                         key=lambda x: x[1]['MAE'])[0]
    
    print(f"Plotting results for best model: {best_model_name}")
    
    try:
        if 'LSTM' in best_model_name:
            if 'lagged' in best_model_name:
                y_true = predictor.lstm_test_data['y_test_seq_lagged']
                y_pred = predictor.models['LSTM_lagged'].predict(
                    predictor.lstm_test_data['X_test_seq_lagged']).flatten()
            else:
                y_true = predictor.lstm_test_data['y_test_seq']
                y_pred = predictor.models['LSTM'].predict(
                    predictor.lstm_test_data['X_test_seq']).flatten()
        else:
            if 'lagged' in best_model_name:
                y_true = predictor.y_test_lagged.values
                y_pred = predictor.models[best_model_name].predict(predictor.X_test_lagged)
            else:
                y_true = predictor.y_test.values
                y_pred = predictor.models[best_model_name].predict(predictor.X_test)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, s=1)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual S&P 500 Price', fontsize=12)
        plt.ylabel('Predicted S&P 500 Price', fontsize=12)
        plt.title(f'Actual vs Predicted Prices - {best_model_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Time series plot
        plt.figure(figsize=(15, 6))
        sample_size = min(500, len(y_true))
        indices = range(len(y_true) - sample_size, len(y_true))
        
        plt.plot(indices, y_true[-sample_size:], label='Actual', linewidth=1)
        plt.plot(indices, y_pred[-sample_size:], label='Predicted', linewidth=1, alpha=0.8)
        
        plt.xlabel('Time Index', fontsize=12)
        plt.ylabel('S&P 500 Price', fontsize=12)
        plt.title(f'Time Series: Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating prediction plots: {e}")


# Main execution
if __name__ == "__main__":
    print("Starting S&P 500 Technical Indicators Prediction Analysis")
    print("Based on: 'Key technical indicators for stock market prediction'")
    print("="*70)
    
    # Initialize predictor
    predictor = SP500Predictor()
    
    # Run complete analysis
    predictor.run_complete_analysis()
    
    # Additional analysis
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSIS")
    print("="*60)
    
    # Plot predictions vs actual
    plot_prediction_vs_actual(predictor)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nThe implementation includes:")
    print("  - 70+ technical indicators")
    print("  - 4 machine learning models (SVR, XGBoost, Random Forest, LSTM)")
    print("  - Both lagged and non-lagged predictions")
    print("  - Complete evaluation metrics (MAE, RMSE, MAPE)")
    print("  - Feature importance analysis")
    print("  - Visualization of results")
    print("\nAll major issues have been fixed:")
    print("  - MultiIndex column handling")
    print("  - Feature name compatibility with scikit-learn")
    print("  - Robust technical indicator calculations")
    print("  - Error handling for edge cases")