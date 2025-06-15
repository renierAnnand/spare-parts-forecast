import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Time series libraries
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Deep learning (optional - install with pip install tensorflow)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - LSTM models will be skipped")

class ImprovedSalesPredictionSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.ensemble_weights = {}
        self.validation_scores = {}
        
    def load_and_prepare_data(self, sales_2024_file, sales_2022_2023_file):
        """Load and prepare the sales data with proper formatting for Streamlit uploads"""
        
        try:
            # Load 2024 data - handle both file paths and uploaded files
            if hasattr(sales_2024_file, 'read'):  # Streamlit uploaded file
                df_2024 = pd.read_excel(sales_2024_file)
            else:  # File path
                df_2024 = pd.read_excel(sales_2024_file)
            
            print("=== 2024 FILE ANALYSIS ===")
            print("Shape:", df_2024.shape)
            print("Original columns:", df_2024.columns.tolist())
            print("First few rows:")
            print(df_2024.head(3))
            
            # Clean 2024 data - handle header row detection
            first_cell = str(df_2024.iloc[0, 0]).strip() if len(df_2024) > 0 else ""
            
            if 'item code' in first_cell.lower() or 'part' in first_cell.lower():
                # First row contains headers, skip it
                df_2024_data = df_2024.iloc[1:].copy()
                print("Detected header row, skipping first row")
            else:
                df_2024_data = df_2024.copy()
                print("No header row detected")
            
            # Reset index and handle column naming
            df_2024_data = df_2024_data.reset_index(drop=True)
            
            # Identify structure based on content
            ncols = df_2024_data.shape[1]
            print(f"Data has {ncols} columns")
            
            # Standard structure: part_code, description, brand, engine, then 12 months
            if ncols >= 16:  # At least 4 metadata + 12 months
                new_cols = ['part_code', 'description', 'brand', 'engine'] + [f'month_{i:02d}' for i in range(1, 13)]
                df_2024_data.columns = new_cols[:ncols]
            elif ncols >= 13:  # At least 1 metadata + 12 months
                new_cols = ['part_code'] + [f'month_{i:02d}' for i in range(1, ncols)]
                df_2024_data.columns = new_cols
            else:
                # Fallback: use original columns
                df_2024_data.columns = [f'col_{i}' for i in range(ncols)]
                df_2024_data = df_2024_data.rename(columns={df_2024_data.columns[0]: 'part_code'})
            
            print("New column names:", df_2024_data.columns.tolist())
            
            # Remove rows with missing part codes
            df_2024_data = df_2024_data.dropna(subset=['part_code'])
            df_2024_data = df_2024_data[df_2024_data['part_code'].astype(str).str.strip() != '']
            
            # Identify month columns
            month_cols = [col for col in df_2024_data.columns if col.startswith('month_')]
            if not month_cols:
                # Try to identify by content or position
                possible_month_cols = df_2024_data.columns[1:13] if ncols >= 13 else df_2024_data.columns[1:]
                month_cols = [col for col in possible_month_cols if col not in ['description', 'brand', 'engine']]
            
            print("Identified month columns:", month_cols)
            
            # Prepare metadata columns
            metadata_cols = ['description', 'brand', 'engine']
            available_metadata = [col for col in metadata_cols if col in df_2024_data.columns]
            
            # Melt to long format
            melt_id_vars = ['part_code'] + available_metadata
            df_2024_long = df_2024_data.melt(
                id_vars=melt_id_vars,
                value_vars=month_cols,
                var_name='month_col',
                value_name='sales'
            )
            
            # Convert month column to actual dates
            def month_col_to_date(month_col):
                try:
                    if 'month_' in month_col:
                        month_num = int(month_col.split('_')[1])
                        return pd.to_datetime(f'2024-{month_num:02d}-01')
                    else:
                        # Try to extract month from column name
                        month_mapping = {
                            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                        }
                        col_lower = str(month_col).lower()
                        for month_name, month_num in month_mapping.items():
                            if month_name in col_lower:
                                return pd.to_datetime(f'2024-{month_num:02d}-01')
                        return pd.to_datetime('2024-01-01')  # Fallback
                except:
                    return pd.to_datetime('2024-01-01')
            
            df_2024_long['date'] = df_2024_long['month_col'].apply(month_col_to_date)
            df_2024_long['sales'] = pd.to_numeric(df_2024_long['sales'], errors='coerce')
            
            print(f"2024 data after processing: {len(df_2024_long)} rows")
            
            # Load 2022-2023 historical data
            if hasattr(sales_2022_2023_file, 'read'):  # Streamlit uploaded file
                df_hist = pd.read_excel(sales_2022_2023_file)
            else:  # File path
                df_hist = pd.read_excel(sales_2022_2023_file)
            
            print("\n=== HISTORICAL FILE ANALYSIS ===")
            print("Shape:", df_hist.shape)
            print("Columns:", df_hist.columns.tolist())
            print("First few rows:")
            print(df_hist.head(3))
            
            # Smart column detection for historical data
            col_names = df_hist.columns.tolist()
            
            # Find part column
            part_col = None
            for col in col_names:
                if 'part' in str(col).lower():
                    part_col = col
                    break
            if part_col is None:
                part_col = col_names[0]  # Use first column as fallback
            
            # Find sales column
            sales_col = None
            for col in col_names:
                if 'sales' in str(col).lower() or 'value' in str(col).lower():
                    sales_col = col
                    break
            if sales_col is None:
                sales_col = col_names[-1]  # Use last column as fallback
            
            # Find month/date column
            date_col = None
            for col in col_names:
                if any(word in str(col).lower() for word in ['month', 'date', 'time']):
                    date_col = col
                    break
            if date_col is None:
                date_col = col_names[1]  # Use second column as fallback
            
            print(f"Detected columns - Part: {part_col}, Date: {date_col}, Sales: {sales_col}")
            
            # Create clean historical dataset
            df_hist_clean = df_hist[[part_col, date_col, sales_col]].copy()
            df_hist_clean.columns = ['part_code', 'date_raw', 'sales']
            
            # Convert dates
            def safe_date_convert(date_val):
                try:
                    if pd.isna(date_val):
                        return pd.NaT
                    
                    # Handle Excel serial dates
                    if isinstance(date_val, (int, float)):
                        if date_val > 40000:  # Reasonable Excel date range
                            return pd.to_datetime('1900-01-01') + pd.to_timedelta(date_val - 2, unit='D')
                    
                    # Try direct conversion
                    return pd.to_datetime(date_val)
                    
                except:
                    return pd.NaT
            
            df_hist_clean['date'] = df_hist_clean['date_raw'].apply(safe_date_convert)
            df_hist_clean = df_hist_clean.dropna(subset=['date'])
            df_hist_clean['sales'] = pd.to_numeric(df_hist_clean['sales'], errors='coerce')
            
            print(f"Historical data after processing: {len(df_hist_clean)} rows")
            
            # Combine datasets
            hist_subset = df_hist_clean[['part_code', 'date', 'sales']].copy()
            current_subset = df_2024_long[['part_code', 'date', 'sales']].copy()
            
            # Ensure part_code is string in both datasets
            hist_subset['part_code'] = hist_subset['part_code'].astype(str)
            current_subset['part_code'] = current_subset['part_code'].astype(str)
            
            combined_df = pd.concat([hist_subset, current_subset], ignore_index=True)
            
            # Add metadata
            if available_metadata:
                metadata_df = df_2024_long[['part_code'] + available_metadata].drop_duplicates()
                metadata_df['part_code'] = metadata_df['part_code'].astype(str)
                combined_df = combined_df.merge(metadata_df, on='part_code', how='left')
            
            # Fill missing metadata columns
            for col in ['description', 'brand', 'engine']:
                if col not in combined_df.columns:
                    combined_df[col] = 'Unknown'
                else:
                    combined_df[col] = combined_df[col].fillna('Unknown')
            
            # Final cleanup
            combined_df = combined_df.dropna(subset=['sales'])
            combined_df = combined_df[combined_df['sales'] >= 0]  # Remove negative sales
            combined_df = combined_df.sort_values(['part_code', 'date']).reset_index(drop=True)
            
            print(f"\n=== FINAL DATASET ===")
            print(f"Total records: {len(combined_df)}")
            print(f"Unique parts: {combined_df['part_code'].nunique()}")
            print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
            print(f"Columns: {combined_df.columns.tolist()}")
            
            if len(combined_df) == 0:
                raise ValueError("No valid data found after processing. Please check your file formats.")
            
            return combined_df
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            print(error_msg)
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            raise Exception(error_msg)
    
    def engineer_features(self, df):
        """Create advanced features for better predictions"""
        
        df_features = df.copy()
        df_features = df_features.sort_values(['part_code', 'date'])
        
        # Time-based features
        df_features['year'] = df_features['date'].dt.year
        df_features['month'] = df_features['date'].dt.month
        df_features['quarter'] = df_features['date'].dt.quarter
        df_features['is_year_end'] = (df_features['month'] == 12).astype(int)
        df_features['is_quarter_end'] = df_features['month'].isin([3, 6, 9, 12]).astype(int)
        
        # Seasonal features
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
        df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
        
        # Part-specific features
        le_brand = LabelEncoder()
        le_engine = LabelEncoder()
        
        df_features['brand_encoded'] = le_brand.fit_transform(df_features['brand'].fillna('Unknown'))
        df_features['engine_encoded'] = le_engine.fit_transform(df_features['engine'].fillna('Unknown'))
        
        # Lag features and rolling statistics
        for part in df_features['part_code'].unique():
            mask = df_features['part_code'] == part
            part_data = df_features[mask].copy()
            
            # Lag features
            for lag in [1, 3, 6, 12]:
                col_name = f'sales_lag_{lag}'
                df_features.loc[mask, col_name] = part_data['sales'].shift(lag)
            
            # Rolling statistics
            for window in [3, 6, 12]:
                df_features.loc[mask, f'sales_roll_mean_{window}'] = part_data['sales'].rolling(window).mean()
                df_features.loc[mask, f'sales_roll_std_{window}'] = part_data['sales'].rolling(window).std()
                df_features.loc[mask, f'sales_roll_min_{window}'] = part_data['sales'].rolling(window).min()
                df_features.loc[mask, f'sales_roll_max_{window}'] = part_data['sales'].rolling(window).max()
            
            # Trend features
            df_features.loc[mask, 'sales_trend'] = part_data['sales'].pct_change(periods=3)
            df_features.loc[mask, 'sales_momentum'] = part_data['sales'].pct_change(periods=1)
            
            # Seasonal decomposition features
            if len(part_data) >= 24:  # Need at least 2 years
                try:
                    decomp = seasonal_decompose(part_data['sales'].dropna(), period=12, extrapolate_trend='freq')
                    df_features.loc[mask, 'seasonal_component'] = decomp.seasonal
                    df_features.loc[mask, 'trend_component'] = decomp.trend
                except:
                    df_features.loc[mask, 'seasonal_component'] = 0
                    df_features.loc[mask, 'trend_component'] = part_data['sales']
        
        # Part category features (based on description keywords)
        df_features['is_valve'] = df_features['description'].str.contains('VALVE', na=False).astype(int)
        df_features['is_guide'] = df_features['description'].str.contains('GUIDE', na=False).astype(int)
        df_features['is_gasket'] = df_features['description'].str.contains('GASKET', na=False).astype(int)
        df_features['is_filter'] = df_features['description'].str.contains('FILTER', na=False).astype(int)
        
        return df_features
    
    def create_time_series_splits(self, df, n_splits=5):
        """Create proper time series cross-validation splits"""
        
        unique_dates = sorted(df['date'].unique())
        total_periods = len(unique_dates)
        
        splits = []
        for i in range(n_splits):
            # Expanding window approach
            train_end_idx = int(total_periods * (0.5 + 0.1 * i))  # Start with 50%, expand by 10% each fold
            test_start_idx = train_end_idx
            test_end_idx = min(train_end_idx + int(total_periods * 0.2), total_periods)  # 20% for testing
            
            if test_end_idx <= test_start_idx:
                continue
                
            train_dates = unique_dates[:train_end_idx]
            test_dates = unique_dates[test_start_idx:test_end_idx]
            
            train_idx = df[df['date'].isin(train_dates)].index
            test_idx = df[df['date'].isin(test_dates)].index
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def build_improved_xgboost(self):
        """XGBoost with proper regularization to prevent overfitting"""
        return xgb.XGBRegressor(
            n_estimators=100,  # Reduced to prevent overfitting
            max_depth=4,       # Reduced depth
            learning_rate=0.05, # Lower learning rate
            subsample=0.8,     # Row sampling
            colsample_bytree=0.8, # Column sampling
            reg_alpha=1.0,     # L1 regularization
            reg_lambda=1.0,    # L2 regularization
            early_stopping_rounds=10,
            random_state=42
        )
    
    def build_improved_prophet(self, df_part):
        """Prophet with proper regularization"""
        
        # Prepare data for Prophet
        prophet_df = df_part[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.01,  # Reduced for less overfitting
            seasonality_prior_scale=1.0,   # Moderate seasonality
            n_changepoints=10,             # Reduced changepoints
            interval_width=0.8
        )
        
        return model, prophet_df
    
    def build_lstm_model(self, input_shape):
        """LSTM model for time series prediction"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def prepare_lstm_data(self, series, lookback=12):
        """Prepare data for LSTM training"""
        X, y = [], []
        for i in range(lookback, len(series)):
            X.append(series[i-lookback:i])
            y.append(series[i])
        return np.array(X), np.array(y)
    
    def train_models_for_part(self, df_part, part_code):
        """Train all models for a specific part"""
        
        if len(df_part) < 24:  # Need at least 2 years of data
            print(f"Insufficient data for part {part_code}")
            return None
        
        df_part = df_part.sort_values('date').reset_index(drop=True)
        
        # Prepare features
        feature_cols = [col for col in df_part.columns if col.startswith(('sales_lag', 'sales_roll', 'month', 'quarter', 
                                                                         'year', 'brand_encoded', 'engine_encoded',
                                                                         'seasonal_component', 'trend_component',
                                                                         'is_valve', 'is_guide', 'is_gasket', 'is_filter'))]
        
        # Remove rows with NaN values (from lag features)
        df_clean = df_part.dropna(subset=feature_cols + ['sales'])
        
        if len(df_clean) < 12:
            return None
        
        X = df_clean[feature_cols].values
        y = df_clean['sales'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Time series cross-validation
        cv_splits = self.create_time_series_splits(df_clean, n_splits=3)
        
        models = {}
        cv_scores = {}
        
                # XGBoost with regularization
        print(f"Training XGBoost for part {part_code}")
        xgb_scores = []
        
        for train_idx, test_idx in cv_splits:
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                # Check for valid data
                if len(y_test) == 0 or np.all(y_test == 0):
                    print(f"Invalid test data for XGBoost CV in part {part_code}")
                    continue
                    
                xgb_model_cv = self.build_improved_xgboost()
                xgb_model_cv.fit(X_train, y_train, 
                             eval_set=[(X_test, y_test)], 
                             verbose=False)
                
                pred = xgb_model_cv.predict(X_test)
                
                # Safe MAPE calculation
                # MAPE = mean(|actual - predicted| / |actual|) * 100
                # Avoid division by zero by adding small epsilon to denominator
                epsilon = 1e-8
                mape_values = np.abs(y_test - pred) / (np.abs(y_test) + epsilon) * 100
                score = np.mean(mape_values)
                
                # Cap extreme values
                score = min(score, 1000)  # Cap at 1000% error
                xgb_scores.append(score)
                
            except Exception as e:
                print(f"XGBoost CV iteration failed for part {part_code}: {e}")
                continue
        
        # Final fit on all data
        try:
            final_xgb = self.build_improved_xgboost()
            final_xgb.fit(X_scaled, y, verbose=False)
            models['xgboost'] = final_xgb
            cv_scores['xgboost'] = np.mean(xgb_scores) if xgb_scores else float('inf')
        except Exception as e:
            print(f"Final XGBoost training failed for part {part_code}: {e}")
        
        # Prophet
        try:
            print(f"Training Prophet for part {part_code}")
            prophet_scores = []
            
            for train_idx, test_idx in cv_splits:
                train_data = df_clean.iloc[train_idx][['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
                test_data = df_clean.iloc[test_idx][['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
                
                try:
                    prophet_model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        seasonality_mode='multiplicative',
                        changepoint_prior_scale=0.01,
                        seasonality_prior_scale=1.0,
                        n_changepoints=10,
                        interval_width=0.8
                    )
                    
                    prophet_model.fit(train_data)
                    forecast = prophet_model.predict(test_data[['ds']])
                    
                    pred = forecast['yhat'].values
                    actual = test_data['y'].values
                    score = mean_absolute_percentage_error(actual, pred)
                    prophet_scores.append(score)
                except Exception as e:
                    print(f"Prophet CV iteration failed: {e}")
                    prophet_scores.append(float('inf'))
            
            # Final Prophet model
            try:
                final_prophet = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.01,
                    seasonality_prior_scale=1.0,
                    n_changepoints=10,
                    interval_width=0.8
                )
                
                prophet_df = df_clean[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
                final_prophet.fit(prophet_df)
                models['prophet'] = final_prophet
                cv_scores['prophet'] = np.mean(prophet_scores) if prophet_scores else float('inf')
            except Exception as e:
                print(f"Final Prophet training failed for part {part_code}: {e}")
                
        except Exception as e:
            print(f"Prophet failed for part {part_code}: {e}")
        
        # Simple Moving Average as baseline
        try:
            print(f"Training Moving Average for part {part_code}")
            ma_scores = []
            
            for train_idx, test_idx in cv_splits:
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Simple 3-month moving average
                if len(y_train) >= 3:
                    ma_pred = np.mean(y_train[-3:])  # Use last 3 months
                    ma_pred_array = np.full(len(y_test), ma_pred)
                    
                    # Safe MAPE calculation for Moving Average
                    if len(y_test) > 0 and not np.all(y_test == 0):
                        epsilon = 1e-8
                        mape_values = np.abs(y_test - ma_pred_array) / (np.abs(y_test) + epsilon) * 100
                        score = np.mean(mape_values)
                        score = min(score, 1000)  # Cap at 1000% error
                        ma_scores.append(score)
            
            # Store moving average model (just the last 3 values)
            if len(y) >= 3:
                models['moving_average'] = {'last_values': y[-3:].tolist()}
                cv_scores['moving_average'] = np.mean(ma_scores) if ma_scores else float('inf')
                
        except Exception as e:
            print(f"Moving Average failed for part {part_code}: {e}")
        
        # Ridge Regression
        try:
            print(f"Training Ridge for part {part_code}")
            ridge_scores = []
            
            for train_idx, test_idx in cv_splits:
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                ridge_model = Ridge(alpha=1.0)
                ridge_model.fit(X_train, y_train)
                pred = ridge_model.predict(X_test)
                
                # Safe MAPE calculation for Ridge
                if len(y_test) > 0 and not np.all(y_test == 0):
                    epsilon = 1e-8
                    mape_values = np.abs(y_test - pred) / (np.abs(y_test) + epsilon) * 100
                    score = np.mean(mape_values)
                    score = min(score, 1000)  # Cap at 1000% error
                    ridge_scores.append(score)
            
            # Final Ridge model
            final_ridge = Ridge(alpha=1.0)
            final_ridge.fit(X_scaled, y)
            models['ridge'] = final_ridge
            cv_scores['ridge'] = np.mean(ridge_scores) if ridge_scores else float('inf')
            
        except Exception as e:
            print(f"Ridge failed for part {part_code}: {e}")
        
        # LSTM (if TensorFlow is available and enough data)
        if TENSORFLOW_AVAILABLE and len(df_clean) >= 36:
            try:
                print(f"Training LSTM for part {part_code}")
                lstm_scores = []
                
                for train_idx, test_idx in cv_splits:
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    if len(y_train) >= 24:  # Need enough data for LSTM
                        X_lstm_train, y_lstm_train = self.prepare_lstm_data(y_train, lookback=12)
                        
                        if len(X_lstm_train) > 0:
                            lstm_model = self.build_lstm_model((12, 1))
                            X_lstm_train = X_lstm_train.reshape((X_lstm_train.shape[0], X_lstm_train.shape[1], 1))
                            
                            lstm_model.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=32, verbose=0)
                            
                            # Simple prediction using last 12 values
                            if len(y_train) >= 12:
                                last_sequence = y_train[-12:].reshape(1, 12, 1)
                                lstm_pred = lstm_model.predict(last_sequence)[0, 0]
                                lstm_pred_array = np.full(len(y_test), lstm_pred)
                                score = mean_absolute_percentage_error(y_test, lstm_pred_array)
                                lstm_scores.append(score)
                
                # Final LSTM model
                if len(y) >= 24:
                    X_lstm_all, y_lstm_all = self.prepare_lstm_data(y, lookback=12)
                    if len(X_lstm_all) > 0:
                        X_lstm_all = X_lstm_all.reshape((X_lstm_all.shape[0], X_lstm_all.shape[1], 1))
                        
                        final_lstm = self.build_lstm_model((12, 1))
                        final_lstm.fit(X_lstm_all, y_lstm_all, epochs=100, batch_size=32, verbose=0)
                        
                        models['lstm'] = final_lstm
                        cv_scores['lstm'] = np.mean(lstm_scores) if lstm_scores else float('inf')
                        
            except Exception as e:
                print(f"LSTM failed for part {part_code}: {e}")
        
        print(f"Completed training for part {part_code}. Models: {list(models.keys())}")
        
        return {
            'models': models,
            'cv_scores': cv_scores,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'recent_sales': y[-12:].tolist() if len(y) >= 12 else y.tolist()  # Store recent sales for prediction
        }
    
    def create_ensemble_predictions(self, models_dict, method='weighted_avg'):
        """Create ensemble predictions with different weighting strategies - DEPRECATED"""
        # This method is now replaced by the logic in predict_next_month
        # Keeping for backward compatibility
        return None
    
    def train_system(self, df):
        """Train the complete prediction system"""
        
        print("Training improved sales prediction system...")
        
        # Engineer features
        print("Engineering features...")
        df_features = self.engineer_features(df)
        
        # Train models for each part
        print("Training models for each part...")
        
        part_results = {}
        for part_code in df_features['part_code'].unique():
            print(f"Training models for part: {part_code}")
            
            df_part = df_features[df_features['part_code'] == part_code].copy()
            result = self.train_models_for_part(df_part, part_code)
            
            if result:
                part_results[part_code] = result
        
        self.models = part_results
        print(f"Training completed for {len(part_results)} parts")
        
        return part_results
    
    def predict_next_month(self, df, part_code, months_ahead=1):
        """Make predictions for the next month(s)"""
        
        if part_code not in self.models:
            return None
        
        model_data = self.models[part_code]
        models = model_data['models']
        cv_scores = model_data['cv_scores']
        
        predictions = {}
        
        try:
            # XGBoost prediction
            if 'xgboost' in models:
                try:
                    scaler = model_data['scaler']
                    feature_cols = model_data['feature_cols']
                    
                    # Get latest data for the part
                    df_part = df[df['part_code'] == part_code].sort_values('date')
                    
                    if len(df_part) > 0:
                        # Engineer features for the latest data
                        df_features = self.engineer_features(df)
                        df_part_features = df_features[df_features['part_code'] == part_code].sort_values('date')
                        
                        # Get the most recent complete record
                        latest_features = df_part_features[feature_cols].dropna()
                        
                        if len(latest_features) > 0:
                            X_latest = scaler.transform(latest_features.iloc[-1:].values)
                            xgb_pred = models['xgboost'].predict(X_latest)[0]
                            predictions['xgboost'] = max(0, xgb_pred)  # Ensure non-negative
                except Exception as e:
                    print(f"XGBoost prediction failed for {part_code}: {e}")
            
            # Prophet prediction
            if 'prophet' in models:
                try:
                    prophet_model = models['prophet']
                    
                    # Get last date and predict next month
                    df_part = df[df['part_code'] == part_code].sort_values('date')
                    if len(df_part) > 0:
                        last_date = df_part['date'].max()
                        next_date = last_date + pd.DateOffset(months=1)
                        
                        future_df = pd.DataFrame({'ds': [next_date]})
                        forecast = prophet_model.predict(future_df)
                        
                        prophet_pred = forecast['yhat'].values[0]
                        predictions['prophet'] = max(0, prophet_pred)
                except Exception as e:
                    print(f"Prophet prediction failed for {part_code}: {e}")
            
            # Moving Average prediction
            if 'moving_average' in models:
                try:
                    last_values = model_data['models']['moving_average']['last_values']
                    ma_pred = np.mean(last_values)
                    predictions['moving_average'] = max(0, ma_pred)
                except Exception as e:
                    print(f"Moving Average prediction failed for {part_code}: {e}")
            
            # Ridge prediction
            if 'ridge' in models:
                try:
                    scaler = model_data['scaler']
                    feature_cols = model_data['feature_cols']
                    
                    # Get latest data for the part
                    df_part = df[df['part_code'] == part_code].sort_values('date')
                    
                    if len(df_part) > 0:
                        # Engineer features for the latest data
                        df_features = self.engineer_features(df)
                        df_part_features = df_features[df_features['part_code'] == part_code].sort_values('date')
                        
                        # Get the most recent complete record
                        latest_features = df_part_features[feature_cols].dropna()
                        
                        if len(latest_features) > 0:
                            X_latest = scaler.transform(latest_features.iloc[-1:].values)
                            ridge_pred = models['ridge'].predict(X_latest)[0]
                            predictions['ridge'] = max(0, ridge_pred)
                except Exception as e:
                    print(f"Ridge prediction failed for {part_code}: {e}")
            
            # LSTM prediction
            if 'lstm' in models:
                try:
                    recent_sales = model_data['recent_sales']
                    if len(recent_sales) >= 12:
                        lstm_input = np.array(recent_sales[-12:]).reshape(1, 12, 1)
                        lstm_pred = models['lstm'].predict(lstm_input)[0, 0]
                        predictions['lstm'] = max(0, lstm_pred)
                except Exception as e:
                    print(f"LSTM prediction failed for {part_code}: {e}")
            
            # If no predictions were successful, use simple average of recent sales
            if not predictions:
                df_part = df[df['part_code'] == part_code].sort_values('date')
                if len(df_part) >= 3:
                    recent_avg = df_part['sales'].tail(3).mean()
                    predictions['fallback'] = max(0, recent_avg)
            
            # Create ensemble prediction
            if predictions:
                # Weight by inverse of CV score (lower error = higher weight)
                weights = {}
                total_weight = 0
                
                for model_name in predictions.keys():
                    if model_name in cv_scores and cv_scores[model_name] != float('inf'):
                        weight = 1 / (cv_scores[model_name] + 0.01)  # Add small epsilon
                        weights[model_name] = weight
                        total_weight += weight
                    else:
                        weights[model_name] = 1.0  # Default weight
                        total_weight += 1.0
                
                if total_weight == 0:
                    ensemble_pred = np.mean(list(predictions.values()))
                else:
                    # Normalize weights and compute weighted average
                    ensemble_pred = 0
                    for model_name, pred in predictions.items():
                        normalized_weight = weights[model_name] / total_weight
                        ensemble_pred += pred * normalized_weight
                
                return {
                    'part_code': part_code,
                    'predicted_sales': ensemble_pred,
                    'individual_predictions': predictions,
                    'models_used': list(predictions.keys()),
                    'cv_scores': cv_scores,
                    'weights_used': weights if total_weight > 0 else {}
                }
            
            return None
            
        except Exception as e:
            print(f"Overall prediction failed for part {part_code}: {e}")
            return None
    
    def evaluate_system_performance(self, df):
        """Evaluate the overall system performance"""
        
        print("Evaluating system performance...")
        
        results = []
        
        for part_code in self.models.keys():
            try:
                model_data = self.models[part_code]
                cv_scores = model_data['cv_scores']
                
                # Filter out infinite scores and invalid values
                valid_scores = {name: score for name, score in cv_scores.items() 
                               if score != float('inf') and not np.isnan(score) and score > 0}
                
                if valid_scores:
                    # Calculate ensemble score (weighted average of individual model scores)
                    weights = {name: 1/(score + 0.01) for name, score in valid_scores.items()}
                    total_weight = sum(weights.values())
                    
                    if total_weight > 0:
                        ensemble_score = sum(score * (weights[name]/total_weight) 
                                           for name, score in valid_scores.items())
                    else:
                        ensemble_score = np.mean(list(valid_scores.values()))
                    
                    # Find best and worst models
                    best_model = min(valid_scores.items(), key=lambda x: x[1])
                    worst_model = max(valid_scores.items(), key=lambda x: x[1])
                    
                    results.append({
                        'part_code': part_code,
                        'ensemble_mape': ensemble_score,
                        'best_model': best_model[0],
                        'best_model_score': best_model[1],
                        'worst_model': worst_model[0],
                        'worst_model_score': worst_model[1],
                        'model_count': len(valid_scores),
                        'total_models_attempted': len(cv_scores)
                    })
                else:
                    # No valid scores - use fallback
                    results.append({
                        'part_code': part_code,
                        'ensemble_mape': 100.0,  # High error score for no valid models
                        'best_model': 'none',
                        'best_model_score': float('inf'),
                        'worst_model': 'none',
                        'worst_model_score': float('inf'),
                        'model_count': 0,
                        'total_models_attempted': len(cv_scores)
                    })
                    
            except Exception as e:
                print(f"Error evaluating part {part_code}: {e}")
                # Add a default entry for failed parts
                results.append({
                    'part_code': part_code,
                    'ensemble_mape': 999.0,  # Very high error score
                    'best_model': 'error',
                    'best_model_score': float('inf'),
                    'worst_model': 'error', 
                    'worst_model_score': float('inf'),
                    'model_count': 0,
                    'total_models_attempted': 0
                })
        
        if not results:
            print("No results to evaluate!")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # Filter out parts with no valid models for summary statistics
        valid_results = results_df[results_df['model_count'] > 0]
        
        print(f"\nSystem Performance Summary:")
        if len(valid_results) > 0:
            print(f"Parts with valid models: {len(valid_results)}/{len(results_df)}")
            print(f"Average Ensemble MAPE: {valid_results['ensemble_mape'].mean():.2f}%")
            print(f"Median Ensemble MAPE: {valid_results['ensemble_mape'].median():.2f}%")
            print(f"Best performing part: {valid_results.loc[valid_results['ensemble_mape'].idxmin(), 'part_code']} ({valid_results['ensemble_mape'].min():.2f}%)")
            
            print(f"\nBest Performing Parts (Top 5):")
            top_performers = valid_results.nsmallest(5, 'ensemble_mape')
            for _, row in top_performers.iterrows():
                print(f"  {row['part_code']}: {row['ensemble_mape']:.2f}% (best: {row['best_model']})")
            
            print(f"\nWorst Performing Parts (Bottom 5):")
            worst_performers = valid_results.nlargest(5, 'ensemble_mape')
            for _, row in worst_performers.iterrows():
                print(f"  {row['part_code']}: {row['ensemble_mape']:.2f}% (best: {row['best_model']})")
        else:
            print("No parts had valid model predictions!")
        
        # Show parts with no valid models
        failed_parts = results_df[results_df['model_count'] == 0]
        if len(failed_parts) > 0:
            print(f"\nParts with no valid models ({len(failed_parts)}):")
            for part_code in failed_parts['part_code'].head(10):  # Show first 10
                print(f"  {part_code}")
        
        return results_df

# Streamlit Integration Functions
def create_streamlit_app():
    """Create a Streamlit app for the prediction system"""
    import streamlit as st
    
    st.title("🔧 Improved Sales Prediction System")
    st.markdown("Upload your sales data files to get started with improved predictions!")
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state['predictor'] = None
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'models_trained' not in st.session_state:
        st.session_state['models_trained'] = False
    if 'performance' not in st.session_state:
        st.session_state['performance'] = None
    
    # File uploaders
    st.header("📁 Upload Data Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sales_2024_file = st.file_uploader(
            "Upload 2024 Sales Data", 
            type=['xlsx', 'xls'],
            help="Upload your SPC Sales per part 2024.xlsx file"
        )
    
    with col2:
        sales_hist_file = st.file_uploader(
            "Upload 2022-2023 Historical Data", 
            type=['xlsx', 'xls'],
            help="Upload your SPC_Sales_2022_2023_Formatted.xlsx file"
        )
    
    if sales_2024_file and sales_hist_file:
        try:
            # Initialize predictor if not already done
            if st.session_state['predictor'] is None:
                st.session_state['predictor'] = ImprovedSalesPredictionSystem()
            
            predictor = st.session_state['predictor']
            
            # Load data with progress bar (only if not already loaded)
            if st.session_state['df'] is None:
                with st.spinner("Loading and preparing data..."):
                    df = predictor.load_and_prepare_data(sales_2024_file, sales_hist_file)
                    st.session_state['df'] = df
            else:
                df = st.session_state['df']
            
            st.success(f"✅ Data loaded successfully!")
            st.info(f"📊 Loaded {len(df)} records for {df['part_code'].nunique()} parts")
            st.info(f"📅 Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
            
            # Show data preview
            if st.checkbox("Show data preview"):
                st.subheader("Data Preview")
                st.dataframe(df.head(10))
                
                # Basic statistics
                st.subheader("Basic Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Parts", df['part_code'].nunique())
                with col2:
                    st.metric("Total Records", len(df))
                with col3:
                    st.metric("Avg Monthly Sales", f"{df['sales'].mean():.0f}")
                with col4:
                    st.metric("Date Range (Months)", df['date'].nunique())
            
            # Training section
            st.header("🎯 Train Prediction Models")
            
            if not st.session_state['models_trained']:
                if st.button("🚀 Start Training", type="primary"):
                    with st.spinner("Training models... This may take a few minutes."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Engineer features
                        progress_bar.progress(20)
                        status_text.text("Engineering features...")
                        df_features = predictor.engineer_features(df)
                        
                        # Train models
                        progress_bar.progress(40)
                        status_text.text("Training models for each part...")
                        results = predictor.train_system(df)
                        
                        progress_bar.progress(80)
                        status_text.text("Evaluating performance...")
                        performance = predictor.evaluate_system_performance(df)
                        
                        # Store results in session state
                        st.session_state['performance'] = performance
                        st.session_state['models_trained'] = True
                        
                        progress_bar.progress(100)
                        status_text.text("Training completed!")
                        st.success("✅ Training completed!")
                        st.rerun()  # Refresh to show results
            else:
                st.success("✅ Models already trained!")
                performance = st.session_state['performance']
                
                # Display results
                st.header("📈 Model Performance")
                
                if performance is not None and len(performance) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Average MAPE", 
                            f"{performance['ensemble_mape'].mean():.2f}%",
                            delta=f"-{abs(25 - performance['ensemble_mape'].mean()):.1f}% vs baseline"
                        )
                    
                    with col2:
                        st.metric(
                            "Parts Successfully Modeled", 
                            f"{len(performance)}/{df['part_code'].nunique()}",
                            delta=f"{len(performance)/df['part_code'].nunique()*100:.0f}% coverage"
                        )
                    
                    # Performance details
                    st.subheader("Best Performing Parts")
                    top_performers = performance.nsmallest(5, 'ensemble_mape')
                    st.dataframe(top_performers[['part_code', 'ensemble_mape', 'best_model']])
                    
                    st.subheader("Parts Needing Attention")
                    bottom_performers = performance.nlargest(5, 'ensemble_mape')
                    st.dataframe(bottom_performers[['part_code', 'ensemble_mape', 'best_model']])
                
                # Predictions section
                st.header("🔮 Generate Predictions")
                
                # Check if we have trained models
                if hasattr(predictor, 'models') and len(predictor.models) > 0:
                    st.info(f"Ready to generate predictions for {len(predictor.models)} parts")
                    
                    if st.button("Generate Next Month Predictions", type="primary"):
                        with st.spinner("Generating predictions..."):
                            predictions = []
                            prediction_progress = st.progress(0)
                            
                            model_parts = list(predictor.models.keys())
                            total_parts = len(model_parts)
                            
                            for i, part_code in enumerate(model_parts):
                                try:
                                    pred = predictor.predict_next_month(df, part_code)
                                    if pred and pred['predicted_sales'] is not None:
                                        predictions.append(pred)
                                except Exception as e:
                                    st.warning(f"Failed to predict for part {part_code}: {str(e)}")
                                
                                # Update progress
                                prediction_progress.progress((i + 1) / total_parts)
                        
                        if predictions:
                            pred_df = pd.DataFrame(predictions)
                            
                            st.subheader("Prediction Results")
                            st.success(f"✅ Generated predictions for {len(predictions)} parts!")
                            
                            # Summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Predicted Sales", f"{pred_df['predicted_sales'].sum():.0f}")
                            with col2:
                                st.metric("Average per Part", f"{pred_df['predicted_sales'].mean():.0f}")
                            with col3:
                                st.metric("Highest Prediction", f"{pred_df['predicted_sales'].max():.0f}")
                            
                            # Top predictions
                            st.subheader("Top 10 Predicted Sales")
                            top_predictions = pred_df.nlargest(10, 'predicted_sales')
                            st.dataframe(top_predictions[['part_code', 'predicted_sales', 'models_used']])
                            
                            # All predictions table
                            with st.expander("View All Predictions"):
                                st.dataframe(pred_df.sort_values('predicted_sales', ascending=False))
                            
                            # Download predictions
                            csv = pred_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name=f"sales_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                            
                            # Store predictions in session state
                            st.session_state['predictions'] = predictions
                        
                        else:
                            st.error("❌ No predictions generated. Please check your data and try again.")
                
                else:
                    st.warning("⚠️ No trained models available. Please train the models first.")
                    st.session_state['models_trained'] = False
                
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.error("Please check your file formats and try again.")
            
            with st.expander("Error Details"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())
    
    else:
        st.info("👆 Please upload both data files to continue")
        
        # Show sample file format
        with st.expander("Expected File Formats"):
            st.markdown("""
            **2024 Sales File should contain:**
            - Part codes in first column
            - Monthly sales data in subsequent columns
            - Headers like: Part Code, Description, Brand, Engine, Jan-2024, Feb-2024, etc.
            
            **Historical File should contain:**
            - Part codes, dates/months, and sales values
            - Headers like: Part, Month, Sales
            """)
    
    # Add a reset button in the sidebar
    with st.sidebar:
        st.header("🔄 Controls")
        if st.button("Reset Application"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        if st.session_state.get('models_trained', False):
            st.success("✅ Models trained")
            st.info(f"📊 {len(st.session_state.get('predictor', {}).models) if st.session_state.get('predictor') else 0} parts modeled")
        
        # Show memory usage info
        if st.session_state.get('df') is not None:
            df_size = st.session_state['df'].memory_usage(deep=True).sum() / 1024 / 1024
            st.info(f"💾 Data size: {df_size:.1f} MB")
# Usage Example
def main():
    """Main execution function"""
    
    # Check if running in Streamlit
    try:
        import streamlit as st
        # If we can import streamlit and we're in a streamlit context
        if hasattr(st, 'session_state'):
            create_streamlit_app()
            return
    except ImportError:
        pass
    
    # Traditional execution for non-Streamlit environments
    predictor = ImprovedSalesPredictionSystem()
    
    # Load and prepare data
    print("Loading data...")
    df = predictor.load_and_prepare_data('SPC Sales per part 2024.xlsx', 'SPC_Sales_2022_2023_Formatted.xlsx')
    
    # Train the system
    results = predictor.train_system(df)
    
    # Evaluate performance
    performance = predictor.evaluate_system_performance(df)
    
    # Make predictions for all parts
    print("\nMaking predictions for next month...")
    predictions = []
    
    for part_code in predictor.models.keys():
        pred = predictor.predict_next_month(df, part_code)
        if pred:
            predictions.append(pred)
    
    # Display predictions
    pred_df = pd.DataFrame(predictions)
    print("\nTop 10 Predicted Sales for Next Month:")
    print(pred_df.nlargest(10, 'predicted_sales')[['part_code', 'predicted_sales']])
    
    return predictor, df, performance, predictions

if __name__ == "__main__":
    main()
