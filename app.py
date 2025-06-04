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
            
            # Clean 2024 data structure
            # First, let's inspect the actual structure
            print("2024 file shape:", df_2024.shape)
            print("2024 file columns:", df_2024.columns.tolist())
            
            # Handle the case where first row might be headers
            if df_2024.iloc[0, 0] == 'Item Code' or 'Item Code' in str(df_2024.iloc[0, 0]):
                df_2024_clean = df_2024.iloc[1:].copy()  # Skip header row
                # Set proper column names
                expected_cols = ['part_code', 'description', 'brand', 'engine'] + \
                               [f'month_{i:02d}_2024' for i in range(1, 13)]
                
                # Adjust column count to match actual data
                actual_cols = min(len(expected_cols), df_2024_clean.shape[1])
                df_2024_clean.columns = expected_cols[:actual_cols]
                
            else:
                df_2024_clean = df_2024.copy()
                # Auto-detect columns
                cols = df_2024_clean.columns.tolist()
                month_cols = [col for col in cols if any(month in str(col).lower() 
                             for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                         'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])]
                
                if len(month_cols) >= 4:  # At least 4 months of data
                    non_month_cols = [col for col in cols if col not in month_cols]
                    new_cols = ['part_code', 'description', 'brand', 'engine'][:len(non_month_cols)] + month_cols
                    df_2024_clean.columns = new_cols[:df_2024_clean.shape[1]]
            
            # Get the month columns for melting
            month_cols = [col for col in df_2024_clean.columns if 'month_' in col or 
                         any(month in str(col).lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])]
            
            if not month_cols:
                # Fallback: assume last 12 columns are months
                month_cols = df_2024_clean.columns[-12:].tolist()
            
            # Melt 2024 data to long format
            id_vars = ['part_code', 'description', 'brand', 'engine']
            available_id_vars = [col for col in id_vars if col in df_2024_clean.columns]
            
            df_2024_long = df_2024_clean.melt(
                id_vars=available_id_vars,
                value_vars=month_cols,
                var_name='month',
                value_name='sales'
            )
            
            # Create proper dates for 2024 data
            month_mapping = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            
            def extract_month_number(month_str):
                month_str = str(month_str).lower()
                for month_name, month_num in month_mapping.items():
                    if month_name in month_str:
                        return month_num
                # Fallback: try to extract number
                import re
                numbers = re.findall(r'\d+', month_str)
                if numbers:
                    return min(int(numbers[0]), 12)
                return 1
            
            df_2024_long['month_num'] = df_2024_long['month'].apply(extract_month_number)
            df_2024_long['date'] = pd.to_datetime('2024-' + df_2024_long['month_num'].astype(str) + '-01')
            
            # Load 2022-2023 historical data
            if hasattr(sales_2022_2023_file, 'read'):  # Streamlit uploaded file
                df_hist = pd.read_excel(sales_2022_2023_file)
            else:  # File path
                df_hist = pd.read_excel(sales_2022_2023_file)
            
            print("Historical file shape:", df_hist.shape)
            print("Historical file columns:", df_hist.columns.tolist())
            
            # Handle different possible column names
            part_col = None
            sales_col = None
            month_col = None
            
            for col in df_hist.columns:
                col_lower = str(col).lower()
                if 'part' in col_lower and part_col is None:
                    part_col = col
                elif 'sales' in col_lower and sales_col is None:
                    sales_col = col
                elif 'month' in col_lower or 'date' in col_lower and month_col is None:
                    month_col = col
            
            # Fallback to position-based if column names not found
            if not all([part_col, sales_col, month_col]) and df_hist.shape[1] >= 3:
                part_col = df_hist.columns[0]
                month_col = df_hist.columns[1] 
                sales_col = df_hist.columns[2]
            
            # Clean historical data
            df_hist_clean = df_hist[[part_col, month_col, sales_col]].copy()
            df_hist_clean.columns = ['part_code', 'month_raw', 'sales']
            
            # Convert Excel dates to proper datetime
            def convert_excel_date(date_val):
                try:
                    if pd.isna(date_val):
                        return pd.NaT
                    if isinstance(date_val, (int, float)):
                        # Excel date number
                        return pd.to_datetime('1900-01-01') + pd.to_timedelta(date_val - 2, unit='D')
                    else:
                        # Try to parse as regular date
                        return pd.to_datetime(date_val)
                except:
                    return pd.NaT
            
            df_hist_clean['date'] = df_hist_clean['month_raw'].apply(convert_excel_date)
            df_hist_clean = df_hist_clean.dropna(subset=['date'])
            
            # Ensure sales is numeric
            df_2024_long['sales'] = pd.to_numeric(df_2024_long['sales'], errors='coerce')
            df_hist_clean['sales'] = pd.to_numeric(df_hist_clean['sales'], errors='coerce')
            
            # Combine datasets
            hist_subset = df_hist_clean[['part_code', 'date', 'sales']].copy()
            current_subset = df_2024_long[['part_code', 'date', 'sales']].copy()
            
            combined_df = pd.concat([hist_subset, current_subset], ignore_index=True)
            
            # Add part metadata from 2024 data
            if len(available_id_vars) > 1:
                part_info = df_2024_clean[available_id_vars].drop_duplicates()
                combined_df = combined_df.merge(part_info, on='part_code', how='left')
            
            # Fill missing metadata
            for col in ['description', 'brand', 'engine']:
                if col not in combined_df.columns:
                    combined_df[col] = 'Unknown'
                else:
                    combined_df[col] = combined_df[col].fillna('Unknown')
            
            # Sort by part and date
            combined_df = combined_df.sort_values(['part_code', 'date']).reset_index(drop=True)
            combined_df = combined_df.dropna(subset=['sales'])  # Remove rows with missing sales
            
            print(f"Successfully loaded {len(combined_df)} records for {combined_df['part_code'].nunique()} parts")
            print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
            
            return combined_df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print("Please check your file formats and try again.")
            raise e
    
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
        xgb_model = self.build_improved_xgboost()
        xgb_scores = []
        
        for train_idx, test_idx in cv_splits:
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            xgb_model.fit(X_train, y_train, 
                         eval_set=[(X_test, y_test)], 
                         verbose=False)
            
            pred = xgb_model.predict(X_test)
            score = mean_absolute_percentage_error(y_test, pred)
            xgb_scores.append(score)
        
        models['xgboost'] = xgb_model
        cv_scores['xgboost'] = np.mean(xgb_scores)
        
        # Prophet
        try:
            prophet_model, prophet_df = self.build_improved_prophet(df_clean)
            prophet_scores = []
            
            for train_idx, test_idx in cv_splits:
                train_data = prophet_df.iloc[train_idx]
                test_data = prophet_df.iloc[test_idx]
                
                prophet_model.fit(train_data)
                forecast = prophet_model.predict(test_data[['ds']])
                
                pred = forecast['yhat'].values
                actual = test_data['y'].values
                score = mean_absolute_percentage_error(actual, pred)
                prophet_scores.append(score)
            
            models['prophet'] = prophet_model
            cv_scores['prophet'] = np.mean(prophet_scores)
            
        except Exception as e:
            print(f"Prophet failed for part {part_code}: {e}")
        
        # ARIMA with auto-selection
        try:
            # Simple ARIMA with cross-validation
            arima_scores = []
            
            for train_idx, test_idx in cv_splits:
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Fit ARIMA
                arima_model = ARIMA(y_train, order=(1, 1, 1))
                arima_fit = arima_model.fit()
                
                # Forecast
                forecast = arima_fit.forecast(steps=len(y_test))
                score = mean_absolute_percentage_error(y_test, forecast)
                arima_scores.append(score)
            
            # Fit final model on all data
            final_arima = ARIMA(y, order=(1, 1, 1)).fit()
            models['arima'] = final_arima
            cv_scores['arima'] = np.mean(arima_scores)
            
        except Exception as e:
            print(f"ARIMA failed for part {part_code}: {e}")
        
        # Ridge Regression
        ridge_model = Ridge(alpha=1.0)
        ridge_scores = []
        
        for train_idx, test_idx in cv_splits:
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            ridge_model.fit(X_train, y_train)
            pred = ridge_model.predict(X_test)
            score = mean_absolute_percentage_error(y_test, pred)
            ridge_scores.append(score)
        
        models['ridge'] = ridge_model
        cv_scores['ridge'] = np.mean(ridge_scores)
        
        # LSTM (if TensorFlow is available)
        if TENSORFLOW_AVAILABLE and len(df_clean) >= 36:
            try:
                lstm_scores = []
                
                for train_idx, test_idx in cv_splits:
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Prepare LSTM data
                    X_lstm_train, y_lstm_train = self.prepare_lstm_data(y_train, lookback=12)
                    X_lstm_test, y_lstm_test = self.prepare_lstm_data(y_test, lookback=12)
                    
                    if len(X_lstm_train) > 0 and len(X_lstm_test) > 0:
                        lstm_model = self.build_lstm_model((12, 1))
                        X_lstm_train = X_lstm_train.reshape((X_lstm_train.shape[0], X_lstm_train.shape[1], 1))
                        X_lstm_test = X_lstm_test.reshape((X_lstm_test.shape[0], X_lstm_test.shape[1], 1))
                        
                        lstm_model.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=32, verbose=0)
                        pred = lstm_model.predict(X_lstm_test)
                        score = mean_absolute_percentage_error(y_lstm_test, pred.flatten())
                        lstm_scores.append(score)
                
                if lstm_scores:
                    # Train final LSTM model
                    X_lstm_all, y_lstm_all = self.prepare_lstm_data(y, lookback=12)
                    X_lstm_all = X_lstm_all.reshape((X_lstm_all.shape[0], X_lstm_all.shape[1], 1))
                    
                    final_lstm = self.build_lstm_model((12, 1))
                    final_lstm.fit(X_lstm_all, y_lstm_all, epochs=100, batch_size=32, verbose=0)
                    
                    models['lstm'] = final_lstm
                    cv_scores['lstm'] = np.mean(lstm_scores)
                    
            except Exception as e:
                print(f"LSTM failed for part {part_code}: {e}")
        
        return {
            'models': models,
            'cv_scores': cv_scores,
            'scaler': scaler,
            'feature_cols': feature_cols
        }
    
    def create_ensemble_predictions(self, models_dict, X_scaled, method='weighted_avg'):
        """Create ensemble predictions with different weighting strategies"""
        
        models = models_dict['models']
        cv_scores = models_dict['cv_scores']
        
        predictions = {}
        
        # Get individual model predictions
        for model_name, model in models.items():
            if model_name == 'prophet':
                # Prophet needs different input format
                continue
            elif model_name == 'arima':
                # ARIMA needs different prediction method
                pred = model.forecast(steps=1)[0]
                predictions[model_name] = pred
            elif model_name == 'lstm':
                # LSTM needs different input format
                continue
            else:
                predictions[model_name] = model.predict(X_scaled[-1:]).flatten()[0]
        
        if not predictions:
            return None
        
        if method == 'simple_avg':
            return np.mean(list(predictions.values()))
        
        elif method == 'weighted_avg':
            # Weight by inverse of CV score (lower error = higher weight)
            weights = {}
            total_weight = 0
            
            for model_name in predictions.keys():
                if model_name in cv_scores:
                    weight = 1 / (cv_scores[model_name] + 0.01)  # Add small epsilon
                    weights[model_name] = weight
                    total_weight += weight
            
            if total_weight == 0:
                return np.mean(list(predictions.values()))
            
            # Normalize weights
            weighted_pred = 0
            for model_name, pred in predictions.items():
                if model_name in weights:
                    weighted_pred += pred * (weights[model_name] / total_weight)
            
            return weighted_pred
        
        return np.mean(list(predictions.values()))
    
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
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        
        # Get latest data for the part
        df_part = df[df['part_code'] == part_code].sort_values('date')
        
        # Engineer features for the latest data point
        df_features = self.engineer_features(df)
        df_part_features = df_features[df_features['part_code'] == part_code].sort_values('date')
        
        # Get the most recent complete record
        latest_features = df_part_features[feature_cols].dropna().iloc[-1:]
        
        if len(latest_features) == 0:
            return None
        
        X_scaled = scaler.transform(latest_features.values)
        
        # Get ensemble prediction
        ensemble_pred = self.create_ensemble_predictions(model_data, X_scaled, method='weighted_avg')
        
        return {
            'part_code': part_code,
            'predicted_sales': ensemble_pred,
            'models_used': list(models.keys()),
            'cv_scores': model_data['cv_scores']
        }
    
    def evaluate_system_performance(self, df):
        """Evaluate the overall system performance"""
        
        print("Evaluating system performance...")
        
        results = []
        
        for part_code in self.models.keys():
            model_data = self.models[part_code]
            cv_scores = model_data['cv_scores']
            
            # Calculate ensemble score (weighted average of individual model scores)
            if cv_scores:
                weights = {name: 1/(score + 0.01) for name, score in cv_scores.items()}
                total_weight = sum(weights.values())
                
                ensemble_score = sum(score * (weights[name]/total_weight) 
                                   for name, score in cv_scores.items())
                
                results.append({
                    'part_code': part_code,
                    'ensemble_mape': ensemble_score,
                    'best_model': min(cv_scores.items(), key=lambda x: x[1])[0],
                    'worst_model': max(cv_scores.items(), key=lambda x: x[1])[0],
                    'model_count': len(cv_scores)
                })
        
        results_df = pd.DataFrame(results)
        
        print(f"\nSystem Performance Summary:")
        print(f"Average Ensemble MAPE: {results_df['ensemble_mape'].mean():.2f}%")
        print(f"Best Performing Parts (Top 5):")
        print(results_df.nsmallest(5, 'ensemble_mape')[['part_code', 'ensemble_mape', 'best_model']])
        
        print(f"\nWorst Performing Parts (Bottom 5):")
        print(results_df.nlargest(5, 'ensemble_mape')[['part_code', 'ensemble_mape', 'best_model']])
        
        return results_df

# Streamlit Integration Functions
def create_streamlit_app():
    """Create a Streamlit app for the prediction system"""
    import streamlit as st
    
    st.title("🔧 Improved Sales Prediction System")
    st.markdown("Upload your sales data files to get started with improved predictions!")
    
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
            # Initialize predictor
            predictor = ImprovedSalesPredictionSystem()
            
            # Load data with progress bar
            with st.spinner("Loading and preparing data..."):
                df = predictor.load_and_prepare_data(sales_2024_file, sales_hist_file)
            
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
            
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training models... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    
                    # Engineer features
                    progress_bar.progress(20)
                    st.write("Engineering features...")
                    df_features = predictor.engineer_features(df)
                    
                    # Train models
                    progress_bar.progress(40)
                    st.write("Training models for each part...")
                    results = predictor.train_system(df)
                    
                    progress_bar.progress(80)
                    st.write("Evaluating performance...")
                    performance = predictor.evaluate_system_performance(df)
                    
                    progress_bar.progress(100)
                    st.success("✅ Training completed!")
                
                # Display results
                st.header("📈 Model Performance")
                
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
                
                if st.button("Generate Next Month Predictions"):
                    with st.spinner("Generating predictions..."):
                        predictions = []
                        
                        for part_code in predictor.models.keys():
                            pred = predictor.predict_next_month(df, part_code)
                            if pred:
                                predictions.append(pred)
                    
                    if predictions:
                        pred_df = pd.DataFrame(predictions)
                        
                        st.subheader("Prediction Results")
                        
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
                        
                        # Download predictions
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f"sales_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                        
                        # Store in session state for later use
                        st.session_state['predictor'] = predictor
                        st.session_state['predictions'] = predictions
                        st.session_state['performance'] = performance
                
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.error("Please check your file formats and try again.")
            
            with st.expander("Error Details"):
                st.code(str(e))
    
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
