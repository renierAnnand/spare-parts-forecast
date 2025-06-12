import streamlit as st

# Configure streamlit FIRST - must be before any other st commands
st.set_page_config(page_title="Enhanced Accuracy Sales Forecasting Dashboard", layout="wide")

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from scipy import stats
from scipy.optimize import minimize
import itertools
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Forecasting libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")


class AdvancedTimeSeriesFeatures:
    """Create advanced features from time series data alone"""
    
    @staticmethod
    def create_lag_features(data, max_lags=12):
        """Create multiple lag features"""
        features = data.copy()
        
        for lag in [1, 2, 3, 6, 12]:
            if lag <= max_lags and len(data) > lag:
                features[f'lag_{lag}'] = data['Sales'].shift(lag)
        
        return features
    
    @staticmethod
    def create_rolling_features(data, windows=[3, 6, 12]):
        """Create rolling statistics features"""
        features = data.copy()
        
        for window in windows:
            if len(data) >= window:
                features[f'rolling_mean_{window}'] = data['Sales'].rolling(window).mean()
                features[f'rolling_std_{window}'] = data['Sales'].rolling(window).std()
                features[f'rolling_min_{window}'] = data['Sales'].rolling(window).min()
                features[f'rolling_max_{window}'] = data['Sales'].rolling(window).max()
        
        return features
    
    @staticmethod
    def create_trend_features(data):
        """Create trend-based features"""
        features = data.copy()
        
        # Linear trend
        if len(data) >= 6:
            features['trend_6m'] = data['Sales'].rolling(6).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 6 else np.nan
            )
        
        if len(data) >= 12:
            features['trend_12m'] = data['Sales'].rolling(12).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 12 else np.nan
            )
        
        # Growth rates
        features['growth_1m'] = data['Sales'].pct_change(1)
        features['growth_3m'] = data['Sales'].pct_change(3)
        features['growth_12m'] = data['Sales'].pct_change(12)
        
        return features
    
    @staticmethod
    def create_seasonal_features(data):
        """Create advanced seasonal features"""
        features = data.copy()
        
        # Month-based features
        features['month'] = features['Month'].dt.month
        features['quarter'] = features['Month'].dt.quarter
        
        # Cyclical encoding for seasonality
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['quarter_sin'] = np.sin(2 * np.pi * features['quarter'] / 4)
        features['quarter_cos'] = np.cos(2 * np.pi * features['quarter'] / 4)
        
        # Seasonal decomposition features
        if len(data) >= 24:
            try:
                decomposition = seasonal_decompose(data['Sales'], model='additive', period=12)
                features['seasonal_component'] = decomposition.seasonal
                features['trend_component'] = decomposition.trend
                features['residual_component'] = decomposition.resid
            except:
                pass
        
        return features


class AdvancedCrossValidation:
    """Advanced cross-validation for time series"""
    
    @staticmethod
    def time_series_cv(data, model_func, n_splits=5, test_size=6):
        """Perform time series cross-validation"""
        if len(data) < n_splits * test_size + 12:
            return None, np.inf
        
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        errors = []
        
        for train_idx, test_idx in tscv.split(data):
            try:
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                # Train model
                forecast, _ = model_func(train_data, forecast_periods=len(test_data))
                
                # Calculate error
                actual = test_data['Sales'].values
                mae = mean_absolute_error(actual, forecast[:len(actual)])
                errors.append(mae)
                
            except Exception:
                continue
        
        return errors, np.mean(errors) if errors else np.inf


class AdaptiveEnsemble:
    """Adaptive ensemble that learns from recent performance"""
    
    def __init__(self, window_size=6):
        self.window_size = window_size
        self.performance_history = {}
    
    def update_performance(self, model_name, actual, forecast):
        """Update model performance history"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        mae = mean_absolute_error([actual], [forecast])
        self.performance_history[model_name].append(mae)
        
        # Keep only recent performance
        if len(self.performance_history[model_name]) > self.window_size:
            self.performance_history[model_name] = self.performance_history[model_name][-self.window_size:]
    
    def get_adaptive_weights(self, model_names):
        """Calculate adaptive weights based on recent performance"""
        weights = {}
        
        for model_name in model_names:
            if model_name in self.performance_history and self.performance_history[model_name]:
                # Use inverse of recent average error
                recent_avg_error = np.mean(self.performance_history[model_name])
                weights[model_name] = 1 / (recent_avg_error + 1e-6)
            else:
                weights[model_name] = 1.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        for model_name in weights:
            weights[model_name] /= total_weight
        
        return weights


def detect_optimal_transformation(data):
    """Automatically detect the best transformation for the data"""
    sales = data['Sales'].values
    transformations = {
        'none': sales,
        'log': np.log1p(sales),
        'sqrt': np.sqrt(sales),
        'box_cox': None
    }
    
    # Try Box-Cox transformation
    try:
        from scipy.stats import boxcox
        bc_data, lambda_param = boxcox(sales + 1)  # Add 1 to handle zeros
        transformations['box_cox'] = bc_data
    except:
        pass
    
    best_transform = 'none'
    best_score = np.inf
    
    # Test each transformation using normality and variance
    for name, transformed_data in transformations.items():
        if transformed_data is not None:
            try:
                # Combine multiple criteria
                skewness = abs(stats.skew(transformed_data))
                kurtosis = abs(stats.kurtosis(transformed_data))
                score = skewness + kurtosis * 0.5
                
                if score < best_score:
                    best_score = score
                    best_transform = name
            except:
                continue
    
    return best_transform


def advanced_outlier_detection(data, method='isolation_forest'):
    """Advanced outlier detection and treatment"""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    
    sales = data['Sales'].values.reshape(-1, 1)
    
    if method == 'isolation_forest':
        detector = IsolationForest(contamination=0.1, random_state=42)
        outliers = detector.fit_predict(sales) == -1
    elif method == 'lof':
        detector = LocalOutlierFactor(contamination=0.1)
        outliers = detector.fit_predict(sales) == -1
    else:
        # IQR method (fallback)
        Q1 = np.percentile(sales, 25)
        Q3 = np.percentile(sales, 75)
        IQR = Q3 - Q1
        outliers = (sales < (Q1 - 1.5 * IQR)) | (sales > (Q3 + 1.5 * IQR))
        outliers = outliers.flatten()
    
    return outliers


def create_advanced_features(data):
    """Create comprehensive feature set from existing data"""
    feature_creator = AdvancedTimeSeriesFeatures()
    
    # Start with base data
    enhanced_data = data.copy()
    
    # Add all feature types
    enhanced_data = feature_creator.create_lag_features(enhanced_data)
    enhanced_data = feature_creator.create_rolling_features(enhanced_data)
    enhanced_data = feature_creator.create_trend_features(enhanced_data)
    enhanced_data = feature_creator.create_seasonal_features(enhanced_data)
    
    return enhanced_data


def run_advanced_ml_ensemble(data, forecast_periods=12, scaling_factor=1.0):
    """Advanced ML ensemble with feature engineering"""
    try:
        # Create advanced features
        enhanced_data = create_advanced_features(data)
        
        # Select feature columns (exclude date and target)
        feature_cols = [col for col in enhanced_data.columns 
                       if col not in ['Month', 'Sales', 'Sales_Original', 'log_transformed']]
        
        # Prepare features and target
        X = enhanced_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
        y = enhanced_data['Sales']
        
        # Remove rows with any remaining NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 12:
            raise ValueError("Insufficient data for ML ensemble")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define multiple models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=0.1, random_state=42)
        }
        
        # Train models with time series CV
        model_weights = {}
        trained_models = {}
        
        for name, model in models.items():
            try:
                # Use recent data for training
                train_size = max(12, int(len(X) * 0.8))
                X_train = X_scaled[-train_size:]
                y_train = y.iloc[-train_size:]
                
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Calculate weight based on recent performance
                y_pred = model.predict(X_train[-6:])  # Last 6 months
                mae = mean_absolute_error(y_train.iloc[-6:], y_pred)
                model_weights[name] = 1 / (mae + 1e-6)
                
            except Exception:
                continue
        
        if not trained_models:
            raise ValueError("No models could be trained")
        
        # Normalize weights
        total_weight = sum(model_weights.values())
        for name in model_weights:
            model_weights[name] /= total_weight
        
        # Generate forecasts
        forecasts = []
        
        for i in range(forecast_periods):
            # Create features for forecast period
            last_date = enhanced_data['Month'].iloc[-1]
            forecast_date = last_date + pd.DateOffset(months=i+1)
            
            # Create forecast features (simplified approach)
            forecast_features = np.zeros(len(feature_cols))
            
            # Fill with simple patterns
            if 'month_sin' in feature_cols:
                month = forecast_date.month
                forecast_features[feature_cols.index('month_sin')] = np.sin(2 * np.pi * month / 12)
            if 'month_cos' in feature_cols:
                forecast_features[feature_cols.index('month_cos')] = np.cos(2 * np.pi * month / 12)
            
            # Use last known lag values
            for col in feature_cols:
                if 'lag_1' in col and len(y) > 0:
                    forecast_features[feature_cols.index(col)] = y.iloc[-1]
                elif 'rolling_mean_12' in col and len(y) >= 12:
                    forecast_features[feature_cols.index(col)] = y.iloc[-12:].mean()
            
            # Scale features
            forecast_features_scaled = scaler.transform([forecast_features])
            
            # Ensemble prediction
            ensemble_pred = 0
            for name, model in trained_models.items():
                pred = model.predict(forecast_features_scaled)[0]
                weight = model_weights.get(name, 0)
                ensemble_pred += pred * weight
            
            forecasts.append(max(ensemble_pred, 0))
        
        forecasts = np.array(forecasts)
        
        # Apply scaling
        forecasts *= scaling_factor
        
        return forecasts, np.mean(list(model_weights.values()))
        
    except Exception as e:
        st.warning(f"⚠️ Advanced ML ensemble failed: {str(e)}. Using fallback.")
        return run_enhanced_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_optimized_prophet(data, forecast_periods=12, scaling_factor=1.0):
    """Highly optimized Prophet with advanced parameter tuning"""
    try:
        if not PROPHET_AVAILABLE:
            return run_enhanced_fallback_forecast(data, forecast_periods, scaling_factor), np.inf
        
        # Create enhanced features for Prophet
        enhanced_data = create_advanced_features(data)
        
        # Prepare Prophet data
        prophet_data = enhanced_data[['Month', 'Sales']].rename(columns={'Month': 'ds', 'Sales': 'y'})
        prophet_data['y'] = prophet_data['y'].clip(lower=0.1)
        
        # Optimize Prophet hyperparameters using grid search
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0, 50.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative']
        }
        
        best_mae = np.inf
        best_params = None
        
        # Use only a subset for speed
        if len(prophet_data) >= 36:
            train_size = len(prophet_data) - 12
            train_data = prophet_data.iloc[:train_size]
            val_data = prophet_data.iloc[train_size:]
            
            # Test limited parameter combinations for speed
            test_combinations = [
                {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 
                 'holidays_prior_scale': 1.0, 'seasonality_mode': 'additive'},
                {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1.0,
                 'holidays_prior_scale': 0.1, 'seasonality_mode': 'multiplicative'},
                {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 50.0,
                 'holidays_prior_scale': 10.0, 'seasonality_mode': 'additive'}
            ]
            
            for params in test_combinations:
                try:
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        **params
                    )
                    
                    # Add custom seasonalities
                    model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
                    
                    model.fit(train_data)
                    
                    future = model.make_future_dataframe(periods=12, freq='MS')
                    forecast = model.predict(future)
                    val_pred = forecast['yhat'].tail(12).values
                    
                    mae = mean_absolute_error(val_data['y'].values, val_pred)
                    if mae < best_mae:
                        best_mae = mae
                        best_params = params
                except:
                    continue
        
        # Use best parameters or default
        if best_params is None:
            best_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 1.0,
                'seasonality_mode': 'additive'
            }
        
        # Train final model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            **best_params
        )
        
        # Add advanced seasonalities
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        if len(prophet_data) >= 24:
            model.add_seasonality(name='semi_annual', period=182.5, fourier_order=4)
        
        model.fit(prophet_data)
        
        future = model.make_future_dataframe(periods=forecast_periods, freq='MS')
        forecast = model.predict(future)
        forecast_values = forecast['yhat'].tail(forecast_periods).values
        
        # Post-process forecasts
        forecast_values = np.maximum(forecast_values, 0) * scaling_factor
        
        return forecast_values, best_mae
        
    except Exception as e:
        st.warning(f"⚠️ Optimized Prophet failed: {str(e)}. Using fallback.")
        return run_enhanced_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_enhanced_sarima(data, forecast_periods=12, scaling_factor=1.0):
    """Enhanced SARIMA with automatic parameter optimization"""
    try:
        if not STATSMODELS_AVAILABLE or len(data) < 24:
            return run_enhanced_fallback_forecast(data, forecast_periods, scaling_factor), np.inf
        
        # Enhanced preprocessing
        sales_series = data['Sales'].copy()
        
        # Apply optimal transformation
        transform_type = detect_optimal_transformation(data)
        if transform_type == 'log':
            sales_series = np.log1p(sales_series)
        elif transform_type == 'sqrt':
            sales_series = np.sqrt(sales_series)
        
        # More comprehensive parameter search
        p_values = [0, 1, 2]
        d_values = [0, 1]
        q_values = [0, 1, 2]
        P_values = [0, 1]
        D_values = [0, 1]
        Q_values = [0, 1]
        
        best_aic = np.inf
        best_params = None
        best_seasonal_params = None
        
        # Use AIC for model selection
        for p, d, q in itertools.product(p_values, d_values, q_values):
            for P, D, Q in itertools.product(P_values, D_values, Q_values):
                try:
                    model = SARIMAX(
                        sales_series,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    
                    fitted_model = model.fit(disp=False, maxiter=100)
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        best_seasonal_params = (P, D, Q, 12)
                        
                except:
                    continue
        
        if best_params is None:
            raise ValueError("No suitable SARIMA parameters found")
        
        # Fit final model
        final_model = SARIMAX(
            sales_series,
            order=best_params,
            seasonal_order=best_seasonal_params,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_final = final_model.fit(disp=False, maxiter=200)
        forecast = fitted_final.forecast(steps=forecast_periods)
        
        # Reverse transformation
        if transform_type == 'log':
            forecast = np.expm1(forecast)
        elif transform_type == 'sqrt':
            forecast = np.square(forecast)
        
        forecast = np.maximum(forecast, 0) * scaling_factor
        
        return forecast, best_aic
        
    except Exception as e:
        st.warning(f"⚠️ Enhanced SARIMA failed: {str(e)}. Using fallback.")
        return run_enhanced_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_enhanced_fallback_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Enhanced fallback with multiple sophisticated approaches"""
    try:
        # Method 1: Seasonal naive with exponential smoothing
        if len(data) >= 24:
            # Decompose the series
            decomposition = seasonal_decompose(data['Sales'], model='additive', period=12)
            
            # Extract components
            trend = decomposition.trend.dropna()
            seasonal = decomposition.seasonal.iloc[-12:]  # Last year's seasonal pattern
            
            # Project trend
            recent_trend = np.polyfit(range(len(trend.tail(12))), trend.tail(12), 1)[0]
            
            forecasts = []
            for i in range(forecast_periods):
                trend_component = trend.iloc[-1] + recent_trend * (i + 1)
                seasonal_component = seasonal.iloc[i % 12]
                forecast_val = trend_component + seasonal_component
                forecasts.append(max(forecast_val, data['Sales'].mean() * 0.1))
            
            return np.array(forecasts) * scaling_factor
        
        # Method 2: Exponential smoothing
        elif len(data) >= 12:
            alpha = 0.3  # Smoothing parameter
            seasonal_alpha = 0.1
            
            # Simple exponential smoothing with seasonality
            level = data['Sales'].mean()
            seasonal_factors = np.ones(12)
            
            if len(data) >= 12:
                for i in range(12):
                    month_data = data[data['Month'].dt.month == (i + 1)]['Sales']
                    if len(month_data) > 0:
                        seasonal_factors[i] = month_data.mean() / data['Sales'].mean()
            
            forecasts = []
            for i in range(forecast_periods):
                month_idx = (data['Month'].iloc[-1].month + i) % 12
                forecast_val = level * seasonal_factors[month_idx]
                forecasts.append(max(forecast_val, level * 0.1))
            
            return np.array(forecasts) * scaling_factor
        
        # Method 3: Simple growth projection
        else:
            base_value = data['Sales'].mean()
            growth_rate = 0.02  # 2% monthly growth assumption
            
            forecasts = []
            for i in range(forecast_periods):
                forecast_val = base_value * (1 + growth_rate) ** i
                forecasts.append(forecast_val)
            
            return np.array(forecasts) * scaling_factor
            
    except Exception:
        # Ultimate fallback
        base_value = data['Sales'].mean() if len(data) > 0 else 1000
        return np.array([base_value * scaling_factor] * forecast_periods)


def create_confidence_intervals(forecasts_dict, confidence_level=0.95):
    """Create confidence intervals for forecasts"""
    ensemble_values = []
    
    for model_name, forecast in forecasts_dict.items():
        if '_Forecast' in model_name:
            ensemble_values.append(forecast)
    
    if len(ensemble_values) < 2:
        return None, None
    
    ensemble_array = np.array(ensemble_values)
    
    # Calculate percentiles for confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(ensemble_array, lower_percentile, axis=0)
    upper_bound = np.percentile(ensemble_array, upper_percentile, axis=0)
    
    return lower_bound, upper_bound


# Load existing functions with enhancements
@st.cache_data
def load_data(uploaded_file):
    """Enhanced data loading with automatic preprocessing"""
    try:
        df = pd.read_excel(uploaded_file)
    except Exception:
        st.error("Could not read the uploaded file. Please ensure it's a valid Excel file.")
        return None

    if "Month" not in df.columns or "Sales" not in df.columns:
        st.error("The file must contain 'Month' and 'Sales' columns.")
        return None

    # Parse dates
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    if df["Month"].isna().any():
        st.error("Some dates could not be parsed. Please check the 'Month' column format.")
        return None

    # Clean sales data
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    df["Sales"] = df["Sales"].abs()

    # Sort by date
    df = df.sort_values("Month").reset_index(drop=True)
    
    # Aggregate multiple entries per month
    original_rows = len(df)
    unique_months = df['Month'].nunique()
    
    if original_rows > unique_months:
        st.info(f"📊 Aggregating {original_rows} data points into {unique_months} monthly totals...")
        df_monthly = df.groupby('Month', as_index=False).agg({'Sales': 'sum'}).sort_values('Month').reset_index(drop=True)
        df_monthly['Sales_Original'] = df_monthly['Sales'].copy()
        df_processed = enhanced_preprocess_data(df_monthly)
        st.success(f"✅ Successfully aggregated to {len(df_processed)} monthly data points")
    else:
        df_processed = enhanced_preprocess_data(df)
    
    return df_processed


def enhanced_preprocess_data(df):
    """Enhanced preprocessing with multiple techniques"""
    df['Sales_Original'] = df['Sales'].copy()
    
    # 1. Advanced outlier detection
    outliers = advanced_outlier_detection(df)
    if outliers.sum() > 0:
        st.info(f"📊 Detected and treated {outliers.sum()} outliers using Isolation Forest")
        # Cap outliers instead of removing
        Q1, Q3 = df['Sales'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df.loc[outliers, 'Sales'] = df.loc[outliers, 'Sales'].clip(lower=lower_bound, upper=upper_bound)
    
    # 2. Missing value treatment
    if df['Sales'].isna().any():
        df['Sales'] = df['Sales'].interpolate(method='time')
    
    # 3. Optimal transformation
    transform_type = detect_optimal_transformation(df)
    if transform_type != 'none':
        st.info(f"📈 Applying {transform_type} transformation for better model performance")
        if transform_type == 'log':
            df['Sales'] = np.log1p(df['Sales'])
        elif transform_type == 'sqrt':
            df['Sales'] = np.sqrt(df['Sales'])
        df['transform_applied'] = transform_type
    else:
        df['transform_applied'] = 'none'
    
    return df


# Enhanced main function with all improvements
def main():
    """Enhanced main function with advanced accuracy features"""
    st.title("🎯 Enhanced Accuracy Sales Forecasting Dashboard")
    st.markdown("**Advanced algorithmic improvements for maximum accuracy without external data**")

    # Sidebar configuration
    st.sidebar.header("⚙️ Enhanced Configuration")
    forecast_year = st.sidebar.selectbox("Select forecast year:", options=[2024, 2025, 2026], index=0)

    # Advanced algorithmic options
    st.sidebar.subheader("🎯 Accuracy Enhancement Options")
    enable_advanced_features = st.sidebar.checkbox("Advanced Feature Engineering", value=True, 
                                                   help="Create lag, rolling, trend, and seasonal features")
    enable_ml_ensemble = st.sidebar.checkbox("ML Ensemble Models", value=True,
                                            help="Random Forest, Gradient Boosting, Ridge, Elastic Net")
    enable_optimized_prophet = st.sidebar.checkbox("Optimized Prophet", value=True,
                                                   help="Hyperparameter tuned Prophet with custom seasonalities")
    enable_enhanced_sarima = st.sidebar.checkbox("Enhanced SARIMA", value=True,
                                                 help="Comprehensive parameter search with transformations")
    enable_confidence_intervals = st.sidebar.checkbox("Confidence Intervals", value=True,
                                                      help="Calculate prediction uncertainty bands")
    
    # Advanced preprocessing options
    st.sidebar.subheader("🔬 Advanced Preprocessing")
    outlier_method = st.sidebar.selectbox(
        "Outlier Detection Method:",
        ["isolation_forest", "lof", "iqr"],
        help="Method for detecting and treating outliers"
    )
    
    transformation_mode = st.sidebar.selectbox(
        "Transformation Mode:",
        ["auto", "none", "log", "sqrt"],
        help="Data transformation approach"
    )

    # File uploads
    st.subheader("📁 Upload Data Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        historical_file = st.file_uploader(
            "📊 Upload Historical Sales Data",
            type=["xlsx", "xls"],
            help="Excel file with 'Month' and 'Sales' columns"
        )
    
    with col2:
        actual_2024_file = st.file_uploader(
            f"📈 Upload {forecast_year} Actual Data (Optional)",
            type=["xlsx", "xls"],
            help="For validation and adaptive learning"
        )

    if historical_file is None:
        st.info("👆 Please upload historical sales data to begin enhanced forecasting.")
        return

    # Load and validate historical data
    hist_df = load_data(historical_file)
    if hist_df is None:
        return

    # Load actual data for validation
    actual_2024_df = None
    scaling_factor = 1.0
    
    if actual_2024_file is not None:
        actual_2024_df = load_actual_2024_data(actual_2024_file, forecast_year)
        if actual_2024_df is not None:
            scaling_factor = detect_and_apply_scaling(hist_df, actual_2024_df)

    # Enhanced data analysis
    st.subheader("📊 Enhanced Data Analysis")
    
    # Create advanced features for analysis
    if enable_advanced_features:
        enhanced_hist_df = create_advanced_features(hist_df)
        st.info(f"🚀 Created {len(enhanced_hist_df.columns) - len(hist_df.columns)} additional features from existing data")
    else:
        enhanced_hist_df = hist_df

    # Display metrics
    unique_months = hist_df['Month'].nunique()
    avg_monthly_sales = hist_df.groupby('Month')['Sales'].sum().mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 Total Months", unique_months)
    with col2:
        st.metric("📈 Avg Monthly Sales", f"{avg_monthly_sales:,.0f}")
    with col3:
        data_quality = min(100, unique_months * 4.17)
        st.metric("🎯 Data Quality Score", f"{data_quality:.0f}%")
    with col4:
        transform_applied = hist_df.get('transform_applied', ['none'])[0] if len(hist_df) > 0 else 'none'
        st.metric("🔧 Transformation", transform_applied.title())

    # Advanced data insights
    with st.expander("🔍 Advanced Data Insights"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Seasonality strength
            if len(hist_df) >= 24:
                try:
                    monthly_data = hist_df.groupby('Month')['Sales'].sum().reset_index()
                    decomposition = seasonal_decompose(monthly_data['Sales'], model='additive', period=12)
                    seasonal_strength = np.var(decomposition.seasonal) / np.var(monthly_data['Sales'])
                    st.metric("📊 Seasonality Strength", f"{seasonal_strength:.2%}")
                except:
                    st.metric("📊 Seasonality", "Analysis failed")
            else:
                st.metric("📊 Seasonality", "Need 24+ months")
        
        with col2:
            # Trend analysis
            if len(hist_df) >= 12:
                try:
                    recent_data = hist_df['Sales'].tail(12)
                    trend_slope = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
                    trend_direction = "📈 Increasing" if trend_slope > 0 else "📉 Decreasing"
                    st.metric("📈 Recent Trend", trend_direction)
                except:
                    st.metric("📈 Recent Trend", "Analysis failed")
            else:
                st.metric("📈 Recent Trend", "Need 12+ months")
        
        with col3:
            # Data stationarity
            try:
                adf_result = adfuller(hist_df['Sales'].dropna())
                is_stationary = adf_result[1] < 0.05
                stationarity_text = "✅ Stationary" if is_stationary else "❌ Non-stationary"
                st.metric("📈 Stationarity", stationarity_text)
            except:
                st.metric("📈 Stationarity", "Analysis failed")
        
        # Feature importance preview
        if enable_advanced_features and len(enhanced_hist_df.columns) > len(hist_df.columns):
            st.subheader("🎯 Generated Features Preview")
            feature_cols = [col for col in enhanced_hist_df.columns 
                           if col not in ['Month', 'Sales', 'Sales_Original', 'transform_applied']]
            
            if len(feature_cols) > 0:
                # Show correlation with target
                correlations = []
                for col in feature_cols[:10]:  # Show top 10
                    try:
                        corr = enhanced_hist_df[col].corr(enhanced_hist_df['Sales'])
                        if not np.isnan(corr):
                            correlations.append({'Feature': col, 'Correlation': f"{corr:.3f}"})
                    except:
                        continue
                
                if correlations:
                    corr_df = pd.DataFrame(correlations)
                    st.dataframe(corr_df, use_container_width=True)

    # Enhanced forecasting
    if st.button("🎯 Generate Enhanced Accuracy Forecasts", type="primary"):
        st.subheader("🎯 Generating Enhanced Accuracy Forecasts...")
        
        st.info("🚀 Using advanced algorithmic improvements for maximum accuracy")
        
        progress_bar = st.progress(0)
        forecast_results = {}
        validation_scores = {}
        
        # Create forecast dates
        forecast_dates = pd.date_range(
            start=f"{forecast_year}-01-01",
            end=f"{forecast_year}-12-01",
            freq='MS'
        )
        
        # Initialize adaptive ensemble
        adaptive_ensemble = AdaptiveEnsemble()
        
        # Run enhanced models
        models_to_run = []
        
        if enable_ml_ensemble:
            models_to_run.append(("ML_Ensemble", run_advanced_ml_ensemble))
        if enable_optimized_prophet:
            models_to_run.append(("Optimized_Prophet", run_optimized_prophet))
        if enable_enhanced_sarima:
            models_to_run.append(("Enhanced_SARIMA", run_enhanced_sarima))
        
        # Always include enhanced fallback
        models_to_run.append(("Enhanced_Fallback", run_enhanced_fallback_forecast))
        
        # Cross-validation setup
        cv_validator = AdvancedCrossValidation()
        
        for i, (model_name, model_func) in enumerate(models_to_run):
            with st.spinner(f"🤖 Running {model_name} with advanced optimizations..."):
                try:
                    # Run cross-validation first
                    cv_errors, cv_score = cv_validator.time_series_cv(
                        enhanced_hist_df if enable_advanced_features else hist_df, 
                        model_func
                    )
                    
                    if cv_score != np.inf:
                        st.info(f"📊 {model_name} CV Score: {cv_score:.2f}")
                    
                    # Generate actual forecast
                    result = model_func(
                        enhanced_hist_df if enable_advanced_features else hist_df, 
                        forecast_periods=12, 
                        scaling_factor=scaling_factor
                    )
                    
                    if isinstance(result, tuple):
                        forecast_values, model_score = result
                    else:
                        forecast_values = result
                        model_score = cv_score
                    
                    # Validate forecast
                    if (isinstance(forecast_values, (list, np.ndarray)) and 
                        len(forecast_values) == 12 and 
                        not np.all(forecast_values == 0) and
                        not np.any(np.isnan(forecast_values))):
                        
                        forecast_results[f"{model_name}_Forecast"] = np.array(forecast_values)
                        validation_scores[model_name] = model_score
                        
                        # Show results
                        min_val, max_val = np.min(forecast_values), np.max(forecast_values)
                        total_val = np.sum(forecast_values)
                        st.success(f"✅ {model_name}: Range {min_val:,.0f}-{max_val:,.0f}, Total: {total_val:,.0f}")
                        
                    else:
                        st.warning(f"⚠️ {model_name} produced invalid results. Skipping.")
                        
                except Exception as e:
                    st.error(f"❌ {model_name} failed: {str(e)}")
            
            progress_bar.progress((i + 1) / len(models_to_run))
        
        if not forecast_results:
            st.error("❌ All enhanced models failed. Please check your data.")
            return
        
        # Create intelligent ensemble
        if len(forecast_results) > 1:
            with st.spinner("🔥 Creating intelligent weighted ensemble..."):
                try:
                    # Calculate weights based on validation scores
                    weights = {}
                    total_inverse_score = 0
                    
                    for model_name, score in validation_scores.items():
                        if score != np.inf and score > 0:
                            inverse_score = 1 / score
                            weights[model_name] = inverse_score
                            total_inverse_score += inverse_score
                        else:
                            weights[model_name] = 0.1
                            total_inverse_score += 0.1
                    
                    # Normalize weights
                    for model_name in weights:
                        weights[model_name] = weights[model_name] / total_inverse_score
                    
                    # Create weighted ensemble
                    ensemble_forecast = np.zeros(12)
                    for model_name, forecast in forecast_results.items():
                        model_key = model_name.replace('_Forecast', '')
                        weight = weights.get(model_key, 0.25)
                        ensemble_forecast += weight * forecast
                    
                    forecast_results["Intelligent_Ensemble"] = ensemble_forecast
                    
                    # Show weights
                    weight_text = ', '.join([f'{k}: {v:.1%}' for k, v in weights.items()])
                    st.info(f"🎯 Intelligent weights: {weight_text}")
                    
                except Exception as e:
                    st.warning(f"⚠️ Ensemble creation failed: {str(e)}")
        
        # Calculate confidence intervals
        if enable_confidence_intervals:
            with st.spinner("📊 Calculating confidence intervals..."):
                try:
                    lower_bound, upper_bound = create_confidence_intervals(forecast_results)
                    if lower_bound is not None:
                        forecast_results["Lower_95%_CI"] = lower_bound
                        forecast_results["Upper_95%_CI"] = upper_bound
                        st.success("✅ 95% confidence intervals calculated")
                except Exception as e:
                    st.warning(f"⚠️ Confidence interval calculation failed: {str(e)}")
        
        # Create results dataframe
        result_df = pd.DataFrame({
            "Month": forecast_dates,
            **forecast_results
        })
        
        # Merge actual data if available
        if actual_2024_df is not None:
            actual_2024_df['Month'] = pd.to_datetime(actual_2024_df['Month'])
            result_df['Month'] = pd.to_datetime(result_df['Month'])
            result_df = result_df.merge(actual_2024_df, on="Month", how="left")
            
            actual_count = result_df[f'Actual_{forecast_year}'].notna().sum()
            st.success(f"📊 Loaded {actual_count} months of actual data for validation")
        
        # Display enhanced results
        st.subheader("🎯 Enhanced Accuracy Results")
        
        # Enhanced forecast summary
        summary_data = []
        for model_name, forecast_values in forecast_results.items():
            if '_Forecast' in model_name or model_name == 'Intelligent_Ensemble':
                if isinstance(forecast_values, (list, np.ndarray)):
                    forecast_array = np.array(forecast_values)
                    cv_score = validation_scores.get(model_name.replace('_Forecast', ''), 'N/A')
                    cv_text = f"{cv_score:.2f}" if cv_score != np.inf and cv_score != 'N/A' else 'N/A'
                    
                    summary_data.append({
                        'Model': model_name.replace('_Forecast', '').replace('_', ' '),
                        'Annual Total': f"{np.sum(forecast_array):,.0f}",
                        'Monthly Avg': f"{np.mean(forecast_array):,.0f}",
                        'Min Month': f"{np.min(forecast_array):,.0f}",
                        'Max Month': f"{np.max(forecast_array):,.0f}",
                        'CV Score': cv_text,
                        'Coefficient of Variation': f"{np.std(forecast_array)/np.mean(forecast_array):.2%}"
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        
        # Enhanced visualization
        st.subheader("📊 Enhanced Forecast Visualization")
        
        # Create advanced chart
        fig = go.Figure()
        
        # Add actual data if available
        actual_col = f'Actual_{forecast_year}'
        if actual_col in result_df.columns and result_df[actual_col].notna().any():
            actual_data = result_df[result_df[actual_col].notna()]
            fig.add_trace(go.Scatter(
                x=actual_data['Month'],
                y=actual_data[actual_col],
                mode='lines+markers',
                name='🎯 ACTUAL',
                line=dict(color='#FF6B6B', width=4),
                marker=dict(size=12, symbol='circle')
            ))
        
        # Add confidence intervals if available
        if 'Lower_95%_CI' in result_df.columns and 'Upper_95%_CI' in result_df.columns:
            fig.add_trace(go.Scatter(
                x=result_df['Month'],
                y=result_df['Upper_95%_CI'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=result_df['Month'],
                y=result_df['Lower_95%_CI'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(width=0),
                name='📊 95% Confidence Band',
                hoverinfo='skip'
            ))
        
        # Add forecast lines
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF9F43']
        model_cols = [col for col in result_df.columns if '_Forecast' in col or col == 'Intelligent_Ensemble']
        
        for i, col in enumerate(model_cols):
            if col == 'Intelligent_Ensemble':
                line_style = dict(color='#6C5CE7', width=4, dash='dash')
                icon = '🧠'
            else:
                line_style = dict(color=colors[i % len(colors)], width=2)
                icon = '🤖'
            
            model_name = col.replace('_Forecast', '').replace('_', ' ').upper()
            fig.add_trace(go.Scatter(
                x=result_df['Month'],
                y=result_df[col],
                mode='lines+markers',
                name=f'{icon} {model_name}',
                line=line_style,
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title='🎯 ENHANCED ACCURACY FORECASTING RESULTS',
            xaxis_title='Month',
            yaxis_title='Sales Volume',
            height=700,
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance analysis
        if actual_col in result_df.columns and result_df[actual_col].notna().any():
            st.subheader("🎯 Enhanced Performance Analysis")
            
            actual_data = result_df[result_df[actual_col].notna()]
            performance_data = []
            
            for col in model_cols:
                model_name = col.replace('_Forecast', '').replace('_', ' ')
                
                # Calculate enhanced metrics
                actual_values = actual_data[actual_col].values
                forecast_values = actual_data[col].values
                
                try:
                    mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
                    mae = mean_absolute_error(actual_values, forecast_values)
                    rmse = np.sqrt(mean_squared_error(actual_values, forecast_values))
                    smape = np.mean(2 * np.abs(forecast_values - actual_values) / 
                                   (np.abs(actual_values) + np.abs(forecast_values))) * 100
                    
                    # Enhanced metrics
                    bias = np.mean(forecast_values - actual_values)
                    bias_pct = (bias / np.mean(actual_values)) * 100
                    
                    cv_score = validation_scores.get(model_name.replace(' ', '_'), 'N/A')
                    cv_text = f"{cv_score:.2f}" if cv_score != np.inf and cv_score != 'N/A' else 'N/A'
                    
                    performance_data.append({
                        'Model': model_name,
                        'MAPE (%)': f"{mape:.1f}%",
                        'SMAPE (%)': f"{smape:.1f}%",
                        'MAE': f"{mae:,.0f}",
                        'RMSE': f"{rmse:,.0f}",
                        'Bias': f"{bias:,.0f}",
                        'Bias (%)': f"{bias_pct:+.1f}%",
                        'CV Score': cv_text,
                        'Accuracy': f"{100 - mape:.1f}%"
                    })
                    
                except Exception as e:
                    continue
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                st.dataframe(perf_df, use_container_width=True)
                
                # Highlight best model
                best_idx = perf_df['MAPE (%)'].str.replace('%', '').astype(float).idxmin()
                best_model = perf_df.iloc[best_idx]
                st.success(f"🏆 Best Model: **{best_model['Model']}** with {best_model['MAPE (%)']} MAPE")
        
        # Enhanced export
        st.subheader("📊 Enhanced Analytics Export")
        
        @st.cache_data
        def create_enhanced_excel_report(result_df, enhanced_hist_df, forecast_year, validation_scores):
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Sheet 1: Enhanced Results
                export_df = result_df.copy()
                export_df['Month'] = export_df['Month'].dt.strftime('%Y-%m-%d')
                export_df.to_excel(writer, sheet_name='Enhanced_Results', index=False)
                
                # Sheet 2: Model Performance
                if summary_data:
                    summary_export_df = pd.DataFrame(summary_data)
                    summary_export_df.to_excel(writer, sheet_name='Model_Performance', index=False)
                
                # Sheet 3: Validation Scores
                if validation_scores:
                    val_df = pd.DataFrame([
                        {'Model': k, 'Validation_Score': v if v != np.inf else 'Failed'}
                        for k, v in validation_scores.items()
                    ])
                    val_df.to_excel(writer, sheet_name='Validation_Scores', index=False)
                
                # Sheet 4: Feature Analysis
                if enable_advanced_features:
                    feature_cols = [col for col in enhanced_hist_df.columns 
                                   if col not in ['Month', 'Sales', 'Sales_Original', 'transform_applied']]
                    if feature_cols:
                        feature_analysis = []
                        for col in feature_cols:
                            try:
                                corr = enhanced_hist_df[col].corr(enhanced_hist_df['Sales'])
                                feature_analysis.append({
                                    'Feature': col,
                                    'Correlation_with_Sales': corr if not np.isnan(corr) else 'N/A',
                                    'Mean': enhanced_hist_df[col].mean(),
                                    'Std': enhanced_hist_df[col].std()
                                })
                            except:
                                continue
                        
                        if feature_analysis:
                            feature_df = pd.DataFrame(feature_analysis)
                            feature_df.to_excel(writer, sheet_name='Feature_Analysis', index=False)
            
            output.seek(0)
            return output
        
        # Generate enhanced report
        excel_data = create_enhanced_excel_report(result_df, enhanced_hist_df, forecast_year, validation_scores)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="🎯 Download Enhanced Analytics Report",
                data=excel_data,
                file_name=f"enhanced_accuracy_forecast_{forecast_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Download CSV Data",
                data=csv,
                file_name=f"enhanced_forecasts_{forecast_year}.csv",
                mime="text/csv"
            )
        
        # Final summary
        st.subheader("🎯 Enhanced Accuracy Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'Intelligent_Ensemble' in result_df.columns:
                ensemble_total = result_df['Intelligent_Ensemble'].sum()
                st.metric("🧠 Intelligent Ensemble", f"{ensemble_total:,.0f}")
        
        with col2:
            successful_models = len([v for v in validation_scores.values() if v != np.inf])
            st.metric("🤖 Successful Models", f"{successful_models}/{len(validation_scores)}")
        
        with col3:
            if enable_advanced_features:
                feature_count = len(enhanced_hist_df.columns) - len(hist_df.columns)
                st.metric("🎯 Features Created", f"{feature_count}")
            else:
                st.metric("🎯 Features", "Basic")
        
        with col4:
            if enable_confidence_intervals and 'Lower_95%_CI' in result_df.columns:
                st.metric("📊 Confidence Intervals", "✅ Available")
            else:
                st.metric("📊 Confidence Intervals", "Not Available")


@st.cache_data
def load_actual_2024_data(uploaded_file, forecast_year):
    """Load actual data for validation"""
    try:
        df = pd.read_excel(uploaded_file)
        
        if "Month" in df.columns and "Sales" in df.columns:
            df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
            df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
            df["Sales"] = df["Sales"].abs()
            
            start = pd.Timestamp(f"{forecast_year}-01-01")
            end = pd.Timestamp(f"{forecast_year+1}-01-01")
            df = df[(df["Month"] >= start) & (df["Month"] < end)]
            
            if df.empty:
                return None
            
            monthly = df.groupby("Month", as_index=False)["Sales"].sum()
            monthly = monthly[monthly["Sales"] > 0]
            monthly = monthly.sort_values("Month").reset_index(drop=True)
            
            return monthly.rename(columns={"Sales": f"Actual_{forecast_year}"})
    except Exception:
        return None


def detect_and_apply_scaling(historical_data, actual_data=None):
    """Detect and apply scaling between historical and actual data"""
    hist_avg = historical_data['Sales'].mean()
    
    if actual_data is not None and len(actual_data) > 0:
        actual_avg = actual_data.iloc[:, 1].mean()
        ratio = actual_avg / hist_avg if hist_avg > 0 else 1
        
        if ratio > 2 or ratio < 0.5:
            st.warning(f"📊 Scale mismatch detected! Scaling factor: {ratio:.2f}")
            return ratio
    
    return 1.0


if __name__ == "__main__":
    main()
