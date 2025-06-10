import streamlit as st

# Configure streamlit FIRST - must be before any other st commands
st.set_page_config(page_title="Advanced AI Sales Forecasting System", layout="wide")

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from scipy import stats
from scipy.optimize import minimize
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib

# Forecasting libraries
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.forecasting.theta import ThetaModel

# Machine learning libraries
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# Try to import optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedMetaLearner(BaseEstimator, RegressorMixin):
    """Advanced meta-learner with multiple stacking options"""
    def __init__(self, meta_model='ridge', cv_folds=5):
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.model = None
        self.feature_importance = None
        
    def fit(self, X, y):
        if self.meta_model == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.meta_model == 'lasso':
            self.model = Lasso(alpha=0.1)
        elif self.meta_model == 'elastic':
            self.model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif self.meta_model == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.model.fit(X, y)
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_)
            
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self):
        return self.feature_importance


def optimize_dtypes(df):
    """Reduce memory usage by optimizing data types"""
    initial_memory = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object' and col != 'Month':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    final_memory = df.memory_usage().sum() / 1024**2
    memory_reduction = (initial_memory - final_memory) / initial_memory * 100
    
    if memory_reduction > 0:
        st.info(f"💾 Memory optimized: {initial_memory:.2f} MB → {final_memory:.2f} MB ({memory_reduction:.1f}% reduction)")
    
    return df


@st.cache_data(ttl=3600)
def load_data_optimized(file_content, file_hash):
    """Load and preprocess data with memory optimization"""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
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
    
    # Optimize data types
    df = optimize_dtypes(df)
    
    # Check if there are multiple entries per month
    original_rows = len(df)
    unique_months = df['Month'].nunique()
    
    if original_rows > unique_months:
        st.info(f"📊 Aggregating {original_rows} data points into {unique_months} monthly totals...")
        
        # Aggregate by month
        df_monthly = df.groupby('Month', as_index=False).agg({
            'Sales': 'sum'
        }).sort_values('Month').reset_index(drop=True)
        
        df_monthly['Sales_Original'] = df_monthly['Sales'].copy()
        
        # Advanced preprocessing
        df_processed = advanced_preprocess_data(df_monthly)
        
        st.success(f"✅ Successfully aggregated to {len(df_processed)} monthly data points")
        
    else:
        df_processed = advanced_preprocess_data(df)
    
    # Force garbage collection
    gc.collect()
    
    return df_processed


def advanced_preprocess_data(df):
    """Enhanced data preprocessing with multiple techniques"""
    df = df.copy()
    df['Sales_Original'] = df['Sales'].copy()
    
    # 1. Advanced Outlier Detection (multiple methods)
    outlier_methods = []
    
    # IQR method
    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = ((df['Sales'] < lower_bound) | (df['Sales'] > upper_bound))
    outlier_methods.append(iqr_outliers)
    
    # Z-score method
    z_scores = np.abs(stats.zscore(df['Sales']))
    zscore_outliers = z_scores > 3
    outlier_methods.append(zscore_outliers)
    
    # Isolation Forest (if enough data)
    if len(df) >= 20:
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_pred = iso_forest.fit_predict(df['Sales'].values.reshape(-1, 1))
        iso_outliers = outlier_pred == -1
        outlier_methods.append(iso_outliers)
    
    # Combine outlier detection methods (majority vote)
    outliers = np.sum(outlier_methods, axis=0) >= len(outlier_methods) / 2
    outliers_detected = outliers.sum()
    
    if outliers_detected > 0:
        st.info(f"📊 Detected {outliers_detected} outliers using ensemble method")
        # Use Winsorization instead of hard capping
        df.loc[outliers, 'Sales'] = df.loc[~outliers, 'Sales'].quantile(0.95)
    
    # 2. Handle missing values with advanced interpolation
    if df['Sales'].isna().any():
        # Try multiple interpolation methods
        df['Sales'] = df['Sales'].interpolate(method='time')
        # Fill any remaining NaNs with seasonal average
        month_avg = df.groupby(df['Month'].dt.month)['Sales'].transform('mean')
        df['Sales'] = df['Sales'].fillna(month_avg)
    
    # 3. Detect and handle structural breaks
    if len(df) >= 24:
        from statsmodels.tsa.stattools import adfuller
        # Check for stationarity
        adf_result = adfuller(df['Sales'])
        if adf_result[1] > 0.05:  # Non-stationary
            st.info("📈 Non-stationary data detected. Applying differencing.")
            df['needs_differencing'] = True
        else:
            df['needs_differencing'] = False
    
    # 4. Advanced transformation selection
    transformations = {
        'none': df['Sales'].copy(),
        'log': np.log1p(df['Sales']),
        'sqrt': np.sqrt(df['Sales']),
        'boxcox': stats.boxcox(df['Sales'] + 1)[0] if (df['Sales'] > 0).all() else df['Sales']
    }
    
    # Select best transformation based on normality
    best_transform = 'none'
    best_normality = 0
    
    for transform_name, transformed_data in transformations.items():
        try:
            _, p_value = stats.normaltest(transformed_data)
            if p_value > best_normality:
                best_normality = p_value
                best_transform = transform_name
        except:
            continue
    
    if best_transform != 'none':
        st.info(f"📊 Applied {best_transform} transformation for better modeling")
        df['Sales'] = transformations[best_transform]
        df['transformation'] = best_transform
        df['transformation_params'] = {'method': best_transform}
        
        if best_transform == 'boxcox':
            df['transformation_params']['lambda'] = stats.boxcox(df['Sales_Original'] + 1)[1]
    else:
        df['transformation'] = 'none'
        df['transformation_params'] = {'method': 'none'}
    
    # 5. Add cyclical encoding for months
    df['month'] = df['Month'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def create_advanced_features(df):
    """Create comprehensive features for ML models"""
    df = df.copy()
    
    # Time features
    df['year'] = df['Month'].dt.year
    df['quarter'] = df['Month'].dt.quarter
    df['dayofyear'] = df['Month'].dt.dayofyear
    df['weekofyear'] = df['Month'].dt.isocalendar().week
    
    # Lag features (multiple lags)
    lag_features = [1, 2, 3, 6, 12, 24] if len(df) > 24 else [1, 3, 6, 12]
    for lag in lag_features:
        if lag < len(df):
            df[f'lag_{lag}'] = df['Sales'].shift(lag)
    
    # Rolling statistics (multiple windows)
    windows = [3, 6, 12, 24] if len(df) > 24 else [3, 6, 12]
    for window in windows:
        if window < len(df):
            df[f'rolling_mean_{window}'] = df['Sales'].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['Sales'].rolling(window=window, min_periods=1).std()
            df[f'rolling_min_{window}'] = df['Sales'].rolling(window=window, min_periods=1).min()
            df[f'rolling_max_{window}'] = df['Sales'].rolling(window=window, min_periods=1).max()
            
            # Exponentially weighted statistics
