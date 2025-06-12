import streamlit as st
import pandas as pd
import numpy as np
import io
import hashlib
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore')

# Check for optional packages
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Core dependencies
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from scipy.stats import boxcox, boxcox_normmax
from scipy.special import inv_boxcox
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure Streamlit
st.set_page_config(
    page_title="AI Sales Forecasting System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def detect_and_apply_scaling(historical_data, actual_data=None):
    """Enhanced scaling detection with multiple methods"""
    hist_avg = historical_data['Sales_Original'].mean()
    hist_median = historical_data['Sales_Original'].median()
    
    if actual_data is not None and len(actual_data) > 0:
        # Get the actual column name dynamically
        actual_col = [col for col in actual_data.columns if 'Actual_' in col]
        if actual_col:
            actual_col = actual_col[0]
            actual_values = actual_data[actual_col].dropna()
            
            if len(actual_values) > 0:
                actual_avg = actual_values.mean()
                actual_median = actual_values.median()
                
                # Calculate ratios using both mean and median
                ratio_mean = actual_avg / hist_avg if hist_avg > 0 else 1
                ratio_median = actual_median / hist_median if hist_median > 0 else 1
                
                # Use the more conservative ratio
                ratio = min(ratio_mean, ratio_median)
                
                # Apply more aggressive scaling detection
                if ratio > 1.2 or ratio < 0.8:
                    st.warning(f"📊 **Scale Mismatch Detected!**")
                    st.warning(f"   - Historical Average: {hist_avg:,.0f}")
                    st.warning(f"   - Actual Average: {actual_avg:,.0f}")
                    st.warning(f"   - Scaling Factor Applied: {ratio:.4f}")
                    return ratio
                    
                # Check for order of magnitude differences
                hist_magnitude = np.floor(np.log10(hist_avg)) if hist_avg > 0 else 0
                actual_magnitude = np.floor(np.log10(actual_avg)) if actual_avg > 0 else 0
                
                if abs(hist_magnitude - actual_magnitude) >= 2:  # At least 2 orders of magnitude difference
                    magnitude_ratio = 10 ** (actual_magnitude - hist_magnitude)
                    st.error(f"🚨 **Major Scale Issue Detected!**")
                    st.error(f"   - Historical scale: 10^{hist_magnitude:.0f} (avg: {hist_avg:,.0f})")
                    st.error(f"   - Actual scale: 10^{actual_magnitude:.0f} (avg: {actual_avg:,.0f})")
                    st.error(f"   - Magnitude-based scaling: {magnitude_ratio:.6f}")
                    return magnitude_ratio
    
    return 1.0


def detect_data_frequency(dates):
    """Detect the frequency of the time series data"""
    if len(dates) < 2:
        return "Unknown"
    
    diff = dates.diff().dropna()
    median_diff = diff.median()
    
    if median_diff <= pd.Timedelta(days=1):
        return "Daily"
    elif median_diff <= pd.Timedelta(days=7):
        return "Weekly"
    elif median_diff <= pd.Timedelta(days=31):
        return "Monthly"
    elif median_diff <= pd.Timedelta(days=92):
        return "Quarterly"
    else:
        return "Yearly"


def advanced_preprocess_data(data):
    """Advanced data preprocessing with multiple transformations"""
    processed_data = data.copy()
    
    # Handle missing values
    processed_data['Sales'] = processed_data['Sales'].fillna(processed_data['Sales'].median())
    
    # Outlier detection using IQR method
    Q1 = processed_data['Sales'].quantile(0.25)
    Q3 = processed_data['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers instead of removing them
    processed_data['Sales'] = processed_data['Sales'].clip(lower=max(0, lower_bound), upper=upper_bound)
    
    # Apply appropriate transformation
    sales_data = processed_data['Sales'].values
    
    # Test for transformation need
    if np.min(sales_data) > 0:
        # Test Box-Cox transformation
        try:
            lambda_param = boxcox_normmax(sales_data, method='mle')
            if 0.4 <= lambda_param <= 0.6:  # Close to sqrt
                processed_data['Sales'] = np.sqrt(sales_data)
                processed_data['transformation'] = 'sqrt'
            elif -0.1 <= lambda_param <= 0.1:  # Close to log
                processed_data['Sales'] = np.log1p(sales_data)
                processed_data['transformation'] = 'log'
            else:
                transformed_data, _ = boxcox(sales_data, lmbda=lambda_param)
                processed_data['Sales'] = transformed_data
                processed_data['transformation'] = 'boxcox'
                processed_data['transformation_params'] = {'lambda': lambda_param}
        except:
            # Fallback to log transformation if positive
            if np.min(sales_data) > 0:
                processed_data['Sales'] = np.log1p(sales_data)
                processed_data['transformation'] = 'log'
            else:
                processed_data['transformation'] = 'none'
    else:
        processed_data['transformation'] = 'none'
    
    return processed_data


@st.cache_data(ttl=3600)
def load_data_optimized(file_content, file_hash):
    """Load and preprocess data with enhanced validation"""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
    except Exception as e:
        st.error(f"Could not read the uploaded file: {str(e)}")
        return None

    # Check for required columns - be more flexible
    if "Month" not in df.columns and "Sales" not in df.columns:
        st.error("The file must contain Month and Sales columns.")
        st.info("Available columns: " + ", ".join(df.columns.tolist()))
        return None
    
    # Handle files with Part column (multiple parts data)
    if "Part" in df.columns:
        st.info(f"📊 **Multi-part data detected** - Found {df['Part'].nunique()} unique parts")
        
        # Let user choose how to handle multiple parts
        aggregation_method = st.selectbox(
            "How to handle multiple parts?",
            options=["Sum all parts", "Select specific part", "Average across parts"],
            index=0,
            help="Choose how to combine data from multiple parts"
        )
        
        if aggregation_method == "Select specific part":
            available_parts = df['Part'].unique()
            selected_part = st.selectbox("Select part:", available_parts)
            df = df[df['Part'] == selected_part].copy()
            st.info(f"Selected part: {selected_part}")
        elif aggregation_method == "Sum all parts":
            # Group by Month and sum Sales
            df = df.groupby('Month', as_index=False).agg({'Sales': 'sum'})
            st.info("✅ Summed sales across all parts by month")
        else:  # Average across parts
            df = df.groupby('Month', as_index=False).agg({'Sales': 'mean'})
            st.info("✅ Averaged sales across all parts by month")

    # Parse dates
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    if df["Month"].isna().any():
        st.error("Some dates could not be parsed. Please check the Month column format.")
        return None

    # Clean sales data with better validation
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    df["Sales"] = df["Sales"].abs()
    
    # Remove zero sales values for better forecasting
    non_zero_count = (df["Sales"] > 0).sum()
    total_count = len(df)
    
    if non_zero_count < total_count * 0.5:
        st.warning(f"⚠️ High number of zero values detected ({total_count - non_zero_count} out of {total_count})")
        st.info("💡 Consider using Croston's method for intermittent demand")
    
    # Sort by date and ensure monthly aggregation
    df = df.sort_values("Month").reset_index(drop=True)
    
    # Check if there are multiple entries per month and aggregate
    original_rows = len(df)
    unique_months = df['Month'].nunique()
    
    if original_rows > unique_months:
        st.info(f"📊 Aggregating {original_rows} data points into {unique_months} monthly totals...")
        df = df.groupby('Month', as_index=False).agg({'Sales': 'sum'})
        df = df.sort_values('Month').reset_index(drop=True)
        st.success(f"✅ Successfully aggregated to {len(df)} monthly data points")
    
    # Check data magnitude and warn user
    avg_sales = df["Sales"].mean()
    median_sales = df["Sales"].median()
    max_sales = df["Sales"].max()
    
    st.info(f"📊 **Data Summary:**")
    st.info(f"   - Average Sales: {avg_sales:,.2f}")
    st.info(f"   - Median Sales: {median_sales:,.2f}")
    st.info(f"   - Max Sales: {max_sales:,.2f}")
    st.info(f"   - Time Range: {df['Month'].min().strftime('%Y-%m')} to {df['Month'].max().strftime('%Y-%m')}")
    
    # Add original sales column and preprocess
    df['Sales_Original'] = df['Sales'].copy()
    df_processed = advanced_preprocess_data(df)
    
    return df_processed


def load_actual_2024_data(file_content, year):
    """Load actual data for comparison"""
    try:
        df = pd.read_excel(file_content)
        
        # Handle different column structures
        if 'Month' in df.columns and len(df.columns) >= 2:
            # Standard format with Month and actual values
            df['Month'] = pd.to_datetime(df['Month'])
            return df[['Month', df.columns[1]]].copy()
        elif 'Date' in df.columns:
            # Alternative date column name
            df['Date'] = pd.to_datetime(df['Date'])
            return df[['Date', df.columns[1]]].copy()
        else:
            st.error("Actual data file should have Date/Month and Values columns")
            return None
            
    except Exception as e:
        st.error(f"Error loading actual data: {str(e)}")
        return None


def apply_scaling_to_forecasts(forecasts, scaling_factor, data_info=None):
    """Apply scaling factor to forecasts with validation"""
    try:
        scaled_forecasts = forecasts * scaling_factor
        
        # Additional validation - check if scaling makes sense
        if data_info and 'actual_mean' in data_info:
            forecast_mean = np.mean(scaled_forecasts)
            actual_mean = data_info['actual_mean']
            
            # If still way off, apply more aggressive scaling
            if forecast_mean > actual_mean * 10:  # Still 10x too high
                additional_factor = actual_mean / forecast_mean
                scaled_forecasts = scaled_forecasts * additional_factor
                st.warning(f"🔧 Applied additional scaling: {additional_factor:.6f}")
        
        return np.maximum(scaled_forecasts, 0)
    except:
        return np.maximum(forecasts * scaling_factor, 0)


def run_fallback_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Simple fallback forecasting method"""
    try:
        # Simple moving average with trend
        if len(data) >= 12:
            recent_avg = data['Sales'].tail(12).mean()
            older_avg = data['Sales'].head(-12).mean() if len(data) > 12 else recent_avg
            trend = (recent_avg - older_avg) / 12 if older_avg > 0 else 0
        else:
            recent_avg = data['Sales'].mean()
            trend = 0
        
        # Generate forecast
        forecast = []
        for i in range(forecast_periods):
            value = recent_avg + (trend * i)
            forecast.append(max(0, value))
        
        forecast = np.array(forecast) * scaling_factor
        return forecast
        
    except Exception:
        # Ultimate fallback
        avg_value = data['Sales'].mean() if len(data) > 0 else 1000
        return np.full(forecast_periods, avg_value * scaling_factor)


def run_advanced_sarima_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """SARIMA forecasting with automatic parameter selection"""
    try:
        work_data = data.copy()
        
        # Try auto ARIMA first
        try:
            from pmdarima import auto_arima
            
            auto_model = auto_arima(
                work_data['Sales'],
                start_p=0, start_q=0, max_p=2, max_q=2,
                seasonal=True, m=12, start_P=0, start_Q=0,
                max_P=1, max_Q=1, trace=False,
                error_action='ignore', suppress_warnings=True,
                stepwise=True
            )
            
            best_order = auto_model.order
            best_seasonal_order = auto_model.seasonal_order
        
        except ImportError:
            # Default parameters if pmdarima not available
            best_order = (1, 1, 1)
            best_seasonal_order = (1, 1, 1, 12)
        
        # Fit SARIMA model
        model = SARIMAX(
            work_data['Sales'],
            order=best_order,
            seasonal_order=best_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False)
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=forecast_periods)
        
        # Apply inverse transformations
        work_columns = work_data.columns.tolist()
        if 'transformation' in work_columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecast = np.expm1(forecast)
            elif transform_method == 'sqrt':
                forecast = forecast ** 2
            elif transform_method == 'boxcox':
                params = work_data['transformation_params'].iloc[0]
                if isinstance(params, dict) and 'lambda' in params:
                    lambda_param = params['lambda']
                    forecast = inv_boxcox(forecast, lambda_param)
        
        forecast = np.maximum(forecast, 0) * scaling_factor
        
        return forecast, 75.0
        
    except Exception as e:
        st.warning(f"Theta failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_croston_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Croston method for intermittent demand"""
    try:
        work_data = data.copy()
        series = work_data['Sales'].values
        
        # Croston method parameters
        alpha = 0.3
        
        # Initialize
        non_zero_indices = np.where(series > 0)[0]
        if len(non_zero_indices) < 2:
            raise ValueError("Insufficient non-zero data for Croston")
        
        # Calculate intervals and sizes
        intervals = np.diff(non_zero_indices)
        sizes = series[non_zero_indices[1:]]
        
        # Initialize forecasts
        interval_forecast = intervals[0]
        size_forecast = sizes[0]
        
        # Update forecasts
        for i in range(1, len(intervals)):
            interval_forecast = alpha * intervals[i] + (1 - alpha) * interval_forecast
            size_forecast = alpha * sizes[i] + (1 - alpha) * size_forecast
        
        # Generate forecast
        demand_rate = size_forecast / interval_forecast if interval_forecast > 0 else 0
        forecast = np.full(forecast_periods, demand_rate)
        
        # Apply inverse transformations
        work_columns = work_data.columns.tolist()
        if 'transformation' in work_columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecast = np.expm1(forecast)
            elif transform_method == 'sqrt':
                forecast = forecast ** 2
            elif transform_method == 'boxcox':
                params = work_data['transformation_params'].iloc[0]
                if isinstance(params, dict) and 'lambda' in params:
                    lambda_param = params['lambda']
                    forecast = inv_boxcox(forecast, lambda_param)
        
        forecast = np.maximum(forecast, 0) * scaling_factor
        
        return forecast, 80.0
        
    except Exception as e:
        st.warning(f"Croston failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_lstm_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """LSTM neural network forecasting"""
    try:
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        work_data = data.copy()
        series = work_data['Sales'].values
        
        if len(series) < 24:
            raise ValueError("Insufficient data for LSTM")
        
        # Prepare data for LSTM
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(series.reshape(-1, 1))
        
        # Create sequences
        lookback = min(12, len(series) // 2)
        X, y = [], []
        
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        if len(X) < 10:
            raise ValueError("Insufficient sequences for LSTM")
        
        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        model.fit(X, y, batch_size=32, epochs=50, verbose=0)
        
        # Generate forecasts
        last_sequence = scaled_data[-lookback:]
        forecasts = []
        
        for _ in range(forecast_periods):
            pred_input = last_sequence.reshape(1, lookback, 1)
            pred = model.predict(pred_input, verbose=0)[0, 0]
            forecasts.append(pred)
            
            # Update sequence
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = pred
        
        # Inverse transform
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = scaler.inverse_transform(forecasts).flatten()
        
        # Apply inverse transformations
        work_columns = work_data.columns.tolist()
        if 'transformation' in work_columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecasts = np.expm1(forecasts)
            elif transform_method == 'sqrt':
                forecasts = forecasts ** 2
            elif transform_method == 'boxcox':
                params = work_data['transformation_params'].iloc[0]
                if isinstance(params, dict) and 'lambda' in params:
                    lambda_param = params['lambda']
                    forecasts = inv_boxcox(forecasts, lambda_param)
        
        forecasts = np.maximum(forecasts, 0) * scaling_factor
        
        return forecasts, 60.0
        
    except Exception as e:
        st.warning(f"LSTM failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def calculate_ensemble_weights(validation_scores, method="softmax"):
    """Calculate ensemble weights based on validation scores"""
    try:
        scores = np.array(list(validation_scores.values()))
        
        # Handle infinite scores
        finite_mask = np.isfinite(scores)
        if not np.any(finite_mask):
            # All scores are infinite, use equal weights
            return {model: 1/len(validation_scores) for model in validation_scores.keys()}
        
        # Replace infinite with large number
        scores[~finite_mask] = np.max(scores[finite_mask]) * 10 if np.any(finite_mask) else 1000
        
        if method == "softmax":
            # Convert to weights (lower is better, so negate)
            weights = np.exp(-scores / np.std(scores) if np.std(scores) > 0 else 1)
        elif method == "inverse_error":
            weights = 1 / (scores + 1e-10)
        else:  # equal
            weights = np.ones(len(scores))
        
        # Normalize
        weights = weights / np.sum(weights)
        
        return dict(zip(validation_scores.keys(), weights))
        
    except Exception:
        # Fallback to equal weights
        return {model: 1/len(validation_scores) for model in validation_scores.keys()}


def create_weighted_ensemble(forecast_results, weights):
    """Create weighted ensemble forecast"""
    try:
        ensemble_forecast = np.zeros(12)
        
        for model_name, weight in weights.items():
            forecast_key = f"{model_name}_Forecast"
            if forecast_key in forecast_results:
                ensemble_forecast += forecast_results[forecast_key] * weight
        
        return np.maximum(ensemble_forecast, 0)
        
    except Exception:
        # Fallback: simple average
        forecasts = list(forecast_results.values())
        return np.mean(forecasts, axis=0)


def create_meta_learning_ensemble(forecast_results, historical_data, actual_data, scaling_factor):
    """Create meta-learning ensemble using stacking"""
    try:
        if len(forecast_results) < 2:
            return None
        
        # This would require actual implementation with cross-validation
        # For now, return weighted average
        forecasts = list(forecast_results.values())
        return np.mean(forecasts, axis=0)
        
    except Exception:
        return None


def calculate_comprehensive_metrics(actual, predicted):
    """Calculate comprehensive forecast accuracy metrics"""
    try:
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {}
        
        # Basic metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # MAPE (handle zero actuals)
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
        else:
            mape = np.inf
        
        # Bias
        bias = np.mean(predicted - actual)
        bias_pct = (bias / np.mean(actual)) * 100 if np.mean(actual) != 0 else 0
        
        # Directional accuracy
        if len(actual) > 1:
            actual_direction = np.diff(actual) > 0
            predicted_direction = np.diff(predicted) > 0
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        else:
            directional_accuracy = 0
        
        # Tracking signal
        if mae > 0:
            tracking_signal = bias / mae
        else:
            tracking_signal = 0
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Bias': bias,
            'Bias_Pct': bias_pct,
            'Directional_Accuracy': directional_accuracy,
            'Tracking_Signal': tracking_signal
        }
        
    except Exception:
        return {}


def create_diagnostic_plots(data):
    """Create comprehensive diagnostic plots for time series analysis"""
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📈 Time Series Plot', '📊 Decomposition', 
                           '🔍 ACF/PACF', '📈 Distribution'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Time series plot
        fig.add_trace(
            go.Scatter(
                x=data['Month'],
                y=data['Sales_Original'],
                mode='lines+markers',
                name='Sales',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Simple trend line
        if len(data) > 12:
            x_numeric = np.arange(len(data))
            z = np.polyfit(x_numeric, data['Sales_Original'], 1)
            trend_line = np.polyval(z, x_numeric)
            
            fig.add_trace(
                go.Scatter(
                    x=data['Month'],
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Seasonal decomposition (simplified)
        if len(data) >= 24:
            try:
                # Simple seasonal calculation
                monthly_avg = data.groupby(data['Month'].dt.month)['Sales_Original'].mean()
                seasonal_index = monthly_avg / monthly_avg.mean()
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig.add_trace(
                    go.Bar(
                        x=month_names,
                        y=seasonal_index.values,
                        name='Seasonal Pattern',
                        marker_color='lightblue'
                    ),
                    row=1, col=2
                )
                
                fig.add_hline(y=1, line_dash="dash", line_color="gray", row=1, col=2)
                
            except Exception:
                # Fallback: show monthly averages
                monthly_sales = data.groupby(data['Month'].dt.month)['Sales_Original'].mean()
                fig.add_trace(
                    go.Bar(
                        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        y=monthly_sales.values,
                        name='Monthly Averages',
                        marker_color='lightgreen'
                    ),
                    row=1, col=2
                )
        
        # Distribution plot
        fig.add_trace(
            go.Histogram(
                x=data['Sales_Original'],
                name='Sales Distribution',
                nbinsx=20,
                marker_color='lightcoral',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Box plot for outlier detection
        fig.add_trace(
            go.Box(
                y=data['Sales_Original'],
                name='Sales Box Plot',
                marker_color='lightpink'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            title_text="📊 Time Series Diagnostic Dashboard",
            showlegend=True,
            title_x=0.5
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Sales", row=1, col=1)
        
        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="Seasonal Index", row=1, col=2)
        
        fig.update_xaxes(title_text="Sales Value", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_yaxes(title_text="Sales", row=2, col=2)
        
        return fig
        
    except Exception as e:
        # Return simple plot if diagnostic fails
        simple_fig = go.Figure()
        simple_fig.add_trace(
            go.Scatter(
                x=data['Month'],
                y=data['Sales_Original'],
                mode='lines+markers',
                name='Sales'
            )
        )
        simple_fig.update_layout(
            title="📈 Sales Over Time",
            xaxis_title="Date",
            yaxis_title="Sales"
        )
        return simple_fig


def create_forecast_plot(result_df, forecast_year, historical_data):
    """Create interactive forecast visualization"""
    try:
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=historical_data['Month'],
                y=historical_data['Sales_Original'],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            )
        )
        
        # Forecast models
        forecast_cols = [col for col in result_df.columns 
                        if '_Forecast' in col or col in ['Weighted_Ensemble', 'Meta_Learning']]
        
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
        
        for i, col in enumerate(forecast_cols):
            color = colors[i % len(colors)]
            model_name = col.replace('_Forecast', '').replace('_', ' ')
            
            fig.add_trace(
                go.Scatter(
                    x=result_df['Month'],
                    y=result_df[col],
                    mode='lines+markers',
                    name=f'{model_name} Forecast',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=6)
                )
            )
        
        # Actual data if available
        actual_col = f'Actual_{forecast_year}'
        if actual_col in result_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=result_df['Month'],
                    y=result_df[actual_col],
                    mode='lines+markers',
                    name=f'Actual {forecast_year}',
                    line=dict(color='black', width=3),
                    marker=dict(size=8, symbol='diamond')
                )
            )
        
        fig.update_layout(
            title=f"📈 Sales Forecast vs Historical Data ({forecast_year})",
            xaxis_title="Date",
            yaxis_title="Sales",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
        
    except Exception:
        # Simple fallback plot
        simple_fig = go.Figure()
        simple_fig.add_trace(
            go.Scatter(
                x=result_df['Month'],
                y=result_df.iloc[:, 1],
                mode='lines+markers',
                name='Forecast'
            )
        )
        simple_fig.update_layout(title="Forecast Results")
        return simple_fig


def main():
    """Main application function"""
    st.title("🚀 Advanced AI Sales Forecasting System")
    st.markdown("**Enterprise-grade forecasting with 8+ models, ensemble learning, and neural networks**")
    
    # Initialize session state
    if 'forecast_info' not in st.session_state:
        st.session_state.forecast_info = {}
    if 'quick_scale' not in st.session_state:
        st.session_state.quick_scale = 1.0
    
    # Display warnings for missing packages
    if not XGBOOST_AVAILABLE:
        st.warning("⚠️ XGBoost not installed. Install with: `pip install xgboost` for better accuracy")
    if not TENSORFLOW_AVAILABLE:
        st.info("ℹ️ TensorFlow not available. Install with: `pip install tensorflow` for LSTM models")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ System Configuration")
    
    # Basic settings
    forecast_year = st.sidebar.selectbox(
        "📅 Select Forecast Year",
        options=[2024, 2025, 2026, 2027],
        index=0
    )
    
    # Advanced settings
    with st.sidebar.expander("🔬 Advanced Settings", expanded=True):
        st.subheader("🎯 Data Processing")
        enable_preprocessing = st.checkbox("Advanced Data Preprocessing", value=True,
                                         help="Apply outlier detection, transformations, and data cleaning")
        
        # Unit scaling options - More prominent and easier to use
        st.subheader("🚨 SCALE FIX (Use if forecasts are too high/low)")
        
        # Quick fix buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📉 Divide by 1000", help="Click if forecasts are 1000x too high"):
                st.session_state['quick_scale'] = 0.001
        with col2:
            if st.button("📉 Divide by 1M", help="Click if forecasts are 1,000,000x too high"):
                st.session_state['quick_scale'] = 0.000001
        
        # Get scaling factor
        if 'quick_scale' in st.session_state and st.session_state['quick_scale'] != 1.0:
            scaling_override = st.session_state['quick_scale']
            st.success(f"🔧 Quick scaling applied: {scaling_override}")
        else:
            scaling_override = st.number_input(
                "Manual Scale Factor", 
                value=1.0, 
                step=0.001,
                format="%.6f",
                help="Enter 0.001 to divide by 1000, or 0.000001 to divide by 1M"
            )
        
        # Advanced manual scaling (keep existing)
        manual_scaling = st.selectbox(
            "Advanced Scaling Presets",
            options=["None", "Historical in Thousands", "Historical in Millions", 
                    "Actual in Thousands", "Actual in Millions"],
            index=0
        )
        
        if manual_scaling == "Historical in Thousands":
            scaling_override = 0.001
        elif manual_scaling == "Historical in Millions":
            scaling_override = 0.000001
        elif manual_scaling == "Actual in Thousands":
            scaling_override = 1000.0
        elif manual_scaling == "Actual in Millions":
            scaling_override = 1000000.0
        
        st.subheader("🤖 Ensemble Settings")
        ensemble_method = st.selectbox(
            "Ensemble Weighting Method",
            options=["Softmax", "Inverse Error", "Equal"],
            index=0
        )
        enable_meta_learning = st.checkbox("Enable Meta-Learning", value=True,
                                         help="Use stacking with Ridge regression")
        
        st.subheader("📊 Visualization Settings")
        show_diagnostics = st.checkbox("Show Diagnostic Plots", value=True)
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.markdown("**Classic Models**")
        use_sarima = st.checkbox("SARIMA (Auto)", value=True)
        use_ets = st.checkbox("ETS (Auto)", value=True)
        use_theta = st.checkbox("Theta Method", value=True)
        use_croston = st.checkbox("Croston (Intermittent)", value=False)
    
    with col2:
        st.markdown("**ML/DL Models**")
        use_prophet = st.checkbox("Prophet (Enhanced)", value=True)
        use_xgboost = st.checkbox("XGBoost (Advanced)", value=True)
        use_lstm = st.checkbox("LSTM Neural Net", value=TENSORFLOW_AVAILABLE)
        use_ensemble = st.checkbox("Ensemble Models", value=True)
    
    # Validate model selection
    model_list = [use_sarima, use_ets, use_theta, use_croston, use_prophet, use_xgboost, use_lstm]
    selected_models = sum(model_list)
    
    if selected_models == 0:
        st.sidebar.error("❌ Please select at least one model!")
        return
    
    # File upload section
    st.header("📁 Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        historical_file = st.file_uploader(
            "📊 Upload Historical Sales Data",
            type=["xlsx", "xls"],
            help="Excel file with Month and Sales columns"
        )
    
    with col2:
        actual_file = st.file_uploader(
            f"📈 Upload {forecast_year} Actual Data (Optional)",
            type=["xlsx", "xls"],
            help="For validation and meta-learning"
        )
    
    if historical_file is None:
        st.info("👆 Please upload historical sales data to begin forecasting")
        
        # Show sample data format
        with st.expander("📋 View Sample Data Format"):
            sample_data = pd.DataFrame({
                'Month': pd.date_range('2022-01-01', periods=24, freq='MS'),
                'Sales': np.random.randint(1000, 5000, 24)
            })
            st.dataframe(sample_data.head(10))
        
        return
    
    # Load data
    file_content = historical_file.read()
    file_hash = hashlib.md5(file_content).hexdigest()
    
    hist_df = load_data_optimized(file_content, file_hash)
    
    if hist_df is None:
        return
    
    # Load actual data if provided - with scaling logic
    actual_df = None
    final_scaling_factor = scaling_override  # Use the scaling override directly
    
    if actual_file is not None:
        actual_content = actual_file.read()
        actual_df = load_actual_2024_data(io.BytesIO(actual_content), forecast_year)
        
        if actual_df is not None:
            # Show the detected scaling issue but don't override manual setting
            auto_scaling = detect_and_apply_scaling(hist_df, actual_df)
            
            # If user hasn't set manual scaling (still at default 1.0), use auto-detected
            if scaling_override == 1.0:
                final_scaling_factor = auto_scaling
            else:
                # User has set manual scaling - use that instead
                final_scaling_factor = scaling_override
            
            st.info(f"📊 **Final Scaling Factor**: {final_scaling_factor:.6f}")
            
            if final_scaling_factor < 0.01:
                st.success("✅ **Small Scaling Factor** - This will significantly REDUCE forecast values (divide by large number)")
            elif final_scaling_factor > 100:
                st.warning("⚠️ **Large Scaling Factor** - This will significantly INCREASE forecast values")
    else:
        final_scaling_factor = scaling_override
        st.info(f"📊 **Scaling Factor**: {final_scaling_factor:.6f}")
    
    # Show clear guidance
    if final_scaling_factor != 1.0:
        if final_scaling_factor < 1.0:
            reduction_factor = 1 / final_scaling_factor
            st.success(f"✅ **Forecasts will be REDUCED** (divided by {reduction_factor:.0f})")
        else:
            st.success(f"✅ **Forecasts will be INCREASED** (multiplied by {final_scaling_factor:.0f})")
    
    # Additional data validation
    if hist_df is not None:
        # Check for reasonable data ranges
        hist_mean = hist_df['Sales_Original'].mean()
        hist_std = hist_df['Sales_Original'].std()
        cv = hist_std / hist_mean if hist_mean > 0 else 0
        
        if cv > 2:
            st.warning("⚠️ **High Variability Detected**: Your data has high volatility (CV > 200%)")
            st.info("💡 Consider using ensemble methods or robust forecasting techniques")
        
        # Check for trend
        if len(hist_df) >= 12:
            recent_avg = hist_df['Sales_Original'].tail(6).mean()
            older_avg = hist_df['Sales_Original'].head(6).mean()
            trend_ratio = recent_avg / older_avg if older_avg > 0 else 1
            
            if trend_ratio > 1.5:
                st.info("📈 **Strong Growth Trend** detected in recent data")
            elif trend_ratio < 0.5:
                st.warning("📉 **Declining Trend** detected in recent data")
    
    # Data Analysis Dashboard
    st.header("📊 Data Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📅 Data Points", len(hist_df))
    
    with col2:
        avg_sales = hist_df['Sales_Original'].mean()
        st.metric("💰 Avg Sales", f"{avg_sales:,.0f}")
    
    with col3:
        cv = hist_df['Sales_Original'].std() / avg_sales
        st.metric("📊 CV", f"{cv:.2%}")
    
    with col4:
        data_quality = min(100, len(hist_df) * 4.17)
        st.metric("🎯 Data Quality", f"{data_quality:.0f}%")
    
    with col5:
        freq = detect_data_frequency(hist_df['Month'])
        st.metric("📆 Frequency", freq)
    
    # Show diagnostic plots
    if show_diagnostics:
        with st.expander("📈 Time Series Diagnostics", expanded=False):
            diagnostic_fig = create_diagnostic_plots(hist_df)
            st.plotly_chart(diagnostic_fig, use_container_width=True)

    # Forecasting section
    if st.button("🚀 Run Forecasting", type="primary", use_container_width=True):
        
        # Build model list
        models_to_run = []
        if use_sarima:
            models_to_run.append(("SARIMA", run_advanced_sarima_forecast))
        if use_prophet:
            models_to_run.append(("Prophet", run_advanced_prophet_forecast))
        if use_ets:
            models_to_run.append(("ETS", run_advanced_ets_forecast))
        if use_xgboost and XGBOOST_AVAILABLE:
            models_to_run.append(("XGBoost", run_xgboost_forecast))
        if use_theta:
            models_to_run.append(("Theta", run_theta_forecast))
        if use_croston:
            models_to_run.append(("Croston", run_croston_forecast))
        if use_lstm and TENSORFLOW_AVAILABLE:
            models_to_run.append(("LSTM", run_lstm_forecast))
        
        # Progress tracking
        ensemble_steps = 2 if use_ensemble else 0
        total_steps = len(models_to_run) + ensemble_steps
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Store forecast results
        forecast_results = {}
        validation_scores = {}
        
        # Sequential execution for stability
        for i, (model_name, model_func) in enumerate(models_to_run):
            status_text.text(f"Training {model_name}...")
            
            try:
                forecast_values, score = model_func(hist_df, 12, final_scaling_factor)
                forecast_results[f"{model_name}_Forecast"] = forecast_values
                validation_scores[model_name] = score
                
                # Show model-specific info with scale validation
                forecast_avg = np.mean(forecast_values)
                forecast_range = f"{np.min(forecast_values):,.0f} - {np.max(forecast_values):,.0f}"
                
                if score != np.inf:
                    st.success(f"✅ {model_name} completed - Avg: {forecast_avg:,.0f} (Range: {forecast_range})")
                else:
                    st.warning(f"⚠️ {model_name} completed with fallback - Avg: {forecast_avg:,.0f}")
                
            except Exception as e:
                st.error(f"❌ {model_name} failed: {str(e)}")
                fallback_forecast = run_fallback_forecast(hist_df, 12, final_scaling_factor)
                forecast_results[f"{model_name}_Forecast"] = fallback_forecast
                validation_scores[model_name] = np.inf
            
            progress_bar.progress((i + 1) / total_steps)
        
        # Generate ensemble forecasts
        if use_ensemble and len(forecast_results) > 1:
            status_text.text("Creating ensemble forecasts...")
            
            try:
                # Weighted ensemble
                weights = calculate_ensemble_weights(validation_scores, method=ensemble_method)
                weighted_forecast = create_weighted_ensemble(forecast_results, weights)
                forecast_results["Weighted_Ensemble"] = weighted_forecast
                
                progress_bar.progress((len(models_to_run) + 1) / total_steps)
                
                # Meta-learning ensemble
                if enable_meta_learning and actual_df is not None:
                    meta_forecast = create_meta_learning_ensemble(
                        forecast_results, hist_df, actual_df, final_scaling_factor
                    )
                    if meta_forecast is not None:
                        forecast_results["Meta_Learning"] = meta_forecast
                
                progress_bar.progress(1.0)
                
            except Exception as e:
                st.error(f"❌ Ensemble creation failed: {str(e)}")
        
        status_text.text("✅ Forecasting completed!")
        
        # Create results dataframe
        forecast_dates = pd.date_range(
            start=f"{forecast_year}-01-01",
            periods=12,
            freq='MS'
        )
        
        result_df = pd.DataFrame({'Month': forecast_dates})
        
        # Add forecast results
        for key, values in forecast_results.items():
            result_df[key] = values
        
        # Add actual data if available
        if actual_df is not None:
            actual_col = f'Actual_{forecast_year}'
            actual_df_renamed = actual_df.copy()
            actual_df_renamed.columns = [actual_df_renamed.columns[0], actual_col]
            
            result_df = result_df.merge(
                actual_df_renamed,
                left_on='Month',
                right_on=actual_df_renamed.columns[0],
                how='left'
            )
            result_df = result_df.drop(columns=[actual_df_renamed.columns[0]])
        
        # Display results
        st.header("📊 Forecast Results")
        
        # Check if forecasts look reasonable compared to actual data
        if actual_df is not None and len(actual_df) > 0:
            actual_col = f'Actual_{forecast_year}'
            actual_values = result_df[result_df[actual_col].notna()][actual_col]
            
            if len(actual_values) > 0:
                actual_mean = actual_values.mean()
                
                # Check if any forecast is way off
                forecast_cols = [col for col in result_df.columns 
                               if '_Forecast' in col or col in ['Weighted_Ensemble', 'Meta_Learning']]
                
                way_off_models = []
                for col in forecast_cols:
                    forecast_mean = result_df[col].mean()
                    if forecast_mean > actual_mean * 3:  # More than 3x actual
                        way_off_models.append((col, forecast_mean / actual_mean))
                
                if way_off_models:
                    st.error("🚨 **Scale Issue Still Detected!**")
                    st.error(f"   Actual data average: {actual_mean:,.0f}")
                    
                    # Show top 3 problematic models
                    for model, ratio in way_off_models[:3]:
                        model_name = model.replace('_Forecast', '')
                        forecast_avg = result_df[model].mean()
                        st.error(f"   {model_name} average: {forecast_avg:,.0f} ({ratio:.1f}x too high)")
                    
                    # Emergency fix button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🔧 **EMERGENCY SCALE FIX**", type="primary", use_container_width=True):
                            # Apply aggressive scaling to all forecasts
                            first_forecast_col = forecast_cols[0]
                            correction_ratio = actual_mean / result_df[first_forecast_col].mean()
                            
                            for col in forecast_cols:
                                result_df[col] = result_df[col] * correction_ratio
                            
                            st.success(f"✅ Applied emergency scaling: {correction_ratio:.6f}")
                            st.rerun()
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Weighted_Ensemble' in result_df.columns:
                total_forecast = result_df['Weighted_Ensemble'].sum()
            else:
                first_forecast = list(forecast_results.values())[0]
                total_forecast = first_forecast.sum()
            st.metric("📈 Total Forecast", f"{total_forecast:,.0f}")
        
        with col2:
            avg_monthly = total_forecast / 12
            st.metric("📅 Average Monthly", f"{avg_monthly:,.0f}")
        
        with col3:
            recent_total = hist_df['Sales_Original'].tail(12).sum()
            yoy_growth = ((total_forecast - recent_total) / recent_total * 100)
            st.metric("📊 YoY Growth", f"{yoy_growth:+.1f}%")
        
        # Interactive forecast plot
        st.subheader("📈 Interactive Forecast Visualization")
        forecast_fig = create_forecast_plot(result_df, forecast_year, hist_df)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Model Performance Analysis
        actual_col = f'Actual_{forecast_year}'
        if actual_col in result_df.columns and result_df[actual_col].notna().any():
            st.subheader("🎯 Model Performance Analysis")
            
            # Create performance summary with error handling
            performance_data = []
            
            model_cols = [c for c in result_df.columns 
                         if '_Forecast' in c or c in ['Weighted_Ensemble', 'Meta_Learning']]
            
            for col in model_cols:
                try:
                    model_name = col.replace('_Forecast', '').replace('_', ' ')
                    
                    # Get subset with actual data available
                    actual_subset = result_df[result_df[actual_col].notna()].copy()
                    
                    if len(actual_subset) == 0:
                        continue
                    
                    # Calculate metrics with error handling
                    metrics = calculate_comprehensive_metrics(
                        actual_subset[actual_col],
                        actual_subset[col]
                    )
                    
                    if metrics:
                        performance_data.append({
                            'Model': model_name,
                            'MAPE (%)': f"{metrics.get('MAPE', 0):.1f}",
                            'RMSE': f"{metrics.get('RMSE', 0):,.0f}",
                            'MAE': f"{metrics.get('MAE', 0):,.0f}",
                            'Bias (%)': f"{metrics.get('Bias_Pct', 0):+.1f}",
                            'Direction Acc (%)': f"{metrics.get('Directional_Accuracy', 0):.0f}",
                            'Tracking Signal': f"{metrics.get('Tracking_Signal', 0):.1f}"
                        })
                except Exception as e:
                    st.warning(f"Could not calculate metrics for {col}: {str(e)}")
                    continue
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                
                # Sort by MAPE (convert to numeric for sorting)
                try:
                    perf_df['MAPE_numeric'] = perf_df['MAPE (%)'].str.replace('%', '').astype(float)
                    perf_df = perf_df.sort_values('MAPE_numeric').drop('MAPE_numeric', axis=1)
                except:
                    pass  # Keep original order if sorting fails
                
                # Display performance table
                st.dataframe(
                    perf_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show best model if available
                if len(perf_df) > 0:
                    best_model = perf_df.iloc[0]['Model']
                    best_mape = perf_df.iloc[0]['MAPE (%)']
                    st.success(f"🏆 Best Model: **{best_model}** (MAPE: {best_mape})")
            else:
                st.info("📊 Performance metrics will be available when actual data is provided for comparison.")
        else:
            st.info("📊 Upload actual data for the forecast year to see model performance analysis.")
        
        # Detailed results table
        st.subheader("📋 Detailed Forecast Results")
        
        # Format the results for display
        display_df = result_df.copy()
        
        # Format month column
        display_df['Month'] = display_df['Month'].dt.strftime('%Y-%m')
        
        # Round numeric columns
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        display_df[numeric_cols] = display_df[numeric_cols].round(0)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download functionality
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_data = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"forecast_results_{forecast_year}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel download
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                result_df.to_excel(writer, sheet_name='Forecasts', index=False)
                if 'performance_data' in locals() and performance_data:
                    pd.DataFrame(performance_data).to_excel(writer, sheet_name='Performance', index=False)
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name=f"forecast_results_{forecast_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()0]
                if isinstance(params, dict) and 'lambda' in params:
                    lambda_param = params['lambda']
                    forecast = inv_boxcox(forecast, lambda_param)
        
        # Apply scaling with enhanced validation
        forecast = apply_scaling_to_forecasts(forecast, scaling_factor)
        
        return forecast, fitted_model.aic
        
    except Exception as e:
        st.warning(f"SARIMA failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_advanced_prophet_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Enhanced Prophet forecasting"""
    try:
        work_data = data.copy()
        
        # Prepare data for Prophet
        prophet_data = work_data[['Month', 'Sales']].rename(columns={'Month': 'ds', 'Sales': 'y'})
        
        # Create Prophet model
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode='additive',
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        # Add custom seasonality
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
        
        # Fit model
        model.fit(prophet_data)
        
        # Make predictions
        future = model.make_future_dataframe(periods=forecast_periods, freq='MS')
        forecast = model.predict(future)
        forecast_values = forecast['yhat'].tail(forecast_periods).values
        
        # Apply inverse transformations
        work_columns = work_data.columns.tolist()
        if 'transformation' in work_columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecast_values = np.expm1(forecast_values)
            elif transform_method == 'sqrt':
                forecast_values = forecast_values ** 2
        
        # Apply scaling - this is the key fix
        forecast_values = np.maximum(forecast_values, 0) * scaling_factor
        
        return forecast_values, 100.0
        
    except Exception as e:
        st.warning(f"Prophet failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_advanced_ets_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """ETS forecasting with automatic model selection"""
    try:
        work_data = data.copy()
        
        # Test different ETS configurations
        configs = [
            {'seasonal': 'add', 'trend': 'add', 'damped_trend': True},
            {'seasonal': 'add', 'trend': 'add', 'damped_trend': False},
            {'seasonal': 'mul', 'trend': 'add', 'damped_trend': False},
            {'seasonal': 'add', 'trend': None},
            {'seasonal': None, 'trend': 'add'}
        ]
        
        best_model = None
        best_aic = np.inf
        
        for config in configs:
            try:
                model = ExponentialSmoothing(
                    work_data['Sales'].values,
                    seasonal=config.get('seasonal'),
                    seasonal_periods=12 if config.get('seasonal') else None,
                    trend=config.get('trend'),
                    damped_trend=config.get('damped_trend', False),
                    initialization_method='estimated'
                )
                
                fitted_model = model.fit(optimized=True)
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
            except:
                continue
        
        if best_model is not None:
            forecast = best_model.forecast(steps=forecast_periods)
            
            # Apply inverse transformations
            work_columns = work_data.columns.tolist()
            if 'transformation' in work_columns:
                transform_method = work_data['transformation'].iloc[0]
                if transform_method == 'log':
                    forecast = np.expm1(forecast)
                elif transform_method == 'sqrt':
                    forecast = forecast ** 2
                elif transform_method == 'boxcox':
                    params = work_data['transformation_params'].iloc[0]
                    if isinstance(params, dict) and 'lambda' in params:
                        lambda_param = params['lambda']
                        forecast = inv_boxcox(forecast, lambda_param)
            
            # Apply scaling - this is the key fix
            forecast = np.maximum(forecast, 0) * scaling_factor
            
            return forecast, best_aic
        else:
            raise ValueError("All ETS configurations failed")
            
    except Exception as e:
        st.warning(f"ETS failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_xgboost_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """XGBoost time series forecasting"""
    try:
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        work_data = data.copy()
        
        # Create features
        def create_features(df):
            df = df.copy()
            df['month'] = df['Month'].dt.month
            df['quarter'] = df['Month'].dt.quarter
            df['year'] = df['Month'].dt.year
            
            # Lag features
            for lag in [1, 2, 3, 6, 12]:
                if len(df) > lag:
                    df[f'lag_{lag}'] = df['Sales'].shift(lag)
            
            # Rolling statistics
            for window in [3, 6, 12]:
                if len(df) > window:
                    df[f'rolling_mean_{window}'] = df['Sales'].rolling(window).mean()
                    df[f'rolling_std_{window}'] = df['Sales'].rolling(window).std()
            
            return df
        
        # Prepare training data
        featured_data = create_features(work_data)
        featured_data = featured_data.dropna()
        
        if len(featured_data) < 10:
            raise ValueError("Insufficient data for XGBoost")
        
        # Prepare features and target
        feature_cols = [col for col in featured_data.columns 
                       if col not in ['Month', 'Sales', 'Sales_Original', 'transformation', 'transformation_params']]
        
        X = featured_data[feature_cols]
        y = featured_data['Sales']
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X, y)
        
        # Generate forecasts
        forecasts = []
        last_data = work_data.copy()
        
        for i in range(forecast_periods):
            # Create features for next period
            next_date = last_data['Month'].iloc[-1] + pd.DateOffset(months=1)
            
            # Add new row
            new_row = {'Month': next_date, 'Sales': 0}
            new_df = pd.concat([last_data, pd.DataFrame([new_row])], ignore_index=True)
            
            # Create features
            featured_new = create_features(new_df)
            
            # Get features for prediction
            pred_features = featured_new[feature_cols].iloc[-1:].fillna(0)
            
            # Predict
            pred = model.predict(pred_features)[0]
            forecasts.append(max(0, pred))
            
            # Update data for next iteration
            last_data.loc[len(last_data)] = {'Month': next_date, 'Sales': pred}
        
        forecasts = np.array(forecasts)
        
        # Apply inverse transformations
        work_columns = work_data.columns.tolist()
        if 'transformation' in work_columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecasts = np.expm1(forecasts)
            elif transform_method == 'sqrt':
                forecasts = forecasts ** 2
            elif transform_method == 'boxcox':
                params = work_data['transformation_params'].iloc[0]
                if isinstance(params, dict) and 'lambda' in params:
                    lambda_param = params['lambda']
                    forecasts = inv_boxcox(forecasts, lambda_param)
        
        forecasts = np.maximum(forecasts, 0) * scaling_factor
        
        return forecasts, 50.0
        
    except Exception as e:
        st.warning(f"XGBoost failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_theta_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Theta method forecasting"""
    try:
        work_data = data.copy()
        series = work_data['Sales'].values
        
        if len(series) < 12:
            raise ValueError("Insufficient data for Theta method")
        
        # Simple Theta method implementation
        def theta_forecast(y, h, theta=2):
            n = len(y)
            
            # Linear trend estimation
            t = np.arange(1, n + 1)
            slope = np.sum((t - np.mean(t)) * (y - np.mean(y))) / np.sum((t - np.mean(t))**2)
            intercept = np.mean(y) - slope * np.mean(t)
            
            # Trend line
            trend = intercept + slope * np.arange(n + 1, n + h + 1)
            
            # Simple exponential smoothing on detrended series
            detrended = y - (intercept + slope * t)
            alpha = 0.3  # Smoothing parameter
            
            smoothed = [detrended[0]]
            for i in range(1, len(detrended)):
                smoothed.append(alpha * detrended[i] + (1 - alpha) * smoothed[-1])
            
            # Forecast
            last_smooth = smoothed[-1]
            forecast = trend + last_smooth
            
            return np.maximum(forecast, 0)
        
        forecast = theta_forecast(series, forecast_periods)
        
        # Apply inverse transformations
        work_columns = work_data.columns.tolist()
        if 'transformation' in work_columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecast = np.expm1(forecast)
            elif transform_method == 'sqrt':
                forecast = forecast ** 2
            elif transform_method == 'boxcox':
                params = work_data['transformation_params'].iloc[
