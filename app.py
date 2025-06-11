import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import hashlib
from datetime import datetime
from prophet import Prophet
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Configure streamlit
st.set_page_config(page_title="Advanced AI Sales Forecasting System", layout="wide")

# Try to import optional libraries
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


@st.cache_data(ttl=3600)
def load_data_optimized(file_content, file_hash):
    """Load and preprocess data"""
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
        df_processed = preprocess_data(df_monthly)
        st.success(f"✅ Successfully aggregated to {len(df_processed)} monthly data points")
    else:
        df['Sales_Original'] = df['Sales'].copy()
        df_processed = preprocess_data(df)
    
    return df_processed


def preprocess_data(df):
    """Basic data preprocessing"""
    df = df.copy()
    
    # Outlier detection using IQR
    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df['Sales'] < lower_bound) | (df['Sales'] > upper_bound))
    
    if outliers.sum() > 0:
        st.info(f"📊 Detected {outliers.sum()} outliers")
        df.loc[outliers, 'Sales'] = df.loc[~outliers, 'Sales'].quantile(0.95)
    
    # Handle missing values
    if df['Sales'].isna().any():
        df['Sales'] = df['Sales'].fillna(df['Sales'].mean())
    
    # Add cyclical encoding for months
    df['month'] = df['Month'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def run_prophet_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Prophet forecasting model"""
    try:
        # Prepare data
        prophet_data = data[['Month', 'Sales']].rename(columns={'Month': 'ds', 'Sales': 'y'})
        
        # Create and fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        model.fit(prophet_data)
        
        # Make predictions
        future = model.make_future_dataframe(periods=forecast_periods, freq='MS')
        forecast = model.predict(future)
        forecast_values = forecast['yhat'].tail(forecast_periods).values
        forecast_values = np.maximum(forecast_values, 0) * scaling_factor
        
        return forecast_values, 100.0
        
    except Exception as e:
        st.warning(f"Prophet failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_xgboost_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Simple XGBoost forecasting"""
    if not XGBOOST_AVAILABLE:
        st.warning("XGBoost not installed. Using fallback forecast.")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf
    
    try:
        # Create simple features
        df = data.copy()
        df['trend'] = np.arange(len(df))
        df['lag_1'] = df['Sales'].shift(1)
        df['lag_12'] = df['Sales'].shift(12)
        df['rolling_mean_3'] = df['Sales'].rolling(window=3, min_periods=1).mean()
        df['rolling_mean_12'] = df['Sales'].rolling(window=12, min_periods=1).mean()
        
        # Remove NaN values
        df = df.dropna()
        
        if len(df) < 12:
            st.warning("Insufficient data for XGBoost.")
            return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf
        
        # Define features and target
        feature_cols = ['trend', 'month_sin', 'month_cos', 'lag_1', 'lag_12', 
                       'rolling_mean_3', 'rolling_mean_12']
        X = df[feature_cols]
        y = df['Sales']
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        
        # Generate forecasts
        predictions = []
        last_known = df.iloc[-1].copy()
        
        for i in range(forecast_periods):
            # Update features
            next_month = last_known['Month'] + pd.DateOffset(months=1)
            
            features = {
                'trend': last_known['trend'] + i + 1,
                'month_sin': np.sin(2 * np.pi * next_month.month / 12),
                'month_cos': np.cos(2 * np.pi * next_month.month / 12),
                'lag_1': predictions[i-1] if i > 0 else last_known['Sales'],
                'lag_12': df['Sales'].iloc[-(12-i)] if i < 12 else predictions[i-12],
                'rolling_mean_3': np.mean(predictions[max(0, i-2):i+1]) if i > 0 else last_known['rolling_mean_3'],
                'rolling_mean_12': np.mean(predictions[max(0, i-11):i+1]) if i > 0 else last_known['rolling_mean_12']
            }
            
            feature_vector = np.array([features[col] for col in feature_cols]).reshape(1, -1)
            pred = model.predict(feature_vector)[0]
            predictions.append(max(pred, 0))
        
        forecasts = np.array(predictions) * scaling_factor
        return forecasts, 50.0
        
    except Exception as e:
        st.warning(f"XGBoost failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_fallback_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Fallback forecasting method"""
    try:
        if len(data) >= 12:
            # Seasonal naive with trend
            seasonal_pattern = data['Sales'].tail(12).values
            
            # Calculate trend
            X_trend = np.arange(len(data)).reshape(-1, 1)
            y_trend = data['Sales'].values
            
            trend_model = HuberRegressor()
            trend_model.fit(X_trend, y_trend)
            
            # Generate forecast
            forecast = []
            last_index = len(data)
            
            for i in range(forecast_periods):
                seasonal_component = seasonal_pattern[i % 12]
                trend_component = trend_model.predict([[last_index + i]])[0] - trend_model.predict([[last_index]])[0]
                forecast_value = seasonal_component + trend_component
                forecast.append(max(forecast_value, seasonal_component * 0.5))
            
            forecast = np.array(forecast)
        else:
            # Simple average for short series
            base_forecast = data['Sales'].mean()
            forecast = np.full(forecast_periods, base_forecast)
        
        return forecast * scaling_factor
        
    except Exception as e:
        # Ultimate fallback
        historical_mean = data['Sales'].mean() if len(data) > 0 else 1000
        return np.array([historical_mean * scaling_factor] * forecast_periods)


def create_simple_ensemble(forecasts_dict):
    """Create simple ensemble forecast"""
    if len(forecasts_dict) <= 1:
        return list(forecasts_dict.values())[0]
    
    # Simple average ensemble
    forecast_array = np.array(list(forecasts_dict.values()))
    ensemble_forecast = np.mean(forecast_array, axis=0)
    
    return ensemble_forecast


def main():
    """Main application function"""
    st.title("🚀 Advanced AI Sales Forecasting System")
    st.markdown("**Enterprise-grade forecasting with multiple models and ensemble learning**")
    
    # Display warnings for missing packages
    if not XGBOOST_AVAILABLE:
        st.warning("⚠️ XGBoost not installed. Install with: `pip install xgboost` for better accuracy")
    if not TENSORFLOW_AVAILABLE:
        st.info("ℹ️ TensorFlow not available. Install with: `pip install tensorflow` for LSTM models")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    forecast_year = st.sidebar.selectbox(
        "📅 Select Forecast Year",
        options=[2024, 2025, 2026, 2027],
        index=1
    )
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    use_prophet = st.sidebar.checkbox("Prophet", value=True)
    use_xgboost = st.sidebar.checkbox("XGBoost", value=True)
    use_fallback = st.sidebar.checkbox("Fallback Model", value=True)
    use_ensemble = st.sidebar.checkbox("Ensemble", value=True)
    
    # File upload
    st.header("📁 Data Upload")
    
    historical_file = st.file_uploader(
        "📊 Upload Historical Sales Data",
        type=["xlsx", "xls"],
        help="Excel file with 'Month' and 'Sales' columns"
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
    
    # Data Analysis Dashboard
    st.header("📊 Data Analysis Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📅 Data Points", len(hist_df))
    
    with col2:
        avg_sales = hist_df['Sales_Original'].mean()
        st.metric("💰 Avg Sales", f"{avg_sales:,.0f}")
    
    with col3:
        cv = hist_df['Sales_Original'].std() / avg_sales
        st.metric("📊 CV", f"{cv:.2%}")
    
    # Forecasting section
    if st.button("🚀 Generate Forecasts", type="primary", use_container_width=True):
        st.header("🔮 Generating Forecasts...")
        
        forecast_results = {}
        
        # Run selected models
        if use_prophet:
            try:
                forecast_values, score = run_prophet_forecast(hist_df, 12, 1.0)
                forecast_results["Prophet_Forecast"] = forecast_values
                st.success("✅ Prophet model completed")
            except Exception as e:
                st.error(f"❌ Prophet failed: {str(e)}")
        
        if use_xgboost:
            try:
                forecast_values, score = run_xgboost_forecast(hist_df, 12, 1.0)
                forecast_results["XGBoost_Forecast"] = forecast_values
                st.success("✅ XGBoost model completed")
            except Exception as e:
                st.error(f"❌ XGBoost failed: {str(e)}")
        
        if use_fallback:
            forecast_results["Fallback_Forecast"] = run_fallback_forecast(hist_df, 12, 1.0)
            st.success("✅ Fallback model completed")
        
        # Create ensemble if multiple models
        if use_ensemble and len(forecast_results) > 1:
            ensemble_forecast = create_simple_ensemble(forecast_results)
            forecast_results["Ensemble_Forecast"] = ensemble_forecast
            st.success("✅ Ensemble model created")
        
        # Create results dataframe
        forecast_dates = pd.date_range(
            start=f"{forecast_year}-01-01",
            end=f"{forecast_year}-12-01",
            freq='MS'
        )
        
        result_df = pd.DataFrame({
            "Month": forecast_dates,
            **forecast_results
        })
        
        # Display results
        st.header("📊 Forecast Results")
        
        # Summary statistics
        if forecast_results:
            total_forecast = list(forecast_results.values())[-1].sum()  # Use last model (or ensemble)
            st.metric("📈 Total Forecast", f"{total_forecast:,.0f}")
        
        # Show forecast table
        st.subheader("📋 Detailed Forecasts")
        
        display_df = result_df.copy()
        display_df['Month'] = display_df['Month'].dt.strftime('%b %Y')
        
        # Round numeric columns
        numeric_cols = [col for col in display_df.columns if col != 'Month']
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "—"
            )
        
        st.dataframe(display_df, use_container_width=True)
        
        # Visualization
        st.subheader("📈 Forecast Visualization")
        
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=hist_df['Month'],
            y=hist_df['Sales_Original'],
            mode='lines',
            name='Historical',
            line=dict(color='gray', width=2)
        ))
        
        # Add forecasts
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (col_name, values) in enumerate(forecast_results.items()):
            model_name = col_name.replace('_Forecast', '')
            fig.add_trace(go.Scatter(
                x=result_df['Month'],
                y=values,
                mode='lines+markers',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=f'Sales Forecast - {forecast_year}',
            xaxis_title='Date',
            yaxis_title='Sales',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download section
        st.header("📥 Export Results")
        
        csv_data = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Forecasts (CSV)",
            data=csv_data,
            file_name=f"Forecasts_{forecast_year}.csv",
            mime="text/csv"
        )
        
        st.success("✅ Forecasting complete!")


if __name__ == "__main__":
    main()
