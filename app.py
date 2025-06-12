import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import hashlib
from datetime import datetime, timedelta
import warnings
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

# Core forecasting functions
def simple_moving_average(data, periods=3):
    """Simple moving average forecast"""
    if len(data) < periods:
        return [data.mean()] * 12
    
    last_values = data.tail(periods).mean()
    return [last_values] * 12

def exponential_smoothing(data, alpha=0.3):
    """Simple exponential smoothing"""
    if len(data) == 0:
        return [0] * 12
    
    smoothed = [data.iloc[0]]
    for i in range(1, len(data)):
        smoothed.append(alpha * data.iloc[i] + (1 - alpha) * smoothed[-1])
    
    last_value = smoothed[-1]
    return [last_value] * 12

def linear_trend_forecast(data):
    """Linear trend forecasting"""
    if len(data) < 2:
        return [data.mean() if len(data) > 0 else 0] * 12
    
    x = np.arange(len(data))
    y = data.values
    
    # Simple linear regression
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    intercept = y.mean() - slope * x.mean()
    
    # Forecast next 12 periods
    forecast = []
    for i in range(12):
        pred = intercept + slope * (len(data) + i)
        forecast.append(max(0, pred))  # Ensure non-negative
    
    return forecast

def seasonal_naive_forecast(data, season_length=12):
    """Seasonal naive forecasting"""
    if len(data) < season_length:
        return [data.mean()] * 12
    
    last_season = data.tail(season_length).values
    forecast = []
    for i in range(12):
        forecast.append(last_season[i % len(last_season)])
    
    return forecast

def run_simple_forecast(data, periods=12, scaling=1.0):
    """Run simple ensemble forecast"""
    try:
        # Multiple simple methods
        ma_forecast = simple_moving_average(data)
        es_forecast = exponential_smoothing(data)
        trend_forecast = linear_trend_forecast(data)
        seasonal_forecast = seasonal_naive_forecast(data)
        
        # Simple ensemble (average)
        ensemble = []
        for i in range(periods):
            avg_pred = np.mean([
                ma_forecast[i],
                es_forecast[i], 
                trend_forecast[i],
                seasonal_forecast[i]
            ])
            ensemble.append(avg_pred * scaling)
        
        return ensemble, 0.15  # Return forecast and dummy score
        
    except Exception:
        # Ultimate fallback
        mean_value = data.mean() if len(data) > 0 else 1000
        return [mean_value * scaling] * periods, 999

def load_excel_data(file_content):
    """Load and process Excel data"""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Show data preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head())
        
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to load Excel file: {str(e)}")
        return None

def process_data(df):
    """Process and aggregate data"""
    if df is None:
        return None
    
    # Let user select columns
    st.subheader("📋 Column Selection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_col = st.selectbox("📅 Date/Month Column", df.columns)
    
    with col2:
        sales_col = st.selectbox("💰 Sales Column", df.columns)
    
    with col3:
        part_col = st.selectbox("🔧 Part Column (Optional)", [None] + list(df.columns))
    
    if not date_col or not sales_col:
        st.warning("⚠️ Please select Date and Sales columns")
        return None
    
    # Handle multiple parts
    if part_col and len(df[part_col].unique()) > 1:
        st.info(f"📊 Found {len(df[part_col].unique())} unique parts")
        
        agg_method = st.selectbox(
            "🎯 Aggregation Method",
            ["Sum all parts", "Largest part only", "Average per part"]
        )
        
        if agg_method == "Sum all parts":
            processed_df = df.groupby(date_col)[sales_col].sum().reset_index()
        elif agg_method == "Largest part only":
            largest_part = df.groupby(part_col)[sales_col].sum().idxmax()
            processed_df = df[df[part_col] == largest_part][[date_col, sales_col]]
        else:  # Average
            processed_df = df.groupby(date_col)[sales_col].mean().reset_index()
    else:
        processed_df = df[[date_col, sales_col]].copy()
    
    # Standardize column names
    processed_df.columns = ['Month', 'Sales']
    
    # Convert to datetime
    try:
        processed_df['Month'] = pd.to_datetime(processed_df['Month'])
    except Exception:
        st.error("❌ Could not convert dates")
        return None
    
    # Sort and clean
    processed_df = processed_df.sort_values('Month').reset_index(drop=True)
    processed_df = processed_df.dropna()
    
    # Remove duplicates by aggregating
    if processed_df.duplicated(subset=['Month']).any():
        processed_df = processed_df.groupby('Month')['Sales'].sum().reset_index()
    
    if len(processed_df) < 3:
        st.error("❌ Need at least 3 months of data")
        return None
    
    st.success(f"✅ Processed {len(processed_df)} months of data")
    return processed_df

def create_basic_plot(historical_data, forecast_data, forecast_year):
    """Create simple forecast plot"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data['Month'],
        y=historical_data['Sales'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Forecast data
    forecast_dates = pd.date_range(
        start=f"{forecast_year}-01-01", 
        periods=len(forecast_data), 
        freq='MS'
    )
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_data,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="📈 Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Sales"
    )
    
    return fig

def main():
    """Main application"""
    st.title("🚀 Sales Forecasting System")
    st.markdown("**Simple, reliable sales forecasting**")
    
    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    
    forecast_year = st.sidebar.selectbox(
        "📅 Forecast Year",
        [2024, 2025, 2026, 2027]
    )
    
    scaling_factor = st.sidebar.number_input(
        "🔧 Scale Factor",
        value=1.0,
        step=0.001,
        format="%.6f",
        help="Use 0.001 to divide by 1000"
    )
    
    # Quick scale buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("÷1000"):
            scaling_factor = 0.001
    with col2:
        if st.button("÷1M"):
            scaling_factor = 0.000001
    
    # File upload
    st.header("📁 Upload Data")
    
    uploaded_file = st.file_uploader(
        "Upload Excel file",
        type=['xlsx', 'xls'],
        help="Excel file with Date and Sales columns"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload an Excel file to start")
        return
    
    # Load and process data
    file_content = uploaded_file.read()
    df = load_excel_data(file_content)
    
    if df is None:
        return
    
    processed_df = process_data(df)
    
    if processed_df is None:
        return
    
    # Show data summary
    st.header("📊 Data Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📅 Months", len(processed_df))
    
    with col2:
        avg_sales = processed_df['Sales'].mean()
        st.metric("💰 Avg Sales", f"{avg_sales:,.0f}")
    
    with col3:
        total_sales = processed_df['Sales'].sum()
        st.metric("📊 Total", f"{total_sales:,.0f}")
    
    # Show current scaling info
    if scaling_factor != 1.0:
        if scaling_factor < 1.0:
            reduction = 1 / scaling_factor
            st.info(f"🔧 Forecasts will be REDUCED (÷{reduction:.0f})")
        else:
            st.info(f"🔧 Forecasts will be INCREASED (×{scaling_factor:.0f})")
    
    # Run forecast
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Creating forecast..."):
            
            # Run simple ensemble forecast
            forecast_values, score = run_simple_forecast(
                processed_df['Sales'], 
                12, 
                scaling_factor
            )
            
            # Show results
            st.header("📈 Forecast Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_forecast = sum(forecast_values)
                st.metric("🎯 Total Forecast", f"{total_forecast:,.0f}")
            
            with col2:
                avg_monthly = total_forecast / 12
                st.metric("📅 Monthly Avg", f"{avg_monthly:,.0f}")
            
            with col3:
                historical_total = processed_df['Sales'].tail(12).sum()
                growth = ((total_forecast - historical_total) / historical_total * 100)
                st.metric("📊 Growth", f"{growth:+.1f}%")
            
            # Create and show plot
            fig = create_basic_plot(processed_df, forecast_values, forecast_year)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Monthly Forecast")
            
            forecast_dates = pd.date_range(
                start=f"{forecast_year}-01-01",
                periods=12,
                freq='MS'
            )
            
            results_df = pd.DataFrame({
                'Month': forecast_dates.strftime('%Y-%m'),
                'Forecast': [round(x, 0) for x in forecast_values]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Download option
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                f"forecast_{forecast_year}.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()
