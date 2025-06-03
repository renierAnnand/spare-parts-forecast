# app.py

"""
Streamlit app for forecasting spare‐parts sales (monthly) and comparing 2024 forecasts
against actuals. Uses SARIMA (via pmdarima) and Prophet, then creates a simple ensemble.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('pmdarima').setLevel(logging.WARNING)

# ----------------------------
# Helper functions
# ----------------------------

@st.cache_data
def load_data(uploaded_file):
    """
    Load the uploaded Excel file into a DataFrame, parse dates, and aggregate monthly sales.
    Expects a sheet with at least columns: 'Month' and 'Sales'.
    """
    try:
        df = pd.read_excel(uploaded_file)
        
        # Validate required columns
        required_cols = ['Month', 'Sales']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure 'Month' is parsed as datetime
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
        
        # Check for invalid dates
        if df['Month'].isna().any():
            raise ValueError("Some dates in 'Month' column could not be parsed")
        
        # Ensure 'Sales' is numeric and non-negative
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
        if df['Sales'].isna().any():
            st.warning("Some sales values could not be converted to numbers. These will be treated as 0.")
            df['Sales'] = df['Sales'].fillna(0)
        
        # Handle negative sales
        if (df['Sales'] < 0).any():
            st.warning("Negative sales values detected. Converting to absolute values.")
            df['Sales'] = df['Sales'].abs()
        
        # Aggregate total sales per month across all parts
        monthly = df.groupby('Month', as_index=False)['Sales'].sum().sort_values('Month')
        
        # Remove months with zero sales
        monthly = monthly[monthly['Sales'] > 0]
        
        return monthly
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def determine_sarima_order(ts, seasonal_period=12):
    """
    Simple heuristic to determine SARIMA order parameters.
    Returns (p,d,q)(P,D,Q,s) tuple.
    """
    try:
        # Default parameters that work well for most sales data
        return (1, 1, 1), (1, 1, 1, seasonal_period)
    except:
        return (0, 1, 0), (0, 1, 0, seasonal_period)

def train_sarima(train_series, seasonal_period=12):
    """
    Fit a SARIMA model on the provided train_series using statsmodels.
    Returns the fitted model or None if training fails.
    """
    try:
        # Validate input
        if len(train_series) < seasonal_period * 2:
            raise ValueError(f"Need at least {seasonal_period * 2} data points for seasonal modeling")
        
        # Check for constant series
        if train_series.var() == 0:
            raise ValueError("Sales data is constant - cannot fit SARIMA model")
        
        # Determine SARIMA parameters
        order, seasonal_order = determine_sarima_order(train_series, seasonal_period)
        
        # Fit SARIMA model
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        return fitted_model
        
    except Exception as e:
        st.error(f"SARIMA training failed: {str(e)}")
        return None

@st.cache_data
def forecast_sarima(_model, n_periods, last_date):
    """
    Given a fitted SARIMA model, forecast n_periods ahead.
    last_date: Timestamp of the last training month (e.g., 2023-12-01).
    Returns a DataFrame with 'Month' and 'SARIMA_Forecast'.
    """
    if _model is None:
        return None
    
    try:
        # Get forecast
        forecast_result = _model.forecast(steps=n_periods)
        conf_int = _model.get_forecast(steps=n_periods).conf_int()
        
        # Ensure predictions are non-negative
        preds = np.maximum(forecast_result.values, 0)
        
        # Build the future months index
        future_months = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), 
                                      periods=n_periods, freq='MS')
        
        df_forecast = pd.DataFrame({
            'Month': future_months,
            'SARIMA_Forecast': preds,
            'SARIMA_Lower': np.maximum(conf_int.iloc[:, 0], 0),
            'SARIMA_Upper': conf_int.iloc[:, 1]
        })
        return df_forecast
    except Exception as e:
        st.error(f"SARIMA forecasting failed: {str(e)}")
        return None

def train_prophet(train_df):
    """
    Fit a Prophet model on the provided train_df with columns ['ds','y'].
    Returns the fitted model or None if training fails.
    """
    try:
        # Validate input
        if len(train_df) < 24:  # Need at least 2 years for good seasonality
            st.warning("Limited data for Prophet. Results may be unreliable.")
        
        m = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=False, 
            daily_seasonality=False,
            seasonality_mode='multiplicative',  # Often better for sales data
            changepoint_prior_scale=0.05  # More conservative changepoint detection
        )
        
        # Suppress Prophet's verbose output
        with st.spinner("Training Prophet model..."):
            m.fit(train_df)
        
        return m
    except Exception as e:
        st.error(f"Prophet training failed: {str(e)}")
        return None

@st.cache_data
def forecast_prophet(_model, periods, last_date):
    """
    Given a fitted Prophet model, forecast `periods` months ahead.
    last_date: Timestamp of the last training month (e.g., 2023-12-01).
    Returns a DataFrame with 'Month' and 'Prophet_Forecast'.
    """
    if _model is None:
        return None
    
    try:
        # Create a dataframe with future periods in monthly frequency
        future = pd.DataFrame({'ds': pd.date_range(start=last_date + pd.offsets.MonthBegin(1),
                                                   periods=periods,
                                                   freq='MS')})
        forecast = _model.predict(future)
        
        # Extract relevant columns and ensure non-negative forecasts
        df_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        df_forecast['yhat'] = np.maximum(df_forecast['yhat'], 0)
        df_forecast = df_forecast.rename(columns={
            'ds': 'Month', 
            'yhat': 'Prophet_Forecast',
            'yhat_lower': 'Prophet_Lower',
            'yhat_upper': 'Prophet_Upper'
        })
        return df_forecast
    except Exception as e:
        st.error(f"Prophet forecasting failed: {str(e)}")
        return None

def calculate_accuracy_metrics(actual, forecast):
    """Calculate MAPE, MAE, and RMSE for actual vs forecast."""
    if len(actual) == 0 or len(forecast) == 0:
        return None
    
    # Remove NaN values
    mask = ~(pd.isna(actual) | pd.isna(forecast))
    actual = actual[mask]
    forecast = forecast[mask]
    
    if len(actual) == 0:
        return None
    
    # Calculate metrics
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    
    return {'MAPE': mape, 'MAE': mae, 'RMSE': rmse}

# ----------------------------
# Streamlit App Layout
# ----------------------------

st.set_page_config(page_title="Spare‐Parts Sales Forecast 2024", layout="wide")
st.title("📈 Spare‐Parts Sales Forecast vs. Actual (2024)")

st.markdown(
    """
    **Instructions:**
    1. Upload your Excel file with columns 'Month' and 'Sales' (and any other part columns).
    2. The app will aggregate to total monthly sales, train SARIMA and Prophet on data through 2023,
       then forecast Jan–Dec 2024.
    3. You'll see a table comparing forecasts to actual 2024 sales (if present), plus accuracy metrics.
    """
)

# Sidebar for parameters
st.sidebar.header("Model Parameters")
seasonal_period = st.sidebar.selectbox("Seasonal Period", [12, 4, 6], index=0, 
                                       help="12 for monthly, 4 for quarterly, 6 for bi-monthly")
forecast_year = st.sidebar.selectbox("Forecast Year", [2024, 2025], index=0)

uploaded_file = st.file_uploader(
    "Upload Excel file with columns 'Month' and 'Sales'",
    type=["xlsx", "xls"]
)

if uploaded_file:
    # Load and aggregate data
    monthly_df = load_data(uploaded_file)
    
    if monthly_df is not None:
        st.success(f"Data loaded successfully: {len(monthly_df)} months of data")
        
        # Show data preview
        with st.expander("Data Preview"):
            st.dataframe(monthly_df.head(10))
            st.write(f"Date range: {monthly_df['Month'].min().strftime('%Y-%m')} to {monthly_df['Month'].max().strftime('%Y-%m')}")
            st.write(f"Total months of data: {len(monthly_df)}")
            
            # Debug: Show what data exists for the forecast year
            forecast_year_data = monthly_df[
                (monthly_df['Month'] >= forecast_start) & 
                (monthly_df['Month'] < forecast_end)
            ]
            if len(forecast_year_data) > 0:
                st.write(f"**Data found for {forecast_year}:**")
                st.dataframe(forecast_year_data)
            else:
                st.write(f"**No data found for {forecast_year}**")
        
        # Dynamic date splitting based on forecast year
        train_cutoff = pd.Timestamp(f"{forecast_year-1}-12-01")
        forecast_start = pd.Timestamp(f"{forecast_year}-01-01")
        forecast_end = pd.Timestamp(f"{forecast_year+1}-01-01")
        
        # Ensure we have enough training data
        if monthly_df['Month'].max() < train_cutoff:
            st.error(f"Your data must include at least through December {forecast_year-1} for training. Latest data: {monthly_df['Month'].max().strftime('%Y-%m')}")
        else:
            # Split data
            train_df = monthly_df[monthly_df['Month'] <= train_cutoff].copy()
            actual_forecast_df = monthly_df[
                (monthly_df['Month'] >= forecast_start) &
                (monthly_df['Month'] < forecast_end)
            ].copy()
            
            if len(train_df) < seasonal_period * 2:
                st.error(f"Need at least {seasonal_period * 2} months of training data. Found: {len(train_df)}")
            else:
                actual_forecast_df = actual_forecast_df.rename(columns={'Sales': f'Actual_{forecast_year}'})
                
                col1, col2 = st.columns(2)
                
                # Train models
                with col1:
                    st.subheader("SARIMA Model")
                    train_series = train_df.set_index('Month')['Sales']
                    
                    with st.spinner("Training SARIMA..."):
                        sarima_model = train_sarima(train_series, seasonal_period)
                    
                    if sarima_model:
                        st.success("SARIMA training completed")
                        st.write("Model: SARIMA(1,1,1)(1,1,1)[12]")
                        sarima_forecast_df = forecast_sarima(sarima_model, n_periods=12, last_date=train_df['Month'].max())
                    else:
                        sarima_forecast_df = None
                
                with col2:
                    st.subheader("Prophet Model")
                    prophet_train = train_df.rename(columns={'Month': 'ds', 'Sales': 'y'})[['ds', 'y']]
                    
                    prophet_model = train_prophet(prophet_train)
                    
                    if prophet_model:
                        st.success("Prophet training completed")
                        prophet_forecast_df = forecast_prophet(prophet_model, periods=12, last_date=train_df['Month'].max())
                    else:
                        prophet_forecast_df = None
                
                # Combine results
                if sarima_forecast_df is not None and prophet_forecast_df is not None:
                    # Merge forecasts
                    forecasts_df = pd.merge(sarima_forecast_df, prophet_forecast_df, on='Month', how='inner')
                    
                    # Simple ensemble = average of SARIMA and Prophet
                    forecasts_df['Ensemble_Forecast'] = (
                        forecasts_df['SARIMA_Forecast'] + forecasts_df['Prophet_Forecast']
                    ) / 2
                    
                    # Create a complete date range for the forecast year
                    complete_dates = pd.date_range(
                        start=forecast_start, 
                        end=pd.Timestamp(f"{forecast_year}-12-01"), 
                        freq='MS'
                    )
                    complete_df = pd.DataFrame({'Month': complete_dates})
                    
                    # Merge forecasts with complete date range
                    result_df = pd.merge(complete_df, forecasts_df, on='Month', how='left')
                    
                    # Merge with actual data - this ensures all actual data is included
                    if len(actual_forecast_df) > 0:
                        st.write(f"**Merging actual data:** {len(actual_forecast_df)} rows")
                        st.write("Actual data to merge:")
                        st.dataframe(actual_forecast_df[['Month', f'Actual_{forecast_year}']])
                        
                        result_df = pd.merge(result_df, actual_forecast_df[['Month', f'Actual_{forecast_year}']], 
                                           on='Month', how='left')
                        
                        # Verify merge worked
                        actual_count = result_df[f'Actual_{forecast_year}'].notna().sum()
                        st.write(f"**After merge:** {actual_count} months have actual data")
                    else:
                        # Add empty actual column if no actual data
                        result_df[f'Actual_{forecast_year}'] = np.nan
                        st.write("No actual data to merge - added empty column")
                    
                    # Round numeric columns for display
                    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
                    result_df[numeric_cols] = result_df[numeric_cols].round(2)
                    
                    # Display results
                    st.subheader(f"Forecast vs. Actual ({forecast_year})")
                    st.dataframe(result_df.set_index('Month'))
                    
                    # Calculate accuracy metrics if we have actuals
                    if f'Actual_{forecast_year}' in result_df.columns and not result_df[f'Actual_{forecast_year}'].isna().all():
                        st.subheader("Accuracy Metrics")
                        col1, col2, col3 = st.columns(3)
                        
                        actual_col = result_df[f'Actual_{forecast_year}'].dropna()
                        ensemble_col = result_df.loc[actual_col.index, 'Ensemble_Forecast']
                        
                        metrics = calculate_accuracy_metrics(actual_col, ensemble_col)
                        if metrics:
                            col1.metric("MAPE", f"{metrics['MAPE']:.1f}%")
                            col2.metric("MAE", f"{metrics['MAE']:.0f}")
                            col3.metric("RMSE", f"{metrics['RMSE']:.0f}")
                    
                    # Plotting
                    st.subheader("Visual Comparison")
                    plot_cols = [f'Actual_{forecast_year}', 'SARIMA_Forecast', 'Prophet_Forecast', 'Ensemble_Forecast']
                    plot_df = result_df.set_index('Month')[plot_cols]
                    st.line_chart(plot_df)
                    
                    # Provide download
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download Forecast vs Actual (CSV)",
                        data=csv,
                        file_name=f"forecast_vs_actual_{forecast_year}.csv",
                        mime="text/csv"
                    )
                
                elif sarima_forecast_df is not None:
                    st.warning("Only SARIMA model succeeded. Showing SARIMA results only.")
                    # Create result with just SARIMA and actual data
                    complete_dates = pd.date_range(start=forecast_start, end=pd.Timestamp(f"{forecast_year}-12-01"), freq='MS')
                    result_df = pd.DataFrame({'Month': complete_dates})
                    result_df = pd.merge(result_df, sarima_forecast_df, on='Month', how='left')
                    
                    if len(actual_forecast_df) > 0:
                        result_df = pd.merge(result_df, actual_forecast_df[['Month', f'Actual_{forecast_year}']], on='Month', how='left')
                    else:
                        result_df[f'Actual_{forecast_year}'] = np.nan
                    
                    # Display results
                    st.dataframe(result_df.set_index('Month'))
                    
                elif prophet_forecast_df is not None:
                    st.warning("Only Prophet model succeeded. Showing Prophet results only.")
                    # Create result with just Prophet and actual data
                    complete_dates = pd.date_range(start=forecast_start, end=pd.Timestamp(f"{forecast_year}-12-01"), freq='MS')
                    result_df = pd.DataFrame({'Month': complete_dates})
                    result_df = pd.merge(result_df, prophet_forecast_df, on='Month', how='left')
                    
                    if len(actual_forecast_df) > 0:
                        result_df = pd.merge(result_df, actual_forecast_df[['Month', f'Actual_{forecast_year}']], on='Month', how='left')
                    else:
                        result_df[f'Actual_{forecast_year}'] = np.nan
                    
                    # Display results
                    st.dataframe(result_df.set_index('Month'))
                else:
                    st.error("Both models failed to train. Please check your data quality and try again.")
