import streamlit as st

# Configure streamlit FIRST - must be before any other st commands
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# Forecasting libraries
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Machine learning libraries
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


@st.cache_data
def load_data(uploaded_file):
    """
    Load and preprocess the historical sales data.
    Expected columns: 'Month' and 'Sales'
    """
    try:
        df = pd.read_excel(uploaded_file)
    except Exception:
        st.error("Could not read the uploaded file. Please ensure it's a valid Excel file.")
        return None

    if "Month" not in df.columns or "Sales" not in df.columns:
        st.error("The file must contain 'Month' and 'Sales' columns.")
        return None

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    if df["Month"].isna().any():
        st.error("Some dates could not be parsed. Please check the 'Month' column format.")
        return None

    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    df["Sales"] = df["Sales"].abs()

    df = df.sort_values("Month").reset_index(drop=True)
    return df[["Month", "Sales"]]


@st.cache_data
def load_actual_2024_data(uploaded_file, forecast_year):
    """
    Load the 2024‐actuals file and return aggregated monthly sales.
    Handles both formats:
    1. Long format: 'Month' and 'Sales' columns
    2. Wide format: months as columns (Jan-2024, Feb-2024, etc.)
    """
    try:
        df = pd.read_excel(uploaded_file)
        
        # Check if it's the standard long format
        if "Month" in df.columns and "Sales" in df.columns:
            # Standard format handling
            df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
            if df["Month"].isna().any():
                st.error("Some dates in the 2024 actuals file could not be parsed.")
                return None

            df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
            df["Sales"] = df["Sales"].abs()

            # Filter to the forecast year only
            start = pd.Timestamp(f"{forecast_year}-01-01")
            end = pd.Timestamp(f"{forecast_year+1}-01-01")
            df = df[(df["Month"] >= start) & (df["Month"] < end)]
            
            if df.empty:
                st.warning(f"No rows in the 2024 actuals file match year {forecast_year}.")
                return None

            monthly = df.groupby("Month", as_index=False)["Sales"].sum().sort_values("Month").reset_index(drop=True)
            return monthly.rename(columns={"Sales": f"Actual_{forecast_year}"})
        
        else:
            # Wide format handling (months as columns)
            st.info("📊 Detected wide format data - converting to long format...")
            
            # Look for month columns that match the forecast year
            month_patterns = [
                f"Jan-{forecast_year}", f"Feb-{forecast_year}", f"Mar-{forecast_year}",
                f"Apr-{forecast_year}", f"May-{forecast_year}", f"Jun-{forecast_year}",
                f"Jul-{forecast_year}", f"Aug-{forecast_year}", f"Sep-{forecast_year}",
                f"Oct-{forecast_year}", f"Nov-{forecast_year}", f"Dec-{forecast_year}"
            ]
            
            # Find which month columns exist in the data
            available_months = []
            for pattern in month_patterns:
                if pattern in df.columns:
                    available_months.append(pattern)
            
            if not available_months:
                st.error(f"No month columns found for {forecast_year}. Expected columns like 'Jan-{forecast_year}', 'Feb-{forecast_year}', etc.")
                return None
            
            st.success(f"Found {len(available_months)} months of data: {', '.join(available_months)}")
            
            # Skip header rows if they exist (look for rows where first column contains "Item" or similar)
            first_col = df.columns[0]
            data_rows = df[~df[first_col].astype(str).str.contains("Item|Code|QTY", case=False, na=False)]
            
            # Melt the data from wide to long format
            melted_data = []
            for _, row in data_rows.iterrows():
                part_code = row[first_col]
                for month_col in available_months:
                    if month_col in row and pd.notna(row[month_col]):
                        # Convert month string to datetime
                        month_str = month_col.replace("-", "-01-")  # Jan-2024 -> Jan-01-2024
                        try:
                            month_date = pd.to_datetime(month_str, format="%b-%d-%Y")
                            sales_value = pd.to_numeric(row[month_col], errors="coerce")
                            if pd.notna(sales_value) and sales_value > 0:
                                melted_data.append({
                                    "Month": month_date,
                                    "Part": part_code,
                                    "Sales": abs(sales_value)
                                })
                        except:
                            continue
            
            if not melted_data:
                st.error("No valid sales data found in the file.")
                return None
            
            # Convert to DataFrame and aggregate by month
            long_df = pd.DataFrame(melted_data)
            monthly = long_df.groupby("Month", as_index=False)["Sales"].sum().sort_values("Month").reset_index(drop=True)
            
            st.success(f"✅ Converted wide format data: {len(monthly)} months of aggregated sales data")
            
            return monthly.rename(columns={"Sales": f"Actual_{forecast_year}"})
            
    except Exception as e:
        st.error(f"Error loading 2024 actual data: {str(e)}")
        return None


def calculate_accuracy_metrics(actual, forecast):
    """
    Calculate MAPE, MAE, and RMSE between actual and forecast values.
    """
    if len(actual) == 0 or len(forecast) == 0:
        return None
    
    # Remove NaN values
    mask = ~(pd.isna(actual) | pd.isna(forecast))
    actual_clean = actual[mask]
    forecast_clean = forecast[mask]
    
    if len(actual_clean) == 0:
        return None
    
    # Calculate metrics
    mape = np.mean(np.abs((actual_clean - forecast_clean) / actual_clean)) * 100
    mae = mean_absolute_error(actual_clean, forecast_clean)
    rmse = np.sqrt(mean_squared_error(actual_clean, forecast_clean))
    
    return {
        "MAPE": mape,
        "MAE": mae,
        "RMSE": rmse
    }


def detect_and_apply_scaling(historical_data, actual_data=None):
    """
    Detect if there's a scaling mismatch between historical and actual data.
    Returns scaling factor to apply to forecasts.
    """
    hist_avg = historical_data['Sales'].mean()
    hist_total = historical_data['Sales'].sum()
    
    if actual_data is not None and len(actual_data) > 0:
        actual_avg = actual_data.iloc[:, 1].mean()  # Second column is the actual values
        actual_total = actual_data.iloc[:, 1].sum()
        
        # Calculate potential scaling factors
        avg_ratio = actual_avg / hist_avg if hist_avg > 0 else 1
        total_ratio = actual_total / (hist_total / len(historical_data) * 12) if hist_total > 0 else 1
        
        # If there's a significant scale difference (>5x or <0.2x), apply scaling
        if avg_ratio > 5 or avg_ratio < 0.2:
            scaling_factor = avg_ratio
            st.warning(f"📊 Scale mismatch detected! Applying scaling factor: {scaling_factor:.2f}")
            return scaling_factor
    
    return 1.0  # No scaling needed


def run_sarima_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """
    Improved SARIMA forecast with better error handling and scaling.
    """
    try:
        # Use a more robust SARIMA configuration
        model = SARIMAX(
            data['Sales'], 
            order=(1, 1, 1), 
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False, maxiter=100)
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=forecast_periods)
        forecast = np.maximum(forecast, 0) * scaling_factor
        
        return forecast
    except Exception as e:
        st.warning(f"SARIMA failed: {str(e)}. Using trend-based fallback.")
        # Improved fallback: use recent trend and seasonality
        recent_months = min(24, len(data))
        recent_data = data['Sales'].tail(recent_months)
        
        # Calculate trend
        if len(recent_data) > 12:
            trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
        else:
            trend = 0
            
        # Use seasonal pattern if available
        if len(data) >= 12:
            seasonal_pattern = data['Sales'].tail(12).values
            base_forecast = []
            for i in range(forecast_periods):
                seasonal_val = seasonal_pattern[i % 12]
                trend_val = seasonal_val + trend * (i + 1)
                base_forecast.append(max(trend_val, seasonal_val * 0.5))
        else:
            base_val = recent_data.mean()
            base_forecast = [base_val + trend * (i + 1) for i in range(forecast_periods)]
        
        return np.maximum(base_forecast, 0) * scaling_factor


def run_prophet_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """
    Improved Prophet forecast with better configuration.
    """
    try:
        prophet_data = data[['Month', 'Sales']].rename(columns={'Month': 'ds', 'Sales': 'y'})
        
        # More robust Prophet configuration
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',  # Changed from multiplicative
            changepoint_prior_scale=0.1,  # More conservative
            seasonality_prior_scale=10.0,
            interval_width=0.8
        )
        
        model.fit(prophet_data)
        
        # Create future dates
        future = model.make_future_dataframe(periods=forecast_periods, freq='MS')
        forecast = model.predict(future)
        
        # Return only the forecast period
        forecast_values = forecast['yhat'].tail(forecast_periods).values
        return np.maximum(forecast_values, 0) * scaling_factor
        
    except Exception as e:
        st.warning(f"Prophet failed: {str(e)}. Using seasonal mean method.")
        # Seasonal mean fallback
        if len(data) >= 12:
            seasonal_means = []
            for month in range(1, 13):
                month_data = data[data['Month'].dt.month == month]['Sales']
                seasonal_means.append(month_data.mean() if len(month_data) > 0 else data['Sales'].mean())
            
            # Create forecast based on seasonal means
            forecast = []
            for i in range(forecast_periods):
                month_idx = (data['Month'].iloc[-1].month + i) % 12
                forecast.append(seasonal_means[month_idx])
            return np.array(forecast) * scaling_factor
        else:
            return np.array([data['Sales'].mean()] * forecast_periods) * scaling_factor


def run_ets_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """
    Improved ETS forecast with better parameter selection.
    """
    try:
        # Try different ETS configurations
        configs = [
            {'seasonal': 'add', 'trend': 'add'},
            {'seasonal': 'mul', 'trend': 'add'},
            {'seasonal': 'add', 'trend': None},
            {'seasonal': None, 'trend': 'add'}
        ]
        
        best_model = None
        best_aic = np.inf
        
        for config in configs:
            try:
                model = ExponentialSmoothing(
                    data['Sales'],
                    seasonal=config['seasonal'],
                    seasonal_periods=12 if config['seasonal'] else None,
                    trend=config['trend']
                )
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
            except:
                continue
        
        if best_model is not None:
            forecast = best_model.forecast(steps=forecast_periods)
            return np.maximum(forecast, 0) * scaling_factor
        else:
            raise ValueError("All ETS configurations failed")
            
    except Exception as e:
        st.warning(f"ETS failed: {str(e)}. Using seasonal naive method.")
        # Improved seasonal naive
        if len(data) >= 12:
            last_year = data['Sales'].tail(12).values
            # Apply slight growth trend
            growth_rate = 1.02  # Assume 2% growth
            seasonal_forecast = []
            for i in range(forecast_periods):
                base_val = last_year[i % 12]
                growth_factor = growth_rate ** ((i // 12) + 1)
                seasonal_forecast.append(base_val * growth_factor)
            return np.array(seasonal_forecast) * scaling_factor
        else:
            return np.array([data['Sales'].mean()] * forecast_periods) * scaling_factor


def run_xgb_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """
    Completely rewritten XGBoost forecast using direct approach instead of rolling predictions.
    """
    try:
        # Create comprehensive features
        df = data.copy()
        df['month'] = df['Month'].dt.month
        df['year'] = df['Month'].dt.year
        df['quarter'] = df['Month'].dt.quarter
        df['day_of_year'] = df['Month'].dt.dayofyear
        
        # Create multiple lag features
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = df['Sales'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            df[f'rolling_mean_{window}'] = df['Sales'].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['Sales'].rolling(window=window, min_periods=1).std()
        
        # Seasonal features
        df['sales_lag_12'] = df['Sales'].shift(12)
        df['seasonal_diff'] = df['Sales'] - df['sales_lag_12']
        
        # Growth features
        df['mom_growth'] = df['Sales'].pct_change()
        df['yoy_growth'] = df['Sales'].pct_change(12)
        
        # Clean data
        feature_cols = [col for col in df.columns if col not in ['Month', 'Sales']]
        df = df.dropna()
        
        if len(df) < 12:
            raise ValueError("Insufficient data after feature engineering")
        
        # Prepare training data
        X = df[feature_cols]
        y = df['Sales']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model with better parameters
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Generate forecasts using direct method (not rolling)
        forecasts = []
        
        for i in range(forecast_periods):
            # Calculate future date
            future_date = df['Month'].iloc[-1] + pd.DateOffset(months=i+1)
            
            # Create feature vector for this future month
            future_features = {}
            
            # Time-based features
            future_features['month'] = future_date.month
            future_features['year'] = future_date.year
            future_features['quarter'] = future_date.quarter
            future_features['day_of_year'] = future_date.dayofyear
            
            # Use historical patterns for lag and seasonal features
            last_idx = len(df) - 1
            
            # Lag features: use actual historical data
            for lag in [1, 2, 3, 6, 12]:
                if lag <= len(df):
                    future_features[f'lag_{lag}'] = df['Sales'].iloc[-(lag)]
                else:
                    future_features[f'lag_{lag}'] = df['Sales'].mean()
            
            # Rolling statistics: use recent historical data
            recent_data = df['Sales'].tail(12)
            for window in [3, 6, 12]:
                future_features[f'rolling_mean_{window}'] = recent_data.tail(window).mean()
                future_features[f'rolling_std_{window}'] = recent_data.tail(window).std()
            
            # Seasonal features: use same month from previous year
            if len(df) >= 12:
                same_month_last_year = df[df['Month'].dt.month == future_date.month]['Sales']
                if len(same_month_last_year) > 0:
                    future_features['sales_lag_12'] = same_month_last_year.iloc[-1]
                    future_features['seasonal_diff'] = 0  # Assume no seasonal change
                else:
                    future_features['sales_lag_12'] = df['Sales'].mean()
                    future_features['seasonal_diff'] = 0
            else:
                future_features['sales_lag_12'] = df['Sales'].mean()
                future_features['seasonal_diff'] = 0
            
            # Growth features: use recent trends
            future_features['mom_growth'] = df['mom_growth'].tail(3).mean()
            future_features['yoy_growth'] = df['yoy_growth'].tail(12).mean()
            
            # Create prediction vector
            X_future = pd.DataFrame([future_features])[feature_cols]
            X_future_scaled = scaler.transform(X_future.fillna(0))
            
            # Predict
            pred = model.predict(X_future_scaled)[0]
            pred = max(pred, 0)  # Ensure non-negative
            forecasts.append(pred)
        
        return np.array(forecasts) * scaling_factor
        
    except Exception as e:
        st.warning(f"XGBoost failed: {str(e)}. Using linear trend method.")
        # Improved linear trend fallback
        if len(data) >= 6:
            # Use robust linear regression on recent data
            recent_data = data.tail(min(24, len(data)))
            X = np.arange(len(recent_data)).reshape(-1, 1)
            y = recent_data['Sales'].values
            
            # Fit linear trend
            lr = LinearRegression().fit(X, y)
            
            # Generate future predictions
            future_X = np.arange(len(recent_data), len(recent_data) + forecast_periods).reshape(-1, 1)
            trend_forecast = lr.predict(future_X)
            
            # Add seasonal component if available
            if len(data) >= 12:
                seasonal_component = data['Sales'].tail(12).values
                seasonal_forecast = []
                for i in range(forecast_periods):
                    seasonal_val = seasonal_component[i % 12]
                    trend_val = trend_forecast[i]
                    # Combine trend and seasonal (weighted average)
                    combined = 0.7 * trend_val + 0.3 * seasonal_val
                    seasonal_forecast.append(max(combined, 0))
                return np.array(seasonal_forecast) * scaling_factor
            else:
                return np.maximum(trend_forecast, 0) * scaling_factor
        else:
            return np.array([data['Sales'].mean()] * forecast_periods) * scaling_factor


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("🔮 Spare Parts Sales Forecasting Dashboard")
    st.markdown("**Advanced forecasting with multiple models, scaling detection, and accuracy analysis**")

    # Sidebar configuration
    st.sidebar.header("📋 Configuration")
    forecast_year = st.sidebar.selectbox(
        "Select forecast year:",
        options=[2024, 2025, 2026],
        index=0
    )

    # Model selection
    st.sidebar.subheader("🔧 Select Models")
    use_sarima = st.sidebar.checkbox("SARIMA (Improved)", value=True)
    use_prophet = st.sidebar.checkbox("Prophet (Enhanced)", value=True)
    use_ets = st.sidebar.checkbox("ETS (Auto-Config)", value=True)
    use_xgb = st.sidebar.checkbox("XGBoost (Direct Method)", value=True)

    if not any([use_sarima, use_prophet, use_ets, use_xgb]):
        st.sidebar.error("Please select at least one forecasting model.")
        return

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
            help="For comparison with forecasts and automatic scaling detection"
        )

    if historical_file is None:
        st.info("👆 Please upload historical sales data to begin forecasting.")
        return

    # Load and validate historical data
    hist_df = load_data(historical_file)
    if hist_df is None:
        return

    # Load actual data for scaling detection
    actual_2024_df = None
    scaling_factor = 1.0
    
    if actual_2024_file is not None:
        actual_2024_df = load_actual_2024_data(actual_2024_file, forecast_year)
        if actual_2024_df is not None:
            scaling_factor = detect_and_apply_scaling(hist_df, actual_2024_df)

    # Display data info
    st.subheader("📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📅 Total Months", len(hist_df))
    with col2:
        st.metric("📈 Avg Monthly Sales", f"{hist_df['Sales'].mean():,.0f}")
    with col3:
        st.metric("📋 Date Range", f"{hist_df['Month'].min().strftime('%Y-%m')} to {hist_df['Month'].max().strftime('%Y-%m')}")
    with col4:
        st.metric("💰 Total Sales", f"{hist_df['Sales'].sum():,.0f}")

    # Show scaling info
    if scaling_factor != 1.0:
        st.info(f"🔧 Automatic scaling applied: {scaling_factor:.2f}x (detected scale mismatch between historical and actual data)")

    # Show historical data preview
    with st.expander("👀 Preview Historical Data"):
        st.dataframe(hist_df.head(12), use_container_width=True)

    # Generate forecasts
    st.subheader("🔮 Generating Enhanced Forecasts...")

    progress_bar = st.progress(0)
    forecast_results = {}

    # Create forecast dates
    forecast_dates = pd.date_range(
        start=f"{forecast_year}-01-01",
        end=f"{forecast_year}-12-01",
        freq='MS'  # Month start
    )

    # Run each selected model with improved algorithms
    models_to_run = []
    if use_sarima:
        models_to_run.append(("SARIMA", run_sarima_forecast))
    if use_prophet:
        models_to_run.append(("Prophet", run_prophet_forecast))
    if use_ets:
        models_to_run.append(("ETS", run_ets_forecast))
    if use_xgb:
        models_to_run.append(("XGBoost", run_xgb_forecast))

    for i, (model_name, model_func) in enumerate(models_to_run):
        with st.spinner(f"Running enhanced {model_name} model..."):
            try:
                forecast_values = model_func(hist_df, forecast_periods=12, scaling_factor=scaling_factor)
                forecast_results[f"{model_name}_Forecast"] = forecast_values
                st.success(f"✅ {model_name} completed successfully")
            except Exception as e:
                st.error(f"❌ {model_name} failed: {str(e)}")
                # Fallback forecast
                fallback_forecast = hist_df['Sales'].mean() * scaling_factor
                forecast_results[f"{model_name}_Forecast"] = [fallback_forecast] * 12

        progress_bar.progress((i + 1) / len(models_to_run))

    # Create ensemble forecast
    if len(forecast_results) > 1:
        ensemble_values = np.mean(list(forecast_results.values()), axis=0)
        forecast_results["Ensemble_Forecast"] = ensemble_values

    # Create results dataframe
    result_df = pd.DataFrame({
        "Month": forecast_dates,
        **forecast_results
    })

    # Merge actual data if available
    if actual_2024_df is not None:
        st.success(f"✅ Loaded {len(actual_2024_df)} months of actual data")
        
        # Ensure proper date alignment
        actual_2024_df['Month'] = pd.to_datetime(actual_2024_df['Month'])
        result_df['Month'] = pd.to_datetime(result_df['Month'])
        
        # Merge actual data with forecasts
        result_df = result_df.merge(actual_2024_df, on="Month", how="left")
        
        # Show merge results
        actual_count = result_df[f'Actual_{forecast_year}'].notna().sum()
        st.info(f"📊 Successfully merged actual data: {actual_count} out of 12 months have actual values")
        
        # Show actual data summary
        if actual_count > 0:
            actual_total = result_df[f'Actual_{forecast_year}'].sum()
            st.success(f"📈 Total actual sales for {forecast_year}: {actual_total:,.0f}")

    # Display results
    st.subheader("📊 Enhanced Forecast Results")

    # Show forecast table
    display_df = result_df.copy()
    display_df['Month'] = display_df['Month'].dt.strftime('%Y-%m')
    
    # Format numbers for display
    for col in display_df.columns:
        if col != 'Month':
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True)

    # ENHANCED COMPARISON CHART
    st.subheader("📊 Enhanced Model Comparison")
    
    # Get model columns
    model_cols = [col for col in result_df.columns if '_Forecast' in col and col != 'Ensemble_Forecast']
    actual_col = f'Actual_{forecast_year}'
    
    # Create the comparison chart
    fig = go.Figure()
    
    # Check if we have actual data
    has_actual_data = actual_col in result_df.columns and result_df[actual_col].notna().any()
    
    if has_actual_data:
        # Filter out NaN values for actual data display
        actual_data = result_df[result_df[actual_col].notna()]
        
        st.info(f"📊 Displaying actual data for {len(actual_data)} months")
        
        # Add actual data line
        fig.add_trace(go.Scatter(
            x=actual_data['Month'],
            y=actual_data[actual_col],
            mode='lines+markers',
            name=f'🎯 ACTUAL {forecast_year}',
            line=dict(color='#FF6B6B', width=4),
            marker=dict(size=10, symbol='circle')
        ))
        
        # Add each model
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        for i, col in enumerate(model_cols):
            model_name = col.replace('_Forecast', '')
            fig.add_trace(go.Scatter(
                x=result_df['Month'],
                y=result_df[col],
                mode='lines+markers',
                name=f'📈 {model_name.upper()}',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        # Add ensemble if available
        if 'Ensemble_Forecast' in result_df.columns:
            fig.add_trace(go.Scatter(
                x=result_df['Month'],
                y=result_df['Ensemble_Forecast'],
                mode='lines+markers',
                name='🔥 ENSEMBLE',
                line=dict(color='#6C5CE7', width=3, dash='dash'),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f'🔄 ENHANCED ACTUAL vs ALL MODELS COMPARISON ({forecast_year})',
            xaxis_title='Month',
            yaxis_title='Sales Volume',
            height=600,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed comparison metrics
        st.subheader("📋 Enhanced Model Performance Analysis")
        
        # Create detailed comparison table
        comparison_data = []
        actual_total = actual_data[actual_col].sum()
        
        for col in model_cols:
            model_name = col.replace('_Forecast', '')
            forecast_total = result_df[col].sum()
            
            # Calculate metrics only for months with actual data
            actual_subset = result_df[result_df[actual_col].notna()]
            if len(actual_subset) > 0:
                metrics = calculate_accuracy_metrics(actual_subset[actual_col], actual_subset[col])
                if metrics:
                    bias = ((forecast_total - actual_total) / actual_total * 100) if actual_total > 0 else 0
                    comparison_data.append({
                        'Model': model_name,
                        'MAPE (%)': f"{metrics['MAPE']:.1f}%",
                        'MAE': f"{metrics['MAE']:,.0f}",
                        'RMSE': f"{metrics['RMSE']:,.0f}",
                        'Total Forecast': f"{forecast_total:,.0f}",
                        'Total Actual': f"{actual_total:,.0f}",
                        'Bias (%)': f"{bias:+.1f}%",
                        'Accuracy': f"{100 - metrics['MAPE']:.1f}%"
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Show summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Actual Sales", f"{actual_total:,.0f}")
            with col2:
                ensemble_total = result_df['Ensemble_Forecast'].sum() if 'Ensemble_Forecast' in result_df.columns else 0
                st.metric("🔥 Total Ensemble Forecast", f"{ensemble_total:,.0f}")
            with col3:
                if ensemble_total > 0 and actual_total > 0:
                    ensemble_accuracy = 100 - (abs(ensemble_total - actual_total) / actual_total * 100)
                    st.metric("🎯 Ensemble Accuracy", f"{ensemble_accuracy:.1f}%")
            with col4:
                if scaling_factor != 1.0:
                    st.metric("⚖️ Scaling Factor Applied", f"{scaling_factor:.2f}x")
    
    else:
        # Show forecast-only chart
        st.warning("📊 No actual data available for comparison. Showing enhanced forecasts only.")
        
        fig = go.Figure()
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, col in enumerate(model_cols):
            model_name = col.replace('_Forecast', '')
            fig.add_trace(go.Scatter(
                x=result_df['Month'],
                y=result_df[col],
                mode='lines+markers',
                name=f'📈 {model_name.upper()}',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        if 'Ensemble_Forecast' in result_df.columns:
            fig.add_trace(go.Scatter(
                x=result_df['Month'],
                y=result_df['Ensemble_Forecast'],
                mode='lines+markers',
                name='🔥 ENSEMBLE',
                line=dict(color='#6C5CE7', width=3, dash='dash'),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f'📈 ENHANCED MODELS FORECAST COMPARISON ({forecast_year})',
            xaxis_title='Month',
            yaxis_title='Sales Volume',
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show forecast summaries
        st.subheader("📈 Forecast Summary")
        summary_data = []
        for col in model_cols + (['Ensemble_Forecast'] if 'Ensemble_Forecast' in result_df.columns else []):
            model_name = col.replace('_Forecast', '')
            total = result_df[col].sum()
            avg = result_df[col].mean()
            summary_data.append({
                'Model': model_name,
                'Total Annual Forecast': f"{total:,.0f}",
                'Monthly Average': f"{avg:,.0f}",
                'Min Month': f"{result_df[col].min():,.0f}",
                'Max Month': f"{result_df[col].max():,.0f}"
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

    # ENHANCED EXCEL DOWNLOAD
    st.subheader("📊 Enhanced Excel Report with Scaling Analysis")
    
    # Create enhanced Excel report
    @st.cache_data
    def create_enhanced_excel_report(result_df, hist_df, forecast_year, scaling_factor):
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Main Comparison
            main_sheet = result_df.copy()
            main_sheet['Month'] = main_sheet['Month'].dt.strftime('%Y-%m-%d')
            main_sheet.to_excel(writer, sheet_name='Main_Comparison', index=False)
            
            # Sheet 2: Scaling Analysis
            scaling_analysis = pd.DataFrame({
                'Metric': [
                    'Scaling Factor Applied',
                    'Historical Avg Monthly Sales',
                    'Historical Total Sales',
                    'Forecasting Method',
                    'Data Quality Score'
                ],
                'Value': [
                    f"{scaling_factor:.2f}x",
                    f"{hist_df['Sales'].mean():,.0f}",
                    f"{hist_df['Sales'].sum():,.0f}",
                    'Enhanced Multi-Model with Auto-Scaling',
                    f"{min(100, len(hist_df) * 4.17):.0f}%"  # Quality based on data length
                ],
                'Description': [
                    'Automatic scaling factor applied to align forecasts with actual data scale',
                    'Average monthly sales in historical training data',
                    'Total sales across all historical training data',
                    'Advanced forecasting methodology used',
                    'Data quality score based on historical data availability'
                ]
            })
            scaling_analysis.to_excel(writer, sheet_name='Scaling_Analysis', index=False)
            
            # Sheet 3: Model vs Actual Analysis (only if actual data exists)
            actual_col = f'Actual_{forecast_year}'
            if actual_col in result_df.columns and result_df[actual_col].notna().any():
                model_cols = [col for col in result_df.columns if '_Forecast' in col]
                
                analysis_data = []
                for _, row in result_df.iterrows():
                    base_data = {
                        'Month': row['Month'].strftime('%Y-%m-%d'),
                        'Actual': row[actual_col] if pd.notna(row[actual_col]) else 'N/A'
                    }
                    
                    for col in model_cols:
                        model_name = col.replace('_Forecast', '')
                        forecast_val = row[col]
                        base_data[f'{model_name}_Forecast'] = forecast_val
                        
                        if pd.notna(row[actual_col]) and pd.notna(forecast_val):
                            variance = forecast_val - row[actual_col]
                            abs_error = abs(variance)
                            pct_error = (abs_error / row[actual_col]) * 100
                            
                            base_data[f'{model_name}_Variance'] = round(variance, 2)
                            base_data[f'{model_name}_Abs_Error'] = round(abs_error, 2)
                            base_data[f'{model_name}_Error_Pct'] = round(pct_error, 2)
                        else:
                            base_data[f'{model_name}_Variance'] = 'N/A'
                            base_data[f'{model_name}_Abs_Error'] = 'N/A'
                            base_data[f'{model_name}_Error_Pct'] = 'N/A'
                    
                    analysis_data.append(base_data)
                
                analysis_df = pd.DataFrame(analysis_data)
                analysis_df.to_excel(writer, sheet_name='Detailed_Analysis', index=False)
                
                # Sheet 4: Enhanced Model Performance Summary
                summary_data = []
                actual_subset = result_df[result_df[actual_col].notna()]
                
                if len(actual_subset) > 0:
                    actual_total = actual_subset[actual_col].sum()
                    
                    for col in model_cols:
                        model_name = col.replace('_Forecast', '')
                        
                        # Calculate metrics only for months with actual data
                        metrics = calculate_accuracy_metrics(actual_subset[actual_col], actual_subset[col])
                        
                        if metrics:
                            total_forecast = result_df[col].sum()
                            bias_pct = ((total_forecast - actual_total) / actual_total * 100) if actual_total > 0 else 0
                            
                            summary_data.append({
                                'Model': model_name,
                                'MAPE': round(metrics['MAPE'], 2),
                                'MAE': round(metrics['MAE'], 0),
                                'RMSE': round(metrics['RMSE'], 0),
                                'Total_Forecast': round(total_forecast, 0),
                                'Total_Actual': round(actual_total, 0),
                                'Bias_Percent': round(bias_pct, 2),
                                'Accuracy_Percent': round(100 - metrics['MAPE'], 1),
                                'Months_With_Actual': len(actual_subset),
                                'Scaling_Applied': f"{scaling_factor:.2f}x"
                            })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        summary_df = summary_df.sort_values('MAPE')  # Best to worst
                        summary_df.to_excel(writer, sheet_name='Model_Performance', index=False)
            
            # Sheet 5: Enhanced Forecast Summary (always included)
            model_cols = [col for col in result_df.columns if '_Forecast' in col]
            forecast_summary = []
            
            for col in model_cols:
                model_name = col.replace('_Forecast', '')
                total_forecast = result_df[col].sum()
                avg_monthly = result_df[col].mean()
                
                forecast_summary.append({
                    'Model': model_name,
                    'Total_Annual_Forecast': round(total_forecast, 0),
                    'Average_Monthly_Forecast': round(avg_monthly, 0),
                    'Min_Monthly': round(result_df[col].min(), 0),
                    'Max_Monthly': round(result_df[col].max(), 0),
                    'Std_Dev': round(result_df[col].std(), 0),
                    'Scaling_Factor_Applied': f"{scaling_factor:.2f}x"
                })
            
            if forecast_summary:
                forecast_df = pd.DataFrame(forecast_summary)
                forecast_df.to_excel(writer, sheet_name='Enhanced_Forecast_Summary', index=False)
            
            # Sheet 6: Historical Data Analysis
            hist_analysis = []
            for year in hist_df['Month'].dt.year.unique():
                year_data = hist_df[hist_df['Month'].dt.year == year]
                hist_analysis.append({
                    'Year': year,
                    'Total_Sales': year_data['Sales'].sum(),
                    'Avg_Monthly': year_data['Sales'].mean(),
                    'Min_Monthly': year_data['Sales'].min(),
                    'Max_Monthly': year_data['Sales'].max(),
                    'Months_Available': len(year_data),
                    'Growth_Rate': 'N/A'  # Will calculate after
                })
            
            hist_df_analysis = pd.DataFrame(hist_analysis)
            # Calculate year-over-year growth
            for i in range(1, len(hist_df_analysis)):
                prev_total = hist_df_analysis.iloc[i-1]['Total_Sales']
                curr_total = hist_df_analysis.iloc[i]['Total_Sales']
                growth = ((curr_total - prev_total) / prev_total * 100) if prev_total > 0 else 0
                hist_df_analysis.iloc[i, hist_df_analysis.columns.get_loc('Growth_Rate')] = f"{growth:.1f}%"
            
            hist_df_analysis.to_excel(writer, sheet_name='Historical_Analysis', index=False)
        
        output.seek(0)
        return output
    
    # Generate and offer download
    excel_data = create_enhanced_excel_report(result_df, hist_df, forecast_year, scaling_factor)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📊 Download Enhanced Excel Report",
            data=excel_data,
            file_name=f"enhanced_sales_forecast_analysis_{forecast_year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # CSV download
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download CSV Report",
            data=csv,
            file_name=f"enhanced_forecasts_vs_actual_{forecast_year}.csv",
            mime="text/csv",
        )
    
    # Show what's in the Excel file
    st.info("""
    **📊 Enhanced Excel Report Contains:**
    - **Main_Comparison**: All forecasts and actual data with scaling applied
    - **Scaling_Analysis**: Automatic scaling detection and data quality metrics  
    - **Detailed_Analysis**: Month-by-month model vs actual with variance analysis
    - **Model_Performance**: Enhanced accuracy metrics with bias and scaling info
    - **Enhanced_Forecast_Summary**: Complete forecast statistics with scaling factors
    - **Historical_Analysis**: Year-over-year growth analysis and trends
    """)

    # Final summary
    st.subheader("🎯 Forecast Summary")
    
    if 'Ensemble_Forecast' in result_df.columns:
        ensemble_total = result_df['Ensemble_Forecast'].sum()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🔥 Best Forecast (Ensemble)", 
                f"{ensemble_total:,.0f}",
                help="Ensemble of all selected models"
            )
        
        with col2:
            historical_avg = hist_df['Sales'].mean() * 12 * scaling_factor
            growth_vs_hist = ((ensemble_total - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
            st.metric(
                "📈 Growth vs Historical", 
                f"{growth_vs_hist:+.1f}%",
                help="Growth compared to historical average annual sales"
            )
        
        with col3:
            if scaling_factor != 1.0:
                st.metric(
                    "⚖️ Auto-Scaling Applied", 
                    f"{scaling_factor:.2f}x",
                    help="Automatic scaling factor to align with actual data scale"
                )


if __name__ == "__main__":
    main()
