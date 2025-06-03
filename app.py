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
from sklearn.preprocessing import OneHotEncoder
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
            month_cols = []
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


def run_sarima_forecast(data, forecast_periods=12):
    """
    Run SARIMA forecast using statsmodels (replacement for pmdarima).
    """
    try:
        # Use a simple SARIMA configuration
        model = SARIMAX(data['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        fitted_model = model.fit(disp=False)
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=forecast_periods)
        forecast = np.maximum(forecast, 0)  # Ensure non-negative
        
        return forecast
    except Exception as e:
        st.warning(f"SARIMA failed: {str(e)}. Using simple trend method.")
        # Fallback to simple trend
        recent_values = data['Sales'].tail(12).values
        trend = np.mean(np.diff(recent_values)) if len(recent_values) > 1 else 0
        base_value = recent_values[-1] if len(recent_values) > 0 else data['Sales'].mean()
        return np.maximum([base_value + trend * i for i in range(1, forecast_periods + 1)], 0)


def run_prophet_forecast(data, forecast_periods=12):
    """
    Run Prophet forecast.
    """
    try:
        prophet_data = data[['Month', 'Sales']].rename(columns={'Month': 'ds', 'Sales': 'y'})
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.fit(prophet_data)
        
        # Create future dates
        future = model.make_future_dataframe(periods=forecast_periods, freq='M')
        forecast = model.predict(future)
        
        # Return only the forecast period
        forecast_values = forecast['yhat'].tail(forecast_periods).values
        return np.maximum(forecast_values, 0)
    except Exception as e:
        st.warning(f"Prophet failed: {str(e)}. Using mean method.")
        return [data['Sales'].mean()] * forecast_periods


def run_ets_forecast(data, forecast_periods=12):
    """
    Run Exponential Smoothing (ETS) forecast.
    """
    try:
        model = ExponentialSmoothing(
            data['Sales'],
            seasonal='add',
            seasonal_periods=12,
            trend='add'
        )
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=forecast_periods)
        return np.maximum(forecast, 0)
    except Exception as e:
        st.warning(f"ETS failed: {str(e)}. Using seasonal naive method.")
        # Fallback to seasonal naive
        if len(data) >= 12:
            seasonal_pattern = data['Sales'].tail(12).values
            return np.tile(seasonal_pattern, forecast_periods // 12 + 1)[:forecast_periods]
        else:
            return [data['Sales'].mean()] * forecast_periods


def run_xgb_forecast(data, forecast_periods=12):
    """
    Run XGBoost forecast using time series features.
    """
    try:
        # Create features
        df = data.copy()
        df['month'] = df['Month'].dt.month
        df['year'] = df['Month'].dt.year
        df['quarter'] = df['Month'].dt.quarter
        
        # Lag features
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = df['Sales'].shift(lag)
        
        # Rolling statistics
        df['rolling_mean_3'] = df['Sales'].rolling(window=3, min_periods=1).mean()
        df['rolling_mean_6'] = df['Sales'].rolling(window=6, min_periods=1).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 12:
            raise ValueError("Not enough data for XGBoost")
        
        # Prepare features and target
        feature_cols = ['month', 'quarter'] + [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
        X = df[feature_cols]
        y = df['Sales']
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Generate forecasts
        forecasts = []
        last_row = df.iloc[-1].copy()
        
        for i in range(forecast_periods):
            # Update time features
            future_date = df['Month'].iloc[-1] + pd.DateOffset(months=i+1)
            last_row['month'] = future_date.month
            last_row['quarter'] = future_date.quarter
            
            # Predict
            X_pred = last_row[feature_cols].values.reshape(1, -1)
            pred = model.predict(X_pred)[0]
            forecasts.append(max(pred, 0))
            
            # Update lag features for next iteration
            for lag in [1, 2, 3, 6, 12]:
                if f'lag_{lag}' in last_row:
                    if lag == 1:
                        last_row[f'lag_{lag}'] = pred
                    else:
                        last_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}']
        
        return forecasts
    except Exception as e:
        st.warning(f"XGBoost failed: {str(e)}. Using linear trend method.")
        # Fallback to linear trend
        if len(data) >= 3:
            X = np.arange(len(data)).reshape(-1, 1)
            y = data['Sales'].values
            lr = LinearRegression().fit(X, y)
            future_X = np.arange(len(data), len(data) + forecast_periods).reshape(-1, 1)
            return np.maximum(lr.predict(future_X), 0)
        else:
            return [data['Sales'].mean()] * forecast_periods


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("🔮 Spare Parts Sales Forecasting Dashboard")
    st.markdown("**Forecast monthly sales using multiple models and compare against actuals**")

    # Sidebar configuration
    st.sidebar.header("📋 Configuration")
    forecast_year = st.sidebar.selectbox(
        "Select forecast year:",
        options=[2024, 2025, 2026],
        index=0
    )

    # Model selection
    st.sidebar.subheader("🔧 Select Models")
    use_sarima = st.sidebar.checkbox("SARIMA", value=True)
    use_prophet = st.sidebar.checkbox("Prophet", value=True)
    use_ets = st.sidebar.checkbox("ETS (Holt-Winters)", value=True)
    use_xgb = st.sidebar.checkbox("XGBoost", value=True)

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
            help="For comparison with forecasts"
        )

    if historical_file is None:
        st.info("👆 Please upload historical sales data to begin forecasting.")
        return

    # Load and validate historical data
    hist_df = load_data(historical_file)
    if hist_df is None:
        return

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

    # Show historical data preview
    with st.expander("👀 Preview Historical Data"):
        st.dataframe(hist_df.head(12), use_container_width=True)

    # Generate forecasts
    st.subheader("🔮 Generating Forecasts...")

    progress_bar = st.progress(0)
    forecast_results = {}

    # Create forecast dates
    last_date = hist_df['Month'].max()
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=12,
        freq='M'
    )

    # Run each selected model
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
        with st.spinner(f"Running {model_name} model..."):
            try:
                forecast_values = model_func(hist_df, forecast_periods=12)
                forecast_results[f"{model_name}_Forecast"] = forecast_values
                st.success(f"✅ {model_name} completed successfully")
            except Exception as e:
                st.error(f"❌ {model_name} failed: {str(e)}")
                forecast_results[f"{model_name}_Forecast"] = [hist_df['Sales'].mean()] * 12

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

    # Load and merge actual 2024 data if provided
    actual_2024_df = None
    if actual_2024_file is not None:
        actual_2024_df = load_actual_2024_data(actual_2024_file, forecast_year)
        if actual_2024_df is not None:
            result_df = result_df.merge(actual_2024_df, on="Month", how="left")

    # Display results
    st.subheader("📊 Forecast Results")

    # Show forecast table
    display_df = result_df.copy()
    display_df['Month'] = display_df['Month'].dt.strftime('%Y-%m')
    
    # Format numbers for display
    for col in display_df.columns:
        if col != 'Month':
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True)

    # Show accuracy metrics if actual data is available
    model_cols = [col for col in result_df.columns if '_Forecast' in col]
    if f'Actual_{forecast_year}' in result_df.columns and not result_df[f'Actual_{forecast_year}'].isna().all():
        st.subheader("🎯 Model Accuracy Analysis")
        
        accuracy_data = []
        for col in model_cols:
            metrics = calculate_accuracy_metrics(result_df[f'Actual_{forecast_year}'], result_df[col])
            if metrics:
                accuracy_data.append({
                    'Model': col.replace('_Forecast', ''),
                    'MAPE (%)': f"{metrics['MAPE']:.1f}%",
                    'MAE': f"{metrics['MAE']:,.0f}",
                    'RMSE': f"{metrics['RMSE']:,.0f}",
                    'Accuracy': f"{100 - metrics['MAPE']:.1f}%"
                })

        if accuracy_data:
            accuracy_df = pd.DataFrame(accuracy_data)
            st.dataframe(accuracy_df, use_container_width=True)

    # 9) ACTUAL VS MODELS COMPARISON CHART
    st.subheader("📊 Actual vs Models Comparison")
    
    # Get model columns
    model_cols = [col for col in result_df.columns if '_Forecast' in col and col != 'Ensemble_Forecast']
    actual_col = f'Actual_{forecast_year}'
    
    # Create the comparison chart
    fig = go.Figure()
    
    # Add actual data line if available
    if actual_col in result_df.columns and not result_df[actual_col].isna().all():
        fig.add_trace(go.Scatter(
            x=result_df['Month'],
            y=result_df[actual_col],
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
            title=f'🔄 ACTUAL vs ALL MODELS COMPARISON ({forecast_year})',
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
        
        # Show accuracy summary table
        st.subheader("📋 Model Accuracy Summary")
        accuracy_data = []
        for col in model_cols:
            metrics = calculate_accuracy_metrics(result_df[actual_col], result_df[col])
            if metrics:
                accuracy_data.append({
                    'Model': col.replace('_Forecast', ''),
                    'MAPE (%)': round(metrics['MAPE'], 1),
                    'MAE': round(metrics['MAE'], 0),
                    'Total Forecast': f"{result_df[col].sum():,.0f}",
                    'Total Actual': f"{result_df[actual_col].sum():,.0f}",
                    'Accuracy': f"{100 - metrics['MAPE']:.1f}%"
                })
        
        if accuracy_data:
            accuracy_df = pd.DataFrame(accuracy_data)
            st.dataframe(accuracy_df, use_container_width=True)
    
    else:
        # Show forecast-only chart
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
            title=f'📈 ALL MODELS FORECAST COMPARISON ({forecast_year})',
            xaxis_title='Month',
            yaxis_title='Sales Volume',
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("📊 Upload 2024 actual data to see model accuracy comparison!")

    # 10) ENHANCED EXCEL DOWNLOAD
    st.subheader("📊 Enhanced Excel Report")
    
    # Create enhanced Excel report
    @st.cache_data
    def create_enhanced_excel_report(result_df, forecast_year):
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Main Comparison
            main_sheet = result_df.copy()
            main_sheet['Month'] = main_sheet['Month'].dt.strftime('%Y-%m-%d')
            main_sheet.to_excel(writer, sheet_name='Main_Comparison', index=False)
            
            # Sheet 2: Model vs Actual Analysis
            if f'Actual_{forecast_year}' in result_df.columns:
                model_cols = [col for col in result_df.columns if '_Forecast' in col]
                actual_col = f'Actual_{forecast_year}'
                
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
                
                # Sheet 3: Model Performance Summary
                summary_data = []
                for col in model_cols:
                    model_name = col.replace('_Forecast', '')
                    metrics = calculate_accuracy_metrics(result_df[actual_col], result_df[col])
                    
                    if metrics:
                        total_forecast = result_df[col].sum()
                        total_actual = result_df[actual_col].sum()
                        bias_pct = ((total_forecast - total_actual) / total_actual * 100) if total_actual > 0 else 0
                        
                        summary_data.append({
                            'Model': model_name,
                            'MAPE': round(metrics['MAPE'], 2),
                            'MAE': round(metrics['MAE'], 0),
                            'RMSE': round(metrics['RMSE'], 0),
                            'Total_Forecast': round(total_forecast, 0),
                            'Total_Actual': round(total_actual, 0),
                            'Bias_Percent': round(bias_pct, 2),
                            'Accuracy_Percent': round(100 - metrics['MAPE'], 1)
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df = summary_df.sort_values('MAPE')  # Best to worst
                    summary_df.to_excel(writer, sheet_name='Model_Performance', index=False)
        
        output.seek(0)
        return output
    
    # Generate and offer download
    excel_data = create_enhanced_excel_report(result_df, forecast_year)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📊 Download Enhanced Excel Report",
            data=excel_data,
            file_name=f"sales_forecast_analysis_{forecast_year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # CSV download
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download CSV Report",
            data=csv,
            file_name=f"forecasts_vs_actual_{forecast_year}.csv",
            mime="text/csv",
        )
    
    # Show what's in the Excel file
    st.info("""
    **📊 Excel Report Contains:**
    - **Main_Comparison**: All forecasts and actual data
    - **Detailed_Analysis**: Each model vs actual with variance, errors, and percentages
    - **Model_Performance**: Summary with MAPE, MAE, RMSE, bias, and accuracy rankings
    """)


if __name__ == "__main__":
    main()
