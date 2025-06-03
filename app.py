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

# ----------------------------
# Helper functions
# ----------------------------

@st.cache_data
def load_data(uploaded_file):
    """
    Load any Excel file with at least ['Month','Sales'] columns,
    parse dates, aggregate to total monthly sales, and return a sorted DataFrame.
    """
    df = pd.read_excel(uploaded_file)
    # Basic validation
    if "Month" not in df.columns or "Sales" not in df.columns:
        st.error("Uploaded file must contain 'Month' and 'Sales' columns.")
        return None

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    if df["Month"].isna().any():
        st.error("Some 'Month' values could not be parsed as dates.")
        return None

    # Convert Sales to numeric
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    df["Sales"] = df["Sales"].abs()  # ensure non-negative

    # Aggregate to total monthly sales (in case there are many part‐level rows)
    monthly = (
        df.groupby("Month", as_index=False)["Sales"]
        .sum()
        .sort_values("Month")
        .reset_index(drop=True)
    )
    return monthly


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
    Fit a SARIMA model using statsmodels.SARIMAX with proven default parameters.
    Returns fitted model or None if training fails.
    """
    try:
        # Validate input
        if len(train_series) < seasonal_period * 2:
            raise ValueError(f"Need at least {seasonal_period * 2} data points for seasonal modeling")
        
        # Check for constant series
        if train_series.var() == 0:
            raise ValueError("Sales data is constant - cannot fit SARIMA model")
        
        # Use proven SARIMA parameters
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


def forecast_sarima(fitted_model, n_periods, last_date):
    """
    Forecast n_periods months ahead from last_date using a fitted SARIMA.
    Returns DataFrame with 'Month','SARIMA_Forecast','SARIMA_Lower','SARIMA_Upper'.
    """
    if fitted_model is None:
        return None

    try:
        # get predictions + conf_int
        pred = fitted_model.get_forecast(steps=n_periods)
        forecast_vals = pred.predicted_mean.clip(lower=0).values
        ci = pred.conf_int().values
        lower = np.maximum(ci[:, 0], 0)
        upper = np.maximum(ci[:, 1], 0)

        future_months = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1), periods=n_periods, freq="MS"
        )
        return pd.DataFrame(
            {
                "Month": future_months,
                "SARIMA_Forecast": forecast_vals,
                "SARIMA_Lower": lower,
                "SARIMA_Upper": upper,
            }
        )
    except Exception as e:
        st.error(f"SARIMA forecasting failed: {str(e)}")
        return None


def train_prophet(train_df):
    """
    Fit a Prophet model on a DataFrame with columns ['ds','y'].
    Returns the fitted Prophet instance.
    """
    if len(train_df) < 24:
        st.warning("Less than 24 months of data: Prophet may underperform.")
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    with st.spinner("Training Prophet..."):
        m.fit(train_df)
    return m


def forecast_prophet(m, periods, last_date):
    """
    Forecast with Prophet for `periods` months beyond last_date.
    Returns DataFrame with ['Month','Prophet_Forecast','Prophet_Lower','Prophet_Upper'].
    """
    if m is None:
        return None
    future = pd.DataFrame(
        {"ds": pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=periods, freq="MS")}
    )
    fc = m.predict(future)
    df = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    df["yhat"] = df["yhat"].clip(lower=0)
    df["yhat_lower"] = df["yhat_lower"].clip(lower=0)
    df["yhat_upper"] = df["yhat_upper"].clip(lower=0)
    return df.rename(
        columns={
            "ds": "Month",
            "yhat": "Prophet_Forecast",
            "yhat_lower": "Prophet_Lower",
            "yhat_upper": "Prophet_Upper",
        }
    )


def train_ets(train_series, seasonal_period=12):
    """
    Fit a Holt-Winters ETS model (multiplicative seasonality by default). Returns fitted model.
    """
    try:
        # If variance is zero (constant series), skip
        if train_series.var() == 0:
            st.warning("Series is constant: ETS will simply hold level.")
        # Multiplicative seasonality is common for sales
        model = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal="mul",
            seasonal_periods=seasonal_period,
        )
        fitted = model.fit(optimized=True)
        return fitted
    except Exception as e:
        st.error(f"ETS training failed: {str(e)}")
        return None


def forecast_ets(fitted_model, n_periods, last_date):
    """
    Forecast n_periods months ahead from last_date using a fitted ETS model.
    Returns DataFrame with ['Month','ETS_Forecast'].
    """
    if fitted_model is None:
        return None
    try:
        forecast_vals = fitted_model.forecast(steps=n_periods).clip(lower=0).values
        future_months = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1), periods=n_periods, freq="MS"
        )
        return pd.DataFrame({"Month": future_months, "ETS_Forecast": forecast_vals})
    except Exception as e:
        st.error(f"ETS forecasting failed: {str(e)}")
        return None


@st.cache_data
def create_ml_features(df):
    """
    Given a DataFrame with ['Month','Sales'], create lag/rolling/month‐of‐year features
    for XGBoost. Returns features DataFrame (indexed by Month) and target Series.
    """
    df = df.copy().set_index("Month").asfreq("MS")
    df["lag_1"] = df["Sales"].shift(1)
    df["lag_12"] = df["Sales"].shift(12)
    df["rolling_mean_3"] = df["Sales"].shift(1).rolling(window=3).mean()
    # cyclical month encoding
    df["month"] = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Drop rows with NaNs due to shifts
    features = df.dropna().copy()
    X = features[["lag_1", "lag_12", "rolling_mean_3", "month_sin", "month_cos"]]
    y = features["Sales"]
    return X, y


def train_xgboost(train_df):
    """
    Train an XGBoost (GradientBoostingRegressor) on lagged features. Returns the trained model
    and the last 12 months of features (to be used for recursive forecasting).
    """
    try:
        # Prepare features and target
        X, y = create_ml_features(train_df[["Month", "Sales"]])
        if len(X) < 12:
            st.warning("Too few rows after feature engineering: XGBoost may not train properly.")
        # Use TimeSeriesSplit to tune lightly
        splitter = TimeSeriesSplit(n_splits=3)
        best_model = None
        best_score = np.inf

        for train_idx, val_idx in splitter.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            score = mean_absolute_error(y_val, preds)
            if score < best_score:
                best_score = score
                best_model = model

        # Finally, retrain on entire set
        best_model.fit(X, y)
        # Save the last 12 months of df (for recursive forecast)
        last_12 = train_df.set_index("Month").asfreq("MS")["Sales"].iloc[-12:]
        return best_model, last_12
    except Exception as e:
        st.error(f"XGBoost training failed: {str(e)}")
        return None, None


def forecast_xgboost(model, last_12_series, n_periods):
    """
    Recursively forecast n_periods months ahead using the trained XGBoost model.
    last_12_series: Pandas Series of the most recent 12 months of 'Sales' indexed by Month.
    Returns DataFrame with ['Month','XGB_Forecast'].
    """
    if model is None or last_12_series is None:
        return None

    try:
        # We'll iterate one month at a time:
        history = last_12_series.copy().to_dict()  # {Timestamp -> value}
        future_months = pd.date_range(
            start=last_12_series.index.max() + pd.offsets.MonthBegin(1), periods=n_periods, freq="MS"
        )
        forecasts = []
        for dt in future_months:
            # Build the feature vector for this dt
            lag_1 = history[dt - pd.offsets.MonthBegin(1)]
            lag_12 = history.get(dt - pd.DateOffset(months=12), np.nan)
            # Compute rolling_mean_3 over the last 3 months
            last1 = history[dt - pd.offsets.MonthBegin(1)]
            last2 = history.get(dt - pd.offsets.MonthBegin(2), np.nan)
            last3 = history.get(dt - pd.offsets.MonthBegin(3), np.nan)
            rm3 = np.nanmean([last1, last2, last3])

            month = dt.month
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            x_vec = np.array([[lag_1, lag_12, rm3, month_sin, month_cos]])
            pred = max(model.predict(x_vec)[0], 0)
            forecasts.append(pred)

            # Add to history to enable the next-step lag calculations
            history[dt] = pred

        return pd.DataFrame({"Month": future_months, "XGB_Forecast": forecasts})
    except Exception as e:
        st.error(f"XGBoost forecasting failed: {str(e)}")
        return None


def create_model_comparison_diagram(result_df, forecast_year):
    """
    Create a comprehensive comparison diagram showing actual vs all models.
    """
    if f'Actual_{forecast_year}' not in result_df.columns or result_df[f'Actual_{forecast_year}'].isna().all():
        return None
    
    # Prepare data
    comparison_data = result_df.copy()
    comparison_data['Month_Name'] = pd.to_datetime(comparison_data['Month']).dt.strftime('%b %Y')
    
    # Get model columns
    model_cols = [col for col in result_df.columns if '_Forecast' in col]
    actual_col = f'Actual_{forecast_year}'
    
    # Create subplots: one main comparison and individual model comparisons
    n_models = len(model_cols)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '📊 All Models vs Actual Comparison',
            '🎯 Actual vs Ensemble Forecast',
            '📈 Model Performance by Month',
            '🔍 Accuracy Distribution'
        ],
        specs=[
            [{"colspan": 2}, None],
            [{"type": "scatter"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Colors for different models
    colors = {
        'Actual': '#1f77b4',  # Blue
        'SARIMA': '#ff7f0e',  # Orange  
        'Prophet': '#2ca02c', # Green
        'ETS': '#d62728',     # Red
        'XGB': '#9467bd',     # Purple
        'Ensemble': '#8c564b' # Brown
    }
    
    # Plot 1: All models comparison (top row, full width)
    fig.add_trace(go.Scatter(
        x=comparison_data['Month_Name'],
        y=comparison_data[actual_col],
        mode='lines+markers',
        name='Actual',
        line=dict(color=colors['Actual'], width=4),
        marker=dict(size=10, symbol='circle'),
        legendgroup='main'
    ), row=1, col=1)
    
    for col in model_cols:
        model_name = col.replace('_Forecast', '')
        fig.add_trace(go.Scatter(
            x=comparison_data['Month_Name'],
            y=comparison_data[col],
            mode='lines+markers',
            name=model_name,
            line=dict(color=colors.get(model_name, '#17becf'), width=2),
            marker=dict(size=6),
            legendgroup='main'
        ), row=1, col=1)
    
    # Plot 2: Actual vs Ensemble (bottom left)
    if 'Ensemble_Forecast' in comparison_data.columns:
        # Scatter plot for correlation
        fig.add_trace(go.Scatter(
            x=comparison_data[actual_col],
            y=comparison_data['Ensemble_Forecast'],
            mode='markers',
            name='Actual vs Ensemble',
            marker=dict(
                size=12,
                color=comparison_data.index,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Month")
            ),
            text=comparison_data['Month_Name'],
            textposition="top center",
            showlegend=False
        ), row=2, col=1)
        
        # Add perfect prediction line
        min_val = min(comparison_data[actual_col].min(), comparison_data['Ensemble_Forecast'].min())
        max_val = max(comparison_data[actual_col].max(), comparison_data['Ensemble_Forecast'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False
        ), row=2, col=1)
    
    # Plot 3: Model accuracy by month (bottom right)
    accuracy_data = []
    for _, row in comparison_data.iterrows():
        for col in model_cols:
            if pd.notna(row[actual_col]) and pd.notna(row[col]):
                error = abs(row[actual_col] - row[col]) / row[actual_col] * 100
                accuracy_data.append({
                    'Month': row['Month_Name'],
                    'Model': col.replace('_Forecast', ''),
                    'MAPE': error
                })
    
    if accuracy_data:
        acc_df = pd.DataFrame(accuracy_data)
        for model in acc_df['Model'].unique():
            model_data = acc_df[acc_df['Model'] == model]
            fig.add_trace(go.Bar(
                x=model_data['Month'],
                y=model_data['MAPE'],
                name=f'{model} MAPE',
                marker_color=colors.get(model, '#17becf'),
                showlegend=False,
                opacity=0.7
            ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="🔄 Comprehensive Model Comparison Dashboard",
        title_x=0.5,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_yaxes(title_text="Sales Volume", row=1, col=1)
    fig.update_xaxes(title_text="Actual Sales", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Sales", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=2)
    fig.update_yaxes(title_text="MAPE (%)", row=2, col=2)
    
    return fig


def create_detailed_excel_report(result_df, hist_df, actual_2024_df, forecast_year):
    """
    Create a detailed Excel report with comprehensive model comparisons.
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Main Results with all models
        result_df.to_excel(writer, sheet_name='All_Models_Comparison', index=False)
        
        # Sheet 2: Model vs Actual Detailed Comparison
        if f'Actual_{forecast_year}' in result_df.columns and not result_df[f'Actual_{forecast_year}'].isna().all():
            model_cols = [col for col in result_df.columns if '_Forecast' in col]
            actual_col = f'Actual_{forecast_year}'
            
            # Create detailed comparison table
            comparison_df = result_df[['Month', actual_col] + model_cols].copy()
            
            # Add variance columns for each model
            for col in model_cols:
                model_name = col.replace('_Forecast', '')
                comparison_df[f'{model_name}_Variance'] = comparison_df[col] - comparison_df[actual_col]
                comparison_df[f'{model_name}_Abs_Error'] = abs(comparison_df[col] - comparison_df[actual_col])
                comparison_df[f'{model_name}_MAPE'] = (abs(comparison_df[col] - comparison_df[actual_col]) / comparison_df[actual_col] * 100).round(2)
            
            comparison_df.to_excel(writer, sheet_name='Model_vs_Actual_Detail', index=False)
        
        # Sheet 3: Monthly Performance Summary
        if f'Actual_{forecast_year}' in result_df.columns and not result_df[f'Actual_{forecast_year}'].isna().all():
            monthly_summary = []
            for _, row in result_df.iterrows():
                month_data = {
                    'Month': row['Month'],
                    'Actual': row[actual_col] if pd.notna(row[actual_col]) else 'N/A'
                }
                
                for col in model_cols:
                    model_name = col.replace('_Forecast', '')
                    forecast_val = row[col]
                    month_data[f'{model_name}_Forecast'] = forecast_val
                    
                    if pd.notna(row[actual_col]) and pd.notna(forecast_val):
                        error = abs(row[actual_col] - forecast_val)
                        pct_error = (error / row[actual_col]) * 100
                        month_data[f'{model_name}_Error'] = round(error, 2)
                        month_data[f'{model_name}_Error_Pct'] = round(pct_error, 2)
                    else:
                        month_data[f'{model_name}_Error'] = 'N/A'
                        month_data[f'{model_name}_Error_Pct'] = 'N/A'
                
                monthly_summary.append(month_data)
            
            monthly_df = pd.DataFrame(monthly_summary)
            monthly_df.to_excel(writer, sheet_name='Monthly_Performance', index=False)
        
        # Sheet 4: Model Ranking and Statistics
        if f'Actual_{forecast_year}' in result_df.columns and not result_df[f'Actual_{forecast_year}'].isna().all():
            ranking_data = []
            for col in model_cols:
                metrics = calculate_accuracy_metrics(result_df[actual_col], result_df[col])
                if metrics:
                    total_forecast = result_df[col].sum()
                    total_actual = result_df[actual_col].sum()
                    bias = ((total_forecast - total_actual) / total_actual * 100) if total_actual > 0 else 0
                    
                    ranking_data.append({
                        'Model': col.replace('_Forecast', ''),
                        'MAPE': round(metrics['MAPE'], 2),
                        'MAE': round(metrics['MAE'], 0),
                        'RMSE': round(metrics['RMSE'], 0),
                        'Total_Forecast': round(total_forecast, 0),
                        'Total_Actual': round(total_actual, 0),
                        'Bias_Pct': round(bias, 2),
                        'Avg_Monthly_Forecast': round(result_df[col].mean(), 0),
                        'Std_Monthly_Forecast': round(result_df[col].std(), 0)
                    })
            
            if ranking_data:
                ranking_df = pd.DataFrame(ranking_data)
                # Sort by MAPE (best to worst)
                ranking_df = ranking_df.sort_values('MAPE')
                ranking_df['Rank'] = range(1, len(ranking_df) + 1)
                ranking_df = ranking_df[['Rank'] + [col for col in ranking_df.columns if col != 'Rank']]
                ranking_df.to_excel(writer, sheet_name='Model_Rankings', index=False)
        
        # Sheet 5: Historical Analysis
        if hist_df is not None:
            hist_analysis = hist_df.copy()
            hist_analysis['Year'] = hist_analysis['Month'].dt.year
            hist_analysis['Month_Name'] = hist_analysis['Month'].dt.strftime('%B')
            hist_analysis['Quarter'] = hist_analysis['Month'].dt.quarter
            
            # Yearly summary
            yearly_summary = hist_analysis.groupby('Year')['Sales'].agg([
                'sum', 'mean', 'std', 'min', 'max', 'count'
            ]).round(2).reset_index()
            yearly_summary.columns = ['Year', 'Total_Sales', 'Avg_Monthly', 'Std_Dev', 'Min_Monthly', 'Max_Monthly', 'Months_Count']
            
            # Calculate year-over-year growth
            yearly_summary['YoY_Growth_Pct'] = yearly_summary['Total_Sales'].pct_change() * 100
            yearly_summary['YoY_Growth_Pct'] = yearly_summary['YoY_Growth_Pct'].round(2)
            
            yearly_summary.to_excel(writer, sheet_name='Historical_Analysis', index=False)
            
            # Monthly historical data
            hist_analysis.to_excel(writer, sheet_name='Historical_Data_Detail', index=False)
        
        # Sheet 6: Executive Summary
        summary_data = []
        
        # Overall forecast summary
        forecast_summary = {
            'Metric': 'Total Forecast (Ensemble)',
            'Value': result_df['Ensemble_Forecast'].sum() if 'Ensemble_Forecast' in result_df.columns else 'N/A',
            'Description': 'Sum of all 12 months ensemble forecast'
        }
        summary_data.append(forecast_summary)
        
        if f'Actual_{forecast_year}' in result_df.columns and not result_df[f'Actual_{forecast_year}'].isna().all():
            actual_total = result_df[actual_col].sum()
            ensemble_total = result_df['Ensemble_Forecast'].sum() if 'Ensemble_Forecast' in result_df.columns else 0
            
            summary_data.extend([
                {
                    'Metric': f'Total Actual {forecast_year}',
                    'Value': actual_total,
                    'Description': f'Sum of actual sales for {forecast_year}'
                },
                {
                    'Metric': 'Forecast Accuracy (Ensemble)',
                    'Value': f"{((1 - abs(ensemble_total - actual_total) / actual_total) * 100):.1f}%" if actual_total > 0 else 'N/A',
                    'Description': 'Overall forecast accuracy percentage'
                },
                {
                    'Metric': 'Best Performing Model',
                    'Value': ranking_df.iloc[0]['Model'] if 'ranking_df' in locals() and not ranking_df.empty else 'N/A',
                    'Description': 'Model with lowest MAPE'
                }
            ])
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
    
    output.seek(0)
    return output


def create_dashboard_charts(result_df, hist_df, forecast_year):
    """
    Create interactive dashboard charts using Plotly.
    """
    # Main forecast vs actual chart
    fig_main = go.Figure()
    
    # Add actual data if available
    if f'Actual_{forecast_year}' in result_df.columns and not result_df[f'Actual_{forecast_year}'].isna().all():
        fig_main.add_trace(go.Scatter(
            x=result_df['Month'],
            y=result_df[f'Actual_{forecast_year}'],
            mode='lines+markers',
            name=f'Actual {forecast_year}',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8)
        ))
    
    # Add forecast lines
    forecast_cols = [col for col in result_df.columns if '_Forecast' in col]
    colors = ['#A23B72', '#F18F01', '#C73E1D', '#7209B7']
    
    for i, col in enumerate(forecast_cols):
        fig_main.add_trace(go.Scatter(
            x=result_df['Month'],
            y=result_df[col],
            mode='lines+markers',
            name=col.replace('_Forecast', ''),
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6)
        ))
    
    fig_main.update_layout(
        title=f'📈 Sales Forecast vs Actual ({forecast_year})',
        xaxis_title='Month',
        yaxis_title='Sales Volume',
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    return fig_main


def create_performance_chart(result_df, forecast_year):
    """
    Create model performance comparison chart.
    """
    if f'Actual_{forecast_year}' not in result_df.columns or result_df[f'Actual_{forecast_year}'].isna().all():
        return None
    
    performance_data = []
    model_cols = [col for col in result_df.columns if '_Forecast' in col]
    
    for col in model_cols:
        metrics = calculate_accuracy_metrics(result_df[f'Actual_{forecast_year}'], result_df[col])
        if metrics:
            performance_data.append({
                'Model': col.replace('_Forecast', ''),
                'MAPE': metrics['MAPE'],
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE']
            })
    
    if not performance_data:
        return None
    
    perf_df = pd.DataFrame(performance_data)
    
    # Create subplots for different metrics
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('MAPE (%)', 'MAE', 'RMSE'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # MAPE
    fig.add_trace(go.Bar(
        x=perf_df['Model'],
        y=perf_df['MAPE'],
        name='MAPE',
        marker_color=colors[0],
        showlegend=False
    ), row=1, col=1)
    
    # MAE
    fig.add_trace(go.Bar(
        x=perf_df['Model'],
        y=perf_df['MAE'],
        name='MAE',
        marker_color=colors[1],
        showlegend=False
    ), row=1, col=2)
    
    # RMSE
    fig.add_trace(go.Bar(
        x=perf_df['Model'],
        y=perf_df['RMSE'],
        name='RMSE',
        marker_color=colors[2],
        showlegend=False
    ), row=1, col=3)
    
    fig.update_layout(
        title_text="📊 Model Performance Comparison",
        height=400,
        showlegend=False
    )
    
    return fig


def create_historical_trend_chart(hist_df):
    """
    Create historical sales trend chart.
    """
    if hist_df is None or len(hist_df) == 0:
        return None
    
    # Monthly trend
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_df['Month'],
        y=hist_df['Sales'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=5)
    ))
    
    # Add trend line
    from sklearn.linear_model import LinearRegression
    X = np.arange(len(hist_df)).reshape(-1, 1)
    y = hist_df['Sales'].values
    lr = LinearRegression().fit(X, y)
    trend = lr.predict(X)
    
    fig.add_trace(go.Scatter(
        x=hist_df['Month'],
        y=trend,
        mode='lines',
        name='Trend Line',
        line=dict(color='#A23B72', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='📈 Historical Sales Trend',
        xaxis_title='Month',
        yaxis_title='Sales Volume',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def calculate_accuracy_metrics(actual, forecast):
    """
    Given two aligned pandas Series (actual, forecast), compute MAPE, MAE, RMSE.
    """
    mask = (~actual.isna()) & (~forecast.isna()) & (actual != 0)
    a = actual[mask]
    f = forecast[mask]
    if len(a) == 0:
        return None

    mape = np.mean(np.abs((a - f) / a)) * 100
    mae = mean_absolute_error(a, f)
    rmse = np.sqrt(mean_squared_error(a, f))
    return {"MAPE": mape, "MAE": mae, "RMSE": rmse}


# ----------------------------
# Streamlit App Layout
# ----------------------------

st.set_page_config(page_title="Spare-Parts Sales Forecast Comparison", layout="wide")
st.title("📊 Spare-Parts Sales: 2024 Forecast vs. Actual")

st.markdown(
    """
Upload one file containing historical data (at least through Dec 2023), and optionally a second file
with the actual 2024 data. Then select which models to run:\n
- **SARIMA** (Seasonal AutoRegressive Integrated Moving Average)  
- **Prophet** (Facebook/Meta's forecasting tool)  
- **ETS** (Exponential Smoothing / Holt–Winters)  
- **XGBoost** (Gradient Boosting with lag features)  

The app will train each selected model on your historical data, forecast for 2024, 
then merge all forecasts alongside the 2024 actuals (if provided) for comparison.
"""
)

# Sidebar: model selection
st.sidebar.header("🎯 Model Selection")
run_sarima = st.sidebar.checkbox("SARIMA", value=True, help="Seasonal ARIMA model")
run_prophet = st.sidebar.checkbox("Prophet", value=True, help="Facebook's forecasting tool")
run_ets = st.sidebar.checkbox("ETS (Holt-Winters)", value=False, help="Exponential smoothing")
run_xgb = st.sidebar.checkbox("XGBoost", value=False, help="Tree-based ML model")

# Sidebar: seasonal period (usually 12 for monthly data)
seasonal_period = st.sidebar.selectbox(
    "Seasonal Period (months)", [12, 6, 4], index=0, help="12 = yearly seasonality"
)
forecast_year = st.sidebar.selectbox("Forecast Year", [2024, 2025], index=0)

# File uploaders
st.subheader("1. Upload Historical Data")
historical_file = st.file_uploader(
    "Excel file with columns ['Month','Sales'] (multiple rows per month will be aggregated)",
    type=["xlsx", "xls"],
    key="hist",
)

st.markdown("---")
st.subheader("2. (Optional) Upload 2024 Actuals for Comparison")
actual_2024_file = st.file_uploader(
    f"Excel file with 2024 actuals (same format). Will merge on 'Month' for {forecast_year}.",
    type=["xlsx", "xls"],
    key="act2024",
)

if historical_file:
    hist_df = load_data(historical_file)
    if hist_df is None or hist_df.empty:
        st.stop()

    st.success(
        f"Historical data loaded: {hist_df['Month'].min().strftime('%Y-%m')} to "
        f"{hist_df['Month'].max().strftime('%Y-%m')} ({len(hist_df)} months)."
    )

    # Load 2024 actuals if provided
    actual_2024_df = None
    if actual_2024_file:
        actual_2024_df = load_actual_2024_data(actual_2024_file, forecast_year)
        if actual_2024_df is not None:
            st.success(
                f"2024 actuals loaded: {actual_2024_df['Month'].min().strftime('%Y-%m')} to "
                f"{actual_2024_df['Month'].max().strftime('%Y-%m')} ({len(actual_2024_df)} months)."
            )

    # Split into train (<= Dec of year_before forecast_year) and actual_2024 (≥ Jan forecast_year)
    cutoff = pd.Timestamp(f"{forecast_year-1}-12-01")
    if hist_df["Month"].max() < cutoff:
        st.error(f"Need data through at least {cutoff.strftime('%Y-%m')}. Please upload more data.")
        st.stop()

    train_df = hist_df[hist_df["Month"] <= cutoff].copy()
    future_actuals = hist_df[
        (hist_df["Month"] >= pd.Timestamp(f"{forecast_year}-01-01"))
        & (hist_df["Month"] < pd.Timestamp(f"{forecast_year+1}-01-01"))
    ].copy()
    if not future_actuals.empty:
        future_actuals = future_actuals.rename(columns={"Sales": f"Actual_{forecast_year}"})

    # Show a preview of train data
    with st.expander("📋 Historical Data Preview"):
        st.dataframe(train_df.head(10))
        st.write(f"Training range: {train_df['Month'].min().strftime('%Y-%m')} to {train_df['Month'].max().strftime('%Y-%m')}")
        st.write(f"Total training months: {len(train_df)}")

    # Check if any models are selected
    if not any([run_sarima, run_prophet, run_ets, run_xgb]):
        st.warning("⚠️ Please select at least one model from the sidebar to run forecasts.")
        st.stop()

    # Prepare containers for forecasts
    sarima_forecast_df = None
    prophet_forecast_df = None
    ets_forecast_df = None
    xgb_forecast_df = None

    last_train_date = train_df["Month"].max()
    n_periods = 12  # always forecasting 12 months ahead

    st.subheader("🚀 Running Forecasts")
    
    # Progress tracking
    models_to_run = sum([run_sarima, run_prophet, run_ets, run_xgb])
    progress_bar = st.progress(0)
    model_count = 0

    # 1) SARIMA
    if run_sarima:
        with st.spinner("Training SARIMA model..."):
            sarima_model = train_sarima(train_df.set_index("Month")["Sales"], seasonal_period)
            if sarima_model:
                sarima_forecast_df = forecast_sarima(sarima_model, n_periods, last_train_date)
                st.success("✅ SARIMA forecast ready.")
            else:
                st.error("❌ SARIMA failed.")
        model_count += 1
        progress_bar.progress(model_count / models_to_run)

    # 2) Prophet
    if run_prophet:
        with st.spinner("Training Prophet model..."):
            prophet_train = train_df.rename(columns={"Month": "ds", "Sales": "y"})[["ds", "y"]]
            prophet_model = train_prophet(prophet_train)
            if prophet_model:
                prophet_forecast_df = forecast_prophet(prophet_model, n_periods, last_train_date)
                st.success("✅ Prophet forecast ready.")
            else:
                st.error("❌ Prophet failed.")
        model_count += 1
        progress_bar.progress(model_count / models_to_run)

    # 3) ETS (Holt-Winters)
    if run_ets:
        with st.spinner("Training ETS (Holt-Winters) model..."):
            ets_model = train_ets(train_df.set_index("Month")["Sales"], seasonal_period)
            if ets_model:
                ets_forecast_df = forecast_ets(ets_model, n_periods, last_train_date)
                st.success("✅ ETS forecast ready.")
            else:
                st.error("❌ ETS failed.")
        model_count += 1
        progress_bar.progress(model_count / models_to_run)

    # 4) XGBoost (tree-based lag features)
    if run_xgb:
        with st.spinner("Training XGBoost model..."):
            xgb_model, last_12 = train_xgboost(train_df[["Month", "Sales"]])
            if xgb_model is not None:
                xgb_forecast_df = forecast_xgboost(xgb_model, last_12, n_periods)
                st.success("✅ XGBoost forecast ready.")
            else:
                st.error("❌ XGBoost failed.")
        model_count += 1
        progress_bar.progress(model_count / models_to_run)

    # 5) Merge all forecasts into one DataFrame
    # Create a complete 12‐month index for forecast_year
    future_months = pd.date_range(
        start=pd.Timestamp(f"{forecast_year}-01-01"), periods=12, freq="MS"
    )
    result_df = pd.DataFrame({"Month": future_months})

    # Merge each forecast if available
    if sarima_forecast_df is not None:
        result_df = result_df.merge(
            sarima_forecast_df[["Month", "SARIMA_Forecast"]], on="Month", how="left"
        )
    if prophet_forecast_df is not None:
        result_df = result_df.merge(
            prophet_forecast_df[["Month", "Prophet_Forecast"]], on="Month", how="left"
        )
    if ets_forecast_df is not None:
        result_df = result_df.merge(
            ets_forecast_df[["Month", "ETS_Forecast"]], on="Month", how="left"
        )
    if xgb_forecast_df is not None:
        result_df = result_df.merge(
            xgb_forecast_df[["Month", "XGB_Forecast"]], on="Month", how="left"
        )

    # If user uploaded "actual_2024_df", merge it:
    if actual_2024_df is not None:
        result_df = result_df.merge(actual_2024_df[["Month", f"Actual_{forecast_year}"]], on="Month", how="left")
    else:
        # If they did not upload a separate actuals file, but historical data already contains some 2024 months:
        if not future_actuals.empty:
            result_df = result_df.merge(
                future_actuals[["Month", f"Actual_{forecast_year}"]], on="Month", how="left"
            )
        else:
            result_df[f"Actual_{forecast_year}"] = np.nan

    # 6) Simple Ensemble = average of all selected model forecasts (ignoring NaNs)
    model_cols = []
    if run_sarima and sarima_forecast_df is not None:
        model_cols.append("SARIMA_Forecast")
    if run_prophet and prophet_forecast_df is not None:
        model_cols.append("Prophet_Forecast")
    if run_ets and ets_forecast_df is not None:
        model_cols.append("ETS_Forecast")
    if run_xgb and xgb_forecast_df is not None:
        model_cols.append("XGB_Forecast")

    if model_cols:
        result_df["Ensemble_Forecast"] = result_df[model_cols].mean(axis=1)

    # Round all numeric columns to two decimals
    for c in result_df.select_dtypes(include="number").columns:
        result_df[c] = result_df[c].round(2)

    # 7) Display the merged table
    st.subheader(f"📊 Forecast vs. Actual ({forecast_year})")
    
    # Show data availability summary
    if f"Actual_{forecast_year}" in result_df.columns and not result_df[f"Actual_{forecast_year}"].isna().all():
        available_months = result_df[f"Actual_{forecast_year}"].notna().sum()
        total_actual = result_df[f"Actual_{forecast_year}"].sum()
        st.info(f"📈 Found actual data for {available_months} out of 12 months. Total actual sales: {total_actual:,.0f}")
    else:
        st.info("📊 No actual data provided - showing forecasts only")
        if model_cols:
            total_ensemble = result_df["Ensemble_Forecast"].sum()
            st.write(f"🎯 **Total ensemble forecast for {forecast_year}: {total_ensemble:,.0f}**")
    
    st.dataframe(result_df.set_index("Month"), use_container_width=True)

    # 8) Calculate and display accuracy metrics if actuals exist
    if f"Actual_{forecast_year}" in result_df.columns and not result_df[f"Actual_{forecast_year}"].isna().all():
        st.subheader("📐 Accuracy Metrics")
        st.write("*Calculated only for months with actual data*")

        metrics = {}
        for col in model_cols + (["Ensemble_Forecast"] if model_cols else []):
            m = calculate_accuracy_metrics(
                result_df[f"Actual_{forecast_year}"], result_df[col]
            )
            if m is not None:
                metrics[col] = m

        if metrics:
            # Build a DataFrame for display
            metrics_df = (
                pd.DataFrame(metrics)
                .T.reset_index()
                .rename(columns={"index": "Model"})
                .loc[:, ["Model", "MAPE", "MAE", "RMSE"]]
            )
            
            # Find best model for each metric
            best_mape = metrics_df.loc[metrics_df["MAPE"].idxmin(), "Model"]
            best_mae = metrics_df.loc[metrics_df["MAE"].idxmin(), "Model"]
            best_rmse = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
            
            st.dataframe(
                metrics_df.style.format({"MAPE": "{:.1f}%", "MAE": "{:.0f}", "RMSE": "{:.0f}"}),
                use_container_width=True
            )
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Best MAPE", best_mape, f"{metrics[best_mape]['MAPE']:.1f}%")
            col2.metric("Best MAE", best_mae, f"{metrics[best_mae]['MAE']:.0f}")
            col3.metric("Best RMSE", best_rmse, f"{metrics[best_rmse]['RMSE']:.0f}")
        else:
            st.info("No overlapping months of actuals and forecasts to compute metrics.")
    else:
        # Show forecast summaries instead
        st.subheader("📈 Forecast Summary")
        if model_cols:
            summary_data = []
            for col in model_cols + ["Ensemble_Forecast"]:
                total = result_df[col].sum()
                avg = result_df[col].mean()
                summary_data.append({
                    "Model": col.replace("_Forecast", ""),
                    "Total Forecast": f"{total:,.0f}",
                    "Monthly Average": f"{avg:,.0f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

    # 9)
