import streamlit as st

# Configure streamlit FIRST - must be before any other st commands
st.set_page_config(page_title="Enhanced Hierarchical Sales Forecasting Dashboard", layout="wide")

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

# Forecasting libraries
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox

# Machine learning libraries
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin

import warnings
warnings.filterwarnings("ignore")


class MetaLearner(BaseEstimator, RegressorMixin):
    """Meta-learner for model stacking"""
    def __init__(self):
        self.meta_model = Ridge(alpha=1.0)
        
    def fit(self, X, y):
        self.meta_model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.meta_model.predict(X)


def process_wide_hierarchical_format(df, year):
    """Process wide format specifically for your data structure"""
    try:
        st.info("🔧 Processing wide hierarchical format...")
        
        # Your data structure: Row 0 has months, Row 1 has headers, Data starts from Row 2
        month_row = df.iloc[0]  # MonthYear row with Jan-2022, Feb-2022, etc.
        header_row = df.iloc[1]  # Item Code, Item Description, Brand, Engine, QTY...
        
        st.info(f"📅 Month row: {month_row.tolist()[:8]}")
        st.info(f"📋 Header row: {header_row.tolist()[:8]}")
        
        # Find month columns - more aggressive search
        month_cols = []
        month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        
        # Check ALL columns for month patterns
        for i in range(len(df.columns)):
            month_val = month_row.iloc[i] if i < len(month_row) else None
            if pd.notna(month_val):
                month_str = str(month_val).lower().strip()
                
                # Check for various month patterns
                for month in month_names:
                    if month in month_str:
                        month_cols.append((i, month_val, month))
                        break
                
                # Also check if it looks like a date pattern
                if any(pattern in month_str for pattern in ['-2022', '-2023', '-2024', '/2022', '/2023', '/2024']):
                    # Try to extract month from date-like patterns
                    for month in month_names:
                        if month in month_str:
                            month_cols.append((i, month_val, month))
                            break
        
        # Remove duplicates while preserving order
        seen_months = set()
        unique_month_cols = []
        for col_idx, col_name, month_name in month_cols:
            if month_name not in seen_months:
                unique_month_cols.append((col_idx, col_name, month_name))
                seen_months.add(month_name)
        
        month_cols = unique_month_cols
        
        st.info(f"📅 Found {len(month_cols)} month columns: {[col[1] for col in month_cols]}")
        
        # If still no month columns, try looking at column headers that contain 'QTY'
        if not month_cols:
            st.warning("🔍 No month columns found, trying QTY column approach...")
            qty_cols = []
            for i in range(4, len(df.columns)):  # Start from column 4
                header_val = header_row.iloc[i] if i < len(header_row) else None
                if pd.notna(header_val) and 'qty' in str(header_val).lower():
                    # This might be a monthly sales column
                    month_val = month_row.iloc[i] if i < len(month_row) else None
                    if pd.notna(month_val):
                        # Try to map to month based on position
                        month_index = len(qty_cols)
                        if month_index < 12:
                            month_name = month_names[month_index]
                            qty_cols.append((i, f"{month_name}-{year}", month_name))
            
            if qty_cols:
                month_cols = qty_cols
                st.info(f"📅 Using QTY columns as months: {[col[1] for col in month_cols]}")
        
        if not month_cols:
            st.error("❌ No month columns found in wide format")
            st.info("🔍 Debug information:")
            st.info(f"First row (months): {month_row.tolist()}")
            st.info(f"Second row (headers): {header_row.tolist()}")
            return None
        
        # Process data rows (starting from row 2)
        long_data = []
        rows_processed = 0
        
        for idx in range(2, len(df)):
            row = df.iloc[idx]
            
            # Extract categorical data (first 4 columns)
            item_code = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
            description = str(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else ""
            brand = str(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else "Unknown"
            engine = str(row.iloc[3]) if len(row) > 3 and pd.notna(row.iloc[3]) else "Unknown"
            
            # Skip empty rows
            if item_code == "" or item_code.lower() in ['nan', 'none', 'null']:
                continue
                
            rows_processed += 1
            
            # Process monthly sales
            for col_idx, col_name, month_name in month_cols:
                if col_idx < len(row):
                    sales_value = row.iloc[col_idx]
                    
                    try:
                        sales_value = float(sales_value) if pd.notna(sales_value) else 0
                    except:
                        sales_value = 0
                    
                    if sales_value > 0:  # Only include positive sales
                        # Create proper date
                        month_num = {
                            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                        }.get(month_name, 1)
                        
                        month_date = pd.Timestamp(year=year, month=month_num, day=1)
                        
                        long_data.append({
                            'Month': month_date,
                            'Sales': sales_value,
                            'Item_Code': item_code,
                            'Description': description,
                            'Brand': brand,
                            'Engine': engine
                        })
        
        st.info(f"📈 Processed {rows_processed} product rows, created {len(long_data)} data points")
        
        if not long_data:
            st.warning(f"❌ No valid sales data found for {year}")
            st.info("🔍 Sample data from first few rows:")
            for idx in range(2, min(5, len(df))):
                row = df.iloc[idx]
                st.info(f"Row {idx}: {row.iloc[:8].tolist()}")
            return None
        
        result_df = pd.DataFrame(long_data)
        
        # Show sample of processed data
        st.info(f"📋 Sample processed data: {len(result_df)} records")
        with st.expander(f"Preview {year} processed data"):
            st.dataframe(result_df.head(10))
        
        return create_monthly_aggregation(result_df, year)
        
    except Exception as e:
        st.error(f"❌ Error in wide format processing: {str(e)}")
        return None


def create_monthly_aggregation(result_df, year):
    """Create monthly aggregation with categorical features"""
    # Aggregate by month (sum across all items)
    monthly_agg = result_df.groupby('Month').agg({
        'Sales': 'sum',
        'Brand': lambda x: '|'.join(x.unique()),
        'Engine': lambda x: '|'.join(x.unique())
    }).reset_index()
    
    # Add enhanced categorical features
    monthly_agg['Year'] = year
    monthly_agg['Sales_Original'] = monthly_agg['Sales'].copy()
    
    # Calculate brand diversity (number of unique brands per month)
    brand_diversity = result_df.groupby('Month')['Brand'].nunique().reset_index()
    brand_diversity.columns = ['Month', 'Brand_Diversity']
    monthly_agg = monthly_agg.merge(brand_diversity, on='Month')
    
    # Calculate engine diversity
    engine_diversity = result_df.groupby('Month')['Engine'].nunique().reset_index()
    engine_diversity.columns = ['Month', 'Engine_Diversity']
    monthly_agg = monthly_agg.merge(engine_diversity, on='Month')
    
    # Top brand sales share per month
    for month in monthly_agg['Month']:
        month_data = result_df[result_df['Month'] == month]
        if len(month_data) > 0:
            brand_sales = month_data.groupby('Brand')['Sales'].sum()
            top_brand_share = brand_sales.max() / brand_sales.sum() if brand_sales.sum() > 0 else 0
            monthly_agg.loc[monthly_agg['Month'] == month, 'Top_Brand_Share'] = top_brand_share
        else:
            monthly_agg.loc[monthly_agg['Month'] == month, 'Top_Brand_Share'] = 0
    
    # Product diversity index
    monthly_agg['Product_Diversity_Index'] = monthly_agg['Brand_Diversity'] * monthly_agg['Engine_Diversity']
    
    st.success(f"✅ Successfully processed {year}: {len(monthly_agg)} months with enhanced categorical features")
    st.info(f"📊 Brand diversity range: {monthly_agg['Brand_Diversity'].min()}-{monthly_agg['Brand_Diversity'].max()}")
    st.info(f"🔧 Engine diversity range: {monthly_agg['Engine_Diversity'].min()}-{monthly_agg['Engine_Diversity'].max()}")
    
    return monthly_agg


def process_hierarchical_format(df, year):
    """Process the enhanced hierarchical format with brands and engines"""
    try:
        st.info(f"🔍 Analyzing hierarchical structure for {year}...")
        
        # Special handling for your specific format
        if len(df.columns) >= 16:
            st.info("📋 Detected wide format with 16+ columns - using optimized processing")
            return process_wide_hierarchical_format(df, year)
        
        # Fallback for other formats
        st.warning("❌ Format not recognized as hierarchical")
        return None
        
    except Exception as e:
        st.error(f"❌ Error processing hierarchical format: {str(e)}")
        return None


def process_standard_format(df, year):
    """Process standard format (Month, Sales columns)"""
    try:
        if "Month" not in df.columns or "Sales" not in df.columns:
            st.error("Standard format requires 'Month' and 'Sales' columns.")
            return None

        # Parse dates
        df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
        df = df.dropna(subset=['Month'])

        # Clean sales data
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
        df["Sales"] = df["Sales"].abs()

        # Filter to the specified year
        df = df[df['Month'].dt.year == year]

        if len(df) == 0:
            st.warning(f"No data found for year {year}")
            return None

        # Aggregate by month
        monthly_df = df.groupby('Month', as_index=False).agg({
            'Sales': 'sum'
        }).sort_values('Month').reset_index(drop=True)

        # Add basic categorical features (placeholders)
        monthly_df['Year'] = year
        monthly_df['Sales_Original'] = monthly_df['Sales'].copy()
        monthly_df['Brand'] = 'Standard'
        monthly_df['Engine'] = 'Standard'
        monthly_df['Brand_Diversity'] = 1
        monthly_df['Engine_Diversity'] = 1
        monthly_df['Top_Brand_Share'] = 1.0

        return monthly_df

    except Exception as e:
        st.error(f"Error processing standard format: {str(e)}")
        return None


@st.cache_data
def load_enhanced_hierarchical_data(uploaded_files):
    """Load and process hierarchical sales data with brand, engine, and product categorization"""
    if not uploaded_files:
        st.error("Please upload at least one data file.")
        return None
    
    all_data = []
    years_processed = []
    
    for uploaded_file in uploaded_files:
        try:
            # Read Excel file
            df = pd.read_excel(uploaded_file)
            
            # Detect the year from the filename
            filename = uploaded_file.name
            if '2022' in filename:
                year = 2022
            elif '2023' in filename:
                year = 2023
            elif '2024' in filename:
                year = 2024
            else:
                year = 2022  # Default
            
            years_processed.append(year)
            
            # Enhanced detection for hierarchical format
            is_hierarchical = False
            
            # Check multiple indicators for hierarchical format
            for check_row in range(min(5, len(df))):
                row_values = df.iloc[check_row].astype(str).str.lower()
                if any('brand' in val for val in row_values) or any('engine' in val for val in row_values):
                    is_hierarchical = True
                    break
                if any('item code' in val for val in row_values) or any('description' in val for val in row_values):
                    is_hierarchical = True
                    break
            
            # Also check if we have monthly columns
            first_row = df.iloc[0].astype(str)
            has_month_cols = any(month in val.lower() for val in first_row for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun'])
            
            if is_hierarchical or has_month_cols:
                st.info(f"📊 Processing enhanced hierarchical data for {year}...")
                
                # Process hierarchical format
                df_processed = process_hierarchical_format(df, year)
                if df_processed is not None:
                    all_data.append(df_processed)
                else:
                    st.warning(f"Failed to process hierarchical format for {year}, trying standard format...")
                    df_processed = process_standard_format(df, year)
                    if df_processed is not None:
                        all_data.append(df_processed)
            
            else:
                # Fallback to standard format
                st.info(f"📊 Processing standard format for {year}...")
                df_processed = process_standard_format(df, year)
                if df_processed is not None:
                    all_data.append(df_processed)
                    
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    if not all_data:
        st.error("No valid data files could be processed.")
        return None
    
    # Combine all years
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('Month').reset_index(drop=True)
    
    # Enhanced preprocessing
    final_df = enhanced_preprocessing(combined_df)
    
    st.success(f"✅ Successfully processed {len(years_processed)} years: {', '.join(map(str, years_processed))}")
    st.info(f"📈 Total data points: {len(final_df)} | Date range: {final_df['Month'].min().strftime('%Y-%m')} to {final_df['Month'].max().strftime('%Y-%m')}")
    
    return final_df


def enhanced_preprocessing(df):
    """Enhanced preprocessing with categorical feature engineering"""
    # Store original sales for reference
    df['Sales_Original'] = df['Sales'].copy()
    
    # 1. Outlier Detection and Treatment using IQR
    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers instead of removing (preserves data points)
    outliers_detected = ((df['Sales'] < lower_bound) | (df['Sales'] > upper_bound)).sum()
    if outliers_detected > 0:
        st.info(f"📊 Detected and capped {outliers_detected} outliers for better model stability")
        df['Sales'] = df['Sales'].clip(lower=lower_bound, upper=upper_bound)
    
    # 2. Handle missing values with interpolation
    if df['Sales'].isna().any():
        df['Sales'] = df['Sales'].interpolate(method='time')
    
    # 3. Data transformation - test for optimal transformation
    skewness = stats.skew(df['Sales'])
    if abs(skewness) > 1:  # Highly skewed data
        st.info(f"📈 Data skewness detected ({skewness:.2f}). Applying log transformation for better modeling.")
        df['Sales'] = np.log1p(df['Sales'])  # log1p handles zeros better
        df['log_transformed'] = True
    else:
        df['log_transformed'] = False
    
    # 4. Enhanced categorical features
    # Year-over-year growth rate
    df['YoY_Growth'] = df['Sales'].pct_change(12)
    
    # Seasonal strength
    if len(df) >= 24:
        try:
            decomposition = seasonal_decompose(df['Sales'], model='additive', period=12)
            df['Seasonal_Component'] = decomposition.seasonal
            df['Trend_Component'] = decomposition.trend
        except:
            df['Seasonal_Component'] = 0
            df['Trend_Component'] = df['Sales']
    else:
        df['Seasonal_Component'] = 0
        df['Trend_Component'] = df['Sales']
    
    # Market concentration metrics
    df['Market_Concentration'] = df['Top_Brand_Share']
    df['Product_Diversity_Index'] = df['Brand_Diversity'] * df['Engine_Diversity']
    
    return df


def run_fallback_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Enhanced fallback forecasting with categorical insights"""
    try:
        work_data = data.copy()
        log_transformed = work_data.get('log_transformed', [False])[0] if len(work_data) > 0 else False
        
        if len(work_data) >= 12:
            # Use seasonal naive with trend and categorical adjustment
            seasonal_pattern = work_data['Sales'].tail(12).values
            recent_trend = np.polyfit(range(len(work_data['Sales'].tail(12))), work_data['Sales'].tail(12), 1)[0]
            
            # Categorical adjustment factor
            adjustment_factor = 1.0
            if 'YoY_Growth' in work_data.columns:
                avg_growth = work_data['YoY_Growth'].tail(6).mean()
                if not pd.isna(avg_growth):
                    adjustment_factor = 1 + avg_growth
            
            forecast = []
            for i in range(forecast_periods):
                seasonal_val = seasonal_pattern[i % 12]
                trend_adjustment = recent_trend * (i + 1)
                categorical_adjustment = seasonal_val * (adjustment_factor - 1)
                
                final_val = seasonal_val + trend_adjustment + categorical_adjustment
                forecast.append(max(final_val, seasonal_val * 0.5))
            
            forecast = np.array(forecast)
            
            # Reverse log transformation if applied
            if log_transformed:
                forecast = np.expm1(forecast)
            
            # Apply scaling
            forecast = forecast * scaling_factor
            
            return forecast
        else:
            base_forecast = work_data['Sales'].mean()
            
            # Reverse log transformation if applied
            if log_transformed:
                base_forecast = np.expm1(base_forecast)
            
            # Apply scaling
            base_forecast = base_forecast * scaling_factor
            
            return np.array([base_forecast] * forecast_periods)
            
    except Exception as e:
        # Ultimate fallback
        historical_mean = data['Sales'].mean() if len(data) > 0 else 1000
        return np.array([historical_mean * scaling_factor] * forecast_periods)


def main():
    """Main function for the enhanced hierarchical forecasting dashboard"""
    st.title("🚀 Enhanced Hierarchical Sales Forecasting Dashboard")
    st.markdown("**Next-generation forecasting with brand/engine categorization and market intelligence**")

    # Sidebar configuration
    st.sidebar.header("⚙️ Enhanced Configuration")
    forecast_year = st.sidebar.selectbox(
        "Select forecast year:",
        options=[2024, 2025, 2026],
        index=0
    )

    # File uploads
    st.subheader("📁 Upload Enhanced Data Files")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Historical Data (Multiple Years)**")
        historical_files = st.file_uploader(
            "📊 Upload Historical Sales Data",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            help="Upload 2022, 2023 data with Brand/Engine columns for enhanced forecasting"
        )

    if not historical_files:
        st.info("👆 Please upload historical sales data files (2022, 2023) to begin enhanced forecasting.")
        return

    # Load and validate enhanced data
    hist_df = load_enhanced_hierarchical_data(historical_files)
    if hist_df is None:
        return

    # Display enhanced analytics
    st.subheader("📊 Enhanced Hierarchical Data Analysis")

    # Enhanced metrics
    unique_months = hist_df['Month'].nunique()
    total_sales = hist_df['Sales'].sum()
    avg_monthly_sales = hist_df['Sales'].mean()
    
    # Calculate categorical insights
    avg_brand_diversity = hist_df.get('Brand_Diversity', pd.Series([1])).mean()
    avg_market_concentration = hist_df.get('Top_Brand_Share', pd.Series([1])).mean()
    avg_product_diversity = hist_df.get('Product_Diversity_Index', pd.Series([1])).mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 Total Months", unique_months)
    with col2:
        st.metric("📈 Avg Monthly Sales", f"{avg_monthly_sales:,.0f}")
    with col3:
        st.metric("🏷️ Avg Brand Diversity", f"{avg_brand_diversity:.1f}")
    with col4:
        st.metric("🎯 Market Concentration", f"{avg_market_concentration:.1%}")

    # Generate a simple forecast
    if st.button("🚀 Generate Simple Forecast", type="primary"):
        st.subheader("📈 Simple Forecast Results")
        
        # Create forecast dates
        forecast_dates = pd.date_range(
            start=f"{forecast_year}-01-01",
            end=f"{forecast_year}-12-01",
            freq='MS'
        )
        
        # Generate simple forecast
        forecast_values = run_fallback_forecast(hist_df, forecast_periods=12, scaling_factor=1.0)
        
        # Create results dataframe
        result_df = pd.DataFrame({
            "Month": forecast_dates,
            "Simple_Forecast": forecast_values
        })
        
        # Display results
        display_df = result_df.copy()
        display_df['Month'] = display_df['Month'].dt.strftime('%Y-%m')
        display_df['Simple_Forecast'] = display_df['Simple_Forecast'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result_df['Month'],
            y=result_df['Simple_Forecast'],
            mode='lines+markers',
            name='📈 Simple Forecast',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='📊 Simple Forecast Results',
            xaxis_title='Month',
            yaxis_title='Sales Volume',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary
        annual_total = result_df['Simple_Forecast'].sum()
        monthly_avg = result_df['Simple_Forecast'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Annual Forecast", f"{annual_total:,.0f}")
        with col2:
            st.metric("📈 Monthly Average", f"{monthly_avg:,.0f}")


if __name__ == "__main__":
    main()
