def show_data_processing_info(hist_df):
    """Show enhanced data processing information"""
    
    # Calculate correct metrics based on unique months
    unique_months = hist_df['Month'].nunique()
    total_sales = hist_df['Sales'].sum()
    avg_monthly_sales = hist_df.groupby('Month')['Sales'].sum().mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 Total Months", unique_months)
    with col2:
        st.metric("📈 Avg Monthly Sales", f"{avg_monthly_sales:,.0f}")
    with col3:
        data_quality = min(100, unique_months * 4.17)  # Quality score based on unique months
        st.metric("🎯 Data Quality Score", f"{data_quality:.0f}%")
    with col4:
        if 'log_transformed' in hist_df.columns and hist_df['log_transformed'].iloc[0]:
            st.metric("🔧 Data Transformation", "Log Applied")
        else:
            st.metric("🔧 Data Transformation", "None Applied")

    # Show additional data insights
    col1, col2 = st.columns(2)
    with col1:
        # Date range
        start_date = hist_df['Month'].min().strftime('%Y-%m')
        end_date = hist_df['Month'].max().strftime('%Y-%m')
        st.metric("📅 Data Range", f"{start_date} to {end_date}")
        
    with col2:
        # Total data points vs unique months
        total_rows = len(hist_df)
        if total_rows > unique_months:
            st.metric("📊 Data Points", f"{total_rows} rows ({unique_months} unique months)")
        else:
            st.metric("📊 Data Points", f"{total_rows}")

    # Show data breakdown if there are multiple entries per month
    if len(hist_df) > unique_months:
        avg_entries_per_month = len(hist_df) / unique_months
        st.info(f"📊 Your data contains multiple entries per month (avg: {avg_entries_per_month:.1f} entries/month). Sales are being aggregated by month for forecasting.")

    # Show seasonality and trend analysis
    col1, col2 = st.columns(2)
    with col1:
        # Seasonality detection - use monthly aggregated data
        monthly_data = hist_df.groupby('Month')['Sales'].sum().reset_index()
        if len(monthly_data) >= 24:
            try:
                decomposition = seasonal_decompose(monthly_data['Sales'], model='additive', period=12)
                seasonal_strength = np.var(decomposition.seasonal) / np.var(monthly_data['Sales'])
                st.metric("📊 Seasonality Strength", f"{seasonal_strength:.2%}")
            except:
                st.metric("📊 Seasonality", "Analysis unavailable")
        else:
            st.metric("📊 Seasonality", "Need 24+ months")
        
    with col2:
        # Trend detection - use monthly aggregated data
        if len(monthly_data) >= 12:
            try:
                recent_trend = np.polyfit(range(len(monthly_data['Sales'].tail(12))), monthly_data['Sales'].tail(12), 1)[0]
                trend_direction = "📈 Increasing" if recent_trend > 0 else "📉 Decreasing"
                st.metric("📈 Recent Trend", trend_direction)
            except:
                st.metric("📈 Recent Trend", "Analysis unavailable")
        else:
            st.metric("📈 Recent Trend", "Need 12+ months")

    # Show preprocessing results
    if 'Sales_Original' in hist_df.columns:
        with st.expander("🔧 Data Preprocessing Results"):
            col1, col2, col3 = st.columns(3)
            with col1:
                outliers_removed = (hist_df['Sales_Original'] != hist_df['Sales']).sum()
                st.metric("🎯 Outliers Handled", outliers_removed)
            with col2:
                if 'log_transformed' in hist_df.columns and hist_df['log_transformed'].iloc[0]:
                    st.info("📊 Log transformation applied to reduce skewness")
            with col3:
                st.metric("✅ Data Points", len(hist_df))@st.cache_data
def load_actual_2024_data(uploaded_file, forecast_year):
    """
    Load actual data with preprocessing - handles SPC format with multiple products per month
    """
    try:
        # Read Excel file - try to handle different sheet structures
        try:
            df = pd.read_excel(uploaded_file, header=None)
        except Exception:
            df = pd.read_excel(uploaded_file)
        
        # Check if it's the standard long format
        if len(df.columns) >= 2 and any(col in str(df.columns).lower() for col in ['month', 'sales']):
            # Standard long format handling
            if "Month" in df.columns and "Sales" in df.columns:
                df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
                if df["Month"].isna().any():
                    st.error("Some dates in the actual file could not be parsed.")
                    return None

                df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
                df["Sales"] = df["Sales"].abs()

                # Filter to the forecast year only
                start = pd.Timestamp(f"{forecast_year}-01-01")
                end = pd.Timestamp(f"{forecast_year+1}-01-01")
                df = df[(df["Month"] >= start) & (df["Month"] < end)]
                
                if df.empty:
                    st.warning(f"No rows match year {forecast_year}.")
                    return None

                # Only include months that have actual non-zero data
                monthly = df.groupby("Month", as_index=False)["Sales"].sum()
                monthly = monthly[monthly["Sales"] > 0]  # Only months with actual sales
                monthly = monthly.sort_values("Month").reset_index(drop=True)
                
                return monthly.rename(columns={"Sales": f"Actual_{forecast_year}"})
        
        else:
            # SPC Wide format handling - specifically for your file structure
            st.info("📊 Detected SPC wide format data - processing multiple products...")
            
            # Convert df to string representation to work with headers
            df_str = df.astype(str)
            
            # Find the header row (should contain month patterns)
            header_row_idx = 0
            for idx, row in df_str.iterrows():
                row_str = ' '.join(row.values)
                if f"-{forecast_year}" in row_str:
                    header_row_idx = idx
                    break
            
            # Extract headers from the identified row
            headers = df.iloc[header_row_idx].values.tolist()
            
            # Find month columns for the forecast year
            month_patterns = [
                f"Jan-{forecast_year}", f"Feb-{forecast_year}", f"Mar-{forecast_year}",
                f"Apr-{forecast_year}", f"May-{forecast_year}", f"Jun-{forecast_year}",
                f"Jul-{forecast_year}", f"Aug-{forecast_year}", f"Sep-{forecast_year}",
                f"Oct-{forecast_year}", f"Nov-{forecast_year}", f"Dec-{forecast_year}"
            ]
            
            # Find which columns contain our target year months
            available_month_cols = {}
            for i, header in enumerate(headers):
                header_str = str(header)
                for month_pattern in month_patterns:
                    if month_pattern in header_str:
                        available_month_cols[month_pattern] = i
                        break
            
            if not available_month_cols:
                st.error(f"No month columns found for {forecast_year}.")
                return None
            
            st.info(f"📅 Found data for months: {', '.join([m.split('-')[0] for m in available_month_cols.keys()])}")
            
            # Skip header rows and find data rows (those with Item Code pattern)
            data_start_row = header_row_idx + 2  # Skip header and QTY row
            
            melted_data = []
            
            # Process each data row
            for idx in range(data_start_row, len(df)):
                row = df.iloc[idx]
                
                # Check if this is a valid data row (has item code pattern)
                first_col = str(row.iloc[0])
                if pd.isna(row.iloc[0]) or first_col == 'nan' or len(first_col) < 3:
                    continue
                
                # Extract sales data for each available month
                for month_pattern, col_idx in available_month_cols.items():
                    if col_idx < len(row):
                        sales_value = row.iloc[col_idx]
                        sales_numeric = pd.to_numeric(sales_value, errors="coerce")
                        
                        if pd.notna(sales_numeric) and sales_numeric > 0:
                            # Convert month pattern to date
                            try:
                                month_str = month_pattern.replace("-", "-01-")
                                month_date = pd.to_datetime(month_str, format="%b-%d-%Y")
                                melted_data.append({
                                    "Month": month_date,
                                    "Sales": abs(sales_numeric),
                                    "Item_Code": first_col
                                })
                            except Exception as e:
                                continue
            
            if not melted_data:
                st.error("No valid sales data found in the file.")
                return None
            
            # Convert to DataFrame and aggregate by month
            long_df = pd.DataFrame(melted_data)
            
            # Group by month and sum all products for that month
            monthly = long_df.groupby("Month", as_index=False)["Sales"].sum()
            monthly = monthly[monthly["Sales"] > 0]  # Only months with actual sales data
            monthly = monthly.sort_values("Month").reset_index(drop=True)
            
            # Show processing results
            processed_months = monthly['Month'].dt.strftime('%b').tolist()
            total_products = long_df['Item_Code'].nunique()
            total_sales = monthly['Sales'].sum()
            
            st.success(f"✅ Successfully processed {total_products} products for: {', '.join(processed_months)}")
            st.info(f"📊 Total sales aggregated: {total_sales:,.0f} units")
            
            return monthly.rename(columns={"Sales": f"Actual_{forecast_year}"})
            
    except Exception as e:
        st.error(f"Error loading actual data: {str(e)}")
        st.error(f"Please ensure the file contains month columns in format 'Jan-{forecast_year}', 'Feb-{forecast_year}', etc.")
        return None
