import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
import io
import logging

st.title("📈 Spare Parts Forecast (Horizontal Layout)")

uploaded_file = st.file_uploader("Upload Excel file with: Part, Month, Sales", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        
        # Display basic info about uploaded data
        st.write(f"Data shape: {df.shape}")
        st.write("Data preview:")
        st.dataframe(df.head())
        
        # Check if data is in correct format
        if 'Part' in df.columns and 'Month' in df.columns and 'Sales' in df.columns:
            st.success("✅ Detected correct long format (Part, Month, Sales)")
        else:
            st.error("Please ensure your file has columns named: Part, Month, Sales")
            st.write("Available columns:", list(df.columns))
            st.stop()
        
        # Process the data
        df['Month'] = pd.to_datetime(df['Month'])
        
        # Show data range information
        date_range = f"{df['Month'].min().strftime('%Y-%m')} to {df['Month'].max().strftime('%Y-%m')}"
        st.write(f"📅 Data covers: {date_range}")
        st.write(f"📦 Total parts: {df['Part'].nunique()}")
        st.write(f"📊 Total records: {len(df)}")
        
        # Clean data
        df = df.dropna()
        df = df.drop_duplicates()
        
        # Group data by part and month
        df_grouped = df.groupby(['Part', pd.Grouper(key='Month', freq='M')])['Sales'].sum().reset_index()
        
        st.write(f"🔄 After cleaning: {df_grouped['Part'].nunique()} parts, {len(df_grouped)} records")
        
        # Determine date ranges - Use ALL historical data for training
        min_date = df_grouped['Month'].min()
        max_date = df_grouped['Month'].max()
        
        # Calculate data span
        data_span_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
        data_span_years = data_span_months / 12
        
        st.write(f"📅 **Historical Data Range**: {min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}")
        st.write(f"📊 **Data Span**: {data_span_months} months ({data_span_years:.1f} years)")
        
        # Forecast starts from the month AFTER the last historical data
        forecast_start = max_date + pd.DateOffset(months=1)
        forecast_end = forecast_start + pd.DateOffset(months=11)  # 12 months forecast
        
        st.success(f"🔮 **Prediction Period**: {forecast_start.strftime('%Y-%m')} to {forecast_end.strftime('%Y-%m')}")
        st.info(f"✅ **Training on**: {data_span_months} months of historical data")
        st.info(f"🎯 **Forecasting**: Next 12 months after {max_date.strftime('%Y-%m')}")
        
        # Create date ranges
        historical_months = pd.date_range(start=min_date, end=max_date, freq='MS')
        forecast_months = pd.date_range(start=forecast_start, end=forecast_end, freq='MS')
        
        # Validate data quality
        if data_span_months < 12:
            st.warning(f"⚠️ **Limited Data**: Only {data_span_months} months available. Consider having at least 12-24 months for better predictions.")
        elif data_span_months >= 24:
            st.success(f"✅ **Excellent Data**: {data_span_months} months of data will produce reliable forecasts!")
        else:
            st.info(f"✅ **Good Data**: {data_span_months} months should produce reasonable forecasts.")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each part
        results_data = []
        total_parts = df_grouped['Part'].nunique()
        
        # Suppress Prophet logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        
        for i, part in enumerate(df_grouped['Part'].unique()):
            progress_bar.progress((i + 1) / total_parts)
            status_text.text(f'Processing part {i + 1}/{total_parts}: {part}')
            
            part_data = df_grouped[df_grouped['Part'] == part][['Month', 'Sales']].dropna()
            
            # Initialize row for this part
            row_data = {'Item Code': part}
            
            # Add historical data
            for month in historical_months:
                month_key = month.strftime('%b-%Y')
                historical_sales = part_data[part_data['Month'].dt.to_period('M') == month.to_period('M')]['Sales']
                row_data[month_key] = int(historical_sales.iloc[0]) if len(historical_sales) > 0 else ''
            
            # Generate forecasts using ALL historical data
            if part_data['Month'].nunique() < 3:
                # Not enough data for forecasting
                for month in forecast_months:
                    month_key = month.strftime('%b-%Y')
                    row_data[month_key] = 'Not enough data (min 3 months)'
            else:
                try:
                    # Use ALL available historical data for training
                    model_df = part_data.rename(columns={'Month': 'ds', 'Sales': 'y'})
                    
                    # Debug info for first part only
                    if i == 0:
                        st.write(f"**Example Training Data for '{part}':**")
                        st.write(f"- Historical data: {model_df['ds'].min().strftime('%Y-%m')} to {model_df['ds'].max().strftime('%Y-%m')}")
                        st.write(f"- Training records: {len(model_df)} months")
                        st.write(f"- Predicting: {forecast_start.strftime('%Y-%m')} to {forecast_end.strftime('%Y-%m')}")
                        st.write(f"- Sample historical sales: {model_df['y'].head(3).tolist()}")
                    
                    # Create Prophet model optimized for your data span
                    if data_span_months >= 24:
                        # Enough data for full seasonality
                        model = Prophet(
                            changepoint_prior_scale=0.1,
                            seasonality_mode='multiplicative',
                            yearly_seasonality=True,
                            weekly_seasonality=False,
                            daily_seasonality=False
                        )
                    else:
                        # Limited data - simpler model
                        model = Prophet(
                            changepoint_prior_scale=0.05,
                            seasonality_mode='additive',
                            yearly_seasonality=False,
                            weekly_seasonality=False,
                            daily_seasonality=False
                        )
                    
                    model.fit(model_df)
                    
                    # Create future predictions for the next 12 months
                    future_dates = pd.DataFrame({'ds': forecast_months})
                    forecast = model.predict(future_dates)
                    
                    # Add realistic forecast values
                    for j, month in enumerate(forecast_months):
                        month_key = month.strftime('%b-%Y')
                        forecast_value = forecast.iloc[j]['yhat']
                        
                        # Ensure non-negative and realistic values
                        predicted_value = max(0, round(forecast_value, 0))
                        row_data[month_key] = int(predicted_value)
                    
                except Exception as e:
                    # Error in forecasting
                    for month in forecast_months:
                        month_key = month.strftime('%b-%Y')
                        row_data[month_key] = f'Error: {str(e)[:20]}'
            
            results_data.append(row_data)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Create results dataframe
        result_df = pd.DataFrame(results_data)
        
        # Reorder columns chronologically
        all_months = list(historical_months) + list(forecast_months)
        month_columns = [month.strftime('%b-%Y') for month in all_months]
        column_order = ['Item Code'] + month_columns
        
        existing_columns = [col for col in column_order if col in result_df.columns]
        result_df = result_df[existing_columns]
        
        # Display summary with validation
        st.success("Forecasting completed!")
        st.write(f"Generated forecasts for {len(result_df)} parts")
        
        # VALIDATION: Check if predictions look realistic
        forecast_cols = [col for col in result_df.columns if any(year in col for year in ['2024', '2025', '2026'])]
        if forecast_cols:
            sample_forecasts = []
            for col in forecast_cols[:3]:  # Check first 3 forecast columns
                col_values = result_df[col].tolist()
                numeric_values = []
                for val in col_values:
                    if isinstance(val, str) and '(' in val:
                        try:
                            numeric_values.append(int(val.split('(')[0].strip()))
                        except:
                            pass
                    elif isinstance(val, (int, float)):
                        numeric_values.append(val)
                
                if numeric_values:
                    sample_forecasts.extend(numeric_values[:5])
            
            if sample_forecasts:
                st.write(f"**Forecast Validation:**")
                st.write(f"- Sample predicted values: {sample_forecasts[:10]}")
                st.write(f"- Average forecast: {sum(sample_forecasts) / len(sample_forecasts):.0f}")
                st.write(f"- Min/Max forecasts: {min(sample_forecasts)} / {max(sample_forecasts)}")
        
        # Show preview
        st.write("**Data preview (first 5 parts and 8 columns):**")
        preview_df = result_df.iloc[:5, :8]
        st.dataframe(preview_df)
        
        # Create Excel file with special formatting
        output = io.BytesIO()
        result_df_clean = result_df.fillna('')
        
        # Create header structure
        month_headers = ['Item Code']
        data_type_headers = ['']
        
        month_columns = [col for col in result_df_clean.columns if col != 'Item Code']
        
        for month_col in month_columns:
            month_headers.append(month_col)
            # Check if this is historical or forecast data
            month_date = pd.to_datetime(month_col, format='%b-%Y', errors='ignore')
            if pd.isna(month_date) or month_date <= max_date:
                data_type_headers.append('Historical QTY')
            else:
                data_type_headers.append('Forecasted QTY')
        
        # Build restructured data
        restructured_data = [month_headers, data_type_headers]
        
        for _, row in result_df_clean.iterrows():
            data_row = [row[col] for col in result_df_clean.columns]
            restructured_data.append(data_row)
        
        # Ensure all rows have same length
        max_cols = max(len(row) for row in restructured_data)
        for row in restructured_data:
            while len(row) < max_cols:
                row.append('')
        
        restructured_df = pd.DataFrame(restructured_data)
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            restructured_df.to_excel(writer, sheet_name='Sales Forecast', index=False, header=False)
            
            # Add formatting
            workbook = writer.book
            worksheet = writer.sheets['Sales Forecast']
            
            # Define formats
            month_header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'align': 'center',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            data_type_header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'align': 'center',
                'fg_color': '#F2F2F2',
                'border': 1,
                'font_size': 9
            })
            
            item_code_format = workbook.add_format({
                'bold': True,
                'fg_color': '#F2F2F2',
                'border': 1,
                'align': 'left'
            })
            
            historical_format = workbook.add_format({
                'fg_color': '#E8F4FD',
                'border': 1,
                'align': 'right',
                'num_format': '#,##0'
            })
            
            forecast_format = workbook.add_format({
                'fg_color': '#FFF2CC',
                'border': 1,
                'align': 'right',
                'num_format': '#,##0'
            })
            
            # Format month header row
            for col_num in range(len(month_headers)):
                worksheet.write(0, col_num, month_headers[col_num], month_header_format)
            
            # Format data type header row
            for col_num in range(len(data_type_headers)):
                worksheet.write(1, col_num, data_type_headers[col_num], data_type_header_format)
            
            # Format data rows
            for row_num in range(2, len(restructured_data)):
                for col_num in range(len(restructured_data[row_num])):
                    cell_value = restructured_data[row_num][col_num]
                    
                    if col_num == 0:  # Item Code column
                        worksheet.write(row_num, col_num, cell_value, item_code_format)
                    else:
                        # Determine if historical or forecast based on actual date
                        month_col = month_headers[col_num] if col_num < len(month_headers) else ''
                        try:
                            month_date = pd.to_datetime(month_col, format='%b-%Y', errors='ignore')
                            if pd.isna(month_date) or month_date <= max_date:
                                worksheet.write(row_num, col_num, cell_value, historical_format)
                            else:
                                worksheet.write(row_num, col_num, cell_value, forecast_format)
                        except:
                            worksheet.write(row_num, col_num, cell_value, forecast_format)
            
            # Set column widths
            worksheet.set_column(0, 0, 20)  # Item Code column
            for col_num in range(1, max_cols):
                worksheet.set_column(col_num, col_num, 12)
            
            # Freeze panes
            worksheet.freeze_panes(2, 1)
        
        output.seek(0)
        
        st.download_button(
            label="📥 Download Excel File (Horizontal Layout)",
            data=output,
            file_name="Parts_Forecast_Horizontal_Layout.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please make sure your Excel file has the columns: Part, Month, Sales")
