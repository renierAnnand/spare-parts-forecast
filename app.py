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
        
        # Determine date ranges
        min_date = df_grouped['Month'].min()
        max_date = df_grouped['Month'].max()
        
        # Forecast next 12 months after last available data
        forecast_start = max_date + pd.DateOffset(months=1)
        forecast_end = forecast_start + pd.DateOffset(months=11)
        
        st.write(f"🎯 Will forecast from {forecast_start.strftime('%Y-%m')} to {forecast_end.strftime('%Y-%m')}")
        
        # Create date ranges
        historical_months = pd.date_range(start=min_date, end=max_date, freq='MS')
        forecast_months = pd.date_range(start=forecast_start, end=forecast_end, freq='MS')
        
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
            
            # Generate forecasts
            if part_data['Month'].nunique() < 6:
                # Not enough data for forecasting
                for month in forecast_months:
                    month_key = month.strftime('%b-%Y')
                    row_data[month_key] = 'Not enough data'
            else:
                try:
                    # Prepare data for Prophet
                    model_df = part_data.rename(columns={'Month': 'ds', 'Sales': 'y'})
                    
                    # Create and fit Prophet model
                    model = Prophet(
                        changepoint_prior_scale=0.05,
                        seasonality_mode='multiplicative',
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False
                    )
                    
                    model.fit(model_df)
                    
                    # Create future predictions
                    future_dates = pd.DataFrame({'ds': forecast_months})
                    forecast = model.predict(future_dates)
                    
                    # Add forecast values
                    for j, month in enumerate(forecast_months):
                        month_key = month.strftime('%b-%Y')
                        forecast_value = forecast.iloc[j]['yhat']
                        row_data[month_key] = int(round(max(0, forecast_value), 0))
                    
                except Exception as e:
                    # Error in forecasting
                    for month in forecast_months:
                        month_key = month.strftime('%b-%Y')
                        row_data[month_key] = 'Error'
            
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
        
        # Display summary
        st.success("Forecasting completed!")
        st.write(f"Generated forecasts for {len(result_df)} parts")
        
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
            if any(year in month_col for year in ['2022', '2023']):
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
                        # Determine if historical or forecast
                        month_col = month_headers[col_num] if col_num < len(month_headers) else ''
                        if any(year in str(month_col) for year in ['2022', '2023']):
                            worksheet.write(row_num, col_num, cell_value, historical_format)
                        else:
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
