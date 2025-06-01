import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
import io

st.title("📈 Spare Parts Forecast (Horizontal Layout)")

uploaded_file = st.file_uploader("Upload Excel file with: Part, Month, Sales", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        
        # Display basic info about uploaded data
        st.write(f"Data shape: {df.shape}")
        st.write("Data preview:")
        st.dataframe(df.head())
        
        # Detect data format and process accordingly
        if 'Part' in df.columns and 'Month' in df.columns and 'Sales' in df.columns:
            # Data is already in long format - perfect!
            st.success("✅ Detected correct long format (Part, Month, Sales)")
            st.write("Data preview:")
            st.dataframe(df.head(10))
            
        else:
            # Check if it's in wide format (months as columns)
            month_columns = [col for col in df.columns if any(month in col for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])]
            
            if len(month_columns) > 0:
                # Data is in wide format - convert to long format
                st.info("Detected wide format data (months as columns). Converting to long format...")
                
                # Assume first column contains part codes
                part_column = df.columns[0]
                
                # Melt the dataframe to convert from wide to long format
                df_long = df.melt(
                    id_vars=[part_column], 
                    value_vars=month_columns,
                    var_name='Month', 
                    value_name='Sales'
                )
                
                # Rename columns to standard format
                df_long = df_long.rename(columns={part_column: 'Part'})
                
                # Convert month strings to datetime
                def parse_month(month_str):
                    try:
                        # Try different formats
                        if '-' in month_str:
                            return pd.to_datetime(month_str, format='%b-%Y')
                        else:
                            # Assume current year if no year specified
                            return pd.to_datetime(f"{month_str}-2023", format='%b-%Y')
                    except:
                        return pd.NaT
                
                df_long['Month'] = df_long['Month'].apply(parse_month)
                df_long = df_long.dropna(subset=['Month'])
                
                # Use the converted long format data
                df = df_long
                st.write("✅ Successfully converted to long format")
                st.write("Converted data preview:")
                st.dataframe(df.head())
                
            else:
                st.error("Unable to detect data format. Please ensure your file has either:")
                st.write("- Columns named 'Part', 'Month', 'Sales' (long format), OR")
                st.write("- First column with part codes and month columns like 'Jan-2022', 'Feb-2022', etc. (wide format)")
                st.write("Available columns:", list(df.columns))
                return
        
        # Continue with the rest of the processing
        # Now process the data
        df['Month'] = pd.to_datetime(df['Month'])
        
        # Show data range information
        date_range = f"{df['Month'].min().strftime('%Y-%m')} to {df['Month'].max().strftime('%Y-%m')}"
        st.write(f"📅 Data covers: {date_range}")
        st.write(f"📦 Total parts: {df['Part'].nunique()}")
        st.write(f"📊 Total records: {len(df)}")
        
        # Clean data
        df = df.dropna()
        df = df.drop_duplicates()
        
        # Group data by part and month, summing sales for any duplicates
        df_grouped = df.groupby(['Part', pd.Grouper(key='Month', freq='M')])['Sales'].sum().reset_index()
        
        st.write(f"🔄 After cleaning: {df_grouped['Part'].nunique()} parts, {len(df_grouped)} records")
        
        # Determine date ranges for historical and forecast data based on available data
        min_date = df_grouped['Month'].min()
        max_date = df_grouped['Month'].max()
        
        # Use available historical data for training
        # For forecasting, predict next 12 months after the last available data
        forecast_start = max_date + pd.DateOffset(months=1)
        forecast_end = forecast_start + pd.DateOffset(months=11)  # 12 months total
        
        st.write(f"🎯 Will forecast from {forecast_start.strftime('%Y-%m')} to {forecast_end.strftime('%Y-%m')}")
        
        # Create date ranges
        historical_months = pd.date_range(start=min_date, end=max_date, freq='MS')
        forecast_months = pd.date_range(start=forecast_start, end=forecast_end, freq='MS')
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Prepare the results structure
        results_data = []
        total_parts = df_grouped['Part'].nunique()
        
        for i, part in enumerate(df_grouped['Part'].unique()):
            progress_bar.progress((i + 1) / total_parts)
            status_text.text(f'Processing part {i + 1}/{total_parts}: {part}')
            
            part_data = df_grouped[df_grouped['Part'] == part][['Month', 'Sales']].dropna()
            
            # Initialize row for this part
            row_data = {'Item Code': part}
            
            # Add historical data columns
            for month in historical_months:
                month_key = month.strftime('%b-%Y')
                historical_sales = part_data[part_data['Month'].dt.to_period('M') == month.to_period('M')]['Sales']
                row_data[month_key] = int(historical_sales.iloc[0]) if len(historical_sales) > 0 else ''
            
            # Check if we have enough data for forecasting (at least 6 months)
            if part_data['Month'].nunique() < 6:
                # Add forecast columns with "Not enough data" message
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
                    
                    # Suppress Prophet's verbose output
                    import logging
                    logging.getLogger('prophet').setLevel(logging.WARNING)
                    
                    model.fit(model_df)
                    
                    # Create future dataframe for forecasting
                    future_dates = pd.DataFrame({'ds': forecast_months})
                    forecast = model.predict(future_dates)
                    
                    # Add forecast data columns
                    for j, month in enumerate(forecast_months):
                        month_key = month.strftime('%b-%Y')
                        forecast_value = forecast.iloc[j]['yhat']
                        row_data[month_key] = int(round(max(0, forecast_value), 0))  # Round to whole numbers, ensure non-negative
                    
                except Exception as e:
                    # Add forecast columns with error message
                    for month in forecast_months:
                        month_key = month.strftime('%b-%Y')
                        row_data[month_key] = 'Error'
            
            results_data.append(row_data)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Create results dataframe
        result_df = pd.DataFrame(results_data)
        
        # Reorder columns: Item Code first, then chronological order
        all_months = list(historical_months) + list(forecast_months)
        month_columns = [month.strftime('%b-%Y') for month in all_months]
        column_order = ['Item Code'] + month_columns
        
        # Only include columns that exist in the dataframe
        existing_columns = [col for col in column_order if col in result_df.columns]
        result_df = result_df[existing_columns]
        
        # Display summary
        st.success("Forecasting completed!")
        st.write(f"Generated forecasts for {len(result_df)} parts")
        
        # Show preview of results
        st.write("**Data preview (first 10 parts and 8 columns):**")
        preview_df = result_df.iloc[:10, :8]  # Show first 10 rows and 8 columns
        st.dataframe(preview_df)
        
        # Create Excel file for download
        output = io.BytesIO()
        
        # Clean the result dataframe to handle NaN values before writing to Excel
        result_df_clean = result_df.fillna('')
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            result_df_clean.to_excel(writer, sheet_name='Sales Forecast', index=False)
            
            # Add formatting
            workbook = writer.book
            worksheet = writer.sheets['Sales Forecast']
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1,
                'align': 'center'
            })
            
            item_code_format = workbook.add_format({
                'bold': True,
                'fg_color': '#F2F2F2',
                'border': 1
            })
            
            historical_format = workbook.add_format({
                'fg_color': '#E8F4FD',
                'border': 1,
                'align': 'right'
            })
            
            forecast_format = workbook.add_format({
                'fg_color': '#FFF2CC',
                'border': 1,
                'align': 'right'
            })
            
            # Format headers
            for col_num, column_name in enumerate(result_df_clean.columns):
                worksheet.write(0, col_num, column_name, header_format)
            
            # Format data rows
            for row_num in range(1, len(result_df_clean) + 1):
                for col_num, column_name in enumerate(result_df_clean.columns):
                    cell_value = result_df_clean.iloc[row_num-1, col_num]
                    
                    if column_name == 'Item Code':
                        worksheet.write(row_num, col_num, cell_value, item_code_format)
                    elif any(year in column_name for year in ['2022', '2023']):  # Historical data
                        worksheet.write(row_num, col_num, cell_value, historical_format)
                    elif any(year in column_name for year in ['2024', '2025']):  # Forecast data
                        worksheet.write(row_num, col_num, cell_value, forecast_format)
                    else:
                        worksheet.write(row_num, col_num, cell_value)
            
            # Auto-adjust column widths
            for i, col in enumerate(result_df_clean.columns):
                if col == 'Item Code':
                    worksheet.set_column(i, i, 20)  # Wider for item codes
                else:
                    worksheet.set_column(i, i, 12)  # Standard width for data columns
            
            # Add header row to distinguish historical vs forecast
            forecast_start_col = len([col for col in result_df_clean.columns if any(year in col for year in ['2022', '2023'])]) + 1
            if forecast_start_col < len(result_df_clean.columns):
                worksheet.merge_range(0, forecast_start_col, 0, len(result_df_clean.columns)-1, 'Forecasted Data (2024-2025)', forecast_format)
        
        output.seek(0)
        
        st.download_button(
            label="📥 Download Excel File (Horizontal Layout)",
            data=output,
            file_name="Parts_Forecast_Horizontal_Layout.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
                        
                        # Create future dataframe for forecasting
                        future_dates = pd.DataFrame({'ds': forecast_months})
                        forecast = model.predict(future_dates)
                        
                        # Add forecast data columns
                        for j, month in enumerate(forecast_months):
                            month_key = month.strftime('%b-%Y')
                            if month_key not in row_data:  # Avoid overwriting historical data
                                forecast_value = forecast.iloc[j]['yhat']
                                row_data[month_key] = round(max(0, forecast_value), 0)  # Round to whole numbers
                        
                    except Exception as e:
                        st.warning(f"Error processing part '{part}': {str(e)}")
                        # Add forecast columns with error message
                        for month in forecast_months:
                            month_key = month.strftime('%b-%Y')
                            if month_key not in row_data:
                                row_data[month_key] = f'Error: {str(e)}'
                
                results_data.append(row_data)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Create results dataframe
            result_df = pd.DataFrame(results_data)
            
            # Reorder columns: Item Code first, then chronological order
            all_months = list(historical_months) + list(forecast_months)
            month_columns = [month.strftime('%b-%Y') for month in all_months]
            column_order = ['Item Code'] + month_columns
            
            # Only include columns that exist in the dataframe
            existing_columns = [col for col in column_order if col in result_df.columns]
            result_df = result_df[existing_columns]
            
            # Display summary
            st.success("Forecasting completed!")
            st.write(f"Generated forecasts for {len(result_df)} parts")
            
            # Show preview of results
            st.write("**Data preview (first 10 parts and 8 columns):**")
            preview_df = result_df.iloc[:10, :8]  # Show first 10 rows and 8 columns
            st.dataframe(preview_df)
            
            # Create Excel file for download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter', options={'nan_inf_to_errors': True}) as writer:
                result_df.to_excel(writer, sheet_name='Sales Forecast', index=False)
                
                # Add formatting
                workbook = writer.book
                worksheet = writer.sheets['Sales Forecast']
                
                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1,
                    'align': 'center'
                })
                
                item_code_format = workbook.add_format({
                    'bold': True,
                    'fg_color': '#F2F2F2',
                    'border': 1
                })
                
                historical_format = workbook.add_format({
                    'fg_color': '#E8F4FD',
                    'border': 1,
                    'align': 'right'
                })
                
                forecast_format = workbook.add_format({
                    'fg_color': '#FFF2CC',
                    'border': 1,
                    'align': 'right'
                })
                
                # Format headers
                for col_num, column_name in enumerate(result_df.columns):
                    worksheet.write(0, col_num, column_name, header_format)
                
                # Format data rows
                for row_num in range(1, len(result_df) + 1):
                    for col_num, column_name in enumerate(result_df.columns):
                        cell_value = result_df.iloc[row_num-1, col_num]
                        
                        if column_name == 'Item Code':
                            worksheet.write(row_num, col_num, cell_value, item_code_format)
                        elif any(year in column_name for year in ['2022', '2023']):  # Historical data
                            worksheet.write(row_num, col_num, cell_value, historical_format)
                        elif any(year in column_name for year in ['2024', '2025']):  # Forecast data
                            worksheet.write(row_num, col_num, cell_value, forecast_format)
                        else:
                            worksheet.write(row_num, col_num, cell_value)
                
                # Auto-adjust column widths
                for i, col in enumerate(result_df.columns):
                    if col == 'Item Code':
                        worksheet.set_column(i, i, 20)  # Wider for item codes
                    else:
                        worksheet.set_column(i, i, 12)  # Standard width for data columns
                
                # Add header row to distinguish historical vs forecast
                worksheet.merge_range('B1:Y1', 'Historical Data (2022-2023)', historical_format)
                forecast_start_col = len([col for col in result_df.columns if any(year in col for year in ['2022', '2023'])]) + 1
                worksheet.merge_range(0, forecast_start_col, 0, len(result_df.columns)-1, 'Forecasted Data (2024-2025)', forecast_format)
            
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
