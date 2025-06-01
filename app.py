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
        
        # Check if required columns exist
        required_columns = ['Part', 'Month', 'Sales']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.write("Available columns:", list(df.columns))
        else:
            df['Month'] = pd.to_datetime(df['Month'])
            
            # Filter for training data (use last 2 years of available data for training)
            max_date = df['Month'].max()
            training_start = max_date - pd.DateOffset(years=2)
            training_data = df[df['Month'] >= training_start]
            
            st.write(f"Using training data from {training_start.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}")
            
            # Clean and group data
            training_data = training_data.dropna()
            training_data = training_data.drop_duplicates()
            df_grouped = training_data.groupby(['Part', pd.Grouper(key='Month', freq='M')])['Sales'].sum().reset_index()
            
            st.write(f"Processing {df_grouped['Part'].nunique()} unique parts...")
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create date ranges for historical and forecast periods
            historical_months = pd.date_range(start='2022-01-01', end='2023-12-31', freq='MS')
            forecast_months = pd.date_range(start='2024-01-01', end='2025-12-31', freq='MS')
            
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
                    row_data[month_key] = historical_sales.iloc[0] if len(historical_sales) > 0 else ''
                
                # Check if we have enough data for forecasting (at least 12 months)
                if part_data['Month'].nunique() < 12:
                    # Add forecast columns with "Not enough data" message
                    for month in forecast_months:
                        month_key = month.strftime('%b-%Y')
                        if month_key not in row_data:  # Avoid overwriting historical data
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
