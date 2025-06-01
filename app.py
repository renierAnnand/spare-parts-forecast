import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
import io

st.title("📈 Spare Parts Forecast vs Actual (May 2025 – April 2026)")

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
            
            # Filter for Jan 2022 to April 2025 only
            df = df[(df['Month'] >= '2022-01-01') & (df['Month'] <= '2025-04-30')]
            df = df.dropna()
            df = df.drop_duplicates()
            
            # Group data by part and month
            df_grouped = df.groupby(['Part', pd.Grouper(key='Month', freq='M')])['Sales'].sum().reset_index()
            
            st.write(f"Processing {df_grouped['Part'].nunique()} unique parts...")
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_results = []
            total_parts = df_grouped['Part'].nunique()
            
            for i, part in enumerate(df_grouped['Part'].unique()):
                progress_bar.progress((i + 1) / total_parts)
                status_text.text(f'Processing part {i + 1}/{total_parts}: {part}')
                
                part_data = df_grouped[df_grouped['Part'] == part][['Month', 'Sales']].dropna()
                
                # First, add all historical data (Jan 2022 - April 2025)
                for _, row in part_data.iterrows():
                    all_results.append({
                        'Part': part,
                        'Month': row['Month'],
                        'Actual_Sales': row['Sales'],
                        'Predicted_Sales': np.nan,
                        'Type': 'Historical'
                    })
                
                # Check if we have enough data for forecasting (at least 24 months)
                if part_data['Month'].nunique() < 24:
                    for month in pd.date_range(start='2025-05-01', end='2026-04-30', freq='MS'):
                        all_results.append({
                            'Part': part,
                            'Month': month,
                            'Actual_Sales': np.nan,
                            'Predicted_Sales': 'Not enough data (need 24+ months)',
                            'Type': 'Forecast'
                        })
                    continue
                
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
                    
                    # Make future predictions
                    future = model.make_future_dataframe(periods=12, freq='M')
                    forecast = model.predict(future)
                    
                    # Extract forecast for May 2025 - April 2026
                    forecast_range = forecast[(forecast['ds'] >= '2025-05-01') & (forecast['ds'] <= '2026-04-30')]
                    
                    # Add future predictions
                    for _, row in forecast_range.iterrows():
                        all_results.append({
                            'Part': part,
                            'Month': row['ds'],
                            'Actual_Sales': np.nan,
                            'Predicted_Sales': round(max(0, row['yhat']), 2),  # Ensure non-negative forecasts
                            'Type': 'Forecast'
                        })
                        
                except Exception as e:
                    st.warning(f"Error processing part '{part}': {str(e)}")
                    for month in pd.date_range(start='2025-05-01', end='2026-04-30', freq='MS'):
                        all_results.append({
                            'Part': part,
                            'Month': month,
                            'Actual_Sales': np.nan,
                            'Predicted_Sales': f'Error: {str(e)}',
                            'Type': 'Forecast'
                        })
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Create results dataframe
            result_df = pd.DataFrame(all_results)
            
            # Sort by Part and Month for better organization
            result_df = result_df.sort_values(['Part', 'Month']).reset_index(drop=True)
            
            # Display summary
            st.success("Forecasting completed!")
            historical_count = len(result_df[result_df['Type'] == 'Historical'])
            forecast_count = len(result_df[result_df['Type'] == 'Forecast'])
            st.write(f"Generated {historical_count} historical records and {forecast_count} forecast records")
            
            # Show summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Historical Data Summary:**")
                historical_data = result_df[result_df['Type'] == 'Historical']
                if not historical_data.empty:
                    st.write(f"- Date range: {historical_data['Month'].min().strftime('%Y-%m')} to {historical_data['Month'].max().strftime('%Y-%m')}")
                    st.write(f"- Total records: {len(historical_data)}")
            
            with col2:
                st.write("**Forecast Data Summary:**")
                forecast_data = result_df[result_df['Type'] == 'Forecast']
                if not forecast_data.empty:
                    st.write(f"- Date range: {forecast_data['Month'].min().strftime('%Y-%m')} to {forecast_data['Month'].max().strftime('%Y-%m')}")
                    st.write(f"- Total records: {len(forecast_data)}")
            
            # Show preview of results
            st.write("**Data preview:**")
            st.dataframe(result_df.head(15))
            
            # Create Excel file for download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, sheet_name='Sales Data & Forecasts', index=False)
                
                # Add some formatting
                workbook = writer.book
                worksheet = writer.sheets['Sales Data & Forecasts']
                
                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                historical_format = workbook.add_format({
                    'fg_color': '#E8F4FD',
                    'border': 1
                })
                
                forecast_format = workbook.add_format({
                    'fg_color': '#FFF2CC',
                    'border': 1
                })
                
                # Format headers
                for col_num, value in enumerate(result_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Apply conditional formatting based on Type
                for row_num, row_data in enumerate(result_df.itertuples(), 1):
                    if row_data.Type == 'Historical':
                        for col_num in range(len(result_df.columns)):
                            worksheet.write(row_num, col_num, 
                                          getattr(row_data, result_df.columns[col_num], ''), 
                                          historical_format)
                    elif row_data.Type == 'Forecast':
                        for col_num in range(len(result_df.columns)):
                            worksheet.write(row_num, col_num, 
                                          getattr(row_data, result_df.columns[col_num], ''), 
                                          forecast_format)
                
                # Auto-adjust column widths
                for i, col in enumerate(result_df.columns):
                    max_length = max(
                        result_df[col].astype(str).map(len).max(),
                        len(str(col))
                    ) + 2
                    worksheet.set_column(i, i, max_length)
                
                # Add a legend
                legend_row = len(result_df) + 3
                worksheet.write(legend_row, 0, "Legend:", header_format)
                worksheet.write(legend_row + 1, 0, "Historical Data", historical_format)
                worksheet.write(legend_row + 2, 0, "Forecast Data", forecast_format)
            
            output.seek(0)
            
            st.download_button(
                label="📥 Download Excel File (Historical + Forecasts)",
                data=output,
                file_name="Sales_Historical_and_Forecasts_2022_2026.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please make sure your Excel file has the columns: Part, Month, Sales")
