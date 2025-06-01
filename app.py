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
                
                # Check if we have enough data (at least 24 months)
                if part_data['Month'].nunique() < 24:
                    for month in pd.date_range(start='2025-05-01', end='2026-04-30', freq='MS'):
                        all_results.append({
                            'Part': part,
                            'Month': month,
                            'Sales': np.nan,
                            'Forecast': 'Not enough data (need 24+ months)'
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
                    
                    # FIXED: Corrected the syntax error here
                    for _, row in forecast_range.iterrows():
                        all_results.append({
                            'Part': part,
                            'Month': row['ds'],
                            'Sales': np.nan,
                            'Forecast': round(max(0, row['yhat']), 2)  # Ensure non-negative forecasts
                        })
                        
                except Exception as e:
                    st.warning(f"Error processing part '{part}': {str(e)}")
                    for month in pd.date_range(start='2025-05-01', end='2026-04-30', freq='MS'):
                        all_results.append({
                            'Part': part,
                            'Month': month,
                            'Sales': np.nan,
                            'Forecast': f'Error: {str(e)}'
                        })
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Create results dataframe
            result_df = pd.DataFrame(all_results)
            
            # Display summary
            st.success("Forecasting completed!")
            st.write(f"Generated forecasts for {len(result_df)} part-month combinations")
            
            # Show preview of results
            st.write("Forecast preview:")
            st.dataframe(result_df.head(10))
            
            # Create Excel file for download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, sheet_name='Forecast vs Actual', index=False)
                
                # Add some formatting
                workbook = writer.book
                worksheet = writer.sheets['Forecast vs Actual']
                
                # Format headers
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                for col_num, value in enumerate(result_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Auto-adjust column widths
                for i, col in enumerate(result_df.columns):
                    max_length = max(
                        result_df[col].astype(str).map(len).max(),
                        len(str(col))
                    ) + 2
                    worksheet.set_column(i, i, max_length)
            
            output.seek(0)
            
            st.download_button(
                label="📥 Download Excel File",
                data=output,
                file_name="Actual_vs_Forecast_2025_2026.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please make sure your Excel file has the columns: Part, Month, Sales")
