import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
import io

st.title("📈 Spare Parts Forecast vs Actual (May 2025 – April 2026)")

uploaded_file = st.file_uploader("Upload Excel file with: Part, Month, Sales", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Month'] = pd.to_datetime(df['Month'])

    # Filter for Jan 2022 to April 2025 only
    df = df[(df['Month'] >= '2022-01-01') & (df['Month'] <= '2025-04-30')]
    df = df.dropna()
    df = df.drop_duplicates()

    df_grouped = df.groupby(['Part', pd.Grouper(key='Month', freq='M')])['Sales'].sum().reset_index()

    all_results = []

    for part in df_grouped['Part'].unique():
        part_data = df_grouped[df_grouped['Part'] == part][['Month', 'Sales']].dropna()

        if part_data['Month'].nunique() < 24:
            for month in pd.date_range(start='2025-05-01', end='2026-04-30', freq='MS'):
                all_results.append({
                    'Part': part,
                    'Month': month,
                    'Sales': np.nan,
                    'Forecast': 'Not enough data'
                })
            continue

        try:
            model_df = part_data.rename(columns={'Month': 'ds', 'Sales': 'y'})
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            model.fit(model_df)

            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)
            forecast_range = forecast[(forecast['ds'] >= '2025-05-01') & (forecast['ds'] <= '2026-04-30')]

            for _, row in forecast_range.iterrows():
                all_results.append({
                    'Part': part,
                    'Month': row['ds'],
                    'Sales': np.nan,
                    'Forecast': round(row['yhat'], 2)
                })
        except Exception as e:
            for month in pd.date_range(start='2025-05-01', end='2026-04-30', freq='MS'):
                all_results.append({
                    'Part': part,
                    'Month': month,
                    'Sales': np.nan,
                    'Forecast': f'Error: {e}'
                })

    result_df = pd.DataFrame(all_results)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, sheet_name='Forecast vs Actual', index=False)
    output.seek(0)

    st.download_button(
        label="📥 Download Excel File",
        data=output,
        file_name="Actual_vs_Forecast_2025_2026.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
