import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Spare Parts Forecast", layout="wide")
st.title("🔮 Spare Parts Sales Forecast (Meta Prophet - May 2025 to April 2026)")

uploaded_file = st.file_uploader("📤 Upload Excel file with columns: Part, Month, Sales", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if {'Part', 'Month', 'Sales'}.issubset(df.columns):
        df['Month'] = pd.to_datetime(df['Month'])
        df = df[(df['Month'] >= '2022-01-01') & (df['Month'] <= '2025-04-30')]
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)

        df_grouped = df.groupby(['Part', pd.Grouper(key='Month', freq='M')])['Sales'].sum().reset_index()

        all_results = []
        all_errors = []

        for part in df_grouped['Part'].unique():
            st.subheader(f"📦 Forecast for: {part}")
            part_data = df_grouped[df_grouped['Part'] == part][['Month', 'Sales']]
            part_data = part_data.rename(columns={'Month': 'ds', 'y': 'y'})

            if len(part_data) >= 24:
                try:
                    model = Prophet(
                        changepoint_prior_scale=0.05,
                        seasonality_mode='multiplicative',
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False
                    )
                    model.fit(part_data)

                    future = model.make_future_dataframe(periods=12, freq='M')
                    forecast = model.predict(future)

                    # Select forecast only from May 2025 to April 2026
                    forecast_range = forecast[(forecast['ds'] >= '2025-05-01') & (forecast['ds'] <= '2026-04-30')]
                    forecast_trimmed = forecast_range[['ds', 'yhat']].rename(columns={'ds': 'Month', 'yhat': 'Forecast'})
                    forecast_trimmed['Part'] = part
                    all_results.append(forecast_trimmed)

                    # Plot full forecast (optional)
                    fig1 = model.plot(forecast)
                    st.pyplot(fig1)

                except Exception as e:
                    st.error(f"Could not process {part}: {e}")
            else:
                st.warning(f"⚠️ Not enough data to forecast for {part} (at least 24 months needed)")

        if all_results:
            result_df = pd.concat(all_results)

            # Save Excel to memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, sheet_name='Forecast_May25_Apr26', index=False)
            output.seek(0)

            st.download_button("📥 Download Forecast Excel",
                               data=output,
                               file_name="Forecast_May2025_Apr2026.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.error("❌ Excel must contain columns: Part, Month, Sales")
