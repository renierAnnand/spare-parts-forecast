import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("🔮 Spare Parts Sales Forecast (Meta Prophet - 12 Months)")

uploaded_file = st.file_uploader("Upload Excel file with columns: Part, Month, Sales", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if {'Part', 'Month', 'Sales'}.issubset(df.columns):
        df['Month'] = pd.to_datetime(df['Month'])
        all_forecasts = []

        for part in df['Part'].unique():
            st.subheader(f"📦 {part}")
            part_data = df[df['Part'] == part][['Month', 'Sales']].rename(columns={'Month': 'ds', 'Sales': 'y'})

            try:
                model = Prophet()
                model.fit(part_data)

                future = model.make_future_dataframe(periods=12, freq='M')
                forecast = model.predict(future)

                fig1 = model.plot(forecast)
                st.pyplot(fig1)

                forecast_trimmed = forecast[['ds', 'yhat']].tail(12)
                forecast_trimmed['Part'] = part
                all_forecasts.append(forecast_trimmed.rename(columns={'ds': 'Month', 'yhat': 'Forecast'}))

            except Exception as e:
                st.error(f"Could not process {part}: {e}")

        if all_forecasts:
            result = pd.concat(all_forecasts)
            st.download_button("📥 Download Forecast CSV", result.to_csv(index=False), file_name="12_month_forecast.csv")

    else:
        st.error("Excel must contain: Part, Month, Sales")

