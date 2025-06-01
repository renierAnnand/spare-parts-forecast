import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from io import BytesIO

st.title("🔮 Spare Parts Sales Forecast (Meta Prophet - 12 Months)")

uploaded_file = st.file_uploader("Upload Excel file with columns: Part, Month, Sales", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if {'Part', 'Month', 'Sales'}.issubset(df.columns):
        df['Month'] = pd.to_datetime(df['Month'])
        all_results = []

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

                forecast_trimmed = forecast[['ds', 'yhat']].rename(columns={'ds': 'Month', 'yhat': 'Forecast'})
                forecast_trimmed['Part'] = part

                # Merge actual and forecast data for comparison
                actual_data = part_data.rename(columns={'ds': 'Month', 'y': 'Sales'})
                merged = pd.merge(forecast_trimmed, actual_data, on=['Part', 'Month'], how='left')

                all_results.append(merged)

            except Exception as e:
                st.error(f"Could not process {part}: {e}")

        if all_results:
            final_result = pd.concat(all_results)

            # Prepare Excel in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_result.to_excel(writer, sheet_name='Forecast vs Actual', index=False)

            st.download_button(
                label="📥 Download Excel: Forecast vs Actual",
                data=output.getvalue(),
                file_name="spare_parts_forecast_vs_actual.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.error("Excel must contain: Part, Month, Sales")
