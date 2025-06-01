import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Spare Parts Forecast", layout="wide")
st.title("🔮 Spare Parts Sales Forecast (Meta Prophet - 12 Months)")

uploaded_file = st.file_uploader("📤 Upload Excel file with columns: Part, Month, Sales", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if {'Part', 'Month', 'Sales'}.issubset(df.columns):
        df['Month'] = pd.to_datetime(df['Month'])
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        
        # Ensure monthly frequency
        df_grouped = df.groupby(['Part', pd.Grouper(key='Month', freq='M')])['Sales'].sum().reset_index()

        all_results = []
        all_errors = []

        for part in df_grouped['Part'].unique():
            st.subheader(f"📦 Forecast for: {part}")
            part_data = df_grouped[df_grouped['Part'] == part][['Month', 'Sales']]
            part_data = part_data.rename(columns={'Month': 'ds', 'Sales': 'y'})

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

                    fig1 = model.plot(forecast)
                    st.pyplot(fig1)

                    forecast_trimmed = forecast[['ds', 'yhat']].rename(columns={'ds': 'Month', 'yhat': 'Forecast'})
                    actuals = part_data.rename(columns={'ds': 'Month', 'y': 'Sales'})
                    comparison = pd.merge(actuals, forecast_trimmed, on='Month', how='outer')
                    comparison['Part'] = part
                    all_results.append(comparison)

                    # Evaluate where both actual and forecast exist
                    eval_data = comparison.dropna(subset=['Sales', 'Forecast'])
                    mae = mean_absolute_error(eval_data['Sales'], eval_data['Forecast'])
                    rmse = mean_squared_error(eval_data['Sales'], eval_data['Forecast'], squared=False)
                    all_errors.append({'Part': part, 'MAE': mae, 'RMSE': rmse})

                    st.write(f"📊 MAE: {mae:.2f}, RMSE: {rmse:.2f}")

                except Exception as e:
                    st.error(f"Could not process {part}: {e}")
            else:
                st.warning(f"⚠️ Not enough data to forecast for {part} (at least 24 months needed)")

        if all_results:
            result_df = pd.concat(all_results)
            error_df = pd.DataFrame(all_errors)

            # Save Excel to memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, sheet_name='Forecast_vs_Actual', index=False)
                error_df.to_excel(writer, sheet_name='Error_Metrics', index=False)
            output.seek(0)

            st.download_button("📥 Download Forecast Excel",
                               data=output,
                               file_name="SpareParts_Forecast_Report.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.error("❌ Excel must contain columns: Part, Month, Sales")
