import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("🔮 Spare Parts Forecasting with Prophet")

st.subheader("Step 1: Upload 2022 (Training) and 2023 (Actual) Excel Files")
file_2022 = st.file_uploader("Upload 2022 Excel file", type="xlsx", key="2022")
file_2023 = st.file_uploader("Upload 2023 Excel file", type="xlsx", key="2023")

def transform_excel(file, year):
    df = pd.read_excel(file)
    df = df.drop(index=0)
    df_long = df.melt(id_vars='MonthYear', var_name='Month', value_name='Quantity')
    df_long.columns = ['Item Code', 'Month', 'Quantity']
    df_long['Date'] = pd.to_datetime(df_long['Month'], format='%b-%Y')
    df_long['Quantity'] = pd.to_numeric(df_long['Quantity'], errors='coerce')
    df_long.dropna(subset=['Quantity'], inplace=True)
    df_long = df_long[df_long['Date'].dt.year == year]
    return df_long

if file_2022:
    df_2022 = transform_excel(file_2022, 2022)
    item_codes = df_2022['Item Code'].unique()
    selected_item = st.selectbox("Select Item Code to Forecast", item_codes)

    df_selected = df_2022[df_2022['Item Code'] == selected_item][['Date', 'Quantity']]
    df_selected = df_selected.rename(columns={'Date': 'ds', 'Quantity': 'y'})

    st.subheader(f"Prophet Forecast for {selected_item}")
    model = Prophet()
    model.fit(df_selected)

    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    if file_2023:
        df_2023 = transform_excel(file_2023, 2023)
        df_actual = df_2023[df_2023['Item Code'] == selected_item][['Date', 'Quantity']]
        df_actual.columns = ['ds', 'y_actual']

        df_merge = pd.merge(forecast[['ds', 'yhat']], df_actual, on='ds', how='left')
        df_merge = df_merge.dropna()

        st.subheader("📊 Forecast vs Actual (2023)")
        st.line_chart(df_merge.set_index('ds'))

        df_merge['error'] = abs(df_merge['yhat'] - df_merge['y_actual'])
        df_merge['percent_error'] = df_merge['error'] / df_merge['y_actual'] * 100

        mae = df_merge['error'].mean()
        mape = df_merge['percent_error'].mean()

        st.markdown(f"**Mean Absolute Error (MAE)**: {mae:.2f}")
        st.markdown(f"**Mean Absolute Percentage Error (MAPE)**: {mape:.2f}%")
