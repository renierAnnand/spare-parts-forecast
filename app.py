import streamlit as st
import pandas as pd
from predictor import SparePartsPredictor

st.set_page_config(page_title="Spare Parts Forecasting", layout="wide")
st.title("🔧 Spare Parts Sales Forecasting App")

# Initialize predictor
if 'predictor' not in st.session_state:
    st.session_state.predictor = SparePartsPredictor()

predictor = st.session_state.predictor

st.sidebar.header("📂 Data Upload")

uploaded_file_base = st.sidebar.file_uploader("Upload Base Year File", type=['xlsx', 'csv'], key='base')
uploaded_file_actual = st.sidebar.file_uploader("Upload Actual Year File", type=['xlsx', 'csv'], key='actual')

base_year = st.sidebar.number_input("Base Year (e.g., 2022)", min_value=2000, max_value=2100, value=2022)
prediction_year = st.sidebar.number_input("Prediction Year (e.g., 2023)", min_value=2000, max_value=2100, value=2023)

if uploaded_file_base:
    st.success(f"Base Year File Loaded for {base_year}")
    df_base = predictor.load_data(uploaded_file_base, base_year)
    st.dataframe(df_base.head(), use_container_width=True)

    if st.button("🚀 Train Model"):
        predictor.train_models(base_year)
        st.success("Model trained successfully!")

        st.subheader("📈 Model Training Summary")
        st.write("Total Parts Trained:", len(predictor.models[base_year]['item_codes']))

if st.button("🔮 Predict Sales"):
    predictions = predictor.predict_next_year(base_year, prediction_year)
    if predictions is not None:
        st.subheader("📊 Predictions")
        st.dataframe(predictions.head(20), use_container_width=True)

if uploaded_file_actual:
    st.success(f"Actual File Loaded for {prediction_year}")
    predictor.load_data(uploaded_file_actual, prediction_year)

    if st.button("📊 Compare with Actual"):
        comparison_df, metrics = predictor.compare_predictions(prediction_year, prediction_year)

        if comparison_df is not None:
            st.subheader("✅ Comparison Table")
            st.dataframe(comparison_df.head(20), use_container_width=True)

            st.subheader("📉 Model Metrics")
            for model, metric in metrics.items():
                st.markdown(f"**{model}**")
                st.write({k: round(v, 2) for k, v in metric.items()})

            st.subheader("📊 Visual Comparison")
            predictor.visualize_comparison(comparison_df, metrics, prediction_year)

st.sidebar.markdown("---")
st.sidebar.info("Developed by Digital CoE - ALKHORAYEF")
