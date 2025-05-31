import streamlit as st
import pandas as pd
from predictor import SparePartsPredictor

st.set_page_config(page_title="Spare Parts Forecasting", layout="wide")
st.title("🔧 Spare Parts Sales Forecasting App")

# Initialize predictor
if 'predictor' not in st.session_state:
    st.session_state.predictor = SparePartsPredictor()

predictor = st.session_state.predictor

st.sidebar.header("📂 Upload Yearly Files")

uploaded_files = {}
for year in [2022, 2023, 2024]:
    uploaded_files[year] = st.sidebar.file_uploader(f"Upload File for {year}", type=['xlsx', 'csv'], key=str(year))

for year, file in uploaded_files.items():
    if file:
        st.success(f"File Loaded for {year}")
        df = predictor.load_data(file, year)
        st.dataframe(df.head(), use_container_width=True)

        if st.button(f"🚀 Train Model for {year}"):
            predictor.train_models(year)
            st.success(f"Model trained for {year}!")
            st.subheader(f"📈 Model Training Summary ({year})")
            st.write("Total Parts Trained:", len(predictor.models[year]['item_codes']))

comparison_results = []

st.header("📊 Predict & Compare Multiple Years")
prediction_year = st.number_input("Prediction Year (e.g., 2025)", min_value=2000, max_value=2100, value=2025)

if st.button("🔮 Predict and Compare"):
    for year in [2022, 2023, 2024]:
        if year in predictor.models:
            st.subheader(f"➡️ Prediction based on {year}")
            predictions = predictor.predict_next_year(year, prediction_year)
            if prediction_year in predictor.training_data:
                comparison_df, metrics = predictor.compare_predictions(prediction_year, prediction_year)
                if comparison_df is not None:
                    st.markdown(f"**Comparison for {year} → {prediction_year}:**")
                    st.dataframe(comparison_df.head(), use_container_width=True)
                    for model, metric in metrics.items():
                        st.markdown(f"**{model}**")
                        st.write({k: round(v, 2) for k, v in metric.items()})
                    predictor.visualize_comparison(comparison_df, metrics, prediction_year)

st.sidebar.markdown("---")
st.sidebar.info("Developed by Digital CoE - ALKHORAYEF")
