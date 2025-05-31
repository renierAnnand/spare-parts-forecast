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

years = st.sidebar.text_input("Enter years to upload (comma-separated, e.g., 2022,2023,2024)", "2022,2023,2024")
try:
    year_list = [int(y.strip()) for y in years.split(",") if y.strip().isdigit()]
except:
    year_list = [2022, 2023, 2024]

uploaded_files = {}
for year in year_list:
    uploaded_files[year] = st.sidebar.file_uploader(f"Upload File for {year}", type=['xlsx', 'csv'], key=str(year))

for year, file in uploaded_files.items():
    if file:
        st.success(f"File Loaded for {year}")
        df = predictor.load_data(file, year)
        st.dataframe(df.head(), use_container_width=True)

# Train selected years
st.sidebar.markdown("---")
selected_train_years = st.sidebar.multiselect("Select years to train models", options=year_list, default=year_list)

for year in selected_train_years:
    if year in predictor.training_data:
        if st.sidebar.button(f"🚀 Train Model for {year}"):
            predictor.train_models(year)
            st.sidebar.success(f"Model trained for {year}!")
            st.subheader(f"📈 Model Training Summary ({year})")
            st.write("Total Parts Trained:", len(predictor.models[year]['item_codes']))

# Prediction & comparison section
st.header("📊 Predict & Compare Across Years")
prediction_year = st.number_input("Enter Prediction Year (e.g., 2025)", min_value=2000, max_value=2100, value=2025)

if st.button("🔮 Run Predictions and Compare"):
    for train_year in selected_train_years:
        if train_year in predictor.models:
            st.subheader(f"➡️ Prediction using model trained on {train_year}")
            predictions = predictor.predict_next_year(train_year, prediction_year)
            if prediction_year in predictor.training_data:
                comparison_df, metrics = predictor.compare_predictions(prediction_year, prediction_year)
                if comparison_df is not None:
                    st.markdown(f"**Comparison for model trained on {train_year} → predicting {prediction_year}:**")
                    st.dataframe(comparison_df.head(), use_container_width=True)
                    for model, metric in metrics.items():
                        st.markdown(f"**{model}**")
                        st.write({k: round(v, 2) for k, v in metric.items()})
                    predictor.visualize_comparison(comparison_df, metrics, prediction_year)

st.sidebar.markdown("---")
st.sidebar.info("Developed by Digital CoE - ALKHORAYEF")
