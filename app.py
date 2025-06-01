import streamlit as st
import pandas as pd
import numpy as np
import io
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

st.set_page_config(layout="wide")
st.title("📈 Spare Parts Forecast (SARIMA)")

uploaded_file = st.file_uploader(
    "Upload Excel file with columns: Part, Month, Sales", type=["xlsx"]
)

if uploaded_file:
    try:
        # 1) Load and show basic info
        df = pd.read_excel(uploaded_file)
        st.write(f"Data shape: {df.shape}")
        st.write("Data preview:")
        st.dataframe(df.head())

        # 2) Verify required columns
        required_cols = {"Part", "Month", "Sales"}
        if not required_cols.issubset(df.columns):
            st.error(
                f"Please ensure your file has columns: Part, Month, Sales. Found: {list(df.columns)}"
            )
            st.stop()

        # 3) Preprocess input
        df = df.dropna(subset=["Part", "Month", "Sales"])
        df["Part"] = df["Part"].astype(str).str.strip()

        try:
            df["Month"] = pd.to_datetime(df["Month"])
        except:
            st.warning("Attempting alternative date parsing...")
            df["Month"] = pd.to_datetime(df["Month"], errors="coerce")

        date_parse_failed = df["Month"].isna().sum()
        if date_parse_failed > 0:
            st.warning(f"⚠️ Removed {date_parse_failed} rows with invalid dates")
            df = df.dropna(subset=["Month"])

        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0).astype(float)

        # Sales data stats
        st.write("### Sales Data Statistics:")
        st.write(f"- Total rows after cleaning: {len(df)}")
        st.write(f"- Non-zero sales rows: {(df['Sales'] > 0).sum()}")
        st.write(f"- Average sales: {df['Sales'].mean():.2f}")
        st.write(f"- Max sales: {df['Sales'].max()}")
        st.write(f"- Parts with any sales: {df[df['Sales'] > 0]['Part'].nunique()}")

        top_parts = df.groupby("Part")["Sales"].sum().sort_values(ascending=False).head(10)
        if len(top_parts) > 0:
            st.write("### Top 10 Parts by Total Sales:")
            st.dataframe(top_parts)

        st.write("### Date Range in Original Data:")
        date_counts = (
            df.groupby(df["Month"].dt.to_period("M"))["Sales"]
            .agg(["count", "sum"])
            .reset_index()
        )
        date_counts["Month"] = date_counts["Month"].dt.to_timestamp()
        st.dataframe(date_counts.set_index("Month"))

        # 4) Aggregate duplicate (Part, Month) pairs
        df["Month"] = pd.to_datetime(df["Month"]).dt.to_period("M").dt.to_timestamp()
        df_grouped = df.groupby(["Part", "Month"])["Sales"].sum().reset_index()

        # 5) Compute overall date range
        min_date = df_grouped["Month"].min()
        max_date = df_grouped["Month"].max()

        if pd.isna(min_date) or pd.isna(max_date):
            st.error(
                "❌ No valid dates found in the data. Please check your Month column format."
            )
            st.stop()

        data_span_months = (
            (max_date.year - min_date.year) * 12
            + (max_date.month - min_date.month)
            + 1
        )
        data_span_years = data_span_months / 12

        st.write(f"📅 Historical data covers: {min_date.strftime('%Y-%m')} → {max_date.strftime('%Y-%m')}")
        st.write(f"📦 Total unique parts: {df_grouped['Part'].nunique()}")
        st.write(f"📊 Total aggregated records: {len(df_grouped)}")
        st.write(f"📊 Data span: {data_span_months} months ({data_span_years:.1f} years)")

        if data_span_months < 12:
            st.warning("⚠️ Fewer than 12 months of history total – forecasts may be unreliable for many parts.")
        elif data_span_months < 24:
            st.info("ℹ️ Between 12 and 24 months of data – consider adding more history if possible.")
        else:
            st.success("✅ Good historical coverage (≥24 months).")

        # 6) Define monthly indices
        min_date = min_date.replace(day=1)
        max_date = max_date.replace(day=1)
        historical_months = pd.date_range(start=min_date, end=max_date, freq="MS")
        forecast_start = max_date + pd.DateOffset(months=1)
        forecast_end = forecast_start + pd.DateOffset(months=11)
        forecast_months = pd.date_range(start=forecast_start, end=forecast_end, freq="MS")

        st.write(f"📅 Historical months range: {historical_months[0].strftime('%Y-%m')} to {historical_months[-1].strftime('%Y-%m')} ({len(historical_months)} months)")
        st.success(f"🔮 Prediction period: {forecast_start.strftime('%Y-%m')} → {forecast_end.strftime('%Y-%m')}")

        # 7) Prepare progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 8) Loop over each unique part
        results_data = []
        parts_list = sorted(df_grouped["Part"].unique())
        total_parts = len(parts_list)

        parts_with_sarima = 0
        parts_with_naive = 0
        parts_failed = 0

        for idx, part in enumerate(parts_list):
            progress_bar.progress((idx + 1) / total_parts)
            status_text.text(f"Processing part {idx + 1}/{total_parts}: {part}")

            try:
                raw = (
                    df_grouped[df_grouped["Part"] == part]
                    .loc[:, ["Month", "Sales"]]
                    .set_index("Month")
                    .sort_index()
                )

                if len(raw) == 0:
                    st.warning(f"No data found for part: {part}")
                    parts_failed += 1
                    continue

                part_index = pd.date_range(start=raw.index.min(), end=max_date, freq="MS")
                part_train = raw.reindex(part_index, fill_value=0).reset_index().rename(
                    columns={"index": "ds", "Sales": "y"}
                )

                part_full_hist = raw.reindex(historical_months, fill_value=0).reset_index().rename(
                    columns={"index": "Month", "Sales": "Sales"}
                )

                row_data = {"Item Code": part}
                for hist_month in historical_months:
                    sales_val = part_full_hist.loc[
                        part_full_hist["Month"] == hist_month, "Sales"
                    ]
                    row_data[hist_month.strftime("%b-%Y")] = int(sales_val.iloc[0]) if len(sales_val) > 0 else 0

                # 8.a) If too few data points (< 8 months) or too few nonzeros, use naive 3-month average
                non_zero_count = (part_train["y"] > 0).sum()
                if part_train["ds"].nunique() < 8 or non_zero_count < 2:
                    last_three = part_train.sort_values("ds").tail(3)["y"]
                    naive_forecast = int(round(last_three.mean(), 0)) if len(last_three) > 0 else 0
                    for fc_month in forecast_months:
                        row_data[fc_month.strftime("%b-%Y")] = naive_forecast

                    row_data["Method"] = "Naive"
                    results_data.append(row_data)
                    parts_with_naive += 1
                    continue

                # 8.b) Split last 12 months for in-sample validation (if ≥ 24 months total)
                if len(part_train) > 24:
                    cutoff_date = max_date - pd.DateOffset(months=12)
                    train_df = part_train[part_train["ds"] <= cutoff_date].copy()
                    valid_df = part_train[part_train["ds"] > cutoff_date].copy()
                else:
                    train_df = part_train.copy()
                    valid_df = pd.DataFrame(columns=part_train.columns)

                # 8.c) If we have a validation window, use auto_arima on train_df and compute CV MAPE
                best_order = None
                best_seasonal_order = None
                cv_mape = np.nan

                if not valid_df.empty and len(train_df) >= 12:
                    try:
                        # Fit auto_arima on train_df to select orders
                        arima_model = auto_arima(
                            train_df["y"],
                            start_p=0,
                            start_q=0,
                            max_p=2,
                            max_q=2,
                            seasonal=True,
                            m=12,
                            start_P=0,
                            start_Q=0,
                            max_P=1,
                            max_Q=1,
                            d=1,
                            D=1,
                            trace=False,
                            error_action="ignore",
                            suppress_warnings=True,
                            stepwise=True,
                        )

                        best_order = arima_model.order
                        best_seasonal_order = arima_model.seasonal_order

                        # Forecast valid window
                        n_valid = len(valid_df)
                        preds_valid = arima_model.predict(n_periods=n_valid)
                        actual_valid = valid_df.sort_values("ds")["y"].values

                        mask = actual_valid != 0
                        if mask.any():
                            cv_mape = (
                                np.mean(np.abs((actual_valid[mask] - preds_valid[mask]) / actual_valid[mask]))
                                * 100
                            )
                        else:
                            cv_mape = np.nan

                    except Exception as e:
                        best_order = None
                        best_seasonal_order = None
                        cv_mape = np.nan

                row_data["CV_MAPE"] = round(cv_mape, 2) if not np.isnan(cv_mape) else np.nan
                if best_order is not None:
                    row_data["Chosen_order"] = f"{best_order}"
                    row_data["Chosen_seasonal_order"] = f"{best_seasonal_order}"
                else:
                    row_data["Chosen_order"] = np.nan
                    row_data["Chosen_seasonal_order"] = np.nan

                # 8.d) Fit final SARIMA on all available history
                try:
                    if best_order is not None:
                        sarima_order = best_order
                        seasonal_order = best_seasonal_order
                        final_model = SARIMAX(
                            part_train["y"],
                            order=sarima_order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
                    else:
                        # Fallback to a default SARIMA (1,1,1)x(1,1,1,12)
                        final_model = SARIMAX(
                            part_train["y"],
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)

                    # Forecast next 12 months
                    forecast_res = final_model.get_forecast(steps=12)
                    preds = forecast_res.predicted_mean.round().astype(int)

                    for i, fc_month in enumerate(forecast_months):
                        yhat = preds[i]
                        row_data[fc_month.strftime("%b-%Y")] = max(0, int(yhat))

                    row_data["Method"] = "SARIMA"
                    results_data.append(row_data)
                    parts_with_sarima += 1

                except Exception as e:
                    # Fallback to naive if SARIMA fails
                    st.warning(f"SARIMA failed for part {part}: {str(e)}. Using naive forecast.")
                    last_three = part_train.sort_values("ds").tail(3)["y"]
                    naive_forecast = int(round(last_three.mean(), 0)) if len(last_three) > 0 else 0
                    for fc_month in forecast_months:
                        row_data[fc_month.strftime("%b-%Y")] = naive_forecast
                    row_data["Method"] = "Naive (SARIMA failed)"
                    results_data.append(row_data)
                    parts_with_naive += 1

            except Exception as e:
                st.error(f"Error processing part {part}: {str(e)}")
                parts_failed += 1
                continue

        progress_bar.empty()
        status_text.empty()

        # 9) Create final DataFrame
        result_df = pd.DataFrame(results_data)

        # Show processing summary
        st.write("### 📊 Processing Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Parts", total_parts)
        with col2:
            st.metric("SARIMA Models", parts_with_sarima)
        with col3:
            st.metric("Naive Forecasts", parts_with_naive)
        with col4:
            st.metric("Failed", parts_failed)

        if "CV_MAPE" in result_df.columns:
            cv_table = (
                result_df[["Item Code", "CV_MAPE", "Method"]]
                .dropna(subset=["CV_MAPE"])
                .sort_values(by="CV_MAPE", ascending=False)
            )
            if len(cv_table) > 0:
                st.write("### 🧪 In-Sample Validation (CV MAPE) by Part")
                st.dataframe(cv_table.head(15))

        # 10) Reorder columns
        month_columns = [m.strftime("%b-%Y") for m in list(historical_months) + list(forecast_months)]
        ordered_cols = ["Item Code"] + [c for c in month_columns if c in result_df.columns]
        final_cols = ordered_cols + [
            c for c in ["CV_MAPE", "Chosen_order", "Chosen_seasonal_order", "Method"] if c in result_df.columns
        ]
        result_df = result_df[final_cols]

        st.success("✅ Forecasting completed!")
        st.write(f"🔢 Number of parts processed: {len(result_df)}")

        # 11) Show a quick preview
        st.write("**Forecast sample (first 10 parts):**")
        preview_cols = ["Item Code"] + month_columns[-3:] + forecast_months[:3].strftime("%b-%Y").tolist() + ["Method"]
        preview_cols = [c for c in preview_cols if c in result_df.columns]
        st.dataframe(result_df[preview_cols].head(10))

        # 12) Prepare Excel download
        excel_df = result_df.loc[:, ["Item Code"] + month_columns].fillna("")

        header_row_1 = ["Item Code"]
        header_row_2 = [""]
        for col in month_columns:
            header_row_1.append(col)
            dt = pd.to_datetime(col, format="%b-%Y", errors="coerce")
            if pd.isna(dt) or dt <= max_date:
                header_row_2.append("Historical QTY")
            else:
                header_row_2.append("Forecasted QTY")

        restructured_data = [header_row_1, header_row_2]
        for _, row in excel_df.iterrows():
            restructured_data.append(row.tolist())

        max_cols = max(len(r) for r in restructured_data)
        for r in restructured_data:
            while len(r) < max_cols:
                r.append("")

        restructured_df = pd.DataFrame(restructured_data)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            restructured_df.to_excel(writer, sheet_name="Sales Forecast", index=False, header=False)
            workbook = writer.book
            worksheet = writer.sheets["Sales Forecast"]

            month_header_format = workbook.add_format(
                {
                    "bold": True,
                    "text_wrap": True,
                    "valign": "top",
                    "align": "center",
                    "fg_color": "#D7E4BC",
                    "border": 1,
                }
            )
            dtype_header_format = workbook.add_format(
                {
                    "bold": True,
                    "text_wrap": True,
                    "valign": "top",
                    "align": "center",
                    "fg_color": "#F2F2F2",
                    "border": 1,
                    "font_size": 9,
                }
            )
            item_code_format = workbook.add_format(
                {
                    "bold": True,
                    "fg_color": "#F2F2F2",
                    "border": 1,
                    "align": "left",
                }
            )
            hist_format = workbook.add_format(
                {
                    "fg_color": "#E8F4FD",
                    "border": 1,
                    "align": "right",
                    "num_format": "#,##0",
                }
            )
            fc_format = workbook.add_format(
                {
                    "fg_color": "#FFF2CC",
                    "border": 1,
                    "align": "right",
                    "num_format": "#,##0",
                }
            )

            for col_num, val in enumerate(header_row_1):
                worksheet.write(0, col_num, val, month_header_format)
            for col_num, val in enumerate(header_row_2):
                worksheet.write(1, col_num, val, dtype_header_format)

            for row_num, data_row in enumerate(restructured_data[2:], start=2):
                for col_num, cell in enumerate(data_row):
                    if col_num == 0:
                        worksheet.write(row_num, col_num, cell, item_code_format)
                    else:
                        col_name = header_row_1[col_num]
                        try:
                            dt = pd.to_datetime(col_name, format="%b-%Y", errors="coerce")
                            if pd.isna(dt) or dt <= max_date:
                                worksheet.write(row_num, col_num, cell, hist_format)
                            else:
                                worksheet.write(row_num, col_num, cell, fc_format)
                        except:
                            worksheet.write(row_num, col_num, cell, fc_format)

            worksheet.set_column(0, 0, 20)
            for col_num in range(1, max_cols):
                worksheet.set_column(col_num, col_num, 14)
            worksheet.freeze_panes(2, 1)

        output.seek(0)
        st.download_button(
            label="📥 Download Excel (SARIMA Forecast)",
            data=output,
            file_name="Parts_Forecast_SARIMA.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.write("Please verify the uploaded Excel has the correct columns (Part, Month, Sales).")
        import traceback
        st.write("Full error trace:")
        st.code(traceback.format_exc())

# CODE REVIEW FINDINGS:

"""
## STRENGTHS:
1. Uses SARIMA which is excellent for seasonal time series
2. Auto-ARIMA for automatic parameter selection
3. Good date handling with explicit normalization to month start
4. Proper MAPE calculation excluding zeros
5. Shows diagnostic information (date ranges, sales stats)
6. Good error handling and fallback mechanisms
7. Professional Excel output with formatting

## POSITIVE ASPECTS OF THIS CODE vs PROPHET:
1. SARIMA is often better for spare parts with strong seasonality
2. Auto-ARIMA removes manual hyperparameter tuning
3. More traditional and interpretable time series approach
4. Better handling of intermittent demand patterns

## POTENTIAL IMPROVEMENTS:

### 1. COMPUTATIONAL EFFICIENCY
- Auto-ARIMA can be slow for many parts
- Consider caching results or parallel processing
- Could reduce search space for faster processing

### 2. HANDLING INTERMITTENT DEMAND
- Many spare parts have intermittent demand (lots of zeros)
- Consider Croston's method or SBA for such parts
- Could add detection for intermittent vs regular demand

### 3. MINIMUM DATA REQUIREMENTS
- Currently requires 12+ months for auto_arima
- Could be relaxed for parts with strong patterns

### 4. FORECAST INTERVALS
- Currently only provides point forecasts
- Could add prediction intervals from SARIMA

### 5. MODEL DIAGNOSTICS
- Could add residual diagnostics
- AIC/BIC values for model selection transparency

## SUGGESTED ENHANCEMENTS:

1. Add option for faster processing:
   - Simple SARIMA without auto-tuning
   - Limit auto_arima search space
   
2. Better handling of sparse data:
   - Detect intermittent demand patterns
   - Use appropriate methods (Croston, SBA)
   
3. Add forecast confidence intervals:
   - Use SARIMA's built-in intervals
   - Show uncertainty in forecasts
   
4. Performance optimization:
   - Process parts in batches
   - Add multiprocessing option
   
5. Additional metrics:
   - Show AIC/BIC for model selection
   - Add RMSE alongside MAPE

The code is production-ready and handles the date issues well. 
The SARIMA approach is appropriate for seasonal spare parts data.
"""
