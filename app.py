import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
import io
import logging

st.set_page_config(layout="wide")
st.title("📈 Spare Parts Forecast (Horizontal Layout)")

uploaded_file = st.file_uploader("Upload Excel file with columns: Part, Month, Sales", type=["xlsx"])

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
            st.error(f"Please ensure your file has columns: Part, Month, Sales. Found: {list(df.columns)}")
            st.stop()

        # 3) Preprocess input
        df = df.dropna(subset=["Part", "Month", "Sales"])
        df["Month"] = pd.to_datetime(df["Month"])
        df["Sales"] = df["Sales"].astype(float)

        # 4) Aggregate any duplicate (Part, Month) pairs
        df_grouped = (
            df.groupby(["Part", pd.Grouper(key="Month", freq="M")])["Sales"]
            .sum()
            .reset_index()
        )

        # 5) Compute overall date range
        min_date = df_grouped["Month"].min()
        max_date = df_grouped["Month"].max()

        data_span_months = (
            (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
        )
        data_span_years = data_span_months / 12

        st.write(f"📅 Historical data covers: {min_date.strftime('%Y-%m')} → {max_date.strftime('%Y-%m')}")
        st.write(f"📦 Total unique parts: {df_grouped['Part'].nunique()}")
        st.write(f"📊 Total aggregated records: {len(df_grouped)}")
        st.write(f"📊 Data span: {data_span_months} months ({data_span_years:.1f} years)")

        # 6) Warn if too little data overall
        if data_span_months < 12:
            st.warning("⚠️ Fewer than 12 months of history total – forecasts may be unreliable for many parts.")
        elif data_span_months < 24:
            st.info("ℹ️ Between 12 and 24 months of data – consider adding more history if possible.")
        else:
            st.success("✅ Good historical coverage (≥24 months).")

        # 7) Define monthly indices
        historical_months = pd.date_range(start=min_date, end=max_date, freq="MS")
        forecast_start = max_date + pd.DateOffset(months=1)
        forecast_end = forecast_start + pd.DateOffset(months=11)
        forecast_months = pd.date_range(start=forecast_start, end=forecast_end, freq="MS")

        st.success(f"🔮 Prediction period: {forecast_start.strftime('%Y-%m')} → {forecast_end.strftime('%Y-%m')}")

        # 8) Prepare progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 9) Loop over each unique part
        results_data = []
        parts_list = df_grouped["Part"].unique()
        total_parts = len(parts_list)

        # Silence Prophet logging
        logging.getLogger("prophet").setLevel(logging.WARNING)

        for idx, part in enumerate(parts_list):
            progress_bar.progress((idx + 1) / total_parts)
            status_text.text(f"Processing part {idx + 1}/{total_parts}: {part}")

            # 9.a) Extract raw monthly series for this part
            raw = (
                df_grouped[df_grouped["Part"] == part]
                .loc[:, ["Month", "Sales"]]
                .set_index("Month")
                .sort_index()
            )

            # 9.b) Build a continuous series from this part’s first sale to max_date
            part_index = pd.date_range(start=raw.index.min(), end=max_date, freq="MS")
            part_train = raw.reindex(part_index, fill_value=0).reset_index().rename(
                columns={"index": "ds", "Sales": "y"}
            )

            # 9.c) Build a complete historical row (including zeros) aligned to global historical_months
            part_full_hist = raw.reindex(historical_months, fill_value=0).reset_index().rename(
                columns={"index": "Month", "Sales": "Sales"}
            )

            # 9.d) Initialize row_data for output
            row_data = {"Item Code": part}
            for hist_month in historical_months:
                sales_val = part_full_hist.loc[
                    part_full_hist["Month"] == hist_month, "Sales"
                ]
                row_data[hist_month.strftime("%b-%Y")] = int(sales_val.iloc[0]) if len(sales_val) > 0 else 0

            # 9.e) If too few data points (<8 months), use naive 3‐month average
            if part_train["ds"].nunique() < 8:
                last_three = part_train.sort_values("ds").tail(3)["y"]
                naive_forecast = int(round(last_three.mean(), 0)) if len(last_three) > 0 else 0
                for fc_month in forecast_months:
                    row_data[fc_month.strftime("%b-%Y")] = naive_forecast

                # Mark validation fields as NaN when using naive
                row_data["CV_MAPE"] = np.nan
                row_data["Chosen_cps"] = np.nan
                row_data["Chosen_sps"] = np.nan

                results_data.append(row_data)
                continue

            # 9.f) Otherwise, split last 12 months for in‐sample validation
            # Ensure that we have at least 12 months to hold out
            if data_span_months > 12:
                cutoff_date = max_date - pd.DateOffset(months=12)
                train_df = part_train[part_train["ds"] <= cutoff_date].copy()
                valid_df = part_train[part_train["ds"] > cutoff_date].copy()
            else:
                # If overall history ≤12 months, use all for training (skip validation)
                train_df = part_train.copy()
                valid_df = pd.DataFrame(columns=part_train.columns)

            # 9.g) Hyperparameter grid‐search over changepoint/seasonality priors, if we have a valid set
            best_mape = np.inf
            best_params = {"cps": 0.05, "sps": 5.0}

            if not valid_df.empty:
                # Only run grid‐search if we have a validation window
                for cps in [0.001, 0.01, 0.05]:
                    for sps in [1.0, 5.0, 10.0]:
                        m_tmp = Prophet(
                            changepoint_prior_scale=cps,
                            seasonality_prior_scale=sps,
                            seasonality_mode="additive",
                            yearly_seasonality=False,
                            weekly_seasonality=False,
                            daily_seasonality=False,
                        )
                        m_tmp.add_seasonality(name="monthly", period=12, fourier_order=3)
                        m_tmp.fit(train_df)

                        # Forecast the validation window
                        future_valid = valid_df[["ds"]].rename(columns={"ds": "ds"})
                        pred_valid = m_tmp.predict(future_valid)["yhat"].values
                        actual_valid = valid_df.sort_values("ds")["y"].values

                        # Compute MAPE (avoid division by zero)
                        mape = np.mean(
                            np.abs((actual_valid - pred_valid) / (actual_valid + 1e-9))
                        ) * 100
                        if mape < best_mape:
                            best_mape = mape
                            best_params = {"cps": cps, "sps": sps}

                # If we never improved (e.g. all infinite), leave defaults
                row_data["CV_MAPE"] = round(best_mape, 2)
                row_data["Chosen_cps"] = best_params["cps"]
                row_data["Chosen_sps"] = best_params["sps"]
            else:
                # No validation window: skip grid search
                row_data["CV_MAPE"] = np.nan
                row_data["Chosen_cps"] = np.nan
                row_data["Chosen_sps"] = np.nan

            # 9.h) Fit final model on ALL available history (part_train)
            final_model = Prophet(
                changepoint_prior_scale=best_params["cps"],
                seasonality_prior_scale=best_params["sps"],
                seasonality_mode="additive" if data_span_months < 36 else "multiplicative",
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
            )
            final_model.add_seasonality(name="monthly", period=12, fourier_order=3)
            final_model.fit(part_train)

            # 9.i) Forecast the next 12 months (forecast_months)
            future_df = pd.DataFrame({"ds": forecast_months})
            forecast = final_model.predict(future_df)

            for j, fc_month in enumerate(forecast_months):
                yhat = forecast.iloc[j]["yhat"]
                row_data[fc_month.strftime("%b-%Y")] = max(0, int(round(yhat, 0)))

            results_data.append(row_data)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # 10) Create final DataFrame
        result_df = pd.DataFrame(results_data)

        # 11) Show in‐sample validation metrics (if any)
        if "CV_MAPE" in result_df.columns:
            cv_table = result_df[["Item Code", "CV_MAPE"]].sort_values(
                by="CV_MAPE", ascending=False
            )
            st.write("### 🧪 In‐Sample Validation (CV MAPE) by Part")
            st.dataframe(cv_table.head(15))

        # 12) Reorder columns: Item Code → all months (historical + forecast) → CV columns
        month_columns = [m.strftime("%b-%Y") for m in list(historical_months) + list(forecast_months)]
        ordered_cols = ["Item Code"] + [c for c in month_columns if c in result_df.columns]
        final_cols = ordered_cols + [c for c in ["CV_MAPE", "Chosen_cps", "Chosen_sps"] if c in result_df.columns]
        result_df = result_df[final_cols]

        st.success("✅ Forecasting completed!")
        st.write(f"🔢 Number of parts processed: {len(result_df)}")

        # 13) Show a quick preview of results
        st.write("**Forecast sample (first 5 parts):**")
        st.dataframe(result_df.iloc[:5, : min(len(final_cols), 8)])

        # 14) Prepare Excel download (only include Item Code + month columns, not CV fields)
        excel_df = result_df.loc[:, ["Item Code"] + month_columns]
        excel_df = excel_df.fillna("")

        # Build a two‐row header: 
        #   Row 1: month names 
        #   Row 2: "Historical QTY" or "Forecasted QTY"
        header_row_1 = ["Item Code"]
        header_row_2 = [""]
        for col in month_columns:
            header_row_1.append(col)
            dt = pd.to_datetime(col, format="%b-%Y", errors="coerce")
            if pd.isna(dt) or dt <= max_date:
                header_row_2.append("Historical QTY")
            else:
                header_row_2.append("Forecasted QTY")

        # Assemble restructured data
        restructured_data = [header_row_1, header_row_2]
        for _, row in excel_df.iterrows():
            restructured_data.append(row.tolist())

        # Pad rows to equal length
        max_cols = max(len(r) for r in restructured_data)
        for r in restructured_data:
            while len(r) < max_cols:
                r.append("")

        restructured_df = pd.DataFrame(restructured_data)

        # Write to Excel with formatting
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            restructured_df.to_excel(writer, sheet_name="Sales Forecast", index=False, header=False)
            workbook = writer.book
            worksheet = writer.sheets["Sales Forecast"]

            # Define formats
            month_header_format = workbook.add_format({
                "bold": True,
                "text_wrap": True,
                "valign": "top",
                "align": "center",
                "fg_color": "#D7E4BC",
                "border": 1,
            })
            dtype_header_format = workbook.add_format({
                "bold": True,
                "text_wrap": True,
                "valign": "top",
                "align": "center",
                "fg_color": "#F2F2F2",
                "border": 1,
                "font_size": 9,
            })
            item_code_format = workbook.add_format({
                "bold": True,
                "fg_color": "#F2F2F2",
                "border": 1,
                "align": "left",
            })
            hist_format = workbook.add_format({
                "fg_color": "#E8F4FD",
                "border": 1,
                "align": "right",
                "num_format": "#,##0",
            })
            fc_format = workbook.add_format({
                "fg_color": "#FFF2CC",
                "border": 1,
                "align": "right",
                "num_format": "#,##0",
            })

            # Write header rows
            for col_num, val in enumerate(header_row_1):
                worksheet.write(0, col_num, val, month_header_format)
            for col_num, val in enumerate(header_row_2):
                worksheet.write(1, col_num, val, dtype_header_format)

            # Write data rows
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

            # Set column widths & freeze panes
            worksheet.set_column(0, 0, 20)
            for col_num in range(1, max_cols):
                worksheet.set_column(col_num, col_num, 14)
            worksheet.freeze_panes(2, 1)

        output.seek(0)
        st.download_button(
            label="📥 Download Excel (Horizontal Layout)",
            data=output,
            file_name="Parts_Forecast_Horizontal_Layout.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.write("Please verify the uploaded Excel has the correct columns (Part, Month, Sales).")
