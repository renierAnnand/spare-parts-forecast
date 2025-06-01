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
        
        # Ensure Part column is string type and strip whitespace
        df["Part"] = df["Part"].astype(str).str.strip()
        
        # Convert Month to datetime - handle various date formats
        try:
            df["Month"] = pd.to_datetime(df["Month"])
        except:
            # Try different date parsing strategies
            st.warning("Attempting alternative date parsing...")
            df["Month"] = pd.to_datetime(df["Month"], errors='coerce')
            
        # Remove rows where date parsing failed
        date_parse_failed = df["Month"].isna().sum()
        if date_parse_failed > 0:
            st.warning(f"⚠️ Removed {date_parse_failed} rows with invalid dates")
            df = df.dropna(subset=["Month"])
            
        df["Sales"] = pd.to_numeric(df["Sales"], errors='coerce').fillna(0).astype(float)
        
        # Debug: Show sales statistics
        st.write("### Sales Data Statistics:")
        st.write(f"- Total rows after cleaning: {len(df)}")
        st.write(f"- Non-zero sales rows: {(df['Sales'] > 0).sum()}")
        st.write(f"- Average sales: {df['Sales'].mean():.2f}")
        st.write(f"- Max sales: {df['Sales'].max()}")
        st.write(f"- Parts with any sales: {df[df['Sales'] > 0]['Part'].nunique()}")
        
        # Show top parts by total sales
        top_parts = df.groupby('Part')['Sales'].sum().sort_values(ascending=False).head(10)
        if len(top_parts) > 0:
            st.write("### Top 10 Parts by Total Sales:")
            st.dataframe(top_parts)

        # 4) Aggregate any duplicate (Part, Month) pairs
        df_grouped = (
            df.groupby(["Part", pd.Grouper(key="Month", freq="M")])["Sales"]
            .sum()
            .reset_index()
        )

        # 5) Compute overall date range
        min_date = df_grouped["Month"].min()
        max_date = df_grouped["Month"].max()

        # Ensure we have valid dates
        if pd.isna(min_date) or pd.isna(max_date):
            st.error("❌ No valid dates found in the data. Please check your Month column format.")
            st.stop()

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
        parts_list = sorted(df_grouped["Part"].unique())  # Sort for consistent output
        total_parts = len(parts_list)
        
        # Track processing statistics
        parts_with_prophet = 0
        parts_with_naive = 0
        parts_failed = 0

        # Silence Prophet logging
        logging.getLogger("prophet").setLevel(logging.WARNING)

        for idx, part in enumerate(parts_list):
            progress_bar.progress((idx + 1) / total_parts)
            status_text.text(f"Processing part {idx + 1}/{total_parts}: {part}")

            try:
                # 9.a) Extract raw monthly series for this part
                raw = (
                    df_grouped[df_grouped["Part"] == part]
                    .loc[:, ["Month", "Sales"]]
                    .set_index("Month")
                    .sort_index()
                )
                
                # Check if we have any data for this part
                if len(raw) == 0:
                    st.warning(f"No data found for part: {part}")
                    parts_failed += 1
                    continue

                # 9.b) Build a continuous series from this part's first sale to max_date
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
                # Also check if we have at least 2 non-zero values for Prophet
                non_zero_count = (part_train["y"] > 0).sum()
                if part_train["ds"].nunique() < 8 or non_zero_count < 2:
                    last_three = part_train.sort_values("ds").tail(3)["y"]
                    naive_forecast = int(round(last_three.mean(), 0)) if len(last_three) > 0 else 0
                    
                    # Option to set minimum forecast value
                    # naive_forecast = max(naive_forecast, 1)  # Uncomment for minimum forecast of 1
                    
                    for fc_month in forecast_months:
                        row_data[fc_month.strftime("%b-%Y")] = naive_forecast

                    # Mark validation fields as NaN when using naive
                    row_data["CV_MAPE"] = np.nan
                    row_data["Chosen_cps"] = np.nan
                    row_data["Chosen_sps"] = np.nan
                    row_data["Method"] = "Naive"

                    results_data.append(row_data)
                    parts_with_naive += 1
                    continue

                # 9.f) Otherwise, split last 12 months for in‐sample validation
                # Ensure that we have at least 12 months to hold out
                if len(part_train) > 12:
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

                if not valid_df.empty and len(train_df) >= 2:
                    # Only run grid‐search if we have a validation window
                    for cps in [0.001, 0.01, 0.05]:
                        for sps in [1.0, 5.0, 10.0]:
                            try:
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
                                future_valid = valid_df[["ds"]].copy()
                                pred_valid = m_tmp.predict(future_valid)["yhat"].values
                                actual_valid = valid_df.sort_values("ds")["y"].values

                                # Compute MAPE (exclude zeros from calculation)
                                mask = actual_valid != 0
                                if mask.any():
                                    mape = np.mean(np.abs((actual_valid[mask] - pred_valid[mask]) / actual_valid[mask])) * 100
                                else:
                                    mape = np.inf
                                    
                                if mape < best_mape:
                                    best_mape = mape
                                    best_params = {"cps": cps, "sps": sps}
                            except Exception as e:
                                # Skip this parameter combination if fitting fails
                                continue

                    # If we never improved (e.g. all infinite), leave defaults
                    row_data["CV_MAPE"] = round(best_mape, 2) if best_mape != np.inf else np.nan
                    row_data["Chosen_cps"] = best_params["cps"]
                    row_data["Chosen_sps"] = best_params["sps"]
                else:
                    # No validation window: skip grid search
                    row_data["CV_MAPE"] = np.nan
                    row_data["Chosen_cps"] = np.nan
                    row_data["Chosen_sps"] = np.nan

                # 9.h) Fit final model on ALL available history (part_train)
                try:
                    # Check if we have enough data for Prophet
                    if len(part_train) < 2 or (part_train["y"] > 0).sum() < 2:
                        # Fallback to naive forecast
                        last_three = part_train.sort_values("ds").tail(3)["y"]
                        naive_forecast = int(round(last_three.mean(), 0)) if len(last_three) > 0 else 0
                        for fc_month in forecast_months:
                            row_data[fc_month.strftime("%b-%Y")] = naive_forecast
                        row_data["Method"] = "Naive"
                        results_data.append(row_data)
                        parts_with_naive += 1
                        continue
                        
                    final_model = Prophet(
                        changepoint_prior_scale=best_params["cps"],
                        seasonality_prior_scale=best_params["sps"],
                        seasonality_mode="additive" if len(part_train) < 36 else "multiplicative",
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

                    row_data["Method"] = "Prophet"
                    results_data.append(row_data)
                    parts_with_prophet += 1
                    
                except Exception as e:
                    # If Prophet fails, fallback to naive forecast
                    st.warning(f"Prophet failed for part {part}: {str(e)}. Using naive forecast.")
                    last_three = part_train.sort_values("ds").tail(3)["y"]
                    naive_forecast = int(round(last_three.mean(), 0)) if len(last_three) > 0 else 0
                    for fc_month in forecast_months:
                        row_data[fc_month.strftime("%b-%Y")] = naive_forecast
                    row_data["Method"] = "Naive (Prophet failed)"
                    results_data.append(row_data)
                    parts_with_naive += 1
                    
            except Exception as e:
                st.error(f"Error processing part {part}: {str(e)}")
                parts_failed += 1
                continue

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # 10) Create final DataFrame
        result_df = pd.DataFrame(results_data)

        # Show processing summary
        st.write("### 📊 Processing Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Parts", total_parts)
        with col2:
            st.metric("Prophet Models", parts_with_prophet)
        with col3:
            st.metric("Naive Forecasts", parts_with_naive)
        with col4:
            st.metric("Failed", parts_failed)

        # 11) Show in‐sample validation metrics (if any)
        if "CV_MAPE" in result_df.columns:
            cv_table = result_df[["Item Code", "CV_MAPE", "Method"]].dropna(subset=["CV_MAPE"]).sort_values(
                by="CV_MAPE", ascending=False
            )
            if len(cv_table) > 0:
                st.write("### 🧪 In‐Sample Validation (CV MAPE) by Part")
                st.dataframe(cv_table.head(15))

        # 12) Reorder columns: Item Code → all months (historical + forecast) → CV columns → Method
        month_columns = [m.strftime("%b-%Y") for m in list(historical_months) + list(forecast_months)]
        ordered_cols = ["Item Code"] + [c for c in month_columns if c in result_df.columns]
        final_cols = ordered_cols + [c for c in ["CV_MAPE", "Chosen_cps", "Chosen_sps", "Method"] if c in result_df.columns]
        result_df = result_df[final_cols]

        st.success("✅ Forecasting completed!")
        st.write(f"🔢 Number of parts processed: {len(result_df)}")

        # 13) Show a quick preview of results
        st.write("**Forecast sample (first 10 parts):**")
        preview_cols = ["Item Code"] + month_columns[-3:] + forecast_months[:3].strftime("%b-%Y").tolist() + ["Method"]
        preview_cols = [c for c in preview_cols if c in result_df.columns]
        st.dataframe(result_df[preview_cols].head(10))

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
        import traceback
        st.write("Full error trace:")
        st.code(traceback.format_exc())
