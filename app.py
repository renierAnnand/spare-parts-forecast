import streamlit as st
import pandas as pd
import numpy as np
import io
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from scipy.stats import boxcox, inv_boxcox

st.set_page_config(layout="wide")
st.title("📈 Spare Parts Forecast (MH-Family SARIMA with Exogenous Regressors)")

# —————————————————————————————
# 1) File Uploaders
# —————————————————————————————

st.write(
    """
    1. Upload an Excel/CSV containing at least these columns:  
       **Part**, **Family**, **Month**, **Sales**.  
       - `Family` groups parts by MH-equipment family (e.g. “Linde-Forks”, “JD-Filters”, …).  
       - `Month` should parseable by pandas (e.g. “2023-05-01” or “May 2023”).  
       - `Sales` = number of units sold that month.  

    2. (Optional) Upload a second file (Excel/CSV) with two columns:  
       **Month**, **FleetHours**—total monthly runtime hours for your MH fleet in Saudi Arabia.  
       This will be used as an exogenous regressor.  
    """
)

uploaded_parts = st.file_uploader(
    "Upload Parts × Family × Month × Sales file", type=["xlsx", "csv"]
)
uploaded_exog = st.file_uploader(
    "Upload FleetHours file (Month, FleetHours)  (Optional)", type=["xlsx", "csv"]
)

if not uploaded_parts:
    st.info("📥 Please upload the Parts×Family×Month×Sales file to proceed.")
    st.stop()

# —————————————————————————————
# 2) Load & Validate Input Data
# —————————————————————————————

@st.cache_data
def load_parts_df(uploaded_file):
    if str(uploaded_file.name).lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

@st.cache_data
def load_exog_df(uploaded_file):
    if str(uploaded_file.name).lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

df = load_parts_df(uploaded_parts)

required_cols = {"Part", "Family", "Month", "Sales"}
if not required_cols.issubset(df.columns):
    st.error(
        f"❌ Your file must contain columns: {required_cols}. Found: {list(df.columns)}"
    )
    st.stop()

# Drop rows with any missing key fields
df = df.dropna(subset=["Part", "Family", "Month", "Sales"]).copy()
df["Part"] = df["Part"].astype(str).str.strip()
df["Family"] = df["Family"].astype(str).str.strip()

# Parse Month → datetime (coerce invalid)
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
n_invalid = df["Month"].isna().sum()
if n_invalid > 0:
    st.warning(f"⚠️ Dropping {n_invalid} rows with invalid Month parse.")
    df = df.dropna(subset=["Month"])

df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0).astype(float)

st.write(f"Data shape after initial cleaning: {df.shape}")
st.write("Preview of uploaded data:")
st.dataframe(df.head())

# —————————————————————————————
# 3) Load Optional Exogenous (FleetHours) + Build IsRamadanMonth Flag
# —————————————————————————————

if uploaded_exog:
    exog_df = load_exog_df(uploaded_exog)
    if not {"Month", "FleetHours"}.issubset(exog_df.columns):
        st.error("❌ Exogenous file must have columns: Month, FleetHours.")
        st.stop()
    exog_df["Month"] = pd.to_datetime(exog_df["Month"], errors="coerce")
    ninv = exog_df["Month"].isna().sum()
    if ninv > 0:
        st.warning(f"⚠️ Dropping {ninv} rows from exogenous data with invalid Month.")
        exog_df = exog_df.dropna(subset=["Month"])
    exog_df["FleetHours"] = pd.to_numeric(exog_df["FleetHours"], errors="coerce").fillna(0)
    # Reindex exog to Month→FleetHours
    fleet_hours = (
        exog_df
        .groupby(pd.Grouper(key="Month", freq="MS"))["FleetHours"]
        .sum()
        .sort_index()
    )
else:
    # If no exog provided, create a zero series placeholder
    fleet_hours = pd.Series(dtype=float)

# Build a simple monthly Ramadan indicator (Saudi Arabia) for 2023–2025
# NOTE: In production, replace this with a full Hijri→Gregorian conversion.
def build_ramadan_flag(idx):
    # Approximate: Ramadan 2023: Mar 23 – Apr 21
    #            Ramadan 2024: Mar 10 – Apr 9
    #            Ramadan 2025: Feb 28 – Mar 29
    # We mark any month overlapping these ranges as IsRamadan=1
    flags = []
    for ts in idx:
        y, m = ts.year, ts.month
        # Check these month ranges:
        if (y == 2023 and m in (3, 4)) or \
           (y == 2024 and m in (3, 4)) or \
           (y == 2025 and m in (2, 3)):
            flags.append(1)
        else:
            flags.append(0)
    return pd.Series(flags, index=idx)

# —————————————————————————————
# 4) Aggregate (Family, Month) at both Family and Part Levels
# —————————————————————————————

# 4.a Family-level: for auto-ARIMA parameter selection
family_ts = (
    df
    .groupby([pd.Grouper(key="Month", freq="MS"), "Family"])["Sales"]
    .sum()
    .unstack(fill_value=0)
    .sort_index()
)

# 4.b Part-level: sum any duplicates
part_ts_df = (
    df
    .groupby([pd.Grouper(key="Month", freq="MS"), "Part", "Family"])["Sales"]
    .sum()
    .reset_index()
)

# —————————————————————————————
# 5) Determine Overall Historical Range & Build Month Indices
# —————————————————————————————

min_date = part_ts_df["Month"].min()
max_date = part_ts_df["Month"].max()
min_date = min_date.replace(day=1)
max_date = max_date.replace(day=1)

historical_months = pd.date_range(start=min_date, end=max_date, freq="MS")
forecast_start = max_date + pd.DateOffset(months=1)
forecast_end = forecast_start + pd.DateOffset(months=11)
forecast_months = pd.date_range(start=forecast_start, end=forecast_end, freq="MS")

st.write(
    f"📅 Historical data: {min_date.strftime('%Y-%m')} → {max_date.strftime('%Y-%m')} "
    f"({len(historical_months)} months)."
)
st.write(
    f"🔮 Forecast horizon: {forecast_start.strftime('%Y-%m')} → {forecast_end.strftime('%Y-%m')}."
)

# Build seasonal Ramadan flags for historical + forecast months
all_months = historical_months.union(forecast_months)
ramadan_flag = build_ramadan_flag(all_months)

# Build a complete exogenous DataFrame indexed by Month: columns: FleetHours, IsRamadan
exog_full = pd.DataFrame(index=all_months)
# 1) Populate FleetHours: reindex to all_months (fill missing with 0)
if not fleet_hours.empty:
    exog_full["FleetHours"] = fleet_hours.reindex(all_months, fill_value=0)
else:
    exog_full["FleetHours"] = 0.0

# 2) Populate Ramadan indicator
exog_full["IsRamadanMonth"] = ramadan_flag

st.write("Here is a preview of the exogenous DataFrame (FleetHours + Ramadan flag):")
st.dataframe(exog_full.head(12))

# —————————————————————————————
# 6) Pre-Fit Auto-ARIMA for Each Family (to get (p,d,q) × (P,D,Q,12))
# —————————————————————————————

st.write("### ▶️ Pre-fitting Auto-ARIMA on each Family (this may take a moment)…")
family_params = {}  # { family: {"order":(p,d,q), "seasonal":(P,D,Q,12)} }

for fam in family_ts.columns:
    ts_fam = family_ts[fam].reindex(historical_months, fill_value=0)
    # Skip families that have almost no data
    if ts_fam.sum() < 1 or (ts_fam > 0).sum() < 3:
        # fallback to default (1,1,1)x(1,1,1,12)
        family_params[fam] = {
            "order": (1, 1, 1),
            "seasonal": (1, 1, 1, 12),
        }
        continue

    try:
        # We include Exogenous = FleetHours + Ramadan for the family fit as well
        exog_fam = exog_full["FleetHours"].loc[historical_months].values.reshape(-1, 1)
        # (Optional) You could also pass IsRamadanMonth as a second column:
        exog_fam2 = np.vstack([
            exog_full["IsRamadanMonth"].loc[historical_months].values,
            exog_fam.flatten()
        ]).T

        fam_model = pm.auto_arima(
            ts_fam,
            exogenous=exog_fam2,
            seasonal=True,
            m=12,
            start_p=0,
            max_p=2,
            start_q=0,
            max_q=2,
            start_P=0,
            max_P=1,
            start_Q=0,
            max_Q=1,
            d=None,
            D=1,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
            n_jobs=1,
        )
        family_params[fam] = {
            "order": fam_model.order,
            "seasonal": fam_model.seasonal_order,
        }
    except Exception as e:
        # If auto_arima fails, fall back
        family_params[fam] = {
            "order": (1, 1, 1),
            "seasonal": (1, 1, 1, 12),
        }

st.success("✅ Finished pre-fitting family-level SARIMA parameters.")
st.write("Family parameters discovered:")
fam_summary = pd.DataFrame.from_dict(
    {fam: {"order": str(vals["order"]), "seasonal": str(vals["seasonal"])} 
     for fam, vals in family_params.items()}, 
    orient="index"
)
st.dataframe(fam_summary)

# —————————————————————————————
# 7) Loop Over Each Part (SKU) to Forecast
# —————————————————————————————

progress_bar = st.progress(0.0)
status_text = st.empty()

results_data = []
parts_list = sorted(part_ts_df["Part"].unique())
total_parts = len(parts_list)

parts_with_sarima = 0
parts_with_naive = 0
parts_failed = 0

# Precompute: number of parts per family (for allocation in fallbacks)
n_parts_in_family = (
    part_ts_df[["Part", "Family"]]
    .drop_duplicates()
    .groupby("Family")["Part"].nunique()
    .to_dict()
)

for idx, part in enumerate(parts_list):
    progress_bar.progress((idx + 1) / total_parts)
    status_text.text(f"Processing part {idx + 1}/{total_parts}: {part}")

    try:
        # Extract raw series for this part
        sub = part_ts_df[part_ts_df["Part"] == part][["Month", "Sales", "Family"]]
        fam = sub["Family"].iloc[0]
        raw = sub.set_index("Month")[["Sales"]].sort_index()
        raw = raw.reindex(historical_months, fill_value=0)  # zero-fill missing months

        # 1) Determine “active” start (first non-zero)
        if (raw["Sales"] > 0).any():
            start_nonzero = raw[raw["Sales"] > 0].index.min()
            ts = raw.loc[start_nonzero:][ "Sales" ].copy()
        else:
            ts = raw["Sales"].copy()

        # If not enough history or non-zeros, fallback to Family average
        non_zero_count = (ts > 0).sum()
        if (len(ts) < 12) or (non_zero_count < 3):
            # Use family’s last 3 months total, then divide equally among SKUs
            fam_hist = family_ts[fam].reindex(historical_months, fill_value=0)
            last3_total = fam_hist.iloc[-3:].sum()
            fallback_per_sku = int(round(last3_total / max(n_parts_in_family[fam], 1), 0))
            fc_vals = [fallback_per_sku] * 12
            method_used = "Naive (family avg)"
            parts_with_naive += 1
        else:
            # 2) Box–Cox transform (add tiny constant to avoid zeros)
            ts_pos = ts + 1e-6
            ts_bc, λ = boxcox(ts_pos)

            # 3) Use pre-fitted family params
            fam_order = family_params[fam]["order"]
            fam_seasonal = family_params[fam]["seasonal"]

            # 4) Build exogenous matrix for “in-sample” (ensuring alignment)
            exog_insample = exog_full.loc[ts.index, ["FleetHours", "IsRamadanMonth"]].values

            # 5) Fit SARIMAX on transformed series
            try:
                model = SARIMAX(
                    ts_bc,
                    exog=exog_insample,
                    order=fam_order,
                    seasonal_order=fam_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                # 6) Forecast next 12 months with exog
                exog_out = exog_full.loc[forecast_months, ["FleetHours", "IsRamadanMonth"]].values
                bc_forecast = model.get_forecast(steps=12, exog=exog_out).predicted_mean
                preds = inv_boxcox(bc_forecast, λ).round().astype(int)
                preds = np.clip(preds, 0, None)  # no negative forecasts

                fc_vals = preds.tolist()
                method_used = f"SARIMA (fam={fam})"
                parts_with_sarima += 1

            except Exception as e:
                # Fallback if SARIMAX blew up
                fam_hist = family_ts[fam].reindex(historical_months, fill_value=0)
                last6 = fam_hist.iloc[-6:].mean()
                fallback2 = int(round(last6 / max(n_parts_in_family[fam], 1), 0))
                fc_vals = [fallback2] * 12
                method_used = "Naive (SARIMAX failed)"
                parts_with_naive += 1

        # 7) Build output row: historical + forecast columns
        row_data = {"Item Code": part}
        # Historical (all months)
        for hist_m in historical_months:
            row_data[hist_m.strftime("%b-%Y")] = int(raw.loc[hist_m, "Sales"])

        # Forecast months
        for i, fc_m in enumerate(forecast_months):
            row_data[fc_m.strftime("%b-%Y")] = int(fc_vals[i])

        row_data["Method"] = method_used
        results_data.append(row_data)

    except Exception as e:
        st.warning(f"⚠️ Skipping part {part} due to error: {e}")
        parts_failed += 1
        continue

progress_bar.empty()
status_text.empty()

# —————————————————————————————
# 8) Summarize & Build Final DataFrame
# —————————————————————————————

st.write("### 📊 Processing Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Parts", total_parts)
c2.metric("SARIMA Models", parts_with_sarima)
c3.metric("Naive Forecasts", parts_with_naive)
c4.metric("Failed", parts_failed)

result_df = pd.DataFrame(results_data)

# Reorder columns: Item Code → all months (hist + forecast) → Method
all_month_cols = [m.strftime("%b-%Y") for m in list(historical_months) + list(forecast_months)]
ordered_cols = ["Item Code"] + [c for c in all_month_cols if c in result_df.columns]
final_cols = ordered_cols + ["Method"]
result_df = result_df[final_cols].copy()

st.success("✅ Forecasting Completed!")
st.write(f"🔢 Parts processed: {len(result_df)}")

# Show a quick preview (last 3 historical + first 3 forecast)
preview_cols = (
    ["Item Code"]
    + all_month_cols[-3:]
    + [m.strftime("%b-%Y") for m in forecast_months[:3]]
    + ["Method"]
)
preview_cols = [c for c in preview_cols if c in result_df.columns]
st.write("**Forecast Sample (first 10 SKUs)**")
st.dataframe(result_df[preview_cols].head(10))

# —————————————————————————————
# 9) Download as Excel (with two-row headers)
# —————————————————————————————

excel_df = result_df[["Item Code"] + all_month_cols].fillna("")

# Build header rows
hdr1 = ["Item Code"]
hdr2 = [""]
for col in all_month_cols:
    hdr1.append(col)
    dt = pd.to_datetime(col, format="%b-%Y", errors="coerce")
    if pd.isna(dt) or (dt <= max_date):
        hdr2.append("Historical QTY")
    else:
        hdr2.append("Forecasted QTY")

two_row = [hdr1, hdr2]
for _, row in excel_df.iterrows():
    two_row.append(row.tolist())

# Pad rows to equal length
max_c = max(len(r) for r in two_row)
for r in two_row:
    while len(r) < max_c:
        r.append("")

hdr_df = pd.DataFrame(two_row)

output = io.BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    hdr_df.to_excel(writer, sheet_name="MH_Sales_Forecast", index=False, header=False)
    wb = writer.book
    ws = writer.sheets["MH_Sales_Forecast"]

    fmt_month = wb.add_format(
        {
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "align": "center",
            "fg_color": "#D7E4BC",
            "border": 1,
        }
    )
    fmt_type = wb.add_format(
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
    fmt_item = wb.add_format(
        {
            "bold": True,
            "fg_color": "#F2F2F2",
            "border": 1,
            "align": "left",
        }
    )
    fmt_hist = wb.add_format(
        {
            "fg_color": "#E8F4FD",
            "border": 1,
            "align": "right",
            "num_format": "#,##0",
        }
    )
    fmt_fc = wb.add_format(
        {
            "fg_color": "#FFF2CC",
            "border": 1,
            "align": "right",
            "num_format": "#,##0",
        }
    )

    # Write headers
    for col_idx, val in enumerate(hdr1):
        ws.write(0, col_idx, val, fmt_month)
    for col_idx, val in enumerate(hdr2):
        ws.write(1, col_idx, val, fmt_type)

    # Write data rows
    for r_idx, row in enumerate(two_row[2:], start=2):
        for c_idx, cell in enumerate(row):
            if c_idx == 0:
                ws.write(r_idx, c_idx, cell, fmt_item)
            else:
                colname = hdr1[c_idx]
                dt = pd.to_datetime(colname, format="%b-%Y", errors="coerce")
                if pd.isna(dt) or (dt <= max_date):
                    ws.write(r_idx, c_idx, cell, fmt_hist)
                else:
                    ws.write(r_idx, c_idx, cell, fmt_fc)

    ws.set_column(0, 0, 20)
    for c in range(1, max_c):
        ws.set_column(c, c, 14)
    ws.freeze_panes(2, 1)

output.seek(0)
st.download_button(
    label="📥 Download Excel (MH Family SARIMA Forecast)",
    data=output,
    file_name="MH_Parts_Forecast.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
