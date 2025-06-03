import streamlit as st
import pandas as pd
import numpy as np
import io
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import boxcox, inv_boxcox

st.set_page_config(layout="wide")
st.title("📈 Spare Parts Forecast (MH-Family SARIMA)") 

# 1) Upload parts data (Part, Family, Month, Sales)
uploaded_parts = st.file_uploader("Upload Parts×Family×Month×Sales", type=["xlsx","csv"])
uploaded_exog  = st.file_uploader("Upload FleetHours (optional)", type=["xlsx","csv"])

if not uploaded_parts:
    st.info("📥 Please upload your Parts×Family×Month×Sales file.")
    st.stop()

@st.cache_data
def load_df(f):
    if f.name.lower().endswith(".csv"):
        return pd.read_csv(f)
    else:
        return pd.read_excel(f)

df = load_df(uploaded_parts)
required = {"Part","Family","Month","Sales"}
if not required.issubset(df.columns):
    st.error(f"❌ Required columns: {required}. Found: {list(df.columns)}")
    st.stop()

df = df.dropna(subset=["Part","Family","Month","Sales"])
df["Part"]   = df["Part"].astype(str).str.strip()
df["Family"] = df["Family"].astype(str).str.strip()
df["Month"]  = pd.to_datetime(df["Month"], errors="coerce")
df = df.dropna(subset=["Month"])
df["Sales"]  = pd.to_numeric(df["Sales"], errors="coerce").fillna(0).astype(float)

# 2) Load optional exogenous (FleetHours)
if uploaded_exog:
    exog_df = load_df(uploaded_exog)
    if not {"Month","FleetHours"}.issubset(exog_df.columns):
        st.error("❌ FleetHours file must have columns Month, FleetHours")
        st.stop()
    exog_df["Month"]       = pd.to_datetime(exog_df["Month"], errors="coerce")
    exog_df = exog_df.dropna(subset=["Month"])
    exog_df["FleetHours"]  = pd.to_numeric(exog_df["FleetHours"], errors="coerce").fillna(0)
    fleet_hours = (
        exog_df
        .groupby(pd.Grouper(key="Month", freq="MS"))["FleetHours"]
        .sum()
        .sort_index()
    )
else:
    fleet_hours = pd.Series(dtype=float)

# 3) Simple Ramadan‐month flag for 2023–2025 (Saudi Arabia)
def build_ramadan_flag(idx):
    flags = []
    for ts in idx:
        y,m = ts.year, ts.month
        if (y==2023 and m in (3,4)) or (y==2024 and m in (3,4)) or (y==2025 and m in (2,3)):
            flags.append(1)
        else:
            flags.append(0)
    return pd.Series(flags, index=idx)

# 4) Aggregate at (Family, Month) and (Part, Month)
family_ts = (
    df
    .groupby([pd.Grouper(key="Month", freq="MS"), "Family"])["Sales"]
    .sum()
    .unstack(fill_value=0)
    .sort_index()
)

part_ts_df = (
    df
    .groupby([pd.Grouper(key="Month", freq="MS"), "Part", "Family"])["Sales"]
    .sum()
    .reset_index()
)

# 5) Compute historical and forecast horizon
min_date = part_ts_df["Month"].min().replace(day=1)
max_date = part_ts_df["Month"].max().replace(day=1)
hist_months   = pd.date_range(start=min_date, end=max_date, freq="MS")
fc_start = max_date + pd.DateOffset(months=1)
fc_end   = fc_start + pd.DateOffset(months=11)
fc_months = pd.date_range(start=fc_start, end=fc_end, freq="MS")

all_months = hist_months.union(fc_months)
ramadan_flag = build_ramadan_flag(all_months)

# Build exogenous DataFrame for all months
exog_full = pd.DataFrame(index=all_months)
exog_full["FleetHours"]      = fleet_hours.reindex(all_months, fill_value=0)
exog_full["IsRamadanMonth"]  = ramadan_flag

# 6) Grid‐search SARIMA parameters per family (statsmodels only)
st.write("▶️ Pre-fitting SARIMA parameters per Family (this may take a moment)…")

ps = ds = qs = range(0,2)       # 0 or 1
Ps = Ds = Qs = range(0,2)       # 0 or 1
s  = 12                         # seasonal period = 12 months

family_params = {}
for fam in family_ts.columns:
    ts_fam = family_ts[fam].reindex(hist_months, fill_value=0)
    if ts_fam.sum() < 1 or (ts_fam>0).sum() < 3:
        family_params[fam] = {
            "order":    (1,1,1),
            "seasonal": (1,1,1,12),
        }
        continue

    best_aic = np.inf
    best_cfg = None

    exog_insample = exog_full.loc[hist_months, ["FleetHours","IsRamadanMonth"]].values

    for p in ps:
        for d in ds:
            for q in qs:
                for P in Ps:
                    for D in Ds:
                        for Q in Qs:
                            order    = (p,d,q)
                            seas_ord = (P,D,Q,s)
                            try:
                                mdl = SARIMAX(
                                    ts_fam,
                                    exog=exog_insample,
                                    order=order,
                                    seasonal_order=seas_ord,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(disp=False)
                                if mdl.aic < best_aic:
                                    best_aic = mdl.aic
                                    best_cfg = (order, seas_ord)
                            except:
                                continue

    if best_cfg is not None:
        family_params[fam] = {
            "order":    best_cfg[0],
            "seasonal": best_cfg[1],
        }
    else:
        family_params[fam] = {
            "order":    (1,1,1),
            "seasonal": (1,1,1,12),
        }

st.success("✅ Family SARIMA parameters fitted.")
fam_summary = pd.DataFrame.from_dict(
    {fam: {"order": str(vals["order"]), "seasonal": str(vals["seasonal"])} for fam, vals in family_params.items()},
    orient="index"
)
st.write("Family parameters (AIC-minimized):")
st.dataframe(fam_summary)

# 7) Loop through each Part to forecast
progress_bar = st.progress(0.0)
status_text  = st.empty()

results = []
parts_list = sorted(part_ts_df["Part"].unique())
total_parts = len(parts_list)

parts_sarima = parts_naive = parts_fail = 0
n_per_family  = (
    part_ts_df[["Part","Family"]]
    .drop_duplicates()
    .groupby("Family")["Part"]
    .nunique()
    .to_dict()
)

for idx, part in enumerate(parts_list):
    progress_bar.progress((idx+1)/total_parts)
    status_text.text(f"Processing part {idx+1}/{total_parts}: {part}")

    try:
        sub = part_ts_df[part_ts_df["Part"] == part][["Month","Sales","Family"]]
        fam = sub["Family"].iloc[0]
        raw = sub.set_index("Month")[["Sales"]].sort_index()
        raw = raw.reindex(hist_months, fill_value=0)

        # “Active” start at first non-zero sale
        if (raw["Sales"]>0).any():
            start_nz = raw[raw["Sales"]>0].index.min()
            ts = raw.loc[start_nz:, "Sales"].copy()
        else:
            ts = raw["Sales"].copy()

        nz_count = (ts>0).sum()

        # Fallback if not enough data
        if (len(ts) < 12) or (nz_count < 3):
            fam_hist   = family_ts[fam].reindex(hist_months, fill_value=0)
            last3_sum  = fam_hist.iloc[-3:].sum()
            fallback   = int(round(last3_sum / max(n_per_family[fam], 1), 0))
            fc_vals    = [fallback]*12
            method     = "Naive (family avg)"
            parts_naive += 1

        else:
            # Box–Cox transform (+tiny epsilon)
            ts_pos = ts + 1e-6
            ts_bc, lam = boxcox(ts_pos)

            fam_ord    = family_params[fam]["order"]
            fam_seas   = family_params[fam]["seasonal"]
            exog_ins   = exog_full.loc[ts.index, ["FleetHours","IsRamadanMonth"]].values

            try:
                mdl = SARIMAX(
                    ts_bc,
                    exog=exog_ins,
                    order=fam_ord,
                    seasonal_order=fam_seas,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                exog_out = exog_full.loc[fc_months, ["FleetHours","IsRamadanMonth"]].values
                bc_fc    = mdl.get_forecast(steps=12, exog=exog_out).predicted_mean
                preds    = inv_boxcox(bc_fc, lam).round().astype(int)
                preds    = np.clip(preds, 0, None)

                fc_vals  = preds.tolist()
                method   = f"SARIMA (fam={fam})"
                parts_sarima += 1

            except:
                # If SARIMAX fails, fallback to last-6-month family average
                fam_hist = family_ts[fam].reindex(hist_months, fill_value=0)
                last6    = int(round(fam_hist.iloc[-6:].mean() / max(n_per_family[fam], 1), 0))
                fc_vals  = [last6]*12
                method   = "Naive (SARIMAX failed)"
                parts_naive += 1

        # Build one output row: historical + forecasts
        row = {"Item Code": part}
        for hm in hist_months:
            row[hm.strftime("%b-%Y")] = int(raw.loc[hm, "Sales"])
        for i, fm in enumerate(fc_months):
            row[fm.strftime("%b-%Y")] = int(fc_vals[i])
        row["Method"] = method
        results.append(row)

    except Exception as e:
        st.warning(f"⚠️ Error for {part}: {e}")
        parts_fail += 1
        continue

progress_bar.empty()
status_text.empty()

# 8) Summarize & produce final DataFrame
st.write("### 📊 Processing Summary")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Parts",      total_parts)
c2.metric("SARIMA Models",    parts_sarima)
c3.metric("Naive Forecasts",  parts_naive)
c4.metric("Failed",           parts_fail)

result_df = pd.DataFrame(results)
all_cols  = [m.strftime("%b-%Y") for m in (list(hist_months)+list(fc_months))]
ordered   = ["Item Code"] + [c for c in all_cols if c in result_df.columns]
final_cols = ordered + ["Method"]
result_df  = result_df[final_cols].copy()

st.success("✅ Forecasting completed!")
st.write(f"🔢 Parts processed: {len(result_df)}")

# Quick preview: last 3 historical + first 3 forecast
preview_cols = (
    ["Item Code"]
    + all_cols[-3:]
    + [m.strftime("%b-%Y") for m in fc_months[:3]]
    + ["Method"]
)
preview_cols = [c for c in preview_cols if c in result_df.columns]
st.write("**Forecast Sample (first 10 SKUs)**")
st.dataframe(result_df[preview_cols].head(10))

# 9) Download as Excel (two-row header as before)…
# (Same logic you already have to build a two-row header and color historical vs. forecast columns.)
