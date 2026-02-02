# -*- coding: utf-8 -*-
"""
Phase 3: Exploratory Data Analysis (EDA) + Clustering

This script performs:
    - Coverage checks and date ranges for macro / rich datasets
    - Distributional analysis (histograms, volatility)
    - Correlation and ACF analysis on daily log returns
    - PCA on rich and macro datasets (top 150 items)
    - KMeans clustering on PCA scores and loadings
    - Silhouette diagnostics for cluster quality
"""

##########################################################
### 0. Imports & Working Directory                     ###
##########################################################

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

import seaborn as sns
from statsmodels.tsa.stattools import acf

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

##########################################################
### 1. Data Import                                     ###
##########################################################

# Set to work for either Github level folder structure or relative path (same folder)

# Location of this script
SCRIPT_DIR = Path(__file__).resolve().parent

# GitHub-style structure 
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # adjust if needed
DATA_DIR_GITHUB = PROJECT_ROOT / "data"

DATE_TAG = "2025-09-30" # Leave as default if using original data.

macro_path_github = DATA_DIR_GITHUB / "macro" / f"macro_daily_final_{DATE_TAG}.parquet"
rich_daily_path_github = DATA_DIR_GITHUB / "rich" / f"rich_df_final_{DATE_TAG}.parquet"
rich_6h_path_github = DATA_DIR_GITHUB / "rich" / f"rich_6h_final_{DATE_TAG}.parquet"

# Fallback: local same-folder files
macro_path_local = SCRIPT_DIR / f"macro_daily_final_{DATE_TAG}.parquet"
rich_daily_path_local = SCRIPT_DIR / f"rich_df_final_{DATE_TAG}.parquet"
rich_6h_path_local = SCRIPT_DIR / f"rich_6h_final_{DATE_TAG}.parquet"

if macro_path_github.exists():
    print("Using GitHub-style data directory structure.")
    macro_df = pd.read_parquet(macro_path_github)
    rich_daily = pd.read_parquet(rich_daily_path_github)
    rich_6H = pd.read_parquet(rich_6h_path_github)
else:
    print("Using local same-folder parquet files.")
    macro_df = pd.read_parquet(macro_path_local)
    rich_daily = pd.read_parquet(rich_daily_path_local)
    rich_6H = pd.read_parquet(rich_6h_path_local)

##########################################################
### 2. Date Range Diagnostics                          ###
##########################################################

def date_range(df: pd.DataFrame, name: str) -> None:
    """
    Print basic date coverage diagnostics for a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing either 'Date' or 'Timestamp'.
    name : str
        Label used in the printed output.
    """
    # Choose appropriate time column
    time_col = "Date" if "Date" in df.columns else "Timestamp"

    # Ensure datetime
    s = pd.to_datetime(df[time_col], errors="coerce")

    # For reporting: unique day count (important for 6H data)
    unique_days = s.dt.floor("D").nunique()

    print(f"--- {name} ---")
    print(f"Using column: {time_col}")
    print("Date range:", s.min(), "→", s.max())
    print("Number of unique days (floored):", unique_days)
    print("Number of unique items:", df["Item ID"].nunique())
    print()


for name, df in [
    ("macro_df", macro_df),
    ("rich_daily", rich_daily),
    ("rich_6H", rich_6H),
]:
    date_range(df, name)

# Interpretation (from printed output):
# - macro_df:    clean 180-day window (2025-02-11 → 2025-08-10)
# - rich_daily:  pulled back to 2021 due to illiquid items
# - rich_6H:     similar long tails because of sparse trading

# The analysis window for this project is restricted to
START = pd.Timestamp("2025-02-11")
END   = pd.Timestamp("2025-08-10")

##########################################################
### 3. Coverage: Which Items Stay Within 180-Day Window ###
##########################################################

# Per-item coverage in the rich-daily dataset
coverage = (
    rich_daily.groupby(["Item ID", "Item Name"])["Date"]
    .agg(first_trade="min", last_trade="max", n_days="nunique")
    .reset_index()
)

# Flag items fully contained in the macro window
coverage["180d_range"] = (
    (coverage["first_trade"] >= START) & (coverage["last_trade"] <= END)
)

inside = coverage["180d_range"].sum()
outside = len(coverage) - inside

print(f"Total items: {len(coverage):,}")
print(f"Inside macro 180-day range: {inside:,} ({inside/len(coverage):.1%})")
print(f"Outside macro 180-day range: {outside:,} ({outside/len(coverage):.1%})")

##########################################################
### 4. Trade Start/End Histograms (Rich Daily)         ###
##########################################################

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

# Overlaid histograms of first and last trade dates
ax.hist(coverage["first_trade"], bins=60, alpha=0.6, color="C0")
ax.hist(coverage["last_trade"],  bins=60, alpha=0.6, color="C1")

# Macro window markers
ln_start = ax.axvline(START, color="tab:green", ls="--", lw=2)
ln_end   = ax.axvline(END,   color="tab:red",   ls="--", lw=2)

ax.set_title("Item First / Last Trade Distributions")
ax.set_xlabel("Date")
ax.set_ylabel("Count")

# Clean legend using proxy artists so labels are clear
handles = [
    Patch(color="C0", alpha=0.6, label="First trade — Rich Daily"),
    Patch(color="C1", alpha=0.6, label="Last trade — Rich Daily"),
    Line2D([0], [0], color="tab:green", ls="--", lw=2, label="Macro (180-day) Start"),
    Line2D([0], [0], color="tab:red",   ls="--", lw=2, label="Macro (180-day) End"),
]
ax.legend(handles=handles, loc="upper left", frameon=False)

plt.tight_layout()
plt.show()

##########################################################
### 5. Monthly Binned Coverage within Extended Window  ###
##########################################################

# Slightly extended window to give bins some visual room
START = pd.Timestamp("2025-02-01")
END   = pd.Timestamp("2025-08-30")

def month_counts_in_window(s: pd.Series) -> pd.DataFrame:
    """
    Count items by first/last trade month within the macro window.

    Parameters
    ----------
    s : pd.Series
        Series of datetime values (first_trade or last_trade).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['month', 'count'].
    """
    m = s[(s >= START) & (s <= END)].dt.to_period("M").dt.to_timestamp()
    return (
        m.value_counts()
         .sort_index()
         .rename_axis("month")
         .rename("count")
         .reset_index()
    )

first_monthly = month_counts_in_window(coverage["first_trade"])
last_monthly  = month_counts_in_window(coverage["last_trade"])

total_items = len(coverage)
first_monthly["pct"] = 100 * first_monthly["count"] / total_items
last_monthly["pct"]  = 100 * last_monthly["count"]  / total_items

# Ensure both tables share the same month index
months = pd.date_range(START, END, freq="MS")
first_monthly = (
    first_monthly.set_index("month")
    .reindex(months, fill_value=0)
    .reset_index(names="month")
)
last_monthly = (
    last_monthly.set_index("month")
    .reindex(months, fill_value=0)
    .reset_index(names="month")
)

# Side-by-side monthly bars with optional percentage overlay
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

width = 10  # days; purely visual bar width
ax.bar(first_monthly["month"] - pd.Timedelta(days=5),
       first_monthly["count"], width=width, label="First trade (count)")
ax.bar(last_monthly["month"] + pd.Timedelta(days=5),
       last_monthly["count"],  width=width, label="Last trade (count)")

# Secondary axis: percentage of all items
ax2 = ax.twinx()
ax2.plot(first_monthly["month"], first_monthly["pct"],
         marker="o", linestyle="-", label="First trade (%)")
ax2.plot(last_monthly["month"],  last_monthly["pct"],
         marker="o", linestyle="-", label="Last trade (%)")

ax.set_title("Items First / Last Seen per Month (Within Macro Window)")
ax.set_xlabel("Month")
ax.set_ylabel("Items (count)")
ax2.set_ylabel("% of all items")

# Monthly ticks
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

# Combine legends from both axes
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)

plt.tight_layout()
plt.show()

# Recompute simple inside/outside counts using extended START / END
coverage["180d_range"] = (
    (coverage["first_trade"] >= START) & (coverage["last_trade"] <= END)
)
inside = coverage["180d_range"].sum()
outside = len(coverage) - inside

print(f"Total items: {len(coverage):,}")
print(f"Inside macro 180-day range: {inside:,} ({inside/len(coverage):.1%})")
print(f"Outside macro 180-day range: {outside:,} ({outside/len(coverage):.1%})")

# Monthly distribution of first trades (for reporting)
coverage["first_month"] = coverage["first_trade"].dt.to_period("M").dt.to_timestamp()
monthly_counts = (
    coverage.loc[
        (coverage["first_trade"] >= START) &
        (coverage["first_trade"] <= END)
    ]
    .groupby("first_month")
    .size()
    .rename("count")
    .reset_index()
)
monthly_counts["pct_of_total"] = 100 * monthly_counts["count"] / len(coverage)

print("\nMonthly distribution (first trades within macro window):")
print(monthly_counts.to_string(index=False,
                               formatters={"pct_of_total": "{:.1f}%".format}))

##########################################################
### 6. Time Series Plot Helper (Daily Prices)          ###
##########################################################

def GP_EZ_Read(x, pos):
    """
    Convert raw GP values to K / M / B strings for axis labels.

    Examples
    --------
    1,500     -> '1.5K'
    2,000,000 -> '2.00M'
    1,000,000,000 -> '1.00B'
    """
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    elif x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    elif x >= 1_000:
        return f"{x/1_000:.1f}K"
    else:
        return f"{x:.0f}"


def plot_daily_price(item_name: str,
                     df_daily: pd.DataFrame,
                     log_price: bool = False) -> None:
    """
    Plot daily average prices for a single item.

    Parameters
    ----------
    item_name : str
        Item name (case-insensitive match on 'Item Name').
    df_daily : pd.DataFrame
        Daily dataset with 'Date' and 'Average Price'.
    log_price : bool, optional
        If True, uses a log y-axis for price.
    """
    data = (
        df_daily.loc[
            df_daily["Item Name"].str.casefold() == item_name.casefold(),
            ["Date", "Average Price"],
        ]
        .dropna()
        .sort_values("Date")
    )

    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    ax.plot(data["Date"], data["Average Price"],
            linewidth=1.8, color="steelblue")

    ax.set_title(f"{item_name} — Daily Average Price")
    ax.set_ylabel("Price (GP)")

    # Apply user-friendly GP formatter
    ax.yaxis.set_major_formatter(FuncFormatter(GP_EZ_Read))

    if log_price:
        ax.set_yscale("log")

    # Add a small vertical margin for visual breathing room
    ax.margins(y=0.3)

    plt.tight_layout()
    plt.show()


# Example time series plots used during EDA
plot_daily_price("Coal", macro_df, log_price=False)
plot_daily_price("Shark", macro_df, log_price=False)
plot_daily_price("Rune Platebody", macro_df, log_price=False)
plot_daily_price("Twisted Bow", macro_df, log_price=False)

##########################################################
### 7. Price / Volume Histograms (Rich Data)           ###
##########################################################

def hist_log_binned(series: pd.Series,
                    title: str,
                    bins: int = 60,
                    clip_q: float | None = None) -> None:
    """
    Histogram with logarithmically-spaced bins on the x-axis.

    Parameters
    ----------
    series : pd.Series
        Input numeric series (e.g., prices or volumes).
    title : str
        Plot title.
    bins : int, optional
        Number of log-spaced bins.
    clip_q : float or None, optional
        Upper quantile to clip extremely large values (e.g., 0.995).
    """
    s = series.dropna()
    s = s[s > 0]

    if clip_q is not None:
        s = s.clip(upper=s.quantile(clip_q))

    lo, hi = s.min(), s.max()
    edges = np.logspace(np.log10(lo), np.log10(hi), bins)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s, bins=edges)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Value (log scale)")
    ax.set_ylabel("Count")
    plt.show()


def hist_log_transformed(series: pd.Series,
                         title: str,
                         step: float = 0.1,
                         clip_q: float | None = None) -> None:
    """
    Histogram of log10-transformed values with linear bins.

    Useful when the raw scale produces a 'solid block' histogram.

    Parameters
    ----------
    series : pd.Series
        Input numeric series.
    title : str
        Plot title.
    step : float, optional
        Bin width in log10 units.
    clip_q : float or None, optional
        Upper quantile for clipping tails.
    """
    s = series.dropna()
    s = s[s > 0]

    if clip_q is not None:
        s = s.clip(upper=s.quantile(clip_q))

    x = np.log10(s)
    lo = np.floor(x.min())
    hi = np.ceil(x.max())

    bins = np.arange(lo, hi + step, step)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("log10(Value)")
    ax.set_ylabel("Count")

    ticks = np.arange(lo, hi + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"1e{int(t)}" for t in ticks])

    plt.show()


# Daily rich histograms (trimmed tails for readability)
hist_log_binned(rich_daily["Average Price"],
                "Daily Avg Price (log-binned)",
                bins=60, clip_q=0.995)
hist_log_binned(rich_daily["Total Volume"],
                "Daily Total Volume (log-binned)",
                bins=60, clip_q=0.995)

# 6H rich histograms
hist_log_transformed(rich_6H["Average Price 6H"],
                     "6H Avg Price (log10-binned)",
                     step=0.1, clip_q=0.995)
hist_log_transformed(rich_6H["Total Volume 6H"],
                     "6H Total Volume (log10-binned)",
                     step=0.1, clip_q=0.995)

##########################################################
### 8. Missingness Diagnostics (Rich 6H vs Macro)      ###
##########################################################

# 6H rich data: per-item-per-day missing fractions
df6 = rich_6H.copy()

# Normalize column names: spaces → underscores
df6.columns = [c.strip().replace(" ", "_") for c in df6.columns]

df6["Timestamp"] = pd.to_datetime(df6["Timestamp"], errors="coerce")
df6["Date"] = df6["Timestamp"].dt.floor("D")

# Choose a 6H price column from candidates
sixh_candidates = ["Avg_High_Price", "Avg_Low_Price", "Average_Price_6H"]
value_col_6h = None
for c in sixh_candidates:
    if c in df6.columns:
        value_col_6h = c
        break

if value_col_6h is None:
    raise ValueError(
        f"No valid price column found in rich_6H. Columns are: {df6.columns.tolist()}"
    )

per_item_day_6h = (
    df6.groupby(["Item_ID", "Date"])[value_col_6h]
       .agg(non_missing="count", total="size")
       .reset_index()
)
per_item_day_6h["missing_frac"] = 1 - (
    per_item_day_6h["non_missing"] / per_item_day_6h["total"]
)

median_missing_6h = per_item_day_6h["missing_frac"].median()
pct_over50_6h = (per_item_day_6h["missing_frac"] > 0.5).mean() * 100

print(f"Median missing fraction (6H rich data): {median_missing_6h:.2%}")
print(f"Item-days with >50% of 6H slots missing: {pct_over50_6h:.1f}%")

# Macro dataset missingness for comparison
print("\n=== Macro Missingness Diagnostics ===")

macro = macro_df.copy()
macro.columns = [c.strip().replace(" ", "_") for c in macro.columns]
macro["Date"] = pd.to_datetime(macro["Date"], errors="coerce")

macro_value_col = "Average_Price"
if macro_value_col not in macro.columns:
    raise ValueError(
        f"{macro_value_col} not found in macro_df columns: {macro.columns.tolist()}"
    )

macro_missing_frac = macro[macro_value_col].isna().mean()
print(f"Missing item-days in macro daily dataset: {macro_missing_frac:.2%}")

##########################################################
### 9. Volatility Computation & Comparison             ###
##########################################################

# Sort before computing returns
rich_daily = rich_daily.sort_values(by=["Item ID", "Date"])
rich_6H = rich_6H.sort_values(by=["Item ID", "Timestamp"])

# Log returns per item
rich_daily["Volatility"] = rich_daily.groupby("Item ID")["Average Price"].transform(
    lambda x: np.log(x / x.shift(1))
)
rich_6H["Volatility"] = rich_6H.groupby("Item ID")["Average Price 6H"].transform(
    lambda x: np.log(x / x.shift(1))
)

# Raw volatility histograms (diagnostic)
sns.histplot(rich_daily["Volatility"].dropna(), bins=100, kde=True)
plt.title("Volatility Distribution (Daily)")
plt.xlabel("Log Return")
plt.show()

sns.histplot(rich_6H["Volatility"].dropna(), bins=100, kde=True)
plt.title("Volatility Distribution (6H)")
plt.xlabel("Log Return")
plt.show()

# Trim tails (1% on each side) to make scale interpretable
low_d, high_d = rich_daily["Volatility"].quantile([0.01, 0.99])
low_6h, high_6h = rich_6H["Volatility"].quantile([0.01, 0.99])

# Trimmed daily volatility
plt.figure(figsize=(7, 4))
sns.histplot(rich_daily["Volatility"].dropna(), bins=100, kde=True)
plt.title("Volatility Distribution (Daily, Trimmed)")
plt.xlabel("Log Return")
plt.ylabel("Count")
plt.xlim(low_d, high_d)

# Trimmed 6H volatility
plt.figure(figsize=(7, 4))
sns.histplot(rich_6H["Volatility"].dropna(), bins=100, kde=True)
plt.title("Volatility Distribution (6H, Trimmed)")
plt.xlabel("Log Return")
plt.ylabel("Count")
plt.xlim(low_6h, high_6h)

plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.gca().yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
plt.tight_layout()
plt.show()

# Overlayed normalized histograms for daily vs 6H
plt.figure(figsize=(8, 4))

vol_d_trimmed = rich_daily["Volatility"].clip(
    lower=low_d, upper=high_d
).dropna()
vol_6h_trimmed = rich_6H["Volatility"].clip(
    lower=low_6h, upper=high_6h
).dropna()

sns.histplot(
    vol_d_trimmed,
    bins=100,
    kde=True,
    stat="density",
    label="Daily",
    color="blue",
)
sns.histplot(
    vol_6h_trimmed,
    bins=100,
    kde=True,
    stat="density",
    label="6H",
    color="orange",
)

plt.title("Volatility Distribution Comparison (Normalized)")
plt.xlabel("Log Return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

##########################################################
### 10. Correlations & ACF Heatmap (Top Items)         ###
##########################################################

# Log returns (if not already present)
rich_daily = rich_daily.sort_values(by=["Item ID", "Date"])
rich_daily["Log Return"] = rich_daily.groupby("Item ID")["Average Price"].transform(
    lambda x: np.log(x / x.shift(1))
)

# Wide format: Date × Item ID → log returns
pivot_df = rich_daily.pivot(index="Date", columns="Item ID", values="Log Return")

# Pairwise Pearson correlations for comovement analysis
correlation_matrix = pivot_df.corr()

# Focus on top 30 most traded items
top_items = rich_daily["Item ID"].value_counts().head(30).index
filtered_corr = correlation_matrix.loc[top_items, top_items]

# Flatten upper triangle to export to Excel
mask = np.triu(np.ones(filtered_corr.shape), k=1).astype(bool)
upper_tri = filtered_corr.where(mask)

stacked_df = upper_tri.stack(dropna=True).to_frame(name="Correlation")
stacked_df.index.names = ["Item ID 1", "Item ID 2"]
stacked_df = stacked_df.reset_index()

# Map Item IDs to names
id_name_map = rich_daily[["Item ID", "Item Name"]].drop_duplicates()

stacked_df = stacked_df.merge(
    id_name_map.rename(columns={"Item ID": "Item ID 1", "Item Name": "Item Name 1"}),
    on="Item ID 1",
    how="left",
)
stacked_df = stacked_df.merge(
    id_name_map.rename(columns={"Item ID": "Item ID 2", "Item Name": "Item Name 2"}),
    on="Item ID 2",
    how="left",
)

# Keep non-trivial positive correlations
stacked_df = stacked_df.query("0 < Correlation < 0.999")
stacked_df = stacked_df.sort_values(by="Correlation", ascending=False)

pd.set_option("display.max_columns", None)
stacked_df.to_excel("top_item_correlations_daily_log_returns.xlsx", index=False)

# --- ACF Heatmap for top items ---

def truncate_label(label: str, max_chars: int = 18) -> str:
    """
    Shorten long item names for plotting on the heatmap axis.
    """
    return label if len(label) <= max_chars else label[:max_chars] + "…"


max_lag = 20  # number of lags to show
acf_values = {}

for item_id in top_items:
    series = pivot_df[item_id].dropna()
    if series.empty:
        continue

    # statsmodels returns lags 0..max_lag; drop lag 0 (always 1)
    acf_vals = acf(series, nlags=max_lag, fft=True)
    acf_values[item_id] = acf_vals[1:]  # keep lags 1..max_lag only

lags = np.arange(1, max_lag + 1)
acf_df = pd.DataFrame(acf_values, index=lags)
acf_df.index.name = "Lag"

# Label columns by item name instead of ID
id_to_name = (
    rich_daily[["Item ID", "Item Name"]]
    .drop_duplicates()
    .set_index("Item ID")["Item Name"]
    .to_dict()
)
acf_df.columns = [id_to_name.get(i, str(i)) for i in acf_df.columns]

plt.figure(figsize=(12, 8), dpi=180)
ax = sns.heatmap(
    acf_df.T,
    cmap="coolwarm",
    center=0,
    vmin=-0.3,
    vmax=0.3,
    annot=False,
)

labels = [truncate_label(lbl) for lbl in acf_df.T.index]
ax.set_yticklabels(labels, rotation=0)

plt.title("ACF Heatmap of Daily Log Returns (Top Items, Lags 1–20)")
plt.xlabel("Lag")
plt.ylabel("Item")
plt.tight_layout()
plt.show()

acf_df.to_excel("top_item_acf_values.xlsx")

##########################################################
### 11. PCA on Rich Daily Data (Top 150 Items)         ###
##########################################################

# Top 150 most frequently traded items in rich_daily
top_item_ids = rich_daily["Item ID"].value_counts().head(150).index
rich_top = rich_daily[rich_daily["Item ID"].isin(top_item_ids)]

# Wide format: Date × Item ID → Average Price
pivot_prices = rich_top.pivot(
    index="Date", columns="Item ID", values="Average Price"
).sort_index()

# Fill missing prices forward/backward
pivot_prices = pivot_prices.fillna(method="ffill").fillna(method="bfill")

# Standardize item price series (mean 0, variance 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot_prices)

# PCA on rich_daily prices
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Cumulative explained variance
cumvar_rich = np.cumsum(pca.explained_variance_ratio_) * 100

plt.figure(figsize=(8, 5))
plt.plot(cumvar_rich)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("PCA on Top 150 Most Traded Items (Rich Daily Prices)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Loadings: items × PCs
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=pivot_prices.columns,
)
loadings.to_excel("pca_loadings_top150.xlsx")

# Scores: dates × PCs
scores = pd.DataFrame(
    X_pca,
    index=pivot_prices.index,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
)
scores.to_excel("pca_scores_top150.xlsx")

##########################################################
### 12. PCA on Macro Data (Top 150 Items, 180-Day)     ###
##########################################################

# Select top 150 items in macro dataset
top_item_ids_macro = macro_df["Item ID"].value_counts().head(150).index
macro_top = macro_df[macro_df["Item ID"].isin(top_item_ids_macro)]

# Wide format: Date × Item ID → Average Price
pivot_prices_macro = macro_top.pivot(
    index="Date", columns="Item ID", values="Average Price"
).sort_index()

# Fill missing values
pivot_prices_macro = pivot_prices_macro.fillna(method="ffill").fillna(method="bfill")

# Standardize
scaler_macro = StandardScaler()
X_scaled_macro = scaler_macro.fit_transform(pivot_prices_macro)

# PCA on macro prices
pca_macro = PCA()
X_pca_macro = pca_macro.fit_transform(X_scaled_macro)

# Cumulative explained variance
cumvar_macro = np.cumsum(pca_macro.explained_variance_ratio_) * 100

plt.figure(figsize=(8, 5))
plt.plot(cumvar_macro)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("PCA on Top 150 Items (Macro Daily Prices, 180-Day Window)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Macro loadings and scores
macro_loadings = pd.DataFrame(
    pca_macro.components_.T,
    columns=[f"PC{i+1}" for i in range(pca_macro.n_components_)],
    index=pivot_prices_macro.columns,
)
macro_loadings.to_excel("macro_pca_loadings_top150.xlsx")

macro_scores = pd.DataFrame(
    X_pca_macro,
    index=pivot_prices_macro.index,
    columns=[f"PC{i+1}" for i in range(pca_macro.n_components_)],
)
macro_scores.to_excel("macro_pca_scores_top150.xlsx")

##########################################################
### 13. Combined PCA Curve: Rich vs Macro              ###
##########################################################

max_k = min(len(cumvar_rich), len(cumvar_macro))
components = np.arange(1, max_k + 1)

plt.figure(figsize=(8, 5))
plt.plot(components, cumvar_rich[:max_k], label="Rich Daily (Top 150)")
plt.plot(
    components,
    cumvar_macro[:max_k],
    label="Macro (Top 150)",
    linestyle="--",
)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("PCA Explained Variance: Rich Daily vs Macro (Top 150 Items)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

##########################################################
### 14. PART A: PCA on Daily Price Scores (Regimes)    ###
##########################################################

# Reload scores from disk for clarity / reproducibility
file_path = "pca_scores_top150.xlsx"
df = pd.read_excel(file_path)

# Drop non-numeric columns (e.g., Date) before PCA
df_numeric = df.drop(columns=["Date"], errors="ignore").select_dtypes(include="number")

# PCA on scores (3-component summary of daily regimes)
pca_scores_model = PCA(n_components=3)
pca_scores = pca_scores_model.fit_transform(df_numeric)

# KMeans clustering on PCA scores (market regime clusters)
kmeans_scores = KMeans(n_clusters=4, n_init=10, random_state=42)
clusters_scores = kmeans_scores.fit_predict(pca_scores)

# PC1 vs PC2 colored by cluster
plt.figure(figsize=(8, 6))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1],
            c=clusters_scores, cmap="viridis")
plt.title("PCA Scores: Rich Daily PC1 vs PC2 with KMeans Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
# ---- ADD LEGEND ----
unique_clusters = np.unique(clusters_scores)
handles = []
cmap = plt.cm.viridis

for cluster in unique_clusters:
    handles.append(
        plt.Line2D(
            [], [], linestyle="none",
            marker="o",
            markersize=8,
            markerfacecolor=cmap(cluster / (len(unique_clusters)-1)),
            label=f"Cluster {cluster}"
        )
    )

plt.legend(handles=handles, title="KMeans Clusters")
plt.tight_layout()
plt.show()

# PC1 vs PC2 colored by PC3
plt.figure(figsize=(8, 6))
sc = plt.scatter(pca_scores[:, 0], pca_scores[:, 1],
                 c=pca_scores[:, 2], cmap="coolwarm")
plt.title("PCA Scores: Rich Daily PC1 vs PC2 Colored by PC3")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(sc, label="PC3 Value")
plt.tight_layout()
plt.show()

# PC2 vs PC3 colored by cluster
plt.figure(figsize=(8, 6))
plt.scatter(pca_scores[:, 1], pca_scores[:, 2],
            c=clusters_scores, cmap="plasma")
plt.title("PCA Scores: Rich PC2 vs PC3 with KMeans Clusters")
plt.xlabel("PC2")
plt.ylabel("PC3")
# ---- ADD LEGEND ----
unique_clusters = np.unique(clusters_scores)
handles = []
cmap = plt.cm.plasma

for cluster in unique_clusters:
    handles.append(
        plt.Line2D(
            [], [], linestyle="none",
            marker="o",
            markersize=8,
            markerfacecolor=cmap(cluster / (len(unique_clusters) - 1)),
            label=f"Cluster {cluster}"
        )
    )

plt.legend(handles=handles, title="KMeans Clusters")
plt.tight_layout()
plt.show()

##########################################################
### 15. PART B: PCA on Loadings (Item-Level Patterns)  ###
##########################################################

# Load rich PCA loadings: item IDs × PCs
df_loadings = pd.read_excel("pca_loadings_top150.xlsx", index_col=0)

# Use first 10 PCs to summarize item behavior
df_subset = df_loadings.iloc[:, :10]

# Standardize loadings prior to clustering
scaler_load = StandardScaler()
scaled_data = scaler_load.fit_transform(df_subset)

# PCA for visualization (3D factor space)
pca_vis = PCA(n_components=3)
pca_scores_load = pca_vis.fit_transform(scaled_data)

# KMeans clustering: item-level sectors (rich data)
kmeans_load = KMeans(n_clusters=4, n_init=10, random_state=42)
clusters_load = kmeans_load.fit_predict(pca_scores_load)

# Plot: PC1 vs PC2 by cluster
plt.figure(figsize=(8, 6))
plt.scatter(
    pca_scores_load[:, 0],
    pca_scores_load[:, 1],
    c=clusters_load,
    cmap="viridis",
    s=50,
)
plt.title("PCA Loadings (Top 150 Items): PC1 vs PC2 with KMeans Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# Plot: PC2 vs PC3 by cluster
plt.figure(figsize=(8, 6))
plt.scatter(
    pca_scores_load[:, 1],
    pca_scores_load[:, 2],
    c=clusters_load,
    cmap="plasma",
    s=50,
)
plt.title("PCA Loadings (Rich) PC2 vs PC3 with KMeans Clusters")
plt.xlabel("PC2")
plt.ylabel("PC3")
plt.tight_layout()
plt.show()

# Save clustered loadings in factor space
results_df = pd.DataFrame(
    pca_scores_load,
    columns=["PC1", "PC2", "PC3"],
    index=df_loadings.index,
)
results_df["Cluster"] = clusters_load
results_df.to_excel("pca_clustered_items_top150.xlsx")

##########################################################
### 16. PART C: Highlight Standout Items (Rich PCA)    ###
##########################################################

# Reload PCA loadings (items × PCs)
loadings_df = pd.read_excel(
    "C:/Users/Drew/Desktop/Data Sci/DATA599/Modeling Scripts/pca_loadings_top150.xlsx",
    index_col=0,
)

# Map Item ID → Item Name from rich daily dataset
rich_df_subset = pd.read_excel(
    "C:/Users/Drew/Desktop/Data Sci/DATA599/Modeling Scripts/rich_df_final_2025-09-29.xlsx"
)
id_name_map = (
    rich_df_subset[["Item ID", "Item Name"]]
    .drop_duplicates()
    .set_index("Item ID")
)

# Take first 10 PCs for standout analysis
df_subset = loadings_df.iloc[:, :10]

# Scale loadings
scaler_standout = StandardScaler()
scaled_data = scaler_standout.fit_transform(df_subset)

# PCA reduction for plotting
pca_vis_standout = PCA(n_components=3)
pca_scores_standout = pca_vis_standout.fit_transform(scaled_data)

# KMeans clustering in this reduced space
kmeans_standout = KMeans(n_clusters=4, n_init=10, random_state=42)
clusters_standout = kmeans_standout.fit_predict(pca_scores_standout)

# Build results DF for rich standout plot
results_df = pd.DataFrame(
    pca_scores_standout,
    columns=["PC1", "PC2", "PC3"],
    index=loadings_df.index,
)
results_df["Cluster"] = clusters_standout

# Identify standout items by extremes of PC1 and PC2
standout_items = pd.concat(
    [
        results_df.nlargest(10, "PC1"),
        results_df.nsmallest(10, "PC1"),
        results_df.nlargest(10, "PC2"),
        results_df.nsmallest(10, "PC2"),
    ]
).drop_duplicates()

# Attach names
standout_items = standout_items.join(id_name_map, how="left")

plt.figure(figsize=(12, 9))
plt.scatter(
    results_df["PC1"],
    results_df["PC2"],
    c=results_df["Cluster"],
    cmap="viridis",
    s=50,
)
for idx, row in standout_items.iterrows():
    label = row["Item Name"] if pd.notna(row["Item Name"]) else str(idx)
    plt.text(row["PC1"], row["PC2"], label, fontsize=12, ha="center")

plt.title(
    "PCA Loadings: Rich Daily PC1 vs PC2 with KMeans Clusters and Item Names"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

##########################################################
### 17. Macro Item-Level Sectors via PCA + KMeans      ###
##########################################################

# Use first 10 macro PCs for clustering
macro_df_subset = macro_loadings.iloc[:, :10]

# Standardize macro loading vectors
scaler_macro_clust = StandardScaler()
macro_scaled = scaler_macro_clust.fit_transform(macro_df_subset)

# PCA for visualization (macro items)
pca_macro_vis = PCA(n_components=3)
macro_pca_scores_vis = pca_macro_vis.fit_transform(macro_scaled)

# KMeans clustering to create macro sectors
kmeans_macro = KMeans(n_clusters=4, n_init=10, random_state=42)
macro_clusters = kmeans_macro.fit_predict(macro_pca_scores_vis)

# Build macro sector results (index = Item ID)
macro_cluster_results = pd.DataFrame(
    macro_pca_scores_vis,
    index=macro_df_subset.index,
    columns=["PC1", "PC2", "PC3"],
)
macro_cluster_results["Cluster"] = macro_clusters

# Map Item ID → Item Name
macro_id_name_map = (
    macro_df[["Item ID", "Item Name"]]
    .drop_duplicates()
    .set_index("Item ID")["Item Name"]
)
macro_cluster_results["Item Name"] = macro_cluster_results.index.map(
    macro_id_name_map
)

macro_cluster_results.to_excel("macro_pca_clustered_items_top150.xlsx")

# Macro PC1 vs PC2 by cluster
plt.figure(figsize=(8, 6))
plt.scatter(
    macro_cluster_results["PC1"],
    macro_cluster_results["PC2"],
    c=macro_cluster_results["Cluster"],
    cmap="viridis",
    s=50,
)
plt.title("Macro PCA Loadings (Top 150 Items): PC1 vs PC2 with KMeans Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# Highlight standout macro items by extremes on PC1 and PC2
standout_macro = pd.concat(
    [
        macro_cluster_results.nlargest(10, "PC1"),
        macro_cluster_results.nsmallest(10, "PC1"),
        macro_cluster_results.nlargest(10, "PC2"),
        macro_cluster_results.nsmallest(10, "PC2"),
    ]
).drop_duplicates()

plt.figure(figsize=(12, 9))
plt.scatter(
    macro_cluster_results["PC1"],
    macro_cluster_results["PC2"],
    c=macro_cluster_results["Cluster"],
    cmap="viridis",
    s=50,
)

for idx, row in standout_macro.iterrows():
    label = row["Item Name"] if pd.notna(row["Item Name"]) else str(idx)
    plt.text(row["PC1"], row["PC2"], label, fontsize=14, ha="center")

plt.title(
    "Macro PCA Loadings: PC1 vs PC2 with KMeans Clusters and Item Names",
    fontsize=18,
)
plt.xlabel("PC1", fontsize=14)
plt.ylabel("PC2", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

##########################################################
### 18. Labelled Loading Plots (Rich + Macro)          ###
##########################################################

def truncate_label(label, max_chars: int = 25) -> str:
    """
    Shorten long item names for plotting in scatter-label plots.
    """
    if not isinstance(label, str):
        label = str(label)
    return label if len(label) <= max_chars else label[:max_chars] + "…"


def plot_loadings_with_labels(
    loading_df: pd.DataFrame,
    standout_df: pd.DataFrame,
    title: str,
    cmap: str = "viridis",
) -> None:
    """
    Scatter plot of PC1 vs PC2 with cluster coloring and text labels
    for a subset of 'standout' items.

    Parameters
    ----------
    loading_df : pd.DataFrame
        DataFrame with columns ['PC1', 'PC2', 'Cluster'].
    standout_df : pd.DataFrame
        Subset of loading_df to label; expected to include 'Item Name'.
    title : str
        Plot title.
    cmap : str, optional
        Matplotlib colormap for cluster coloring.
    """
    plt.figure(figsize=(14, 9), dpi=180)

    # Base scatter for all items
    sc = plt.scatter(
        loading_df["PC1"],
        loading_df["PC2"],
        c=loading_df["Cluster"],
        cmap=cmap,
        s=50,
        alpha=0.85,
    )

    # Spread labels radially around their points to reduce overlap
    n_labels = len(standout_df)
    for i, (idx, row) in enumerate(standout_df.iterrows()):
        label = row.get("Item Name", str(idx))
        label = truncate_label(label, max_chars=25)

        angle = 4.25 * np.pi * i / max(n_labels, 1)
        dx = 0.25 * np.cos(angle)
        dy = 0.25 * np.sin(angle)

        plt.text(
            row["PC1"] + dx,
            row["PC2"] + dy,
            label,
            fontsize=14,
            ha="center",
            va="center",
        )

    plt.title(title, fontsize=20)
    plt.xlabel("PC1", fontsize=14)
    plt.ylabel("PC2", fontsize=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()


# Rich labelled plot
plot_loadings_with_labels(
    loading_df=results_df,
    standout_df=standout_items,
    title="Rich Daily PCA Loadings: PC1 vs PC2 with KMeans Clusters and Item Names",
)

# Macro labelled plot
plot_loadings_with_labels(
    loading_df=macro_cluster_results,
    standout_df=standout_macro,
    title="Macro PCA Loadings: PC1 vs PC2 with KMeans Clusters and Item Names",
)

# Quick sanity check on abundance of 'Raw ...' items
macro_top["Item Name"].str.startswith("Raw").mean()
# ~35% of top 150 macro items start with 'Raw', reflecting OSRS naming scheme rather than a data filtering issue.

##########################################################
### 19.  Regex/Classification Attempt                  ###
##########################################################

def classify_item_name(name: str) -> str:
    """
    Very rough OSRS sector tagging from item name.
    Adjust / extend patterns as needed.
    """
    if not isinstance(name, str):
        return "Unknown"

    n = name.lower()

    # Equipment
    if any(word in n for word in [
        "sword", "bow", "dagger", "axe", "scimitar", "mace", "spear",
        "staff", "wand", "helm", "helmet", "platebody", "platelegs",
        "chainbody", "shield", "boots", "gloves", "ring", "amulet",
        "body", "chaps", "robe", "cape"
    ]):
        return "Equipment"

    # Food / potions
    if any(word in n for word in [
        "potion", "brew", "overload", "tea", "stew", "pie", "pizza",
        "cake", "wine", "shark", "monkfish", "lobster", "tuna",
        "salmon", "trout", "karambwan", "bass"
    ]):
        return "Food/Potion"

    # Resources / skilling mats
    if any(word in n for word in [
        "ore", "bar", "log", "plank", "herb", "seed", "fish",
        "raw", "bone", "hide", "essence", "rune", "bolt", "arrow",
        "scale", "leaf", "sapling", "gem"
    ]):
        return "Resource"
    # Fallback
    return "Other"
macro_cluster_results["Sector"] = macro_cluster_results["Item Name"].apply(classify_item_name)

# Raw counts
cluster_sector_counts = pd.crosstab(
    macro_cluster_results["Cluster"],
    macro_cluster_results["Sector"]
)

# Row-normalised (percentage per cluster)
cluster_sector_props = pd.crosstab(
    macro_cluster_results["Cluster"],
    macro_cluster_results["Sector"],
    normalize="index"
)

print(cluster_sector_counts)
print(cluster_sector_props.round(2))
cluster_sector_counts.to_excel("cluster_sector_counts.xlsx"); cluster_sector_props.round(2).to_excel("cluster_sector_props.xlsx")
##########################################################
### 20. Silhouette Scores (Rich vs Macro Clusters)     ###
##########################################################

# Silhouette scores for Rich Daily PCA scores (regime clustering)
sil_scores_rich = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = km.fit_predict(pca_scores)
    score_k = silhouette_score(pca_scores, labels_k)
    sil_scores_rich.append({"K": k, "Silhouette Score": score_k})

silhouette_rich_df = pd.DataFrame(sil_scores_rich)
print("\nSilhouette scores for Rich Daily PCA scores:")
print(silhouette_rich_df)
silhouette_rich_df.to_excel("silhouette_scores_rich_pca_scores.xlsx", index=False)

# Silhouette scores for Macro PCA loadings (item sectors)
sil_scores_macro = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = km.fit_predict(macro_pca_scores_vis)
    score_k = silhouette_score(macro_pca_scores_vis, labels_k)
    sil_scores_macro.append({"K": k, "Silhouette Score": score_k})

silhouette_macro_df = pd.DataFrame(sil_scores_macro)
print("\nSilhouette scores for Macro PCA loadings (PC1–PC3 space):")
print(silhouette_macro_df)
silhouette_macro_df.to_excel("silhouette_scores_macro_loadings.xlsx", index=False)

