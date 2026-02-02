# -*- coding: utf-8 -*-
"""
Phase 4 & 5: Data Partitioning, Factor Modeling (VAR/FAVAR),
and Neural Baselines (LSTM / GRU)

Notes
-----
- This script assumes an environment compatible with TensorFlow-based
  models (GRU / LSTM), e.g.:

    * Python 3.10.19
    * tensorflow 3.12.1

This script performs:
    
    1. Partition rich-daily returns into TRAIN / VAL / TEST windows.
    2. Apply split-safe interpolation + TRAIN-mean imputation.
    3. Fit PCA on TRAIN-only standardized returns to obtain factors.
    4. Configure a VAR model on factors (FAVAR) in either:
         - "breadth" mode (high K, p = 1)
         - "depth"   mode (lower K, p = 2)
    5. Produce rolling 1-step-ahead forecasts on VAL and TEST.
    6. Back-project factor forecasts into item space and compute errors.
    7. Train LSTM / GRU neural baselines on top-50 liquid items.
    8. Run a structural-break (Bai–Perron-style) analysis on log prices.
"""

##########################################################
### 0. Imports & Global Settings                       ###
##########################################################

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import ruptures as rpt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.tsa_model import ValueWarning
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

import warnings
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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
### 2. Configuration                                   ###
##########################################################

# VAR configuration mode:
#   "breadth" → higher K to capture more variance (p=1)
#   "depth"   → lower K to allow a richer lag structure (p=2)
MODE = "breadth"          # "breadth" or "depth"

# Target explained variance for factor selection
BREADTH_VAR_TARGET = 0.90 # ~90% variance, p = 1
DEPTH_VAR_TARGET   = 0.80 # ~80% variance, p = 2
DEPTH_P            = 2    # Max lag for DEPTH mode

##########################################################
### 3. Phase 4: Windowing & Return Construction        ###
##########################################################

# Macro analysis window (consistent with earlier phases)
START = pd.Timestamp("2025-02-11")
END   = pd.Timestamp("2025-08-10")

# Restrict rich-daily data to the macro window (no hard liquidity filters)
rdw = rich_daily.query("Date >= @START and Date <= @END").copy()

# Wide panel of prices → log returns
prices_w = (
    rdw.pivot(index="Date", columns="Item ID", values="Average Price")
       .sort_index()
)
prices_w.index = pd.to_datetime(prices_w.index)

# Simple log-difference returns (NaN in first row per column is expected)
rets_w = np.log(prices_w).diff()

##########################################################
### 4. Train / Val / Test Split                        ###
##########################################################

# Fixed calendar-based split
train_start = pd.Timestamp("2025-05-01")
train_end   = pd.Timestamp("2025-07-25")
val_end     = pd.Timestamp("2025-08-01")  # TEST = 2025-08-02..2025-08-10

X_train = rets_w.loc[train_start:train_end].copy()
X_val   = rets_w.loc[train_end + pd.Timedelta(days=1):val_end].copy()
X_test  = rets_w.loc[val_end   + pd.Timedelta(days=1):END].copy()

##########################################################
### 5. Split-Safe Imputation (Soft Gap Handling)       ###
##########################################################

def impute_block(block: pd.DataFrame, train_means: pd.Series) -> pd.DataFrame:
    """
    Perform time-based interpolation within a split, then
    fill remaining gaps with TRAIN-set means.

    Parameters
    ----------
    block : pd.DataFrame
        Submatrix (TRAIN / VAL / TEST) with datetime index.
    train_means : pd.Series
        Column-wise means computed on TRAIN only.

    Returns
    -------
    pd.DataFrame
        Imputed block with time interpolation + mean fill.
    """
    b = block.copy()

    if not isinstance(b.index, pd.DatetimeIndex):
        b.index = pd.to_datetime(b.index)

    b = b.sort_index()
    b = b.interpolate(method="time", limit_direction="both", axis=0)

    aligned_means = train_means.reindex(b.columns)
    b = b.fillna(aligned_means)

    return b

# TRAIN-based means for soft filling
train_means = X_train.mean(skipna=True)

X_train_i = impute_block(X_train, train_means)
X_val_i   = impute_block(X_val,   train_means)
X_test_i  = impute_block(X_test,  train_means)

# Defensive column drops:
# 1) Remove columns that are entirely NaN after imputation.
non_allnan = X_train_i.notna().any(axis=0)
X_train_i  = X_train_i.loc[:, non_allnan]
X_val_i    = X_val_i.loc[:, non_allnan]
X_test_i   = X_test_i.loc[:, non_allnan]

# 2) Remove (near) constant columns to avoid singularities.
eps = 1e-12
non_const = X_train_i.std(ddof=0) > eps
X_train_i = X_train_i.loc[:, non_const]
X_val_i   = X_val_i.loc[:, non_const]
X_test_i  = X_test_i.loc[:, non_const]

print("After soft filters (TRAIN days, features):", X_train_i.shape)

##########################################################
### 6. Phase 5A: PCA → Factors (TRAIN-Only Fit)        ###
##########################################################

# Standardize TRAIN / VAL / TEST using TRAIN fit only
scaler = StandardScaler()
Z_tr = pd.DataFrame(
    scaler.fit_transform(X_train_i),
    index=X_train_i.index,
    columns=X_train_i.columns,
)
Z_va = pd.DataFrame(
    scaler.transform(X_val_i),
    index=X_val_i.index,
    columns=X_val_i.columns,
)
Z_te = pd.DataFrame(
    scaler.transform(X_test_i),
    index=X_test_i.index,
    columns=X_test_i.columns,
)

# PCA on standardized TRAIN returns
pca = PCA()
pca.fit(Z_tr)
cumvar = np.cumsum(pca.explained_variance_ratio_)

def k_for_target(cumvar_arr: np.ndarray, target: float, max_k: int) -> int:
    """
    Choose the smallest K such that cumulative variance ≥ target,
    constrained by max_k and K ≥ 1.
    """
    if len(cumvar_arr) == 0:
        return 1
    k = int(np.searchsorted(cumvar_arr, target) + 1)
    return max(1, min(k, max_k))

K_90 = k_for_target(cumvar, BREADTH_VAR_TARGET, Z_tr.shape[1])
K_80 = k_for_target(cumvar, DEPTH_VAR_TARGET,   Z_tr.shape[1])

print(
    f"Top-K(90%) guess = {K_90} ({cumvar[K_90-1]*100:.1f}%),  "
    f"Top-K(80%) guess = {K_80} ({cumvar[K_80-1]*100:.1f}%)"
)

# Full factor series using TRAIN-only PCA fit
F_tr_full = pd.DataFrame(pca.transform(Z_tr), index=Z_tr.index)
F_va_full = pd.DataFrame(pca.transform(Z_va), index=Z_va.index)
F_te_full = pd.DataFrame(pca.transform(Z_te), index=Z_te.index)

cols = [f"PC{i+1}" for i in range(F_tr_full.shape[1])]
for _df in (F_tr_full, F_va_full, F_te_full):
    _df.columns = cols

def pmax_feasible(T: int, K: int) -> int:
    """
    Feasible maximum VAR lag p for T observations and K variables,
    using a simple (T - 1) > p(K + 1) rule-of-thumb.
    """
    return max(1, int((T - 1) // (K + 1)))

def to_daily_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Warp factor time index to a regular daily grid and fill gaps.
    This ensures VAR sees a regular time series in 'D' frequency.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    out = df.reindex(full)
    out.index.freq = "D"
    return out

# Force daily frequency for all factor panels
F_tr_full = to_daily_grid(F_tr_full).ffill().bfill()
F_va_full = to_daily_grid(F_va_full).ffill().bfill()
F_te_full = to_daily_grid(F_te_full).ffill().bfill()

# MODE-dependent choice of K and p
if MODE == "depth":
    T_train_full = len(F_tr_full)
    K_cap = max(1, int((T_train_full - 1) // DEPTH_P - 1))
    K = min(K_80, K_cap)
    p = min(DEPTH_P, pmax_feasible(T_train_full, K))
    mode_msg = (
        f"[Config] DEPTH mode: target var≈{DEPTH_VAR_TARGET*100:.0f}%, "
        f"p={DEPTH_P} → using K={K}, p={p}"
    )
else:
    K = K_90
    p = 1
    mode_msg = (
        f"[Config] BREADTH mode: target var≈{BREADTH_VAR_TARGET*100:.0f}%, "
        f"using K={K}, p={p}"
    )

print(mode_msg, f"(cumvar@K={cumvar[K-1]*100:.1f}%)")

# Truncate to K leading PCs
F_tr = F_tr_full.iloc[:, :K].copy()
F_va = F_va_full.iloc[:, :K].copy()
F_te = F_te_full.iloc[:, :K].copy()

T_train = len(F_tr)
print(f"T_train={T_train}, feasible p≤{pmax_feasible(T_train, K)}")

##########################################################
### 7. VAR Fit & Rolling Forecasts (VAL / TEST)        ###
##########################################################

def rolling_forecast(F_train: pd.DataFrame,
                     F_future: pd.DataFrame,
                     p: int) -> pd.DataFrame:
    """
    Rolling one-step-ahead forecast in factor space using VAR(p).

    At each forecast date:
        - re-fit VAR on history (TRAIN + realized subset of future)
        - forecast 1-step ahead

    Parameters
    ----------
    F_train : pd.DataFrame
        Initial factor history (TRAIN).
    F_future : pd.DataFrame
        Future factor path (VAL or TEST).
    p : int
        VAR lag order.

    Returns
    -------
    pd.DataFrame
        Predicted factors aligned with F_future index.
    """
    preds = []
    history = F_train.copy()

    for t in F_future.index:
        res = VAR(history).fit(p)
        f1 = res.forecast(history.values[-p:], steps=1)[0]
        preds.append(pd.Series(f1, index=history.columns, name=t))
        history = pd.concat([history, F_future.loc[[t]]])

    return pd.DataFrame(preds)

# Validation forecasts
Fhat_va = rolling_forecast(F_tr, F_va, p)
mae_va  = (Fhat_va - F_va).abs().mean().median()
rmse_va = ((Fhat_va - F_va)**2).mean().pow(0.5).median()
print("VAL median MAE (factors):", mae_va,
      "| VAL median RMSE (factors):", rmse_va)

# Test forecasts (TRAIN+VAL history)
F_trva  = pd.concat([F_tr, F_va])
Fhat_te = rolling_forecast(F_trva, F_te, p)
mae_te  = (Fhat_te - F_te).abs().mean().median()
rmse_te = ((Fhat_te - F_te)**2).mean().pow(0.5).median()
print("TEST median MAE (factors):", mae_te,
      "| TEST median RMSE (factors):", rmse_te)

##########################################################
### 8. Diagnostics (Dimensions & Config Summary)       ###
##########################################################

print("Train days × features:", X_train_i.shape)
print("Val   days × features:", X_val_i.shape)
print("Test  days × features:", X_test_i.shape)
print(
    f"[Final VAR config] K={K}, p={p}, T_train={T_train}, "
    f"cumvar@K={cumvar[K-1]*100:.1f}%"
)

##########################################################
### 9. Back-Projection to Item Returns (VAL / TEST)    ###
##########################################################

# PCA loading matrix (columns = PCs)
V  = pca.components_.T
mu = scaler.mean_
sd = np.sqrt(scaler.var_)
V_K = V[:, :K]

# Factor forecasts → standardized returns
Zhat_va = Fhat_va.values @ V_K.T
Zhat_te = Fhat_te.values @ V_K.T

cols_items = X_train_i.columns

Rhat_va = pd.DataFrame(
    Zhat_va * sd + mu,
    index=Fhat_va.index,
    columns=cols_items,
)
Rhat_te = pd.DataFrame(
    Zhat_te * sd + mu,
    index=Fhat_te.index,
    columns=cols_items,
)

# Align actual returns on VAL / TEST to the same dates
R_va = X_val_i.loc[Rhat_va.index, cols_items]
R_te = X_test_i.loc[Rhat_te.index, cols_items]

# Item-level error metrics
mae_item_va  = (Rhat_va - R_va).abs().median()
rmse_item_va = ((Rhat_va - R_va)**2).mean()**0.5

mae_item_te  = (Rhat_te - R_te).abs().median()
rmse_item_te = ((Rhat_te - R_te)**2).mean()**0.5

print(
    "VAL item-space: median(MAE)=",
    mae_item_va.median(),
    " median(RMSE)=",
    rmse_item_va.median(),
)
print(
    "TEST item-space: median(MAE)=",
    mae_item_te.median(),
    " median(RMSE)=",
    rmse_item_te.median(),
)

print("Worst 5 (VAL MAE):")
print(mae_item_va.sort_values(ascending=False).head(5))
print("Best 5 (VAL MAE):")
print(mae_item_va.sort_values().head(5))

##########################################################
### 10. Final VAR(1) Fit for Reporting (TEST)          ###
##########################################################

# For the final reporting run, fix p = 1 and
# refit VAR on TRAIN+VAL factors, then forecast TEST.
F_trainval = pd.concat([F_tr, F_va])
print(f"Final fit sample: {F_trainval.shape[0]} days × {F_trainval.shape[1]} factors")

res = VAR(F_trainval).fit(1)

hist = F_trainval.copy()
preds = []
for t in F_te.index:
    f_next = res.forecast(hist.values[-1:], steps=1)[0]
    preds.append(pd.Series(f_next, index=hist.columns, name=t))
    hist = pd.concat([hist, F_te.loc[[t]]])

Fhat_te = pd.DataFrame(preds)
print(f"Forecasted factors for {len(Fhat_te)} test days.")

# Back-project to item returns with fixed K
V  = pca.components_.T[:, :K]
mu = scaler.mean_
sd = np.sqrt(scaler.var_)

Zhat_te = Fhat_te.values @ V.T
cols_items = X_train_i.columns
Rhat_te = pd.DataFrame(
    Zhat_te * sd + mu,
    index=Fhat_te.index,
    columns=cols_items,
)

R_te = X_test_i.loc[Rhat_te.index, cols_items]

mae_item  = (Rhat_te - R_te).abs().median()
rmse_item = ((Rhat_te - R_te)**2).mean()**0.5

print("=== Out-of-Sample Test Results ===")
print(f"Median Item MAE : {mae_item.median():.6f}")
print(f"Median Item RMSE: {rmse_item.median():.6f}")

print("\nTop 5 Best Forecasted Items:")
print(mae_item.nsmallest(5))
print("\nTop 5 Worst Forecasted Items:")
print(mae_item.nlargest(5))

##########################################################
### 11. Interpretability: Add Item Names & Export      ###
##########################################################

# Map item IDs to names from rich_daily
name_map = (
    rich_daily[["Item ID", "Item Name"]]
    .drop_duplicates()
    .set_index("Item ID")["Item Name"]
)

def with_names(series: pd.Series, value_col_name: str = "MAE") -> pd.DataFrame:
    """
    Attach item names to a per-item metric series (e.g., MAE or RMSE).
    """
    df = series.to_frame(value_col_name)
    df.insert(0, "Item Name", df.index.map(name_map).fillna("«unknown»"))
    return df

best10_named  = with_names(mae_item.nsmallest(10))
worst10_named = with_names(mae_item.nlargest(10))

print("\nTop 10 Best Forecasted Items (TEST, MAE):")
print(best10_named)
print("\nTop 10 Worst Forecasted Items (TEST, MAE):")
print(worst10_named)

# Persist per-item errors for external analysis
with_names(mae_item).sort_values("MAE").to_csv(
    "var_test_item_mae.csv", index_label="Item ID"
)
with_names(rmse_item, value_col_name="RMSE").sort_values("RMSE").to_csv(
    "var_test_item_rmse.csv", index_label="Item ID"
)

##########################################################
### 12. Helper Functions: y-padding & GP Formatter     ###
##########################################################

def _ypads(y_a: pd.Series,
           y_f: pd.Series,
           frac: float = 0.25) -> tuple[float, float]:
    """
    Compute symmetric y-limits around the combined range of
    actual and forecast series, with a fractional padding.
    """
    y_all = pd.concat([y_a, y_f], axis=0)
    y_min, y_max = float(y_all.min()), float(y_all.max())
    rng = y_max - y_min
    pad = max(1e-6, frac * rng) if rng > 0 else 1e-6
    return y_min - pad, y_max + pad

def GP_EZ_Read(x, pos):
    """
    Format GP values in K / M / B notation for plot axes.
    """
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    elif x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    elif x >= 1_000:
        return f"{x/1_000:.1f}K"
    else:
        return f"{x:.0f}"

##########################################################
### 13. Price-Level Reconstruction for TEST            ###
##########################################################

# Reconstruct GP price levels from predicted returns on TEST
prices_w_ffill = prices_w.sort_index().ffill()

test_dates = Rhat_te.index
t0 = test_dates.min() - pd.Timedelta(days=1)  # base day for cumprod
cols_items = X_train_i.columns

base_prices = prices_w_ffill.loc[t0, cols_items]
cum_fore = np.exp(Rhat_te).cumprod()
price_forecast_te = cum_fore.mul(base_prices, axis=1)

price_actual_te = prices_w_ffill.loc[test_dates, cols_items]

def plot_item_returns(iid: int,
                      y_min: float | None = None,
                      y_max: float | None = None,
                      pad_frac: float = 0.3) -> None:
    """
    Plot test-period actual vs forecasted returns for a single item.
    """
    nm = name_map.get(iid, "«unknown»")
    y_a = R_te[iid]
    y_f = Rhat_te[iid]

    ylo, yhi = _ypads(y_a, y_f, frac=pad_frac)
    if y_min is not None:
        ylo = y_min
    if y_max is not None:
        yhi = y_max

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    ax.plot(R_te.index, y_a, label="Actual")
    ax.plot(R_te.index, y_f, "--", label="Forecast")

    ax.set_ylim(ylo, yhi)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=25)

    ax.set_title(
        f"{nm} (ID {iid}) | MAE={mae_item[iid]:.3f}, RMSE={rmse_item[iid]:.3f}"
    )
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def plot_item_prices(iid: int,
                     y_min: float | None = None,
                     y_max: float | None = None,
                     pad_frac: float = 0.05) -> None:
    """
    Plot test-period actual vs forecasted GP price levels for one item.
    """
    nm = name_map.get(iid, "«unknown»")
    y_a = price_actual_te[iid]
    y_f = price_forecast_te[iid]

    ylo, yhi = _ypads(y_a, y_f, frac=pad_frac)
    if y_min is not None:
        ylo = y_min
    if y_max is not None:
        yhi = y_max

    fig, ax = plt.subplots(figsize=(8, 3), dpi=130)
    ax.plot(y_a.index, y_a, label="Actual GP")
    ax.plot(y_f.index, y_f, "--", label="Forecast GP")

    ax.set_ylim(ylo, yhi)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45)

    ax.yaxis.set_major_formatter(FuncFormatter(GP_EZ_Read))
    ax.set_title(f"{nm} (ID {iid}) — Price Level")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

# Example plots used in the paper (fixed-scale and free-scale variants)
plot_item_prices(22305, y_min=15_800_000,    y_max=16_000_000)
plot_item_prices(1319,  y_min=30_000,    y_max=50_000)
plot_item_returns(22305, y_min=-0.025, y_max=0.025)
plot_item_prices(31111, y_min=0, y_max=400)
plot_item_prices(31115)
plot_item_prices(26382)
plot_item_prices(20997, y_min=1_550_000_000, y_max=1_625_000_000)
plot_item_prices(30753, y_min=50_000_000, y_max=100_000_000)
plot_item_prices(22810, y_min=0, y_max=26_000)
plot_item_prices(2581)
plot_item_prices(22978)

# “Good” vs “hard” example items for qualitative illustration
pick_good = [22305, 1319]
pick_hard = [3769, 4319]
for iid in (pick_good + pick_hard):
    plot_item_prices(iid)

##########################################################
### 14. VAR Item-Level MAE Histogram & Coverage CDF    ###
##########################################################

mae_vals = mae_item.values

def plot_var_mae_hist(mae_vals: np.ndarray,
                      clip: float = 1.0,
                      bins: int = 60) -> None:
    """
    Histogram of item-level MAE values, clipped at a threshold
    for readability, with a note on excluded tails.
    """
    bulk = mae_vals[mae_vals <= clip]
    tail_count = int((mae_vals > clip).sum())
    med = float(np.median(bulk))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.hist(bulk, bins=bins, range=(0, clip),
            edgecolor="white", color="steelblue")

    ax.axvline(med, color="red", linestyle="--", lw=3)

    label_text = (
        f"Median = {med:.3f}  |  Excludes {tail_count} items > {clip}"
    )
    ax.legend([label_text], loc="upper right",
              frameon=False, fontsize=11)

    ax.set_xlim(0, clip)
    ax.set_xlabel("MAE", fontsize=13)
    ax.set_ylabel("Item count", fontsize=13)
    ax.set_title(
        "Distribution of Item-level MAE (VAR, test)\n"
        "(x-axis clipped at 1.0)",
        fontsize=14,
        pad=10,
    )
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.2, linestyle=":")
    plt.tight_layout()
    plt.show()

plot_var_mae_hist(mae_item.values)

# Coverage CDF: fraction of items with MAE ≤ threshold
sorted_mae = np.sort(mae_vals)
plt.figure(dpi=130)
plt.plot(sorted_mae, np.arange(len(sorted_mae)) / len(sorted_mae))
plt.xlabel("MAE threshold")
plt.ylabel("Fraction of items ≤ threshold")
plt.title("Forecast Accuracy Coverage (VAR)")
plt.grid(True, ls=":", alpha=0.5)
plt.tight_layout()
plt.show()

##########################################################
### 15. Liquidity vs Forecast Error                    ###
##########################################################

# Average volume (TRAIN window) as a liquidity proxy
liq_map = (
    rich_daily
    .query("Date >= @train_start and Date <= @train_end")
    .groupby("Item ID")["Total Volume"]
    .mean()
)

mae_liq = (
    pd.DataFrame({"MAE": mae_item})
    .join(liq_map.rename("Avg Volume"))
    .dropna()
)

plt.figure(figsize=(5, 4), dpi=130)
plt.scatter(mae_liq["Avg Volume"], mae_liq["MAE"], alpha=0.45)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Average Volume (log)")
plt.ylabel("MAE (log)")
plt.title("Forecast Error vs Liquidity (VAR, Test)")
plt.grid(True, ls=":", alpha=0.5)
plt.tight_layout()
plt.show()

##########################################################
### 16. VAR Residual Diagnostics                       ###
##########################################################

# Residual ACF/PACF and Ljung–Box tests on a subset of factors
resid = res.resid
check_cols = resid.columns[:6]

for c in check_cols:
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=120)
    plot_acf(resid[c].dropna(), lags=10, ax=ax[0])
    plot_pacf(resid[c].dropna(), lags=10, method="ywm", ax=ax[1])
    ax[0].set_title(f"{c} Residual ACF")
    ax[1].set_title(f"{c} Residual PACF")
    plt.tight_layout()
    plt.show()

for c in check_cols:
    lb = acorr_ljungbox(resid[c].dropna(), lags=[10], return_df=True)
    print(f"{c}: Ljung–Box p(10) = {lb['lb_pvalue'].iloc[0]:.3f}")

##########################################################
### 17. SECTION 2: LSTM and GRU Modeling               ###
##########################################################

print("### GRU/LSTM outputs start here ###")

# 1) Liquidity ranking → top-50 items for deep models
liq_map_top50 = (
    rich_daily
    .query("Date >= @train_start and Date <= @train_end")
    .groupby("Item ID")["Total Volume"]
    .mean()
    .sort_values(ascending=False)
)
top50_ids = liq_map_top50.head(50).index
print("Top-50 liquid Item IDs:\n", top50_ids)

# 2) Returns & volume panels for top-50
rets_top = rets_w.loc[:, top50_ids]

vol_w = (
    rich_daily
    .pivot(index="Date", columns="Item ID", values="Total Volume")
    .sort_index()
)
vol_w.index = pd.to_datetime(vol_w.index)
vol_top = vol_w.loc[rets_top.index, top50_ids]

print("rets_top shape:", rets_top.shape)
print("vol_top shape:", vol_top.shape)

# 3) Clean NaNs for deep models (more aggressive handling)
rets_top_clean = (
    rets_top.copy()
            .sort_index()
            .interpolate(method="time", limit_direction="both", axis=0)
            .fillna(0.0)
)

vol_top_clean = (
    vol_top.copy()
           .sort_index()
           .ffill()
           .bfill()
           .fillna(0.0)
)

# 4) Derived features: rolling volatility and log volume
vol7_top    = rets_top_clean.rolling(7, min_periods=1).std()
log_vol_top = np.log1p(vol_top_clean)

print("NaNs after cleaning:")
print("  rets_top_clean:", np.isnan(rets_top_clean.values).sum())
print("  vol_top_clean :", np.isnan(vol_top_clean.values).sum())
print("  vol7_top      :", np.isnan(vol7_top.values).sum())
print("  log_vol_top   :", np.isnan(log_vol_top.values).sum())

##########################################################
### 18. Sliding Window Builder for Deep Models         ###
##########################################################

WINDOW = 30  # sequence length (days) for LSTM/GRU

def make_sequences(feat_all,
                   target_all,
                   top_ids,
                   window: int,
                   feature_mode: str = "price_only"):
    """
    Construct sliding-window sequences for sequence models.

    Parameters
    ----------
    feat_all : DataFrame or MultiIndex DataFrame
        Feature panel. For 'price_only', this is returns only.
        For 'price_plus_volume', this is a MultiIndex with
        ('ret', iid), ('log_vol', iid), ('vol_7', iid).
    target_all : pd.DataFrame
        Target returns panel (same index as feat_all).
    top_ids : list-like
        List of item IDs to include.
    window : int
        Window length in days.
    feature_mode : {"price_only", "price_plus_volume"}
        Determines feature stacking.

    Returns
    -------
    X : np.ndarray
        3D array of shape (n_sequences, window, n_features).
    y : np.ndarray
        1D array of target returns.
    meta : pd.DataFrame
        Metadata with columns ['Date', 'Item ID'] for each sequence.
    """
    X_list, y_list, meta = [], [], []
    dates = target_all.index

    for iid in top_ids:
        if feature_mode == "price_only":
            f_item = feat_all[iid].values.reshape(-1, 1)
        else:
            f_item = np.stack(
                [
                    feat_all[("ret", iid)].values,
                    feat_all[("log_vol", iid)].values,
                    feat_all[("vol_7", iid)].values,
                ],
                axis=-1,
            )

        t_item = target_all[iid].values

        for i in range(window, len(dates)):
            t_date = dates[i]
            X_list.append(f_item[i-window:i, :])
            y_list.append(t_item[i])
            meta.append((t_date, iid))

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    meta = pd.DataFrame(meta, columns=["Date", "Item ID"])

    return X, y, meta

##########################################################
### 19. Deep Model Wrapper (LSTM / GRU)                ###
##########################################################

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def run_deep_models(FEATURE_MODE: str,
                    tag: str,
                    return_models: bool = False):
    """
    Train LSTM and GRU baselines on top-50 liquid items
    under a given feature configuration, and compare to VAR.

    Parameters
    ----------
    FEATURE_MODE : {"price_only", "price_plus_volume"}
        Input feature configuration.
    tag : str
        Tag used in output CSV filenames.
    return_models : bool, optional
        If True, returns models and full sequence data.

    Returns
    -------
    comp : pd.DataFrame
        Per-item MAE comparison (VAR vs LSTM vs GRU).
    or (if return_models=True):
        (lstm_model, gru_model, X_all, y_all, meta_all, comp)
    """
    print(f"\n=== Running deep models with FEATURE_MODE = {FEATURE_MODE} ===")

    # Prepare feature panel
    if FEATURE_MODE == "price_only":
        feat_all   = rets_top_clean
        n_features = 1
    else:
        feat_all = pd.concat(
            {"ret": rets_top_clean, "log_vol": log_vol_top, "vol_7": vol7_top},
            axis=1,
        )
        n_features = 3

    target_all = rets_top_clean

    # Build sequences
    X_all, y_all, meta_all = make_sequences(
        feat_all=feat_all,
        target_all=target_all,
        top_ids=top50_ids,
        window=WINDOW,
        feature_mode=FEATURE_MODE,
    )

    print("Sequence dataset shapes:")
    print("  X_all:", X_all.shape, " y_all:", y_all.shape)
    print(meta_all.head())

    # Align sequences with TRAIN / VAL / TEST windows
    meta_all["Date"] = pd.to_datetime(meta_all["Date"])

    train_mask = (meta_all["Date"] >= train_start) & (meta_all["Date"] <= train_end)
    val_mask   = (meta_all["Date"] >  train_end)   & (meta_all["Date"] <= val_end)
    test_mask  = (meta_all["Date"] >  val_end)     & (meta_all["Date"] <= END)

    X_train = X_all[train_mask.values]; y_train = y_all[train_mask.values]
    X_val   = X_all[val_mask.values];   y_val   = y_all[val_mask.values]
    X_test  = X_all[test_mask.values];  y_test  = y_all[test_mask.values]

    meta_train = meta_all[train_mask].reset_index(drop=True)
    meta_val   = meta_all[val_mask].reset_index(drop=True)
    meta_test  = meta_all[test_mask].reset_index(drop=True)

    print("Train sequences:", X_train.shape, "| targets:", y_train.shape)
    print("Val   sequences:", X_val.shape,   "| targets:", y_val.shape)
    print("Test  sequences:", X_test.shape,  "| targets:", y_test.shape)

    # LSTM model
    lstm_model = Sequential(
        [
            LSTM(64, input_shape=(WINDOW, n_features)),
            Dropout(0.2),
            Dense(1),
        ]
    )
    lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    es = EarlyStopping(monitor="val_mae", patience=8, restore_best_weights=True)

    lstm_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=1,
    )

    # GRU model
    gru_model = Sequential(
        [
            GRU(64, input_shape=(WINDOW, n_features)),
            Dropout(0.2),
            Dense(1),
        ]
    )
    gru_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    gru_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=1,
    )

    # Test predictions
    yhat_lstm = lstm_model.predict(X_test).ravel()
    yhat_gru  = gru_model.predict(X_test).ravel()

    mae_lstm  = mean_absolute_error(y_test, yhat_lstm)
    rmse_lstm = mean_squared_error(y_test, yhat_lstm) ** 0.5
    mae_gru   = mean_absolute_error(y_test, yhat_gru)
    rmse_gru  = mean_squared_error(y_test, yhat_gru) ** 0.5

    print(f"LSTM  Test MAE={mae_lstm:.5f}  RMSE={rmse_lstm:.5f}")
    print(f"GRU   Test MAE={mae_gru:.5f}  RMSE={rmse_gru:.5f}")

    # Assemble per-item comparison (VAR vs LSTM vs GRU)
    pred_df = meta_test.copy()
    pred_df["y_true"]    = y_test
    pred_df["yhat_lstm"] = yhat_lstm
    pred_df["yhat_gru"]  = yhat_gru

    pred_df["err_lstm"] = (pred_df["y_true"] - pred_df["yhat_lstm"]).abs()
    pred_df["err_gru"]  = (pred_df["y_true"] - pred_df["yhat_gru"]).abs()

    mae_lstm_items = pred_df.groupby("Item ID")["err_lstm"].mean().rename("MAE_LSTM")
    mae_gru_items  = pred_df.groupby("Item ID")["err_gru"].mean().rename("MAE_GRU")

    var_mae_top50 = mae_item.reindex(top50_ids).rename("MAE_VAR")

    comp = (
        pd.concat([var_mae_top50, mae_lstm_items, mae_gru_items], axis=1)
        .join(name_map.rename("Item Name"))
    )
    comp.index.name = "Item ID"
    comp.insert(0, "Item Name", comp.pop("Item Name"))

    print(f"\n=== Head of VAR vs LSTM vs GRU (top-50 items) — {FEATURE_MODE} ===")
    print(comp.head(10))

    print("\n=== Median MAE across top-50 liquid items ===")
    print("VAR  median MAE :", comp["MAE_VAR"].median())
    print("LSTM median MAE :", comp["MAE_LSTM"].median())
    print("GRU  median MAE :", comp["MAE_GRU"].median())

    best_model = comp[["MAE_VAR", "MAE_LSTM", "MAE_GRU"]].idxmin(axis=1)
    print("\n=== Count of items where each model has lowest MAE ===")
    print(best_model.value_counts())

    out_name = f"top50_var_lstm_gru_{tag}.csv"
    comp.to_csv(out_name, index_label="Item ID")
    print(f"\nWrote {out_name}")

    if return_models:
        return lstm_model, gru_model, X_all, y_all, meta_all, comp
    else:
        return comp

##########################################################
### 20. Run Deep Models in Both Feature Modes          ###
##########################################################

lstm_model, gru_model, X_all_po, y_all_po, meta_all_po, comp_price_only = run_deep_models(
    FEATURE_MODE="price_only",
    tag="price_only",
    return_models=True,
)
comp_price_vol = run_deep_models(
    FEATURE_MODE="price_plus_volume",
    tag="price_plus_volume",
    return_models=False,
)

##########################################################
### 21. Model Comparison Visuals (VAR vs LSTM vs GRU)  ###
##########################################################

# Boxplot of MAE across models (price-only feature set)
plt.figure(dpi=180)
plt.boxplot(
    [
        comp_price_only["MAE_VAR"].dropna(),
        comp_price_only["MAE_LSTM"].dropna(),
        comp_price_only["MAE_GRU"].dropna(),
    ],
    tick_labels=["VAR", "LSTM", "GRU"],
)
plt.ylabel("MAE (test)")
plt.title("MAE Distribution by Model (price_only)")
plt.tight_layout()
plt.show()

# Zoomed-in boxplot for fine differences
plt.figure(dpi=180)
plt.boxplot(
    [
        comp_price_only["MAE_VAR"].dropna(),
        comp_price_only["MAE_LSTM"].dropna(),
        comp_price_only["MAE_GRU"].dropna(),
    ],
    tick_labels=["VAR", "LSTM", "GRU"],
)
plt.ylim(0, 0.25)
plt.ylabel("MAE (test)")
plt.title("MAE Distributions by Model (Zoomed In)")
plt.tight_layout()
plt.show()

# Median MAE bars (price-only)
meds_po = {
    "VAR":  comp_price_only["MAE_VAR"].median(),
    "LSTM": comp_price_only["MAE_LSTM"].median(),
    "GRU":  comp_price_only["MAE_GRU"].median(),
}

plt.figure(dpi=180)
plt.bar(list(meds_po.keys()), list(meds_po.values()))

# Annotate bars with numeric values
for i, (k, v) in enumerate(meds_po.items()):
    plt.text(
        i,
        v + 0.001,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=12,
    )

plt.ylim(0, 0.05)  # align with similar MAE barplots
plt.ylabel("Median MAE (Test)")
plt.title("Top-50 Liquid Items: Median MAE by Model (Price-Only)")
plt.tight_layout()
plt.show()

# Combined chart: LSTM / GRU across feature sets
med_lstm_po = comp_price_only["MAE_LSTM"].median()
med_gru_po  = comp_price_only["MAE_GRU"].median()

med_lstm_pv = comp_price_vol["MAE_LSTM"].median()
med_gru_pv  = comp_price_vol["MAE_GRU"].median()

feature_sets = ["price_only", "price+volume+vol7"]
x = np.arange(len(feature_sets))
width = 0.38

fig, ax = plt.subplots(figsize=(9, 6), dpi=160)

bars_lstm = ax.bar(x - width/2, [med_lstm_po, med_lstm_pv], width, label="LSTM")
bars_gru  = ax.bar(x + width/2, [med_gru_po,  med_gru_pv],  width, label="GRU")

def _label_bars(bars, dy: float = 0.0006, fontsize: int = 14):
    """
    Label bar tops with their heights, offset by dy.
    """
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width()/2,
            h + dy,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )

_label_bars(bars_lstm)
_label_bars(bars_gru)

# Align y-limits with previous MAE barplots
plt.draw()
current_ylim = ax.get_ylim()
ax.set_ylim(current_ylim[0], 0.05)

ax.set_ylabel("Median MAE (test)", fontsize=14)
ax.set_xlabel("Feature Configuration", fontsize=14)
ax.set_title("Top-50 Items: LSTM vs GRU — Median MAE by Feature Set", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(feature_sets, fontsize=13)
ax.tick_params(axis="y", labelsize=13)
ax.grid(axis="y", linestyle=":", alpha=0.35)
ax.legend(frameon=False, fontsize=14, loc="upper left")

plt.tight_layout()
plt.show()

# Scatter: VAR vs LSTM (price-only), with 45° line + annotated outliers
plt.figure(figsize=(6, 6), dpi=130)
plt.scatter(
    comp_price_only["MAE_VAR"],
    comp_price_only["MAE_LSTM"],
    alpha=0.7,
    color="steelblue",
    edgecolor="white",
)

lim_max = max(comp_price_only[["MAE_VAR", "MAE_LSTM"]].max()) * 1.05
plt.plot([0, lim_max], [0, lim_max], "r--", lw=1, label="Equal error line")

for iid, row in comp_price_only.iterrows():
    nm = row["Item Name"]
    if isinstance(nm, str) and "Demon tear" in nm:
        plt.text(
            row["MAE_VAR"] * 1.02,
            row["MAE_LSTM"] * 1.02,
            "Demon Tear",
            fontsize=8,
            color="darkred",
            weight="bold",
            ha="right",
            va="bottom",
        )

for iid, row in comp_price_only.iterrows():
    nm = row["Item Name"]
    if (row["MAE_VAR"] > 0.09) or (row["MAE_LSTM"] > 0.09):
        if not (isinstance(nm, str) and "Demon tear" in nm):
            plt.text(
                row["MAE_VAR"] * 1.02,
                row["MAE_LSTM"] * 1.02,
                nm,
                fontsize=8,
                color="darkred",
            )

plt.xlabel("VAR MAE")
plt.ylabel("LSTM MAE")
plt.title(
    "Per-Item Forecast Comparison: VAR vs LSTM\n"
    "(top-50 liquid items, price-only)"
)
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

##########################################################
### 22. Granger Causality on PCA Factors (PC1–PC4)     ###
##########################################################

# Combine TRAIN + VAL factors (no TEST leakage)
F_all = pd.concat([F_tr_full, F_va_full])

pcs = ["PC1", "PC2", "PC3", "PC4"]
factors_df = F_all[pcs]

maxlag = 2
gc_matrix = pd.DataFrame(index=pcs, columns=pcs, dtype=float)

def gc_pvalue(x: pd.Series, y: pd.Series) -> float:
    """
    Return minimum Granger p-value across lags 1..maxlag using
    the ssr_chi2test statistic.
    """
    test = grangercausalitytests(
        pd.concat([y, x], axis=1).dropna(),
        maxlag=maxlag,
        verbose=False,
    )
    return min(test[lag][0]["ssr_chi2test"][1] for lag in range(1, maxlag + 1))

for cause in pcs:
    for effect in pcs:
        if cause == effect:
            gc_matrix.loc[effect, cause] = None
        else:
            gc_matrix.loc[effect, cause] = gc_pvalue(
                factors_df[cause],
                factors_df[effect],
            )

print("\n=== Granger Causality p-values (PC1–PC4, lag=1–2) ===")
print(gc_matrix.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "—"))

##########################################################
### 23. Economic Backtest: High-Value Items            ###
##########################################################

print("### ECONOMIC BACKTEST: High-value item simulation ###")

# Hand-picked focus items (mega-rares, BIS gear, controls)
focus_ids = [
    20997, 22486, 27277,     # Mega-rares
    4151, 11832, 11834, 11826, 11828, 11830, 11785,
    11802, 11804, 11806, 11808,  # GS set
    11235, 12924, 12902, 22324, 12817, 12831, 12833,  # controls
]

focus_ids = [iid for iid in focus_ids if iid in rets_w.columns]

rets_focus = rets_w.loc[:, focus_ids].dropna(how="all")
feat_focus = rets_focus.copy().fillna(0.0)

X_focus, y_focus, meta_focus = make_sequences(
    feat_all=feat_focus,
    target_all=rets_focus.fillna(0.0),
    top_ids=focus_ids,
    window=WINDOW,
    feature_mode="price_only",
)

# Predict with previously trained LSTM / GRU
yhat_focus_lstm = lstm_model.predict(X_focus, verbose=0).ravel()
yhat_focus_gru  = gru_model.predict(X_focus, verbose=0).ravel()

pred_focus = meta_focus.copy()
pred_focus["y_true"]     = y_focus
pred_focus["yhat_lstm"]  = yhat_focus_lstm
pred_focus["yhat_gru"]   = yhat_focus_gru

# Directional accuracy and sign-based PnL
pred_focus["signal_lstm"] = np.sign(pred_focus["yhat_lstm"])
pred_focus["signal_gru"]  = np.sign(pred_focus["yhat_gru"])
pred_focus["actual"]      = np.sign(pred_focus["y_true"])

pred_focus["lstm_correct"] = (
    pred_focus["signal_lstm"] == pred_focus["actual"]
).astype(int)
pred_focus["gru_correct"]  = (
    pred_focus["signal_gru"]  == pred_focus["actual"]
).astype(int)

pred_focus["pnl_lstm"] = pred_focus["signal_lstm"] * pred_focus["y_true"]
pred_focus["pnl_gru"]  = pred_focus["signal_gru"]  * pred_focus["y_true"]

summary = (
    pred_focus
    .groupby("Item ID")[["lstm_correct", "gru_correct", "pnl_lstm", "pnl_gru"]]
    .mean()
    .rename(columns={"lstm_correct": "DA_LSTM", "gru_correct": "DA_GRU"})
)
summary["TotalPnL_LSTM"] = pred_focus.groupby("Item ID")["pnl_lstm"].sum()
summary["TotalPnL_GRU"]  = pred_focus.groupby("Item ID")["pnl_gru"].sum()
summary["Item Name"]     = summary.index.map(name_map)

summary = summary[
    ["Item Name", "DA_LSTM", "DA_GRU", "TotalPnL_LSTM", "TotalPnL_GRU"]
].sort_values("TotalPnL_LSTM", ascending=False)

print("\nBacktest summary (head):")
print(summary.head(10))

# Cumulative PnL over time (aggregated across focus items)
cum_pnl = (
    pred_focus
    .groupby("Date")[["pnl_lstm", "pnl_gru"]]
    .sum()
    .cumsum()
)
cum_pnl.plot(
    figsize=(8, 4),
    title="Simulated cumulative returns (LSTM vs GRU)\nHigh-value focus items",
    ylabel="Cumulative return units",
)
plt.tight_layout()
plt.show()

##########################################################
### 24. Objective 3: Bai–Perron Structural Break Tests ###
##########################################################

# Item pool for structural break analysis
# (not all items are necessarily plotted)
all_items = {
    30753: "Oathplate platebody",
    27612: "Venator bow (uncharged)",
    29580: "Tormented synapse",
    11834: "Bandos tassets",
    566:   "Soul rune",
    31111: "Demon tear",
    27277: "Tumeken's shadow (uncharged)",
    31115: "Eye of ayak (uncharged)",
    12924: "Toxic blowpipe (empty)",
    22810: "Dragon knife(p++)",
    31088: "Avernic treads",
    11229: "Dragon arrow(p++)",
    22486: "Scythe of vitur (uncharged)",
    20997: "Twisted Bow",
    565:   "Blood Rune",
    564:   "Cosmic Rune",
    23685: "Divine super combat potion(4)",
    385:   "Shark",
    383:   "Raw shark",
    13441: "Anglerfish",
    13439: "Raw Anglerfish",
    1513:  "Magic logs",
    23733: "Divine Ranging Potion (4)",
    12934: "Zulrah Scales",
    2:     "Cannonball",
    24417: "Inquisitor's mace",
    24488: "Inquisitor's Armour Set",
    20724: "Ancestral Robe Bottom",
    29577: "Burning Claws",
    11286: "Draconic Visage",
    24511: "Harmonized Orb",
    24514: "Volatile Orb",
    24517: "Eldritch Orb",
    27690: "Voidwaker",
    28307: "Ultor ring",
    28310: "Venator ring",
    28313: "Magus ring",
    28316: "Bellator ring",
    28334: "Awakener's orb",
    28338: "Soulreaper Axe",
    29025: "Blood Moon Tassets",
    29796: "Noxious Halberd",
    29806: "Arenea Boots",
    30070: "DragonHunter Wand",
    1046:  "Purple Party Hat",
    19493: "Zenyte",
    1006:  "Red Wood Logs",
    20205: "Golden Chef's",
    4153:  "Granite Maul",
    25859: "Enhanced crystal weapon seed",
    26235: "Zaryte vambraces",
}

def get_log_price_series(rich_daily: pd.DataFrame, item_id: int):
    """
    Extract a log price series for a given item over the macro window.

    Returns
    -------
    logp : pd.Series or None
        Daily log prices (gp) on [START, END], or None if insufficient data.
    """
    sub = (
        macro_df[macro_df["Item ID"] == item_id]
        .copy()
        .sort_values("Date")
    )
    sub["Date"] = pd.to_datetime(sub["Date"])
    sub = sub[(sub["Date"] >= START) & (sub["Date"] <= END)]
    sub = sub.dropna(subset=["Average Price"])
    sub = sub[sub["Average Price"] > 0]

    if sub.empty:
        return None

    s = sub.set_index("Date")["Average Price"]

    # Regular daily grid with interpolation
    full_idx = pd.date_range(START, END, freq="D")
    s = s.reindex(full_idx).interpolate().bfill().ffill()

    logp = np.log(s.astype(float))
    logp.name = all_items.get(item_id, f"Item {item_id}")
    return logp

def detect_breaks_logprice(log_series: pd.Series,
                           max_breaks: int = 4):
    """
    Detect structural breaks in log price using PELT with an
    approximate BIC-style penalty.

    Parameters
    ----------
    log_series : pd.Series
        Daily log price series.
    max_breaks : int
        Maximum number of breakpoints to retain.

    Returns
    -------
    list of pd.Timestamp
        Break dates within the series.
    """
    y = log_series.values.reshape(-1, 1)
    n = len(y)

    algo = rpt.Pelt(model="l2").fit(y)

    # BIC-style penalty: c * log(n),
    # c is a small tuning constant (here 3.0).
    pen = 3.0 * np.log(n)
    bkpt_indices = algo.predict(pen=pen)

    # Drop the final endpoint (len(series))
    bkpt_indices = bkpt_indices[:-1]

    if max_breaks is not None:
        bkpt_indices = bkpt_indices[:max_breaks]

    dates = log_series.index[bkpt_indices].to_list()
    return dates

# Run detection for all items in the pool
rows = []
log_series_dict = {}

for item_id, name in all_items.items():
    logp = get_log_price_series(rich_daily, item_id)
    if logp is None or len(logp) < 40:
        print(f"Skipping {item_id} ({name}): insufficient data in window")
        continue

    breaks = detect_breaks_logprice(logp, max_breaks=4)
    log_series_dict[item_id] = logp

    rows.append(
        {
            "Item ID": item_id,
            "Item Name": name,
            "n_obs": len(logp),
            "n_breaks": len(breaks),
            "break_dates": ", ".join(d.strftime("%Y-%m-%d") for d in breaks),
        }
    )

break_summary = pd.DataFrame(rows).set_index("Item ID")
print("=== Structural break summary (log price levels) ===")
print(break_summary)
break_summary.to_csv("structural_break_summary.csv", index_label="Item ID")

##########################################################
### 25. Structural Break Plots with Event Markers      ###
##########################################################

events = {
    "Yama release":              pd.Timestamp("2025-05-14"),
    "Final Dawn release":        pd.Timestamp("2025-07-23"),
    "Summer Sweep-Up: Combat":   pd.Timestamp("2025-06-25"),
    "Sailing skilling poll":     pd.Timestamp("2025-07-31"),
    "Varlamore & Summer tweaks": pd.Timestamp("2025-08-06"),
}

def plot_item_with_breaks(item_id: int, event_dict=None) -> None:
    """
    Plot price levels with detected breakpoints and optional
    vertical event markers (patch releases, polls, etc.).
    """
    logp = log_series_dict.get(item_id)
    if logp is None:
        print(f"No log series stored for {item_id}")
        return

    name = all_items[item_id]
    breaks = detect_breaks_logprice(logp, max_breaks=4)

    # Tall + slightly higher DPI for better readability in the paper (previous verision too small)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)

    # Base price series
    ax.plot(
        logp.index,
        np.exp(logp.values),
        label="Price level",
        linewidth=1.3,
    )

    # Structural breaks: make them bold and clearly distinct
    for d in breaks:
        ax.axvline(d,color="red",  linestyle="--", alpha=0.9, linewidth=2,  zorder=3,
        )

    # Event markers: lighter and thinner so they don't compete 
    if event_dict is not None:
        for label, d in event_dict.items():
            if START <= d <= END:
                ax.axvline(
                    d,
                    color="blue",
                    linestyle=":",
                    alpha=0.5,
                    linewidth=1,
                    zorder=2,
                )
                ax.text(
                    d,
                    ax.get_ylim()[1],
                    label,
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=11,
                    alpha=0.8,
                )

    ax.set_title(f"{name} (ID {item_id}) — Bai–Perron Test", fontsize=13)
    ax.set_ylabel("GP", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(GP_EZ_Read))

    # Larger tick labels
    ax.tick_params(axis="both", labelsize=10)

    ax.grid(alpha=0.3)

    # Simple legend 
    # Proxy handles for breaks/events
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="red", linestyle="--", linewidth=2, label="Structural break"),
        Line2D([], [], color="blue", linestyle=":", linewidth=2, label="Event marker"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.show()


# Example Objective 3 plots used in the discussion
# (Not all items in all_items are plotted here.)

# Problem items: Bad data or False positive break from API price backfill
plot_item_with_breaks(30753, event_dict=events)   # Oathplate Platebody
plot_item_with_breaks(31111, event_dict=events)   # Demon Tear
# Confirmed breaks
plot_item_with_breaks(4153,  event_dict=events)   # Granite Maul
plot_item_with_breaks(29577, event_dict=events)   # Burning Claws
plot_item_with_breaks(566,   event_dict=events)   # Soul Rune

# High-level equipment
plot_item_with_breaks(29580, event_dict=events)   # Tormented synapse
plot_item_with_breaks(11834, event_dict=events)   # Bandos tassets
plot_item_with_breaks(12924, event_dict=events)   # Toxic Blowpipe

# Mega rares
plot_item_with_breaks(27277, event_dict=events)   # Tumeken's Shadow
plot_item_with_breaks(22486, event_dict=events)   # Scythe of Vitur
plot_item_with_breaks(20997, event_dict=events)   # Twisted Bow

# Popular ranged ammunition
plot_item_with_breaks(12934, event_dict=events)   # Zulrah Scales
plot_item_with_breaks(22810, event_dict=events)   # Dragon knife(p++)
plot_item_with_breaks(11229, event_dict=events)   # Dragon arrow(p++)
plot_item_with_breaks(2,      event_dict=events)  # Cannonball

# Runes
plot_item_with_breaks(565, event_dict=events)     # Blood Rune
plot_item_with_breaks(564, event_dict=events)     # Cosmic Rune

# Food and potions
plot_item_with_breaks(23685, event_dict=events)   # Divine super combat potion(4)
plot_item_with_breaks(23733, event_dict=events)   # Divine ranging potion(4)
plot_item_with_breaks(385,   event_dict=events)   # Shark
plot_item_with_breaks(383,   event_dict=events)   # Raw Shark
plot_item_with_breaks(13441, event_dict=events)   # Anglerfish
plot_item_with_breaks(13439, event_dict=events)   # Raw Anglerfish

# Miscellaneous
plot_item_with_breaks(1513,  event_dict=events)   # Magic Logs
plot_item_with_breaks(20205, event_dict=events)   # Golden Chef's Hat

