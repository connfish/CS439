# ===========================
# Portfolio Optimization with Hierarchical Clustering
# Final Project script — reads ./history.csv
# ===========================
# Expected CSV columns:
# Date,Open,High,Low,Close,Adj Close,Volume,Symbol
# e.g., 2008-01-29,9.50,9.99,8.57,8.75,0.702589,1489000,AACG

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ---------------------------
# DATA CONFIG
# ---------------------------
CSV_PATH   = "history.csv"   # CSV in the same folder as main.py
DATE_COL   = "Date"
PRICE_COL  = "Adj Close"
SYMBOL_COL = "Symbol"

# ---------------------------
# GENERAL CONFIG
# ---------------------------
ANN  = 252     # trading days/year
SEED = 42
np.random.seed(SEED)


# ---------------------------
# Small helper for figure window titles
# ---------------------------
def _set_window_title(title: str):
    fig = plt.gcf()
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass


# ---------------------------
# 1) Load & prepare data
# ---------------------------
def load_prices() -> pd.DataFrame:
    """
    Read prices from history.csv, pivot to wide matrix (index=Date, columns=Symbol, values=Adj Close),
    forward-fill, drop all-NaN symbols, and (if available) keep last ~20 years.
    """
    df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL])
    df = df[[DATE_COL, SYMBOL_COL, PRICE_COL]].dropna()
    prices = (
        df.pivot(index=DATE_COL, columns=SYMBOL_COL, values=PRICE_COL)
          .sort_index()
          .dropna(axis=1, how="all")
          .ffill()
          .dropna(how="all")
    )

    # Cap to ~20 years if present (keeps evaluation manageable & matches slides)
    last_date = prices.index.max()
    if pd.notna(last_date):
        cutoff = last_date - pd.DateOffset(years=20)
        prices = prices[prices.index >= cutoff]

    return prices

def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = np.log(prices / prices.shift(1))
    return rets.dropna(how="all").dropna(axis=0)


# ---------------------------
# 2) Covariance / Correlation
# ---------------------------
def shrinkage_cov_corr(returns: pd.DataFrame):
    """Ledoit–Wolf shrinkage covariance + correlation (robust)"""
    lw  = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    std = np.sqrt(np.diag(cov))
    corr = cov.div(std, axis=0).div(std, axis=1)
    return cov, corr


# ---------------------------
# 3) Clustering utilities
# ---------------------------
def corr_to_dist(corr: pd.DataFrame) -> pd.DataFrame:
    """Lopez de Prado distance: sqrt(0.5*(1 - corr))"""
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist.values, 0.0)
    return dist

def hierarchical_clustering(dist: pd.DataFrame, method: str = "ward"):
    condensed = squareform(dist.values, checks=False)
    return linkage(condensed, method=method)

def get_quasi_diag(Z: np.ndarray) -> list:
    """Leaf order (seriation) from linkage"""
    N = Z.shape[0] + 1
    def _dfs(node):
        if node < N:
            return [node]
        left  = int(Z[node - N, 0])
        right = int(Z[node - N, 1])
        return _dfs(left) + _dfs(right)
    return _dfs(2 * N - 2)

def extract_macro_clusters(Z, labels, k=3):
    """Three macro clusters to match report/slides."""
    clusters = fcluster(Z, t=k, criterion="maxclust")
    return pd.Series(clusters, index=labels, name="cluster")


# ---------------------------
# 4) HRP (manual implementation)
# ---------------------------
def get_ivp(cov_sub: np.ndarray) -> np.ndarray:
    iv = 1.0 / np.diag(cov_sub)
    return iv / iv.sum()

def cluster_variance(cov: pd.DataFrame, names: list) -> float:
    sub = cov.loc[names, names].values
    w   = get_ivp(sub)
    return float(w.T @ sub @ w)

def hrp_weights(cov: pd.DataFrame, ordered_names: list) -> pd.Series:
    w = pd.Series(1.0, index=ordered_names)
    stacks = [ordered_names]
    while stacks:
        new = []
        for cl in stacks:
            if len(cl) <= 1:
                continue
            split = len(cl) // 2
            L, R = cl[:split], cl[split:]
            vL = cluster_variance(cov, L)
            vR = cluster_variance(cov, R)
            alpha = 1.0 - vL / (vL + vR)
            w[L] *= alpha
            w[R] *= (1.0 - alpha)
            new += [L, R]
        stacks = new
    return (w / w.sum()).sort_values(ascending=False)


# ---------------------------
# 5) Benchmarks & Metrics
# ---------------------------
def eq_weight(names) -> pd.Series:
    n = len(names)
    return pd.Series(1.0 / n, index=names)

def gmv_weights(cov: pd.DataFrame) -> pd.Series:
    """Global Minimum Variance (closed-form, long-only by clipping)."""
    names = cov.columns
    S = cov.values
    ones = np.ones((len(names), 1))
    try:
        w = np.linalg.solve(S, ones)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(S) @ ones
    w = (w / w.sum()).ravel()
    w = np.clip(w, 0, None)
    w = w / w.sum()
    return pd.Series(w, index=names)

def portfolio_metrics(returns: pd.DataFrame, weights: pd.Series):
    w = weights.reindex(returns.columns).fillna(0).values
    mu = returns.mean().values
    C  = returns.cov().values
    ann_ret = float(np.dot(w, mu) * ANN)
    ann_vol = float(np.sqrt(w @ (C * ANN) @ w))
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    nav     = (1 + returns @ w).cumprod()
    peak    = nav.cummax()
    mdd     = (nav / peak - 1.0).min()
    return ann_ret, ann_vol, sharpe, float(mdd), nav

def cluster_exposures(weights: pd.Series, labels: pd.Series) -> pd.Series:
    df = pd.DataFrame({"w": weights}).join(labels)
    return df.groupby("cluster")["w"].sum().sort_index()

def membership_strength(corr: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """'Core-ness': in-cluster mean corr – best out-cluster mean corr."""
    out = {}
    for t in corr.columns:
        c = int(labels.loc[t])
        in_ids  = labels.index[labels == c]
        out_ids = labels.index[labels != c]

        in_vals = corr.loc[t, in_ids].drop(t, errors="ignore")
        in_mean = in_vals.mean() if len(in_vals) else np.nan

        if len(out_ids):
            grouped = corr.loc[t, out_ids].groupby(labels.loc[out_ids])
            out_mean = grouped.mean().max()
        else:
            out_mean = np.nan

        out[t] = float(in_mean - out_mean) if (pd.notna(in_mean) and pd.notna(out_mean)) else np.nan
    return pd.Series(out, name="membership_strength").dropna().sort_values(ascending=False)


# ---------------------------
# 6) Plot helpers (titles match visuals)
# ---------------------------
def plot_corr_heatmap(corr: pd.DataFrame, title="Correlation Heatmap (Shrinkage)"):
    plt.figure(figsize=(9,6))
    im = plt.imshow(corr.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    n = len(corr)
    max_labels = 40
    if n <= max_labels:
        tick_pos = np.arange(n)
    else:
        step = max(1, n // max_labels)
        tick_pos = np.arange(0, n, step)
    plt.xticks(tick_pos, corr.columns[tick_pos], rotation=90, fontsize=6)
    plt.yticks(tick_pos, corr.index[tick_pos], fontsize=6)
    plt.title(title)
    plt.tight_layout()
    _set_window_title("Correlation Heatmap")

def plot_corr_heatmap_reordered(corr: pd.DataFrame, Z, title="Correlation Heatmap (Reordered by Clustering)"):
    order_idx = get_quasi_diag(Z)
    ordered   = [corr.columns[i] for i in order_idx]
    corr_ord  = corr.loc[ordered, ordered]

    plt.figure(figsize=(9,6))
    im = plt.imshow(corr_ord.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    n = len(corr_ord)
    max_labels = 40
    if n <= max_labels:
        tick_pos = np.arange(n)
    else:
        step = max(1, n // max_labels)
        tick_pos = np.arange(0, n, step)
    plt.xticks(tick_pos, corr_ord.columns[tick_pos], rotation=90, fontsize=6)
    plt.yticks(tick_pos, corr_ord.index[tick_pos], fontsize=6)
    plt.title(title)
    plt.tight_layout()
    _set_window_title("Reordered Correlation Heatmap")

def plot_dendrogram_sampled(corr: pd.DataFrame,
                            title="Ward Hierarchical Clustering Dendrogram",
                            max_names: int = 60):
    # choose up to max_names tickers with highest avg|corr|
    tickers = corr.columns.tolist()
    if len(tickers) <= max_names:
        sel = tickers
    else:
        avg_corr = corr.abs().mean().sort_values(ascending=False)
        sel = avg_corr.head(max_names).index.tolist()

    corr_sub = corr.loc[sel, sel]
    dist_sub = corr_to_dist(corr_sub)
    Z_sub    = hierarchical_clustering(dist_sub, method="ward")

    plt.figure(figsize=(11,4))
    dendrogram(Z_sub, labels=sel, leaf_rotation=90, leaf_font_size=8)
    plt.title(title)
    plt.tight_layout()
    _set_window_title("Dendrogram (sampled with tickers)")

def plot_weights(weights: pd.Series, title):
    top = weights.sort_values(ascending=False).head(30)
    plt.figure(figsize=(11,4))
    top.plot(kind="bar")
    plt.ylabel("Weight")
    plt.title(f"{title} (Top 30 names)")
    plt.tight_layout()
    _set_window_title("Portfolio Weights")

def plot_cum(nav_dict, title="Cumulative Growth (HRP vs EW vs GMV)"):
    plt.figure(figsize=(10,4))
    for name, nav in nav_dict.items():
        plt.plot(nav.index, nav.values, label=name)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    _set_window_title(title)

def plot_drawdowns(nav_dict, title="Drawdowns"):
    plt.figure(figsize=(10,3.5))
    for name, nav in nav_dict.items():
        dd = nav / nav.cummax() - 1.0
        plt.plot(dd.index, dd.values, label=name)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    _set_window_title(title)

def plot_pca_scatter(corr: pd.DataFrame, labels: pd.Series, title="PCA (2D) of Correlation Structure"):
    # PCA on distance matrix for a planar view
    dist = corr_to_dist(corr)
    X = dist.values
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X)

    plt.figure(figsize=(7,5))
    for cid in sorted(labels.unique()):
        mask = (labels.values == cid)
        plt.scatter(coords[mask,0], coords[mask,1], s=20, label=f"Cluster {cid}")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    _set_window_title("PCA 2D")


# ---------------------------
# 7) Walk-forward backtest (3y train, 1y test)
# ---------------------------
def walk_forward_oos(returns: pd.DataFrame,
                     train_years=3, test_years=1,
                     linkage_method="ward"):
    """
    Rolling walk-forward:
      - fit on 'train_years'
      - apply to next 'test_years'
    Concatenate all test slices; return cumulative series for HRP/EW/GMV.
    """
    idx = returns.index
    years = sorted({d.year for d in idx})
    if len(years) < (train_years + test_years + 1):
        return None

    hrp_nav, ew_nav, gmv_nav = [], [], []
    hrp_idx, ew_idx, gmv_idx = [], [], []

    # Map year -> (start_date, end_date)
    year_to_dates = {}
    for y in years:
        dts = idx[idx.year == y]
        if len(dts):
            year_to_dates[y] = (dts.min(), dts.max())

    for i in range(0, len(years) - (train_years + test_years) + 1):
        tr_span = years[i:i+train_years]
        te_span = years[i+train_years:i+train_years+test_years]

        tr_start, _ = year_to_dates[tr_span[0]]
        _, tr_end   = year_to_dates[tr_span[-1]]
        te_start, _ = year_to_dates[te_span[0]]
        _, te_end   = year_to_dates[te_span[-1]]

        tr = returns[(returns.index >= tr_start) & (returns.index <= tr_end)]
        te = returns[(returns.index >= te_start) & (returns.index <= te_end)]

        if len(tr) < 60 or len(te) < 20:
            continue

        cov, corr = shrinkage_cov_corr(tr)
        dist = corr_to_dist(corr)
        Z    = hierarchical_clustering(dist, method=linkage_method)
        order_idx = get_quasi_diag(Z)
        ordered   = [corr.columns[i] for i in order_idx]

        w_hrp = hrp_weights(cov, ordered)
        w_ew  = eq_weight(list(corr.columns))
        w_gmv = gmv_weights(cov)

        def _nav(te_rets, w):
            wv = w.reindex(te_rets.columns).fillna(0).values
            return (1 + te_rets @ wv).cumprod()

        hrp_nav.append(_nav(te, w_hrp)); hrp_idx.append(te.index)
        ew_nav.append(_nav(te, w_ew));   ew_idx.append(te.index)
        gmv_nav.append(_nav(te, w_gmv)); gmv_idx.append(te.index)

    if not hrp_nav:
        return None

    # Stitch sequentially
    def stitch(series_list, index_list):
        nav = None
        for s, ix in zip(series_list, index_list):
            s = pd.Series(s, index=ix)
            if nav is None:
                nav = s
            else:
                s = s * nav.iloc[-1]
                nav = pd.concat([nav, s])
        return nav

    return {
        "HRP": stitch(hrp_nav, hrp_idx),
        "EW":  stitch(ew_nav,  ew_idx),
        "GMV": stitch(gmv_nav, gmv_idx),
    }


# ---------------------------
# 8) Robustness: linkage sensitivity summary
# ---------------------------
def linkage_sensitivity(returns: pd.DataFrame, linkages=("ward","average","complete")):
    rows = []
    for meth in linkages:
        cov, corr = shrinkage_cov_corr(returns)
        dist = corr_to_dist(corr)
        Z    = hierarchical_clustering(dist, method=meth)
        order_idx = get_quasi_diag(Z)
        ordered   = [corr.columns[i] for i in order_idx]
        w = hrp_weights(cov, ordered)
        r, v, s, dd, _ = portfolio_metrics(returns, w)
        rows.append({"linkage": meth, "AnnRet": r, "AnnVol": v, "Sharpe": s, "MaxDD": dd})
    return pd.DataFrame(rows).set_index("linkage").sort_values("Sharpe", ascending=False)


# ---------------------------
# 9) Main
# ---------------------------
def main():
    # Load & filter by availability
    prices = load_prices()
    print(f"Loaded prices: {prices.shape[0]} dates, {prices.shape[1]} symbols")

    # keep symbols with at least 80% coverage
    min_days = int(0.80 * prices.shape[0])
    prices = prices.loc[:, prices.notna().sum() >= min_days]

    returns = to_log_returns(prices)
    print(f"Returns shape: {returns.shape}")

    # Shrinkage risk model + base plots
    cov, corr = shrinkage_cov_corr(returns)
    plot_corr_heatmap(corr)  # Base heatmap
    dist = corr_to_dist(corr)
    Z    = hierarchical_clustering(dist, method="ward")
    plot_corr_heatmap_reordered(corr, Z)  # Reordered heatmap

    # Dendrogram (Ward) + three macro clusters
    plot_dendrogram_sampled(corr)
    labels = extract_macro_clusters(Z, corr.columns, k=3)
    print("\nMacro clusters (k=3) — counts:")
    print(labels.value_counts().sort_index())

    # Cluster membership strength (“core vs border”)
    ms = membership_strength(corr, labels)
    print("\nTop 10 strongest cluster members:")
    print(ms.head(10).round(3).to_string())

    # HRP weights (tree-aware) and benchmarks
    order_idx = get_quasi_diag(Z)
    ordered   = [corr.columns[i] for i in order_idx]
    w_hrp     = hrp_weights(cov, ordered)
    w_ew      = eq_weight(list(corr.columns))
    w_gmv     = gmv_weights(cov)

    plot_weights(w_hrp, "HRP Weights (Dendrogram-aware)")

    # In-sample performance & curves
    def summarize(name, w):
        r, v, s, dd, nav = portfolio_metrics(returns, w)
        print(f"{name:>6} | AnnRet {r:7.2%} | Vol {v:6.2%} | Sharpe {s:5.2f} | MaxDD {dd:6.2%}")
        return nav

    print("\nPerformance summary (annualized, in-sample):")
    nav_hrp = summarize("HRP", w_hrp)
    nav_ew  = summarize("EW",  w_ew)
    nav_gmv = summarize("GMV", w_gmv)

    plot_cum({"HRP": nav_hrp, "EW": nav_ew, "GMV": nav_gmv}, title="Cumulative Growth (In-Sample)")
    plot_drawdowns({"HRP": nav_hrp, "EW": nav_ew, "GMV": nav_gmv}, title="Drawdowns (In-Sample)")

    # Weight by macro cluster (ties back to dendrogram structure)
    ce = pd.DataFrame({
        "HRP": cluster_exposures(w_hrp, labels),
        "EW":  cluster_exposures(w_ew,  labels),
        "GMV": cluster_exposures(w_gmv, labels),
    })
    print("\nWeight by macro cluster (sums to 1):")
    print(ce.round(3).to_string())

    # PCA 2-D map of structure
    plot_pca_scatter(corr, labels)

    # Walk-forward backtest
    wf = walk_forward_oos(returns, train_years=3, test_years=1, linkage_method="ward")
    if wf is not None:
        plot_cum(wf, title="Cumulative Growth (Walk-Forward Out-of-Sample)")
        plot_drawdowns(wf, title="Drawdowns (Walk-Forward OOS)")

    # Linkage sensitivity summary
    sens = linkage_sensitivity(returns)
    print("\nLinkage sensitivity (HRP, in-sample):")
    print(sens.assign(
        AnnRet=lambda d: (d["AnnRet"]*100).round(2),
        AnnVol=lambda d: (d["AnnVol"]*100).round(2),
        Sharpe=lambda d: d["Sharpe"].round(2),
        MaxDD=lambda d: (d["MaxDD"]*100).round(2)
    )[["AnnRet","AnnVol","Sharpe","MaxDD"]].to_string())

    # Render all figures
    plt.show()


if __name__ == "__main__":
    main()
