# ===========================
# Portfolio Optimization with Hierarchical Clustering
# Final Project script (history.csv)
# ===========================
# CSV schema expected:
# Date,Open,High,Low,Close,Adj Close,Volume,Symbol
# e.g., 2008-01-29,9.500000,9.990000,8.570000,8.750000,0.702589,1489000,AACG

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ---------------------------
# CONFIG
# ---------------------------
CSV_PATH   = "history.csv"   # keep your path
DATE_COL   = "Date"
PRICE_COL  = "Adj Close"
SYMBOL_COL = "Symbol"
ANN        = 252             # trading days/year
SEED       = 42
np.random.seed(SEED)


# ---------------------------
# 0) Small helper for window titles
# ---------------------------
def _set_window_title(title: str):
    """Set the window title for the current figure if the backend supports it."""
    fig = plt.gcf()
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        # Some backends (e.g. inline / notebook) won't support this
        pass


# ---------------------------
# 1) Load & prepare data
# ---------------------------
def load_prices(csv_path: str) -> pd.DataFrame:
    """Pivot to wide prices: index=Date, columns=Symbol, values=Adj Close.

    Keep all rows that have at least one price, forward-fill missing values,
    drop symbols that are entirely missing, then restrict to the last 20 years.
    """
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])
    df = df[[DATE_COL, SYMBOL_COL, PRICE_COL]].dropna()

    prices = (
        df.pivot(index=DATE_COL, columns=SYMBOL_COL, values=PRICE_COL)
          .sort_index()
          .dropna(axis=1, how="all")     # drop symbols that are all NaN
          .ffill()                       # forward-fill missing values
          .dropna(how="all")             # drop rows that are all NaN
    )

    # Restrict to last 20 years (if we have that much history)
    last_date = prices.index.max()
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
    """Ledoit–Wolf shrinkage covariance + correlation (robust vs sample)."""
    lw  = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    std = np.sqrt(np.diag(cov))
    corr = cov.div(std, axis=0).div(std, axis=1)
    return cov, corr


# ---------------------------
# 3) Clustering utilities
# ---------------------------
def corr_to_dist(corr: pd.DataFrame) -> pd.DataFrame:
    """Lopez de Prado distance: sqrt(0.5*(1 - corr))."""
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist.values, 0.0)
    return dist


def hierarchical_clustering(dist: pd.DataFrame, method: str = "ward"):
    """Ward linkage per slides."""
    condensed = squareform(dist.values, checks=False)
    return linkage(condensed, method=method)


def plot_dendrogram(corr: pd.DataFrame,
                    title="Dendrogram (Ward, sampled tickers)",
                    figsize=(11, 4),
                    max_names: int = 60):
    """
    Plot a dendrogram WITH TICKER LABELS.

    For large universes, we:
      - pick up to `max_names` representative tickers
        (those with highest average absolute correlation),
      - recompute distance / linkage on that subset,
      - and show a full dendrogram with ticker labels.
    """
    tickers = corr.columns.tolist()
    n = len(tickers)

    if n <= max_names:
        selected = tickers
    else:
        # Pick the most "connected" tickers by average |corr|
        avg_corr = corr.abs().mean().sort_values(ascending=False)
        selected = avg_corr.head(max_names).index.tolist()

    corr_sub = corr.loc[selected, selected]
    dist_sub = corr_to_dist(corr_sub)
    Z_sub    = hierarchical_clustering(dist_sub, method="ward")

    plt.figure(figsize=figsize)
    dendrogram(
        Z_sub,
        labels=selected,
        leaf_rotation=90,
        leaf_font_size=8
    )
    plt.title(title)
    plt.tight_layout()
    _set_window_title("Dendrogram (sampled with tickers)")


def extract_macro_clusters(Z, labels, k=3):
    """
    Extract k macro clusters (consistent with slides’ 'three macro clusters').
    Returns: pd.Series index=ticker, value=cluster_id {1..k}
    """
    clusters = fcluster(Z, t=k, criterion="maxclust")
    return pd.Series(clusters, index=labels, name="cluster")


# ---------------------------
# 4) HRP (manual)
# ---------------------------
def get_quasi_diag(Z: np.ndarray) -> list:
    N = Z.shape[0] + 1

    def _dfs(node):
        if node < N:
            return [node]
        left  = int(Z[node - N, 0])
        right = int(Z[node - N, 1])
        return _dfs(left) + _dfs(right)

    return _dfs(2 * N - 2)


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
            vL   = cluster_variance(cov, L)
            vR   = cluster_variance(cov, R)
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
    """
    Global Minimum Variance (GMV) weights using closed-form solution.
    We enforce long-only by clipping negatives to 0 and renormalizing.
    """
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
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    # Max drawdown from cumulative returns
    nav = (1 + returns @ w).cumprod()
    peak = nav.cummax()
    dd = (nav / peak - 1.0).min()
    return ann_ret, ann_vol, sharpe, float(dd), nav


def cluster_exposures(weights: pd.Series, labels: pd.Series) -> pd.Series:
    """Sum weights by cluster id."""
    df = pd.DataFrame({"w": weights}).join(labels)
    return df.groupby("cluster")["w"].sum().sort_index()


def membership_strength(corr: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """
    Simple 'membership strength': in-cluster mean corr – best out-cluster mean corr.
    Higher => sits in cluster core; near 0 => border asset.
    """
    tickers = corr.columns
    out = {}
    for t in tickers:
        c = int(labels.loc[t])
        in_ids  = labels.index[labels == c]
        out_ids = labels.index[labels != c]

        # in-cluster, excluding itself
        in_vals = corr.loc[t, in_ids].drop(t, errors="ignore")
        in_mean = in_vals.mean() if len(in_vals) > 0 else np.nan

        # best out-of-cluster mean
        if len(out_ids) > 0:
            grouped = corr.loc[t, out_ids].groupby(labels.loc[out_ids])
            out_mean = grouped.mean().max()
        else:
            out_mean = np.nan

        out[t] = float(in_mean - out_mean) if (not np.isnan(in_mean) and not np.isnan(out_mean)) else np.nan

    return pd.Series(out, name="membership_strength").dropna().sort_values(ascending=False)


# ---------------------------
# 6) Plot helpers
# ---------------------------
def plot_corr_heatmap(corr: pd.DataFrame,
                      title="Shrinkage Correlation (Ledoit–Wolf)"):
    """Heatmap with a limited number of axis labels for readability."""
    n = len(corr)

    plt.figure(figsize=(9, 6))
    im = plt.imshow(corr.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)

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


def plot_weights(weights: pd.Series, title):
    """Bar chart of top 30 names only, to keep it readable."""
    top = weights.sort_values(ascending=False).head(30)
    plt.figure(figsize=(11, 4))
    top.plot(kind="bar")
    plt.ylabel("Weight")
    plt.title(f"{title} (Top 30 names)")
    plt.tight_layout()
    _set_window_title("Portfolio Weights")


def plot_cum(nav_dict):
    plt.figure(figsize=(10, 4))
    for name, nav in nav_dict.items():
        plt.plot(nav.index, nav.values, label=name)
    plt.legend()
    plt.title("Cumulative Growth (HRP vs EW vs GMV)")
    plt.tight_layout()
    _set_window_title("Cumulative Returns")


def plot_drawdowns(nav_dict):
    plt.figure(figsize=(10, 3.5))
    for name, nav in nav_dict.items():
        dd = nav / nav.cummax() - 1.0
        plt.plot(dd.index, dd.values, label=name)
    plt.legend()
    plt.title("Drawdowns")
    plt.tight_layout()
    _set_window_title("Drawdowns")


# ---------------------------
# 7) Main
# ---------------------------
def main():
    # --- Load & returns
    prices = load_prices(CSV_PATH)
    print(f"Loaded prices: {prices.shape[0]} dates, {prices.shape[1]} symbols")

    # Require reasonable price history per symbol
    min_days = int(0.80 * prices.shape[0])
    prices = prices.loc[:, prices.notna().sum() >= min_days]

    rets = to_log_returns(prices)
    print(f"Returns shape: {rets.shape}")

    # --- Covariance / correlation
    cov, corr = shrinkage_cov_corr(rets)
    plot_corr_heatmap(corr)

    # --- Full-universe distance & linkage for clustering / HRP
    dist = corr_to_dist(corr)
    Z    = hierarchical_clustering(dist, method="ward")

    # Dendrogram with actual ticker labels on a sampled subset
    plot_dendrogram(corr)

    # Cluster labels (k=3 macro clusters, full universe)
    labels = extract_macro_clusters(Z, corr.columns, k=3)
    print("\nMacro clusters (k=3) — counts:")
    print(labels.value_counts().sort_index())

    # Membership strength (core vs border)
    ms = membership_strength(corr, labels)
    print("\nTop 10 strongest cluster members:")
    print(ms.head(10).round(3).to_string())

    # --- HRP weights (manual, dendrogram-aware)
    order_idx = get_quasi_diag(Z)
    ordered   = [corr.columns[i] for i in order_idx]
    w_hrp     = hrp_weights(cov, ordered)
    plot_weights(w_hrp, "HRP Weights (Dendrogram-aware)")

    # --- Benchmarks: Equal-Weight and Global Min-Variance
    w_ew  = eq_weight(list(corr.columns))
    w_gmv = gmv_weights(cov)

    # --- Metrics
    def summarize(name, w):
        r, v, s, dd, nav = portfolio_metrics(rets, w)
        print(f"{name:>6} | AnnRet {r:7.2%} | Vol {v:6.2%} | Sharpe {s:5.2f} | MaxDD {dd:6.2%}")
        return nav

    print("\nPerformance summary (annualized):")
    nav_hrp = summarize("HRP", w_hrp)
    nav_ew  = summarize("EW",  w_ew)
    nav_gmv = summarize("GMV", w_gmv)

    plot_cum({"HRP": nav_hrp, "EW": nav_ew, "GMV": nav_gmv})
    plot_drawdowns({"HRP": nav_hrp, "EW": nav_ew, "GMV": nav_gmv})

    # --- Cluster exposures (weight by macro cluster)
    ce_hrp = cluster_exposures(w_hrp, labels)
    ce_ew  = cluster_exposures(w_ew,  labels)
    ce_gmv = cluster_exposures(w_gmv, labels)

    exposures = pd.DataFrame({"HRP": ce_hrp, "EW": ce_ew, "GMV": ce_gmv})
    print("\nWeight by macro cluster (sum to 1):")
    print(exposures.round(3).to_string())

    # --- Concentration checks
    print("\nWeight concentration diagnostics:")
    for name, w in [("HRP", w_hrp), ("EW", w_ew), ("GMV", w_gmv)]:
        hhi = float((w**2).sum())
        wmax = float(w.max())
        print(f"{name:>6} | HHI {hhi:0.4f} | Max single name weight {wmax:0.3f}")

    # Finally, draw all figures at once
    plt.show()


if __name__ == "__main__":
    main()
