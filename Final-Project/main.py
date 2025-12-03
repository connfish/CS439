# ===========================
# HRP Portfolio Optimization
# ===========================
# Works with a CSV: Date,Open,High,Low,Close,Adj Close,Volume,Symbol
# Example row:
# 2008-01-29,9.500000,9.990000,8.570000,8.750000,0.702589,1489000,AACG

import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# Try to import PyPortfolioOpt HRP for cross-checks (optional)
try:
    from pypfopt.hierarchical_risk_parity import HRPOpt as PpoHRPOpt
    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False

# Optional time-series bootstrap (stub)
try:
    from arch.bootstrap import StationaryBootstrap
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

# ---------------------------
# CONFIG
# ---------------------------
CSV_PATH = "history.csv"        # <-- point to your file
DATE_COL = "Date"
PRICE_COL = "Adj Close"
SYMBOL_COL = "Symbol"
ANNUALIZATION = 252


# ---------------------------
# 1) Load & prepare data
# ---------------------------
def load_prices(csv_path: str) -> pd.DataFrame:
    """Load CSV and pivot to wide prices: index=Date, columns=Symbol, values=Adj Close."""
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])
    # Keep only necessary columns
    df = df[[DATE_COL, SYMBOL_COL, PRICE_COL]].dropna()
    # Pivot to wide format
    prices = df.pivot(index=DATE_COL, columns=SYMBOL_COL, values=PRICE_COL).sort_index()
    # Drop columns with all NaNs and forward-fill small gaps if any
    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    return prices


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns."""
    rets = np.log(prices / prices.shift(1))
    return rets.dropna(how="all").dropna(axis=0)


# ---------------------------
# 2) Covariance / Correlation
# ---------------------------
def shrinkage_cov_corr(returns: pd.DataFrame):
    """Ledoit-Wolf shrinkage covariance and correlation."""
    lw = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    std = np.sqrt(np.diag(cov))
    corr = cov.div(std, axis=0).div(std, axis=1)
    return cov, corr


# ---------------------------
# 3) Clustering helpers
# ---------------------------
def corr_to_dist(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert correlation to distance using distance = sqrt(0.5*(1-corr))."""
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist.values, 0.0)
    return dist


def hierarchical_clustering(dist: pd.DataFrame, method: str = "ward"):
    """Run hierarchical clustering on condensed distance matrix."""
    # squareform expects condensed upper triangle
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=method)
    return Z


def plot_dendrogram(Z, labels, title="Hierarchical Clustering Dendrogram", figsize=(10,4)):
    plt.figure(figsize=figsize)
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------
# 4) HRP (manual implementation)
#    Following Lopez de Prado 2016
# ---------------------------
def get_quasi_diag(linkage_mat: np.ndarray) -> list:
    """
    Given a linkage matrix Z, perform the seriation (quasi-diagonalization)
    to get an ordered list of original asset indices.
    """
    # number of original items
    N = linkage_mat.shape[0] + 1

    # Start with the final cluster id
    def _extract_order(curr_id):
        """Depth-first traversal to retrieve leaf order."""
        if curr_id < N:
            return [curr_id]
        left = int(linkage_mat[curr_id - N, 0])
        right = int(linkage_mat[curr_id - N, 1])
        return _extract_order(left) + _extract_order(right)

    return _extract_order(2 * N - 2)  # Root cluster id


def get_ivp(cov: pd.DataFrame) -> np.ndarray:
    """Inverse-variance portfolio weights for a covariance submatrix."""
    iv = 1.0 / np.diag(cov)
    w = iv / iv.sum()
    return w


def get_cluster_var(cov: pd.DataFrame, items: list) -> float:
    """Cluster variance using IVP weights."""
    subcov = cov.loc[items, items].values
    w = get_ivp(pd.DataFrame(subcov))
    return float(w.T @ subcov @ w)


def hrp_allocation(cov: pd.DataFrame, ordered_tickers: list) -> pd.Series:
    """
    Recursive bisection to allocate weights along the hierarchy.
    cov: covariance dataframe
    ordered_tickers: list of tickers in quasi-diagonal order
    """
    weights = pd.Series(1.0, index=ordered_tickers)
    clusters = [ordered_tickers]

    while len(clusters) > 0:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = int(len(cluster) / 2)
            left = cluster[:split]
            right = cluster[split:]

            var_left = get_cluster_var(cov, left)
            var_right = get_cluster_var(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)

            weights[left] *= alpha
            weights[right] *= (1.0 - alpha)

            new_clusters.extend([left, right])
        clusters = new_clusters

    return (weights / weights.sum()).sort_values(ascending=False)


# ---------------------------
# 5) Evaluation
# ---------------------------
def portfolio_metrics(returns: pd.DataFrame, weights: pd.Series):
    """Compute annualized return, volatility, sharpe (no RF)."""
    w = weights.reindex(returns.columns).fillna(0).values
    mu_daily = returns.mean().values
    cov_daily = returns.cov().values

    ann_ret = float(np.dot(w, mu_daily) * ANNUALIZATION)
    ann_vol = float(np.sqrt(w @ (cov_daily * ANNUALIZATION) @ w))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe


def plot_weights(weights: pd.Series, title="HRP Portfolio Weights"):
    weights.sort_values(ascending=False).plot(kind="bar", figsize=(10,4))
    plt.ylabel("Weight")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(corr: pd.DataFrame, title="Correlation Matrix"):
    plt.figure(figsize=(8,6))
    im = plt.imshow(corr.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------
# 6) Linkage sensitivity (optional)
# ---------------------------
def linkage_sensitivity(returns: pd.DataFrame, methods=("single","complete","average","ward")):
    """Run HRP with different linkage methods and compare Sharpe."""
    cov, corr = shrinkage_cov_corr(returns)
    dist = corr_to_dist(corr)

    rows = []
    for m in methods:
        Z = hierarchical_clustering(dist, method=m)
        order_idx = get_quasi_diag(Z)
        tickers = list(corr.columns)
        ordered = [tickers[i] for i in order_idx]
        w = hrp_allocation(cov, ordered)
        r, v, s = portfolio_metrics(returns, w)
        rows.append({"method": m, "ann_return": r, "ann_vol": v, "sharpe": s})
    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    return out


# ---------------------------
# 7) Stationary bootstrap (stub)
# ---------------------------
def bootstrap_sharpe_stub(returns: pd.DataFrame, linkage_method="ward", reps=500, block_len=5, random_state=42):
    """
    Example stub for time-series stationary bootstrap to get a distribution of Sharpe ratios.
    Requires 'arch' package. If not available, returns None.
    """
    if not ARCH_AVAILABLE:
        print("arch not installed; skipping bootstrap.")
        return None

    rng = np.random.default_rng(random_state)
    cov, corr = shrinkage_cov_corr(returns)
    dist = corr_to_dist(corr)
    Z = hierarchical_clustering(dist, method=linkage_method)
    order_idx = get_quasi_diag(Z)
    tickers = list(corr.columns)
    ordered = [tickers[i] for i in order_idx]

    bs = StationaryBootstrap(block_len, returns.values, random_state=random_state)
    sharpes = []
    for _ in range(reps):
        sample = next(bs.bootstrap(1))[0][0]   # resampled (T x N)
        sample_df = pd.DataFrame(sample, columns=returns.columns, index=None)
        # Recompute cov on the sample
        lw = LedoitWolf().fit(sample_df.values)
        cov_s = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
        w = hrp_allocation(cov_s, ordered)
        r, v, s = portfolio_metrics(sample_df, w)
        sharpes.append(s)
    return pd.Series(sharpes, name=f"sharpe_{linkage_method}")


# ---------------------------
# 8) Main runner
# ---------------------------
def main():
    # Load
    prices = load_prices(CSV_PATH)
    print(f"Loaded prices: {prices.shape[0]} dates, {prices.shape[1]} symbols")

    # Filter symbols with too many NaNs (should be minimal after ffill/dropna)
    min_days = int(0.8 * prices.shape[0])
    valid = prices.columns[prices.notna().sum() >= min_days]
    prices = prices[valid]

    # Returns
    returns = to_log_returns(prices)
    print(f"Returns shape: {returns.shape}")

    # Cov/Correlation
    cov, corr = shrinkage_cov_corr(returns)

    # Plots
    plot_corr_heatmap(corr, title="Shrinkage Correlation (Ledoit-Wolf)")
    dist = corr_to_dist(corr)
    Z = hierarchical_clustering(dist, method="ward")
    plot_dendrogram(Z, labels=corr.columns, title="Dendrogram (ward)")

    # HRP (manual)
    order_idx = get_quasi_diag(Z)
    tickers = list(corr.columns)
    ordered = [tickers[i] for i in order_idx]
    weights_hrp = hrp_allocation(cov, ordered)
    print("\nManual HRP Weights:")
    print(weights_hrp.round(6))
    plot_weights(weights_hrp, title="HRP Weights (manual)")

    ann_ret, ann_vol, sharpe = portfolio_metrics(returns, weights_hrp)
    print(f"\nManual HRP — Expected Annual Return: {ann_ret:.2%}, Volatility: {ann_vol:.2%}, Sharpe: {sharpe:.2f}")

    # HRP via PyPortfolioOpt (if available) for verification
    if PYPFOPT_AVAILABLE:
        hrp = PpoHRPOpt(returns)
        weights_ppopt = pd.Series(hrp.optimize())
        weights_ppopt.name = "ppopt_hrp"
        print("\nPyPortfolioOpt HRP Weights:")
        print(weights_ppopt.round(6))
        plot_weights(weights_ppopt, title="HRP Weights (PyPortfolioOpt)")

    # Linkage sensitivity
    print("\nLinkage sensitivity (Sharpe ranked):")
    ls = linkage_sensitivity(returns)
    print(ls.to_string(index=False))

    # Bootstrap stub (optional)
    if ARCH_AVAILABLE:
        print("\nBootstrapping Sharpe (stub; this can take time on large data)...")
        s_series = bootstrap_sharpe_stub(returns, linkage_method="ward", reps=200, block_len=5, random_state=123)
        if s_series is not None:
            print(f"Bootstrap Sharpe (median) = {s_series.median():.3f}, 95% CI ≈ [{s_series.quantile(0.025):.3f}, {s_series.quantile(0.975):.3f}]")
            # Quick histogram
            plt.figure(figsize=(7,4))
            plt.hist(s_series.dropna().values, bins=30)
            plt.title("Bootstrapped Sharpe (ward)")
            plt.xlabel("Sharpe")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()
    else:
        print("\narch not installed — skip bootstrap (optional).")

if __name__ == "__main__":
    # For scripts: run main(). In a notebook, call main() in a cell.
    main()
