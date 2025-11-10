#!/usr/bin/env python3
"""
Visualize large stock history CSVs by symbol and date range.

CSV schema (header required):
Date,Open,High,Low,Close,Adj Close,Volume,Symbol

Outputs:
- PNG chart(s) in --outdir (if --save)
- On-screen chart (unless --no-show)

Examples:
  # List symbols found in the CSV
  python visualize_history.py --csv history.csv --list-symbols

  # Plot AAPL for 2020-01-01 to 2024-12-31 and save PNG
  python visualize_history.py --csv history.csv --symbol AAPL --start 2020-01-01 --end 2024-12-31 --save

Tip: On very large files, add --fast to downsample to business-day means.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# I/O helpers (chunked reads)
# ---------------------------

USECOLS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Symbol"]
DTYPE = {
    "Open": "float64", "High": "float64", "Low": "float64",
    "Close": "float64", "Adj Close": "float64",
    "Volume": "float64",   # volume may exceed int32; keep as float for safety
    "Symbol": "string",
}

def list_available_symbols(csv_path: str, chunksize: int = 500_000) -> List[str]:
    symbols = set()
    for chunk in pd.read_csv(
        csv_path, usecols=["Symbol"], dtype={"Symbol": "string"},
        chunksize=chunksize, low_memory=True
    ):
        symbols.update(chunk["Symbol"].dropna().unique().tolist())
    syms = sorted(s for s in symbols if s)
    return syms


def load_symbol_frame(
    csv_path: str,
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    frames = []
    for chunk in pd.read_csv(
        csv_path, usecols=USECOLS, dtype=DTYPE, chunksize=chunksize, low_memory=True
    ):
        # Filter symbol within chunk
        c = chunk[chunk["Symbol"] == symbol]
        if not c.empty:
            frames.append(c)

    if not frames:
        raise ValueError(f"No rows found for symbol '{symbol}' in {csv_path}")

    df = pd.concat(frames, ignore_index=True)

    # Parse dates, sort, and de-dup just in case
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"])

    # Filter date range
    if start:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Date"] <= pd.to_datetime(end)]

    if df.empty:
        raise ValueError(
            f"No rows remain for {symbol} after applying date range "
            f"(start={start}, end={end})."
        )

    df = df.set_index("Date")
    return df


# ---------------------------
# Plotting
# ---------------------------

def plot_history(
    df: pd.DataFrame,
    symbol: str,
    outdir: Optional[str] = None,
    save: bool = False,
    show: bool = True,
    ma_windows: Iterable[int] = (20, 50, 200),
    fast: bool = False,
) -> Optional[str]:
    # Optional downsample for speed on giant frames
    if fast:
        # Business-day frequency average
        df_plot = df.resample("B").mean(numeric_only=True)
    else:
        df_plot = df

    # Compute moving averages
    for w in ma_windows:
        df_plot[f"MA{w}"] = df_plot["Close"].rolling(w, min_periods=1).mean()

    # Build figure: price on top, volume below (shared x)
    fig = plt.figure(figsize=(12, 7))
    ax_price = fig.add_axes([0.08, 0.35, 0.88, 0.58])  # [left, bottom, width, height]
    ax_vol = fig.add_axes([0.08, 0.10, 0.88, 0.20], sharex=ax_price)

    # Price lines
    ax_price.plot(df_plot.index, df_plot["Close"], label="Close", linewidth=1.2)
    for w in ma_windows:
        ax_price.plot(df_plot.index, df_plot[f"MA{w}"], label=f"MA{w}", linewidth=1.0)

    ax_price.set_title(f"{symbol} — Close & Moving Averages")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")
    ax_price.grid(True, alpha=0.25)

    # Volume bars
    ax_vol.bar(df_plot.index, df_plot["Volume"].fillna(0.0), width=1.0)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(True, axis="y", alpha=0.25)

    # Tidy x labels
    fig.autofmt_xdate()

    saved_path = None
    if save:
        outdir_path = Path(outdir or "charts")
        outdir_path.mkdir(parents=True, exist_ok=True)
        # Build filename from min/max date visible
        idx = df_plot.index
        start_str = idx.min().strftime("%Y%m%d")
        end_str = idx.max().strftime("%Y%m%d")
        saved_path = outdir_path / f"{symbol}_{start_str}_{end_str}.png"
        fig.savefig(saved_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(saved_path) if saved_path else None


# ---------------------------
# CLI
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize a large stock history CSV by symbol/date range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="Path to history.csv")
    p.add_argument("--symbol", help="Ticker symbol to plot (e.g., AAPL)")
    p.add_argument("--start", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", help="End date (YYYY-MM-DD)")
    p.add_argument("--list-symbols", action="store_true", help="List available symbols and exit")
    p.add_argument("--outdir", default="charts", help="Directory to save charts")
    p.add_argument("--save", action="store_true", help="Save chart as PNG")
    p.add_argument("--no-show", action="store_true", help="Do not display the chart window")
    p.add_argument("--ma", nargs="*", type=int, default=[20, 50, 200],
                   help="Moving-average windows (space-separated, e.g., --ma 20 100)")
    p.add_argument("--fast", action="store_true", help="Downsample to business-day means for speed")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    # Parse known args to avoid Jupyter/IPython's extra flags if someone runs it there
    args, _ = parser.parse_known_args(argv)

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")

    if args.list_symbols:
        syms = list_available_symbols(csv_path)
        if not syms:
            print("No symbols found.")
        else:
            print(f"{len(syms)} symbol(s) found:")
            print(", ".join(syms))
        return

    if not args.symbol:
        # If not provided, try to pick the first symbol in the file
        syms = list_available_symbols(csv_path)
        if not syms:
            raise SystemExit("No symbols found; cannot auto-select a symbol.")
        args.symbol = syms[0]
        print(f"[info] --symbol not provided; using first symbol found: {args.symbol}")

    df = load_symbol_frame(csv_path, args.symbol, args.start, args.end)
    out = plot_history(
        df=df,
        symbol=args.symbol,
        outdir=args.outdir,
        save=bool(args.save),
        show=not args.no_show,
        ma_windows=args.ma,
        fast=args.fast,
    )
    if out:
        print(f"Saved chart: {out}")


if __name__ == "__main__":
    main()
