"""
Second-order Stochastic Dominance (SSD) portfolio optimizer.

Formulation: Kuosmanen (2004), doubly stochastic matrix. Find the long-only,
fully invested portfolio with the highest mean return whose return distribution
SSD-dominates a reference distribution (the index or the equally weighted
portfolio) over a lookback window of T scenarios:

    max_{w, W}   (1/T) * sum_t (R w)_t
    s.t.         R w >= W y                      (state-by-state, W y = anti-spread of y)
                 sum_i w_i = 1,  w >= 0
                 W >= 0,  sum_s W[t, s] = 1  for all t,  sum_t W[t, s] = 1  for all s

Variables: N + T^2. Constraints: T inequalities, 2T + 1 equalities.

Usage:
    python ssd_optimizer.py --data file.xlsx            # monthly out-of-sample evaluation
    python ssd_optimizer.py --data file.xlsx --latest   # weights for the last window only
The optimizer itself can also be imported: from ssd_optimizer import ssd_optimize
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

log = logging.getLogger("ssd_optimizer")

BENCH = "sp100"
WEIGHT_TOL = 1e-5   # weights below this are set to zero and the rest renormalized


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Read the Excel workbook. Required sheets: df_ret_final, composition,
    Date, nomi, Index (all read without headers)."""
    sheets = pd.read_excel(path, sheet_name=None, header=None)

    dates = pd.DatetimeIndex(pd.to_datetime(sheets["Date"].iloc[:, 0]), name="Date")
    names = sheets["nomi"].iloc[0, :].tolist()

    returns = sheets["df_ret_final"].copy()
    composition = sheets["composition"].copy()
    for df in (returns, composition):
        df.index = dates
        df.columns = names

    benchmark = sheets["Index"].iloc[:, 0].fillna(0.0)
    benchmark.index = dates
    benchmark.name = BENCH

    dup = returns.columns.duplicated()
    if dup.any():
        log.info("Dropping duplicate asset names: %s", returns.columns[dup].tolist())
        returns = returns.loc[:, ~dup]
        composition = composition.loc[:, ~composition.columns.duplicated()]

    return returns, composition, benchmark


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #
def ssd_optimize(asset_returns, reference_returns):
    """SSD-dominating portfolio with maximum mean return (Kuosmanen 2004).

    asset_returns     : (T, N) array-like of asset returns (scenarios equally likely)
    reference_returns : (T,) array-like, the distribution to be dominated
    Returns the weight vector (N,), or None if the solver fails
    (infeasible or numerical problem).
    """
    R = np.asarray(asset_returns, dtype=float)
    y = np.asarray(reference_returns, dtype=float).ravel()
    T, N = R.shape
    if y.shape[0] != T:
        raise ValueError("asset_returns and reference_returns must have the same length")
    # Decision vector x = [w | vec(W)]: N portfolio weights followed by the T x T
    # doubly stochastic matrix flattened row-major, so W[t, s] sits at N + t*T + s.
    # W is only an auxiliary device to encode SSD linearly; it costs T^2 variables,
    # which is what drives the runtime.
    n_vars = N + T * T

    # Objective: maximize mean portfolio return (linprog minimizes, hence the minus).
    # Only w enters the objective; the W block has zero cost.
    c = np.concatenate([-R.mean(axis=0), np.zeros(T * T)])

    # Dominance constraint R w >= W y, written as -R w + W y <= 0.
    # kron(I_T, y') places y on the t-th block, so row t computes sum_s W[t, s] * y[s].
    A_ub = sparse.hstack(
        [-sparse.csr_matrix(R), sparse.kron(sparse.identity(T), y.reshape(1, T))],
        format="csr",
    )
    b_ub = np.zeros(T)

    # Equalities: budget constraint sum(w) = 1, plus the doubly stochastic conditions.
    # kron(I_T, 1') sums each row of W; kron(1', I_T) sums each column of W.
    # (One of the 2T sum constraints is linearly redundant; HiGHS presolve removes it.)
    A_eq = sparse.vstack(
        [
            sparse.hstack([sparse.csr_matrix(np.ones((1, N))), sparse.csr_matrix((1, T * T))]),
            sparse.hstack([sparse.csr_matrix((T, N)), sparse.kron(sparse.identity(T), np.ones((1, T)))]),
            sparse.hstack([sparse.csr_matrix((T, N)), sparse.kron(np.ones((1, T)), sparse.identity(T))]),
        ],
        format="csr",
    )
    b_eq = np.concatenate([[1.0], np.ones(T), np.ones(T)])

    # Bounds: w >= 0 (no short selling); 0 <= W[t, s] <= 1 (valid redistribution weights)
    lower = np.zeros(n_vars)
    upper = np.concatenate([np.full(N, np.inf), np.ones(T * T)])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=np.column_stack([lower, upper]), method="highs")
    if not res.success:
        # Typically infeasible: no long-only portfolio can dominate the reference.
        log.warning("Solver failed: %s", res.message)
        return None
    return res.x[:N]         # W is discarded; only the portfolio weights are needed


def clean_weights(w: np.ndarray, tol: float = WEIGHT_TOL) -> np.ndarray:
    """Zero out numerical residuals (simplex leaves ~1e-10 weights) and renormalize."""
    w = np.where(w < tol, 0.0, w)
    return w / w.sum()


def ssd_min_slack(weights, asset_returns, reference_returns) -> float:
    """In-sample SSD check: min over k of the cumulative difference between the sorted
    portfolio returns and the sorted reference returns, divided by T.
    A value >= 0 (up to solver tolerance) means the portfolio SSD-dominates the reference.

    For equally likely scenarios, SSD is equivalent to the cumulative sums of the sorted
    portfolio returns never falling below those of the sorted reference (Lorenz-curve
    ordering). This is an independent check of the LP solution, not part of the LP.
    """
    p = np.sort(np.asarray(asset_returns, dtype=float) @ np.asarray(weights, dtype=float))
    y = np.sort(np.asarray(reference_returns, dtype=float).ravel())
    return float(np.min(np.cumsum(p - y)) / len(y))


# --------------------------------------------------------------------------- #
# Window preparation shared by the rolling run and the latest-date run
# --------------------------------------------------------------------------- #
def select_assets(returns: pd.DataFrame, composition: pd.DataFrame, rows: slice) -> pd.DataFrame:
    """Investable universe for the estimation window `rows`.

    An asset is kept only if it has complete returns AND is an index member on every day
    of the window. This keeps delisted or not-yet-listed stocks out of the sample, and the
    composition matrix limits survivorship bias.
    """
    raw_R = returns.iloc[rows]
    raw_C = composition.iloc[rows]
    valid = raw_R.notna().all(axis=0) & (raw_C.fillna(0) == 1).all(axis=0)
    return raw_R.loc[:, valid]


def build_reference(is_R: pd.DataFrame, benchmark: pd.Series, rows: slice, reference: str) -> pd.Series:
    """Return distribution that the optimized portfolio has to SSD-dominate."""
    if reference == "index":
        return benchmark.iloc[rows].fillna(0.0)
    if reference == "equal_weight":
        return is_R.mean(axis=1)          # 1/N portfolio of the selected assets
    raise ValueError("reference must be 'index' or 'equal_weight'")


def optimize_window(is_R: pd.DataFrame, ref: pd.Series):
    """Solve the SSD problem on a prepared window.

    Returns (weights Series indexed by asset, info dict). On solver failure the
    weights fall back to 1/N and info['fallback'] is True.
    """
    t0 = time.time()
    raw = ssd_optimize(is_R, ref)
    fallback = raw is None or raw.sum() <= 0
    if fallback:
        # Keep the backtest running with a neutral portfolio; failures are counted and logged.
        w = np.ones(is_R.shape[1]) / is_R.shape[1]
    else:
        w = clean_weights(raw)
    info = {
        "assets": is_R.shape[1],
        "fallback": fallback,
        "solve_seconds": time.time() - t0,
        "in_sample_mean_portfolio": float((is_R.to_numpy() @ w).mean()),
        "in_sample_mean_reference": float(ref.mean()),
        "ssd_min_slack": ssd_min_slack(w, is_R, ref),
    }
    return pd.Series(w, index=is_R.columns), info


# --------------------------------------------------------------------------- #
# Rolling evaluation
# --------------------------------------------------------------------------- #
def monthly_rebalance_locations(index: pd.DatetimeIndex) -> list[int]:
    """Integer positions of the last trading day of every month."""
    last = index.to_series().groupby(index.to_period("M")).last()
    return [index.get_loc(d) for d in last]


def run_rolling(returns, composition, benchmark, lookback, reference="index", start_date=None):
    """Monthly rebalancing. The weights chosen at a month-end use the `lookback` rows
    before it and are applied (constant weights, daily rebalanced) until the next one.

    Timing (no look-ahead): the estimation window is rows [curr - lookback, curr), i.e.
    it excludes the rebalancing day; that day's return is already out-of-sample.
    """
    # Rebalancing on the last trading day actually present in the data for each month
    # (a calendar month-end that falls on a weekend would otherwise drop the month).
    locs = monthly_rebalance_locations(returns.index)
    start_pos = returns.index.searchsorted(pd.Timestamp(start_date)) if start_date else 0
    first = next((i for i, loc in enumerate(locs[:-1])
                  if loc >= start_pos and loc - lookback >= 0), None)
    if first is None:
        raise ValueError("No rebalancing date with a full lookback window on/after the start date.")

    start_row = locs[first]
    end_row = locs[-1]                      # out-of-sample rows: [start_row, end_row)
    log.info("Month-end rebalancing dates: %d | first: %s | optimizations: %d",
             len(locs), returns.index[start_row].date(), len(locs) - 1 - first)

    port_ret = np.zeros(len(returns))       # daily out-of-sample portfolio return
    weights_history, records = {}, []

    for i in range(first, len(locs) - 1):
        curr, nxt = locs[i], locs[i + 1]
        curr_date = returns.index[curr]

        window = slice(curr - lookback, curr)
        is_R = select_assets(returns, composition, window)
        if is_R.empty:
            log.info("%s: empty universe, holding cash for the period.", curr_date.date())
            records.append({"date": curr_date, "assets": 0, "fallback": np.nan})
            continue
        ref = build_reference(is_R, benchmark, window, reference)

        w, info = optimize_window(is_R, ref)

        # Out-of-sample: apply the weights from the rebalancing day up to the next one.
        # Missing returns of held assets (e.g. delistings) are treated as 0 and reported.
        X_oos = returns.iloc[curr:nxt][is_R.columns]
        held = w[w > 0].index
        info["held_assets"] = int(len(held))
        info["held_with_missing_oos"] = int(X_oos[held].isna().any().sum())
        port_ret[curr:nxt] = X_oos.fillna(0.0).to_numpy() @ w.to_numpy()

        weights_history[curr_date] = w
        records.append({"date": curr_date, **info})
        log.info("%s | assets: %d | held: %d | %.1fs | in-sample mean: portfolio %.5f vs "
                 "reference %.5f%s",
                 curr_date.date(), info["assets"], info["held_assets"], info["solve_seconds"],
                 info["in_sample_mean_portfolio"], info["in_sample_mean_reference"],
                 " | FALLBACK 1/N" if info["fallback"] else "")

    # Build the wealth paths on exactly the out-of-sample rows so the portfolio and the
    # benchmark start together at 1.0 on the row before the first rebalancing.
    rows = slice(start_row, end_row)
    daily = pd.DataFrame({"SSD portfolio": port_ret[rows],
                          BENCH: benchmark.iloc[rows].fillna(0.0).to_numpy()},
                         index=returns.index[rows])
    wealth = (1.0 + daily).cumprod()
    # initial wealth of 1 on the row before the first out-of-sample day
    wealth.loc[returns.index[start_row - 1]] = 1.0
    wealth = wealth.sort_index()

    weights_df = pd.DataFrame(weights_history).T.fillna(0.0)
    log_df = pd.DataFrame(records).set_index("date")
    return wealth, weights_df, log_df


# --------------------------------------------------------------------------- #
# Performance metrics and plot
# --------------------------------------------------------------------------- #
def calculate_metrics(wealth: pd.Series, periods_per_year=252, risk_free_rate=0.0) -> dict:
    rets = wealth.pct_change().dropna()
    if rets.empty:
        return {}

    total = wealth.iloc[-1] / wealth.iloc[0] - 1
    ann_ret = (1 + total) ** (periods_per_year / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol != 0 else np.nan

    index = (1 + rets).cumprod()
    dd = (index - index.cummax()) / index.cummax()
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    # Downside volatility = std of negative returns only
    down_vol = rets[rets < 0].std() * np.sqrt(periods_per_year)
    sortino = (ann_ret - risk_free_rate) / down_vol if down_vol != 0 else np.nan

    longest = run = 0
    for under in (dd < 0):
        run = run + 1 if under else 0
        longest = max(longest, run)

    return {
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Max Drawdown Duration (days)": longest,
    }


def plot_wealth(wealth: pd.DataFrame, path: Path, title: str):
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(wealth.index, wealth["SSD portfolio"], color="cyan", linewidth=1.5,
                label="SSD portfolio")
        ax.plot(wealth.index, wealth[BENCH], color="white", linestyle="--", linewidth=2.2,
                alpha=0.8, label="Benchmark (SP100)")
        for col, color in (("SSD portfolio", "cyan"), (BENCH, "white")):
            ax.annotate(f"{wealth[col].iloc[-1]:.2f}x", xy=(wealth.index[-1], wealth[col].iloc[-1]),
                        xytext=(5, 0), textcoords="offset points", color=color,
                        fontweight="bold", va="center")
        ax.set_title(title, fontsize=16, loc="left")
        ax.set_ylabel("Wealth (log scale)")
        ax.set_xlabel("Date")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(loc="upper left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="Path to the Excel workbook")
    ap.add_argument("--out", default="results", help="Output directory (default: results)")
    ap.add_argument("--lookback", type=int, default=250, help="Lookback window T in trading days (default: 250)")
    ap.add_argument("--reference", choices=["index", "equal_weight"], default="index",
                    help="Distribution to be SSD-dominated (default: index)")
    ap.add_argument("--start-date", default=None,
                    help="Rolling run: first rebalancing on/after this date (default: earliest possible)")
    ap.add_argument("--latest", action="store_true",
                    help="Solve once on the last T rows of the data and write the weights "
                         "(no backtest)")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    returns, composition, benchmark = load_data(args.data)

    if args.latest:
        # No out-of-sample period follows, so the window includes the last available row.
        window = slice(len(returns) - args.lookback, len(returns))
        is_R = select_assets(returns, composition, window)
        if is_R.empty:
            raise ValueError("Empty universe in the last lookback window.")
        ref = build_reference(is_R, benchmark, window, args.reference)
        w, info = optimize_window(is_R, ref)
        w = w[w > 0].sort_values(ascending=False)
        w.rename("weight").to_csv(out / "latest_weights.csv", header=True)
        log.info("Window: %s to %s | assets: %d | held: %d | SSD slack: %.2e%s",
                 returns.index[window.start].date(), returns.index[-1].date(),
                 info["assets"], len(w), info["ssd_min_slack"],
                 " | FALLBACK 1/N" if info["fallback"] else "")
        print(w.round(4).to_string())
        return

    wealth, weights, opt_log = run_rolling(returns, composition, benchmark, args.lookback,
                                           reference=args.reference, start_date=args.start_date)
    n_fail = int(opt_log["fallback"].fillna(0).astype(bool).sum())
    log.info("Optimization failures (fallback to 1/N): %d", n_fail)

    metrics = pd.DataFrame({c: calculate_metrics(wealth[c]) for c in wealth.columns})
    print(metrics.round(4))

    wealth.to_csv(out / "wealth.csv")
    weights.to_csv(out / "weights.csv")
    opt_log.to_csv(out / "optimization_log.csv")
    metrics.to_csv(out / "metrics.csv")
    plot_wealth(wealth, out / "wealth.png",
                f"SSD portfolio vs benchmark (reference: {args.reference}, T = {args.lookback})")
    log.info("Results written to %s", out.resolve())


if __name__ == "__main__":
    main()
