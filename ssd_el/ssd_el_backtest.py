"""
Statistical arbitrage with Second-order Stochastic Dominance (SSD) and
Empirical Likelihood (EL) scenario probabilities.

Pipeline, repeated at every monthly rebalancing date:
    1. Universe filter: assets with complete data and index membership over the
       whole lookback window.
    2. SD pre-test: find assets dominated by the benchmark (FSD / SSD / TSD).
    3. EL reweighting: minimum-relative-entropy probabilities so that the
       benchmark mean lies inside [L, U]. L and U are percentiles of the rolling
       benchmark means available *at that date* (expanding history, no look-ahead).
    4. SSD portfolio LP (CVaR / lower-partial-moment linearization) solved with
       SciPy HiGHS, once per reference distribution (index, dominated asset, 1/N).
    5. Out-of-sample wealth update for five strategies (3 spreads + 2 mean-reversion).

The first rebalancing happens on the first month-end on or after --start-date,
so that the history used to calibrate [L, U] is already populated.

Usage:
    python ssd_el_backtest.py --data path/to/Datisp100agg.xlsx --lookback 175 \
        --start-date 2007-01-01 --out results
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
from scipy.optimize import linprog, minimize

log = logging.getLogger("ssd_el")

BENCH = "sp100"
STRATEGY_LABELS = [
    "Strat 1 (vs Index)",
    "Strat 2 (vs Dominated)",
    "Strat 3 (vs 1/N)",
    "Strat 4 (MR on S1)",
    "Strat 5 (MR on S2)",
]
N_REFERENCES = 3  # index, dominated asset, equally weighted portfolio
N_STRATEGIES = 5


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
# Stochastic dominance pre-test
# --------------------------------------------------------------------------- #
def sd_dominance_test(asset_ret: pd.DataFrame, bench_ret: pd.Series):
    """Vectorized test of benchmark dominance over each asset.

    Returns (series, name, type, dominated_list). Among dominated assets the
    one with the lowest mean return is selected. If no asset is dominated the
    benchmark itself is returned with type "No SD".
    """
    assets = asset_ret.to_numpy()
    bench = bench_ret.to_numpy().ravel()

    # quantile-wise difference between sorted benchmark and each sorted asset
    diff = np.sort(bench)[:, None] - np.sort(assets, axis=0)
    cum_diff = np.cumsum(diff, axis=0)
    cum_cum = np.cumsum(cum_diff, axis=0)

    fsd = np.all(diff > 0, axis=0)
    ssd = np.all(cum_diff >= 0, axis=0) & ~fsd
    tsd = np.all(cum_cum >= 0, axis=0) & ~fsd & ~ssd

    status = np.zeros(assets.shape[1], dtype=int)  # priority FSD > SSD > TSD
    status[fsd], status[ssd], status[tsd] = 1, 2, 3
    dominated = np.where(status > 0)[0]

    if dominated.size == 0:
        return bench_ret, BENCH, "No SD", asset_ret.columns[:0]

    worst = dominated[np.argmin(assets[:, dominated].mean(axis=0))]
    kind = {1: "fsd", 2: "ssd", 3: "tsd"}[status[worst]]
    return asset_ret.iloc[:, worst], asset_ret.columns[worst], kind, asset_ret.columns[dominated]


# --------------------------------------------------------------------------- #
# Empirical Likelihood probabilities
# --------------------------------------------------------------------------- #
def el_bounds(rolling_means: pd.Series, curr: int, percentiles) -> tuple[float, float, int]:
    """Point-in-time [L, U]: percentiles of the rolling benchmark means whose
    window ends strictly before row `curr` (the first out-of-sample row).

    `rolling_means` is a backward-looking rolling mean of the full benchmark, so
    the value at position k depends only on rows <= k. Slicing to [:curr] uses
    no information from the out-of-sample period.
    Returns (L, U, number of rolling observations used).
    """
    past = rolling_means.iloc[:curr].dropna()
    lower, upper = np.percentile(past, percentiles)
    return float(lower), float(upper), len(past)


def el_weighting(factor_returns: np.ndarray, lower: float, upper: float):
    """Minimum relative-entropy (Empirical Likelihood) probabilities.

    If the sample mean lies in [lower, upper] the probabilities are 1/T
    (no tilt). Otherwise the mean is moved to the nearest bound; the tilting
    parameter is obtained from the dual problem  min_lam  -sum(log(1 + lam * g)).

    Returns (p, info) where info reports whether the tilt was active.
    """
    T = len(factor_returns)
    mean = float(np.mean(factor_returns))
    info = {"sample_mean": mean, "tilted": False, "side": None, "lam": 0.0, "converged": True}

    if lower <= mean <= upper:
        return np.full(T, 1.0 / T), info

    side = "below_L" if mean < lower else "above_U"
    target = lower if mean < lower else upper
    g = factor_returns - target

    def objective(lam):
        arg = 1.0 + lam * g
        return np.inf if np.any(arg <= 0) else -np.sum(np.log(arg))

    def jacobian(lam):
        return -np.sum(g / (1.0 + lam * g))

    res = minimize(objective, x0=np.array([0.0]), jac=jacobian, method="BFGS")
    lam = float(res.x[0])
    p = (1.0 / T) / (1.0 + lam * g)
    p = p / p.sum()

    info.update(tilted=True, side=side, lam=lam, converged=bool(res.success))
    return p, info


# --------------------------------------------------------------------------- #
# SSD portfolio optimization
# --------------------------------------------------------------------------- #
def optimize_ssd_el(X, y, p=None):
    """Maximize expected return under probabilities p subject to SSD over y.

    SSD is imposed as lower-partial-moment (expected shortfall) constraints at
    every benchmark scenario y_s:
        sum_t p_t * max(0, y_s - x_t.w)  <=  sum_t p_t * max(0, y_s - y_t)
    The max(.) is linearized with theta[s, t] >= y_s - x_t.w, theta >= 0,
    giving K + T^2 variables and T + T^2 inequality constraints.

    X : (T, K) asset returns, y : (T,) reference returns, p : (T,) probabilities.
    Returns the weight vector (K,), or None if the solver fails.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    T, K = X.shape
    p = np.full(T, 1.0 / T) if p is None else np.asarray(p, dtype=float)

    # Right-hand side: exact LPM of the reference at every threshold y_s
    lpm_ref = np.maximum(0.0, y[:, None] - y[None, :]) @ p

    n_vars = K + T * T
    c = np.zeros(n_vars)
    c[:K] = -(p @ X)

    A_eq = sparse.csr_matrix(([1.0] * K, ([0] * K, range(K))), shape=(1, n_vars))
    b_eq = np.array([1.0])

    # Set 1 (T rows): sum_t p_t * theta[s, t] <= lpm_ref[s]
    A1 = sparse.hstack(
        [sparse.csr_matrix((T, K)), sparse.kron(sparse.identity(T), p[None, :])],
        format="csr",
    )
    # Set 2 (T^2 rows, row s*T+t): -theta[s, t] - x_t.w <= -y_s
    A2 = sparse.hstack(
        [sparse.csr_matrix(np.tile(-X, (T, 1))), -sparse.identity(T * T)],
        format="csr",
    )

    A_ub = sparse.vstack([A1, A2], format="csr")
    b_ub = np.concatenate([lpm_ref, -np.repeat(y, T)])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    if not res.success:
        log.warning("Solver failed: %s", res.message)
        return None
    return res.x[:K]


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def monthly_rebalance_locations(index: pd.DatetimeIndex) -> list[int]:
    """Integer positions of the last trading day of every month."""
    last = index.to_series().groupby(index.to_period("M")).last()
    return [index.get_loc(d) for d in last]


def run_backtest(returns, composition, benchmark, lookback, el_percentiles=(10, 90),
                 start_date=None, min_el_obs=250, exclude_targets=("Lehman",),
                 mr_mode="inverse"):
    locs = monthly_rebalance_locations(returns.index)
    log.info("Month-end rebalancing dates available: %d", len(locs))

    # Backward-looking rolling means of the benchmark (used point-in-time)
    rolling_means = benchmark.rolling(lookback).mean()
    n_rolling = rolling_means.notna().to_numpy().cumsum()   # count up to each row (inclusive)

    start_pos = returns.index.searchsorted(pd.Timestamp(start_date)) if start_date else 0
    first = next(
        (i for i, loc in enumerate(locs[:-1])
         if loc >= start_pos
         and loc - lookback >= 0
         and n_rolling[loc - 1] >= min_el_obs),
        None,
    )
    if first is None:
        raise ValueError(
            "No rebalancing date satisfies: on/after start date, a full lookback window, "
            f"and at least {min_el_obs} rolling benchmark means of history. "
            "Use an earlier --start-date or a smaller --min-el-obs."
        )

    start_row = locs[first]
    log.info("First rebalancing: %s (%d rolling means available for [L, U])",
             returns.index[start_row].date(), n_rolling[start_row - 1])

    W = np.zeros((len(returns), N_STRATEGIES))
    tt = start_row - 1              # wealth row *before* the first out-of-sample day
    W[tt, :] = 1.0

    weights_history, pool_history, target_history = {}, {}, {}
    el_records = []
    failures = 0

    for i in range(first, len(locs) - 1):
        curr, nxt = locs[i], locs[i + 1]
        curr_date = returns.index[curr]
        window = slice(curr - lookback, curr)
        freq = nxt - curr

        raw_R = returns.iloc[window]
        raw_C = composition.iloc[window]
        bench_is = benchmark.iloc[window].fillna(0.0)

        # Universe: complete data AND index member over the whole window
        valid = raw_R.notna().all(axis=0) & (raw_C.fillna(0) == 1).all(axis=0)
        is_R = raw_R.loc[:, valid]

        if is_R.empty:
            log.info("%s: no valid assets, holding wealth flat.", curr_date.date())
            W[tt + 1: tt + freq + 1, :] = W[tt, :]
            tt += freq
            continue

        t0 = time.time()
        tgt_series, tgt_name, dom_type, dom_list = sd_dominance_test(is_R, bench_is)
        references = [
            bench_is,
            tgt_series if tgt_name != BENCH else bench_is,
            is_R.mean(axis=1),
        ]

        # Point-in-time EL probabilities
        lower, upper, n_obs = el_bounds(rolling_means, curr, el_percentiles)
        el_probs, el_info = el_weighting(bench_is.to_numpy(), lower, upper)
        el_records.append({"date": curr_date, "L": lower, "U": upper,
                           "n_rolling_obs": n_obs, **el_info})

        portfolios = np.zeros((N_REFERENCES, is_R.shape[1]))
        for s, ref in enumerate(references):
            w = optimize_ssd_el(is_R, ref, p=el_probs)
            if w is None or w.sum() <= 0:
                w = np.ones(is_R.shape[1]) / is_R.shape[1]
                failures += 1
            else:
                w = np.where(w < 1e-5, 0.0, w)
                w = w / w.sum()
            portfolios[s] = w
        elapsed = time.time() - t0

        weights_history[curr_date] = pd.DataFrame(portfolios, columns=is_R.columns)
        pool_history[curr_date] = dom_list
        target_history[curr_date] = tgt_name

        # ---- out-of-sample wealth update -------------------------------------
        X_oos = returns.iloc[curr:nxt][is_R.columns].fillna(0.0).to_numpy()
        oos_index = benchmark.iloc[curr:nxt].fillna(0.0).to_numpy()
        if tgt_name != BENCH and not any(k in tgt_name for k in exclude_targets):
            oos_dom = returns.iloc[curr:nxt][tgt_name].fillna(0.0).to_numpy()
        else:
            oos_dom = oos_index
        oos_refs = np.column_stack([oos_index, oos_dom, X_oos.mean(axis=1)])

        factors = 1.0 + (X_oos @ portfolios.T - oos_refs)   # 1 + spread vs reference

        if mr_mode == "inverse":
            mr = 1.0 / (factors[:, :2] + 1e-10)
        elif mr_mode == "short":
            mr = 1.0 - (factors[:, :2] - 1.0)
        else:
            raise ValueError("mr_mode must be 'inverse' or 'short'")

        growth = np.hstack([factors, mr])
        W[tt + 1: tt + freq + 1, :] = W[tt, :] * np.cumprod(growth, axis=0)
        tt += freq

        log.info("%s | assets: %d | dominated: %s (%s) | EL tilt: %s | %.1fs | wealth: %s",
                 curr_date.date(), is_R.shape[1], tgt_name, dom_type,
                 el_info["side"] or "no", elapsed, np.round(W[tt], 4))

    log.info("Backtest complete. Optimization failures: %d", failures)

    rows = slice(start_row - 1, tt + 1)
    wealth = pd.DataFrame(W[rows], index=returns.index[rows], columns=STRATEGY_LABELS)

    b = benchmark.iloc[rows].copy()
    b.iloc[0] = 0.0
    wealth[BENCH] = (1.0 + b).cumprod()

    el_diag = pd.DataFrame(el_records).set_index("date")
    return wealth, weights_history, pool_history, target_history, failures, el_diag


def summarize_el(el_diag: pd.DataFrame) -> dict:
    """Counter of EL tilting across optimized rebalancing dates."""
    n = len(el_diag)
    tilted = int(el_diag["tilted"].sum())
    return {
        "rebalancing_dates": n,
        "tilt_active": tilted,
        "tilt_active_pct": 100.0 * tilted / n if n else float("nan"),
        "tilt_below_L": int((el_diag["side"] == "below_L").sum()),
        "tilt_above_U": int((el_diag["side"] == "above_U").sum()),
        "el_not_converged": int((el_diag["tilted"] & ~el_diag["converged"]).sum()),
    }


# --------------------------------------------------------------------------- #
# Performance metrics and plots
# --------------------------------------------------------------------------- #
def calculate_metrics(wealth: pd.Series, periods_per_year=252, risk_free_rate=0.0) -> dict:
    rets = wealth.pct_change().dropna()

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


def plot_wealth(wealth: pd.DataFrame, columns: list[str], title: str, path: Path):
    colors = dict(zip(STRATEGY_LABELS, ["cyan", "lime", "magenta", "yellow", "orange"]))
    styles = dict(zip(STRATEGY_LABELS, ["-", "-", "-", ":", ":"]))

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(14, 8))
        for col in columns:
            ax.plot(wealth.index, wealth[col], label=col, color=colors[col],
                    linestyle=styles[col], linewidth=1.5)
            ax.annotate(f"{wealth[col].iloc[-1]:.2f}x", xy=(wealth.index[-1], wealth[col].iloc[-1]),
                        xytext=(5, 0), textcoords="offset points", color=colors[col],
                        fontweight="bold", va="center")
        ax.plot(wealth.index, wealth[BENCH], label="Benchmark (SP100)", color="white",
                linestyle="--", linewidth=2.5, alpha=0.8)
        ax.annotate(f"{wealth[BENCH].iloc[-1]:.2f}x", xy=(wealth.index[-1], wealth[BENCH].iloc[-1]),
                    xytext=(5, 0), textcoords="offset points", color="white",
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
    ap.add_argument("--lookback", type=int, default=175, help="Lookback window T in trading days")
    ap.add_argument("--start-date", default="2007-01-01",
                    help="First rebalancing on/after this date (default: 2007-01-01). "
                         "History before it is used only to calibrate [L, U].")
    ap.add_argument("--min-el-obs", type=int, default=250,
                    help="Minimum number of rolling benchmark means required to calibrate "
                         "[L, U] at the first rebalancing (default: 250)")
    ap.add_argument("--el-percentiles", type=float, nargs=2, default=(10, 90),
                    metavar=("LOW", "HIGH"),
                    help="Percentiles of rolling benchmark means defining [L, U]")
    ap.add_argument("--mr-mode", choices=["inverse", "short"], default="inverse",
                    help="Mean-reversion strategies: 1/(1+spread) or 1-spread")
    ap.add_argument("--exclude-target", nargs="*", default=["Lehman"],
                    help="Dominated-asset names containing these strings use the index "
                         "as out-of-sample reference")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    returns, composition, benchmark = load_data(args.data)

    wealth, weights, pools, targets, failures, el_diag = run_backtest(
        returns, composition, benchmark, args.lookback,
        el_percentiles=tuple(args.el_percentiles),
        start_date=args.start_date, min_el_obs=args.min_el_obs,
        exclude_targets=tuple(args.exclude_target), mr_mode=args.mr_mode,
    )

    el_summary = summarize_el(el_diag)
    log.info("EL tilt active in %d of %d rebalancing dates (%.1f%%): %d below L, %d above U; "
             "%d tilts did not converge.",
             el_summary["tilt_active"], el_summary["rebalancing_dates"],
             el_summary["tilt_active_pct"], el_summary["tilt_below_L"],
             el_summary["tilt_above_U"], el_summary["el_not_converged"])

    metrics = pd.DataFrame({c: calculate_metrics(wealth[c]) for c in wealth.columns})
    print(metrics.round(4))

    wealth.to_csv(out / "wealth.csv")
    metrics.to_csv(out / "metrics.csv")
    el_diag.to_csv(out / "el_diagnostics.csv")
    pd.to_pickle({"weights": weights, "dominated_pool": pools, "dominated_target": targets},
                 out / "history.pkl")
    plot_wealth(wealth, STRATEGY_LABELS, "Wealth comparison: all strategies (EL framework)",
                out / "wealth_all.png")
    plot_wealth(wealth, [STRATEGY_LABELS[1], STRATEGY_LABELS[4]],
                "Wealth comparison: Strat 2, Strat 5 and benchmark (EL framework)",
                out / "wealth_dominated_vs_reverse.png")
    log.info("Optimization failures: %d. Results written to %s", failures, out.resolve())


if __name__ == "__main__":
    main()
