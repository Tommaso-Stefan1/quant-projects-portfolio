# SSD Portfolio Optimizer (Kuosmanen doubly stochastic matrix)

Long-only portfolio optimizer under **second-order stochastic dominance (SSD)**: it finds the portfolio with the highest mean return whose return distribution SSD-dominates a reference distribution (the index or the equally weighted portfolio) over a lookback window. It can be used as a library, to get the latest weights, or in a rolling monthly evaluation.

## Computational Note: Execution Time

- **Algorithm:** Kuosmanen (2004) SSD Doubly Stochastic Matrix.
- **Complexity:** O(T^2) variables per optimization (250^2 = 62,500 variables).
- **Estimated Runtime:** ~4-7 hours using the SciPy HiGHS solver.

T is the lookback window (`--lookback`, default 250).

## Method

For T equally likely scenarios, `R` is the (T × N) matrix of asset returns and `y` the reference returns:

```
max_{w, W}   (1/T) · Σ_t (R w)_t
s.t.         R w ≥ W y
             Σ_i w_i = 1,   w ≥ 0
             W ≥ 0,   Σ_s W[t,s] = 1 ∀t,   Σ_t W[t,s] = 1 ∀s
```

`W` is a doubly stochastic matrix, so `W y` has the same mean as `y` with less dispersion (a mean-preserving anti-spread of `y`). If `R w ≥ W y` scenario by scenario, then `R w` SSD-dominates `y` (Kuosmanen, 2004). The problem is a linear program with N + T² variables, T inequality constraints and 2T + 1 equality constraints, solved with SciPy HiGHS (`method="highs"`, which selects between dual simplex and interior point automatically).

Dominance is imposed **in-sample** only; out-of-sample persistence is not guaranteed.

## Design notes

- **Sparse construction.** The constraint matrix (about 62,500 columns at T = 250) is built with Kronecker products of identity and ones matrices and kept sparse throughout.
- **No look-ahead.** The estimation window ends the day before the rebalancing date, and that day's return is out-of-sample.
- **Independent check.** `ssd_min_slack` verifies SSD on the solution through the sorted cumulative-sum criterion, separately from the LP. On random test cases the optimum also matches a lower-partial-moment formulation of the same problem.
- **Failure handling.** Infeasible or failed solves fall back to 1/N and are counted, so a rolling run never stops silently on one date.

## Usage

```bash
pip install -r requirements.txt

# rolling monthly evaluation against the index
python ssd_optimizer.py --data path/to/Datisp100agg.xlsx --lookback 250 --out results

# weights for the most recent window only (one optimization)
python ssd_optimizer.py --data path/to/Datisp100agg.xlsx --latest
```

As a library:

```python
from ssd_optimizer import ssd_optimize, clean_weights, ssd_min_slack

w = ssd_optimize(asset_returns, reference_returns)   # (T, N) array, (T,) array -> weights or None
if w is not None:
    w = clean_weights(w)                              # zero residuals < 1e-5, renormalize
    print(ssd_min_slack(w, asset_returns, reference_returns))   # >= 0: SSD holds in-sample
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | required | Excel workbook (see below) |
| `--out` | `results` | Output directory |
| `--lookback` | `250` | Lookback window T (trading days) |
| `--reference` | `index` | Distribution to be dominated: `index` or `equal_weight` |
| `--start-date` | earliest possible | Rolling run: first rebalancing on/after this date |
| `--latest` | off | Solve once on the last T rows and write the weights (no backtest) |

### Rolling evaluation

At every month-end (last trading day of the month) the optimizer uses the `lookback` rows **before** that date. The universe is the set of assets with complete returns and index membership (composition matrix) over the whole window. The weights are applied, as constant daily-rebalanced weights, until the next month-end. The `--latest` run uses the last T rows **including** the final one, since no out-of-sample period follows.

### Input data

The data file is not included. The workbook must contain these sheets (no headers):

| Sheet | Content |
|-------|---------|
| `df_ret_final` | Daily asset returns (dates × assets) |
| `composition` | Index membership matrix, 1 = member (dates × assets) |
| `Date` | Dates, first column |
| `nomi` | Asset names, first row |
| `Index` | Daily benchmark returns |

### Output

Rolling run: `wealth.csv` (SSD portfolio and benchmark), `weights.csv` (date × asset), `metrics.csv` (annualized return and volatility, Sharpe, Calmar, Sortino, max drawdown and duration), `optimization_log.csv` and `wealth.png`.

`optimization_log.csv` has one row per rebalancing date: number of assets, whether the 1/N fallback was used, solve time, in-sample mean return of the portfolio and of the reference, `ssd_min_slack`, number of held assets and how many of them have missing out-of-sample returns.

`--latest`: `latest_weights.csv` with the non-zero weights.

## Notes

- If the solver fails (for example an infeasible problem when the reference cannot be dominated by the available assets), the weights fall back to 1/N. The log reports how many dates used the fallback.
- Weights below 1e-5 are set to zero and the rest renormalized, which can leave a tiny in-sample SSD violation. `ssd_min_slack` in the log shows its size.
- Missing out-of-sample asset returns (delistings, mergers) are set to 0. `held_with_missing_oos` counts how many held assets are affected.
- Missing benchmark returns are set to 0.
- Sortino uses the standard deviation of negative returns only.
- Tested with Python packages numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, matplotlib 3.10.8. The optimum matches an independent lower-partial-moment formulation of the same SSD problem on random test cases.

## Reference

Kuosmanen, T. (2004). Efficient diversification according to stochastic dominance criteria. *Management Science*, 50(10), 1390–1406.
