# Statistical Arbitrage with Stochastic Dominance and Empirical Likelihood

Monthly-rebalanced backtest that builds portfolios under **second-order stochastic dominance (SSD)** constraints, using **Empirical Likelihood (EL)** scenario probabilities in place of the uniform 1/T weights.

## Computational Note: Execution Time

- **Algorithm:** Second-order Stochastic Dominance (SSD) with CVaR formulation.
- **Complexity:** O(T²) variables per optimization (250² = 62,500 variables).
- **Estimated Runtime:** ~4–7 hours using the SciPy HiGHS solver.

T is the lookback window (`--lookback`). The default is 175 (175² = 30,625 variables). In the original notebook run (T = 175, 91 assets), one rebalancing date (3 LPs) took about 110 s.

## Method

**1. Universe filter.** At each rebalancing date, only assets with complete returns and index membership over the entire lookback window are kept.

**2. Stochastic dominance pre-test.** Sorted benchmark returns are compared with each sorted asset (FSD, SSD, TSD). Among the dominated assets, the one with the lowest mean return is used as a second reference distribution.

**3. EL reweighting (relative entropy minimization).** Scenario probabilities are the minimum-relative-entropy weights such that the benchmark mean lies inside [L, U]. If the sample mean is already inside [L, U], the probabilities stay at 1/T (no tilt). Otherwise the mean is moved to the nearest bound, and the tilt is found through the dual problem. This reweights the empirical distribution toward a mean within its historical range (regime adjustment).

L and U are the 10th and 90th percentiles of the rolling benchmark means (window = T) that were available **at the rebalancing date**. The history expands at each rebalancing, and no out-of-sample data enters the calibration. The first rebalancing is the first month-end on or after `--start-date` (default 2007-01-01), so that the distribution used for [L, U] is already populated. Data before the start date is used only for this calibration.

**4. SSD portfolio LP (CVaR linearization).** Maximize expected return under the EL probabilities, subject to long-only, fully invested weights and

```
sum_t p_t · max(0, y_s − x_t·w)  ≤  sum_t p_t · max(0, y_s − y_t)      for every scenario s = 1..T
```

The constraint is imposed at every point of the empirical benchmark distribution. The `max(0, ·)` shortfall term is linearized with auxiliary variables `θ[s,t] ≥ y_s − x_t·w, θ ≥ 0`, which gives T² extra variables and T + T² constraints. This is the Rockafellar–Uryasev linearization of expected shortfall; imposing it at all thresholds is equivalent to CVaR dominance at all tail levels (Ogryczak & Ruszczyński, 2002). The LP is solved with SciPy HiGHS (`method="highs"`, which selects between dual simplex and interior point automatically), once per reference distribution: (1) the index, (2) the dominated asset, (3) the equally weighted portfolio.

**5. Strategies.** Out-of-sample daily factors are `1 + (portfolio return − reference return)`:

| # | Strategy |
|---|----------|
| 1 | vs Index |
| 2 | vs Dominated asset |
| 3 | vs 1/N |
| 4 | Mean reversion on Strat 1 (`1 / factor`) |
| 5 | Mean reversion on Strat 2 (`1 / factor`) |

`--mr-mode short` uses `1 − spread` instead of `1 / factor` for strategies 4 and 5. The two are not equivalent: `1/(1+s) − (1−s) = s²/(1+s)`, which is positive, so the `inverse` mode overstates the return of an actual short position. In `inverse` mode, wealth of strategies 4 and 5 is exactly the reciprocal of the wealth of strategies 1 and 2.

## EL tilt counter

At every rebalancing date the script records whether the EL tilt was active. The log ends with a summary line, for example:

```
EL tilt active in 14 of 47 rebalancing dates (29.8%): 8 below L, 6 above U; 0 tilts did not converge.
```

The per-date detail is saved in `el_diagnostics.csv` (columns: `L`, `U`, `n_rolling_obs`, `sample_mean`, `tilted`, `side`, `lam`, `converged`).

The counter is needed to interpret any difference between EL and equal-probability results, because on dates without a tilt the EL probabilities equal 1/T. Under stationary returns and 10/90 percentiles, a tilt on roughly 20% of dates is what the calibration alone would produce.

## Usage

```bash
pip install -r requirements.txt
python ssd_el_backtest.py --data path/to/Datisp100agg.xlsx --lookback 175 --start-date 2007-01-01 --out results
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | required | Excel workbook (see below) |
| `--out` | `results` | Output directory |
| `--lookback` | `175` | Lookback window T (trading days) |
| `--start-date` | `2007-01-01` | First rebalancing on/after this date; earlier data only calibrates [L, U] |
| `--min-el-obs` | `250` | Minimum rolling benchmark means required to calibrate [L, U] at the first rebalancing |
| `--el-percentiles` | `10 90` | Percentiles defining [L, U] |
| `--mr-mode` | `inverse` | `inverse` or `short` |
| `--exclude-target` | `Lehman` | Dominated-asset names containing these strings use the index as the out-of-sample reference |

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

`wealth.csv`, `metrics.csv` (annualized return and volatility, Sharpe, Calmar, Sortino, max drawdown and duration), `el_diagnostics.csv`, `history.pkl` (weights, dominated pool and target per date), and two PNG wealth plots.

## Notes

- Results start at `--start-date` and are not comparable with a backtest starting earlier.
- Sortino uses the standard deviation of negative returns only.
- Missing out-of-sample asset returns (delistings, mergers) are set to 0. Solver failures fall back to equal weights and are counted in the log.
- Tested with Python packages numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, matplotlib 3.10.8.

## References

- Kuosmanen, T. (2004). Efficient diversification according to stochastic dominance criteria. *Management Science*, 50(10), 1390–1406.
- Ogryczak, W. & Ruszczyński, A. (2002). Dual stochastic dominance and related mean-risk models. *SIAM Journal on Optimization*, 13(1), 60–78.
- Post, T., Karabatı, S. & Arvanitis, S. (2018). Portfolio optimization based on stochastic dominance and empirical likelihood. *Journal of Econometrics*, 206(1), 167–186.
- Rockafellar, R. T. & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2, 21–42.
