"""
metrics.py
──────────
Evaluation metrics (spec §7):

  1. Deflated Sharpe Ratio (DSR)
     Corrects the naive Sharpe for:
       - Number of independent trials (hyperparameter search)
       - Non-normality of returns (skewness, kurtosis)
       - Sample length

  2. Probability of Backtest Overfitting (PBO)
     Via Combinatorially Symmetric Cross-Validation (CSCV).
     Reference: Bailey & Lopez de Prado (2014).

  3. Regime-conditioned performance statistics.

Public API:
    deflated_sharpe_ratio(returns, n_trials, ...)
    probability_of_backtest_overfitting(returns_matrix, ...)
    regime_performance_breakdown(trade_log, ...)
"""

from __future__ import annotations

import math
import itertools
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# 1. Deflated Sharpe Ratio
# ─────────────────────────────────────────────────────────────────────────────

def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    sharpe_benchmark: float = 0.0,
    annualization: int = 252,
) -> dict:
    """
    Compute the Deflated Sharpe Ratio (DSR).

    DSR tests whether the observed Sharpe Ratio (SR) is statistically
    significant given the number of trials run during development, and
    corrects for non-Gaussian return distributions.

    Formula (Bailey & Lopez de Prado 2013):
        E[max SR] ≈ (1 - γ) * Z^{-1}(1 - 1/n_trials)
                   + γ * Z^{-1}(1 - 1/(n_trials * e))
        where γ ≈ 0.5772 (Euler-Mascheroni constant)

        SR_hat (bias-corrected) = SR * (1 - skew*SR/6 + (kurtosis-3)*SR^2/24)^{-1/2}

        DSR = PSR - SR_benchmark / SR_std_dev
        PSR (Probability of SR) = Φ(SR_hat / sqrt((1-skew*SR+excess_kurt*SR^2/4) / (T-1)))

    Returns dict with:
        sr            : naive annualized Sharpe ratio
        psr           : probability of SR exceeding benchmark
        dsr           : deflated Sharpe ratio (SR after multiple-testing correction)
        e_max_sr      : expected maximum SR under multiple trials
        is_significant: bool (DSR > 0 → significant at ~95% level)
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    returns = returns[np.isfinite(returns)]
    T = len(returns)

    if T < 10:
        return {"sr": 0.0, "psr": 0.0, "dsr": 0.0, "e_max_sr": np.nan, "is_significant": False}

    mu    = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma < 1e-12:
        return {"sr": 0.0, "psr": 0.5, "dsr": 0.0, "e_max_sr": np.nan, "is_significant": False}

    # Annualized naive Sharpe
    sr_ann  = mu / sigma * np.sqrt(annualization)

    # Per-period SR (for PSR formula which works on per-period stats)
    sr_pp   = mu / sigma

    # Skewness and excess kurtosis
    skew    = float(stats.skew(returns))
    kurt    = float(stats.kurtosis(returns))  # excess kurtosis

    # Variance of SR estimator (Lo 2002 / ILopez 2013)
    sr_var  = (1.0 - skew * sr_pp + (kurt + 1.0) * sr_pp ** 2 / 4.0) / (T - 1)
    sr_std  = max(np.sqrt(sr_var), 1e-10)

    # PSR: P(SR > SR_benchmark) under the estimated distribution
    psr = float(stats.norm.cdf((sr_pp - sharpe_benchmark) / sr_std))

    # Expected maximum SR across n_trials (Bailey & Lopez de Prado 2013)
    gamma_em = 0.5772156649   # Euler-Mascheroni constant
    e_max_sr = (
        (1.0 - gamma_em) * stats.norm.ppf(1.0 - 1.0 / max(n_trials, 1))
        + gamma_em * stats.norm.ppf(1.0 - 1.0 / (max(n_trials, 1) * math.e))
    )
    # Convert expected max SR from standardized units to per-period SR
    # (times the SR std so we can compare to our estimated SR)
    e_max_sr_pp = e_max_sr * sr_std + sharpe_benchmark

    # DSR: SR adjusted for the expected maximum SR under multiple testing
    dsr = float(stats.norm.cdf((sr_pp - e_max_sr_pp) / sr_std))

    return {
        "sr":             sr_ann,
        "sr_per_period":  sr_pp,
        "psr":            psr,
        "dsr":            dsr,
        "e_max_sr":       e_max_sr_pp * np.sqrt(annualization),
        "skewness":       skew,
        "excess_kurtosis": kurt,
        "n_obs":          T,
        "n_trials":       n_trials,
        "is_significant": dsr > 0.95,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Probability of Backtest Overfitting (PBO)
# ─────────────────────────────────────────────────────────────────────────────

def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_subsets: int = 4,
    annualization: int = 252,
) -> dict:
    """
    Compute PBO via Combinatorially Symmetric Cross-Validation (CSCV).

    Reference: Bailey, Borwein, Lopez de Prado, Zhu (2014)
    "Pseudo-Mathematics and Financial Charlatanism: The Effects of
    Backtest Overfitting on Out-of-Sample Performance."

    Algorithm:
      1. Organize T observations into S equal sub-periods.
      2. For all C(S, S/2) splits into IS (in-sample) and OOS halves:
         a. Select the best strategy on IS (highest SR).
         b. Record that strategy's OOS SR rank (relative to other strategies).
      3. PBO = fraction of CSCV runs where the best IS strategy underperforms
               the median OOS SR.

    Parameters
    ----------
    returns_matrix : (T, N) array — T observations × N strategies/configurations.
                     Each column is the return stream of one strategy variant.
    n_subsets      : Number of sub-periods S (even integer). More = more combos.
    annualization  : Bars per year for SR computation.

    Returns dict:
        pbo         : fraction of runs where best IS choice is below median OOS
        logit_pbo   : logit-transformed PBO for boundary-safe display
        median_oos_sr: median OOS SR of the best IS strategy
    """
    returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
    T, N = returns_matrix.shape

    if T < n_subsets * 4:
        return {"pbo": np.nan, "logit_pbo": np.nan, "median_oos_sr": np.nan,
                "n_combinations": 0}

    s        = n_subsets
    half_s   = s // 2

    # Divide rows into S equal blocks
    block_size  = T // s
    blocks      = [returns_matrix[i * block_size:(i + 1) * block_size] for i in range(s)]

    block_idx   = list(range(s))
    combos      = list(itertools.combinations(block_idx, half_s))
    n_combos    = len(combos)

    oos_sr_best_is = []

    def _sharpe(rets: np.ndarray) -> float:
        mu, sd = rets.mean(), rets.std(ddof=1)
        if sd < 1e-12:
            return 0.0
        return mu / sd * np.sqrt(annualization)

    for is_blocks in combos:
        oos_blocks = tuple(b for b in block_idx if b not in is_blocks)

        is_ret  = np.concatenate([blocks[i] for i in is_blocks],  axis=0)
        oos_ret = np.concatenate([blocks[i] for i in oos_blocks], axis=0)

        is_sr  = np.array([_sharpe(is_ret[:, n])  for n in range(N)])
        oos_sr = np.array([_sharpe(oos_ret[:, n]) for n in range(N)])

        best_is_idx  = int(np.argmax(is_sr))
        best_oos_sr  = oos_sr[best_is_idx]
        median_oos   = np.median(oos_sr)

        oos_sr_best_is.append(best_oos_sr - median_oos)  # negative → overfit

    oos_sr_best_is = np.array(oos_sr_best_is)
    pbo   = float((oos_sr_best_is < 0).mean())

    # Logit-transform for stability near boundaries
    pbo_clipped = np.clip(pbo, 1e-6, 1 - 1e-6)
    logit_pbo   = float(np.log(pbo_clipped / (1 - pbo_clipped)))

    median_oos_sr = float(np.median(oos_sr_best_is))

    return {
        "pbo":           pbo,
        "logit_pbo":     logit_pbo,
        "median_oos_delta_sr": median_oos_sr,
        "n_combinations": n_combos,
        "is_overfit":    pbo > 0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Regime-conditioned performance
# ─────────────────────────────────────────────────────────────────────────────

def regime_performance_breakdown(
    trade_log: pd.DataFrame,
    regime_col: str = "regime",
    pnl_col: str = "net_pnl",
    notional_col: str = "notional",
    annualization: int = 252,
) -> pd.DataFrame:
    """
    Compute per-regime performance statistics.

    Parameters
    ----------
    trade_log  : DataFrame with columns: regime, net_pnl, notional, t0.
    Returns DataFrame with per-regime:
        n_trades, win_rate, mean_ret, std_ret, sharpe, ann_return
    """
    if regime_col not in trade_log.columns:
        trade_log = trade_log.copy()
        trade_log[regime_col] = "unknown"

    rows = []
    for regime, grp in trade_log.groupby(regime_col):
        ret    = grp[pnl_col] / grp[notional_col].clip(lower=1.0)
        n      = len(grp)
        wr     = float((grp[pnl_col] > 0).mean())
        mr     = float(ret.mean())
        sr_std = float(ret.std(ddof=1))
        sr     = mr / max(sr_std, 1e-10) * np.sqrt(annualization)
        ann_r  = mr * annualization

        rows.append({
            "regime":     regime,
            "n_trades":   n,
            "win_rate":   wr,
            "mean_ret":   mr,
            "std_ret":    sr_std,
            "sharpe":     sr,
            "ann_return": ann_r,
        })

    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full evaluation report
# ─────────────────────────────────────────────────────────────────────────────

def generate_evaluation_report(
    backtest_result,
    n_trials: int,
    returns_matrix: Optional[np.ndarray] = None,
    label_df: Optional[pd.DataFrame] = None,
    annualization: int = 252,
) -> dict:
    """
    Generate a comprehensive evaluation report dictionary.

    Parameters
    ----------
    backtest_result : BacktestResult from backtester.py
    n_trials        : Number of hyperparameter trials run (for DSR)
    returns_matrix  : (T, N) strategy return streams for PBO (optional)
    label_df        : For regime-split enrichment (optional)

    Returns a report dict with all metrics.
    """
    report = {}

    # Basic stats
    daily_rets = backtest_result.daily_returns.dropna().values
    report["basic"] = backtest_result.summary()

    # DSR
    report["dsr_stats"] = deflated_sharpe_ratio(
        daily_rets, n_trials=n_trials, annualization=annualization
    )

    # PBO (if provided)
    if returns_matrix is not None and returns_matrix.shape[1] >= 2:
        report["pbo_stats"] = probability_of_backtest_overfitting(
            returns_matrix, annualization=annualization
        )

    # Regime breakdown
    if len(backtest_result.trade_log) > 0:
        tl = backtest_result.trade_log.copy()
        if label_df is not None and "regime" in label_df.columns:
            regime_map = label_df.set_index(["ticker", "t0"])["regime"].to_dict()
            tl["regime"] = tl.apply(
                lambda r: regime_map.get((r["ticker"], r["t0"]), -1), axis=1
            )
        report["regime_breakdown"] = regime_performance_breakdown(tl).to_dict(orient="records")

    return report


def print_report(report: dict) -> None:
    """Pretty-print the evaluation report."""
    print("\n" + "="*60)
    print("BACKTEST EVALUATION REPORT")
    print("="*60)

    if "basic" in report:
        print("\n── Basic Performance ──")
        for k, v in report["basic"].items():
            print(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    if "dsr_stats" in report:
        d = report["dsr_stats"]
        print(f"\n── Deflated Sharpe Ratio ──")
        print(f"  Naive SR (annualized)  : {d.get('sr', 0):.4f}")
        print(f"  PSR (prob SR > 0)      : {d.get('psr', 0):.4f}")
        print(f"  Deflated SR (DSR)      : {d.get('dsr', 0):.4f}")
        print(f"  Expected Max SR ({d.get('n_trials',0)} trials): {d.get('e_max_sr', 0):.4f}")
        print(f"  Skewness               : {d.get('skewness', 0):.4f}")
        print(f"  Excess Kurtosis        : {d.get('excess_kurtosis', 0):.4f}")
        sig = "✓ SIGNIFICANT" if d.get("is_significant") else "✗ NOT SIGNIFICANT"
        print(f"  {sig}")

    if "pbo_stats" in report:
        p = report["pbo_stats"]
        print(f"\n── Probability of Backtest Overfitting (PBO) ──")
        print(f"  PBO                    : {p.get('pbo', 0):.4f}")
        print(f"  Logit PBO              : {p.get('logit_pbo', 0):.4f}")
        print(f"  Median OOS delta SR    : {p.get('median_oos_delta_sr', 0):.4f}")
        print(f"  N combinations         : {p.get('n_combinations', 0)}")
        print(f"  {'⚠ OVERFIT' if p.get('is_overfit') else '✓ NOT OVERFIT'}")

    if "regime_breakdown" in report:
        rb = pd.DataFrame(report["regime_breakdown"])
        print(f"\n── Regime-Conditioned Performance ──")
        print(rb.to_string(index=False))

    print("\n" + "="*60)


if __name__ == "__main__":
    # Quick test of all metrics
    rng = np.random.default_rng(42)

    # Simulate daily returns (slightly positive, with fat tails)
    T = 500
    daily_rets = rng.standard_t(df=4, size=T) * 0.008 + 0.0003

    print("── Testing Deflated Sharpe Ratio ──")
    dsr = deflated_sharpe_ratio(daily_rets, n_trials=50)
    for k, v in dsr.items():
        print(f"  {k}: {v}")

    print("\n── Testing PBO ──")
    N = 6   # 6 strategy variants
    ret_mat = rng.standard_normal((T, N)) * 0.01
    ret_mat[:, 0] += 0.0003   # one slightly better strategy
    pbo = probability_of_backtest_overfitting(ret_mat, n_subsets=4)
    print(f"  PBO: {pbo['pbo']:.4f}, N combos: {pbo['n_combinations']}")

    print("\n✓ Metrics module tests passed.")
