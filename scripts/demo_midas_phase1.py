"""Demo the Phase 1 Midas improvements (importance threshold + stability).

Runs two mini-demos with realistic-looking synthetic data to show
what the user would see in real usage:

  1. Feature importance threshold filtering: train a 3-class model
     on 20 features where only 3 carry real signal, then retrain
     with importance_threshold=0.05 and report how many features
     were dropped.

  2. Param stability report: build a set of fake OptimizationResults
     representing walk-forward windows (some params stable across
     windows, some unstable), compute the stability report and
     print the diagnostic output.

Usage:
    uv run python -m scripts.demo_midas_phase1
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.midas.optimizer import OptimizationResult
from src.midas.trainer import MidasTrainer, TrainerConfig
from src.midas.walk_forward import (
    WalkForwardOptunaConfig,
    _print_param_stability,
    compute_param_stability,
)


def demo_importance_threshold() -> None:
    """Demo #1: feature importance filtering."""
    print("=" * 70)
    print("DEMO 1 — Feature importance threshold")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n, n_features = 2000, 20

    data: dict[str, list[float]] = {}
    for i in range(n_features):
        data[f"feat_{i:02d}"] = rng.normal(0, 1, n).tolist()
    df = pl.DataFrame(data)

    # Only feat_00, feat_07, and feat_13 carry real signal
    signal = (
        0.8 * df["feat_00"].to_numpy()
        + 0.5 * df["feat_07"].to_numpy()
        + 0.3 * df["feat_13"].to_numpy()
        + rng.normal(0, 0.3, n)
    )
    target = np.where(signal > 0.2, 1, np.where(signal < -0.2, 2, 0)).astype(np.int32)

    print(f"\nDataset: {n} rows, {n_features} features, 3-class target")
    print("Truth: only 3 features carry real signal (feat_00, feat_07, feat_13)")

    # Pass 1 — no filter
    trainer_raw = MidasTrainer(TrainerConfig(
        n_estimators=100, importance_threshold=0.0,
    ))
    raw_result = trainer_raw.train(df, target)
    raw_total = sum(raw_result.feature_importance.values())
    sorted_imp = sorted(
        raw_result.feature_importance.items(), key=lambda x: x[1], reverse=True,
    )
    print("\n--- Without threshold (all features kept) ---")
    print(f"  kept: {len(raw_result.feature_names)} features")
    print(f"  val_log_loss: {raw_result.val_log_loss:.4f}")
    print("  top 5 by importance (% of total):")
    for name, imp in sorted_imp[:5]:
        pct = imp / raw_total * 100
        print(f"    {name:<12} {pct:>6.2f}%")

    # Pass 2 — with threshold 5%
    trainer_filtered = MidasTrainer(TrainerConfig(
        n_estimators=100, importance_threshold=0.05,
    ))
    filtered_result = trainer_filtered.train(df, target)
    print("\n--- With importance_threshold=0.05 (drop if < 5% of total gain) ---")
    print(f"  kept: {len(filtered_result.feature_names)} features")
    print(f"  val_log_loss: {filtered_result.val_log_loss:.4f}")
    print(f"  features kept: {filtered_result.feature_names}")

    dropped = len(raw_result.feature_names) - len(filtered_result.feature_names)
    print(f"\n  → dropped {dropped} features that contributed < 5% of total gain")


def demo_param_stability() -> None:
    """Demo #2: parameter stability across windows."""
    print("\n" + "=" * 70)
    print("DEMO 2 — Parameter stability across walk-forward windows")
    print("=" * 70)

    # Simulate 8 walk-forward windows where:
    # - k_sl converges around 0.5 (stable, CV ~5%)
    # - k_tp drifts between 1.0 and 4.0 (unstable, CV ~45%)
    # - atr_period stays at 14 (perfectly stable)
    # - rr_ratio bounces around (unstable)
    rng = np.random.default_rng(0)
    results: list[OptimizationResult] = []
    for _ in range(8):
        results.append(OptimizationResult(
            best_inner_params={
                "k_sl": float(0.5 + rng.normal(0, 0.025)),
                "k_tp": float(rng.uniform(1.0, 4.0)),
                "rr_ratio": float(rng.uniform(1.5, 4.5)),
            },
            best_outer_params={
                "atr_period": 14,
                "scalp__roc_fast": int(rng.integers(3, 15)),
            },
        ))

    print(f"\nSimulated {len(results)} walk-forward windows")
    print("Truth:")
    print("  - k_sl: stable (values around 0.5)")
    print("  - k_tp: unstable (uniform 1.0-4.0)")
    print("  - rr_ratio: unstable (uniform 1.5-4.5)")
    print("  - atr_period: perfectly stable (always 14)")
    print("  - scalp__roc_fast: unstable (random 3-15)")

    # Programmatic report
    report = compute_param_stability(results, cv_threshold=15.0)
    print("\n--- Programmatic report ---")
    print(f"  total params analysed: {report.total}")
    print(f"  converged (CV < 15%): {report.converged}")
    print(f"  converged ratio: {report.converged_ratio * 100:.0f}%")

    # Pretty print + warning check
    print("\n--- Pretty print output (what you'd see in a real run) ---")
    config = WalkForwardOptunaConfig(
        stability_cv_threshold=15.0,
        stability_warn_ratio=0.30,
    )
    _print_param_stability(
        results,
        cv_threshold=config.stability_cv_threshold,
        warn_ratio=config.stability_warn_ratio,
    )


def main() -> None:
    demo_importance_threshold()
    demo_param_stability()
    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
