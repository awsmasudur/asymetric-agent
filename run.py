"""CLI: run the paired experiment and print metrics + pre-registered tests.

Examples:
  # offline validation:
  python run.py --provider mock --sessions 8 --rounds 30

  # scaled run on Bedrock us-east-1 (resolves inference profiles by substring):
  python run.py --provider bedrock --region us-east-1 \
      --observer claude-sonnet --decider claude-haiku --auditor claude-haiku \
      --sessions 15 --rounds 30
"""

from __future__ import annotations

import argparse
import os

from drift.config import ExperimentConfig
from drift.orchestrator import run_experiment
from drift.metrics import load_rows, compute_curves, run_analysis


def main():
    ap = argparse.ArgumentParser(description="Token-budget language drift experiment")
    ap.add_argument("--provider", default="mock", choices=["mock", "bedrock"])
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--observer", default="", help="bedrock inference-profile selector")
    ap.add_argument("--decider", default="")
    ap.add_argument("--auditor", default="")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--b-max", type=int, default=120)
    ap.add_argument("--b-min", type=int, default=8)
    ap.add_argument("--memory", default="full", help="decider/auditor: full | current | window:k")
    ap.add_argument("--observer-window", type=int, default=8,
                    help="how many prior rounds the Observer can see (0 = stateless)")
    ap.add_argument("--no-observer-feedback", action="store_true",
                    help="hide the Decider's guesses from the Observer")
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--analyze-only", default="", help="run_id to analyze without running")
    args = ap.parse_args()

    cfg = ExperimentConfig(
        provider=args.provider,
        region=args.region,
        observer_model=args.observer,
        decider_model=args.decider,
        auditor_model=args.auditor,
        n_sessions=args.sessions,
        n_rounds=args.rounds,
        b_max=args.b_max,
        b_min=args.b_min,
        memory=args.memory,
        observer_memory_window=args.observer_window,
        observer_feedback=not args.no_observer_feedback,
        run_dir=args.run_dir,
    )

    if args.analyze_only:
        run_id = args.analyze_only
    else:
        run_id = run_experiment(cfg)

    db_path = os.path.join(cfg.run_dir, "results.sqlite")
    rows = load_rows(db_path, run_id=run_id)

    curves = compute_curves(rows)
    print("\n=== Metric curves (by round, per condition) ===")
    print(f"{'cond':10s} {'rnd':>3s} {'len':>6s} {'novel':>6s} {'reuse':>6s} "
          f"{'Bacc':>5s} {'Cacc':>5s} {'gap':>5s}")
    for m in curves:
        print(f"{m.condition:10s} {m.round_index:3d} {m.mean_msg_len:6.1f} "
              f"{m.functional_novelty:6.2f} {m.convention_reuse:6.2f} "
              f"{m.decider_accuracy:5.2f} {m.auditor_accuracy:5.2f} {m.opacity_gap:5.2f}")

    from drift.metrics import budget_compliance, schematization, schema_rigidity
    comp = budget_compliance(rows)
    print("\n=== Budget compliance (truncation vs voluntary compression) ===")
    for cond, d in sorted(comp.items()):
        print(f"{cond:10s} n={d['n']:3d} truncation_rate={d['truncation_rate']:.2f} "
              f"voluntary={d['voluntary_n']:3d} over_budget_rate={d['over_budget_rate']:.2f}")

    sch = schematization(rows)
    print("\n=== Schematization (does it schematize+abbreviate, or encrypt?) ===")
    print("(schema_collapse: 0=template constant; higher=sheds fields under pressure)")
    print(f"{'cond':10s} {'collapse':>8s} {'fields':>6s} {'ws_drop':>7s} "
          f"{'abbrev':>6s} {'coined':>6s} {'full':>6s}")
    for cond, d in sorted(sch.items()):
        print(f"{cond:10s} {d['schema_collapse']:8.2f} {d['mean_fields']:6.1f} "
              f"{d['ws_density_drop']:+7.2f} {d['abbrev_rate']:6.2f} "
              f"{d['coined_rate']:6.2f} {d['full_rate']:6.2f}")

    # Actual token spend from logged usage (0 if provider didn't report usage).
    import sqlite3 as _sql
    _c = _sql.connect(db_path)
    _tot = _c.execute(
        "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(total_output_tokens),0) "
        "FROM rounds WHERE run_id=?", (run_id,)).fetchone()
    _c.close()
    in_tok, out_tok = _tot
    if in_tok or out_tok:
        # Haiku 4.5 approx pricing per 1M tokens; verify current us-east-1 rates.
        cost = in_tok / 1e6 * 1.00 + out_tok / 1e6 * 5.00
        print("\n=== Actual token spend (logged) ===")
        print(f"input={in_tok:,}  output={out_tok:,}  "
              f"est_cost_USD~{cost:.2f} (Haiku 4.5 @ $1/$5 per 1M; verify rates)")

    rig = schema_rigidity(rows)
    print("\n=== Schema rigidity (over-budget/truncation by budget bin, hi->lo) ===")
    for cond, bins in sorted(rig.items()):
        print(f"{cond}:")
        for b in bins:
            if b is None:
                continue
            print(f"  budget[{b['budget_lo']:3d}-{b['budget_hi']:3d}] n={b['n']:2d} "
                  f"over_budget={b['over_budget_rate']:.2f} trunc={b['truncation_rate']:.2f}")

    print("\n=== Pre-registered analysis (final third, treatment vs control) ===")
    results = run_analysis(rows)
    for r in results:
        p = "n/a" if r.p_value is None else f"{r.p_value:.4f}"
        verdict = "PASS" if r.passed else "----"
        print(f"[{verdict}] {r.hypothesis:28s} treat={r.treatment_mean:5.2f} "
              f"ctrl={r.control_mean:5.2f} diff={r.diff:+5.2f} "
              f"n={r.n_pairs} test={r.test} p={p}")


if __name__ == "__main__":
    main()
