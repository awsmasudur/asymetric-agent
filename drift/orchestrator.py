"""Session orchestrator: paired treatment/control runs on identical data.

For each session:
  1. Generate one seeded round stream (shared by both arms).
  2. Run the TREATMENT arm (shrinking budget) and the CONTROL arm (fixed budget)
     over that same stream.
  3. In each arm: Observer speaks under the round's budget; Decider labels from
     the message stream (memory mode); Auditor labels offline from the same
     stream. Everything is logged per round.

The Auditor uses a separate provider instance and only ever sees Observer
messages, so it measures outsider opacity without perturbing A/B.
"""

from __future__ import annotations

import time
from typing import List, Tuple

from .config import ExperimentConfig, shrinking_schedule, fixed_schedule
from .providers import make_provider
from .agents import Observer, Decider, Auditor, PriorTurn
from .task import generate_rounds, Round
from .logging_store import Store, RoundLog


def _build_agents(cfg: ExperimentConfig) -> Tuple[Observer, Decider, Auditor]:
    obs_p = make_provider(cfg.provider, "observer", cfg.observer_model, cfg.region)
    dec_p = make_provider(cfg.provider, "decider", cfg.decider_model, cfg.region)
    aud_p = make_provider(cfg.provider, "auditor", cfg.auditor_model, cfg.region)
    return (
        Observer(obs_p, headroom=cfg.headroom,
                 memory_window=cfg.observer_memory_window,
                 feedback=cfg.observer_feedback),
        Decider(dec_p, memory=cfg.memory),
        Auditor(aud_p, memory=cfg.memory),
    )


def _run_arm(cfg: ExperimentConfig, rounds: List[Round], schedule: List[int],
             condition: str, session: int, seed: int, store: Store, run_id: str):
    observer, decider, auditor = _build_agents(cfg)
    history: List[str] = []            # A's messages, for the decoders
    obs_history: List[PriorTurn] = []  # A's cross-round memory (with B feedback)
    for rnd in rounds:
        budget = schedule[rnd.index]
        # Observer sees only PRIOR rounds (B has not yet decided this round).
        turn = observer.observe(rnd.index, rnd.record, budget,
                                ground_hint=rnd.label, history=obs_history)
        history.append(turn.message)

        b_label, b_in, b_out = decider.decide_usage(history)
        c_label, c_in, c_out = auditor.decide_usage(history)

        round_input = turn.input_tokens + b_in + c_in
        round_output = turn.output_tokens + b_out + c_out

        # Now that B has decided, record this round in A's memory as feedback.
        obs_history.append(PriorTurn(
            round_index=rnd.index, budget=budget,
            message=turn.message, decider_label=b_label,
        ))

        store.log_round(RoundLog(
            run_id=run_id,
            session=session,
            condition=condition,
            seed=seed,
            round_index=rnd.index,
            budget=budget,
            message=turn.message,
            stop_reason=turn.stop_reason,
            voluntary=int(turn.voluntary),
            output_tokens=turn.output_tokens,
            true_label=rnd.label,
            decider_label=b_label,
            auditor_label=c_label,
            decider_correct=int(b_label == rnd.label),
            auditor_correct=int(c_label == rnd.label),
            input_tokens=round_input,
            total_output_tokens=round_output,
        ))
    store.flush()


def run_experiment(cfg: ExperimentConfig) -> str:
    run_id = f"{cfg.provider}-{int(time.time())}"
    store = Store(cfg.run_dir, run_id)
    store.record_run(cfg.to_dict())

    treat_sched = shrinking_schedule(cfg.b_max, cfg.b_min, cfg.n_rounds)
    ctrl_sched = fixed_schedule(cfg.b_max, cfg.n_rounds)
    # Third arm: budget fixed at the LOW floor for all rounds. Fixed-low vs
    # treatment isolates whether the shrinking *trajectory* does anything beyond
    # raw scarcity (both reach the same floor; only treatment gets there
    # gradually). Fixed-low vs control isolates scarcity itself.
    fixedlow_sched = fixed_schedule(cfg.b_min, cfg.n_rounds)

    for s in range(cfg.n_sessions):
        seed = cfg.base_seed + s
        rounds = generate_rounds(seed=seed, n_rounds=cfg.n_rounds)
        # Paired: identical data stream, three budget schedules.
        _run_arm(cfg, rounds, treat_sched, "treatment", s, seed, store, run_id)
        _run_arm(cfg, rounds, ctrl_sched, "control", s, seed, store, run_id)
        _run_arm(cfg, rounds, fixedlow_sched, "fixed_low", s, seed, store, run_id)
        print(f"[session {s}] seed={seed} done (treatment + control + fixed_low)")

    store.close()
    print(f"run_id={run_id}  db={store.db_path}  jsonl={store.jsonl_path}")
    return run_id
