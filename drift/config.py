"""Experiment configuration and the budget schedule."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List


@dataclass
class ExperimentConfig:
    # provider
    provider: str = "mock"          # "mock" | "bedrock"
    region: str = "us-east-1"
    observer_model: str = ""        # selector for bedrock inference profile
    decider_model: str = ""
    auditor_model: str = ""

    # session shape
    n_rounds: int = 30
    b_max: int = 120                # starting / control budget (soft, tokens)
    b_min: int = 8                  # floor for the shrinking schedule
    headroom: int = 32              # max_tokens = budget*tokens_per_word + headroom
    memory: str = "full"            # decider/auditor: "full" | "current" | "window:k"
    observer_memory_window: int = 8  # how many prior rounds the Observer sees
    observer_feedback: bool = True   # show the Observer the Decider's guesses

    # experiment scale
    n_sessions: int = 5             # smoke default; scaled run uses >=15
    base_seed: int = 1000

    # bookkeeping
    run_dir: str = "runs"

    def to_dict(self) -> dict:
        return asdict(self)


def shrinking_schedule(b_max: int, b_min: int, n_rounds: int) -> List[int]:
    """Geometric decay from b_max at round 0 to ~b_min at the final round."""
    if n_rounds <= 1:
        return [b_max]
    ratio = (b_min / b_max) ** (1.0 / (n_rounds - 1))
    sched = []
    val = float(b_max)
    for _ in range(n_rounds):
        sched.append(max(b_min, round(val)))
        val *= ratio
    sched[-1] = max(b_min, sched[-1])
    return sched


def fixed_schedule(b_max: int, n_rounds: int) -> List[int]:
    return [b_max] * n_rounds


if __name__ == "__main__":
    print("shrinking:", shrinking_schedule(120, 8, 30))
    print("fixed:", fixed_schedule(120, 30)[:5], "...")
