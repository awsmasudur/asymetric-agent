"""Synthetic record-classification task with hidden ground truth.

The Observer (A) sees a Record. The label is a deterministic function of the
record (see `classify`), hidden from every agent. Because the rule is fixed but
the surface features vary each round, A is pushed to communicate *which features
mattered* rather than restating the row -- exactly the pressure that should
produce compressed shorthand under a tight budget.

Design notes:
- Records carry both "functional" fields (drive the label) and "entity" fields
  (IDs, names -- pure noise for novelty-confound control). The entity fields let
  us verify that the metrics correctly strip topic-driven novelty.
- Everything is seeded so treatment and control arms see identical data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import List

# The three classes the Decider must choose between.
CLASSES = ["APPROVE", "REVIEW", "DECLINE"]

REGIONS = ["NA", "EU", "APAC", "LATAM", "MEA"]
_FIRST = ["Ada", "Alan", "Grace", "Linus", "Katherine", "Dennis", "Barbara", "Ken"]
_LAST = ["Lovelace", "Turing", "Hopper", "Torvalds", "Johnson", "Ritchie", "Liskov", "Thompson"]


@dataclass
class Record:
    """A single transaction-like record shown only to the Observer."""

    # --- entity / noise fields (must NOT drive novelty metrics) ---
    txn_id: str
    account_holder: str
    # --- functional fields (drive the hidden label) ---
    amount: int
    region: str
    account_age_days: int
    flag_velocity: bool  # unusually rapid recent activity
    flag_mismatch: bool  # billing/shipping or geo mismatch
    prior_declines: int

    def to_prompt(self) -> str:
        """Human-readable form the Observer sees."""
        return (
            f"txn_id={self.txn_id}\n"
            f"account_holder={self.account_holder}\n"
            f"amount={self.amount}\n"
            f"region={self.region}\n"
            f"account_age_days={self.account_age_days}\n"
            f"flag_velocity={self.flag_velocity}\n"
            f"flag_mismatch={self.flag_mismatch}\n"
            f"prior_declines={self.prior_declines}"
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Round:
    index: int
    record: Record
    label: str


def classify(r: Record) -> str:
    """Deterministic ground-truth rule. Hidden from all agents.

    A small risk score with hard override rules. Tuned so all three classes
    occur with reasonable frequency across random records.
    """
    # Hard decline: strong fraud signal.
    if r.prior_declines >= 4:
        return "DECLINE"
    if r.flag_velocity and r.flag_mismatch and r.prior_declines >= 1:
        return "DECLINE"

    score = 0
    if r.amount >= 9000:
        score += 2
    elif r.amount >= 3000:
        score += 1

    if r.account_age_days < 30:
        score += 2
    elif r.account_age_days < 180:
        score += 1

    if r.flag_velocity:
        score += 1
    if r.flag_mismatch:
        score += 1
    score += r.prior_declines  # 0..3 here (>=4 handled above)

    if r.region in ("MEA", "LATAM"):
        score += 1

    if score >= 6:
        return "DECLINE"
    if score >= 3:
        return "REVIEW"
    return "APPROVE"


def _make_record(rng: random.Random, i: int) -> Record:
    return Record(
        txn_id=f"TX{rng.randint(10**6, 10**7 - 1)}",
        account_holder=f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
        amount=rng.choice([50, 120, 300, 800, 1500, 3200, 5000, 9000, 15000]),
        region=rng.choice(REGIONS),
        account_age_days=rng.choice([3, 15, 45, 120, 300, 700, 1500]),
        flag_velocity=rng.random() < 0.35,
        flag_mismatch=rng.random() < 0.30,
        prior_declines=rng.choice([0, 0, 0, 1, 1, 2, 3, 4]),
    )


def generate_rounds(seed: int, n_rounds: int) -> List[Round]:
    """Deterministic stream of rounds for a session. Same seed -> same stream,
    so treatment and control arms are compared on identical data."""
    rng = random.Random(seed)
    rounds: List[Round] = []
    for i in range(n_rounds):
        rec = _make_record(rng, i)
        rounds.append(Round(index=i, record=rec, label=classify(rec)))
    return rounds


def label_distribution(rounds: List[Round]) -> dict:
    dist = {c: 0 for c in CLASSES}
    for rnd in rounds:
        dist[rnd.label] += 1
    return dist


# Tokens that come from record *values* (entities/numbers) -- stripped before
# computing functional-vocabulary novelty so that new IDs/names/amounts don't
# masquerade as invented language.
def entity_tokens(rounds: List[Round]) -> set:
    ent: set = set()
    for rnd in rounds:
        r = rnd.record
        ent.update(str(r.txn_id).lower().split())
        ent.update(str(r.account_holder).lower().split())
        ent.add(str(r.amount))
        ent.add(str(r.account_age_days))
        ent.add(str(r.prior_declines))
        ent.add(r.region.lower())
    return ent


if __name__ == "__main__":
    rs = generate_rounds(seed=7, n_rounds=30)
    print("label distribution:", label_distribution(rs))
    for rnd in rs[:3]:
        print(f"--- round {rnd.index} label={rnd.label} ---")
        print(rnd.record.to_prompt())
