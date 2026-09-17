"""Local logging: SQLite (queryable) + JSONL (raw transcripts).

One row per round-per-arm captures everything the metrics and analysis need:
budget, the Observer message, stop_reason/voluntary flag, token count, the true
label, and the Decider/Auditor labels.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class RoundLog:
    run_id: str
    session: int
    condition: str      # "treatment" | "control"
    seed: int
    round_index: int
    budget: int
    message: str
    stop_reason: str
    voluntary: int      # 1 if stop_reason == end_turn
    output_tokens: int
    true_label: str
    decider_label: str
    auditor_label: str
    decider_correct: int
    auditor_correct: int
    input_tokens: int = 0    # summed across observer+decider+auditor for the round
    total_output_tokens: int = 0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
    run_id TEXT, session INTEGER, condition TEXT, seed INTEGER,
    round_index INTEGER, budget INTEGER, message TEXT, stop_reason TEXT,
    voluntary INTEGER, output_tokens INTEGER, true_label TEXT,
    decider_label TEXT, auditor_label TEXT,
    decider_correct INTEGER, auditor_correct INTEGER,
    input_tokens INTEGER DEFAULT 0, total_output_tokens INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created REAL, config_json TEXT
);
"""


class Store:
    def __init__(self, run_dir: str, run_id: str):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.run_id = run_id
        self.db_path = os.path.join(run_dir, "results.sqlite")
        self.jsonl_path = os.path.join(run_dir, f"{run_id}.jsonl")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self._jsonl = open(self.jsonl_path, "a", encoding="utf-8")

    def record_run(self, config: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, created, config_json) VALUES (?,?,?)",
            (self.run_id, time.time(), json.dumps(config)),
        )
        self.conn.commit()

    def log_round(self, row: RoundLog):
        d = asdict(row)
        self.conn.execute(
            "INSERT INTO rounds VALUES ("
            ":run_id,:session,:condition,:seed,:round_index,:budget,:message,"
            ":stop_reason,:voluntary,:output_tokens,:true_label,:decider_label,"
            ":auditor_label,:decider_correct,:auditor_correct,"
            ":input_tokens,:total_output_tokens)",
            d,
        )
        self._jsonl.write(json.dumps(d, ensure_ascii=False) + "\n")

    def flush(self):
        self.conn.commit()
        self._jsonl.flush()

    def close(self):
        self.flush()
        self._jsonl.close()
        self.conn.close()
