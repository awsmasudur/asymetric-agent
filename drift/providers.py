"""Model-agnostic LLM provider abstraction.

Two providers:
- MockProvider   : deterministic, offline, free. Simulates an Observer that
                   compresses harder as its budget shrinks (so we can validate
                   the harness + metrics + stats end-to-end with zero API cost).
- BedrockProvider: AWS Bedrock (region us-east-1 by default). Resolves model IDs
                   at runtime via inference profiles rather than bare on-demand
                   model IDs, and reports stop_reason so the orchestrator can
                   apply the voluntary-compression filter.

Every provider returns a `Completion` carrying the text, the stop_reason, and
token counts. The orchestrator -- not the provider -- owns budget logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class Completion:
    text: str
    stop_reason: str  # "end_turn" | "max_tokens" | "stop_sequence" | other
    input_tokens: int
    output_tokens: int
    model_id: str


class Provider(Protocol):
    name: str

    def complete(
        self,
        system: str,
        messages: List[Message],
        max_tokens: int,
        temperature: float = 0.4,
    ) -> Completion:
        ...


# ---------------------------------------------------------------------------
# Mock provider: simulates budget-driven compression + emergent shorthand
# ---------------------------------------------------------------------------

# Deterministic "shorthand codebook" the mock Observer drifts toward as budget
# tightens. These are the invented, reused, non-dictionary tokens the metrics
# should detect. The mock always encodes the *same* decision info, so accuracy
# stays high (insider still decodable) while surface form gets cryptic.
_CODEBOOK = {
    "APPROVE": ["ok", "grn", "a+", "clr"],
    "REVIEW": ["chk", "amb", "r?", "hold"],
    "DECLINE": ["blk", "red", "x!", "nope"],
}

_APPROX_TOK = re.compile(r"\S+")


def _approx_tokens(text: str) -> int:
    return len(_APPROX_TOK.findall(text))


class MockProvider:
    """Offline stand-in.

    It expects the caller to embed two hints in the user message so it can
    behave deterministically:
      - a line 'GROUND_HINT=<CLASS>' telling the mock Observer the true class
        (mock only -- real Observer never gets the label; this is a simulation
        shortcut so the offline pipeline is decodable and testable).
      - a line 'BUDGET=<n>' with the soft token budget for this turn.
    For Decider/Auditor turns the message stream contains the codebook tokens,
    which the mock decodes back to a class.
    """

    name = "mock"

    def __init__(self, model_id: str = "mock-observer", role: str = "observer", seed: int = 0):
        self.model_id = model_id
        self.role = role
        self.seed = seed

    def complete(self, system, messages, max_tokens, temperature=0.4) -> Completion:
        joined = "\n".join(m.content for m in messages)
        if self.role == "observer":
            text = self._observer_message(joined)
        else:  # decider or auditor: decode the stream
            text = self._decode(joined)

        in_tok = _approx_tokens(system) + _approx_tokens(joined)
        out_tok = _approx_tokens(text)
        # Voluntary stop unless we genuinely exceeded max_tokens.
        stop = "end_turn" if out_tok <= max_tokens else "max_tokens"
        if stop == "max_tokens":
            text = " ".join(_APPROX_TOK.findall(text)[:max_tokens])
            out_tok = _approx_tokens(text)
        return Completion(text, stop, in_tok, out_tok, self.model_id)

    def _observer_message(self, joined: str) -> str:
        gh = re.search(r"GROUND_HINT=(\w+)", joined)
        bud = re.search(r"BUDGET=(\d+)", joined)
        cls = gh.group(1) if gh else "REVIEW"
        budget = int(bud.group(1)) if bud else 120

        # Pull a couple of functional cues from the record to make verbose
        # messages look natural (and give the metrics real English to compare).
        amount = re.search(r"amount=(\d+)", joined)
        age = re.search(r"account_age_days=(\d+)", joined)
        vel = "flag_velocity=True" in joined
        mis = "flag_mismatch=True" in joined

        code = _CODEBOOK[cls][0]

        if budget >= 80:
            # Verbose, natural language -- lots of dictionary words.
            reasons = []
            if amount:
                reasons.append(f"the amount is {amount.group(1)}")
            if age:
                reasons.append(f"the account age is {age.group(1)} days")
            if vel:
                reasons.append("there is a velocity flag")
            if mis:
                reasons.append("there is a mismatch flag")
            reason_txt = ", and ".join(reasons) if reasons else "the signals are clean"
            return (
                f"Looking at this record, {reason_txt}. Based on all of that, "
                f"my recommendation for this transaction is to {cls.lower()} it. "
                f"[{code}]"
            )
        elif budget >= 40:
            # Mid: shorter, mixes English with the code token.
            tag = _CODEBOOK[cls][1]
            hint = "hi-amt" if (amount and int(amount.group(1)) >= 3000) else "lo-amt"
            flags = ("V" if vel else "") + ("M" if mis else "")
            return f"{hint} {flags} -> {cls.lower()} {tag}"
        else:
            # Tight budget: pure invented shorthand, reused across rounds.
            tag = _CODEBOOK[cls][2]
            flags = ("v" if vel else "") + ("m" if mis else "")
            return f"{tag}{flags}"

    def _decode(self, joined: str) -> str:
        """Decider/Auditor: find the most recent codebook token or class word."""
        low = joined.lower()
        # Prefer explicit class words if present (verbose rounds).
        best = None
        best_pos = -1
        for cls in _CODEBOOK:
            pos = low.rfind(cls.lower())
            if pos > best_pos:
                best_pos, best = pos, cls
            for tok in _CODEBOOK[cls]:
                pos = low.rfind(tok)
                if pos > best_pos:
                    best_pos, best = pos, cls
        return best or "REVIEW"


# ---------------------------------------------------------------------------
# Bedrock provider (us-east-1) -- resolves inference profiles at runtime
# ---------------------------------------------------------------------------


class BedrockProvider:
    """Bedrock Converse API provider.

    Model IDs are resolved via inference profiles so this works on current-gen
    models that require a profile rather than a bare on-demand model ID.
    """

    name = "bedrock"

    def __init__(
        self,
        model_selector: str,
        region: str = "us-east-1",
        role: str = "observer",
    ):
        import boto3  # imported lazily so mock runs need no AWS deps

        self.region = region
        self.role = role
        self._bedrock = boto3.client("bedrock", region_name=region)
        self._runtime = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = self._resolve_profile(model_selector)

    def _resolve_profile(self, selector: str) -> str:
        """Match `selector` (substring, case-insensitive) against available
        inference profiles and return the profile ARN/ID to invoke."""
        paginator = self._bedrock.get_paginator("list_inference_profiles")
        candidates = []
        for page in paginator.paginate():
            for prof in page.get("inferenceProfileSummaries", []):
                name = prof.get("inferenceProfileName", "")
                pid = prof.get("inferenceProfileId", "")
                arn = prof.get("inferenceProfileArn", "")
                if selector.lower() in name.lower() or selector.lower() in pid.lower():
                    candidates.append(pid or arn)
        if not candidates:
            raise RuntimeError(
                f"No Bedrock inference profile matched '{selector}' in {self.region}. "
                f"Run `aws bedrock list-inference-profiles --region {self.region}`."
            )
        # Prefer the shortest match (usually the canonical profile).
        return sorted(candidates, key=len)[0]

    def complete(self, system, messages, max_tokens, temperature=0.4) -> Completion:
        conv = [
            {"role": m.role if m.role in ("user", "assistant") else "user",
             "content": [{"text": m.content}]}
            for m in messages
        ]
        resp = self._runtime.converse(
            modelId=self.model_id,
            system=[{"text": system}] if system else [],
            messages=conv,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        out = resp["output"]["message"]["content"][0]["text"]
        usage = resp.get("usage", {})
        # Bedrock stopReason: "end_turn" | "max_tokens" | "stop_sequence" | ...
        stop = resp.get("stopReason", "end_turn")
        return Completion(
            text=out,
            stop_reason=stop,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            model_id=self.model_id,
        )


def make_provider(kind: str, role: str, model_selector: str = "", region: str = "us-east-1", seed: int = 0) -> Provider:
    if kind == "mock":
        return MockProvider(model_id=f"mock-{role}", role=role, seed=seed)
    if kind == "bedrock":
        return BedrockProvider(model_selector=model_selector, region=region, role=role)
    raise ValueError(f"unknown provider kind: {kind}")
