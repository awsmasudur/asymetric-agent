# Token-Budget Language Drift in Asymmetric Agents

Do two agents coordinating under a shrinking token budget drift into compressed,
cryptic shorthand — and is that shorthand a genuine, grounded code or just
shorter text? This harness tests it with a scorable classification task, an
information asymmetry (Observer sees data, Decider doesn't), and paired
shrinking-budget vs fixed-budget conditions.

It runs three budget conditions on identical data — a roomy fixed budget
(control), a tight fixed budget (fixed-low), and a shrinking budget (treatment)
— and measures whether the messages drift into a private code, stay readable,
and how their structure changes under pressure.

## Layout

```
drift/
  task.py            synthetic records + hidden ground-truth classifier
  providers.py       provider abstraction: mock (offline) + bedrock (us-east-1)
  agents.py          Observer / Decider / Auditor
  config.py          ExperimentConfig + budget schedules
  orchestrator.py    paired treatment/control session runner
  logging_store.py   SQLite + JSONL logging
  metrics.py         drift/convention/grounding metrics + analysis
run.py               CLI: run + analyze
preflight_bedrock.py read-only AWS check + inference-profile lister
```

## 1. Validate cheaply offline (no AWS, no cost)

The mock provider deterministically simulates an Observer that compresses harder
as its budget shrinks. Use it to confirm the pipeline and metrics before spending
anything.

```powershell
pip install scipy
python run.py --provider mock --sessions 8 --rounds 30
```

On the mock, compression and vocabulary-growth signals show up and the task stays
decodable; the opacity and emergence tests stay flat because the mock's codebook
is fixed and shared by construction — those only become meaningful with real LLMs.

## 2. Preflight Bedrock (read-only, no token cost)

Verifies credentials and lists inference profiles in us-east-1 so you can pick
model selectors. Makes no model invocations.

```powershell
pip install boto3
python preflight_bedrock.py --region us-east-1 --match claude
```

Copy a distinctive substring of a profile id/name to use as a selector below.

## 3. Run the Bedrock pilot (opt-in — this spends tokens)

Start with a small pilot (a few sessions) before scaling. The harness runs all
three conditions per session. Pick selectors from the preflight output.

```powershell
python run.py --provider bedrock --region us-east-1 `
  --observer <observer-selector> `
  --decider  <decider-selector> `
  --auditor  <auditor-selector> `
  --sessions 3 --rounds 30
```

Notes:
- The word budget is enforced with token headroom (`max_tokens ≈ word_budget ×
  1.5 + 32`), never at the exact budget, so the API's hard stop isn't mistaken
  for voluntary compression. Only messages that stopped on their own
  (`stop_reason == "end_turn"`) count toward the drift metrics; truncated ones
  are logged separately.
- Using the **same model** for all three roles isolates the effect of the
  information asymmetry and the budget from any difference in model capability.
- Results append to `runs/results.sqlite`; re-analyze any run without re-running:
  `python run.py --analyze-only <run_id>`.

## Cost / safety

- The mock phase is free and offline.
- `preflight_bedrock.py` calls only control-plane list APIs (no inference).
- Only step 3 invokes models and incurs cost. Start with `--sessions 3` to gauge
  spend before scaling to the >=15 sessions the analysis wants for tight
  intervals.
