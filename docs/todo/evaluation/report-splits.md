Break every metric down so failures can be explained. ^desc

## Why
One average hides where the system fails.

## How
- Splits: question type, ambiguity label, note tier, source, correct vs wrong.
- Questions whose gold document the SLM read cut (33, list in `exploration/slm_fields/runs/mine-r16-e2/cut_gold_questions.json`; [[fix-cut-long-documents]], parked).

**Done when:** `report.py` prints the splits.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 4. Card: [[Run the final evaluation]].
