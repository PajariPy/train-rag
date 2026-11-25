Judge every run twice on the same answers: the strict rule (before 20 Nov) and today's lenient one. ^desc

## Why
Lenient counts a right mentioned case as correct; only the strict number compares with run 31 (77.6%).

## How
- Two prompts, `judge_strict` and `judge_lenient`, both in `JUDGES` (`evaluation/registry.py`).
- `facts_covered` only from the case the judge marked correct.
- Check: rejudge a handful of run-31 answers strictly; verdicts match run 31.

## Cost (estimate, 23 Nov)
**≈ $0.10** for the check (≈ 20 run-31 answers judged strictly at ≈ $0.004). The second judge on every run is counted in [[full-evaluation-run]] (≈ +$2.3 per run).

**Done when:** both verdicts stored per answer; the run-31 check matches.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 3. Card: [[Run the final evaluation]].
