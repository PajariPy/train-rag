Judge every run twice on the same answers: the strict rule (before 20 Nov) and today's lenient one.

## Why
Lenient counts a right mentioned case as correct; only the strict number compares with run 31 (77.6%).

## How
- Two prompts, `judge_strict` and `judge_lenient`, both in `JUDGES` (`evaluation/registry.py`).
- `facts_covered` only from the case the judge marked correct.
- Check: rejudge a handful of run-31 answers strictly; verdicts match run 31.

**Done when:** both verdicts stored per answer; the run-31 check matches.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 3. Board: [[Run the final evaluation]].
