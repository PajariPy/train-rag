# Evaluation drift: what the benchmark no longer measures

Status: brainstorm notes, 23 November 2026. Full design:
[2026-11-23 evaluation update](../superpowers/specs/2026-11-23-evaluation-update-design.md).
Todo card: [Prepare the evaluation for the final run](../todo/Prepare%20the%20evaluation%20for%20the%20final%20run.md).

The benchmark evaluation (EnterpriseRAG-Bench, 500 questions, one gold answer, judged on
correctness × completeness; last comparable result run 31, `hybrid-ft-rerank`, 77.6%) was
built for a system that returns **one answer from one document**. Since then the pipeline
(`field-agent`, `field-agent-clarify`, 20 Nov) was tuned on other sets (Slack dev 200,
dev-contradictions 83) and changed what an answer is. Eight things drifted.

| # | Drift | Where | Effect on today's numbers |
|---|---|---|---|
| 1 | **Persona and memory notes are weak and not graded by quality.** Notes come only from `expected_doc_ids[0]`; clean notes are vague on purpose ("no deeper than the question"); 25% hold one wrong detail (seed 7); no note says outcome or customer type. Scores aren't split by note type. | `evaluation/memory_notes.py`, `prompts.yaml: memory_note_*`, `clarify.py: write_memories` | Can't tell whether asking fails because of the agent or because the hint was useless. Questions with no gold document (`info_not_found`) would fail on `[0]`. |
| 2 | **An answer is now the main case plus similar cases, cited.** The judge became lenient on 20 Nov: a matching *mentioned* case counts as right. | `prompts.yaml: judge_instructions` | Not comparable with run 31 (strict rule). The lenient rule rewards listing many cases. |
| 3 | **Citations are stored but not scored.** `cited_doc_ids` is saved; no metric reads it. `invalid_extra_docs` counts *retrieved* documents, which the agent now returns on purpose. | `evaluation/pipeline.py`, `evaluation/metrics.py` | "Right document cited, as main or as mention", the thing we built, is unmeasured. |
| 4 | **Completeness can rise from mentions.** Facts of a near-twin case may be marked covered. | judge `facts_covered` | Completeness goes up for the wrong reason. |
| 5 | **Benchmark questions have no ambiguity label.** The pipeline is built around consistent / specific / ambiguous; only the 83 dev-contradictions questions carry one. For an ambiguous question the single gold answer may be one of several valid twins. | `exploration/slack_extraction/ambiguity.py` (Slack dev only) | Can't tell whether asking or side by side helps where it should, nor whether a "wrong" answer picked a valid twin. |
| 6 | **Asking has no metric on the benchmark.** Asks per question, asks on ambiguous vs other questions. | `evaluation/clarify_report.py` (dev-oriented) | `field-agent` and `field-agent-clarify` can't be compared fairly. |
| 7 | **Tuning set ≠ test set.** Tuned on Slack; the benchmark spans 9 sources; field search covers only mined (Slack) documents. | registry, mined fields | The total hides that non-Slack questions run on the first search only. Needs a per-source split. |
| 8 | **Noise and open choices.** Agent run-to-run noise 6–10 points; the cap of 2 mentioned cases isn't confirmed (`answer_check.py`, ~$0.80). | todo cards | One run misleads. |

Also missing, decided in the same session:

- **Common metrics** a reader recognises: MRR, nDCG, precision@k, hit rate; RAGAS
  faithfulness, answer relevance, context precision, context recall; agent effort (steps,
  tool calls, tokens per question).
- **Refusal:** no off-topic questions in the test set, and the pipeline has no "no
  document answers" decision. The 20 look-alike questions (`info_not_found`) already exist.
- **Operations:** only a run's total time and tokens are stored; nothing per stage.

Known gaps left for later: multi-turn chats (needs the query-from-history step first);
questions about private data the tools don't hold (article only).

## Decisions (23 Nov)

- Judge every run both ways: strict (pre-20 Nov rule, compares with run 31) and lenient
  (today's rule, matches the product).
- Ambiguity labels on all 500 questions, with the twin documents that also answer fully
  stored as alternate gold.
- Memory notes v3 in four tiers: vague 40%, concrete 25%, wrong 25%, none 10%.
- RAGAS: own Gemini judge on all questions, the `ragas` library on a seeded sample of ~100
  to show they agree.
- Out-of-scope set: 50–100 off-topic questions, built in rounds (generate → search wide →
  rate each document → accept or dismiss and teach the generator). General-knowledge
  questions are refused. Rounds because the document set will grow: the check is stored
  per question and rerun after new documents arrive, so a question that became answerable
  leaves the set instead of turning into a false "refusal is right".
- Refusal detection: a fourth verdict, "no document answers", in the agent's existing check
  turn.
- Operations: time every stage per question; p50 / p95 latency, cost per 1,000 questions.
