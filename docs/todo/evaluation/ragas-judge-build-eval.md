Build our own Gemini judge for the four RAGAS metrics, and check it against the `ragas`
library before the full run uses it. ^desc

## Why
RAGAS names (faithfulness, answer relevance, context precision, context recall) are what
readers know. The library is blind to our answer layout (main case + mentioned similar
cases) and to alternate gold, and makes ≈ 10–15 calls per question; our judge makes one.
Checking the two against each other lets the article say how close our numbers are to the
library, instead of "RAGAS-style metrics we defined ourselves".

## The RAGAS sample (decision 23 Nov)
RAGAS is scored on **one fixed sample of 100 benchmark questions**, used both for the check
against the library (step 2) and in the full run ([[full-evaluation-run]]); not on all 575.
- **Stratified by question type**, proportional to the 500: basic 35, semantic 25,
  intra-document 8, project-related 8, constrained 6, conflicting 4, completeness 4,
  miscellaneous 4, info-not-found 4, high-level 2. Within each type, spread over the gold
  document's source; fixed seed.
- **Checked against the full set:** gold-document source within 2 points for every source
  (preview 23 Nov: Gmail 9.0% vs 11.0% is the largest gap). Once the ambiguity labels exist
  ([[ambiguity-labels-alternate-gold]]), check that share too; redraw with another seed if
  any split is more than 5 points off.
- Off-topic questions are left out (no document should answer, so the context metrics mean
  nothing). Context recall is averaged only over questions with a gold document (not the 4
  info-not-found ones).
- Report the RAGAS means with a 95% confidence interval (100 questions: about ± 0.05–0.1 on a
  0–1 scale), overall only; per type the groups are too small.
- Store the sample's question ids with the seed, so every run uses the same 100.

## How
1. **Build:** one Gemini call per question with the RAGAS definitions; aware of mentions and
   of alternate gold ([[ambiguity-labels-alternate-gold]]).
2. **Check against the library**, before the full run, with no new pipeline calls:
   - the stored run-31 answers of **the RAGAS sample** (question, answer, retrieved
     documents are in the database);
   - `ragas` with Gemini through `langchain-google-genai`, and our judge, on the same 100;
   - report the mean difference and the correlation per metric;
   - the main case + mentions layout is checked on the smoke answers
     ([[smoke-test-clarify]]), where the library can't follow it.
3. If a metric disagrees badly, fix our prompt (or drop that metric) before
   [[full-evaluation-run]] uses the judge on all 575 answers.

## Cost (estimate, 23 Nov)
**≈ $3–6 for building and checking:** our judge on ≈ 50 test answers while building (≈ $0.50)
and on the 100-answer sample (≈ $1); the library on the same 100 (≈ 10–15 calls each,
≈ $2–4). Afterwards ≈ $0.009 per question, ≈ $0.90 per run on the 100-question sample, counted in
[[full-evaluation-run]]. Cheapest cut: skip step 2 (≈ −$3–5); the report then calls
the numbers "our judge, RAGAS definitions" without the agreement.

**Done when:** the judge gives four numbers per answer, and an agreement table per metric
against `ragas` is written.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 4. Card: [[Run the final evaluation]].
