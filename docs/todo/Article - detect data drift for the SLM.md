For the articles only (theoretical, no build): how to know that newly arriving data hasn't
drifted away from what the SLM was trained on, so it can still process it reliably.

## Why
The SLM route (see [[Article - strategy for messier documents]]) is cheap because the model
is trained once and then runs on every new document. That only holds while new documents
look like the training data: new tools, templates, topics or writing styles can quietly
lower extraction quality, and nothing in the pipeline would notice.

## How
Still up to research. Starting idea: measure accuracy on a sample of new data and compare it
with the accuracy on the old data (e.g. re-label a small fresh sample with the LLM, score the
SLM against it, compare with the calibration-set score).

Questions to research:
- what signals show drift without labels (input statistics, embedding distribution shift,
  SLM confidence from logits, share of empty or malformed outputs);
- how often to check, and how big a sample is needed;
- what threshold triggers re-labelling and retraining.

## Result (22 Nov)
Draft approved: `docs/articles/009_detecting_data_drift_for_the_slm.md`, the next chapter of
the same article as 008 (branch `feature-messier-data-strategy`). With labels: user
satisfaction correlated with ingestion, confirmed by fresh LLM labels against the
calibration score. Without labels: input statistics, nearest-neighbour distance to a golden
set of embeddings (Sun et al. 2022) with KS on the distances, SLM confidence (logits or a
correctness classifier), broken output, retrieval scores; KS, chi-square and PSI tests.
When to act: SLA with an error budget, look immediately when 10–20% is burnt and it's on
track to run out. Centroid distance rejected for a many-topic corpus; domain classifier and
Failing Loudly noted for later.
