Give the 3,151 long documents the SLM only partly read full fields, by the cheapest SLM method
that matches DeepSeek on the full text. ^desc

## The gap
The SLM saw at most 5,000 prompt tokens (`generate_vllm.py`, `PROMPT_TOKENS`). The extraction
instructions take ≈ 1,600 of them, so ≈ 3,400 tokens are left for the document; a longer
document lost its end.

| Source | Cut | Share |
|---|---|---|
| **Fireflies** (meeting transcripts) | 2,624 of 10,173 | **25.8%** |
| Confluence | 281 of 5,188 | 5.4% |
| Google Drive | 154 of 25,107 | 0.6% |
| Gmail | 81 of 121,390 | 0.1% |
| Slack, Linear, GitHub, HubSpot, Jira | 11 | ≈ 0% |

0.6% of the corpus, but a quarter of the meetings lost their ending, where decisions and
action items usually are: the fields field search relies on. Cut documents are ≈ 3,600
tokens at the median (lost ≈ 5%), ≈ 4,600 at p90 (lost ≈ a quarter), up to ≈ 10k (lost two
thirds).

**Why it happened:** the limit came from training (`--max-tokens 6144` per example: a time and
memory budget for one RTX 4090, not measured at its limit). Scoring and mining reused it so
the model never got inputs longer than in training. It wasn't checked against the length
per source: the held-out score set had only 268 Fireflies documents among 12.5k, so the
score couldn't show it. Training had the same blind spot: DeepSeek labelled full documents,
the model saw them cut. In vLLM GPU memory was not the limit (weights ≈ 4.5 GB).

**Lesson:** check a length limit against the length distribution per source before training
and before mining; chunk long documents instead of cutting them.

## Parked (23 Nov)
Not done for the final evaluation: it barely moves the score.
- **33 of 500 questions (6.6%)** have a gold document among the cut ones; for **9 (1.8%)** all
  gold documents are cut. Gold sources: Confluence 29, Fireflies 12, Google Drive 2; mostly
  `project_related` (18 of its 40).
- The cut only affects field search; the general search (BM25 + gte over chunks) reads these
  documents in full, and field search mainly earns its keep on Slack (1 Slack document cut).
- The 33 questions are marked in `exploration/slm_fields/runs/mine-r16-e2/cut_gold_questions.json` and reported as their own split
  ([[report-splits]]).

**Reopen if** meetings and wiki pages need good field search (a quarter of Fireflies lost its
end), or the split shows these questions fail clearly more often. Cheap version then (≈ $1):
DeepSeek on a sample of 500 cut Fireflies + Confluence documents (≈ $0.75 off-peak) to pick
the method, then the winning SLM method on all 3,151 on the local GPU.

## How (cheap)
1. **Reference:** DeepSeek labels the 3,151 documents on their **full** text, same prompt as
   the training labels (`exploration/slack_extraction/label.py`, `deepseek-flash`), off-peak.
2. **SLM methods** on the same 3,151 documents:

   | Method | What the SLM sees | Context |
   |---|---|---|
   | A · cut (have it) | first ≈ 3,400 tokens | end lost |
   | B · head + tail | two pieces, fields merged | tail lacks context |
   | C · tail with a header | tail + title + the head's fields (people, topic) | most context kept, within training length |
   | D · one long pass | the whole document, limit ≈ 11k | longer than any training input |

3. **Score** each method against DeepSeek: field word F1 per field, like
   `exploration/slm_fields/score_heldout.py`; Fireflies reported apart, plus decisions and
   action items apart.
4. The best method's fields replace the cut ones in
   `exploration/slm_fields/runs/mine-r16-e2/labels.jsonl` (raw outputs kept), then
   [[import-embed-mined-fields]].

If retraining ever happens: chunk long training documents instead of cutting them, so labels
and visible text match.

## Cost (estimate, 23 Nov)
**≈ $5–6.** DeepSeek off-peak ≈ $4–7 (≈ 4,000 input tokens per document at $0.15/M, the
instructions cached; ≈ 1,500 output tokens at $0.60/M; ≈ $0.0015 × 3,151); double at peak
(UTC 01–03 and 06–09 on weekdays). SLM runs B–D ≈ 10 min of generation each: free on the local
8 GB GPU if vLLM fits with small batches, else ≈ $0.50 on one rented 4090 for all three.

**Done when:** methods A–D scored against DeepSeek, the best one chosen, its fields in the
labels file.

Card: [[Run the final evaluation]].
