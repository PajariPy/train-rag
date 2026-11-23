# Synthetic Training Questions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** About 2,000 train and 400 dev one-document questions (basic, semantic, intra-document
reasoning, miscellaneous) written with the benchmark's prompts, each with its source document
as the positive and 4–5 hard negatives picked by the reranker's biggest drop.

**Architecture:** The pure logic (type and source quotas, the biggest-drop rule) lives in a new
`train_rag.training` package with tests. Three resumable scripts in
`exploration/training_questions/` run the pipeline: pick sources → generate questions with
DeepSeek → search, rerank and mine negatives. Each writes a data file the next one reads.

**Tech Stack:** Python 3.13, uv, SQLAlchemy/pgvector (Postgres on 5434), httpx (DeepSeek
`deepseek-flash`), `RetrievalPipeline` (gte + BM25 + RRF), `GteModernBertReranker`,
`RunProgress`, pytest, ruff, mypy strict.

**Spec:** `docs/brainstorm/005_synthetic_training_questions.md` (with brainstorm 004 for the
drop rule).

## Global Constraints

- Type mix: basic 175 : semantic 125 : intra_document_reasoning 40 : miscellaneous 20, scaled
  to the run size. Generate 2,600 train + 520 dev (2,000 + 400 kept, about 30% skipped).
- Dev is picked first, by document. Dev documents and their 20 nearest by gte are excluded
  from train sources. Dev is for every hypothesis and setting; the benchmark is checked once
  at the end.
- Per-source mix per type (paper Table 10, order confluence, fireflies, github, gmail,
  google_drive, hubspot, jira, linear, slack):
  basic 19, 11, 22, 15, 24, 18, 21, 24, 21; semantic 15, 4, 14, 18, 14, 14, 23, 10, 13;
  intra 2, 3, 2, 9, 0, 1, 2, 9, 12. Miscellaneous: random from the misc pool.
- Misc pool: documents whose benchmark path (`generated_data/uuid_index.json`) starts with
  `slack/sports/`, `slack/memes/`, `confluence/misc/`, `jira/misc-requests/`,
  `fireflies/interviews/`, `github/hackathons/`, or contains `/misc-assets/` or
  `/_temp_uploads/`.
- Intra-document sources: `char_count >= 3000` (the benchmark's `--min-doc-length`).
- Excluded as sources: every `expected_doc_ids` of all 500 questions plus each one's 20
  nearest documents by gte. Excluded as negatives: every `expected_doc_ids`.
- Prompts: the benchmark's `BASIC_QUERIES_PROMPT` (basic and miscellaneous),
  `SEMANTIC_QUERIES_PROMPT`, `INTRA_DOCUMENT_REASONING_PROMPT` and
  `INTRA_DOCUMENT_REASONING_ANSWER_GENERATION` (validation), copied verbatim with an MIT
  attribution line.
- Drop rule: window 30, at most 5 negatives, taken in rank order just below the drop; skip the
  question when its source document isn't above the drop.
- Seed 7 for all sampling. DeepSeek budget: stop if spend passes $2.00 (balance $2.38).
- Code style: the python-refine rules (descriptive names, strict types, Google docstrings,
  blank lines, files under 300 lines). Checks: `uv run ruff check . && uv run mypy src tests && uv run pytest`.
- Git: the user commits. Don't commit; work on the current branch.

## Review Focus

- A document with an empty or tiny body (some Slack/Gmail rows): the generator returns junk.
  Skip sources under 200 characters; test in Task 1 (`test_short_documents_are_not_sources`).
- DeepSeek returns a multi-line answer, quotes or "Query:" prefixes despite the prompt: strip
  to one clean line; test in Task 2 (`test_clean_question`).
- The source document missing from the reranked list (search never found it): it must still be
  scored, so it's added to the list before reranking; test in Task 3 (`test_positive_is_scored`).
- Fewer than 5 candidates below the drop: keep what's there, but skip if zero; test in Task 1
  (`test_no_negatives_means_skip`).
- An interrupted run: every script resumes from its output file without paying twice; checked
  by rerunning each script once after it finishes (Tasks 2 and 3, verification steps).

---

### Task 1: Quotas and the drop rule (`train_rag.training`)

**Files:**
- Create: `src/train_rag/training/__init__.py`, `src/train_rag/training/quotas.py`,
  `src/train_rag/training/drop_rule.py`
- Test: `tests/training/test_quotas.py`, `tests/training/test_drop_rule.py`

**Interfaces:**
- Produces:
  - `TYPE_WEIGHTS: dict[str, int]` and `SOURCE_WEIGHTS: dict[str, dict[str, int]]` (Global
    Constraints values).
  - `split_by_weight(total: int, weights: Mapping[str, int]) -> dict[str, int]`: largest
    remainder rounding; values sum to `total`; zero weights get 0.
  - `is_usable_source(char_count: int, question_type: str) -> bool`: ≥ 200 characters, and
    ≥ 3,000 for `intra_document_reasoning`.
  - `biggest_drop(scores: Sequence[float], window: int = 30) -> int`: scores sorted high to
    low; returns how many sit above the largest step between neighbours in the first
    `window + 1`.
  - `pick_negatives(scores: Mapping[str, float], positive: str, excluded: AbstractSet[str],
    count: int = 5, window: int = 30) -> list[str] | None`: None = skip the question.

- [ ] **Step 1: Write the failing tests**

```python
def test_split_by_weight_sums_and_rounds():
    assert split_by_weight(2600, TYPE_WEIGHTS) == {
        "basic": 1264, "semantic": 903, "intra_document_reasoning": 289, "miscellaneous": 144}
    assert split_by_weight(10, {"a": 1, "b": 0}) == {"a": 10, "b": 0}

def test_short_documents_are_not_sources():
    assert not is_usable_source(150, "basic")
    assert is_usable_source(500, "basic")
    assert not is_usable_source(2999, "intra_document_reasoning")

def test_biggest_drop_counts_documents_above_it():
    assert biggest_drop([0.95, 0.80, 0.79, 0.78]) == 1
    assert biggest_drop([0.90, 0.89, 0.70, 0.69]) == 2

def test_pick_negatives_takes_just_below_the_drop():
    scores = {"pos": 0.95, "a": 0.80, "b": 0.79, "c": 0.78, "d": 0.77}
    assert pick_negatives(scores, "pos", set(), count=2) == ["a", "b"]

def test_positive_below_the_drop_skips_the_question():
    scores = {"x": 0.95, "pos": 0.80, "a": 0.79}
    assert pick_negatives(scores, "pos", set()) is None

def test_excluded_and_above_positive_never_negatives():
    scores = {"pos": 0.95, "evaldoc": 0.80, "a": 0.79}
    assert pick_negatives(scores, "pos", {"evaldoc"}) == ["a"]

def test_no_negatives_means_skip():
    assert pick_negatives({"pos": 0.9}, "pos", set()) is None
```

(2600 × 175/360 = 1263.9, × 125/360 = 902.8, × 40/360 = 288.9, × 20/360 = 144.4; floors sum to
2,597, and the three largest remainders get +1.)

- [ ] **Step 2: Run** `uv run pytest tests/training -v` — expected FAIL (module missing).
- [ ] **Step 3: Implement** the five names above. In `pick_negatives`: sort by score; if the
  positive's rank is greater than `biggest_drop`, return None; candidates = documents after
  the drop, not excluded, scoring below the positive; return the first `count`, or None if
  empty.
- [ ] **Step 4: Run** `uv run pytest tests/training -v` — expected PASS; then the full checks.

### Task 2: Pick sources and generate questions

**Files:**
- Create: `exploration/training_questions/prompts.py` (the four benchmark prompts, verbatim,
  MIT attribution), `exploration/training_questions/sources.py`,
  `exploration/training_questions/generate.py`
- Data: `exploration/training_questions/data/uuid_index.json` (downloaded once from the
  benchmark repo), `data/sources.json`, `data/questions.jsonl`
- Test: `tests/training/test_clean_question.py` (with `clean_question` placed in
  `src/train_rag/training/questions.py` so it is testable)

**Interfaces:**
- Consumes: Task 1 quotas.
- Produces:
  - `data/sources.json`: `[{"doc_id", "source_type", "question_type", "split": "train" | "dev"}]`,
    3,120 rows.
  - `data/questions.jsonl`: one line per kept question `{"id": "trq_00001", "split", "question_type",
    "doc_id", "source_type", "question", "valid": bool}`.
  - `clean_question(raw: str) -> str | None` in `train_rag.training.questions`: first
    non-empty line, strips quotes, "Query:" prefixes and markdown; None if empty or over
    400 characters.

- [ ] **Step 1: Write the failing test** `test_clean_question`:
  `clean_question('"Query: What is X?"\n') == "What is X?"`,
  `clean_question("") is None`, `clean_question("a" * 500) is None`.
- [ ] **Step 2: Run** it — FAIL. **Step 3: Implement** `clean_question`. **Step 4:** PASS.
- [ ] **Step 5: `sources.py`**: exclusion set (expected docs + 20 nearest by gte each, one
  pgvector query per expected doc); misc pool from `uuid_index.json`; per type, sample
  `split_by_weight` per source from documents passing `is_usable_source`, seed 7, no document
  used twice. Pick dev first (520), add dev documents and their 20 nearest to the exclusion
  set, then pick train (2,600). Write `data/sources.json`. Run it; expected: 3,120 rows,
  0 overlap with the benchmark exclusion set, 0 train documents among dev's 20 nearest
  (print all three checks).
- [ ] **Step 6: `generate.py`**: DeepSeek chat completions (as in
  `exploration/fact_extraction/label.py`), prompt by type (basic prompt for miscellaneous),
  document as `{document_title}` + `{document_contents}`, 8 threads, append each result to
  `questions.jsonl` (resume by `doc_id`), track spend from `usage` tokens at $0.30/$1.20
  per million (peak; conservative), stop at $2.00, `RunProgress` with spend in the note.
  Intra-document questions get the validation call; `valid` from its JSON.
- [ ] **Step 7: Pilot** `generate.py --limit 20`; read the 20 questions; confirm they look
  like benchmark questions of their type and the spend per question is about $0.0005.
  **Ask the user before the full run** (paid).
- [ ] **Step 8: Full run**, then rerun once to confirm it does nothing (resume works).

### Task 3: Mine negatives and write the training set

**Files:**
- Create: `exploration/training_questions/mine.py`
- Data: `data/rerank_scores.json` (cache), `data/training_set.jsonl`
- Test: `tests/training/test_drop_rule.py` (add `test_positive_is_scored` for the helper
  `candidates_with_positive(fused: list[str], positive: str) -> list[str]` in
  `train_rag.training.drop_rule`: the positive is appended when missing, never duplicated)

**Interfaces:**
- Consumes: `data/questions.jsonl` (valid rows only), Task 1 `pick_negatives`,
  `candidates_with_positive`.
- Produces: `data/training_set.jsonl` and `data/dev_set.jsonl`: `{"id", "question_type", "question", "positive",
  "negatives": [doc_id, ...], "scores": {doc_id: float}}`.

- [ ] **Step 1: Write and run the failing test** `test_positive_is_scored`, implement, PASS.
- [ ] **Step 2: `mine.py`**: for each valid question: `RetrievalPipeline.load(rerank=False)`,
  `semantic_search` + `keyword_search`, `reciprocal_rank_fusion` to 50 (as
  `exploration/retriever_attribution/attribution.py`), `candidates_with_positive`, rerank
  title + content with `GteModernBertReranker` (as `exploration/knee_check/knee_sample.py`),
  cache scores per question, `pick_negatives` with the exclusion set from Task 2. Write kept
  rows to `training_set.jsonl` or `dev_set.jsonl` by split. `RunProgress`. Expected time
  about 2.5 h on the laptop (3,120 × ~51 documents × ~59 ms).
- [ ] **Step 3: Run** on 50 dev questions first; check skip rate and grey-zone size look like the
  knee check (about 9% skipped, median 1 above the drop). Then the full run, then rerun once
  (does nothing).
- [ ] **Step 4: Print the report**, for train and dev separately: kept questions per type, skip rate per type, grey-zone size
  (median, p75), negatives per question, total DeepSeek spend. Add it to
  `docs/brainstorm/005_synthetic_training_questions.md` as "First run".
