import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def imports():
    import html
    import json
    import re
    import statistics

    import altair as alt
    import marimo as mo
    import numpy as np
    import yaml
    from sqlmodel import Session

    from train_rag.data_loaders import EnterpriseRagBenchLoader
    from train_rag.database import get_engine
    from train_rag.embedder import MiniLmEmbedder
    from train_rag.models import QuestionType
    from train_rag.pipelines.data_ingestion import CHUNKERS
    from train_rag.processing.normalize import (
        normalize_content,
        normalize_documents,
        normalize_text,
    )
    from train_rag.repository.document import (
        count_documents_by_source,
        ids_of_doc_ids,
        load_documents,
    )
    from train_rag.repository.evaluation import results_of_run
    from train_rag.repository.question import load_questions

    return (
        CHUNKERS,
        EnterpriseRagBenchLoader,
        MiniLmEmbedder,
        QuestionType,
        Session,
        alt,
        count_documents_by_source,
        get_engine,
        html,
        ids_of_doc_ids,
        json,
        load_documents,
        load_questions,
        mo,
        normalize_content,
        normalize_documents,
        normalize_text,
        np,
        re,
        results_of_run,
        statistics,
        yaml,
    )


@app.cell
def settings(mo):
    PIPELINE_RUN = 8
    AGENT_RUN = 14
    # Stage 2: the same pipeline as run 8, rerun after it started citing its documents.
    CITED_RUN = 15
    FAILURES = mo.notebook_dir().parent / "exploration" / "retrieval_failures"
    DOCUMENTS_FILE = "data/enterprise_rag_bench/data/documents/test.parquet"
    return AGENT_RUN, CITED_RUN, DOCUMENTS_FILE, FAILURES, PIPELINE_RUN


@app.cell
def _(mo):
    mo.md("""
    # Search misses on semantic questions

    Data: run 8 (the pipeline: BM25 and gte, fused, top 50 reranked, top 5 to Gemini) and
    run 14 (the agent, on the questions run 8 got wrong), read from Postgres, plus two files
    from the scripts in `exploration/retrieval_failures/` (not committed):

    - `analyze.py` writes `data/failures.json`: ranks and scores of the expected document
      and the pipeline's top 5;
    - `case_texts.py` writes `data/cases.json`: questions, answers, agent searches and where
      each expected document is in the dataset;
    - `candidates.py` writes `data/candidates.json`: the fused top 50 of every semantic
      question.

    Stage 2 runs the first two again on every wrong answer of the cited run, into
    `data/cases_run15.json` and `data/failures_run15.json`.

    Each takes the output path as its first argument; the first two also take question ids.
    `exploration/retrieval_failures/answer_passages.yaml` holds the passage of each expected
    document that contains the answer.
    """)
    return


@app.cell
def load_runs(
    AGENT_RUN,
    EnterpriseRagBenchLoader,
    PIPELINE_RUN,
    QuestionType,
    Session,
    count_documents_by_source,
    get_engine,
    load_questions,
    results_of_run,
):
    engine = get_engine()
    with Session(engine) as _session:
        questions = {
            _question.id: _question
            for _question in load_questions(
                _session, EnterpriseRagBenchLoader.name, QuestionType.SEMANTIC
            )
        }
        pipeline_results = {
            _result.question_id: _result for _result in results_of_run(_session, PIPELINE_RUN)
        }
        agent_results = {
            _result.question_id: _result for _result in results_of_run(_session, AGENT_RUN)
        }
        corpus_sizes = count_documents_by_source(_session, EnterpriseRagBenchLoader.name)
    return agent_results, corpus_sizes, engine, pipeline_results, questions


@app.cell
def outcomes(
    FAILURES,
    agent_results,
    corpus_sizes,
    json,
    pipeline_results,
    questions,
):
    OUTCOMES = [
        "answered right",
        "wrong, document given to the LLM",
        "wrong, document found but ranked below 5",
        "wrong, document not found",
    ]
    # The fused top 50 per question: the documents the reranker reorders (candidates.py).
    candidates = json.loads((FAILURES / "data" / "candidates.json").read_text())

    def outcome(question):
        """What happened to one question in the pipeline run."""
        result = pipeline_results[question.id]
        expected = set(question.expected_doc_ids)
        if result.correct:
            return OUTCOMES[0]
        if expected & set(result.retrieved_doc_ids):
            return OUTCOMES[1]
        if expected & set(candidates[question.id]):
            return OUTCOMES[2]
        return OUTCOMES[3]

    question_outcomes = {
        _id: (_question.source_types[0].value, outcome(_question))
        for _id, _question in questions.items()
    }
    _corpus_total = sum(corpus_sizes.values())
    source_rows = []
    for _source in sorted(corpus_sizes):
        _mine = [_id for _id, (_s, _) in question_outcomes.items() if _s == _source]
        _counts = {_o: sum(question_outcomes[_id][1] == _o for _id in _mine) for _o in OUTCOMES}
        _rescued = sum(
            1
            for _id in _mine
            if question_outcomes[_id][1] != OUTCOMES[0]
            and _id in agent_results
            and agent_results[_id].correct
        )
        source_rows.append(
            {
                "source": _source,
                "corpus documents": corpus_sizes[_source],
                "share of corpus": corpus_sizes[_source] / _corpus_total,
                "questions": len(_mine),
                **_counts,
                "not found rate": _counts[OUTCOMES[3]] / len(_mine) if _mine else 0.0,
                "rescued by agent": _rescued,
            }
        )
    source_rows.sort(key=lambda _row: -_row["not found rate"])
    return OUTCOMES, question_outcomes, source_rows


@app.cell
def _(OUTCOMES, mo, question_outcomes, source_rows):
    _total = len(question_outcomes)
    _counts = {_o: sum(_v[1] == _o for _v in question_outcomes.values()) for _o in OUTCOMES}
    _lines = [
        "| Source | Corpus documents | Questions | Answered right | Wrong, document given "
        "| Wrong, ranked below 5 | Wrong, not found | Not found rate | Rescued by agent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _row in source_rows:
        _lines.append(
            f"| {_row['source']} | {_row['corpus documents']:,} "
            f"({_row['share of corpus']:.0%}) | {_row['questions']} "
            + "".join(f"| {_row[_o]} " for _o in OUTCOMES)
            + f"| **{_row['not found rate']:.0%}** | {_row['rescued by agent']} |"
        )
    _lines.append(
        f"| **all** | | {_total} "
        + "".join(f"| {_counts[_o]} " for _o in OUTCOMES)
        + f"| **{_counts[OUTCOMES[3]] / _total:.0%}** "
        f"| {sum(_row['rescued by agent'] for _row in source_rows)} |"
    )
    mo.md(
        "## Outcomes by source\n\n"
        "Every semantic question has one expected document, so each question belongs to one "
        "source. Outcomes are from run 8:\n\n"
        "- **answered right**;\n"
        "- **wrong, document given to the LLM**: the expected document was among the 5 given "
        "to Gemini;\n"
        "- **wrong, document found but ranked below 5**: it was in the fused top 50 that the "
        "reranker reorders, but not in the final 5;\n"
        "- **wrong, document not found**: it wasn't in the fused top 50.\n\n"
        "*Not found rate* is the share of the source's questions in the last group. *Rescued "
        "by agent* counts wrong answers the agent (run 14) got right. The top 50 lists come "
        "from `candidates.py`, re-run after run 8; the approximate vector index can return "
        "slightly different results between runs.\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def _(OUTCOMES, alt, mo, source_rows):
    # Outcome colours: green for right, then amber, orange and red as the expected document
    # gets further from the LLM. Colour is backed by the legend and the table above.
    _colors = ["#2f855a", "#d69e2e", "#dd6b20", "#c53030"]
    _values = [
        {
            "source": _row["source"],
            "outcome": _o,
            "order": _i,
            "questions": _row[_o],
            "share": _row[_o] / _row["questions"] if _row["questions"] else 0.0,
        }
        for _row in source_rows
        for _i, _o in enumerate(OUTCOMES)
    ]
    _chart = (
        alt.Chart(alt.Data(values=_values))
        .mark_bar(stroke="white", strokeWidth=2)
        .encode(
            y=alt.Y("source:N", sort=[_row["source"] for _row in source_rows], title=None),
            x=alt.X(
                "share:Q",
                stack="normalize",
                title="Share of the source's questions",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(domain=OUTCOMES, range=_colors),
                legend=alt.Legend(orient="top", title=None),
            ),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("source:N"),
                alt.Tooltip("outcome:N"),
                alt.Tooltip("questions:Q"),
                alt.Tooltip("share:Q", format=".0%"),
            ],
        )
        .properties(height=alt.Step(30), width="container")
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def load_cases(FAILURES, json, yaml):
    failures = json.loads((FAILURES / "data" / "failures.json").read_text())
    cases = {
        _case["question_id"]: _case
        for _case in json.loads((FAILURES / "data" / "cases.json").read_text())
    }
    excerpts = yaml.safe_load((FAILURES / "answer_passages.yaml").read_text())
    return cases, excerpts, failures


@app.cell
def _(mo):
    mo.md("""
    ## 20 cases

    The 20 cases are the first 20 by question id among the 50 semantic questions whose
    expected document was nowhere in the fused top 50 of run 8. Measured over the whole
    corpus of 511,962 documents:

    - **BM25 rank and score**, scoring every document against the question (the top
      document's score is given for scale);
    - **gte rank and cosine similarity** of the document's best chunk, against every chunk
      of every document (exact search, not the approximate index);
    - **the reranker's score** for the question with the document, computed the same way the
      pipeline would (title and first chunk);
    - **the pipeline's top 5**, with each one's rank in BM25 and gte,
      its scores in both and its reranker score;
    - **whether the agent found it** in its extra searches (run 14).
    """)
    return


@app.cell
def _(cases, failures, mo):
    _lines = [
        "| Question | Source | Expected document | BM25 rank (score / top) "
        "| gte rank (similarity) | Rerank: expected vs best pick | Agent found |",
        "|---|---|---|---:|---:|---|:---:|",
    ]
    for _record in failures:
        _expected = _record["expected"][0]
        _best = max(_pick["rerank_score"] for _pick in _record["picked"])
        _beats = " **beats**" if _expected["rerank_score"] > _best else ""
        _title = cases[_record["question_id"]]["documents"][0]["title"][:80]
        _lines.append(
            f"| {_record['question_id']} | {_expected['source']} | {_title} "
            f"| {_expected['bm25_rank']:,} ({_expected['bm25_score']:.1f} / "
            f"{_record['top_bm25_score']:.1f}) "
            f"| {_expected['gte_rank']:,} ({_expected['gte_similarity']:.3f}) "
            f"| {_expected['rerank_score']:.3f} vs {_best:.3f}{_beats} "
            f"| {'yes' if _expected['agent_found'] else 'no'} |"
        )
    mo.md(
        "### Overview\n\n"
        "Ranks are out of 511,962 documents. *Rerank vs picks* compares the reranker's score "
        "for the expected document with its best score among the five documents actually "
        "picked; **beats** marks a higher score for the expected document.\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def _(failures, mo, statistics):
    _expected = [_record["expected"][0] for _record in failures]
    _bm25 = [_e["bm25_rank"] for _e in _expected]
    _gte = [_e["gte_rank"] for _e in _expected]
    _best = [min(_b, _g) for _b, _g in zip(_bm25, _gte, strict=True)]

    def _within(ranks, depth):
        return sum(_rank <= depth for _rank in ranks)

    _rows = [
        f"| Median rank | {statistics.median(_bm25):,.0f} | {statistics.median(_gte):,.0f} "
        f"| {statistics.median(_best):,.0f} |"
    ]
    for _depth in (100, 200, 500, 1000):
        _rows.append(
            f"| Within the top {_depth:,} | {_within(_bm25, _depth)} | {_within(_gte, _depth)} "
            f"| {_within(_best, _depth)} |"
        )
    mo.md(
        "### Ranks\n\n| | BM25 | gte | Better of the two |\n|---|---:|---:|---:|\n"
        + "\n".join(_rows)
    )
    return


@app.cell
def _(failures, mo, statistics):
    _median = statistics.median

    def _quartiles(values):
        return " / ".join(f"{_q:.3f}" for _q in statistics.quantiles(values, n=4))

    _expected = [_record["expected"][0] for _record in failures]
    _first = [_record["picked"][0] for _record in failures]
    _picks = [_pick for _record in failures for _pick in _record["picked"]]
    _top = {_record["question_id"]: _record["top_bm25_score"] for _record in failures}
    _ratio_expected = [
        _e["bm25_score"] / _record["top_bm25_score"]
        for _e, _record in zip(_expected, failures, strict=True)
    ]
    _ratio_first = [
        _p["bm25_score"] / _record["top_bm25_score"]
        for _p, _record in zip(_first, failures, strict=True)
    ]
    _ratio_picks = [
        _pick["bm25_score"] / _record["top_bm25_score"]
        for _record in failures
        for _pick in _record["picked"]
    ]
    _fifth = [min(_p["rerank_score"] for _p in _record["picked"]) for _record in failures]

    def _row(name, key, digits):
        return (
            f"| {name} | {_median([_e[key] for _e in _expected]):.{digits}f} "
            f"| {_median([_p[key] for _p in _first]):.{digits}f} "
            f"| {_median([_p[key] for _p in _picks]):.{digits}f} |"
        )

    mo.md(f"""
    ### Median scores

    Median scores of the expected documents and of the pipeline's picks. *#1 pick* is the document the pipeline put
    first; *all picks* are its five picks for all 20 questions (100 documents). BM25 scores
    are not comparable across questions, so the table also gives each score as a share of
    that question's top BM25 score.

    | Score | Expected document | #1 pick | All picks |
    |---|---:|---:|---:|
    {_row("BM25 score", "bm25_score", 1)}
    | BM25 score / question's top score | {_median(_ratio_expected):.2f} | {_median(_ratio_first):.2f} | {_median(_ratio_picks):.2f} |
    {_row("gte cosine similarity", "gte_similarity", 3)}
    {_row("Reranker score", "rerank_score", 3)}

    The median top BM25 score per question is {_median(list(_top.values())):.1f}. Ranges
    and quartiles:

    - **BM25 / top score:** expected documents run from {min(_ratio_expected):.2f} to
      {max(_ratio_expected):.2f} of the top score.
    - **gte similarity:** expected documents have quartiles
      {_quartiles([_e["gte_similarity"] for _e in _expected])}, the picks
      {_quartiles([_p["gte_similarity"] for _p in _picks])}.
    - **Reranker score:** expected documents have quartiles
      {_quartiles([_e["rerank_score"] for _e in _expected])}, the picks
      {_quartiles([_p["rerank_score"] for _p in _picks])}. The fifth pick has a median of
      {_median(_fifth):.3f}.
    """)
    return


@app.cell
def small_chunks(
    CHUNKERS,
    FAILURES,
    MiniLmEmbedder,
    cases,
    failures,
    json,
    load_by_doc_id,
    mo,
    normalize_documents,
    normalize_text,
    np,
):
    # The MiniLM ingestion settings: 600-character chunks with 100 characters of context
    # from each neighbour and the title on top, embedded with all-MiniLM-L6-v2.
    _chunker = CHUNKERS[MiniLmEmbedder.name]
    _embedder = MiniLmEmbedder()

    def chunk_scores(question, doc_ids):
        """Each document's chunks and their cosine similarity to the question."""
        _documents = load_by_doc_id(doc_ids)
        _query = np.array(_embedder.embed([normalize_text(question)])[0])
        _scores = {}
        for _clean in normalize_documents([_documents[_id] for _id in doc_ids]):
            _texts = _chunker.chunks(_clean.title, _clean.text)
            _similarities = np.array(_embedder.embed(_texts)) @ _query
            _best = int(np.argmax(_similarities))
            _scores[_clean.doc_id] = {
                "chunks": len(_texts),
                "best": float(_similarities[_best]),
                "best_position": _best,
                "best_text": _texts[_best],
                "texts": _texts,
                "similarities": [float(_x) for _x in _similarities],
            }
        return _scores

    # gte's own top 50 per question, before fusion with BM25 (gte_top50.py).
    gte_top50 = json.loads((FAILURES / "data" / "gte_top50.json").read_text())
    small_chunk_scores = {}
    with mo.status.progress_bar(
        total=len(failures), title="Chunking and embedding with MiniLM"
    ) as _bar:
        for _record in failures:
            _qid = _record["question_id"]
            _target = _record["expected"][0]["doc_id"]
            _ids = [_target]
            _ids += [_hit["doc_id"] for _hit in gte_top50[_qid]]
            _ids += [_p["doc_id"] for _p in _record["picked"]]
            _ids = list(dict.fromkeys(_ids))
            small_chunk_scores[_qid] = chunk_scores(cases[_qid]["question"], _ids)
            _bar.update()
    return gte_top50, small_chunk_scores


@app.cell
def _(failures, gte_top50, mo, small_chunk_scores, statistics):
    def _rank(target, others):
        """Rank of the target among itself and the other documents (1 = top)."""
        return 1 + sum(_other > target for _other in others)

    _lines = [
        "| Question | Pool | gte: target score | gte: rank in pool | MiniLM: target score "
        "| MiniLM: rank in pool | Change | MiniLM: pool's #1 score |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    _gte_ranks, _mini_ranks = [], []
    for _record in failures:
        _qid = _record["question_id"]
        _target = _record["expected"][0]
        _others = [_hit for _hit in gte_top50[_qid] if _hit["doc_id"] != _target["doc_id"]]
        _scores = small_chunk_scores[_qid]
        _gte_rank = _rank(_target["gte_similarity"], [_h["gte_similarity"] for _h in _others])
        _mini_target = _scores[_target["doc_id"]]["best"]
        _mini_others = [_scores[_h["doc_id"]]["best"] for _h in _others]
        _mini_rank = _rank(_mini_target, _mini_others)
        _gte_ranks.append(_gte_rank)
        _mini_ranks.append(_mini_rank)
        _change = (
            "higher" if _mini_rank < _gte_rank else "lower" if _mini_rank > _gte_rank else "same"
        )
        _lines.append(
            f"| {_qid} | {len(_others) + 1} | {_target['gte_similarity']:.3f} | {_gte_rank} "
            f"| {_mini_target:.3f} | {_mini_rank} | {_change} "
            f"| {max([_mini_target, *_mini_others]):.3f} |"
        )
    _pairs = list(zip(_gte_ranks, _mini_ranks, strict=True))
    _higher = sum(_m < _g for _g, _m in _pairs)
    _lower = sum(_m > _g for _g, _m in _pairs)

    def _within(ranks, depth):
        return sum(_r <= depth for _r in ranks)

    _summary = "| Target in the pool's | gte | MiniLM |\n|---|---:|---:|\n" + "\n".join(
        f"| top {_d} | {_within(_gte_ranks, _d)} | {_within(_mini_ranks, _d)} |"
        for _d in (1, 5, 10, 20)
    )
    _summary += (
        f"\n| median rank | {statistics.median(_gte_ranks):g} "
        f"| {statistics.median(_mini_ranks):g} |"
    )
    mo.md(
        "## Small chunks: MiniLM vs gte on gte's top 50\n\n"
        "For each question the pool is gte's own top 50 for the question (semantic search "
        "alone, before fusion; `gte_top50.py`) plus the **target**, the expected document. "
        "The pool is ranked twice:\n\n"
        "- **gte**: the gte-modernbert-base similarity of each document's best chunk "
        "(24,000-character chunks, so mostly one vector per document);\n"
        "- **MiniLM**: every document re-chunked with the MiniLM ingestion settings "
        "(600-character chunks, 100 characters of context, the title on top), embedded with "
        "all-MiniLM-L6-v2, scored by its best chunk.\n\n"
        "*Rank in pool* is the target's place (1 = top). Under gte the target ranks last "
        "by construction, since it was outside gte's top 50. In 0185 the target is already in "
        "gte's top 50, so its pool has 50 documents.\n\n"
        f"MiniLM ranks the target higher than gte in {_higher} of {len(failures)} questions, "
        f"lower in {_lower}, the same in {len(failures) - _higher - _lower}.\n\n"
        + _summary
        + "\n\n"
        + "\n".join(_lines)
    )
    return


@app.cell
def answer_chunks(excerpts, failures, mo, re, small_chunk_scores, statistics):
    def _words(text):
        """Lowercase words longer than three characters."""
        return {_w for _w in re.findall(r"[a-z0-9_.\-]+", text.lower()) if len(_w) > 3}

    answer_chunk = {}
    _lines = [
        "| Question | Chunks | Answer chunk # | Answer words in it | Answer chunk score "
        "| Its rank among the document's chunks | Best chunk score | Best chunk # |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    _top = 0
    _gaps = []
    for _record in failures:
        _qid = _record["question_id"]
        _scores = small_chunk_scores[_qid][_record["expected"][0]["doc_id"]]
        _wanted = _words(excerpts[_qid])
        _overlap = [len(_wanted & _words(_t)) / len(_wanted) for _t in _scores["texts"]]
        _index = max(range(len(_overlap)), key=_overlap.__getitem__)
        _similarity = _scores["similarities"][_index]
        _rank = 1 + sum(_x > _similarity for _x in _scores["similarities"])
        _top += _rank == 1
        _gaps.append(_scores["best"] - _similarity)
        answer_chunk[_qid] = {"index": _index, "rank": _rank, "overlap": _overlap[_index]}
        _lines.append(
            f"| {_qid} | {_scores['chunks']} | {_index + 1} | {_overlap[_index]:.0%} "
            f"| {_similarity:.3f} | {_rank} | {_scores['best']:.3f} "
            f"| {_scores['best_position'] + 1} |"
        )
    mo.md(
        "## Does a small chunk hold the answer?\n\n"
        "Each expected document cut into the MiniLM chunks (600 characters, 100 characters of "
        "context). The *answer chunk* is the chunk sharing the most words (over three "
        "letters) with the answer passage in `answer_passages.yaml`; *answer words in it* is "
        "that share. Its MiniLM similarity to the question is ranked among all chunks of the "
        "same document.\n\n"
        f"The answer chunk is the document's best chunk in {_top} of {len(failures)} "
        f"questions. Median gap between the best chunk and the answer chunk: "
        f"{statistics.median(_gaps):.3f}.\n\n" + "\n".join(_lines)
    )
    return


@app.cell
def _(failures, mo):
    case_picker = mo.ui.dropdown(
        options=[_record["question_id"] for _record in failures],
        value=failures[0]["question_id"],
        label="Question",
    )
    return (case_picker,)


@app.cell
def document_texts(
    EnterpriseRagBenchLoader,
    Session,
    engine,
    html,
    ids_of_doc_ids,
    load_documents,
    mo,
    normalize_content,
):
    def load_by_doc_id(doc_ids):
        """Documents keyed by their dataset id."""
        with Session(engine) as _session:
            _ids = ids_of_doc_ids(_session, EnterpriseRagBenchLoader.name, doc_ids)
            return {_d.doc_id: _d for _d in load_documents(_session, sorted(_ids))}

    def document_name(document):
        """A document's type and name (the first line of its title)."""
        return f"[{document.source_type}] {document.title.splitlines()[0]}"

    def document_view(document):
        """A document's type, name and id, then its full text, wrapped and scrollable."""
        return mo.vstack(
            [
                mo.md(
                    f"**Type:** {document.source_type} · **Name:** "
                    f"{document.title.splitlines()[0]} · **Id:** `{document.doc_id}`"
                ),
                mo.Html(
                    "<pre style='white-space: pre-wrap; max-height: 32rem; overflow-y: auto'>"
                    f"{html.escape(normalize_content(document.content))}</pre>"
                ),
            ]
        )

    return document_name, document_view, load_by_doc_id


@app.cell(hide_code=True)
def case_helpers(document_name, document_view, load_by_doc_id, mo):
    def judge_view(result, facts):
        """The judge's verdict, its reasoning and which gold facts the answer covered."""
        if result is None:
            return "*Not judged.*"
        _verdict = "correct" if result.correct else "wrong"
        _covered = "\n".join(
            f"- {'✅' if _hit else '❌'} {_fact}"
            for _fact, _hit in zip(facts, result.facts_covered, strict=False)
        )
        return (
            f"**Judge:** {_verdict}, completeness {result.completeness or 0:.2f}, "
            f"score {result.score or 0:.2f}.\n\n> {result.judge_reasoning}\n\n"
            f"Gold facts covered:\n\n{_covered}"
        )


    def quote(passage):
        """A passage as a markdown block quote."""
        return "\n".join(f"> {_line}" if _line else ">" for _line in passage.splitlines())


    def picks_table(picked):
        """The pipeline's picks: rank in each search, BM25 score, gte similarity, reranker."""
        _rows = "\n".join(
            f"| {_i} | {_p['source']} | {_p['title'][:90]} | "
            + ", ".join(f"{_name} #{_rank}" for _name, _rank in sorted(_p["ranks"].items()))
            + f" | {_p['bm25_score']:.1f} | {_p['gte_similarity']:.3f} | {_p['rerank_score']:.3f} |"
            for _i, _p in enumerate(picked, 1)
        )

        return (
            "| # | Type | Name | Rank in each search | BM25 score | gte similarity | Reranker |\n"
            "|---:|---|---|---|---:|---:|---:|\n" + _rows
        )


    def agent_search_views(searches, expected_ids):
        """The agent's own searches, each with its documents to open; ✅ marks an expected one."""
        _found = load_by_doc_id(
            [_d["doc_id"] for _search in searches for _d in _search["documents"]]
        )
        _views = []
        for _search in searches:
            _tool = "keyword" if _search["tool"] == "keyword_search" else "search"
            if _search["refused"]:
                _views.append(mo.md(f"- {_tool} `{_search['query']}`: refused, search limit reached"))
                continue
            _items = {}
            for _i, _d in enumerate(_search["documents"], 1):
                _mark = " ✅" if _d["doc_id"] in expected_ids else ""
                _doc = _found[_d["doc_id"]]
                _label = f"{_i}. {document_name(_doc)} ({len(_doc.content):,} characters){_mark}"
                _items[_label] = document_view(_doc)
            _views.append(mo.md(f"- {_tool} `{_search['query']}`:"))
            _views.append(mo.accordion(_items, multiple=True))
        if not _views:
            _views.append(mo.md("None: it answered from the first search."))

        return _views


    return agent_search_views, judge_view, picks_table, quote


@app.cell
def _(
    DOCUMENTS_FILE,
    agent_results,
    agent_search_views,
    case_picker,
    cases,
    document_name,
    document_view,
    excerpts,
    failures,
    judge_view,
    load_by_doc_id,
    mo,
    picks_table,
    pipeline_results,
    questions,
    quote,
):
    _qid = case_picker.value
    _case = cases[_qid]
    _record = next(_r for _r in failures if _r["question_id"] == _qid)
    _excerpt = excerpts[_qid]
    _document = _case["documents"][0]
    _expected = _record["expected"][0]
    _source = load_by_doc_id([_document["doc_id"]])[_document["doc_id"]]

    _text = "\n\n".join(
        [
            f"### {_qid}: {document_name(_source)}",
            f"**Query:** {_case['question']}",
            f"**Expected answer:** {_case['gold_answer']}",
            f"**Pipeline answer (run 8):** {_case['pipeline_answer']}",
            judge_view(pipeline_results.get(_qid), questions[_qid].answer_facts),
            f"**Source document:** `{_document['doc_id']}`, {_document['chars']:,} characters, "
            f"row {_document['row_index']:,} of `{DOCUMENTS_FILE}`. Scores: BM25 "
            f"#{_expected['bm25_rank']:,} ({_expected['bm25_score']:.1f} of a top "
            f"{_record['top_bm25_score']:.1f}), gte #{_expected['gte_rank']:,} "
            f"({_expected['gte_similarity']:.3f}), reranker {_expected['rerank_score']:.3f}. "
            "The passage that holds the answer:",
            quote(_excerpt),
        ]
    )
    _rest2 = "\n\n".join(
        [
            "**Pipeline's top 5:**",
            picks_table(_record["picked"]),
            "**Agent's own searches** (run 14), ✅ marks the expected document; open one "
            "to read it:",
        ]
    )
    mo.vstack(
        [
            mo.md(
                "## Cases\n\nThe agent's first search is the question itself, so its results "
                "are the pipeline's top 5; only its own searches are listed."
            ),
            case_picker,
            mo.md(_text),
            mo.accordion(
                {
                    f"Full source document: {document_name(_source)} "
                    f"({_document['chars']:,} characters)": document_view(_source)
                }
            ),
            mo.md(_rest2),
            *agent_search_views(_case["agent_searches"], {_document["doc_id"]}),
            mo.accordion(
                {
                    "Agent answer (run 14) and its judge": mo.md(
                        (_case["agent_answer"] or "")
                        + "\n\n"
                        + judge_view(agent_results.get(_qid), questions[_qid].answer_facts)
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def stage2_setup(CITED_RUN, mo):
    import contextlib
    import importlib
    import sys
    from concurrent.futures import ThreadPoolExecutor

    from dotenv import load_dotenv
    from google import genai

    from train_rag.evaluation.system import NOT_FOUND_ANSWER
    from train_rag.evaluation.usage import UsageMeter

    AMBIGUITY = mo.notebook_dir().parent / "exploration" / "question_ambiguity"
    # Local modules, loaded by name from their folder: a plain `from answers import` makes
    # marimo take them for missing PyPI packages and offer to install them. Reloaded on
    # every run, so edits to them reach the running kernel.
    if str(AMBIGUITY) not in sys.path:
        sys.path.insert(0, str(AMBIGUITY))
    for _name in ["llm_cache", "answers", "citations"]:
        importlib.reload(importlib.import_module(_name))
    _answers = importlib.import_module("answers")  # the per-document check
    _citations = importlib.import_module("citations")
    WORKERS, RatingCache, rate = _answers.WORKERS, _answers.RatingCache, _answers.rate
    CITATION_CACHE, cite, cited_ids = (
        _citations.CITATION_CACHE,
        _citations.cite,
        _citations.cited_ids,
    )
    JsonCache = importlib.import_module("llm_cache").JsonCache

    rate_button = mo.ui.run_button(
        label="Rate the cited documents that aren't cached yet, with Gemini (about $0.15)"
    )
    mo.vstack(
        [
            mo.md(
                "## Stage 2: the document each wrong answer was taken from\n\n"
                f"For every wrong answer of run {CITED_RUN} (the run 8 pipeline, rerun "
                "once it cited its documents): if the expected document wasn't given to "
                "Gemini, the documents the answer cites are each rated on its own: how "
                "fully it answers the question, without the gold answer "
                "(`document_answer_instructions` in "
                "`exploration/question_ambiguity/prompts.yaml`). Ratings are cached there, "
                "so rerunning reads the cache and the button only pays for new calls."
            ),
            rate_button,
        ]
    )
    return (
        CITATION_CACHE,
        JsonCache,
        NOT_FOUND_ANSWER,
        RatingCache,
        ThreadPoolExecutor,
        UsageMeter,
        WORKERS,
        cite,
        cited_ids,
        contextlib,
        genai,
        load_dotenv,
        rate,
        rate_button,
    )


@app.cell
def stage2_llm(
    CITATION_CACHE,
    CITED_RUN,
    JsonCache,
    RatingCache,
    Session,
    ThreadPoolExecutor,
    UsageMeter,
    WORKERS,
    cite,
    cited_ids,
    contextlib,
    engine,
    genai,
    load_by_doc_id,
    load_dotenv,
    mo,
    questions,
    rate,
    rate_button,
    results_of_run,
):
    with Session(engine) as _session:
        _run = {_r.question_id: _r for _r in results_of_run(_session, CITED_RUN)}
    wrong_results = {
        _qid: _result
        for _qid, _result in _run.items()
        if _result.correct is False and _qid in questions
    }
    _documents = load_by_doc_id(
        sorted(
            {
                _d
                for _qid, _result in wrong_results.items()
                for _d in [*_result.retrieved_doc_ids, *questions[_qid].expected_doc_ids]
            }
        )
    )
    # Runs made since the pipeline cites its documents store them; older ones are asked.
    _stored = any(_result.cited_doc_ids for _result in _run.values())
    _to_cite = [
        _qid
        for _qid, _result in wrong_results.items()
        if not set(questions[_qid].expected_doc_ids) & set(_result.retrieved_doc_ids)
    ]
    _ratings = RatingCache()
    _citations = JsonCache(CITATION_CACHE)
    _usage = UsageMeter()
    _client = None
    if rate_button.value:
        load_dotenv(mo.notebook_dir().parent / ".env")
        _client = genai.Client()

    def _cite(qid):
        _result = wrong_results[qid]
        if _stored:
            return _result.cited_doc_ids, ""
        _given = [_documents[_d] for _d in _result.retrieved_doc_ids]
        _citation = cite(
            _client, _usage, _citations, questions[qid].text, _result.answer or "", _given
        )
        if _citation is None:
            return None, ""

        return cited_ids(_citation, _given), _citation.reasoning

    def _rate(pair):
        return rate(_client, _usage, _ratings, questions[pair[0]].text, _documents[pair[1]])

    def _map(function, items):
        if _client is None:
            return [function(_item) for _item in items]
        with ThreadPoolExecutor(WORKERS) as _pool:
            return list(_pool.map(function, items))

    _busy = (
        mo.status.spinner(title="Citing and rating with Gemini")
        if _client is not None
        else contextlib.nullcontext()
    )
    try:
        with _busy:
            citations = dict(zip(_to_cite, _map(_cite, _to_cite), strict=True))
            _pairs = list(
                dict.fromkeys(
                    (_qid, _d) for _qid, (_ids, _) in citations.items() for _d in _ids or []
                )
            )
            _map(_rate, _pairs)
    finally:
        if _client is not None:
            _citations.save()
            _ratings.save()
    doc_ratings = {
        _qid: {
            _d: _rating.model_dump()
            for _d in [*_result.retrieved_doc_ids, *questions[_qid].expected_doc_ids]
            if (_rating := _ratings.rating(questions[_qid].text, _documents[_d])) is not None
        }
        for _qid, _result in wrong_results.items()
    }
    _known = [_ids for _ids, _ in citations.values() if _ids is not None]
    _rated = sum(_d in doc_ratings[_q] for _q, _d in _pairs)
    _note = (
        f"Wrong answers without the expected document: {len(_to_cite)}; "
        f"citing at least one document: {sum(bool(_ids) for _ids in _known)}, "
        f"citing none: {sum(not _ids for _ids in _known)}"
        + (
            f", citations not known yet: {len(_to_cite) - len(_known)}"
            if len(_known) < len(_to_cite)
            else ""
        )
        + f". Cited documents rated: {_rated} of {len(_pairs)}."
    )
    if _usage.calls:
        _note += f" {_usage.report()}"
    mo.md(_note)
    return citations, doc_ratings, wrong_results


@app.cell
def stage2_outcomes(citations, doc_ratings, questions, wrong_results):
    PIE = [
        "expected document retrieved",
        "cited document answers fully",
        "cited document answers partly",
        "cited document doesn't answer",
        "no document cited",
    ]

    def stage2_outcome(qid):
        """Sort one wrong answer by the document it was taken from, or None if not known yet."""
        _result = wrong_results[qid]
        if set(questions[qid].expected_doc_ids) & set(_result.retrieved_doc_ids):
            return PIE[0]
        _ids, _ = citations[qid]
        if _ids is None:
            return None
        if not _ids:
            return PIE[4]
        if any(_d not in doc_ratings[qid] for _d in _ids):
            return None
        _coverage = {doc_ratings[qid][_d]["coverage"] for _d in _ids}
        if "full" in _coverage:
            return PIE[1]
        if "partial" in _coverage:
            return PIE[2]
        return PIE[3]

    stage2 = {_qid: stage2_outcome(_qid) for _qid in wrong_results}
    return PIE, stage2


@app.cell(hide_code=True)
def stage2_details(FAILURES, json):
    # Case texts and ranks for every wrong answer of the cited run, written by
    # case_texts.py and analyze.py into these two files (ranks appear once analyze.py ran).
    stage2_cases = {
        _case["question_id"]: _case
        for _case in json.loads((FAILURES / "data" / "cases_run15.json").read_text())
    }
    _ranks_file = FAILURES / "data" / "failures_run15.json"
    stage2_failures = (
        {_record["question_id"]: _record for _record in json.loads(_ranks_file.read_text())}
        if _ranks_file.exists()
        else {}
    )


    def positions(qid):
        """Where the expected document ranks in BM25 and in gte, over the whole corpus."""
        _record = stage2_failures.get(qid)
        if not _record or not _record["expected"]:
            return "positions not measured"
        _e = _record["expected"][0]

        return f"BM25 #{_e['bm25_rank']:,} · gte #{_e['gte_rank']:,}"


    def rerank_note(qid):
        """The reranker's score for the expected document and for the current #1 pick."""
        _record = stage2_failures.get(qid)
        if not _record or not _record["expected"]:
            return ""
        _expected = _record["expected"][0]["rerank_score"]
        _first = _record["picked"][0]["rerank_score"]
        _verdict = "prefers the expected document" if _expected > _first else "prefers the #1 pick"

        return f" · reranker: expected {_expected:.3f} vs #1 pick {_first:.3f} ({_verdict})"


    return positions, rerank_note, stage2_cases, stage2_failures


@app.cell
def _(
    CITED_RUN,
    NOT_FOUND_ANSWER,
    PIE,
    alt,
    mo,
    question_outcomes,
    stage2,
    wrong_results,
):
    _known = [_o for _o in stage2.values() if _o is not None]
    _values = [
        {
            "outcome": _o,
            "order": _i,
            "answers": _known.count(_o),
            "label": f"{_known.count(_o)} ({_known.count(_o) / len(_known):.0%})",
        }
        for _i, _o in enumerate(PIE)
        if _known.count(_o)
    ]
    _colors = ["#2b6cb0", "#2f855a", "#d69e2e", "#c53030", "#a0aec0"]
    _base = alt.Chart(alt.Data(values=_values)).encode(
        theta=alt.Theta("answers:Q", stack=True),
        order=alt.Order("order:Q"),
        color=alt.Color(
            "outcome:N",
            sort=PIE,
            scale=alt.Scale(domain=PIE, range=_colors),
            legend=alt.Legend(title=None, orient="right", labelLimit=320),
        ),
        tooltip=[alt.Tooltip("outcome:N"), alt.Tooltip("answers:Q"), alt.Tooltip("label:N")],
    )
    _pie = (
        _base.mark_arc(outerRadius=130, stroke="white", strokeWidth=2)
        + _base.mark_text(radius=155, fontSize=13).encode(text="label:N")
    ).properties(width=340, height=340)
    _pending = len(stage2) - len(_known)
    _sources = sorted({question_outcomes[_q][0] for _q in stage2})
    _not_found = {
        _q for _q, _r in wrong_results.items() if (_r.answer or "").strip() == NOT_FOUND_ANSWER
    }

    def _row(outcome, qids):
        _row = {
            "Outcome": outcome,
            "Answers": len(qids),
            "Share": f"{len(qids) / len(stage2):.0%}",
            "Said 'not found'": len(qids & _not_found),
        }
        for _source in _sources:
            _row[_source] = sum(question_outcomes[_q][0] == _source for _q in qids)

        return _row

    _groups = [(_o, {_q for _q, _x in stage2.items() if _x == _o}) for _o in PIE]
    if _pending:
        _groups.append(("not cited or rated yet", {_q for _q, _x in stage2.items() if _x is None}))
    _table = mo.ui.table(
        [_row(_o, _qids) for _o, _qids in _groups] + [_row("all wrong answers", set(stage2))],
        selection=None,
        pagination=False,
        label="Wrong answers by outcome; source columns count the question's source type",
    )
    mo.vstack(
        [
            mo.md(
                f"**Wrong answers of run {CITED_RUN}: {len(stage2)}**"
                + (f" ({_pending} not cited or rated yet)" if _pending else "")
            ),
            mo.hstack([_pie, _table], justify="start", align="center", gap=2),
        ]
    )
    return


@app.cell(hide_code=True)
def stage2_problems(PIE, mo, stage2, stage2_failures):
    _problems = [_q for _q, _o in sorted(stage2.items()) if _o in PIE[3:]]
    _wins = []
    _lines = [
        "| Question | Outcome | Source | Expected document | BM25 rank (score / top) "
        "| gte rank (similarity) | Better of the two | Reranker: expected | Reranker: #1 pick "
        "| Expected minus #1 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _qid in _problems:
        _record = stage2_failures.get(_qid)
        if not _record or not _record["expected"]:
            _lines.append(f"| {_qid} | {stage2[_qid]} | | *not measured* | | | | | | |")
            continue
        _e = _record["expected"][0]
        _first = _record["picked"][0]["rerank_score"]
        _gap = _e["rerank_score"] - _first
        _wins.append(_gap > 0)
        _lines.append(
            f"| {_qid} | {stage2[_qid]} | {_e['source']} | {_e['title'][:70]} "
            f"| {_e['bm25_rank']:,} ({_e['bm25_score']:.1f} / {_record['top_bm25_score']:.1f}) "
            f"| {_e['gte_rank']:,} ({_e['gte_similarity']:.3f}) "
            f"| {min(_e['bm25_rank'], _e['gte_rank']):,} "
            f"| {_e['rerank_score']:.3f} | {_first:.3f} "
            f"| {'**' if _gap > 0 else ''}{_gap:+.3f}{'**' if _gap > 0 else ''} |"
        )
    mo.md(
        f"### Problem cases: {len(_problems)}\n\n"
        "Wrong answers where no document was cited or the cited one doesn't answer. Ranks of "
        "the expected document are out of the whole corpus, measured by `analyze.py` on the "
        "current indexes (after the 2026-11-04 re-embed and BM25 rebuild). The reranker columns "
        "score the expected document and the pipeline's current #1 pick for the question, the "
        "same way the pipeline does (title and first chunk). A **bold** positive gap means the "
        "reranker prefers the expected document: it would have ranked it first had it been in "
        f"the top 50. That holds for **{sum(_wins)} of {len(_wins)}** measured cases.\n\n"
        + "\n".join(_lines)
    )

    return


@app.cell(hide_code=True)
def stage2_small_chunks(FAILURES, json, mo, statistics):
    _file = FAILURES / "data" / "gte_small_chunks.json"
    if not _file.exists():
        _view = mo.md(
            "### Smaller gte chunks\n\n*Not measured yet: run `gte_small_chunks.py` into "
            "`data/gte_small_chunks.json`.*"
        )
    else:
        _rows = json.loads(_file.read_text())
        _sizes = ["stored", "4000", "2000", "1000"]
        _names = {"stored": "24,000 (stored)", "4000": "4,000", "2000": "2,000", "1000": "1,000"}

        def _within(size, depth):
            return sum(_row[size]["rank"] <= depth for _row in _rows)

        _summary = (
            "| Expected document in the pool's | "
            + " | ".join(_names[_s] for _s in _sizes)
            + " |\n|---|"
            + "---:|" * len(_sizes)
            + "\n"
            + "\n".join(
                f"| top {_d} | " + " | ".join(str(_within(_s, _d)) for _s in _sizes) + " |"
                for _d in (1, 5, 10, 20, 50)
            )
            + "\n| median rank | "
            + " | ".join(f"{statistics.median(_r[_s]['rank'] for _r in _rows):g}" for _s in _sizes)
            + " |"
        )
        _lines = [
            "| Question | Pool | "
            + " | ".join(f"{_names[_s]}: rank (similarity)" for _s in _sizes)
            + " | Chunks at 1,000 |",
            "|---|---:|" + "---:|" * len(_sizes) + "---:|",
        ]
        for _row in _rows:
            _lines.append(
                f"| {_row['question_id']} | {_row['pool']} | "
                + " | ".join(
                    f"{_row[_s]['rank']} ({_row[_s]['target']:.3f})" for _s in _sizes
                )
                + f" | {_row['1000']['chunks']} |"
            )
        _view = mo.md(
            "### Smaller gte chunks\n\n"
            "For each problem case the pool is gte's own top 50 on the stored chunks, plus the "
            "expected document and the pipeline's picks (`gte_small_chunks.py`). Every pool "
            "document is re-chunked at 4,000, 2,000 and 1,000 characters (200, 200 and 100 "
            "characters of context, the title on top), embedded with gte-modernbert-base and "
            "scored by its best chunk. *Rank* is the expected document's place in the pool "
            "(1 = top). On the stored chunks it is outside gte's top 50, so it ranks last there "
            "unless gte already found it. The pool is small, so this is optimistic: over the "
            "whole corpus, documents outside the pool can rise too.\n\n"
            + _summary
            + "\n\n"
            + "\n".join(_lines)
        )
    _view

    return


@app.cell
def _(mo, positions, stage2):
    stage2_picker = mo.ui.dropdown(
        options={
            f"{_qid} · {_o or 'not cited or rated yet'} · {positions(_qid)}": _qid
            for _qid, _o in sorted(stage2.items())
        },
        label="Wrong answer",
    )
    return (stage2_picker,)


@app.cell
def _(
    CITED_RUN,
    DOCUMENTS_FILE,
    agent_results,
    agent_search_views,
    citations,
    doc_ratings,
    document_name,
    document_view,
    judge_view,
    load_by_doc_id,
    mo,
    picks_table,
    positions,
    questions,
    quote,
    rerank_note,
    stage2,
    stage2_cases,
    stage2_failures,
    stage2_picker,
    wrong_results,
):
    _unrated = []
    _qid = stage2_picker.value
    if _qid is None:
        _parts = [mo.md("*Pick a wrong answer to see its documents.*")]
    else:
        _result = wrong_results[_qid]
        _question = questions[_qid]
        _ratings = doc_ratings[_qid]
        _cited, _why = citations.get(_qid, (None, ""))
        _ids = list(dict.fromkeys([*_result.retrieved_doc_ids, *_question.expected_doc_ids]))
        _docs = load_by_doc_id(_ids)
        _rows = []
        _unrated = []
        for _d in _ids:
            _r = _ratings.get(_d)
            _marks = [
                f"retrieved #{_result.retrieved_doc_ids.index(_d) + 1}"
                if _d in _result.retrieved_doc_ids
                else "not retrieved"
            ]
            if _cited and _d in _cited:
                _marks.append("**cited**")
            if _d in _question.expected_doc_ids:
                _marks.append("**expected**")
            _head = f"**{document_name(_docs[_d])}** ({' · '.join(_marks)}): "
            if _r is None:
                _unrated.append(_head.removesuffix(": "))
                continue
            _body = f"**{_r['coverage']}**. {_r['reasoning']}"
            if _r["coverage"] != "none":
                _body += f"\n\n> Answer: {_r['answer']}\n>\n> Quote: “{_r['quote']}”"
            _rows.append(_head + _body)
        _citation = f"**Why these are cited:** {_why}\n\n" if _why else ""
        _parts = [
            mo.md(
                f"### {_qid}: {stage2[_qid] or 'not cited or rated yet'}\n\n"
                f"**Expected document position:** {positions(_qid)}{rerank_note(_qid)}\n\n"
                f"**Question:** {_question.text}\n\n"
                f"**Gold answer:** {_question.gold_answer}\n\n"
                f"**Pipeline answer (run {CITED_RUN}):** {_result.answer}\n\n"
                + judge_view(_result, _question.answer_facts)
                + "\n\n"
                + _citation
                + (
                    "**Rated documents**, each rated on its own:\n\n"
                    + "\n\n---\n\n".join(_rows)
                    if _rows
                    else "**Rated documents:** none (no document was cited)."
                )
            ),
            mo.accordion(
                {
                    f"{'Expected' if _d in _question.expected_doc_ids else 'Retrieved'}: "
                    f"{document_name(_docs[_d])} ({len(_docs[_d].content):,} characters)": (
                        document_view(_docs[_d])
                    )
                    for _d in _ids
                },
                multiple=True,
            ),
        ]

        _case = stage2_cases.get(_qid)
        _record = stage2_failures.get(_qid)
        _expected_text = []
        if _case and _case["documents"]:
            _document = _case["documents"][0]
            _scores = ""
            if _record and _record["expected"]:
                _e = _record["expected"][0]
                _scores = (
                    f" Scores: BM25 #{_e['bm25_rank']:,} ({_e['bm25_score']:.1f} of a top "
                    f"{_record['top_bm25_score']:.1f}), gte #{_e['gte_rank']:,} "
                    f"({_e['gte_similarity']:.3f}), reranker {_e['rerank_score']:.3f}."
                )
            _expected_text.append(
                f"**Expected document:** `{_document['doc_id']}`, {_document['chars']:,} "
                f"characters, row {_document['row_index']:,} of `{DOCUMENTS_FILE}`.{_scores} "
                "The window that shares the most words with the gold answer:\n\n"
                + quote(_document["passage"])
            )
        if _record:
            _expected_text.append(
                "**Pipeline's top 5 now** (after the 2026-11-04 re-embed and BM25 rebuild; "
                f"run {CITED_RUN}'s own top 5 is listed above):\n\n" + picks_table(_record["picked"])
            )
        else:
            _expected_text.append(
                "*Ranks and scores not measured yet: run `analyze.py` into "
                "`data/failures_run15.json`.*"
            )
        _parts.append(mo.md("\n\n".join(_expected_text)))
        if _qid in agent_results and _case:
            _parts.append(
                mo.md(
                    "**Agent's own searches** (run 14), ✅ marks the expected document; open one "
                    "to read it:"
                )
            )
            _parts.extend(
                agent_search_views(_case["agent_searches"], set(_question.expected_doc_ids))
            )
            _parts.append(
                mo.accordion(
                    {
                        "Agent answer (run 14) and its judge": mo.md(
                            (_case["agent_answer"] or "")
                            + "\n\n"
                            + judge_view(agent_results.get(_qid), _question.answer_facts)
                        ),
                    }
                )
            )
        else:
            _parts.append(mo.md("*The agent (run 14) didn't try this question.*"))
    if _qid is not None and _unrated:
        _parts.append(
            mo.md(
                "**Not rated:** only the documents an answer cites are rated, so these were "
                "never checked on their own:\n\n" + "\n".join(f"- {_u}" for _u in _unrated)
            )
        )
    mo.vstack([stage2_picker, *_parts])

    return


if __name__ == "__main__":
    app.run()
