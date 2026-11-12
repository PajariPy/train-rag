import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def intro(mo):
    mo.md(r"""
    # 007 · The embedding space, in 2D

    Every document has one gte-modernbert vector (768 dimensions; a document is one chunk).
    UMAP squeezes them into 2D so we can *see* the semantic search failures:

    - **background**: a random sample of the corpus, coloured by source;
    - **candidates**: for each of the 46 semantic failures, gte's top 50 plus the right document;
    - **questions**: the 46 failure questions, embedded the way search embeds them, and
      projected into the same map.

    UMAP keeps neighbourhoods, not distances: two points close in 2D are usually close in 768D,
    but far-apart clusters say little about how far apart they really are.
    """)
    return


@app.cell
def imports():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import umap
    from sqlalchemy import text

    from train_rag.database import get_engine
    from train_rag.embedder import GteModernBertEmbedder
    from train_rag.processing.normalize import normalize_text
    from train_rag.tables import EMBEDDING_TABLES

    return (
        EMBEDDING_TABLES,
        GteModernBertEmbedder,
        Path,
        alt,
        get_engine,
        json,
        mo,
        normalize_text,
        np,
        text,
        umap,
    )


@app.cell
def config(Path):
    # Documents sampled for the background, and the seed that picks them
    SAMPLE = 30_000
    SEED = 7
    ROOT = Path(__file__).parent.parent
    POOLS = ROOT / "exploration/retrieval_failures/data/pools_run15.json"
    RERANK_SCORES = ROOT / "exploration/fact_extraction/data/rerank_pilot_scores.json"
    CACHE = ROOT / "exploration/clarification/data/umap_2d.npz"
    return CACHE, POOLS, RERANK_SCORES, SAMPLE, SEED


@app.cell
def load_failures(POOLS, RERANK_SCORES, json, mo):
    def rank_by_score(scores: dict[str, float]) -> dict[str, int]:
        """Rank documents by score.

        Args:
            scores: Reranker score of each document id.

        Returns:
            Each document's rank, 1 = highest score.
        """
        ordered = sorted(scores, key=lambda doc_id: -scores[doc_id])

        return {doc_id: rank for rank, doc_id in enumerate(ordered, 1)}


    # The 46 failures: question, right document, candidates (gte top 50 + right document)
    _pools = json.loads(POOLS.read_text())
    failures = _pools["pools"]
    titles = {
        _doc_id: _text.split("\n", 1)[0][:100] for _doc_id, _text in _pools["documents"].items()
    }

    # Reranker rank (full text) of every candidate, per question
    _scores = json.loads(RERANK_SCORES.read_text())
    rerank_rank = {
        _question_id: rank_by_score(_question_scores["rerank_text"])
        for _question_id, _question_scores in _scores.items()
        if _question_id != "_timing"
    }
    candidate_ids = sorted({_doc_id for _pool in failures.values() for _doc_id in _pool["members"]})

    mo.md(f"{len(failures)} failures, {len(candidate_ids):,} candidate documents")
    return candidate_ids, failures, rerank_rank, titles


@app.cell
def load_vectors(
    EMBEDDING_TABLES,
    SAMPLE,
    SEED,
    candidate_ids,
    get_engine,
    mo,
    np,
    text,
):
    def parse_vector(vector: str) -> np.ndarray:
        """Read pgvector's text form as an array.

        Args:
            vector: The vector as pgvector prints it, e.g. "[0.1,0.2,...]".

        Returns:
            The vector as float32.
        """
        return np.fromstring(vector[1:-1], sep=",", dtype=np.float32)


    with get_engine().connect() as _connection:
        # A random background sample of the corpus, plus every candidate document
        _source_of = dict(
            _connection.execute(text("select doc_id, source_type from documents")).all()
        )
        _random = np.random.default_rng(SEED)
        _sample = set(_random.choice(sorted(_source_of), SAMPLE, replace=False))
        background_ids = sorted(_sample - set(candidate_ids))

        # One gte vector per document: every document is one chunk
        _rows = []
        for _table in EMBEDDING_TABLES[768].values():
            _rows += _connection.execute(
                text(
                    f"select doc_id, embedding::text from {_table.__tablename__} "
                    "where model = 'gte-modernbert-base' and field = 'chunk' and position = 0 "
                    "and doc_id = any(:ids)"
                ),
                {"ids": background_ids + candidate_ids},
            ).all()

    doc_ids = [_doc_id for _doc_id, _vector in _rows]
    doc_vectors = np.stack([parse_vector(_vector) for _doc_id, _vector in _rows])
    doc_source = [str(_source_of[_doc_id]) for _doc_id in doc_ids]

    mo.md(f"Loaded **{len(doc_ids):,}** document vectors ({doc_vectors.shape[1]} dimensions)")
    return doc_ids, doc_source, doc_vectors


@app.cell
def questions(GteModernBertEmbedder, failures, mo, normalize_text, np):
    # The questions, embedded exactly as semantic search embeds them
    question_ids = list(failures)
    question_vectors = np.array(
        GteModernBertEmbedder().embed([normalize_text(failures[_q]["question"]) for _q in question_ids]),
        dtype=np.float32,
    )
    mo.md(f"Embedded **{len(question_ids)}** questions")
    return question_ids, question_vectors


@app.cell
def fit_umap(
    CACHE,
    SEED,
    doc_ids,
    doc_vectors,
    mo,
    np,
    question_vectors,
    umap,
):
    # Fit UMAP on the documents (cosine, like search), then place the questions in the same map.
    # The result is cached on disk; delete CACHE to refit.
    if CACHE.exists() and list(np.load(CACHE, allow_pickle=True)["doc_ids"]) == doc_ids:
        _cached = np.load(CACHE, allow_pickle=True)
        doc_xy, question_xy = _cached["doc_xy"], _cached["question_xy"]
        _how = "loaded from cache"
    else:
        with mo.status.spinner(title=f"UMAP on {len(doc_ids):,} vectors…"):
            _reducer = umap.UMAP(
                n_neighbors=30, min_dist=0.1, metric="cosine", random_state=SEED, low_memory=True
            )
            doc_xy = _reducer.fit_transform(doc_vectors)
            question_xy = _reducer.transform(question_vectors)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, doc_ids=np.array(doc_ids, dtype=object), doc_xy=doc_xy, question_xy=question_xy)
        _how = "fitted and cached"
    mo.md(f"UMAP: {len(doc_xy):,} documents and {len(question_xy)} questions in 2D ({_how})")
    return doc_xy, question_xy


@app.cell
def points(
    candidate_ids,
    doc_ids,
    doc_source,
    doc_xy,
    failures,
    question_ids,
    question_xy,
    titles,
):
    # One row per point, for the charts
    _right_ids = {_pool["expected"] for _pool in failures.values()}
    _candidate_set = set(candidate_ids)


    def role_of(doc_id: str) -> str:
        """Tell what a document is in this map.

        Args:
            doc_id: The document's dataset id.

        Returns:
            "right document", "candidate" or "background".
        """
        if doc_id in _right_ids:
            return "right document"

        return "candidate" if doc_id in _candidate_set else "background"


    doc_points = [
        {
            "doc_id": _doc_id,
            "x": float(_xy[0]),
            "y": float(_xy[1]),
            "source": _source,
            "role": role_of(_doc_id),
            "title": titles.get(_doc_id, ""),
        }
        for _doc_id, _xy, _source in zip(doc_ids, doc_xy, doc_source, strict=True)
    ]
    question_points = [
        {
            "question_id": _question_id,
            "x": float(_xy[0]),
            "y": float(_xy[1]),
            "question": failures[_question_id]["question"][:120],
        }
        for _question_id, _xy in zip(question_ids, question_xy, strict=True)
    ]
    return doc_points, question_points


@app.cell
def overview(alt, doc_points, mo, question_points):
    _background = (
        alt.Chart(alt.Data(values=[_point for _point in doc_points if _point["role"] == "background"]))
        .mark_circle(size=6, opacity=0.35)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color("source:N", title="Source"),
        )
    )
    _candidates = (
        alt.Chart(alt.Data(values=[_point for _point in doc_points if _point["role"] != "background"]))
        .mark_point(size=40, filled=True, stroke="black", strokeWidth=0.4)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("source:N"),
            shape=alt.Shape(
                "role:N",
                scale=alt.Scale(domain=["candidate", "right document"], range=["circle", "diamond"]),
                title="Failure documents",
            ),
            size=alt.Size(
                "role:N",
                scale=alt.Scale(domain=["candidate", "right document"], range=[25, 120]),
                legend=None,
            ),
            tooltip=["title:N", "source:N", "role:N"],
        )
    )
    _questions = (
        alt.Chart(alt.Data(values=question_points))
        .mark_point(shape="cross", size=90, color="black", filled=True)
        .encode(x="x:Q", y="y:Q", tooltip=["question_id:N", "question:N"])
    )

    mo.vstack(
        [
            mo.md(
                "### The whole map\n"
                "Small dots: a random 30k-document sample, coloured by source. "
                "Outlined points: the 46 failures' candidates (◆ = the right document). "
                "✚ = the failure questions."
            ),
            (_background + _candidates + _questions)
            .properties(width=900, height=650)
            .interactive(),
        ]
    )
    return


@app.cell
def picker(failures, mo, question_ids):
    question_picker = mo.ui.dropdown(
        options={
            f"{_question_id} · {failures[_question_id]['question'][:90]}": _question_id
            for _question_id in question_ids
        },
        value=f"qst_0178 · {failures['qst_0178']['question'][:90]}",
        label="Failure",
    )
    question_picker
    return (question_picker,)


@app.cell
def zoom(
    alt,
    doc_points,
    failures,
    mo,
    question_picker,
    question_points,
    rerank_rank,
    titles,
):
    _question_id = question_picker.value
    _pool = failures[_question_id]
    _gte_rank = {_doc_id: _rank for _rank, _doc_id in enumerate(_pool["gte_top50"], 1)}
    _members = set(_pool["members"])
    _rows = [
        {
            **_point,
            "reranker rank": rerank_rank[_question_id].get(_point["doc_id"]),
            "gte rank": _gte_rank.get(_point["doc_id"]),
            "is right": _point["doc_id"] == _pool["expected"],
        }
        for _point in doc_points
        if _point["doc_id"] in _members
    ]
    _question_row = next(_point for _point in question_points if _point["question_id"] == _question_id)

    _documents = (
        alt.Chart(alt.Data(values=_rows))
        .mark_point(filled=True, stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False)),
            color=alt.Color("source:N", title="Source"),
            shape=alt.Shape(
                "is right:N",
                scale=alt.Scale(domain=[False, True], range=["circle", "diamond"]),
                title="Right document",
            ),
            size=alt.Size("is right:N", scale=alt.Scale(domain=[False, True], range=[60, 300]), legend=None),
            tooltip=["title:N", "source:N", "reranker rank:Q", "gte rank:Q"],
        )
    )
    _labels = (
        alt.Chart(alt.Data(values=_rows))
        .mark_text(dx=9, dy=-7, fontSize=10)
        .encode(x="x:Q", y="y:Q", text="reranker rank:Q")
    )
    _question = (
        alt.Chart(alt.Data(values=[_question_row]))
        .mark_point(shape="cross", size=200, color="black", filled=True)
        .encode(x="x:Q", y="y:Q", tooltip=["question:N"])
    )
    _right_rank = rerank_rank[_question_id].get(_pool["expected"])

    mo.vstack(
        [
            mo.md(
                f"### {_question_id}: the question and its {len(_rows)} candidates\n"
                f"**Q:** {_pool['question']}\n\n"
                f"Right document (◆): *{titles.get(_pool['expected'], '')}*, "
                f"reranker rank **{_right_rank}**. Numbers next to points = reranker rank on full text."
            ),
            (_documents + _labels + _question).properties(width=900, height=600).interactive(),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
