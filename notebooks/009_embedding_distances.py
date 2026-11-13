import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def intro(mo):
    mo.md(r"""
    # 009 · How far apart are the gte vectors? The raw cosine values

    2D maps (notebook 008) bend distances, so here we read the cosine similarities themselves,
    the numbers a fine-tuned embedder would have to move:

    1. **baseline**: how similar two *random* documents are (how much of the space gte uses);
    2. **two questions**: question → its own top 50, → the other's top 50, and documents among themselves;
    3. **the margin**: for every cached question, how far the right document is from rank 1 and rank 50.
    """)
    return


@app.cell
def imports():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from sqlalchemy import text

    from train_rag.database import get_engine
    from train_rag.tables import EMBEDDING_TABLES

    return EMBEDDING_TABLES, Path, get_engine, go, json, mo, np, text


@app.cell
def config(Path):
    TOP = 50
    RANDOM_PER_SOURCE = 150
    CACHE = Path(__file__).parent.parent / "exploration/embedding_map/data/top50.json"
    return CACHE, RANDOM_PER_SOURCE, TOP


@app.cell
def load(CACHE, EMBEDDING_TABLES, RANDOM_PER_SOURCE, get_engine, json, mo, np, text):
    def unit(rows: list[list[float]]) -> np.ndarray:
        """Stack vectors and scale them to unit length.

        Args:
            rows: The vectors.

        Returns:
            An (n, d) array of unit vectors.
        """
        matrix = np.array(rows, dtype=np.float64)

        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    # gte top 50 per question, cached by notebook 008; plus a random sample per source
    neighbourhoods = json.loads(CACHE.read_text())
    with get_engine().connect() as _connection:
        question_info = {
            _row.id: _row
            for _row in _connection.execute(
                text(
                    "select id, question_type, text, expected_doc_ids from questions where id = any(:ids)"
                ),
                {"ids": list(neighbourhoods)},
            ).all()
        }
        _random_rows = []
        for _table in EMBEDDING_TABLES[768].values():
            _random_rows += [
                json.loads(_vector)
                for (_vector,) in _connection.execute(
                    text(
                        f"select embedding::text from {_table.__tablename__} "
                        "where model = 'gte-modernbert-base' and field = 'chunk' "
                        "order by random() limit :limit"
                    ),
                    {"limit": RANDOM_PER_SOURCE},
                ).all()
            ]
    random_vectors = unit(_random_rows)
    mo.md(
        f"{len(neighbourhoods)} questions; {len(random_vectors):,} random documents for the baseline"
    )
    return neighbourhoods, question_info, random_vectors, unit


@app.cell
def baseline(go, mo, np, random_vectors):
    # Anisotropy: if random pairs already sit at cosine ~0.6, the useful range is narrow
    _similarity = random_vectors @ random_vectors.T
    random_pairs = _similarity[np.triu_indices(len(random_vectors), 1)]
    _mean_vector = np.linalg.norm(random_vectors.mean(axis=0))
    _figure = go.Figure(go.Histogram(x=random_pairs, nbinsx=80, marker={"color": "#2a78d6"}))
    _figure.update_layout(
        title="Cosine similarity of random document pairs",
        xaxis_title="cosine similarity",
        yaxis_title="pairs",
        height=320,
        template="plotly_white",
    )
    mo.vstack(
        [
            mo.md(
                f"### 1. Baseline\nRandom pairs: mean **{random_pairs.mean():.3f}**, "
                f"5th–95th percentile {np.percentile(random_pairs, 5):.3f}–"
                f"{np.percentile(random_pairs, 95):.3f}. Length of the mean unit vector: "
                f"**{_mean_vector:.3f}** (0 = directions spread evenly, 1 = all the same direction)."
            ),
            _figure,
        ]
    )
    return (random_pairs,)


@app.cell
def pickers(mo, neighbourhoods, np, question_info, unit):
    # Default pair: the two questions farthest apart
    _ids = sorted(neighbourhoods)
    _vectors = unit([neighbourhoods[_question_id]["question_vector"] for _question_id in _ids])
    _first, _second = np.unravel_index(np.argmin(_vectors @ _vectors.T), (len(_ids), len(_ids)))
    _options = {
        f"{_question_id} · {question_info[_question_id].question_type} · "
        f"{question_info[_question_id].text[:80]}": _question_id
        for _question_id in _ids
    }
    _label_of = {_question_id: _label for _label, _question_id in _options.items()}
    first_picker = mo.ui.dropdown(_options, value=_label_of[_ids[_first]], label="🟢 Green")
    second_picker = mo.ui.dropdown(_options, value=_label_of[_ids[_second]], label="🟠 Orange")
    mo.vstack([mo.md("### 2. Two questions"), first_picker, second_picker])
    return first_picker, second_picker


@app.cell
def pair_values(TOP, first_picker, go, mo, neighbourhoods, np, random_pairs, second_picker, unit):
    def upper(matrix: np.ndarray) -> np.ndarray:
        """Take the values above the diagonal of a square similarity matrix.

        Args:
            matrix: A square similarity matrix.

        Returns:
            The pairwise values, each pair once.
        """
        return matrix[np.triu_indices(len(matrix), 1)]

    _first, _second = first_picker.value, second_picker.value
    _question = {
        _id: unit([neighbourhoods[_id]["question_vector"]])[0] for _id in (_first, _second)
    }
    _docs = {
        _id: unit([_doc["vector"] for _doc in neighbourhoods[_id]["documents"][:TOP]])
        for _id in (_first, _second)
    }
    _groups = {
        "random document pairs": random_pairs,
        "🟢 question → own top 50": _docs[_first] @ _question[_first],
        "🟢 question → 🟠 top 50": _docs[_second] @ _question[_first],
        "🟠 question → own top 50": _docs[_second] @ _question[_second],
        "🟠 question → 🟢 top 50": _docs[_first] @ _question[_second],
        "🟢 docs ↔ 🟢 docs": upper(_docs[_first] @ _docs[_first].T),
        "🟠 docs ↔ 🟠 docs": upper(_docs[_second] @ _docs[_second].T),
        "🟢 docs ↔ 🟠 docs": (_docs[_first] @ _docs[_second].T).ravel(),
    }
    _figure = go.Figure(
        [
            go.Box(x=_values, name=_name, boxpoints="outliers", marker={"color": "#2a78d6"})
            for _name, _values in reversed(_groups.items())
        ]
    )
    _figure.update_layout(
        xaxis_title="cosine similarity", height=480, showlegend=False, template="plotly_white"
    )

    # The full similarity matrix: question 1, its docs, question 2, its docs
    _all = np.vstack([_question[_first], _docs[_first], _question[_second], _docs[_second]])
    _labels = (
        ["🟢 Q"]
        + [f"🟢 {_rank}" for _rank in range(1, TOP + 1)]
        + ["🟠 Q"]
        + [f"🟠 {_rank}" for _rank in range(1, TOP + 1)]
    )
    _heatmap = go.Figure(
        go.Heatmap(
            z=_all @ _all.T, x=_labels, y=_labels, colorscale="Blues", colorbar={"title": "cos"}
        )
    )
    _heatmap.update_layout(
        height=700, width=760, yaxis={"autorange": "reversed"}, template="plotly_white"
    )

    mo.vstack(
        [
            mo.ui.table(
                [
                    {
                        "pairs": _name,
                        "mean": round(float(np.mean(_values)), 3),
                        "min": round(float(np.min(_values)), 3),
                        "max": round(float(np.max(_values)), 3),
                    }
                    for _name, _values in _groups.items()
                ],
                selection=None,
            ),
            _figure,
            mo.md(
                "Similarity matrix, ordered: 🟢 question, its top 50 by rank, 🟠 question, its top 50."
            ),
            _heatmap,
        ]
    )
    return


@app.cell
def margins(TOP, mo, neighbourhoods, question_info, unit):
    # What fine-tuning has to change: the right document's similarity against the top 50
    _rows = []
    for _question_id, _neighbourhood in sorted(neighbourhoods.items()):
        _expected = set(question_info[_question_id].expected_doc_ids)
        if not _expected:
            continue
        _question = unit([_neighbourhood["question_vector"]])[0]
        _similarity = unit([_doc["vector"] for _doc in _neighbourhood["documents"]]) @ _question
        _right = [
            (_rank, float(_value))
            for _rank, (_doc, _value) in enumerate(
                zip(_neighbourhood["documents"], _similarity, strict=True), 1
            )
            if _doc["doc_id"] in _expected
        ]
        _best_rank, _best_value = max(_right, key=lambda _item: _item[1])
        _rows.append(
            {
                "question": _question_id,
                "type": question_info[_question_id].question_type,
                "cos rank 1": round(float(_similarity[0]), 3),
                "cos rank 50": round(float(_similarity[TOP - 1]), 3),
                "cos right doc": round(_best_value, 3),
                "right doc rank": _best_rank if _best_rank <= TOP else "> 50",
                "gap to rank 1": round(float(_similarity[0]) - _best_value, 3),
                "rank 1 − rank 50": round(float(_similarity[0] - _similarity[TOP - 1]), 3),
            }
        )
    mo.vstack(
        [
            mo.md(
                "### 3. The margin\n"
                "**gap to rank 1** = how much the right document must gain (or competitors lose) to come "
                "first; compare it with **rank 1 − rank 50**, the whole width of the top 50."
            ),
            mo.ui.table(_rows, selection=None, page_size=25),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
