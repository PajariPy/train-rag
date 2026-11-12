import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def intro(mo):
    mo.md(r"""
    # 008 · Several questions' top 50, in 2D: PCA, t-SNE or UMAP

    Pick benchmark questions (by default the 10 most different ones) and a projection.
    Each question gets gte's **top 50 by embedding alone**; everything is fitted together, so
    one plot shares one coordinate system. One colour per question; the question is the ★.
    Click a legend entry to hide a question.
    """)
    return


@app.cell
def imports():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.colors
    import plotly.graph_objects as go
    import umap
    from scipy.spatial.distance import pdist, squareform
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sqlalchemy import text

    from train_rag.database import get_engine

    return (
        PCA,
        Path,
        TSNE,
        get_engine,
        go,
        plotly,
        json,
        mo,
        np,
        pdist,
        silhouette_score,
        squareform,
        text,
        umap,
    )


@app.cell
def config(Path, plotly):
    # Documents per question, seed, projection settings, and one colour per question.
    # t-SNE needs perplexity < (n - 1) / 3; UMAP needs n_neighbors < n.
    TOP = 50
    SEED = 7
    PERPLEXITY = 30
    NEIGHBOURS = 15
    DEFAULT_COUNT = 10
    COLOURS = plotly.colors.qualitative.T10
    CACHE = Path(__file__).parent.parent / "exploration/embedding_map/data/top50.json"
    return CACHE, COLOURS, DEFAULT_COUNT, NEIGHBOURS, PERPLEXITY, SEED, TOP


@app.cell
def load(CACHE, DEFAULT_COUNT, get_engine, json, np, text):
    # gte top 50 per question (exact search over the whole corpus), cached by an earlier run
    neighbourhoods = json.loads(CACHE.read_text())
    with get_engine().connect() as _connection:
        question_info = {
            _row.id: _row
            for _row in _connection.execute(
                text("select id, question_type, text from questions where id = any(:ids)"),
                {"ids": list(neighbourhoods)},
            ).all()
        }

    # Default selection: farthest-point sampling over the question vectors
    _ids = sorted(neighbourhoods)
    _vectors = np.array([neighbourhoods[_question_id]["question_vector"] for _question_id in _ids])
    _vectors /= np.linalg.norm(_vectors, axis=1, keepdims=True)
    _similarity = _vectors @ _vectors.T
    _chosen = list(np.unravel_index(np.argmin(_similarity), _similarity.shape))
    while len(_chosen) < min(DEFAULT_COUNT, len(_ids)):
        _chosen.append(int(np.argmin(_similarity[_chosen].max(axis=0))))
    default_ids = [_ids[_index] for _index in _chosen]
    return default_ids, neighbourhoods, question_info


@app.cell
def pickers(default_ids, mo, question_info):
    _options = {
        f"{_question_id} · {_row.question_type} · {_row.text[:80]}": _question_id
        for _question_id, _row in sorted(question_info.items())
    }
    _label_of = {_question_id: _label for _label, _question_id in _options.items()}
    question_picker = mo.ui.multiselect(
        _options, value=[_label_of[_question_id] for _question_id in default_ids], label="Questions"
    )
    method_picker = mo.ui.dropdown(["PCA", "t-SNE", "UMAP"], value="UMAP", label="Projection")
    mo.vstack([question_picker, method_picker])
    return method_picker, question_picker


@app.cell
def points(TOP, mo, neighbourhoods, np, question_info, question_picker):
    # Rows: each question followed by its top 50; vectors scaled to unit length
    selected = list(question_picker.value)
    mo.stop(len(selected) < 2, mo.md("Pick at least two questions."))
    rows = []
    for _group, _question_id in enumerate(selected):
        _neighbourhood = neighbourhoods[_question_id]
        rows.append(
            {
                "group": _group,
                "is_question": True,
                "vector": _neighbourhood["question_vector"],
                "hover": f"{_question_id}: {question_info[_question_id].text}",
            }
        )
        rows += [
            {
                "group": _group,
                "is_question": False,
                "vector": _doc["vector"],
                "hover": f"{_question_id} rank {_rank} · cos {1 - _doc['distance']:.3f} · "
                f"{_doc['source']}: {_doc['title']}",
            }
            for _rank, _doc in enumerate(_neighbourhood["documents"][:TOP], 1)
        ]
    vectors = np.array([_row["vector"] for _row in rows], dtype=np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    groups = np.array([_row["group"] for _row in rows])
    mo.md(f"**{len(rows)} points** from {len(selected)} questions.")
    return groups, rows, selected, vectors


@app.cell
def helpers(np, pdist, squareform):
    def twonn_dimension(points: np.ndarray) -> float:
        """Estimate intrinsic dimension with TwoNN (Facco et al., 2017).

        Args:
            points: The points, one per row; exact duplicates are dropped first.

        Returns:
            The maximum-likelihood dimension from the ratio of 2nd to 1st neighbour distance.
        """
        distinct = np.unique(np.round(points, 6), axis=0)
        distances = np.sort(squareform(pdist(distinct)), axis=1)[:, 1:3]
        ratios = distances[:, 1] / np.maximum(distances[:, 0], 1e-12)

        return float(len(ratios) / np.sum(np.log(ratios)))

    return (twonn_dimension,)


@app.cell
def assumptions(
    NEIGHBOURS,
    PCA,
    PERPLEXITY,
    groups,
    mo,
    np,
    rows,
    silhouette_score,
    twonn_dimension,
    vectors,
):
    # Do the methods' assumptions hold on these points?
    _raw_norms = np.linalg.norm(np.array([_row["vector"] for _row in rows]), axis=1)
    _variance = np.cumsum(PCA().fit(vectors).explained_variance_ratio_)
    _size = len(vectors)
    mo.accordion(
        {
            "Assumption checks": mo.ui.table(
                [
                    {
                        "check": "unit length (Euclidean ranks like cosine)",
                        "value": f"max |norm − 1| = {np.max(np.abs(_raw_norms - 1)):.4f}",
                    },
                    {
                        "check": "PCA: structure in few linear directions",
                        "value": f"2 PCs keep {100 * _variance[1]:.1f}%; "
                        f"{int(np.searchsorted(_variance, 0.9) + 1)} PCs for 90%",
                    },
                    {
                        "check": "UMAP: low-dimensional manifold",
                        "value": f"TwoNN intrinsic dimension = {twonn_dimension(vectors):.1f}",
                    },
                    {
                        "check": "t-SNE: perplexity < (n − 1) / 3",
                        "value": f"{PERPLEXITY} < {(_size - 1) / 3:.1f}",
                    },
                    {"check": "UMAP: n_neighbors < n", "value": f"{NEIGHBOURS} < {_size}"},
                    {
                        "check": "the groups are apart in 768D",
                        "value": f"silhouette (cosine) = "
                        f"{silhouette_score(vectors, groups, metric='cosine'):.3f}",
                    },
                ],
                selection=None,
            )
        }
    )
    return


@app.cell
def project(
    NEIGHBOURS,
    PCA,
    PERPLEXITY,
    SEED,
    TSNE,
    method_picker,
    np,
    umap,
    vectors,
):
    def reduce(method: str, points: np.ndarray) -> np.ndarray:
        """Project points to 2D with the chosen method.

        Args:
            method: "PCA", "t-SNE" or "UMAP".
            points: Unit vectors, one per row.

        Returns:
            The (n, 2) coordinates.
        """
        if method == "PCA":
            return PCA(n_components=2).fit_transform(points)
        if method == "t-SNE":
            return TSNE(perplexity=PERPLEXITY, init="pca", random_state=SEED).fit_transform(points)

        return umap.UMAP(
            n_neighbors=NEIGHBOURS, min_dist=0.1, init="pca", random_state=SEED
        ).fit_transform(points)

    layout = reduce(method_picker.value, vectors)
    return (layout,)


@app.cell
def plot(COLOURS, go, groups, layout, method_picker, rows, selected, silhouette_score):
    _figure = go.Figure()
    for _group, _question_id in enumerate(selected):
        _colour = COLOURS[_group % len(COLOURS)]
        for _is_question in (False, True):
            _indices = [
                _index
                for _index, _row in enumerate(rows)
                if groups[_index] == _group and _row["is_question"] is _is_question
            ]
            _figure.add_trace(
                go.Scatter(
                    x=layout[_indices, 0],
                    y=layout[_indices, 1],
                    mode="markers",
                    name=_question_id,
                    legendgroup=_question_id,
                    showlegend=not _is_question,
                    marker={
                        "color": _colour,
                        "symbol": "star" if _is_question else "circle",
                        "size": 22 if _is_question else 8,
                        "opacity": 1 if _is_question else 0.75,
                        "line": {"width": 1.5 if _is_question else 0.5, "color": "black"},
                    },
                    hovertext=[rows[_index]["hover"] for _index in _indices],
                    hoverinfo="text",
                )
            )
    _figure.update_xaxes(showticklabels=False)
    _figure.update_yaxes(showticklabels=False, scaleanchor="x", scaleratio=1)
    _figure.update_layout(
        title=f"{method_picker.value} · 2D silhouette {silhouette_score(layout, groups):.3f}",
        height=750,
        template="plotly_white",
    )
    _figure
    return


if __name__ == "__main__":
    app.run()
