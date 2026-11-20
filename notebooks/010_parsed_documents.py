import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def imports():
    import html
    import random
    from collections import defaultdict

    import marimo as mo

    from train_rag.data_loaders import EnterpriseRagBenchLoader
    from train_rag.models import Artifact, SourceType
    from train_rag.models.parsed.base import Comment, ParsedDocument
    from train_rag.parsers import default_registry
    from train_rag.processing.normalize import normalize_content, normalize_text

    return (
        Artifact,
        Comment,
        EnterpriseRagBenchLoader,
        ParsedDocument,
        SourceType,
        default_registry,
        defaultdict,
        html,
        mo,
        normalize_content,
        normalize_text,
        random,
    )


@app.cell
def _(mo):
    mo.md("""
    # Parsed documents, easy to read

    The same samples as notebook 001, shown the way our parsers read them: Slack as speaker
    and message, Gmail as one block per email, Jira, Linear and GitHub as fields plus
    comments, Fireflies with its summary, date and attendees. Every document also keeps its
    full cleaned text (`normalize_content`, what gets chunked and embedded) under
    **Full text**, so nothing the parser leaves out is lost. Google Drive and Confluence
    have no parser, so they show the cleaned text only.

    Without the database, the LLM-inferred meeting dates, attendees and Slack splits aren't
    used: the parsers fall back to what the text itself says.
    """)
    return


@app.cell
def load_corpus(EnterpriseRagBenchLoader, defaultdict, default_registry, mo):
    data_dir = mo.notebook_dir().parent / "data" / "enterprise_rag_bench"
    corpus = EnterpriseRagBenchLoader(data_dir).load()
    registry = default_registry()

    artifacts_by_source = defaultdict(list)
    for _artifact in corpus.artifacts:
        artifacts_by_source[_artifact.source_type].append(_artifact)
    return artifacts_by_source, registry


@app.cell
def controls(mo):
    samples_per_type = mo.ui.slider(1, 5, value=2, label="Samples per type", show_value=True)
    reshuffle = mo.ui.button(value=0, on_click=lambda count: count + 1, label="Reshuffle")
    mo.hstack([samples_per_type, reshuffle], justify="start", gap=2)
    return reshuffle, samples_per_type


@app.cell
def renderers(Comment, ParsedDocument, html, mo, normalize_text):
    def text_block(text: str) -> str:
        """Escaped text in a wrapping block that keeps line breaks.

        Args:
            text: Plain text.

        Returns:
            The HTML block.
        """
        return (
            '<div style="white-space: pre-wrap; word-break: break-word; margin: 0 0 12px 0">'
            f"{html.escape(normalize_text(text))}</div>"
        )

    def field_rows(fields: dict[str, object]) -> str:
        """Short fields as a two-column list, skipping empty ones.

        Args:
            fields: Label -> value; lists are joined with commas.

        Returns:
            The HTML rows.
        """
        rows = []
        for label, value in fields.items():
            if value in (None, "", []):
                continue
            shown = ", ".join(str(item) for item in value) if isinstance(value, list) else value
            rows.append(f"<div><b>{html.escape(label)}:</b> {html.escape(str(shown))}</div>")

        return "".join(rows)

    def section(label: str, text: str | None) -> str:
        """A labelled block of longer text, or nothing when it's empty.

        Args:
            label: Heading of the block.
            text: Its text.

        Returns:
            The HTML block.
        """
        if not text:
            return ""
        heading = f"<h4 style='margin: 12px 0 4px'>{html.escape(label)}</h4>" if label else ""

        return heading + text_block(text)

    def comments(items: list[Comment] | None) -> str:
        """Ticket or review comments, one block each with author, role and time.

        Args:
            items: The comments.

        Returns:
            The HTML blocks.
        """
        blocks = []
        for comment in items or []:
            role = f" ({comment.role})" if comment.role else ""
            posted = f" · {comment.posted_at:%Y-%m-%d %H:%M}" if comment.posted_at else ""
            blocks.append(
                f"<div><b>{html.escape(comment.author + role)}</b>{posted}</div>"
                + text_block(comment.text)
            )

        if not blocks:
            return ""

        return "<h4 style='margin: 12px 0 4px'>Comments</h4>" + "".join(blocks)

    def parsed_body(parsed: ParsedDocument) -> str:
        """The parsed fields of one document, laid out by its type.

        Args:
            parsed: The parser's result.

        Returns:
            The HTML of its fields and texts.
        """
        kind = type(parsed).__name__
        data = parsed.model_dump()
        if kind == "SlackThread":
            header = field_rows({"Channel": data["channel"], "Participants": data["participants"]})
            messages = []
            for message in data["messages"]:
                speaker = message["speaker"] or "?"
                if message["role"]:
                    speaker += f" ({message['role']})"
                messages.append(
                    f"<div><b>{html.escape(speaker)}</b></div>" + text_block(message["text"])
                )
            return header + "<h4 style='margin: 12px 0 4px'>Messages</h4>" + "".join(messages)
        if kind == "EmailThread":
            emails = []
            for number, message in enumerate(data["messages"], 1):
                emails.append(
                    f"<h4 style='margin: 12px 0 4px'>Email {number}</h4>"
                    + field_rows(
                        {
                            "From": message["sender"],
                            "To": message["recipients"],
                            "Cc": message["cc"],
                            "Date": message["date"],
                            "Subject": message["subject"],
                            "Attachments": message["attachments"],
                        }
                    )
                    + text_block(message["body"])
                )
            return field_rows({"Participants": data["participants"]}) + "".join(emails)
        if kind in ("JiraTicket", "LinearTicket"):
            header = field_rows(
                {
                    "Participants": data["participants"],
                    "First activity": data["first_activity_at"],
                    "Environment": data.get("environment"),
                }
            )
            return (
                header
                + section("Summary", data.get("issue_summary"))
                + section("Description", data["description"])
                + section("Impact", data.get("impact"))
                + comments(getattr(parsed, "comments", None))
            )
        if kind == "GithubPullRequest":
            return (
                field_rows({"Participants": data["participants"]})
                + section("Description", data["description"])
                + comments(getattr(parsed, "review_comments", None))
            )
        if kind == "FirefliesMeeting":
            attendees = [
                person["name"] for person in data["attendees"] or data["attendees_inferred"]
            ]
            return field_rows(
                {
                    "Held on": data["held_on"],
                    "Start": data["start_time"],
                    "Duration": data["duration"],
                    "Attendees": attendees,
                    "Topics": data["topics"],
                }
            ) + section("Summary", data["summary"])
        if kind == "HubspotRecord":
            return field_rows({"Company": data["company"]})

        return ""

    return parsed_body, section


@app.cell
def render_document(Artifact, mo, normalize_content, parsed_body, registry, section):
    # Parsers that keep only header fields, not the body, so the text is shown open.
    HEADER_ONLY = {"HubspotRecord", "FirefliesMeeting"}

    def render(artifact: Artifact) -> object:
        """One document: its parsed view, then its full cleaned text.

        The text is folded when the parsed view already holds the body (Slack, Gmail,
        Jira, Linear, GitHub) and shown open when the parser keeps only header fields
        (HubSpot, Fireflies) or there is no parser.

        Args:
            artifact: The raw document.

        Returns:
            The marimo element.
        """
        parsed = registry.parse(artifact)
        kind = type(parsed).__name__ if parsed is not None else "no parser"
        body = parsed_body(parsed) if parsed is not None else ""
        full_text = normalize_content(artifact.content)
        header = mo.md(f"`{artifact.id}` · parsed as **{kind}** · {len(full_text):,} chars")
        if kind in HEADER_ONLY or not body:
            return mo.vstack([header, mo.Html(body + section("Text", full_text))])
        folded = mo.accordion({"Full text": mo.Html(section("", full_text))})

        return mo.vstack([header, mo.Html(body), folded])

    return (render,)


@app.cell
def document_samples(
    SourceType,
    artifacts_by_source,
    mo,
    random,
    render,
    reshuffle,
    samples_per_type,
):
    document_rng = random.Random(reshuffle.value)
    document_tabs = {}
    for _source_type in SourceType:
        _of_type = artifacts_by_source[_source_type]
        _picked = document_rng.sample(_of_type, min(samples_per_type.value, len(_of_type)))
        document_tabs[f"{_source_type.value} ({len(_of_type):,})"] = mo.accordion(
            {
                _artifact.title.split("\n")[0][:120] or _artifact.id: render(_artifact)
                for _artifact in _picked
            },
            multiple=True,
        )

    mo.vstack([mo.md("## Document samples"), mo.ui.tabs(document_tabs)])
    return


if __name__ == "__main__":
    app.run()
