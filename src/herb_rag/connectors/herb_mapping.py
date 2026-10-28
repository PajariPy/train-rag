"""Convert raw HERB records into normalized models."""

from herb_rag.models import Artifact, Question, QuestionType, SourceType
from herb_rag.models.herb_raw import (
    RawAnswerableQuestion,
    RawDocument,
    RawMeetingChat,
    RawMeetingTranscript,
    RawProduct,
    RawPullRequest,
    RawSlackMessage,
    RawUrl,
)


def slack_to_artifact(product: str, raw: RawSlackMessage) -> Artifact:
    """Convert a Slack message.

    Args:
        product: Product the message belongs to.
        raw: Raw Slack message.

    Returns:
        The normalized artifact.
    """
    sender = raw.message.user

    return Artifact(
        id=raw.id,
        product=product,
        source_type=SourceType.SLACK,
        content=sender.text,
        created_at=sender.timestamp,
        author_id=sender.user_id,
        metadata={"channel": raw.channel.name, "channel_id": raw.channel.channel_id},
    )


def document_to_artifact(product: str, raw: RawDocument) -> Artifact:
    """Convert an internal document.

    Args:
        product: Product the document belongs to.
        raw: Raw document.

    Returns:
        The normalized artifact.
    """
    metadata = {"document_type": raw.type, "link": raw.document_link}
    if raw.feedback is not None:
        metadata["feedback"] = raw.feedback

    return Artifact(
        id=raw.id,
        product=product,
        source_type=SourceType.DOCUMENT,
        content=raw.content,
        created_at=raw.date,
        author_id=raw.author,
        metadata=metadata,
    )


def transcript_to_artifact(product: str, raw: RawMeetingTranscript) -> Artifact:
    """Convert a meeting transcript.

    Args:
        product: Product the meeting belongs to.
        raw: Raw meeting transcript.

    Returns:
        The normalized artifact.
    """
    return Artifact(
        id=raw.id,
        product=product,
        source_type=SourceType.MEETING_TRANSCRIPT,
        content=raw.transcript,
        created_at=raw.date,
        metadata={
            "meeting_type": raw.document_type,
            "participants": ",".join(raw.participants),
        },
    )


def chat_to_artifact(product: str, raw: RawMeetingChat) -> Artifact:
    """Convert a meeting chat log.

    Args:
        product: Product the meeting belongs to.
        raw: Raw meeting chat.

    Returns:
        The normalized artifact.
    """
    return Artifact(
        id=raw.id,
        product=product,
        source_type=SourceType.MEETING_CHAT,
        content=raw.text,
    )


def url_to_artifact(product: str, raw: RawUrl) -> Artifact:
    """Convert a shared URL.

    Args:
        product: Product the link was shared in.
        raw: Raw URL record.

    Returns:
        The normalized artifact.
    """
    return Artifact(
        id=raw.id,
        product=product,
        source_type=SourceType.URL,
        content=raw.description,
        metadata={"link": raw.link},
    )


def pull_request_to_artifact(product: str, raw: RawPullRequest) -> Artifact:
    """Convert a GitHub pull request.

    Args:
        product: Product the pull request belongs to.
        raw: Raw pull request.

    Returns:
        The normalized artifact.
    """
    reviewers = ",".join(review.user.login for review in raw.reviews)

    return Artifact(
        id=raw.id,
        product=product,
        source_type=SourceType.PULL_REQUEST,
        content=f"{raw.title}\n\n{raw.summary}",
        created_at=raw.created_at,
        author_id=raw.user.login,
        metadata={
            "link": raw.link,
            "number": raw.number,
            "state": raw.state,
            "merged": raw.merged,
            "reviewers": reviewers,
        },
    )


def product_to_artifacts(product: str, raw: RawProduct) -> list[Artifact]:
    """Convert every artifact of a product file.

    Args:
        product: Product name.
        raw: Raw product file.

    Returns:
        All artifacts of the product, grouped by source type.
    """
    artifacts = [slack_to_artifact(product, item) for item in raw.slack]
    artifacts += [document_to_artifact(product, item) for item in raw.documents]
    artifacts += [transcript_to_artifact(product, item) for item in raw.meeting_transcripts]
    artifacts += [chat_to_artifact(product, item) for item in raw.meeting_chats]
    artifacts += [url_to_artifact(product, item) for item in raw.urls]
    artifacts += [pull_request_to_artifact(product, item) for item in raw.prs]

    return artifacts


def answerable_to_question(product: str, index: int, raw: RawAnswerableQuestion) -> Question:
    """Convert an answerable question.

    Args:
        product: Product the question is about.
        index: Position of the question in the product file, used for its id.
        raw: Raw answerable question.

    Returns:
        The normalized question; a single string answer becomes a one-item list.
    """
    answers = [raw.ground_truth] if isinstance(raw.ground_truth, str) else raw.ground_truth

    return Question(
        id=f"{product}/answerable/{index}",
        product=product,
        text=raw.question,
        answerable=True,
        question_type=QuestionType(raw.type),
        answers=answers,
        citation_ids=raw.citations,
    )


def product_to_questions(product: str, raw: RawProduct) -> list[Question]:
    """Convert the answerable and unanswerable questions of a product file.

    Args:
        product: Product name.
        raw: Raw product file.

    Returns:
        Answerable questions followed by unanswerable ones.
    """
    questions = [
        answerable_to_question(product, index, item)
        for index, item in enumerate(raw.answerable_questions)
    ]
    questions += [
        Question(id=f"{product}/unanswerable/{index}", product=product, text=text, answerable=False)
        for index, text in enumerate(raw.unanswerable_questions)
    ]

    return questions
