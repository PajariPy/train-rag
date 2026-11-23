r"""Label documents with DeepSeek: the reference for scoring, and the SLM's training data.

Each item of data/inputs.jsonl (build_inputs.py) goes to DeepSeek (`deepseek-flash`) as a
strict tool call on the beta endpoint, so the server enforces the answer's JSON schema (built
from schema.py); the answer is then validated again against pydantic models built from the same
schema. Name values must be copied word for word; one that can't be found in the text is kept
but flagged. Relations that break the schema are dropped before saving. Prompt frozen on 17
November at fc8ae8878988 (the hash printed at start; it covers the prompt and the schema).
Each label is appended to data/labels.jsonl as it arrives, keyed by item id and that hash, so
a crash loses nothing, a rerun never pays twice and a change starts fresh labels. A run that
crosses into peak hours leaves the rest unlabelled; rerun later to finish.

DeepSeek charges double in its peak hours (01:00-04:00 and 06:00-10:00 UTC on weekdays); the
script refuses to start then unless --peak is given. About $0.0009 a thread off-peak.

Usage:
    uv run python exploration/slack_extraction/label.py [--use test train] [--sample 20]
        [--limit N] [--peak]
"""

import argparse
import hashlib
import json
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from schema import CLASSES, FIELDS, MULTI_CLASSES, RELATIONS

from train_rag.database import get_engine
from train_rag.progress.recorder import RunProgress

HERE = Path(__file__).parent
INPUTS = HERE / "data" / "inputs.jsonl"  # from build_inputs.py
LABELS = HERE / "data" / "labels.jsonl"  # one label per line, appended as it arrives
ENV = HERE.parents[1] / ".env"
# The beta endpoint enforces a tool's JSON schema ("strict" tool calls); JSON mode only
# guarantees valid JSON.
URL = "https://api.deepseek.com/beta/chat/completions"
TOOL = "save_fields"
MODEL = "deepseek-flash"
WORKERS = 32
RETRIES = 3
MAX_TOKENS = 8_000  # 4,000 cut off long documents (568 of 25,633 failed on 17 Nov)
# Dollars per million tokens: (cache-miss input, cache-hit input, output), off-peak and peak.
PRICES = {"off-peak": (0.15, 0.003, 0.60), "peak": (0.30, 0.006, 1.20)}
PEAK_HOURS = {1, 2, 3, 6, 7, 8, 9}  # UTC, weekdays
SAMPLE_SEED = 17

INSTRUCTIONS = """You extract structured fields from one company document: a Slack thread, an
email, a wiki page, a ticket or a meeting note. In a Slack thread each line is one message:
"speaker (role): text".

Fields (form: description):
{fields}

Labels (thread_kind and status: exactly one value; area: one to three values):
{classes}

Relations between values you listed (head -> tail):
{relations}

Rules:
- name fields: copy the value exactly as written, character for character; never fix
  spelling or expand it. One name per value, no lists inside a value.
- phrase fields: at most 12 words; use the document's own words where you can, shortened.
- sentence fields: one short sentence, at most 25 words, that makes sense on its own
  (say what was decided or asked, not "they agreed to do it").
- List each distinct value once. Never put the same value in two fields: pick the one
  that fits best. Examples: "small-int8" is a model, not also a product and a version;
  "Smart Failover" is a feature, not also a product; "2026-05-10" is a date_time, not
  also an identifier; a named sheet or dashboard is a document, not also a component.
  Product vs feature vs component: product = what customers buy (a plan, tier, SKU);
  feature = a capability users see inside a product; component = an internal service,
  module, endpoint or tool.
- Speakers count as persons; a speaker's role in brackets can give a team or a client.
  Bots (names ending in -bot, or automated posts) and roles (QA, oncall) are not persons
  or teams; leave them out.
- date_time: dates, times, deadlines and windows only ("4/17", "EOD Friday", "Q2"); not
  durations or intervals ("45s", "48h", "7 days", "2-4 weeks", "6-month", "~40m"):
  leave those out.
- action_item: the task itself, without the owner's name (the owner goes in "owns").
- "topic": exactly one value. "doc_date": at most one value.
- "concept": up to 6 general terms a reader would search with, even if the document never
  names a product (for a question: the concepts it asks about).
- A search question is labelled the same way: its fields are what it names or asks about;
  status is "not stated".
- Leave a field empty when the document has no value of it. Don't guess and don't
  fill fields with generic words ("the issue", "the team", "today").
- Relations: only when the document states or clearly shows the link. Head and tail must
  be values you listed in the fields named for that relation, written the same way, and
  never the same value.

Call save_fields once with everything you found."""


class Strict(BaseModel):
    """Rejects keys the schema doesn't name, so a made-up field fails validation."""

    model_config = ConfigDict(extra="forbid")


# One model per part of the answer, built from schema.py: every field must be present (a list,
# empty if nothing found), every label one of its allowed values, every relation a known name.
Fields = create_model(  # type: ignore[call-overload]
    "Fields", __base__=Strict, **{name: (list[str], ...) for name in FIELDS}
)
Labels = create_model(  # type: ignore[call-overload]
    "Labels",
    __base__=Strict,
    **{name: (Literal[values], ...) for name, (_, values) in CLASSES.items()},
    **{name: (list[Literal[values]], ...) for name, (_, values) in MULTI_CLASSES.items()},
)


class Relation(Strict):
    """One relation between two extracted values."""

    relation: Literal[tuple(RELATIONS)]  # type: ignore[valid-type]
    head: str
    tail: str


class ThreadLabel(Strict):
    """DeepSeek's answer for one document."""

    fields: Fields  # type: ignore[valid-type]
    labels: Labels  # type: ignore[valid-type]
    relations: list[Relation]


def answer_schema() -> dict[str, Any]:
    """The answer's JSON schema in the form DeepSeek's strict mode accepts.

    Every object lists all its properties as required and forbids others; no $defs.

    Returns:
        The schema for the save_fields tool's arguments.
    """

    def strict_object(properties: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    strings = {"type": "array", "items": {"type": "string"}}

    fields = strict_object({name: strings for name in FIELDS})
    labels = strict_object(
        {name: {"type": "string", "enum": list(values)} for name, (_, values) in CLASSES.items()}
        | {
            name: {"type": "array", "items": {"type": "string", "enum": list(values)}}
            for name, (_, values) in MULTI_CLASSES.items()
        }
    )
    relation = strict_object(
        {
            "relation": {"type": "string", "enum": list(RELATIONS)},
            "head": {"type": "string"},
            "tail": {"type": "string"},
        }
    )

    return strict_object(
        {"fields": fields, "labels": labels, "relations": {"type": "array", "items": relation}}
    )


def instructions() -> str:
    """The system prompt, built from the schema."""
    fields = "\n".join(f"- {name} ({form}): {text}" for name, (form, text) in FIELDS.items())

    classes = "\n".join(
        f"- {name}: {text}; one of: {', '.join(values)}" for name, (text, values) in CLASSES.items()
    )
    classes += "\n" + "\n".join(
        f"- {name}: {text}; from: {', '.join(values)}"
        for name, (text, values) in MULTI_CLASSES.items()
    )

    relations = "\n".join(
        f"- {name}: {description} ({' or '.join(heads)} -> {' or '.join(tails)})"
        for name, (description, heads, tails) in RELATIONS.items()
    )

    return INSTRUCTIONS.format(fields=fields, classes=classes, relations=relations)


def parse_first(arguments: str) -> dict[str, Any]:
    """Parse the first JSON value in the tool-call arguments, ignoring anything after it.

    DeepSeek sometimes closes the object with one brace too many ("...}]}}}").

    Args:
        arguments: The raw arguments string.

    Returns:
        The parsed object.
    """
    value, _end = json.JSONDecoder().raw_decode(arguments.strip())

    if not isinstance(value, dict):
        raise ValueError("the arguments are not a JSON object")

    return value


def fix_labels(answer: dict[str, Any]) -> dict[str, Any]:
    """Repair the label fields before validation.

    Strict mode doesn't enforce the enums: DeepSeek sometimes writes a value from another
    list ("HR or people" as an area, "sales and contracts" as thread_kind). Without this,
    pydantic rejects the whole answer (568 of 25,633 items on 17 Nov).
    - `thread_kind` / `status` off the list -> "other" / "not stated".
    - `area`: near-misses are mapped by their first word ("HR or people" -> "HR and
      expenses"); anything else is dropped; "other" if nothing is left.

    Args:
        answer: The parsed tool-call arguments.

    Returns:
        The same answer, with only allowed labels.
    """
    labels = answer.get("labels")

    if not isinstance(labels, dict):
        return answer

    for name, fallback in (("thread_kind", "other"), ("status", "not stated")):
        if name in labels and labels[name] not in CLASSES[name][1]:
            labels[name] = fallback

    allowed = MULTI_CLASSES["area"][1]

    if not isinstance(labels.get("area"), list):
        return answer

    areas: list[str] = []

    for value in labels["area"]:
        if value not in allowed:
            first = str(value).split()[0].lower() if str(value).split() else ""
            matches = [area for area in allowed if area.split()[0].lower() == first]
            value = matches[0] if len(matches) == 1 else None

        if value and value not in areas:
            areas.append(value)

    labels["area"] = areas or ["other"]

    return answer


def peak_now() -> bool:
    """Whether DeepSeek's peak prices apply right now."""
    now = datetime.now(UTC)

    return now.weekday() < 5 and now.hour in PEAK_HOURS


def missing_values(label: ThreadLabel, text: str) -> list[str]:
    """Name values that don't appear in the text (ignoring case): copied wrong or made up."""
    lowered = text.lower()

    values = [
        value
        for name, values in label.fields.model_dump().items()
        if FIELDS.get(name, ("", ""))[0] == "name"
        for value in values
    ]

    return sorted({value for value in values if value.lower() not in lowered})


def clean(label: ThreadLabel) -> tuple[ThreadLabel, int]:
    """Drop relations that break the schema, so training never teaches them.

    The schema already makes the relation name a known one; this checks what a schema can't:
    head and tail differ, and each end is a value listed under a field the relation allows.

    Args:
        label: DeepSeek's answer.

    Returns:
        The answer with only valid relations, and how many were dropped.
    """
    kept = []

    for relation in label.relations:

        if relation.head == relation.tail:
            continue

        _, heads, tails = RELATIONS[relation.relation]
        head_ok = any(relation.head in getattr(label.fields, field) for field in heads)
        tail_ok = any(relation.tail in getattr(label.fields, field) for field in tails)

        if head_ok and tail_ok:
            kept.append(relation)

    return label.model_copy(update={"relations": kept}), len(label.relations) - len(kept)


class Labeller:
    """Calls DeepSeek, appends the labels and counts the tokens; safe across worker threads."""

    def __init__(self, key: str, allow_peak: bool) -> None:
        """Load the keys already labelled and set up the client.

        Args:
            key: The DeepSeek API key.
            allow_peak: Keep calling during DeepSeek's peak hours (double price).
        """
        self.system = instructions()
        self.schema = answer_schema()
        self.version = hashlib.sha256(
            f"{MODEL}{self.system}{json.dumps(self.schema, sort_keys=True)}".encode()
        ).hexdigest()[:12]

        self.client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=180)
        self.lock = threading.Lock()
        self.allow_peak = allow_peak

        self.done: set[str] = set()

        if LABELS.exists():
            with LABELS.open() as lines:  # not splitlines(): texts hold U+2028 separators
                self.done = {json.loads(line)["key"] for line in lines}

        self.output = LABELS.open("a")
        self.tokens = {"miss": 0, "hit": 0, "output": 0}
        self.counts = {"labelled": 0, "skipped": 0, "failed": 0, "paused": 0, "flagged": 0}
        # Priced at the start: a run that crosses into peak hours pauses, it doesn't pay double.
        self.prices = PRICES["peak" if peak_now() else "off-peak"]
        self.reasons: Counter[str] = Counter()  # why attempts failed, for the final report

    def key(self, item_id: str) -> str:
        """The label's key: the item and the prompt version."""
        return f"{item_id}:{self.version}"

    def label(self, item: dict[str, str]) -> None:
        """Label one item unless a label for it and this prompt exists.

        Args:
            item: A line of inputs.jsonl (id, use, source, text).
        """
        if self.key(item["id"]) in self.done:
            with self.lock:
                self.counts["skipped"] += 1

            return

        if peak_now() and not self.allow_peak:  # the run crossed into peak hours: leave it
            with self.lock:
                self.counts["paused"] += 1

            return

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": item["text"]},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": TOOL,
                        "description": "Save the fields, labels and relations found.",
                        "strict": True,
                        "parameters": self.schema,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": TOOL}},
            # On by default: it spends the whole token budget reasoning and returns no answer.
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
        }

        for _attempt in range(RETRIES):
            try:
                response = self.client.post(URL, json=payload)
                response.raise_for_status()

                data = response.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                self.count_tokens(usage)  # failed attempts are paid for too

                if choice.get("finish_reason") == "length":  # cut off: unusable as training
                    self.note_failure("cut off at max_tokens")
                    continue

                arguments = choice["message"]["tool_calls"][0]["function"]["arguments"]
                label = ThreadLabel.model_validate(fix_labels(parse_first(arguments)))
            except (httpx.HTTPError, ValidationError, KeyError, ValueError) as error:
                self.note_failure(type(error).__name__)
                continue

            cleaned, dropped = clean(label)
            missing = missing_values(label, item["text"])
            row = {
                "key": self.key(item["id"]),
                "id": item["id"],
                "use": item["use"],
                "source": item["source"],
                **cleaned.model_dump(),
                "dropped_relations": dropped,
                "not_in_text": missing,
                "output_tokens": usage.get("completion_tokens", 0),
            }

            with self.lock:
                self.output.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.output.flush()

                self.done.add(row["key"])
                self.counts["labelled"] += 1
                self.counts["flagged"] += bool(missing)

            return

        with self.lock:
            self.counts["failed"] += 1

    def count_tokens(self, usage: dict[str, int]) -> None:
        """Add one answer's token usage to the totals.

        Args:
            usage: The `usage` block of a DeepSeek answer.
        """
        with self.lock:
            self.tokens["hit"] += usage.get("prompt_cache_hit_tokens", 0)
            self.tokens["miss"] += usage.get(
                "prompt_cache_miss_tokens", usage.get("prompt_tokens", 0)
            )
            self.tokens["output"] += usage.get("completion_tokens", 0)

    def note_failure(self, reason: str) -> None:
        """Count one failed attempt by its reason.

        Args:
            reason: A short reason, e.g. "cut off at max_tokens" or an error class name.
        """
        with self.lock:
            self.reasons[reason] += 1

    def cost(self) -> float:
        """Dollars for the tokens counted, at the price that applied when the run started."""
        miss, hit, output = self.prices

        return (
            self.tokens["miss"] * miss + self.tokens["hit"] * hit + self.tokens["output"] * output
        ) / 1e6


def main() -> None:
    """Label the items of inputs.jsonl, optionally only some uses or the first --limit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="label only the first N items picked")
    parser.add_argument("--use", nargs="*", help="only these uses (test, train, extraction_test)")
    parser.add_argument("--sample", type=int, help="a seeded random N items instead")
    parser.add_argument("--peak", action="store_true", help="run even at peak prices")
    parser.add_argument("--batch", help="only the items added in this batch (inputs.jsonl)")
    parser.add_argument("--workflow", help="workflow key, to report as one of its steps")
    parser.add_argument("--step", help="the step's name in that workflow")
    args = parser.parse_args()

    if peak_now() and not args.peak:
        raise SystemExit("DeepSeek peak hours: prices are doubled. Rerun later or pass --peak.")

    with INPUTS.open() as lines:  # not splitlines(): texts hold U+2028 separators
        items = [json.loads(line) for line in lines]

    if args.use:
        items = [item for item in items if item["use"] in args.use]

    if args.batch:
        items = [item for item in items if item.get("batch") == args.batch]

    if args.sample:
        items = random.Random(SAMPLE_SEED).sample(items, args.sample)

    items = items[: args.limit]
    labeller = Labeller(dotenv_values(ENV)["DEEPSEEK_API_KEY"] or "", args.peak)

    print(f"{len(items):,} items, prompt {labeller.version}", flush=True)

    progress = RunProgress(
        get_engine(), "label fields with DeepSeek", len(items),
        workflow=args.workflow, step=args.step,
    )
    step = max(1, len(items) // 200)

    with progress, ThreadPoolExecutor(WORKERS) as pool:
        for done, _ in enumerate(pool.map(labeller.label, items), 1):
            if done % step == 0 or done == len(items):
                progress.advance(
                    "labelling", step if done % step == 0 else done % step,
                    note=f"DeepSeek: ${labeller.cost():.2f}, {labeller.counts}",
                )

    labeller.output.close()

    print(
        f"done: {labeller.counts}, {labeller.tokens} tokens, about ${labeller.cost():.4f}; "
        f"failed attempts: {dict(labeller.reasons)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
