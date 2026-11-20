"""Agent field search: an LLM agent searches the mined fields in several rounds.

The model is Gemini Flash (since 20 Nov; --model deepseek reruns the earlier setup, see
agent_llm.py). For each of the 200 Slack dev questions, the agent gets one tool, `search_fields`:
a free-text query compared by gte with one profile of every labelled document (subject =
topic + concepts, problem = bug + cause + fix + question, names, or all three), optionally
limited to some business areas. It sees the top 10 of each search, each with its id, mined
topic, key facts (bug, fix, names) and a snippet of the text, so it can judge whether a
result answers the question. It may search again (at most ROUNDS searches), then calls
`finish` with the ids it believes answer the question. Its candidate list is those picks
first, then every search's top 300 fused by reciprocal rank.

Measured as in field_pool.py: where the right thread lands in the agent's list (top 10 /
50 / 100), for the questions today's search reaches and those it misses. Same pool and
caveat: the ~24,400 labelled documents, not the full corpus.

Paid: DeepSeek about $0.40 for the 200 questions; Gemini about 5-6x that. Usage:
    uv run python exploration/slack_extraction/agent_search.py [--limit N] [--model M]
"""

import argparse
import json
import re
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import numpy as np
from agent_llm import DeepSeekChat, GeminiChat
from dotenv import dotenv_values
from field_match import BASELINE, HERE, embed_profiles, load_labels
from google import genai
from google.genai import errors as genai_errors
from label import ENV, peak_now
from schema import MULTI_CLASSES

from train_rag.embedder import FineTunedGteEmbedder
from train_rag.evaluation.person import Person

# v4 (sources and channels filters) with DeepSeek; earlier: v1 agent_search_result.json (no
# snippets), v2 agent_search_snippets_result.json, v3 agent_search_documented_result.json.
RESULT = HERE / "data" / "agent_search_v4_result.json"
RESULT_GEMINI = HERE / "data" / "agent_search_v4_gemini_result.json"
INPUTS = HERE / "data" / "inputs.jsonl"
SNIPPET = 300  # characters of document text shown per result
QUESTIONS = HERE.parents[1] / "exploration" / "training_questions" / "data" / "questions.jsonl"
ROUNDS = 4
SHOWN = 10
KEPT = 300
RRF_K = 60
DEPTHS = (10, 50, 100)
WORKERS = 12
AREAS = list(MULTI_CLASSES["area"][1])
SOURCES = {
    "slack": "chat thread", "gmail": "email", "linear": "ticket (Linear)",
    "jira": "ticket (Jira)", "github": "pull request", "confluence": "wiki page",
    "google_drive": "shared doc or sheet", "fireflies": "meeting transcript",
    "hubspot": "CRM record",
}
PROFILES = {"subject": (0,), "problem": (1,), "names": (2,), "all": (0, 1, 2)}

# What each area holds, for the agent (the extractor saw only the names).
AREA_GUIDE = {
    "billing and pricing": "invoices, charges, credits, refunds, plans and price changes",
    "sales and contracts": "deals, quotes, renewals, commitments, partner bundles, contract terms",
    "customer support": "tickets, customer replies, support macros and wording, escalations",
    "security and compliance": "security reviews, questionnaires, SOC 2, access, audits, PII",
    "legal": "contract clauses, DPAs, terms, legal review",
    "incidents and reliability": "outages, SEVs, errors, postmortems, on-call, failover",
    "performance and latency": "latency, throughput, p95/p99, batching, caching, regressions",
    "deploy and release": "rollouts, canaries, config changes, release plans, rollbacks",
    "infrastructure and capacity": "GPUs, clusters, regions, quotas, capacity planning, hosts",
    "models and evaluation": "model quality, benchmarks, evals, quantization, fine-tuning",
    "API and SDK": "endpoints, rate limits, client libraries, retries, API errors",
    "product and UX": "features, UI, onboarding flows, product decisions",
    "docs and developer relations": "documentation, tutorials, community, webinars, examples",
    "observability and logging": "metrics, dashboards, alerts, logs, tracing",
    "data and storage": "databases, retention, backups, data pipelines, storage",
    "finance and procurement": "budgets, vendors, purchase orders, spend",
    "HR and expenses": "hiring, people topics, expense claims, policies for staff",
    "office and social": "office life, events, social chat",
    "other": "anything that fits none of the above",
}

SYSTEM_TEMPLATE = """You find the document that answers a search question in a company's
knowledge base (Slack threads, tickets, emails, docs). You can't read documents directly: an
extractor mined fields from every document, and you search those fields by meaning (an
embedding comparison, so wording close to the extractor's short phrases works best).

## search_fields(query, profile, areas)
profile = which mined fields your query is compared with:
- subject: topic + general concepts. Best first try. Query like a short topic line:
  "duplicate billing after rate limit rollout".
- problem: bug + root cause + fix + question. For "what broke / why / how was it fixed":
  "429 errors after burst limit change; fixed by idempotency keys".
- names: people, teams, clients, products, components, versions, ids. Only when you know a
  name, e.g. one you saw in earlier results: "acme-inc rate_limit.v2".
- all: the three summed. A broad fallback.

areas = optional filter. Every document has 1-3 areas from this list (most important first);
a document passes if any of its areas is in your list. Documents per area in the index:
{areas}

sources = optional filter by kind of document (documents per source):
{sources}

channels = optional filter by Slack channel (keeps only chat threads from those channels):
{channels}
Use sources and channels when you know them, e.g. from the person: "a chat thread in the
release channel" -> sources ["slack"], channels ["eng-releases"]. Results show each
document's source and channel.
Use 1-3 areas when the question clearly belongs to them (it usually does: a question about
refunds is "billing and pricing"; one about what support told customers is "customer
support"). Results show each document's areas: if the right kind of document shows other
areas, adjust or drop the filter.

## Results
Each search shows the top {shown}: [id], topic, areas, key facts and the first {snippet}
characters of the text. Read the snippets: they show whether a document really answers the
question, and give names and terms for the next search.

## How to search
1. Start with subject (or problem) in your own short words, with areas if clear.
2. Read the results. Reuse names and terms you see; try the likely symptom, fix or component.
3. At most {rounds} searches. Then call finish with the ids of the documents that answer
   the question, best first. Pick only documents whose topic or snippet fits; empty if none.

Example: "How did support resolve the customer hit by throttling and duplicate charges?"
  search_fields("duplicate charges after throttling rollout", "subject",
                ["billing and pricing", "customer support"])
  → result [d812] "acme-inc 429s and billing spikes after rate-policy canary" fits
  search_fields("acme-inc rate-policy canary 429 billing", "names") → confirms d812
  finish(["d812"])"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_fields",
            "description": "Search the documents' mined fields by meaning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "what to look for, a few words"},
                    "profile": {"type": "string", "enum": list(PROFILES)},
                    "areas": {
                        "type": "array",
                        "items": {"type": "string", "enum": AREAS},
                        "description": "optional: keep documents with any of these areas",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(SOURCES)},
                        "description": "optional: keep documents of these kinds",
                    },
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "optional: keep chat threads from these Slack channels",
                    },
                },
                "required": ["query", "profile"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Stop searching and pick the documents that answer the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "ids from the results, best first",
                    }
                },
                "required": ["ids"],
            },
        },
    },
]


class FieldIndex:
    """The labelled pool's profile vectors, searchable by a query text."""

    def __init__(self) -> None:
        """Load labels and cached vectors, and the gte model for queries."""
        labels = load_labels()
        vectors = embed_profiles(labels)
        self.pool = sorted(item for item in labels if not item.startswith("question:"))
        self.matrix = np.stack([vectors[item] for item in self.pool])
        self.areas = [set(labels[item]["labels"]["area"]) for item in self.pool]
        texts: dict[str, str] = {}
        with INPUTS.open() as file:
            for line in file:
                row = json.loads(line)
                texts[row["id"]] = row["text"]
        self.sources = [labels[item]["source"] for item in self.pool]
        self.channels = [
            match.group(1) if (match := re.match(r"Channel: #(\S+)", texts.get(item, ""))) else ""
            for item in self.pool
        ]
        self.cards = [
            f"({SOURCES.get(source, source)}{' #' + channel if channel else ''}) "
            + self.card(labels[item], texts.get(item, ""))
            for item, source, channel in zip(self.pool, self.sources, self.channels, strict=True)
        ]
        counts = Counter(area for found in self.areas for area in found)
        guide = "\n".join(
            f"- {area} ({counts[area]:,}): {AREA_GUIDE[area]}" for area in AREAS
        )
        per_source = Counter(self.sources)
        per_channel = Counter(channel for channel in self.channels if channel)
        self.channel_names = set(per_channel)
        self.system = SYSTEM_TEMPLATE.format(
            areas=guide, shown=SHOWN, snippet=SNIPPET, rounds=ROUNDS,
            sources="\n".join(f"- {s} ({per_source[s]:,}): {SOURCES[s]}" for s in SOURCES),
            channels=", ".join(f"{c} ({n:,})" for c, n in per_channel.most_common()),
        )
        self.embedder = FineTunedGteEmbedder(batch_size=8)
        self.lock = threading.Lock()

    @staticmethod
    def card(label: dict[str, Any], text: str) -> str:
        """One result as the agent sees it: topic, key facts and a text snippet."""
        fields = label["fields"]
        topic = (fields["topic"] or ["?"])[0]
        facts = [
            f"{name}: {'; '.join(fields[name][:2])}"
            for name in ("bug", "fix", "client", "component", "person")
            if fields[name]
        ]
        snippet = " ".join(text.split())[:SNIPPET]

        return (
            f"{topic}\n   areas: {', '.join(label['labels']['area'])}"
            f"{chr(10) + '   ' + ' | '.join(facts) if facts else ''}\n   text: {snippet}…"
        )

    def search(
        self,
        query: str,
        profile: str,
        areas: list[str],
        sources: list[str] | None = None,
        channels: list[str] | None = None,
    ) -> list[int]:
        """Pool positions of the best KEPT documents for one search."""
        with self.lock:  # one GPU model, many threads
            vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)

        vector /= np.linalg.norm(vector) + 1e-9
        scores = sum(self.matrix[:, column] @ vector for column in PROFILES[profile])

        if areas:
            allowed = set(areas)
            mask = np.array([bool(found & allowed) for found in self.areas])
            scores = np.where(mask, scores, -np.inf)

        if sources:
            mask = np.array([source in sources for source in self.sources])
            scores = np.where(mask, scores, -np.inf)

        if channels:
            mask = np.array([channel in channels for channel in self.channels])
            scores = np.where(mask, scores, -np.inf)

        return [int(i) for i in np.argsort(-scores)[:KEPT] if np.isfinite(scores[i])]


ASK_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": "Ask the person who asked the question one short clarifying question.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "one short question"}},
            "required": ["question"],
        },
    },
}

ASK_GUIDE = """

## ask_user(question)
You may also ask the person who asked, at most {max_asks} times; asking doesn't use a search.
Most questions fit several documents (they rarely name a client, team, time or place), so
after your first search, ask unless one result clearly and fully answers the question; a
reply often tells two look-alikes apart. Ask the way a colleague would,
about what a person remembers: what kind of document it was (a chat thread, a ticket, an
email, a doc, a meeting), which team or customer, roughly when, what it was about. Never ask
for names, ids, numbers, exact dates or titles. Their memory can be wrong: check it against
the results."""


Chat = GeminiChat | DeepSeekChat
# What a failed run raises: HTTP and API errors, and malformed model output.
AGENT_ERRORS = (httpx.HTTPError, genai_errors.APIError, KeyError, ValueError)


def make_chat(model: str) -> Callable[[str, str], Chat]:
    """A conversation starter for the agent's model, with its API key from .env.

    Args:
        model: "gemini" (default for new runs) or "deepseek" (the runs before 20 Nov).

    Returns:
        A function that starts a conversation from the system and user messages.
    """
    env = dotenv_values(ENV)
    if model == "deepseek":
        if peak_now():
            raise SystemExit("DeepSeek peak hours: prices are doubled. Rerun later.")
        http = httpx.Client(
            headers={"Authorization": f"Bearer {env['DEEPSEEK_API_KEY']}"}, timeout=120
        )
        return lambda system, user: DeepSeekChat(http, system, user)
    client = genai.Client(api_key=env["GEMINI_API_KEY"])
    return lambda system, user: GeminiChat(client, system, user)


def run_agent(
    chat: Callable[[str, str], Chat],
    index: FieldIndex,
    question: str,
    person: Person | None = None,
    max_asks: int = 0,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any], list[int], list[dict[str, str]]]:
    """Let the agent search for one question.

    Args:
        chat: Starts a conversation with the model, from the system and user messages
            (`gemini_chat` or `deepseek_chat`).
        index: The field index.
        question: The search question.
        person: Answers ask_user (anything with `reply(ask) -> str`), or None: no asking.
        max_asks: How many questions the agent may ask the person.

    Returns:
        The candidate list (picks first, then fused positions), the searches made, the
        token usage with its cost in dollars, the picks and the questions asked with their
        replies.
    """
    asking = person is not None and max_asks > 0
    system = index.system + (ASK_GUIDE.format(max_asks=max_asks) if asking else "")
    conversation = chat(system, f"Search question: {question}")
    asks: list[dict[str, str]] = []
    searches: list[dict[str, Any]] = []
    picked: list[int] = []
    results: list[list[int]] = []

    for _turn in range(ROUNDS + max_asks + 2):
        tools = TOOLS if len(searches) < ROUNDS else TOOLS[1:]  # out of searches: must pick
        if asking and len(asks) < max_asks:
            tools = [*tools, ASK_TOOL]
        calls = conversation.step(tools)
        finish = [call for call in calls if call.name == "finish"]

        if finish:
            ids = finish[0].args.get("ids") or []
            picked = [int(i[1:]) for i in map(str, ids) if i[1:].isdigit()]
            picked = [i for i in picked if 0 <= i < len(index.pool)]
            break

        if not calls:
            break

        for call in calls:
            if call.name == "ask_user":
                if person is None or not asking or len(asks) >= max_asks:
                    reply = "no questions left"
                else:
                    try:
                        ask = str(call.args["question"])
                        reply = person.reply(ask)
                        asks.append({"ask": ask, "reply": reply})
                    except KeyError as error:
                        reply = f"bad call: {error}"
                conversation.reply(call, reply)
                continue

            if len(searches) >= ROUNDS:  # several calls in one turn: still at most ROUNDS
                conversation.reply(call, "no searches left: call finish")
                continue

            try:
                arguments = call.args
                query = str(arguments["query"])
                profile = arguments.get("profile", "all")
                profile = profile if profile in PROFILES else "all"
                areas = [a for a in arguments.get("areas") or [] if a in AREAS]
                sources = [x for x in arguments.get("sources") or [] if x in SOURCES]
                channels = [
                    x.lstrip("#") for x in arguments.get("channels") or []
                    if x.lstrip("#") in index.channel_names
                ]
                found = index.search(query, profile, areas, sources, channels)
                searches.append({"query": query, "profile": profile, "areas": areas,
                                 "sources": sources, "channels": channels})
                results.append(found)
                shown = "\n".join(
                    f"[d{i}] {index.cards[i]}" for i in found[:SHOWN]
                )
            except (ValueError, KeyError) as error:
                shown = f"bad call: {error}"

            conversation.reply(call, shown)

    usage: dict[str, Any] = {**conversation.usage, "cost": conversation.cost()}
    fused: dict[int, float] = {}
    for found in results:
        for rank, position in enumerate(found):
            fused[position] = fused.get(position, 0.0) + 1 / (RRF_K + rank + 1)

    rest = [p for p in sorted(fused, key=lambda p: -fused[p]) if p not in picked]

    return list(dict.fromkeys(picked)) + rest, searches, usage, picked, asks


def main() -> None:
    """Run the agent on the dev questions and report where the right thread lands."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", choices=("gemini", "deepseek"), default="gemini")
    args = parser.parse_args()

    baseline: dict[str, dict[str, Any]] = json.loads(BASELINE.read_text())
    rows = [json.loads(line) for line in QUESTIONS.open()]
    texts = {row["doc_id"]: row["question"] for row in rows if row["split"] == "dev_slack"}
    index = FieldIndex()
    position = {item: i for i, item in enumerate(index.pool)}
    questions = [q for q in sorted(baseline) if q in position and q in texts]
    questions = questions[: args.limit] if args.limit else questions

    chat = make_chat(args.model)
    result = RESULT if args.model == "deepseek" else RESULT_GEMINI

    def one(doc_id: str) -> dict[str, Any]:
        try:
            fused, searches, usage, picked, _asks = run_agent(chat, index, texts[doc_id])
        except AGENT_ERRORS as error:
            return {"doc_id": doc_id, "error": str(error)}

        target = position[doc_id]
        rank = fused.index(target) + 1 if target in fused else None

        return {
            "doc_id": doc_id, "rank": rank, "searches": searches, "usage": usage,
            "picked": [index.pool[i] for i in picked], "picked_right": target in picked,
            "group": "reached" if doc_id in baseline[doc_id]["scores"] else "missed",
        }

    with ThreadPoolExecutor(WORKERS) as pool:
        runs = list(pool.map(one, questions))

    done = [run for run in runs if "error" not in run]
    cost = sum(run["usage"]["cost"] for run in done)

    report: dict[str, Any] = {
        "model": args.model, "questions": len(questions), "errors": len(runs) - len(done),
        "cost": round(cost, 3),
        "searches per question": round(np.mean([len(r["searches"]) for r in done]), 2),
        "area filter used": round(
            np.mean([any(s["areas"] for s in r["searches"]) for r in done]), 2
        ),
    }
    for group in ("reached", "missed"):
        ranks = [r["rank"] for r in done if r["group"] == group]
        report[group] = {
            "n": len(ranks),
            "right thread picked": round(
                100 * np.mean([r["picked_right"] for r in done if r["group"] == group]), 1
            ),
            "picks per question": round(
                float(np.mean([len(r["picked"]) for r in done if r["group"] == group])), 1
            ),
        } | {
            f"top{d}": round(100 * sum(1 for r in ranks if r and r <= d) / max(len(ranks), 1), 1)
            for d in DEPTHS
        }

    result.write_text(json.dumps({"report": report, "runs": runs}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
