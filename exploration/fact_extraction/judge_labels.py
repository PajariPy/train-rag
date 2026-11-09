r"""Check fact labels for made-up content and for what RAG needs. A check; changes no label.

Three checks, cheapest first:

- specifics (free, every label): each number, id and URL in a fact must appear in the
  document. Catches the most harmful hallucination, an invented specific, but also flags
  reformatted ones ("1.2k" for "1,200"), so it is a screen, not a verdict.
- hallucination (Gemini, a sample stratified by source): every fact gets supported /
  unsupported / contradicted against the document, with the passage as evidence.
- rag (Gemini, the benchmark questions whose expected documents are labelled): is each gold
  fact in the document, is it kept in the extraction, and could the question be answered
  from the extraction alone?

Labels default to DeepSeek's (data/labels.json); --labels FILE judges another model's output
in the same {doc_id: {about, entities, facts}} shape. Verdicts are cached per label set in
data/judge_<name>.json, so a rerun only pays for what is new.

Usage (from the main checkout, which has the data):
    uv run --project ~/projects/rag-implementation/train-rag python \
        exploration/fact_extraction/judge_labels.py specifics
    ... judge_labels.py hallucination --sample 300
    ... judge_labels.py rag
"""

import argparse
import json
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.evaluation.usage import UsageMeter
from train_rag.processing.normalize import normalize_documents
from train_rag.progress.recorder import RunProgress
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.repository.question import load_questions

HERE = Path(__file__).parent
DATA = HERE / "data"
ENV_FILE = Path.home() / "projects/rag-implementation/train-rag/.env"
MODEL = "gemini-3.8-flash"
MAX_CHARS = 30_000  # the same cut as when labelling
WORKERS = 8
SEED = 7
PROMPTS = yaml.safe_load((HERE / "judge_prompts.yaml").read_text())
# Specifics: anything with a digit (amounts, dates, ids, versions, hashes) and URLs.
SPECIFIC = re.compile(r"https?://\S+|\S*\d\S*")


class FactVerdict(BaseModel):
    """One extracted fact against the document."""

    number: int
    verdict: Literal["supported", "unsupported", "contradicted"]
    evidence: str
    note: str


class LabelVerdict(BaseModel):
    """All of one label against its document."""

    facts: list[FactVerdict]
    about: Literal["supported", "unsupported", "contradicted"]
    entities_not_in_document: list[str]


class OutputFactVerdict(BaseModel):
    """One fact of the fine-tuned model's output against the document."""

    number: int
    verdict: Literal["supported", "unsupported", "contradicted"]
    note: str


class ReferenceFactKept(BaseModel):
    """One fact of the reference label: does the output keep it?"""

    number: int
    kept: Literal["yes", "partial", "no"]


class CompareVerdict(BaseModel):
    """A model output against its document (hallucination) and the reference (completeness)."""

    output_facts: list[OutputFactVerdict]
    reference_facts: list[ReferenceFactKept]


class GoldFactVerdict(BaseModel):
    """One gold fact of a question: stated by the document, kept by the extraction?"""

    number: int
    in_document: bool
    in_extraction: Literal["yes", "partial", "no"]


class RagVerdict(BaseModel):
    """One question against the extracted facts of its expected documents."""

    gold_facts: list[GoldFactVerdict]
    answerable: Literal["yes", "partial", "no"]
    note: str


def squash(text: str) -> str:
    """Lowercase, without thousands separators and trailing punctuation noise."""
    return re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())


def spaced(text: str) -> str:
    """Numbers set apart from units and hyphens.

    "30-minute" and "130ms" become "30 minute" and "130 ms", as "30 minutes" and "130 ms"
    read in a document.
    """
    text = re.sub(r"(?<=\d)-(?=[a-z])|(?<=[a-z])-(?=\d)", " ", text)

    return re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", text)


def missing_specifics(fact: str, document: str, spaced_document: str) -> list[str]:
    """The numbers, ids and URLs of a fact that the document doesn't contain.

    A token counts as contained as written or with its numbers and units set apart.
    """
    found = []
    for token in SPECIFIC.findall(fact):
        token = squash(token).strip(".,;:!?()[]{}\"'")
        if token and token not in document and spaced(token) not in spaced_document:
            found.append(token)

    return found


def load_texts(doc_ids: list[str]) -> dict[str, tuple[str, str, str]]:
    """Source, title and normalized text (as labelled) of each document."""
    with Session(get_engine()) as session:
        ids = ids_of_doc_ids(session, EnterpriseRagBenchLoader.name, doc_ids)
        records = load_documents(session, sorted(ids))
    sources = {record.doc_id: str(record.source_type) for record in records}

    return {
        doc.doc_id: (
            sources[doc.doc_id],
            doc.title.splitlines()[0] if doc.title else "",
            doc.text[:MAX_CHARS],
        )
        for doc in normalize_documents(records)
    }


def numbered(items: list[str]) -> str:
    """Items as a numbered list."""
    return "\n".join(f"{number}. {item}" for number, item in enumerate(items, 1))


class Judge:
    """Asks Gemini for structured verdicts, keeps them in a cache file, counts the cost."""

    def __init__(self, cache: Path) -> None:
        """Load the cache and connect."""
        load_dotenv(ENV_FILE)
        self.client = genai.Client()
        self.usage = UsageMeter()
        self.cache_path = cache
        self.cache: dict[str, dict] = json.loads(cache.read_text()) if cache.exists() else {}
        self.lock = threading.Lock()

    def ask(self, key: str, instructions: str, prompt: str, schema: type[BaseModel]) -> dict:
        """One verdict, from the cache if it is there."""
        if key in self.cache:
            return self.cache[key]
        response = self.client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=instructions,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        self.usage.add(response)
        if not isinstance(response.parsed, schema):
            raise ValueError(f"no verdict for {key}")
        verdict = response.parsed.model_dump()
        with self.lock:
            self.cache[key] = verdict
            self.cache_path.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False))

        return verdict

    def run(self, title: str, jobs: dict[str, tuple[str, str, type[BaseModel]]]) -> dict[str, dict]:
        """Many verdicts in parallel, shown on the progress page; failures are left out."""
        results, failed = {}, 0
        progress = RunProgress(get_engine(), title, len(jobs))
        with progress, ThreadPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(self.ask, key, *job): key for key, job in jobs.items()}
            for done, future in enumerate(as_completed(futures), 1):
                try:
                    results[futures[future]] = future.result()
                except Exception as error:  # noqa: BLE001 - report and go on
                    failed += 1
                    print(f"  {futures[future]}: {error}")
                progress.advance("judging", 1, note=f"Gemini: ${self.usage.cost():.3f}")
                if done % 25 == 0:
                    print(f"  {done}/{len(jobs)} · ${self.usage.cost():.3f}", flush=True)
        print(f"judged {len(results)}, failed {failed}, this run ${self.usage.cost():.3f}")

        return results


def check_specifics(labels: dict[str, dict]) -> None:
    """The free check: specifics of every fact looked up in the document."""
    texts = load_texts(sorted(labels))
    per_source: dict[str, Counter] = defaultdict(Counter)
    examples = []
    for doc_id, label in labels.items():
        if doc_id not in texts:
            continue
        source, title, text = texts[doc_id]
        document = squash(f"{title}\n{text}")
        spaced_document = spaced(document)
        for fact in label["facts"]:
            missing = missing_specifics(fact, document, spaced_document)
            per_source[source]["facts"] += 1
            per_source[source]["flagged"] += bool(missing)
            if missing and len(examples) < 400:
                examples.append({"doc_id": doc_id, "fact": fact, "missing": missing})
    print(f"{'source':14} {'facts':>7} {'flagged':>8}")
    total = Counter()
    for source, counts in sorted(per_source.items()):
        total.update(counts)
        print(f"{source:14} {counts['facts']:>7} {counts['flagged'] / counts['facts']:>8.1%}")
    print(f"{'all':14} {total['facts']:>7} {total['flagged'] / total['facts']:>8.1%}")
    out = DATA / "specifics_check.json"
    out.write_text(json.dumps(examples, indent=1, ensure_ascii=False))
    print(f"flagged examples -> {out}")


def judge_hallucination(labels: dict[str, dict], name: str, sample: int, only: set[str]) -> None:
    """Gemini checks every fact of a sample of labels against its document."""
    pool = sorted(only & set(labels)) if only else sorted(labels)
    texts = load_texts(pool)
    by_source: dict[str, list[str]] = defaultdict(list)
    for doc_id in pool:
        if doc_id in texts and labels[doc_id]["facts"]:
            by_source[texts[doc_id][0]].append(doc_id)
    # The same share of every source, at least a few of each.
    generator = random.Random(SEED)
    total = sum(len(v) for v in by_source.values())
    picked: list[str] = []
    least = 1 if sample < 100 else 5
    for _source, doc_ids in sorted(by_source.items()):
        want = len(doc_ids) if sample >= total else min(
            len(doc_ids), max(least, round(sample * len(doc_ids) / total))
        )
        picked += generator.sample(doc_ids, want)

    judge = Judge(DATA / f"judge_hallucination_{name}.json")
    jobs = {}
    for doc_id in picked:
        source, title, text = texts[doc_id]
        label = labels[doc_id]
        prompt = PROMPTS["hallucination_prompt"].format(
            source=source,
            title=title,
            text=text,
            about=label["about"],
            entities="\n".join(label["entities"]),
            facts=numbered(label["facts"]),
        )
        jobs[doc_id] = (PROMPTS["hallucination_instructions"], prompt, LabelVerdict)
    verdicts = judge.run(f"judge {name} labels: hallucinations", jobs)

    per_source: dict[str, Counter] = defaultdict(Counter)
    bad = []
    for doc_id, verdict in verdicts.items():
        source = texts[doc_id][0]
        counts = per_source[source]
        counts["docs"] += 1
        counts["about_bad"] += verdict["about"] != "supported"
        counts["entities_bad"] += len(verdict["entities_not_in_document"])
        counts["entities"] += len(labels[doc_id]["entities"])
        doc_bad = False
        for fact in verdict["facts"]:
            counts["facts"] += 1
            counts[fact["verdict"]] += 1
            if fact["verdict"] != "supported":
                doc_bad = True
                index = fact["number"] - 1
                facts = labels[doc_id]["facts"]
                bad.append(
                    {
                        "doc_id": doc_id,
                        "source": source,
                        "verdict": fact["verdict"],
                        "fact": facts[index] if 0 <= index < len(facts) else "?",
                        "evidence": fact["evidence"],
                        "note": fact["note"],
                    }
                )
        counts["docs_with_bad_fact"] += doc_bad

    header = f"{'source':14} {'docs':>5} {'facts':>6} {'unsup':>7} {'contra':>7} {'docs≥1':>7}"
    print(header + f" {'about':>6} {'ent.':>6}")
    total = Counter()
    for c in per_source.values():
        total.update(c)
    for source, c in [*sorted(per_source.items()), ("all", total)]:
        print(
            f"{source:14} {c['docs']:>5} {c['facts']:>6} {c['unsupported'] / c['facts']:>7.1%} "
            f"{c['contradicted'] / c['facts']:>7.1%} {c['docs_with_bad_fact'] / c['docs']:>7.1%} "
            f"{c['about_bad'] / c['docs']:>6.1%} {c['entities_bad'] / max(1, c['entities']):>6.1%}"
        )
    out = DATA / f"hallucinations_{name}.json"
    out.write_text(json.dumps(bad, indent=1, ensure_ascii=False))
    print(f"unsupported and contradicted facts -> {out}")


def judge_rag(labels: dict[str, dict], name: str) -> None:
    """Gemini checks whether the extraction keeps the gold facts of the benchmark questions."""
    with Session(get_engine()) as session:
        questions = [
            q
            for q in load_questions(session, EnterpriseRagBenchLoader.name)
            if q.answer_facts
            and q.expected_doc_ids
            and all(d in labels for d in q.expected_doc_ids)
        ]
    texts = load_texts(sorted({d for q in questions for d in q.expected_doc_ids}))

    judge = Judge(DATA / f"judge_rag_{name}.json")
    jobs = {}
    for question in questions:
        documents = "\n\n".join(
            f'<document id="{d}" source="{texts[d][0]}" title="{texts[d][1]}">\n'
            f"{texts[d][2]}\n</document>"
            for d in question.expected_doc_ids
            if d in texts
        )
        extractions = "\n\n".join(
            f'<extraction id="{d}">\nabout: {labels[d]["about"]}\n'
            f"entities: {', '.join(labels[d]['entities'])}\n"
            f"facts:\n{numbered(labels[d]['facts'])}\n</extraction>"
            for d in question.expected_doc_ids
        )
        prompt = PROMPTS["rag_prompt"].format(
            question=question.text,
            answer=question.gold_answer,
            gold_facts=numbered(question.answer_facts),
            documents=documents,
            extractions=extractions,
        )
        jobs[question.id] = (PROMPTS["rag_instructions"], prompt, RagVerdict)
    verdicts = judge.run(f"judge {name} labels: gold facts kept", jobs)

    types_of = {q.id: str(q.question_type) for q in questions}
    per_type: dict[str, Counter] = defaultdict(Counter)
    rows = []
    for question in questions:
        verdict = verdicts.get(question.id)
        if verdict is None:
            continue
        for key in (types_of[question.id], "all"):
            c = per_type[key]
            c["questions"] += 1
            c[f"answerable_{verdict['answerable']}"] += 1
            for gold in verdict["gold_facts"]:
                c["gold"] += 1
                c["in_document"] += gold["in_document"]
                c[f"kept_{gold['in_extraction']}"] += 1
                c["kept_of_in_doc"] += gold["in_document"] and gold["in_extraction"] == "yes"
        rows.append(
            {
                "question_id": question.id,
                "type": types_of[question.id],
                "question": question.text,
                "gold_facts": question.answer_facts,
                **verdict,
            }
        )

    print(
        f"{'type':26} {'qs':>4} {'gold':>5} {'in doc':>7} {'kept':>6} {'partial':>8} "
        f"{'kept/in doc':>12} {'answerable':>11} {'partly':>7}"
    )
    for kind, c in sorted(per_type.items(), key=lambda kv: (kv[0] == "all", kv[0])):
        print(
            f"{kind:26} {c['questions']:>4} {c['gold']:>5} {c['in_document'] / c['gold']:>7.0%} "
            f"{c['kept_yes'] / c['gold']:>6.0%} {c['kept_partial'] / c['gold']:>8.0%} "
            f"{c['kept_of_in_doc'] / max(1, c['in_document']):>12.0%} "
            f"{c['answerable_yes'] / c['questions']:>11.0%} "
            f"{c['answerable_partial'] / c['questions']:>7.0%}"
        )
    out = DATA / f"rag_check_{name}.json"
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"per question -> {out}")


def judge_compare(outputs: dict[str, dict], name: str, sample: int) -> None:
    """Gemini checks model outputs against the document and against DeepSeek's labels."""
    reference = {v["doc_id"]: v for v in json.loads((DATA / "labels.json").read_text()).values()}
    texts = load_texts(sorted(outputs))
    by_source: dict[str, list[str]] = defaultdict(list)
    for doc_id in sorted(outputs):
        if doc_id in texts and doc_id in reference:
            by_source[texts[doc_id][0]].append(doc_id)
    total = sum(len(v) for v in by_source.values())
    generator = random.Random(SEED)
    picked: list[str] = []
    for _source, doc_ids in sorted(by_source.items()):
        want = min(len(doc_ids), max(1, round(sample * len(doc_ids) / total)))
        picked += generator.sample(doc_ids, want)

    judge = Judge(DATA / f"judge_compare_{name}.json")
    jobs = {}
    for doc_id in picked:
        source, title, text = texts[doc_id]
        out, ref = outputs[doc_id], reference[doc_id]
        prompt = PROMPTS["compare_prompt"].format(
            source=source,
            title=title,
            text=text,
            ref_about=ref["about"],
            ref_facts=numbered(ref["facts"]),
            out_about=out["about"],
            out_entities=", ".join(out["entities"]),
            out_facts=numbered(out["facts"]),
        )
        jobs[doc_id] = (PROMPTS["compare_instructions"], prompt, CompareVerdict)
    verdicts = judge.run(f"judge {name}: hallucination and completeness vs DeepSeek", jobs)

    per_source: dict[str, Counter] = defaultdict(Counter)
    bad = []
    for doc_id, verdict in verdicts.items():
        c = per_source[texts[doc_id][0]]
        c["docs"] += 1
        doc_bad = False
        for fact in verdict["output_facts"]:
            c["facts"] += 1
            c[fact["verdict"]] += 1
            if fact["verdict"] != "supported":
                doc_bad = True
                index = fact["number"] - 1
                facts = outputs[doc_id]["facts"]
                bad.append(
                    {
                        "doc_id": doc_id,
                        "source": texts[doc_id][0],
                        "verdict": fact["verdict"],
                        "fact": facts[index] if 0 <= index < len(facts) else "?",
                        "note": fact["note"],
                    }
                )
        c["docs_with_bad_fact"] += doc_bad
        for ref in verdict["reference_facts"]:
            c["ref"] += 1
            c[f"kept_{ref['kept']}"] += 1

    total_counts: Counter = Counter()
    for c in per_source.values():
        total_counts.update(c)
    print(
        f"{'source':14} {'docs':>5} {'facts':>6} {'unsup':>7} {'contra':>7} {'docs≥1':>7}"
        f" {'ref':>5} {'kept':>6} {'partial':>8} {'missing':>8}"
    )
    for source, c in [*sorted(per_source.items()), ("all", total_counts)]:
        print(
            f"{source:14} {c['docs']:>5} {c['facts']:>6} {c['unsupported'] / c['facts']:>7.1%} "
            f"{c['contradicted'] / c['facts']:>7.1%} {c['docs_with_bad_fact'] / c['docs']:>7.1%} "
            f"{c['ref']:>5} {c['kept_yes'] / c['ref']:>6.0%} {c['kept_partial'] / c['ref']:>8.0%} "
            f"{c['kept_no'] / c['ref']:>8.0%}"
        )
    out_path = DATA / f"compare_{name}.json"
    out_path.write_text(json.dumps(bad, indent=1, ensure_ascii=False))
    print(f"unsupported and contradicted output facts -> {out_path}")


def main() -> None:
    """Run one check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["specifics", "hallucination", "rag", "compare"])
    parser.add_argument("--labels", type=Path, default=None, help="default: DeepSeek's labels")
    parser.add_argument("--sample", type=int, default=300, help="hallucination: labels to judge")
    parser.add_argument(
        "--split",
        choices=["train", "pilot", "all"],
        default="train",
        help="hallucination: which labelled documents to sample from",
    )
    parser.add_argument(
        "--same-docs",
        type=Path,
        default=None,
        help="hallucination: judge exactly the documents of this earlier verdict file",
    )
    args = parser.parse_args()

    if args.labels is None:
        name = "deepseek"
        labels = {v["doc_id"]: v for v in json.loads((DATA / "labels.json").read_text()).values()}
    else:
        name = args.labels.stem
        labels = json.loads(args.labels.read_text())

    if args.check == "compare":
        judge_compare(labels, name, args.sample)
    elif args.check == "specifics":
        check_specifics(labels)
    elif args.check == "hallucination":
        sample = json.loads((DATA / "sample.json").read_text())
        only = {
            "train": {r["doc_id"] for r in sample["train"]},
            "pilot": set(sample["pilot"]),
            "all": set(),
        }[args.split]
        if args.same_docs is not None:
            only = set(json.loads(args.same_docs.read_text()))
            args.sample = len(only)
        judge_hallucination(labels, name, args.sample, only)
    else:
        judge_rag(labels, name)


main()
