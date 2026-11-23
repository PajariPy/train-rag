# Evaluation with Clarifying Questions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the snippet agent on the 46 semantic failures when it may ask a simulated
person up to 3 clarifying questions, next to its zero-shot score (run 18), split by clean
and misremembered memory notes.

**Architecture:** Memory notes live in a new `question_memories` table, written once by a
Gemini note writer. A `SimulatedPerson` (Gemini) answers from one note. A
`ClarifyingSnippetAgent` subclasses `SnippetAgentSystem` with an `ask_user` tool whose
budget and person travel in a per-question budget object. The evaluation runner hands a
fresh person to systems that can ask; everything else keeps seeing only the question text.

**Tech Stack:** Python 3.13, sqlmodel/Postgres, google-genai (Gemini 3.8 Flash), pytest
with fakes (no DB or network in tests), `uv`.

**Spec:** `docs/superpowers/specs/2026-11-09-clarification-eval-design.md`

## Global Constraints

- Checks before calling a task done: `uv run ruff check . && uv run mypy src tests && uv run pytest`.
- Git (CLAUDE.md): work on a `feature-<name>` branch from the latest `dev`; **commit only
  when the user asks** — the "Commit" steps below mean "stage nothing, leave uncommitted"
  unless the user has said to commit. Never push to `dev`/`main`, never merge.
- Every LLM prompt lives in `src/train_rag/prompts.yaml`, loaded with `prompt(name, **values)`.
- Tables: sqlmodel in `src/train_rag/tables/`, queries in `src/train_rag/repository/`,
  pydantic models in `src/train_rag/models/`. Google docstrings, strict typing.
- Model: Gemini 3.8 Flash (`GEMINI_MODEL` from `evaluation/pipeline_answer.py`).
- Asks per question: **3**. Searches 5 and reads 3 stay unchanged.
- Wrong-detail share: **0.25** of the notes, chosen with `random.Random(seed=7)` over the
  sorted question ids: `round(46 * 0.25) = 12` notes.
- The agent never sees the note, the gold answer or the expected documents. The person
  never sees the gold answer or the document.
- Every paid run (note writing, evaluation) is confirmed with the user before it starts;
  long runs report through `RunProgress`.

## Review Focus

- **The person is asked something the note doesn't cover** → replies "I don't remember"-style, never invents. Pinned by `test_person_prompt_forbids_inventing` (Task 2) and checked by eye in the note review (Task 6).
- **The agent asks a 4th time** → gets `ASK_LIMIT_REACHED`, the person is not called. Pinned in Task 3.
- **A system that can't ask is evaluated with a person available** → runs exactly as before. Pinned in Task 4.
- **A question id in the ids file has no memory note** → the run stops before any paid call, naming the missing ids. Pinned in Task 5.
- **The note writer returns a note with an exact title/number from the document** → caught in the user's review; the leak check flags notes sharing a 4+ character token sequence with the document title. Pinned by `test_leaks_title` in Task 1.

---

### Task 0: Branch

- [ ] **Step 1:** Ask the user which base to use: the spec relies on uncommitted files of
  `feature-fact-extraction` (`exploration/retrieval_failures/data/pools_run15.json`).
  Default if they don't mind: `git switch dev && git pull && git switch -c feature-clarification-eval`
  (untracked files come along). Expected: on `feature-clarification-eval`.

### Task 1: Memory notes — model, table, repository, picking and leak check

**Files:**
- Create: `src/train_rag/models/memory.py`, `src/train_rag/tables/question_memory.py`, `src/train_rag/repository/question_memory.py`, `src/train_rag/evaluation/memory_notes.py`
- Modify: `src/train_rag/tables/__init__.py` (export `QuestionMemoryRecord`)
- Test: `tests/evaluation/test_memory_notes.py`

**Interfaces:**
- Produces:
  - `class MemoryNote(BaseModel)`: `question_id: str`, `note: str`, `wrong_detail: str | None = None`; property `clean -> bool`.
  - `class QuestionMemoryRecord(SQLModel, table=True)`, `__tablename__ = "question_memories"`: `question_id: str` (primary key), `note: str` (Text), `wrong_detail: str | None` (Text, nullable), `model: str`, `prompt_version: str`, `created_at: datetime` (default now, UTC).
  - `upsert_memories(session: Session, records: Sequence[QuestionMemoryRecord]) -> None` (on conflict on `question_id`: update every column).
  - `load_memories(session: Session, question_ids: Sequence[str]) -> dict[str, MemoryNote]`.
  - In `memory_notes.py`: `WRONG_SHARE = 0.25`, `SEED = 7`, `pick_wrong(question_ids: Sequence[str], share: float = WRONG_SHARE, seed: int = SEED) -> set[str]`, `leaks_title(note: str, title: str) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pick_wrong_takes_a_quarter_reproducibly() -> None:
    ids = [f"qst_{n:04d}" for n in range(46)]
    first = pick_wrong(ids)
    assert len(first) == 12
    assert first == pick_wrong(list(reversed(ids)))  # order of input doesn't matter
    assert first <= set(ids)

def test_pick_wrong_seed_changes_choice() -> None:
    ids = [f"qst_{n:04d}" for n in range(46)]
    assert pick_wrong(ids, seed=1) != pick_wrong(ids, seed=2)

def test_leaks_title() -> None:
    title = "introduce-kernel-stability-thresholds-and-precision-annealing"
    assert leaks_title("- a PR about precision annealing", title)
    assert not leaks_title("- a PR about falling back to full precision", title)

def test_memory_note_clean() -> None:
    assert MemoryNote(question_id="q", note="- a PR").clean
    assert not MemoryNote(question_id="q", note="- a Jira ticket", wrong_detail="says Jira; it is Linear").clean
```

`leaks_title` rule: normalize both to lowercase words (split on non-alphanumerics); a leak
is any two consecutive title words of 4+ characters each appearing consecutively in the
note.

- [ ] **Step 2:** Run `uv run pytest tests/evaluation/test_memory_notes.py -v` → FAIL (imports missing).
- [ ] **Step 3:** Implement the interfaces above. `pick_wrong`: `random.Random(seed).sample(sorted(ids), round(len(ids) * share))`.
- [ ] **Step 4:** Run the tests → PASS; run the full checks.
- [ ] **Step 5: Commit** (only if the user asked): `feat: memory notes table and picking`.

### Task 2: Note writer and simulated person

**Files:**
- Modify: `src/train_rag/evaluation/memory_notes.py`, `src/train_rag/prompts.yaml`
- Create: `src/train_rag/evaluation/person.py`
- Test: `tests/evaluation/test_memory_notes.py`, `tests/evaluation/test_person.py`

**Interfaces:**
- Consumes: `MemoryNote` (Task 1); `UsageMeter`; `GEMINI_MODEL`.
- Produces:
  - `NOTE_PROMPT_VERSION = "v1"`.
  - `write_note(client: genai.Client, usage: UsageMeter, question: Question, document: str, wrong: bool) -> MemoryNote` — one Gemini call with JSON output `{"note": str, "wrong_detail": str | null}`; when `wrong` is False, `wrong_detail` is forced to None whatever the model returns; when True and the model returns null, raise `ValueError`.
  - `class Person(Protocol)`: `def reply(self, ask: str) -> str`.
  - `class SimulatedPerson`: `__init__(self, client: genai.Client, usage: UsageMeter, question: str, note: MemoryNote)`; `reply(ask) -> str` keeps the exchange history and sends it each call; thinking LOW.
- Prompts (exact copy is left to the builder, rules are not):
  - `memory_note_instructions`: 2–4 short bullet points; kind of document/where it lived, team or role, roughly when, what it was about in plain words; NEVER exact titles, ids, code/product names usable as search keys, numbers, exact dates, people's names, or the answer. With `{wrong}` true: exactly one plausible, nearby wrong detail (neighbouring team, similar source, nearby time), and describe it in `wrong_detail`.
  - `person_instructions`: you are the employee who asked `{question}`; you remember only `{note}`; answer the assistant in one short casual sentence; answer only what is asked; if the note doesn't cover it, say you don't remember; state everything in the note with the same confidence.

- [ ] **Step 1: Write the failing tests** (fake `client.models.generate_content`, reuse `FakeModels` from `tests/evaluation/test_agent.py`):

```python
def test_write_note_clean_drops_wrong_detail() -> None:
    # model returns {"note": "- a PR", "wrong_detail": "x"} with wrong=False
    assert write_note(client, usage, question, "doc", wrong=False).wrong_detail is None

def test_write_note_wrong_requires_detail() -> None:
    # model returns {"note": "- a PR", "wrong_detail": None} with wrong=True
    with pytest.raises(ValueError):
        write_note(client, usage, question, "doc", wrong=True)

def test_write_note_prompt_has_question_and_document_not_seed_choice_leak() -> None:
    write_note(client, usage, question, "DOCUMENT TEXT", wrong=True)
    sent = models.requests[0]
    assert "DOCUMENT TEXT" in sent and question.text in sent

def test_person_sees_note_and_history_never_gold() -> None:
    person = SimulatedPerson(client, usage, question.text, MemoryNote(question_id="q", note="- a PR"))
    person.reply("Was it a PR or a ticket?"); person.reply("Which team?")
    second = models.requests[1]
    assert "- a PR" in second and "Was it a PR or a ticket?" in second
    assert question.gold_answer not in second

def test_person_prompt_forbids_inventing() -> None:
    text = prompt("person_instructions", question="q", note="n")
    assert "don't remember" in text.lower() or "do not remember" in text.lower()
```

- [ ] **Step 2:** Run both test files → FAIL.
- [ ] **Step 3:** Implement `write_note`, `SimulatedPerson`, and the two prompts.
- [ ] **Step 4:** Tests → PASS; full checks.
- [ ] **Step 5: Commit** (only if asked): `feat: note writer and simulated person`.

### Task 3: The clarifying snippet agent

**Files:**
- Modify: `src/train_rag/evaluation/agent.py` (let `answer` accept a budget), `src/train_rag/evaluation/agent_tools.py` (ask tool), `src/train_rag/evaluation/system.py` (`ClarifyingSystem`), `src/train_rag/prompts.yaml`
- Create: `src/train_rag/evaluation/agent_clarify.py`
- Test: `tests/evaluation/test_agent_clarify.py`

**Interfaces:**
- Consumes: `Person` (Task 2), `SnippetAgentSystem`, `Budget`.
- Produces:
  - `AgentSystem.answer(self, question: str, budget: Budget | None = None) -> SystemAnswer` (None → `Budget()`, behaviour unchanged).
  - `ASK_TOOL = "ask_user"`; `clarify_tools() -> types.Tool` = `snippet_tools()` declarations + `ask_user(question: str)`, description from prompt `ask_tool_description`.
  - `class ClarifyBudget(Budget)` (dataclass): `person: Person | None = None`, `asks: int = 0`.
  - `class ClarifyingSystem(RagSystem)` (abstract, in `system.py`): `answer_with(self, question: str, person: Person) -> SystemAnswer`. Its `answer(question)` (no person) behaves as the parent agent.
  - `class ClarifyingSnippetAgent(SnippetAgentSystem, ClarifyingSystem)`: `name = "agent-snippets-clarify"`, `MAX_ASKS = 3`, `ASK_LIMIT_REACHED = "No more questions: answer from what you have."`; instructions prompt `clarify_agent_instructions` (the snippet instructions plus: ask only when results leave real doubt; the person remembers the big picture, not details; their memory may be wrong; at most `{max_asks}` questions). `can_call` is True while asks are left too. `run_tool` for `ASK_TOOL`: over budget → `{"error": ASK_LIMIT_REACHED}, []`; else `asks += 1`, `{"reply": person.reply(question)}, []`.

- [ ] **Step 1: Write the failing tests** (fake Gemini as in `test_agent_snippets.py`, a `FakePerson` recording asks):

```python
def test_ask_reaches_person_and_reply_goes_back() -> None:
    # turns: ask_user("Which team?"), final_answer("A")
    result = agent.answer_with("q?", person)
    assert person.asks == ["Which team?"]
    assert any(s.tool == "ask_user" and s.output == "the inference team" for s in result.trace)

def test_fourth_ask_is_refused() -> None:
    # turns: 4 × ask_user, final_answer
    agent.answer_with("q?", person)
    assert len(person.asks) == 3
    # the 4th tool step output equals ASK_LIMIT_REACHED

def test_asks_do_not_use_search_budget() -> None:
    # turns: 3 × ask_user, then 4 × keyword_search, final_answer → all 4 searches ran
    ...

def test_without_person_behaves_like_snippet_agent() -> None:
    assert ClarifyingSnippetAgent(...).answer("q?").answer == SnippetAgentSystem(...).answer("q?").answer
```

- [ ] **Step 2:** Run `uv run pytest tests/evaluation/test_agent_clarify.py -v` → FAIL.
- [ ] **Step 3:** Implement. Existing agent tests must keep passing untouched.
- [ ] **Step 4:** `uv run pytest tests/evaluation -v` → PASS; full checks.
- [ ] **Step 5: Commit** (only if asked): `feat: snippet agent that can ask the user`.

### Task 4: Runner hands a person to clarifying systems

**Files:**
- Modify: `src/train_rag/evaluation/pipeline.py`
- Test: `tests/evaluation/test_pipeline.py`

**Interfaces:**
- Consumes: `ClarifyingSystem`, `Person` (Tasks 2–3).
- Produces: `PersonFactory = Callable[[Question], Person]`; `evaluate_question(..., person_factory: PersonFactory | None = None)`, same keyword added to `evaluate_questions` and `run_evaluation`. Rule: if `isinstance(system, ClarifyingSystem)` and a factory is given → `system.answer_with(question.text, person_factory(question))`; otherwise `system.answer(question.text)`. A fresh person per question.

- [ ] **Step 1: Failing tests:** `test_clarifying_system_gets_a_fresh_person_per_question` (factory called once per question, with that question); `test_plain_system_ignores_person_factory` (factory never called, answer unchanged).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS; full checks.
- [ ] **Step 5: Commit** (only if asked): `feat: evaluation passes a person to clarifying systems`.

### Task 5: Commands — write notes, run by id list, report

**Files:**
- Modify: `src/train_rag/main.py`, `src/train_rag/evaluate.py`, `src/train_rag/evaluation/registry.py`
- Create: `src/train_rag/evaluation/clarify_report.py`, `exploration/clarification/failures_run15.txt` (the 46 ids, one per line, from the `pools` keys of `pools_run15.json`)
- Test: `tests/evaluation/test_clarify_report.py`, `tests/evaluation/test_memory_notes.py`

**Interfaces:**
- Consumes: everything above; `results_of_run(session, run_id)`; repository document loaders (`ids_of_doc_ids`, `load_documents`, `normalize_documents`) for the right document's text (first expected doc id).
- Produces:
  - `read_question_ids(path: Path) -> list[str]` (strip, skip blank lines and `#` comments).
  - `missing_memories(question_ids: Sequence[str], memories: Mapping[str, MemoryNote]) -> list[str]`.
  - `evaluate(..., question_ids: list[str] | None = None)` — filters questions to these ids (error listing unknown ids). For `agent-snippets-clarify` it loads memories, **raises before any Gemini call** if `missing_memories` is non-empty, and passes `person_factory=lambda q: SimulatedPerson(genai.Client(), PERSON_USAGE, q.text, memories[q.id])`.
  - `PERSON_USAGE = UsageMeter()` in `registry.py`; the evaluate printout reports `USAGE` and `PERSON_USAGE` separately.
  - CLI: `evaluate --question-ids FILE`; new command `write-memories --question-ids FILE [--seed 7] [--redo ID ...]` that picks wrong notes, writes and upserts them with `RunProgress`, then prints every note with its `wrong_detail` and a `LEAK?` marker where `leaks_title` is true. `--redo` rewrites only the given ids, keeping their clean/wrong choice. New command `clarify-report --run RUN --baseline 18 --question-ids FILE`.
  - `clarify_report(baseline: Mapping[str, bool | None], run: Mapping[str, bool | None], asks: Mapping[str, int], memories: Mapping[str, MemoryNote]) -> str` — rows: Zero-shot (baseline), With clarification, — clean notes, — note with a wrong detail; columns Questions, Correct, Accuracy; then "asked at least once: N of M", "average asks: x.x". `asks` is counted from each result's trace steps with `tool == "ask_user"`.

- [ ] **Step 1: Failing tests:**

```python
def test_report_splits_clean_and_wrong() -> None:
    memories = {"a": MemoryNote(question_id="a", note="n"),
                "b": MemoryNote(question_id="b", note="n", wrong_detail="w")}
    text = clarify_report({"a": False, "b": True}, {"a": True, "b": False}, {"a": 2, "b": 0}, memories)
    assert "Zero-shot" in text and "50%" in text
    assert "asked at least once: 1 of 2" in text

def test_missing_memories_listed() -> None:
    assert missing_memories(["a", "b"], {"a": MemoryNote(question_id="a", note="n")}) == ["b"]

def test_read_question_ids_skips_blanks_and_comments(tmp_path) -> None:
    path = tmp_path / "ids.txt"; path.write_text("# failures\nqst_1\n\n qst_2 \n")
    assert read_question_ids(path) == ["qst_1", "qst_2"]
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement; register `ClarifyingSnippetAgent` in `SYSTEMS`. **Step 4:** Run → PASS; full checks.
- [ ] **Step 5: Commit** (only if asked): `feat: memory-note and clarification commands`.

### Task 6: Write the notes and get them reviewed (paid, ~$0.50)

- [ ] **Step 1:** Ask the user to approve the spend. Then `uv run python -m train_rag.main write-memories --question-ids exploration/clarification/failures_run15.txt`. Expected: 46 notes stored, 12 with a `wrong_detail`, printed.
- [ ] **Step 2:** Show the user ~10 notes (at least 3 with a wrong detail, every `LEAK?` one). Rewrite rejected ones with `--redo`; repeat until approved.

### Task 7: Clarification run and report (paid, ~$3–4)

- [ ] **Step 1:** Ask the user to approve the spend. Run `uv run python -m train_rag.main evaluate --system agent-snippets-clarify --question-ids exploration/clarification/failures_run15.txt`. Expected: a stored run on 46 questions, with costs printed for the agent and the person separately.
- [ ] **Step 2:** `uv run python -m train_rag.main clarify-report --run <new id> --baseline 18 --question-ids exploration/clarification/failures_run15.txt`. Expected: the zero-shot row reads 46 / 7 / 15%.
- [ ] **Step 3:** Read ~5 traces (asks and replies) for leaks or over-helpful replies; report to the user with the table. Add the results to the research map (step 7 card) and a new article, after the user agrees.
