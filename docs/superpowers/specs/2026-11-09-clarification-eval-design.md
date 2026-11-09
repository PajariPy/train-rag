# Evaluation with clarifying questions — design

Date: 2026-11-09. Status: draft for approval.

## Why

46 semantic questions fail because search can't single out the right document among
near-duplicates (run 15). Embedding changes, reranker settings, deeper lists and fact
extraction have all been tried (articles 003–006, research map
`docs/research/semantic_search_failure_problem.canvas`); the best end-to-end result
reaches 9 of 46. The question text alone often doesn't say enough.

A real assistant would ask the person what they mean. Today's evaluation can't measure
that: a system only ever sees the question text and must answer in one go. This design adds a
second way to score a system: **accuracy with clarification**, where the agent may ask the
person up to 3 questions, answered by a simulated person who remembers the document the way
humans do — the big picture, rarely the details, sometimes one detail wrong.

## Goals

- For each of the 46 failures, a stored **memory note**: what the person plausibly remembers
  about the right document.
- A **simulated person** (an LLM) that answers the agent's questions only from that note.
- The **snippet agent** (run 18, 60.8% on semantic) with a new tool, **ask the user**, usable
  up to 3 times per question.
- A report with **zero-shot accuracy** and **accuracy with clarification** on the same
  questions, split into clean notes and notes with a misremembered detail.

## Non-goals

- All 125 semantic questions, or other question types. Start with the 46 failures; widen if
  the result is interesting (the user's call).
- Changing search, the reranker or the judge.
- A real user interface for clarifying questions.

## Anti-goals (would make it a failure even if it runs)

- **Notes that give the answer away**: exact titles, ids, numbers, names, dates, or the
  answer itself. Then "accuracy with clarification" only measures leakage.
- **The agent seeing the note or the gold answer.** It must only see the person's replies.
- **A person who over-answers**: long, helpful paragraphs no real person would write.
- **Wrong details we can't identify**: every misremembered detail is labelled.

## The memory note

One per question, written once and stored with the evaluation set.

- **Content:** 2–4 short points of what a person would plausibly remember: the kind of
  document or where it lived (a PR, a ticket, a Slack channel, an email, a meeting), the team
  or role, roughly when, and what it was about in plain words.
- **Never:** exact titles, ids, product or code names that would be a search key, numbers,
  exact dates, names of people, or the answer to the question.
- **Misremembered detail:** about a quarter of the notes (~11–12 of 46) hold exactly one
  detail that is slightly wrong but plausible and close to the truth: a neighbouring team,
  a similar source (Jira vs Linear, Slack vs email), a nearby time ("last month" when it
  was three months ago). Never invented from nothing.
- **Label:** each note records `wrong_detail`: null for a clean note, otherwise a short
  description, e.g. `"says Jira; the document is in Linear"`.

Example (qst_0178, a GitHub PR about precision annealing):

> - It was a PR, I think from the inference team.
> - Something about automatically falling back to full precision when nodes get flaky.

**Written by** Gemini 3.8 Flash from the question, the gold answer and the right document,
with a prompt that states the rules above. Which notes get a wrong detail is chosen at random
(fixed seed) before generation, so the share is controlled, not left to the model.

**Checked by** the user: about 10 notes, including some with a wrong detail, are shown for
review before any evaluation run. Notes that leak are rewritten.

**Stored** in a new table `question_memories` (sqlmodel table + repository functions, like
the existing `questions` table): question id, note text, `wrong_detail`, the model and
prompt version that wrote it, created at.

## The simulated person

- Gemini 3.8 Flash, one call per agent question, low thinking.
- Sees: the original question, its memory note, and the conversation so far. Never the
  right document or the gold answer.
- Answers in **one short sentence**, casually, only from the note. Anything the note
  doesn't cover → "I don't remember" (or similar). States the wrong detail as confidently
  as the true ones.
- Doesn't volunteer the whole note at once: answers what was asked.

## The agent

- `SnippetAgentSystem` plus a tool **`ask_user(question)`** with its own budget of 3 asks
  per question, separate from searches and reads. After 3, the tool returns a "no more
  questions: answer from what you have" message, like the other limits.
- The agent decides itself whether and when to ask. Its instructions say: ask only when the
  search results leave real doubt; the person remembers the big picture, not exact details;
  their memory may be wrong.
- Each ask and reply is recorded in the trace, like searches.
- Interface: today `RagSystem.answer(question)` sees only the question text. The clarifying
  agent also needs a way to reach the person, but must not see the note. It gets a
  per-question **`Person`** object (`ask(text) -> reply`) that holds the note privately.
  The rest of the evaluation keeps the "only the question text" rule.

## Running and reporting

- **Questions:** exactly the 46 retrieval failures of run 15: the keys of `pools` in
  `exploration/retrieval_failures/data/pools_run15.json`. Not `--wrong-in 15`, which returns
  all 57 wrong answers, including the ones that aren't retrieval failures. The evaluate
  command needs a way to take an explicit list of question ids (e.g. `--question-ids FILE`).
- **Zero-shot baseline:** run 18's stored results on the same 46 questions (no new cost).
  Checked: run 18 holds all 46, and the snippet agent got **7 of 46** right.
- The agent's first, automatic search of the question stays as it is; asking comes after
  the agent has seen those results.
- **Clarification run:** a new system name, e.g. `agent-snippets-clarify`, in `SYSTEMS`,
  stored as a normal run with the same Gemini judge.
- **Report (one table):**

  | | Questions | Correct | Accuracy |
  |---|---:|---:|---:|
  | Zero-shot (run 18) | 46 | 7 | 15% |
  | With clarification | 46 | | |
  | — clean notes | ~35 | | |
  | — note with a wrong detail | ~11 | | |

  Plus: questions where the agent asked at all, average asks per question, and cost per
  question for each mode (the agent and the simulated person counted separately).
- **Progress:** `RunProgress` is wired in before the first run (as for every long run).

## Cost

- Writing 46 notes: well under $0.50.
- One clarification run on 46 questions: roughly $3–4 (snippet agent ~$0.057 per question,
  plus extra turns and the person's short replies).
- Every paid step is confirmed with the user before it runs.

## Decided (by the user)

- Evaluation gains a second mode: accuracy with clarifying questions, next to zero-shot.
- Person-memory notes plus an LLM playing the person, not fixed answers.
- Replies are short and human: big picture, rarely exact details.
- Some notes are slightly wrong, and each such note is labelled so it can be scored apart.
- Up to 3 rounds of questions.
- The snippet agent with an "ask the user" tool.
- Start with the 46 failures.

## Calls made by Claude (check these)

- About a quarter of the notes get a wrong detail; chosen at random with a fixed seed.
- Gemini 3.8 Flash writes the notes and plays the person.
- Notes are stored in a new database table, not a JSON file.
- The zero-shot baseline reuses run 18 instead of a fresh run.
- The name "accuracy with clarification".
- Asks have their own budget; searches and reads keep theirs (5 and 3).
- A new `--question-ids FILE` option on the evaluate command to run exactly the 46.
- The first automatic search stays; the agent asks after seeing it.
- A note that leaks is rewritten by re-running the note writer for that question, and shown
  again for review.

## Left to the builder

- Prompt wording for the note writer, the person and the agent's ask instructions.
- Exact table columns and the shape of the `Person` interface.
- How the report is printed (extend `format_report` or a separate script).
