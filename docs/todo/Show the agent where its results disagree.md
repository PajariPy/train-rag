Let the search agent notice when its top results describe different cases (near-twin threads
with different facts), and ask the person about what separates them. An LLM classifier
decides whether the results agree; only when they don't, a second LLM with a prompt made for
that case helps the agent. Neither LLM answers the question.

## Why
On the Slack semantic dev set, for 76.5% of questions another shown document answers the
question fully, with different facts (another fix, owner, deadline). The final top 5 holds
*an* answering document for 88% of questions but the right thread for only 55.5% (agent v3
top 100 + reranker). The gap is choosing which case the person means, and only the person can
close it. Asking alone helped only where the reply could become a filter (v4: top 10 27% →
39% on the questions it asked). See [[Test mined fields as search filters]] and article 007.

## How
Models: **Gemini** (`gemini-3.8-flash`) for both LLM steps, as for the judge and the
simulated person.

1. **Agreement classifier** (Gemini, after each search, or once before `finish`): given the
   question and the top 5 results (mined fields + snippet), does it say they describe the
   **same case** or **different cases**? Output: `agree` / `disagree`, and for `disagree` the
   groups of result ids that belong together. No answer to the question.
2. **Disagreement step** (Gemini, only on `disagree`): a prompt written for that case. It gets
   the groups and their context fields (channel, source, team, client, area, time, outcome)
   and returns what separates the cases **in terms a person remembers** (where it was
   discussed, which team, roughly when, what kind of case), plus one suggested clarifying
   question. Never names, ids or numbers, since the person doesn't remember those.
3. The agent sees that contrast in the tool output and decides whether to ask
   (`ask_user`) and which filter to use with the reply.
   If asking isn't possible or the person doesn't know, it **presents the results as they
   are**: the cases side by side ("two threads match: one where … , one where …"), instead
   of picking one.

Calibration set (18 Nov): `exploration/slack_extraction/data/dev_contradictions.jsonl`,
84 questions, 28 each consistent / specific / ambiguous (`contradictions.py`; research map
semantic search failures, data preprocessing problem, step 7). Expected behaviour: consistent → answer; specific → pick the
matching document, no asking; ambiguous → ask, or show the cases side by side.

Code: extend `exploration/slack_extraction/agent_search.py` (v4: sources and channels
filters, `ask_user`) and `clarify_agent.py` (memory notes, simulated person).

Measure as before, on the 200 dev questions: right thread in first 10 / 50 / 100, picks,
final top 5 after the reranker; split by clean / wrong memory notes; plus how often the
classifier says `disagree` and whether the right thread's group is the one chosen. Repeat
runs 2–3 times: the agent's run-to-run noise is 6–10 points at top 10.

Check first: the memory notes hold kind of document, team, time and gist, but not customer
type or outcome. Decide whether a person would remember those before relying on them.

Paid (ask first): Gemini classifier ≈ 200 questions × up to 4 calls, disagreement step on
part of them, plus DeepSeek for the agent (≈ $0.40 a run).

## Result (20 Nov): classifier
Code: `exploration/slack_extraction/agreement.py` (prompt versions in `PROMPTS`, results in
`data/agreement_<version>_<input>[_s<k>]_result.json`). Research map: semantic search failures,
data preprocessing problem, step 8.

- **Hand review:** 5 labels changed, 1 question dropped (its detail gave the answer away):
  83 questions, 27 consistent / 27 specific / 29 ambiguous
  (`data/dev_contradictions_review.json`; `auto_class` keeps the generated label).
- **Input size**, baseline prompt, ask precision / recall: 300-character snippets (what the
  agent sees) 0.89 / 0.59; first 3,000 characters 0.85 / 0.76; up to 12,000 0.82 / 0.79.
  The classifier needs real text; full text and the first 3,000 characters score the same
  within noise.
- **Prompts**, ask F1 on 3,000 characters: baseline 0.80 / 0.81 (two runs); v2 (rules
  for details and iterations) 0.77; v3 (each document's case listed first) 0.80 / 0.72;
  v4 (each document judged on its own) 0.76. The same prompt flips the decision on 3–6
  questions between runs (about ±0.05 F1), so none beats the baseline.
- Answers from a wrong or mixed set: 1–3 of 83 in every run. Most missed asks answer from
  the right thread alone: the question's wording fits the thread it was written from.
- Specific vs consistent stays blurry (10–14 specific called consistent), with the same
  behaviour (answer from the right thread), so only "ask or not" is scored.
- Cost: ~$0.4–0.7 a run of 83 questions.

**Decisions:**
- Keep the baseline prompt, and stop tuning prompts on this set.
- The agent does the check itself on its final top 5, **in full** (cut at 12,000
  characters, like the answer step). The documents are already in its context, so the check
  is one more turn on a cached prefix: little extra cost, and no cut that could drop the
  detail that separates two cases.
- The agent runs on Gemini Flash (`agent_llm.py`; `--model deepseek` reruns the
  earlier setup). The check costs ~$0.005 a question. Earlier agent numbers are DeepSeek runs.

## Result (20 Nov): the answer mentions similar cases, with citations
Code: `exploration/slack_extraction/answer_check.py`. Gemini answers each of the 83
questions from the right thread and the other shown documents (shuffled), and a second
call checks where the right thread's answer is.

- **Answer rule** (`src/train_rag/prompts.yaml`: document_answer, agent, snippet agent and
  keyword agent instructions): when several similar documents answer fully or in part, the
  answer gives the best-fitting one as the main answer and mentions the others briefly.
- **Citations** (document_answer only): the model cites `[n]` inline after every fact,
  including each mentioned case. `link_citations` in `pipeline_answer.py` turns these into
  `[doc_id]`, so the app can open the document on click.

| | Rule only | Rule + citations |
|---|---|---|
| Right answer in the response | 95.2% | **96.4%** |
| as the main answer / only as a mentioned case | 83.1% / 12.0% | 84.3% / 12.0% |
| ambiguous questions: only as a mentioned case | 31.0% | 27.6% |
| Right thread cited inline | — | 97.6% |
| Fact lines with a citation | — | 612 of 623 (the rest are headings) |
| Mentions other cases | 81% | 90% |

Cost: $0.81 and $0.85 a run. Limits: the right thread is always among the documents (this
measures the answer step only), and the answers list up to 5 other cases at full length.

## Next
1. Cap the mentions: at most 2 other cases, one line each, only when their answer differs
   (rerun answer_check.py, ~$0.80).
2. The benchmark judge counts an answer as right when one of its mentioned cases matches
   (`judge_instructions`, changed 20 Nov): decide whether to keep it for the final run.
3. Put the check and the answer rule into the Gemini agent, then the end-to-end evaluation.
