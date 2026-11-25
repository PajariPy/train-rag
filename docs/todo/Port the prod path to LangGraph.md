Write the production path with LangGraph and LangChain chat models, when the prod code is
prepared: the field agent (general and field search, the check, asking the person) and the
cited answer. Nothing else. ^desc

## Why
The agent loop (`evaluation/field_agent/loop.py`) is already a hand-written state machine.
LangGraph gives it `interrupt()` and a checkpointer, so a real person can answer the agent's
questions between turns. Chat models give one layer for Gemini, DeepSeek and an online SLM.
Porting all of `src/` (the older agents, the judge, the labelling scripts) would be a lot of
work for code that never ships.

## How
- Scope: `field_agent/` (`loop.py`, `chat.py`, `searches.py`, `tools.py`, `system.py`),
  `pipeline_answer.py`, `query_rewrite.py`, and `person.py` (it becomes `interrupt()`).
- The agent becomes a `StateGraph`: agent → searches / ask → finish → rerank → check → ask again
  or answer.
- LLM calls go through `ChatGoogleGenerativeAI`. Token and cost counting becomes a callback that
  reads `usage_metadata`.
- Main risk, to test first with a spike: every turn must send all tools and limit the callable
  ones with `allowed_function_names` (mode ANY), and Gemini's thought signatures must
  round-trip. Without both, the implicit cache and the forced tool call are lost.
- Keep our vector store and models. No `PGVector` and no LangChain `Document`: graph tools call
  `RetrievalPipeline` and `FieldIndex` directly.
- The SLM stays on offline vLLM for mining. Online query-field extraction (`ChatOpenAI` on
  vLLM's server) would be one more node, only if that gets decided.
- Leave out the evaluation harness, the judge, the older agents and the exploration scripts.
  They stay on `google.genai`.
- Same accuracy before deploying. The prod agent gets a thin `RagSystem` adapter, so the same
  runner, judge and questions score it.
  1. Free: a test compares the exact requests (prompt, tools, `allowed_function_names`,
     thinking level) of the old and the prod code.
  2. Paid, ask first: the prod run's scores fall within the run-to-run noise of the recorded
     `field-agent-clarify` run.
- Example of LangGraph in this repo: `exploration/training_questions/generate.py`.

## Parked (22 Nov)
The first design covered every agent and all the LLM calls, and that was too much. Reopen when
the prod code is prepared. Until then, keep new prod-path code from adding more direct
`google.genai` coupling where that's easy.
