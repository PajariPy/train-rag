# Slack splitter

One-off helper for data exploration, kept outside `src`. The Slack parser splits a thread
at lines that start with a speaker. About 4,400 threads have messages it cannot split: several
messages on one line (`... verification. deploy-bot: canary paused ...`), messages separated
by an escaped line break, or lines that start with a timestamp (`(10:02) samir_infra: ...`).
Inside the same threads, code and log lines (`Host: ...`, `ERROR: ...`) are sometimes taken
for speakers. An LLM (Gemini) lists where each message starts: its speaker and the first
words of its text, quoted exactly. The thread is cut at those words, so the text stays
verbatim, and an answer whose quotes are not found is dropped.

Before running it on the data, it is evaluated on 50 threads whose messages are known:
clean threads changed the way the merged ones look (joined inline, by an escaped line
break, or with timestamps), and threads with text that must not be split (action items
naming owners, labels, config lines). The parser is scored too, as a baseline.

```bash
uv run python exploration/slack_splitter/build_dataset.py   # data/dataset.jsonl
uv run python exploration/slack_splitter/evaluate.py        # data/predictions.jsonl, data/report.md
```

Then every thread with merged messages is split in one Gemini batch job (half price):

```bash
uv run python exploration/slack_splitter/run_batch.py count     # threads and cost estimate
uv run python exploration/slack_splitter/run_batch.py submit    # data/batch_job.json
uv run python exploration/slack_splitter/run_batch.py status
uv run python exploration/slack_splitter/run_batch.py collect   # data/batch_splits.jsonl
uv run python exploration/slack_splitter/store_messages.py      # Postgres table inferred_messages
```

The stored splits reach the parser through
`default_registry(inferred_messages=load_inferred_messages(session))`; those threads get the
format `messages_inferred`.

The scripts read `GEMINI_API_KEY` from the repository's `.env` (git-ignored).
