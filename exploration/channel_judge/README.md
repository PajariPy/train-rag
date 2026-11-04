# Channel judge

One-off helper for data exploration, kept outside `src`. About 3,400 Slack threads have
no channel in their title. Many look like short-lived group chats for one issue, too many
to review by hand, so an LLM judge (Gemini) reads each thread and says whether the text itself
reveals the channel.

Before trusting it on real threads, it is evaluated on 50 labelled threads:

- 25 where the channel is hidden in the text: a header line, an export banner, a `#name`
  remark inside a message, a plain-words mention ("the eng-releases channel"), or a bot
  notice about the thread.
- 25 that do not reveal their own channel: 15 mention no channel at all, and 10 mention a
  different one (a pin, a digest, a cross-post), which must still count as no channel.

All 50 are real threads whose channel was known from the title; the title is replaced
with a file name like the real channel-less threads.

```bash
uv run python exploration/channel_judge/build_dataset.py                   # data/dataset.jsonl
uv run python exploration/channel_judge/evaluate.py    # data/predictions.jsonl, data/report.md
```

`evaluate.py` reads `GEMINI_API_KEY` from the repository's `.env` (git-ignored).
