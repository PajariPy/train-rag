# Meeting date judge

One-off helper for data exploration, kept outside `src`. About 1,500 Fireflies meetings
have no date the parser can find. Most still say a lot about when they happened (action
items due on given dates, "yesterday's" file names, "next Tuesday"), so an LLM judge
(Gemini) reads each meeting and estimates the date it was held.

Before trusting it on real meetings, it is evaluated on 70 meetings whose date is known.
The date is removed from each one first: header lines with a date and every line that
repeats it.

- 50 synthetic cases, from meetings with no other date left in the text:
  - 25 with a clue hidden in one of five ways: someone states today's date, a dated
    event "2 days ago", a deadline "one week from today", a "week of" banner, or a file
    name stamped with today's date.
  - 25 that do not reveal the date: 15 with no date at all, and 10 with a date that says
    nothing about when the meeting was held (an event years ago, an expiry years ahead).
- 20 real cases, from meetings that keep their own dates in the text (mostly action item
  due dates), to see how close the judge gets on clues as they occur in the data.

```bash
uv run python exploration/date_judge/build_dataset.py   # data/dataset.jsonl
uv run python exploration/date_judge/evaluate.py        # data/predictions.jsonl, data/report.md
```

Then every meeting without a date is judged in one Gemini batch job (half price, results
within 24 hours). The request and raw result files are large and git-ignored.

```bash
uv run python exploration/date_judge/run_batch.py submit    # data/batch_job.json
uv run python exploration/date_judge/run_batch.py status
uv run python exploration/date_judge/run_batch.py collect   # data/batch_predictions.jsonl, data/batch_report.md
uv run python exploration/date_judge/store_dates.py         # Postgres table inferred_dates
```

The stored dates reach the parser through `default_registry(load_inferred_dates(session))`:
a meeting without a header date gets `held_on_inferred` (date, precision, evidence, and
whether the evidence states the year), and is no longer flagged for the date.

The scripts read `GEMINI_API_KEY` from the repository's `.env` (git-ignored).
