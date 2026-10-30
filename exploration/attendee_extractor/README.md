# Attendee extractor

One-off helper for data exploration, kept outside `src`. A few Fireflies meetings have no
attendee list, and their speaker lines are in forms the parser does not recognize
(`Speaker 1 (Jasmine):`, `[S00:00] Ari (Redwood AE):`). An LLM (Gemini) lists the people
who speak in each of them, with their role when the text gives one.

The lists go to the Postgres table `inferred_attendees`. The Fireflies parser attaches them
as `attendees_inferred` when it finds no attendees itself:
`default_registry(inferred_attendees=load_inferred_attendees(session))`.

```bash
uv run python exploration/attendee_extractor/extract.py
```

It reads `GEMINI_API_KEY` from the repository's `.env` (git-ignored). Rerunning replaces
the lists it stored before.
