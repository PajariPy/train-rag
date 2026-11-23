For the articles only (descriptive, no build): how the pipeline would handle messier
documents than the benchmark's: images and screenshots, tables and spreadsheets, fragmented
Slack dialogs.

## Why
The benchmark is clean (see [[Approach human-style data problems]]); readers need to know
what changes on real data.

## How
Per kind of mess: what breaks (parsing, chunking, mining, search), the strategy (OCR / vision
model, table-to-text, thread reassembly, context from neighbouring messages), and how to
test it.

## Result (22 Nov)
Draft approved: `docs/articles/008_strategy_for_messier_documents.md` (written on the
`Messier-data-stategy` branch). One common strategy (meaningful text → LLM labels → SLM, why
the SLM pays off, hallucinations and the two answer paths), then short notes for images,
tables (normalization of complex tables) and Slack (dependency graph over messages, linked
by metadata rules or an LLM, merged into one clean thread).

May need rework when it's folded into the full article. Follow-up:
[[Article - detect data drift for the SLM]].
