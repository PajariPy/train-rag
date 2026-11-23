Describe in the article how the assistant should handle questions about personal or private
data the connected tools don't hold (salary, vacation days, HR records, someone's health).

## Why
EnterpriseRAG-Bench has no such data, so the off-topic set can't test it (see
[[Detect out-of-scope questions]]). A real deployment gets these questions on day one, and a
look-alike document (an HR policy, a payroll thread) can produce a confident wrong answer
or leak someone else's data.

## How
Article only, no build:
- the kinds: the asker's own records, other people's records, sensitive categories;
- refuse with "not in our documents" vs route to the owning system (HR tool, payroll);
- access rules: answers only from documents the asker may see (ties to [[Add guardrails]]);
- how it would be tested: a small hand-written set per kind.

Design context: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`.
