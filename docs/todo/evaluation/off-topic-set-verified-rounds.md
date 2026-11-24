Build and keep the off-topic set the careful way: every candidate checked by a wide search and document ratings, rerunnable when the corpus grows. ^desc

## Parked (23 Nov)
Too expensive for the final evaluation (≈ $10–16, almost all ratings);
[[off-topic-set-build]] generates the set with one call and a hand review instead.
**Reopen if** the corpus grows enough that a reviewed set goes stale, the set has to be large,
or the hand review lets answerable questions through.

## Why
A search miss can look like "not in the corpus". And the corpus will grow: a question that is
off-topic today can become answerable tomorrow.

## How
1. **Rounds** until 50–100 are accepted: Gemini writes candidates → wide search (BM25 + gte +
   field search, top 50) → each document rated full / partial / none (the `ambiguity.py`
   rater) → accept only if all are none; else dismiss and add a line to the generator's
   instructions ("the corpus covers X").
2. **Store** each accepted question's check (documents, ratings, corpus size / import date) and
   the generator's accumulated instructions.
3. **Recheck command:** after new documents arrive, rerun steps 1's search and rating on the
   stored set; move questions that became answerable out, reported, not silently dropped;
   extend the set from the stored instructions.

## Cost (estimate, 23 Nov)
**≈ $10–16** (≈ 5,500–7,500 ratings × $0.0021); ≈ $5–7 rating the top 20 instead of 50. A
recheck costs the ratings of the stored set again.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 5. Card: [[Run the final evaluation]].
