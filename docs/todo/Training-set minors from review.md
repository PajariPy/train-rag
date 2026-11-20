Small issues found in the review of the synthetic training set:

- Neighbours of evaluation documents come from the first chunk only.
- The score cache is keyed by `doc_id` and goes stale if questions are regenerated.
