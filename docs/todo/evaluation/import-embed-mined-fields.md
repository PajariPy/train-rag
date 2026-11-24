Load the SLM's fields for all 511,958 documents into `document_fields` and embed their
profiles, so field search covers the whole corpus. ^desc

## Why
Today field search sees only the ~26k DeepSeek-labelled documents; the final evaluation must
search the whole corpus. The mining is done ([[Mine the corpus with the SLM]]).

## How
- Import the fields as mined; 3,151 long documents were read cut ([[fix-cut-long-documents]], parked).
- Start the local Postgres (port 5434).
- `uv run train-rag import-fields --file exploration/slm_fields/runs/mine-r16-e2/labels.jsonl --extractor slm-fields-r16-e2`
  (field search prefers `slm-` rows, no code change).
- `uv run train-rag embed-fields`: subject, problem and names profiles with the fine-tuned
  gte (≈ 1.5M strings, local GPU, hours). Wire the progress page before the run.
- The field index needs a faster load at 512k documents (open point from the pipeline
  update).
- Check: rows per source; `exploration/slm_fields/field_search_compare.py` on the full pool
  as a smoke check; a few documents per source read by hand.

## Cost
Free (local).

**Done when:** every document has a `slm-fields-r16-e2` row with profile vectors, and the
field index loads all of them.

## Result (23 Nov)
- Imported 511,962 `slm-fields-r16-e2` rows; embedded the three profiles in 3 h 14 min (local GPU).
- Loading all profiles into memory crashed the PC (well over 30 GB), so field search now runs in
  Postgres: `train-rag index-fields` copies the preferred row of each document with its vectors into
  `field_search_vectors` and builds one StreamingDiskANN cosine index per profile (≈ 330 MB each,
  table ≈ 9.5 GB). `StoredFieldIndex` searches there; nothing is loaded into memory.
- Check against an exact scan (20 semantic questions): overlap @10 = 0.985–1.0, top-1 identical,
  overlap @300 = 0.66–0.82 (only the tail differs); 253 ms median vs 1,967 ms exact.
- `all` profile: the inner-product index ranked summed vectors almost at random (0.7% overlap),
  so `all` merges each profile's nearest documents and scores them exactly.

Card: [[Run the final evaluation]].
