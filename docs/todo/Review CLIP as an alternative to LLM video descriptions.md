Review CLIP-style image–text embedding models as an alternative to having an LLM write text descriptions of videos. ^desc

## Why
The plan for messier documents ([[Article - strategy for messier documents]]) turns non-text content into text that search can match. For videos, the obvious way is an LLM (or vision LLM) describing each one. That is slow and costly per video. A CLIP-style model embeds video frames and the question in the same vector space, so a question could find a video directly, with no description written first.

## How
- Compare the options on a few sample videos:
  - a vision LLM writing a description, then the usual text search;
  - CLIP-style embeddings of sampled frames (CLIP, SigLIP);
  - video-level models.
- Questions to answer:
  - recall on questions about what happens in the video;
  - how speech in the video is covered (a transcript is probably still needed);
  - cost and time per video;
  - how frame vectors fit next to the text vectors in Postgres.
- Decide whether to use one approach, the other, or both: CLIP for finding the video, a description or transcript for the answer.
