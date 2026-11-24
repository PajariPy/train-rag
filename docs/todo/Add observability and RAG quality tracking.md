Track the RAG in production so problems show up early: per-request traces, and quality
metrics from real chats. ^desc

## Why
Offline tests don't show drift: new documents, new kinds of questions, a slower model. We need
to see when retrieval or answers get worse, from the chats themselves.

## How
1. Traces per request: query, prepared query, searches, documents, scores, asks, answer,
   latency, cost.
2. Chat metrics: follow-up rate, rephrased questions, thumbs up / down, refusals, asks per
   chat, "not found" rate.
3. Alerts and a dashboard; a weekly sample judged offline.
