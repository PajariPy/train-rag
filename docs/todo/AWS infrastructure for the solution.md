Describe in the article how the whole solution would run on AWS: services, data flow and a rough monthly cost. Not deployed. ^desc

## Why
Everything so far runs on a laptop plus rented Vast.ai GPUs. A reader who wants to take the
approach to production needs to see where each part would live and what it would cost; the
article should show that without me paying for a deployment.

## How
Article only, no build. Map each part of the pipeline to a service:
- **documents and ingestion:** connectors (Slack, Confluence, Jira, Gmail, Fireflies, …) →
  S3 raw store; EventBridge schedule + SQS queue + Lambda/ECS workers for new and changed
  documents;
- **database and vectors:** Postgres with pgvector on RDS / Aurora (documents, chunks,
  embeddings, mined fields, field vectors), as today;
- **embedder and reranker** (fine-tuned gte, cross-encoder): a small GPU endpoint
  (SageMaker or ECS on g5/g6), or CPU if latency allows;
- **SLM field mining** (fine-tuned Qwen, vLLM): AWS Batch on spot GPUs, for the backfill and
  the daily new documents; the backfill cost from our run ($16 on 3 × RTX 4090) vs AWS prices;
- **agent and API:** ECS Fargate service (FastAPI, the agent loop, later LangGraph, see
  [[Port the prod path to LangGraph]]); the LLM through the Gemini API, or Bedrock as the
  AWS-native option;
- **secrets, access, monitoring:** Secrets Manager, IAM, per-user document access (ties to
  [[Add guardrails]] and [[Private and personal data questions]]), CloudWatch and traces
  ([[Add observability and RAG quality tracking]]);
- a diagram of the flow (ingest → mine → embed → search → answer);
- a rough monthly cost table per part, for the benchmark's corpus size (≈ 512k documents).

Free: writing only.
