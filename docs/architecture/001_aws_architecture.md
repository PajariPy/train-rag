# train-rag on AWS (draft 1)

*23–24 Nov 2026 · AWS architecture research · prices come in the next round*

Two parts share one database and three SageMaker model endpoints. A nightly Fargate task pulls only what changed in the nine sources, sends it to the SLM and the embedder, writes to Postgres and stops. An always-on query service behind a load balancer runs the pipeline, calls the embedder and the reranker, and streams the answer.

![[aws_architecture.png]]

Icons: AWS Architecture Icons. The diagram only shows services and how they connect; an arrow with two heads is a request and its response. What each service does is in the table below. The same diagram as a vector file, `aws_architecture.svg`, and as an editable Excalidraw file, `aws_architecture.excalidraw`. All of them are generated from `aws_architecture.json` by the `doc-diagram` skill (`python ~/.claude/skills/doc-diagram/scripts/diagram.py aws_architecture.json --png`); edits made by hand in Excalidraw are lost if it runs again.

## Services

| Service | Role | What it does |
|---|---|---|
| EventBridge Scheduler | Nightly trigger | Starts the ingestion workflow at 02:00. Weekly is the same rule with another cron expression. |
| AWS Step Functions | Ingestion workflow | Runs the Fargate task, retries or alerts on failure, then restarts the query service so it loads the new BM25 index (a control call, not drawn). |
| AWS Fargate | Ingestion task | An ECS task on Fargate: Step Functions starts it, it runs once and stops. One CPU container: fetches what changed, sends new documents to the SLM and gte endpoints, writes Postgres, rebuilds BM25 into S3, moves the watermarks and stops. |
| NAT gateway | Outbound internet | Lets tasks in private subnets call the source APIs, and Gemini if we keep it. |
| Data sources | Jira, Slack, Fireflies … | The nine SaaS APIs. Only changes since the watermark are fetched (table below). |
| SageMaker async endpoint | SLM field extractor | Qwen3.5-2B + LoRA on a GPU instance. Scales to zero between runs and wakes up when ingestion sends work; requests queue meanwhile. |
| SageMaker real-time endpoint | gte embedder | The fine-tuned gte on GPU. Query vectors for every question, document vectors for ingestion. Always on. |
| SageMaker real-time endpoint | gte reranker | gte-reranker-modernbert on GPU; scores ~100 documents per question. Always on. |
| Amazon S3 | Raw changes, BM25 index | Keeps each night's raw changes for replay, and the BM25 index files that ingestion writes and the query tasks load at start. |
| PostgreSQL (RDS or EC2) | Index database | Documents, vectors, extracted fields, watermarks and content hashes. RDS or EC2 is still open (see below). |
| Application Load Balancer | Streaming endpoint | The HTTPS entry point. Streams answers as server-sent events and spreads requests over the query tasks; its request count drives scaling. |
| AWS Fargate | Query service | An ECS service on Fargate: always keeps N tasks running behind the ALB. The pipeline without the models: BM25 from the loaded index, vector search in Postgres, calls to the embedder and reranker endpoints, the field agent, the cited answer. CPU only. |
| Amazon Bedrock or Gemini | Answer LLM | The model behind the field agent and the answer. Bedrock stays inside AWS; Gemini goes out through the NAT gateway. |
| Amazon ECR | Container images | The ingestion and query images, and our own model containers if the stock SageMaker ones don't fit. |
| Amazon CloudWatch | Logs, metrics, alarms | Logs of every service, endpoint invocation metrics, the in-flight-requests metric for scaling, alarms on failed runs. |
| AWS Secrets Manager | API keys | Source API tokens, the Gemini key and the database password. |

## Ingestion, step by step

1. **EventBridge Scheduler** starts a Step Functions run at 02:00 (or weekly; it's one cron expression).
2. Step Functions runs **one Fargate task**. It reads each source's watermark and the content hashes from Postgres, asks the source APIs for what changed, and keeps the raw changes in S3.
3. Only new or changed documents go to the **SLM async endpoint** for fields and to the **gte endpoint** for vectors. No reranker is needed here. The task waits for the SLM results.
4. It adds, updates and deletes rows in Postgres, rebuilds the BM25 index into S3, and moves the watermarks forward.
5. Step Functions then rolls the query service so it loads the new BM25 index. On failure it alerts, the watermarks stay where they were, and the next run (or a replay from S3) retries.

The models now run on GPU, so the CPU worry is gone. The **first load** of 500k documents and **full re-runs** after an SLM retrain go through the same async endpoint, scaled out to several instances for that run (or as a SageMaker batch transform job). Before relying on this, measure documents changed per night (our dataset is a fixed snapshot, so there is no real number yet) and the endpoint's cold start, which adds minutes to each nightly run.

### Finding only new, changed and deleted data

Each source keeps a watermark: the time of the last good run, minus a small overlap. Most APIs can filter on "updated since"; deletions are the hard part.

| Source | Changed since | Deletions |
|---|---|---|
| Jira | JQL `updated >= "…"` | webhook or weekly id check |
| Confluence | CQL `lastmodified >= "…"` | weekly id check |
| Slack | `conversations.history` with `oldest`; replies via `conversations.replies` | edits and deletes need the Events API or a re-read window |
| Gmail | `history.list` from the last `historyId` | reported (messagesDeleted) |
| Google Drive | `changes.list` with a saved page token | reported (removed files) |
| GitHub | `since` on issues and comments | rare; weekly id check |
| Linear | GraphQL filter `updatedAt gt …` | archived items via `includeArchived` |
| HubSpot | CRM search on `hs_lastmodifieddate` | list with `archived=true` |
| Fireflies | GraphQL `transcripts(fromDate: …)` | weekly id check |

- A content hash per document skips anything that came back unchanged, so the SLM and the embedder only see real changes.
- The weekly id check lists every id at the source and deletes what's gone. It's cheap because it reads ids, not content.
- Check each API's exact filter before building; the table is a starting point.

## Query service and scaling

- **ECS service on Fargate, 1 vCPU and 2 GB per task,** behind an Application Load Balancer. The task holds no models, so it needs no GPU: it loads the BM25 index, searches Postgres, and calls the gte and reranker endpoints and the LLM. The app is async (FastAPI), so one task answers many questions at once while it waits on the other services.
- **Load BM25 memory-mapped** (`bm25s.BM25.load(path, mmap=True)`). The task downloads the index from S3 to its own disk at start. In the load test this used 550 MiB idle and 1.37 GB at full load instead of the whole 2 GB, at the same speed.
- **Two tasks, for availability, not capacity.** One task handled ~25 questions/s in the load test (below). The second keeps the service up during restarts and deploys.
- **Two things scale separately.** ECS tasks scale on a custom "questions in flight per task" metric (requests per task says little for an async app). The SageMaker endpoints scale on `InvocationsPerInstance`.
- **Keep one instance per real-time endpoint warm.** A new GPU instance takes minutes to start, so scaling covers daily peaks, not sudden spikes. Fargate query tasks start in under a minute.
- **Streaming** works through the ALB as server-sent events; raise the ALB idle timeout above the longest answer.
- **No Lambdas** on this path. Lambda runs one request per instance and bills the whole wait for the LLM and endpoints, while one async Fargate task serves ~100 questions at once. API Gateway HTTP APIs also can't stream.

### Load test (24 Nov)

Setup in `exploration/load_test/` (`bash exploration/load_test/run.sh`): an async FastAPI stand-in for the query service in Docker, limited to 1 CPU and 2 GB. BM25 is real (512k documents, called through a thread like the pipeline); the embedder (50 ms), Postgres (50 ms), reranker (300 ms), field-agent LLM call (1.5 s) and the streamed answer (first token after 0.8 s, then 3 s) are waits. Those waits are my assumptions, not measured. Real benchmark questions, closed loop, 30 s per step.

Total answer time in seconds, p50 / p95, in-memory index vs memory-mapped:

| Users at once | Questions/s | In memory | Memory-mapped | CPU (1 core) |
|---|---|---|---|---|
| 1 | 0.2 | 5.74 / 5.77 | 5.77 / 5.80 | 0.01 |
| 50 | 8.3 | 5.79 / 5.88 | 5.79 / 5.97 | 0.36 |
| 100 | 17.0 | 5.80 / 6.31 | 5.80 / 6.16 | 0.69 |
| 200 | 25 | 7.99 / 8.59 | 8.18 / 8.46 | 1.00 |
| 300 | 25 | 12.08 / 12.30 | 11.80 / 12.30 | 1.00 |

- The core runs out at ~25 questions/s (about 40 ms of CPU per question, mostly BM25). Up to ~100 questions at once the answer time doesn't move.
- BM25 searches run one at a time (a lock around the shared stemmer); past 100 users they queue, so that is the first thing to fix for more headroom.
- In memory, the container sat at its 2 GB limit (1,543 reclaim events, no crash). Memory-mapped: 550 MiB idle, 1.37 GB peak.
- With real traffic unknown, the open question is the time per question of the real pipeline, not the task size.

## What changed from the first sketch

- **RDS doesn't offer our index** (decide). We use pgvectorscale's StreamingDiskANN. RDS and Aurora support pgvector (HNSW, IVFFlat), but I found no sign of pgvectorscale; there is an open request on AWS re:Post. Options:
  - RDS with HNSW: managed, but re-check recall and give the instance RAM for the index.
  - Postgres on EC2 with the same `timescaledb-ha` image we run now: identical results, we run backups.
  - Timescale Cloud on AWS.
- **Our models go to SageMaker, not Bedrock.** Bedrock Custom Model Import doesn't accept embedding models, and the reranker isn't a text-generation model either. Its Qwen support lists Qwen2/2.5-VL and Qwen3, not Qwen3.5. SageMaker endpoints take any container. Checked 24 Nov: AWS's vLLM image `server-sagemaker-cuda-v2.8` runs vLLM 0.33.0, the same version we mine with, lists Qwen3.5-2B, and loads a model from S3; we serve the merged LoRA, which avoids an open vLLM bug where Qwen3.5 LoRA adapters are silently ignored. Hugging Face's TEI image (1.9.3) lists ModernBERT and gte-reranker-modernbert-base; loading our fine-tuned gte from S3 there is still to test.
- **GPU only where the models are.** With the models on SageMaker, ingestion and the query service are plain CPU containers on Fargate. The SLM endpoint scales to zero between nightly runs; the embedder and reranker stay on.
- **BM25 lives in files, not in the database.** bm25s saves its index to disk. Ingestion rebuilds it into S3; the query tasks load it memory-mapped at start, so each ingestion ends with a rolling restart. A real BM25 inside Postgres (ParadeDB `pg_search`, Timescale `pg_textsearch`, VectorChord-BM25) isn't available on RDS; if we pick Postgres on EC2 or Timescale, BM25 could move into the database and the rebuild and restart would go away.
- **LLM: a Bedrock model or Gemini** (decide). Today the agent and the answer call Gemini 3.8 Flash. Keeping it needs outbound internet through the NAT gateway and the key in Secrets Manager. A Bedrock model keeps all traffic and billing in AWS, but needs a new evaluation run.
- **ECR for images.** Elastic Container Registry holds our images, including any custom model containers.

## Open for the next round

- Database: RDS + HNSW, Postgres on EC2 with DiskANN, or Timescale Cloud.
- LLM: a Bedrock model or Gemini.
- Endpoint instance types from the load test, and whether gte and the reranker share one GPU instance (SageMaker inference components) to halve the always-on cost.
- Fargate size for the ingestion task (the query task is 1 vCPU / 2 GB, from the load test).
- Time per question of the real pipeline, to replace the waits assumed in the load test.
- Where BM25 lives: files (today) or a BM25 extension in Postgres; depends on the database choice.
- Nightly or weekly ingestion, and how stale answers may be.
- Prices: the always-on embedder and reranker endpoints, SLM endpoint hours per night, Fargate hours, the database, NAT gateway, LLM tokens.

## Reference: the Lambda + pgvector article

[Building a Production RAG Pipeline on AWS Lambda and pgvector](https://harun.dev/blog/building-a-production-rag-pipeline-on-aws-lambda-pgvector) (Harun R. Rayhan, 4 Apr 2026). Small SaaS knowledge base: Lambda + API Gateway, RDS t3.micro with pgvector HNSW, RDS Proxy, Bedrock Titan V2 + Claude 3 Haiku, ~$54/month at ~10k queries/day. What carries over to us:
- Chunking mattered more than the embedding model (~30% vs ~5%).
- Put RDS Proxy (or a pool) in front of Postgres when many tasks connect.
- Throttle bulk embedding calls; log retrieval separately from generation.
- Not covered there: hybrid search, reranking, VPC/NAT cost, measured recall, corpus size (likely small).

## Sources

- [pgvectorscale request on AWS re:Post](https://repost.aws/questions/QUEgTTEzDeQdCoo5UJIEGAYw/add-pgvectorscale-extension-to-amazon-rds-aurora-postgresql-for-diskann-hnsw-hybrid-vector-search)
- [RDS for PostgreSQL supports pgvector 0.8.0](https://aws.amazon.com/about-aws/whats-new/2024/11/amazon-rds-for-postgresql-pgvector-080)
- [Bedrock Custom Model Import](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html)
