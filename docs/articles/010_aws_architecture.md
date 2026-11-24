# Taking the pipeline to AWS

> [!note] Status
> Draft. Architecture and one load test; prices come in the next round. Working notes with
> every detail: `docs/architecture/001_aws_architecture.md`.

So far everything ran on my machine and on rented GPUs: Postgres in Docker, the models in
Python, an evaluation script calling Gemini. This article is the first design for running it
as a service on AWS: what goes where, why, and what one load test says about the size of it.

The system has two parts that share one database. A nightly job pulls only what changed in the
nine sources (Jira, Slack, Gmail, Fireflies and the rest), runs the SLM and the embedder on the
new documents, writes them to Postgres and stops. An always-on query service takes questions,
runs the pipeline and streams the answer back.

## The diagram

![[aws_architecture.png]]

The picture only shows services and how they connect; an arrow with two heads is a request and
its response. What each one does is in the table below. I tried drawing the steps inside each
service first, and the diagram grew large and still couldn't hold a proper description.

| Service | Role | What it does |
|---|---|---|
| EventBridge Scheduler | Nightly trigger | Starts the ingestion workflow at 02:00. Weekly is the same rule with another cron expression. |
| AWS Step Functions | Ingestion workflow | Runs the ingestion task, retries or alerts on failure, then restarts the query service so it loads the new BM25 index. |
| AWS Fargate | Ingestion task | One CPU container: fetches what changed, sends new documents to the SLM and gte endpoints, writes Postgres, rebuilds BM25 into S3, moves the watermarks and stops. |
| NAT gateway | Outbound internet | Lets the tasks in private subnets reach the source APIs, and Gemini if I keep it. |
| SageMaker async endpoint | SLM field extractor | Qwen3.5-2B with the merged LoRA on a GPU. Scales to zero between runs; requests queue while it wakes up. |
| SageMaker real-time endpoint | gte embedder | The fine-tuned gte. Query vectors for every question, document vectors for ingestion. Always on. |
| SageMaker real-time endpoint | gte reranker | Scores ~100 documents per question. Always on. |
| Amazon S3 | Raw changes, BM25 index | Each night's raw changes (for replay) and the BM25 index files. |
| PostgreSQL (RDS or EC2) | Index database | Documents, vectors, extracted fields, watermarks and content hashes. Which Postgres is still open. |
| Application Load Balancer | Streaming endpoint | The HTTPS entry point; streams answers as server-sent events. |
| AWS Fargate | Query service · 1 vCPU, 2 GB | The pipeline without the models: BM25, vector search, calls to the endpoints, the field agent, the cited answer. |
| Amazon Bedrock or Gemini | Answer LLM | The model behind the field agent and the answer. |
| ECR, CloudWatch, Secrets Manager | Shared | Container images; logs, metrics and alarms; API keys and the database password. |

## Ingestion: only what changed

Re-reading 500k documents every night would be slow and expensive, so each source keeps a
watermark: the time of the last good run, minus a small overlap. Most APIs can filter on
"updated since" (JQL `updated >=` in Jira, `history.list` in Gmail, `changes.list` in Drive,
`fromDate` in Fireflies). A content hash per document skips anything that came back unchanged,
so the SLM and the embedder only see real changes.

Deletions are the hard part. Gmail and Drive report them; for the rest, a weekly job lists
every id at the source and deletes what's gone. It reads ids, not content, so it's cheap.

The whole run is one Fargate task, started by Step Functions. Fargate has no time limit, and it
bills only the minutes the task runs. If a run fails, the watermarks stay where they were and
the next run, or a replay from the raw changes in S3, picks it up.

## Where the models run

My first idea was Amazon Bedrock for all three models. It doesn't work: Bedrock's Custom Model
Import doesn't take embedding models, the reranker isn't a text-generation model either, and its
Qwen support didn't list Qwen3.5.

SageMaker endpoints take any container, so the models go there:

- **The SLM** on an async endpoint with AWS's vLLM image. Its current version runs vLLM 0.33.0,
  the same version I mined the corpus with, and its changelog lists Qwen3.5-2B. I serve the
  merged LoRA rather than an adapter on top of the base model: vLLM has an open bug where
  Qwen3.5 LoRA adapters are silently ignored, so the base model would answer and nothing would
  fail. The async endpoint scales to zero between nights, and the first load of 500k documents
  goes through the same endpoint with more instances.
- **The embedder and the reranker** on real-time endpoints with Hugging Face's embeddings
  container (TEI), which lists ModernBERT and gte-reranker-modernbert-base. Whether it loads my
  fine-tuned gte from S3 is still to test.

With the models on SageMaker, both of my own containers need no GPU, so they run on Fargate.

## How big the query service has to be

The query service mostly waits: for the embedder, Postgres, the reranker and two LLM calls. The
only real CPU work is BM25. The code is already async (FastAPI), so while one question waits,
the same process works on others.

To size it, I load-tested a stand-in on one CPU and 2 GB, the smallest sensible Fargate task.
BM25 was real: the full index of 512k documents, called the same way as in the pipeline. The
other services were waits of the lengths I expect (50 ms embedder, 50 ms Postgres, 300 ms
reranker, 1.5 s field agent, then a streamed answer), about 5.8 seconds per question in total.
Those waits are my assumptions; I haven't timed the real pipeline on AWS yet. Each step kept a
fixed number of users asking real benchmark questions for 30 seconds.

I ran it twice: once with the BM25 index loaded into memory, once memory-mapped (the index stays
in a file and the operating system reads in only the parts a search touches).

Total answer time in seconds, median / 95th percentile:

| Users at once | Questions/s | Index in memory | Index memory-mapped | CPU used |
|---|---|---|---|---|
| 1 | 0.2 | 5.74 / 5.77 | 5.77 / 5.80 | 0.01 |
| 50 | 8.3 | 5.79 / 5.88 | 5.79 / 5.97 | 0.36 |
| 100 | 17.0 | 5.80 / 6.31 | 5.80 / 6.16 | 0.69 |
| 200 | 25 | 7.99 / 8.59 | 8.18 / 8.46 | 1.00 |
| 300 | 25 | 12.08 / 12.30 | 11.80 / 12.30 | 1.00 |

What I take from it:

- **One core handles about 25 questions per second.** Up to about 100 questions at once, the
  answer time doesn't move at all. Past that, the core is full and new questions wait in line.
  Each question costs about 40 ms of CPU, mostly BM25.
- **The index has to be memory-mapped.** Loaded into memory, the BM25 index alone filled the
  2 GB task, and the container ran at its limit the whole time. Memory-mapped, it used 550 MB
  idle and 1.37 GB at full load, and the answer times were the same.
- **25 questions per second is a lot.** At about 6 seconds per answer, that is around 150
  people waiting for an answer at the same moment. So I'll run two tasks, and the second one is
  only there to keep the service up while the first restarts.

The first thing to break under more load is BM25: searches run one at a time behind a lock
around the shared stemmer, so past 100 users they queue. That's an easy fix if it's ever needed.

## Why not Lambda

Lambda looks cheaper because it costs nothing when nobody asks. But a Lambda instance handles
one request at a time, and it bills the whole time the request takes, including the seconds it
spends waiting for the reranker and the LLM. An async Fargate task does that waiting for about
100 questions at once and costs the same whether it waits or not. Lambda would also have to
load the BM25 index on every cold start, and it can't stream through an API Gateway HTTP API.
For a service that mostly waits, a small always-on task fits better.

## What's still open

- **Which Postgres.** My vector index is pgvectorscale's StreamingDiskANN, which I found no sign
  of on RDS or Aurora. The options are RDS with HNSW (managed, but recall needs re-checking),
  Postgres on EC2 with the same image I run today, or Timescale Cloud.
- **Where BM25 lives.** Today it's a file the nightly job rebuilds, followed by a restart of the
  query service. A real BM25 inside Postgres (ParadeDB's `pg_search`, Timescale's
  `pg_textsearch`) would remove both, but none of them runs on RDS. So this choice follows
  the database one.
- **Bedrock or Gemini for the LLM.** Gemini 3.8 Flash is what I evaluated with; a Bedrock model
  would keep all traffic and billing in AWS but needs a new evaluation run.
- **The cost of the always-on endpoints.** The embedder and reranker endpoints are the biggest
  fixed cost. Putting both on one GPU instance would roughly halve it.
- **The real time per question,** to replace the waits I assumed in the load test.
- **Prices,** in the next round.

## Sources

- [Building a Production RAG Pipeline on AWS Lambda and pgvector](https://harun.dev/blog/building-a-production-rag-pipeline-on-aws-lambda-pgvector), the article I started from
- [pgvectorscale request on AWS re:Post](https://repost.aws/questions/QUEgTTEzDeQdCoo5UJIEGAYw/add-pgvectorscale-extension-to-amazon-rds-aurora-postgresql-for-diskann-hnsw-hybrid-vector-search)
- [Bedrock Custom Model Import](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html)
- [AWS vLLM container for SageMaker](https://aws.github.io/deep-learning-containers/vllm/deployment/sagemaker/) and its [changelog](https://aws.github.io/deep-learning-containers/vllm/changelog/)
- [vLLM issue #49354: Qwen3.5 LoRA adapters ignored](https://github.com/vllm-project/vllm/issues/49354)
- [Text Embeddings Inference: supported models](https://huggingface.co/docs/text-embeddings-inference/supported_models)
- [SageMaker asynchronous inference](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html)
