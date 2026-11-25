# What the models cost on AWS

> [!note] Status
> Draft, kept separate for now. Benchmarks of the three models and a first cost projection
> for the AWS architecture (article 010). Not merged into 010 until the reranker cost is
> investigated ([[Investigate the reranker cost and how to reduce it]]).

The architecture in article 010 sizes the query service from a load test: one small Fargate
task handles about **25 questions per second**. This article uses the same rate as the target for
the three models behind it: the embedder, the reranker and, for ingestion, the SLM.

## How the numbers were measured

- **Local GPU:** NVIDIA RTX 2000 Ada Generation Laptop GPU, 8 GB. It's the same architecture
  (Ada) as the NVIDIA L4 in SageMaker's `ml.g6` instances, but a smaller chip.
- **Models and settings as in the pipeline:**
  - the fine-tuned gte-modernbert embedder and gte-reranker-modernbert-base;
  - float16, run through sentence-transformers;
  - the reranker sees documents shortest first.
- **Data:** real benchmark questions (median 38 tokens) and real documents (median 1,139
  tokens; 90% under 2,187).
- **SLM:** no new run. The number comes from mining the whole corpus with vLLM 0.33.0 on a
  rented RTX 4090: about 5.7 documents per second.
- **Scripts and raw results:** `exploration/gpu_bench/` (`bench.py`, `bench_extra.py`).

## Measured on the local GPU

| Model | Setting | Result |
|---|---|---|
| Embedder, questions | one question at a time | 34 questions/s (29 ms each) |
| Embedder, questions | batches of 8 / 32 | 200 / 654 questions/s |
| Embedder, questions | on the CPU (Intel i7-13800H, 1–4 threads) | 0.75–1.8 s per question |
| Embedder, documents | whole documents, 4 per batch | 19.6 documents/s |
| Reranker | 100 documents × 1,024 tokens | 2.65 s per question |
| Reranker | 50 documents × 1,024 tokens | 1.37 s per question |
| Reranker | 100 documents × 512 tokens | 1.18 s per question |
| Reranker | 50 documents × 512 tokens | 0.62 s per question |

Reranking time grows with the number of documents and with their length, and halving either
roughly halves the time. The 100 × 1,024 setting is what the field agent uses today; the plain
retrieval pipeline reranks 50.

## From the local GPU to SageMaker

The projection assumes an L4 or A10G GPU is **2–3× faster than the local GPU** for these
models. Two anchors support that range:

- The same document embedding ran about 5× faster on an RTX 4090 than on the local GPU.
- An L4 has about 121 TFLOPS of float16 tensor compute and an A10G about 70; both sit between
  the local GPU and a 4090.

Prices are US East on-demand list prices from third-party price lists. Billing is per hour
whether the endpoint is busy or not:
- `ml.g6.xlarge` (L4): about $1.13/h, or ~$820 a month always on;
- `ml.g5.xlarge` (A10G): about $1.41/h, or ~$1,030 a month always on.

The projection is a range, not a measurement. One short run on a real instance would replace it.

## Embedder: cheap

At 25 questions per second the embedder needs only a fraction of one GPU: the local GPU alone
embeds 200 questions per second in batches of 8. It can share a GPU with the reranker. The two
compete only when questions overlap, and then an embedding waits a few tens of milliseconds
behind a reranker batch. Together they use under 2 GB of memory.

On the CPU, as it runs today, it's too slow (up to 1.8 s per question). An optimised CPU
runtime such as ONNX might change that; it isn't tested.

## Reranker: the real cost

| Rerank setting | One L4 handles (projected) | L4 GPUs for 25 questions/s | Monthly cost (SageMaker) |
|---|---|---|---|
| 100 documents × 1,024 tokens | ~0.8–1.1 questions/s | ~25–30 | ~$21,000–25,000 |
| 50 documents × 512 tokens | ~3–5 questions/s | ~5–8 | ~$4,000–6,500 |

Two things follow:

- **The reranker decides the bill.** Everything else in the architecture costs a fraction of this.
- **It also adds latency:** about 1.1 s per answer at 100 × 1,024 and about 0.25–0.3 s at
  50 × 512. The load test in article 010 assumed 300 ms, which only holds for the lighter
  setting.

25 questions per second held all day would be about 2 million questions a day. One L4 at
50 × 512 answers about 4 questions per second, around 300,000 a day, which may already cover
real traffic. So the number of GPUs depends on the traffic the system will actually see, and on
how much reranking quality each setting keeps.

## Ingestion: cost per 1,000 documents

| Part | Speed (projected) | Cost per 1,000 documents |
|---|---|---|
| SLM on an A10G async endpoint (`ml.g5.xlarge`) | ~2–3 documents/s | ~$0.13–0.20 |
| SLM on an L4 async endpoint (`ml.g6.xlarge`) | ~1.5–2 documents/s | ~$0.16–0.21 |
| Document embedding (on the shared GPU) | ~40–60 documents/s | under $0.01 |
| Fargate ingestion task (waits for the SLM) | ~7 min per 1,000 documents | under $0.01 |
| **Total** | | **about $0.15–0.25** |

The SLM's text generation depends mostly on memory speed: about 1,008 GB/s on the RTX 4090,
~600 GB/s on the A10G and ~300 GB/s on the L4. That is why the measured 5.7 documents per
second drops to the speeds in the table.

On top of the per-document cost, the async endpoint starts from zero every night: about 5–10
minutes of GPU time, roughly $0.15. Examples:
- **5,000 changed documents a night:** about $1–1.50.
- **First load of 500k documents:** about $75–125, taking 2–3 days on one GPU or ~10 hours on
  five. For comparison, the same run cost $16 on rented RTX 4090s.

## Open questions

- **Rerank depth and length:** how much quality do 50 documents and 512 tokens lose compared
  with 100 × 1,024?
- **Expected traffic:** questions per day and the peak. This decides between one GPU and
  dozens.
- **Ways to cut the reranker cost:** working hours only, plain EC2 instead of SageMaker, cheaper
  GPUs, a smaller or distilled reranker, or reranking only some questions. See
  [[Investigate the reranker cost and how to reduce it]].
- **A real-instance check:** one hour on an L4 to replace the 2–3× projection.

## Sources

- [NVIDIA L4](https://www.nvidia.com/en-zz/data-center/l4/) and [L4 vs A10G specifications](https://getdeploying.com/gpus/nvidia-a10g-vs-nvidia-l4)
- [RTX 2000 Ada Laptop GPU specifications](https://www.notebookcheck.net:443/NVIDIA-RTX-2000-Ada-Generation-Laptop-GPU-Benchmarks-and-Specs.744893.0.html)
- SageMaker instance prices: [CloudRoute](https://www.cloudroutehq.com/aws/sage-maker-pricing), [Holori](https://calculator.holori.com/aws/sagemaker/ml.g6.xlarge), [cloudburn](https://cloudburn.io/blog/amazon-sagemaker-ai-pricing)
