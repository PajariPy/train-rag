# Renting GPUs on Vast.ai (for agents)

How to run GPU work (training, embedding the corpus) on rented machines. Everything here was
used on 12 November 2026 for the embedder fine-tune (article 006, `exploration/embedder_finetune/`).

## Rules

- **Ask the user before renting anything.** It costs money. Say the GPU, the price per hour,
  the expected hours and the total.
- **Check the credit first and while it runs** (`credit` in `vast/common.sh`, or
  `uvx vastai show user`). When the credit runs out, Vast stops the machine and everything on
  it is lost. Leave a margin: estimate, then add 30%.
- **Never print the API key** (`~/.config/vastai/vast_api_key`) or the keys in `.env`. The
  scripts send `WANDB_API_KEY` to the machine on ssh's stdin, never on a command line.
- **Always destroy with `-y`** (`uvx vastai destroy instance ID -y`). Without it the CLI waits
  for a confirmation and the machine keeps billing.
- **Don't leave a machine idle.** A watcher (a background loop) should wake you when a job
  ends. An idle 4-GPU machine costs ~$2.7/h.
- **At the end, check that nothing is running:** `uvx vastai show instances`.

## Access

| What | Where |
|---|---|
| CLI | `uvx vastai ...` (not installed in the project) |
| API key | `~/.config/vastai/vast_api_key` (600) |
| SSH key | `~/.ssh/vastai`; connect with `-i ~/.ssh/vastai -o IdentitiesOnly=yes` |
| W&B key | `WANDB_API_KEY` in `.env` (entity `rag-train`) |

## The scripts: `exploration/embedder_finetune/vast/`

One machine stays up for several jobs; its id, host and port are kept in `vast/.instance`
(or `vast/.instance-NAME` with `MACHINE=NAME`, so two rentals can run side by side).

| Script | What it does |
|---|---|
| `up.sh` | Rents the cheapest reliable offer, installs uv and the same package versions as the laptop, downloads gte, uploads `train.py` and the training data. Destroys the machine if setup fails. Options: `GPU=RTX_4090`, `NUM_GPUS=4`, `MACHINE=embed`, `FILTER='geolocation notin [CN]'`. |
| `run.sh NAME [args]` | Uploads the current `train.py` and queues a training run. Runs wait for each other on a lock file, so several can be queued. Returns at once. |
| `fetch.sh` | Shows each run's state and copies finished runs' logs and `result.json` to `runs/NAME`. `MODEL=1` also copies the model (300 MB for gte). |
| `embed.sh MODEL_DIR` | On a multi-GPU machine: uploads the corpus and the model, starts one `embed_corpus.py` per GPU, waits, and copies the vectors to `data/vectors/`. |
| `down.sh` | Fetches everything, then destroys the machine. |

Typical session:

```bash
cd exploration/embedder_finetune/vast
./up.sh                                           # ~5 min
./run.sh lr5e-6-e2 --lr 5e-6 --epochs 2 --mini-batch 2 --encode-batch 16
./fetch.sh                                        # any time
MODEL=1 ./fetch.sh                                # copy the trained model too
./down.sh                                         # always
```

## Lessons (what went wrong once)

- **Detach remote jobs properly.** `ssh host "nohup cmd &"` can keep ssh open forever. Write
  the job to a file on the machine, then `ssh -n host "setsid nohup bash job.sh > /dev/null
  2>&1 < /dev/null &"`, as `run.sh` and `embed.sh` do. With `pkill -f`, use a bracket pattern
  (`'[t]rain.py'`) so it doesn't kill its own ssh command.
- **GPU memory (24 GB, 4090), full-length documents (up to 8,192 tokens):** gte-modernbert
  fits 2 documents per forward pass (14 GB); 4 or 8 run out of memory. Qwen3-Embedding-0.6B
  fits 2 with `--grad-checkpointing` (7 GB). Try a tiny smoke run first (`--limit 32
  --max-steps 2 --dev-limit 10 --no-wandb`).
- **Disk:** a Qwen-size checkpoint with optimizer state is ~7 GB. A full disk killed a run
  at 60%. `up.sh` asks for 60 GB and `train.py` keeps one checkpoint.
- **Hosts in China:** Hugging Face is blocked there; filter them out with
  `FILTER='geolocation notin [CN]'` when the machine has to download a model.
- **Speed:** the first boot of a template can take minutes; `up.sh` uses the "PyTorch (Vast)"
  template, which is cached on many hosts. Copying ~800 MB back took ~10 minutes.
- **Costs seen** (12 Nov): 1 × RTX 4090 $0.37–0.39/h; 4 × RTX 4090 $1.9 (China) to $2.7/h
  (Hungary). gte fine-tune, 2 epochs: ~30 min of training + 2 × 4 min dev checks. Corpus
  re-embedding with gte on 4 × 4090: 21 min of compute, ~50 min of machine time with setup
  and copying, ~$2.30.
