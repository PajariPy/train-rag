"""Load test for the stand-in query service: how many questions can one core take at once?

Closed loop: N users each ask a real benchmark question, read the streamed answer to the
end, and ask the next one. N steps up; each step runs WARMUP + DURATION seconds and counts
the questions started inside the measured window. The step stops the run when errors pass
1% or the 95th percentile answer time grows past 1.5x the first step's median.

The users run in several processes so the client is not the bottleneck. The server's CPU and
memory are read from its container's cgroup files.

Usage (from the project root, server already running; run.sh does all of this):
    uv run python exploration/load_test/loadgen.py --container loadtest --out results.json
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import random
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

QUESTIONS = Path("data/enterprise_rag_bench/data/questions/test.parquet")


def load_questions() -> list[str]:
    """Read the benchmark questions."""
    return pd.read_parquet(QUESTIONS)["question"].tolist()


async def users(url: str, n: int, stop_at: float, questions: list[str], seed: int) -> list[tuple]:
    """Run n users until stop_at; return (start, ttft, total, ok) for every question."""
    rng = random.Random(seed)
    records: list[tuple] = []
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), limits=limits) as client:

        async def user() -> None:
            while time.time() < stop_at:
                start = time.time()
                ttft, ok = None, False
                try:
                    async with client.stream("POST", f"{url}/ask", json={"question": rng.choice(questions)}) as r:
                        async for line in r.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            if ttft is None:
                                ttft = time.time() - start
                            if '"done"' in line:
                                ok = r.status_code == 200
                except (httpx.HTTPError, OSError):
                    ok = False
                records.append((start, ttft, time.time() - start, ok))
                if not ok:
                    await asyncio.sleep(0.5)

        await asyncio.gather(*(user() for _ in range(n)))
    return records


def run_users(args: tuple) -> list[tuple]:
    """Process entry point: run one share of the users."""
    url, n, stop_at, questions, seed = args
    return asyncio.run(users(url, n, stop_at, questions, seed))


def cgroup_dir(container: str) -> Path:
    """Find the container's cgroup v2 folder."""
    cid = subprocess.run(["docker", "inspect", "-f", "{{.Id}}", container], capture_output=True, text=True,
                         check=True).stdout.strip()
    for path in (Path(f"/sys/fs/cgroup/system.slice/docker-{cid}.scope"), Path(f"/sys/fs/cgroup/docker/{cid}")):
        if path.exists():
            return path
    raise SystemExit(f"cgroup folder of {container} not found")


def cpu_usec(cg: Path) -> int:
    """Total CPU time the container used so far."""
    for line in (cg / "cpu.stat").read_text().splitlines():
        if line.startswith("usage_usec"):
            return int(line.split()[1])
    return 0


def pct(values: list[float], q: float) -> float:
    """Percentile q (0-100) of values."""
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]


def main() -> None:
    """Step the number of users up and print one row per step."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8800")
    ap.add_argument("--container", required=True)
    ap.add_argument("--levels", default="1,10,25,50,100,200,300,400,600,800,1000")
    ap.add_argument("--warmup", type=float, default=8)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    questions = load_questions()
    cg = cgroup_dir(args.container)
    health = httpx.get(f"{args.url}/health", timeout=10).json()
    print(f"server: {health}, idle memory {int((cg / 'memory.current').read_text()) / 2**20:.0f} MiB")
    header = (f"{'users':>6} {'q/s':>6} {'ttft p50':>9} {'ttft p95':>9} {'total p50':>10} {'total p95':>10} "
              f"{'errors':>7} {'cpu':>5} {'mem MiB':>8} {'bm25 ms':>8} {'lag max ms':>10}")
    print(header)
    rows, base = [], None
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.procs, mp_context=ctx) as pool:
        for n in [int(x) for x in args.levels.split(",")]:
            httpx.get(f"{args.url}/stats", timeout=10)  # reset
            t0 = time.time()
            window = (t0 + args.warmup, t0 + args.warmup + args.duration)
            procs = min(args.procs, n)
            shares = [n // procs + (1 if i < n % procs else 0) for i in range(procs)]
            futures = [pool.submit(run_users, (args.url, s, window[1], questions, i * 7919 + n))
                       for i, s in enumerate(shares)]
            # sample the container while the users run
            mem_max, cpu_start, cpu_t = 0, None, None
            while not all(f.done() for f in futures):
                now = time.time()
                mem_max = max(mem_max, int((cg / "memory.current").read_text()))
                if cpu_start is None and now >= window[0]:
                    cpu_start, cpu_t = cpu_usec(cg), now
                if cpu_start is not None and now <= window[1]:
                    cpu_end, cpu_end_t = cpu_usec(cg), now
                time.sleep(0.5)
            cores = (cpu_end - cpu_start) / 1e6 / (cpu_end_t - cpu_t) if cpu_start is not None else float("nan")
            records = [r for f in futures for r in f.result()]
            server = httpx.get(f"{args.url}/stats", timeout=10).json()
            measured = [r for r in records if window[0] <= r[0] < window[1]]
            ok = [r for r in measured if r[3]]
            errors = len(measured) - len(ok)
            ttft = [r[1] for r in ok if r[1] is not None]
            total = [r[2] for r in ok]
            row = {
                "users": n, "questions_per_s": len(ok) / args.duration,
                "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95),
                "total_p50": pct(total, 50), "total_p95": pct(total, 95),
                "errors": errors, "error_rate": errors / max(1, len(measured)),
                "cpu_cores": cores, "memory_max_mib": mem_max / 2**20,
                "bm25_avg_ms": server["bm25_avg_ms"], "loop_lag_max_ms": server["loop_lag_max_ms"],
            }
            rows.append(row)
            print(f"{n:>6} {row['questions_per_s']:>6.1f} {row['ttft_p50']:>9.2f} {row['ttft_p95']:>9.2f} "
                  f"{row['total_p50']:>10.2f} {row['total_p95']:>10.2f} {errors:>7} {cores:>5.2f} "
                  f"{row['memory_max_mib']:>8.0f} {row['bm25_avg_ms']:>8.1f} {row['loop_lag_max_ms']:>10.0f}",
                  flush=True)
            if base is None:
                base = row["total_p50"]
            if row["error_rate"] > 0.01 or row["total_p95"] > 1.5 * base:
                print(f"stopped: {'errors over 1%' if row['error_rate'] > 0.01 else 'p95 answer time over 1.5x baseline'}")
                break
            time.sleep(3)
    oom = (cg / "memory.events").read_text()
    args.out.write_text(json.dumps({"server": health, "rows": rows, "memory_events": oom}, indent=1))
    print(f"memory events: {' '.join(oom.split())}")


if __name__ == "__main__":
    main()
