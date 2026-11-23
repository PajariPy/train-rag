"""The mining progress page, served by each rented machine on port 8000 (stdlib only).

`/status.json` describes this machine's part: phase, documents done per shard, speed, ETA,
GPU use and the cost so far. `/` is a page that polls this machine's status and its peers'
(PEERS in job.env, the other machines' public URLs), so any one URL shows all three parts.
Only these two paths are served: never the prompts or the outputs.

Usage (on the machine, started by mine_job.sh): python3 mine_status.py
"""

import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("/workspace/mine")
PORT = 8000
PROGRESS = re.compile(r"progress docs (\d+)/(\d+)(?: \((\d+) s\))?")


def read(name: str) -> str:
    """Read a small state file of the job, or an empty string."""
    path = ROOT / name
    return path.read_text().strip() if path.exists() else ""


def settings() -> dict[str, str]:
    """Read job.env (KEY=value lines)."""
    values = {}
    for line in read("job.env").splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def shard_progress(name: str, documents: int) -> dict[str, float]:
    """Documents done in one shard and the speed of its runs, from its log."""
    if (ROOT / "out" / f"{name}.done").exists():
        done = documents
    else:
        out = ROOT / "out" / f"{name}.jsonl"
        done = sum(1 for _ in out.open()) if out.exists() else 0

    mined, seconds = 0, 0
    base: int | None = None
    last = last_seconds = 0
    log = ROOT / "logs" / f"{name}.log"
    if log.exists():
        for match in PROGRESS.finditer(log.read_text(errors="replace")):
            count, spent = int(match.group(1)), match.group(3)
            if spent is None:  # a (re)start: the count it resumed from
                if base is not None:
                    mined += last - base
                    seconds += last_seconds
                base, last, last_seconds = count, count, 0
            elif base is not None:
                last, last_seconds = count, int(spent)
        if base is not None:
            mined += last - base
            seconds += last_seconds

    return {"documents": documents, "done": done, "mined": mined, "seconds": seconds}


def gpu() -> str:
    """GPU use and memory from nvidia-smi, or an empty string."""
    try:
        line = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        use, used, total = (part.strip() for part in line.split(","))
        return f"{use}% · {int(used) / 1024:.1f}/{int(total) / 1024:.0f} GB"
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return ""


def status() -> dict[str, object]:
    """This machine's part, as the page shows it."""
    config = settings()
    part = json.loads(read("part.json") or "{}")
    shards = {
        name.removesuffix(".jsonl.gz"): info["documents"]
        for name, info in part.get("shards", {}).items()
    }
    rows = [{"name": name, **shard_progress(name, size)} for name, size in sorted(shards.items())]
    done = sum(row["done"] for row in rows)
    total = sum(row["documents"] for row in rows)
    mined = sum(row["mined"] for row in rows)
    seconds = sum(row["seconds"] for row in rows)
    rate = mined / seconds if seconds else 0.0
    started = int(read("started") or time.time())
    end = int(read("finished") or time.time())
    hours = (time.time() - started) / 3600
    price = float(config.get("PRICE") or 0)

    return {
        "part": config.get("PART", "?"),
        "phase": read("phase") or "starting",
        "error": read("error"),
        "done": done,
        "total": total,
        "docs_per_second": round(rate, 2),
        "eta_seconds": round((total - done) / rate) if rate and done < total else None,
        "elapsed_seconds": round(end - started),
        "gpu": gpu(),
        "price_per_hour": price,
        "cost": round(hours * price, 2),
        "shards": rows,
        "peers": [peer for peer in config.get("PEERS", "").split() if peer],
        "updated": round(time.time()),
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corpus mining</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1f; --muted:#6b6b70; --bar:#e6e6e3;
  --fill:#2f6fde; --ok:#1f8a4c; --bad:#c2372f; --line:#e2e2df; }
@media (prefers-color-scheme: dark) { :root { --bg:#141416; --card:#1e1e21; --ink:#ededf0;
  --muted:#9a9aa1; --bar:#2c2c30; --fill:#5b8ff0; --ok:#3fbf75; --bad:#ef6a60; --line:#2c2c30; } }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--ink);
  font:15px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width:760px; margin:0 auto; }
h1 { font-size:20px; margin:0 0 4px; } .muted { color:var(--muted); font-size:13px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; margin:12px 0; }
.head { display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap; }
.bar { height:10px; background:var(--bar); border-radius:5px; overflow:hidden; margin:8px 0; }
.bar > div { height:100%; background:var(--fill); }
.finished .bar > div { background:var(--ok); } .err { color:var(--bad); }
.shards { display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr));
  gap:6px; margin-top:8px; font-size:13px; }
.shards .bar { height:5px; margin:3px 0 0; }
</style></head><body><main>
<h1>Mining the corpus with the SLM</h1>
<div class="muted" id="total">loading…</div>
<div id="parts"></div>
</main><script>
const fmt = n => n.toLocaleString("en-US");
const dur = s => s == null ? "–" : (s >= 3600 ? Math.floor(s / 3600) + " h " : "") +
  Math.round((s % 3600) / 60) + " min";
async function get(url) {
  try { const r = await fetch(url + "/status.json", {cache: "no-store"}); return await r.json(); }
  catch (e) { return {url, unreachable: true}; }
}
function card(s) {
  if (s.unreachable) return `<div class="card"><b>${s.url}</b>
    <div class="muted">unreachable: stopped, destroyed or rebooting</div></div>`;
  const pct = s.total ? 100 * s.done / s.total : 0;
  const shards = s.shards.map(x => `<div>${x.name} · ${fmt(x.done)}/${fmt(x.documents)}
    <div class="bar"><div style="width:${100 * x.done / x.documents}%"></div></div>
    </div>`).join("");
  return `<div class="card ${s.phase === "finished" ? "finished" : ""}">
    <div class="head"><b>Part ${s.part} · ${s.phase}</b>
    <span class="muted">${fmt(s.done)} / ${fmt(s.total)} documents · ${pct.toFixed(1)}%</span></div>
    <div class="bar"><div style="width:${pct}%"></div></div>
    <div class="muted">${s.docs_per_second} docs/s · ETA ${dur(s.eta_seconds)} ·
    running ${dur(s.elapsed_seconds)} · $${s.cost} at $${s.price_per_hour}/h ·
    GPU ${s.gpu || "–"}</div>
    ${s.error ? `<div class="err">${s.error}</div>` : ""}
    <div class="shards">${shards}</div></div>`;
}
async function refresh() {
  const own = await get("");
  const peers = own.unreachable ? [] : await Promise.all(own.peers.map(get));
  const all = [own, ...peers].sort((a, b) => String(a.part).localeCompare(String(b.part)));
  document.getElementById("parts").innerHTML = all.map(card).join("");
  const live = all.filter(s => !s.unreachable);
  const done = live.reduce((n, s) => n + s.done, 0);
  const total = live.reduce((n, s) => n + s.total, 0);
  const eta = Math.max(...live.map(s => s.eta_seconds || 0));
  const cost = live.reduce((n, s) => n + s.cost, 0);
  document.getElementById("total").textContent = `${fmt(done)} / ${fmt(total)} documents
    (${total ? (100 * done / total).toFixed(1) : 0}%) · ETA ${dur(eta)} ·
    $${cost.toFixed(2)} so far ·
    updated ${new Date().toLocaleTimeString()}`;
}
refresh(); setInterval(refresh, 30000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    """Serves the page and the status, nothing else."""

    def do_GET(self) -> None:  # noqa: N802 (http.server's name)
        """Answer `/` and `/status.json`."""
        if self.path in ("/", "/index.html"):
            body, kind = PAGE.encode(), "text/html; charset=utf-8"
        elif self.path == "/status.json":
            body, kind = json.dumps(status()).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        """Keep the log quiet."""


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
