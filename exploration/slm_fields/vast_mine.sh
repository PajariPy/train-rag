#!/usr/bin/env bash
# Mine the corpus with the field SLM on rented RTX 4090s, one machine per part, with no laptop
# needed while they run. Ask the user before `launch`: it costs money.
#
#   vast_mine.sh launch   rent one machine per part, upload its shards, the adapter and the
#                         scripts, start mine_job.sh detached, print the progress page URLs, exit
#   vast_mine.sh fetch    copy the outputs back (any time); a machine whose part is complete and
#                         copied is destroyed; a stopped one is started again to copy it
#   vast_mine.sh status   each machine's state, its page's summary, the credit
#   vast_mine.sh watch    while the laptop is on: copy the live counts to the local progress page
#
# Inputs: runs/mine-r16-e2/part{1,2,3}/shard_NN.jsonl.gz and manifest.json
# (build_corpus_prompts.py), the adapter in runs/fields-r16-e2/adapter.
# Outputs: runs/mine-r16-e2/out/partN/shard_NN.jsonl (+ .done when complete).
# State: runs/mine-r16-e2/state/partN.env per live machine.
#
# Each machine stops itself (GPU billing ends, disk kept) when its job fails, when nobody
# fetches within GRACE_HOURS of finishing, or after MAX_HOURS (mine_job.sh).
#
# Options: PARTS="1 2 3"; SMOKE=1 (one machine, 2,000 documents of part 1, GRACE_HOURS=0 so it
# tests the self-stop); GRACE_HOURS=3; MAX_HOURS=16; SKIP="offer ids".
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$HERE/../.." && pwd)
RUN=$HERE/runs/mine-r16-e2
ADAPTER=$HERE/runs/fields-r16-e2/adapter
STATE=$RUN/state
TEMPLATE=305ac3ffd3e42e0d9ad1f4ae14729ec2  # "PyTorch (Vast)"
SMOKE=${SMOKE:-}
PARTS=${PARTS:-1 2 3}
GRACE_HOURS=${GRACE_HOURS:-3}
MAX_HOURS=${MAX_HOURS:-16}
[ -n "$SMOKE" ] && { PARTS=smoke; GRACE_HOURS=0; MAX_HOURS=2; }
V="uvx vastai"
OPTS="-i $HOME/.ssh/vastai -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30"
OFFER_QUERY="gpu_name=RTX_4090 num_gpus=1 reliability>0.99 rentable=true disk_space>=80 inet_down>500 inet_down_cost<0.005 inet_up_cost<0.005 cuda_vers>=13.2 direct_port_count>=4 geolocation notin [CN]"

log() { echo "$(date +%H:%M:%S) $*"; }
credit() { $V show user --raw 2>/dev/null | python3 -c "import json,sys; print(f\"{json.load(sys.stdin).get('credit',0):.2f}\")"; }
instance_json() { timeout 60 $V show instance "$1" --raw 2>/dev/null; }
field() { python3 -c "import json,sys
d=json.load(sys.stdin); v=d
for k in sys.argv[1].split('.'):
    v = v[int(k)] if isinstance(v, list) else (v or {}).get(k)
print('' if v is None else v)" "$1" 2>/dev/null; }
wf() {
  local key; key=$(cat "$STATE/workflow.key" 2>/dev/null) || return 0
  (cd "$ROOT_DIR" && timeout 60 uv run train-rag workflow "$1" "$key" "${@:2}") > /dev/null 2>&1 || true
}
part_dir() { [ "$1" = smoke ] && echo "$RUN/smoke" || echo "$RUN/part$1"; }
load_state() {  # sets INSTANCE HOST PORT PRICE URL
  INSTANCE=""; HOST=""; PORT=""; PRICE=""; URL=""
  # shellcheck disable=SC1090
  [ -f "$STATE/part$1.env" ] && source "$STATE/part$1.env"
  SSH="ssh $OPTS -p $PORT root@$HOST"
}

# Read the machine's ssh and page addresses; prints "HOST PORT URL" or nothing.
addresses() {
  local ssh_url host_port
  ssh_url=$(timeout 60 $V ssh-url "$1" 2>/dev/null | tail -1)
  host_port=${ssh_url#ssh://root@}
  local page; page=$(instance_json "$1" | python3 -c "import json,sys
d=json.load(sys.stdin)
ports=(d.get('ports') or {}).get('8000/tcp') or []
ip=(d.get('public_ipaddr') or '').strip()
print(f'http://{ip}:{ports[0][\"HostPort\"]}' if ports and ip else '')" 2>/dev/null)
  [ -n "$host_port" ] && echo "${host_port%:*} ${host_port##*:} $page"
}

wait_ssh() {  # INSTANCE -> sets HOST PORT URL SSH, or returns 1
  local status=""
  for _ in $(seq 1 40); do
    status=$(instance_json "$1" | field actual_status)
    [ "$status" = running ] && break
    sleep 10
  done
  [ "$status" = running ] || return 1
  read -r HOST PORT URL <<< "$(addresses "$1")"
  [ -n "$HOST" ] && [ -n "$PORT" ] || return 1
  SSH="ssh $OPTS -p $PORT root@$HOST"
  for _ in $(seq 1 30); do timeout 30 $SSH -n true 2>/dev/null && return 0; sleep 10; done
  return 1
}

make_smoke() {  # the first 2,000 documents of part 1's first shard
  mkdir -p "$RUN/smoke"
  python3 - "$RUN" <<'PY'
import gzip, json, sys
from pathlib import Path
run = Path(sys.argv[1])
with gzip.open(run / "part1" / "shard_01.jsonl.gz", "rt") as src:
    lines = [next(src) for _ in range(2001)]
with gzip.open(run / "smoke" / "shard_01.jsonl.gz", "wt") as out:
    out.writelines(lines)
(run / "smoke" / "part.json").write_text(json.dumps(
    {"documents": 2000, "shards": {"shard_01.jsonl.gz": {"documents": 2000}}}))
PY
}

part_json() {  # PART -> writes the part's entry of the manifest to its folder
  [ "$1" = smoke ] && return 0
  python3 -c "import json,sys
m=json.load(open(sys.argv[1])); print(json.dumps(m['parts']['part'+sys.argv[2]]))" \
    "$RUN/manifest.json" "$1" > "$(part_dir "$1")/part.json"
}

cmd_launch() {
  [ -f "$ADAPTER/adapter_model.safetensors" ] || { log "no adapter in $ADAPTER"; exit 1; }
  [ -f "$RUN/manifest.json" ] || { log "run build_corpus_prompts.py first"; exit 1; }
  mkdir -p "$STATE"
  [ -n "$SMOKE" ] && make_smoke
  for p in $PARTS; do
    [ -f "$STATE/part$p.env" ] && { log "part $p already has a machine (state/part$p.env)"; exit 1; }
    part_json "$p"
  done
  log "credit: \$$(credit)"
  if [ -z "$SMOKE" ] && [ ! -f "$STATE/workflow.key" ]; then
    local steps=(); for p in $PARTS; do steps+=("Part $p"); done  # one name per step
    (cd "$ROOT_DIR" && timeout 60 uv run train-rag workflow start "Mine the corpus" "${steps[@]}") \
      > "$STATE/workflow.key" 2>/dev/null \
      || rm -f "$STATE/workflow.key"
  fi

  local offers; offers=$($V search offers "$OFFER_QUERY" -o dph --raw 2>/dev/null | python3 -c "import json,sys
for o in [o for o in json.load(sys.stdin) if str(o['id']) not in '${SKIP:-}'.split()][:8]:
    print(o['id'], round(o['dph_total'],3))")
  [ -z "$offers" ] && { log "no offer"; exit 1; }

  # Rent one machine per part; a machine that doesn't boot is destroyed and the next offer tried.
  local rented=() used=""
  RENTED_IDS=""
  trap 'for i in $RENTED_IDS; do log "launch aborted: destroying $i"; timeout 120 $V destroy instance "$i" -y > /dev/null 2>&1; done' EXIT
  for p in $PARTS; do
    local ok=""
    while read -r offer price <&3; do
      case " $used " in *" $offer "*) continue ;; esac
      used="$used $offer"
      log "part $p: renting offer $offer at \$$price/h"
      INSTANCE=$($V create instance "$offer" --template_hash "$TEMPLATE" --disk 80 \
        --env '-p 8000:8000' --label "mine-part$p" --raw 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])" 2>/dev/null)
      [ -z "$INSTANCE" ] && { log "create failed; next offer"; continue; }
      RENTED_IDS="$RENTED_IDS $INSTANCE"
      if wait_ssh "$INSTANCE" && [ -n "$URL" ]; then
        printf 'INSTANCE=%s\nHOST=%s\nPORT=%s\nPRICE=%s\nURL=%s\n' \
          "$INSTANCE" "$HOST" "$PORT" "$price" "$URL" > "$STATE/part$p.env"
        rented+=("$p"); ok=1; break
      fi
      log "offer $offer never answered (or no page port); destroying it"
      timeout 120 $V destroy instance "$INSTANCE" -y > /dev/null 2>&1
      RENTED_IDS=${RENTED_IDS/ $INSTANCE/}
    done 3<<< "$offers"
    [ -z "$ok" ] && { log "no machine for part $p; stopping the launch"; exit 1; }
  done

  # Upload and start each part; its job.env lists the other machines' pages.
  for p in "${rented[@]}"; do
    load_state "$p"
    local peers=""
    for q in "${rented[@]}"; do [ "$q" != "$p" ] && peers="$peers $(. "$STATE/part$q.env"; echo "$URL")"; done
    printf 'PART=%s\nPRICE=%s\nGRACE_HOURS=%s\nMAX_HOURS=%s\nPEERS="%s"\n' \
      "$p" "$PRICE" "$GRACE_HOURS" "$MAX_HOURS" "${peers# }" > "$STATE/job$p.env"
    log "part $p: uploading to $HOST:$PORT"
    timeout 60 $SSH -n 'mkdir -p /workspace/mine/shards' || { log "part $p: ssh failed"; exit 1; }
    local dir; dir=$(part_dir "$p")
    timeout 1800 scp $OPTS -P "$PORT" "$dir"/shard_*.jsonl.gz "root@$HOST:/workspace/mine/shards/" \
      && timeout 600 scp $OPTS -P "$PORT" -r "$ADAPTER" "$HERE/merge_lora.py" "$HERE/generate_vllm.py" \
        "$HERE/mine_job.sh" "$HERE/mine_status.py" "$dir/part.json" "root@$HOST:/workspace/mine/" \
      && timeout 60 scp $OPTS -P "$PORT" "$STATE/job$p.env" "root@$HOST:/workspace/mine/job.env" \
      || { log "part $p: upload failed"; exit 1; }
    timeout 45 $SSH -n "cd /workspace/mine && setsid nohup bash mine_job.sh > job.log 2>&1 < /dev/null &"
    RENTED_IDS=${RENTED_IDS/ $INSTANCE/}  # started: the machine looks after itself now
    wf step "Part $p" running --note "RTX 4090 at \$$PRICE/h; page $URL"
  done
  trap - EXIT

  sleep 20
  log "started; progress pages (any one shows every part):"
  for p in "${rented[@]}"; do
    load_state "$p"
    if timeout 15 curl -fsS "$URL/status.json" > /dev/null; then echo "  part $p: $URL"
    else echo "  part $p: $URL (not answering yet)"; fi
  done
  log "credit: \$$(credit). The laptop may go off; run 'vast_mine.sh fetch' near the end."
}

cmd_fetch() {
  local any=""
  for f in "$STATE"/part*.env; do
    [ -f "$f" ] || continue
    any=1
    local p; p=$(basename "$f" .env); p=${p#part}
    load_state "$p"
    local status; status=$(instance_json "$INSTANCE" | field actual_status)
    log "part $p: instance $INSTANCE is ${status:-gone}"
    if [ -z "$status" ]; then log "part $p: instance not found; its files are lost"; continue; fi
    if [ "$status" != running ]; then
      log "part $p: starting it again to copy the files"
      timeout 60 $V start instance "$INSTANCE" > /dev/null 2>&1
      if ! wait_ssh "$INSTANCE"; then log "part $p: didn't come back (GPU busy?); try later"; continue; fi
      sed -i "s/^HOST=.*/HOST=$HOST/; s/^PORT=.*/PORT=$PORT/; s#^URL=.*#URL=$URL#" "$f"
    fi

    local out="$RUN/out/part$p"; mkdir -p "$out"
    local phase error
    phase=$(timeout 30 $SSH -n 'cat /workspace/mine/phase 2>/dev/null')
    error=$(timeout 30 $SSH -n 'cat /workspace/mine/error 2>/dev/null')
    # Hosts can upload slowly (0.4 MB/s seen); xz on the machine shrinks the outputs ~7x.
    # The archive is copied with resume (a retry continues it), then unpacked here.
    log "part $p: compressing on the machine"
    timeout 1800 $SSH -n 'cd /workspace/mine && rm -f fetch.tar.xz.tmp &&
      tar -cf - out logs job.log setup.log $(ls stop.log 2>/dev/null) | xz -T0 -6 > fetch.tar.xz.tmp &&
      mv fetch.tar.xz.tmp fetch.tar.xz && ls -l fetch.tar.xz | awk "{print \$5}"' \
      > "$out/.archive_size" || { log "part $p: compressing failed; try again"; continue; }
    log "part $p: copying $(( $(cat "$out/.archive_size") / 1000000 )) MB"
    timeout 7200 rsync --partial --append-verify -e "ssh $OPTS -p $PORT" \
      "root@$HOST:/workspace/mine/fetch.tar.xz" "$out/fetch.tar.xz" 2>/dev/null \
      || { log "part $p: copy failed; try again"; continue; }
    local tmp="$out/.unpack"; rm -rf "$tmp"; mkdir -p "$tmp"
    tar -xJf "$out/fetch.tar.xz" -C "$tmp" || { log "part $p: archive broken; try again"; rm -f "$out/fetch.tar.xz"; continue; }
    cp -r "$tmp/out/." "$out/"; rm -rf "$out/logs"; cp -r "$tmp/logs" "$tmp"/*.log "$out/" 2>/dev/null
    rm -rf "$tmp" "$out/fetch.tar.xz" "$out/.archive_size"

    # Complete = every shard has .done and as many lines as the manifest says.
    local check; check=$(python3 - "$(part_dir "$p")/part.json" "$out" <<'PY'
import json, sys
from pathlib import Path
part, out = json.load(open(sys.argv[1])), Path(sys.argv[2])
done = complete = 0
for name, info in part["shards"].items():
    stem = name.removesuffix(".jsonl.gz")
    lines = sum(1 for _ in (out / f"{stem}.jsonl").open()) if (out / f"{stem}.jsonl").exists() else 0
    done += lines
    complete += (out / f"{stem}.done").exists() and lines == info["documents"]
print(done, part["documents"], int(complete == len(part["shards"])))
PY
)
    local got total full; read -r got total full <<< "$check"
    log "part $p: $got/$total documents copied; phase: ${phase:-?}${error:+; error: $error}"
    wf progress "Part $p" "$got" "$total" --unit documents --note "phase: ${phase:-?}"

    if [ "$full" = 1 ]; then
      timeout 30 $SSH -n 'touch /workspace/mine/fetched'
      timeout 120 $V destroy instance "$INSTANCE" -y > /dev/null 2>&1
      log "part $p: complete; instance $INSTANCE destroyed"
      wf step "Part $p" success --note "$got documents; machine destroyed"
      mkdir -p "$STATE/done"; mv "$f" "$STATE/done/"
    elif [ "$phase" = failed ] || [ "$phase" = stopping ]; then
      log "part $p: stopped (${error:-see $out/stop.log}). Left as is: fix, then relaunch this part."
      wf step "Part $p" failed --note "${error:-stopped}; $got/$total copied"
    fi
  done
  [ -z "$any" ] && log "no machines in $STATE"
  log "live instances: $($V show instances --raw 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))'); credit: \$$(credit)"
}

cmd_status() {
  for f in "$STATE"/part*.env; do
    [ -f "$f" ] || { log "no machines in $STATE"; break; }
    local p; p=$(basename "$f" .env); p=${p#part}
    load_state "$p"
    local status; status=$(instance_json "$INSTANCE" | field actual_status)
    local page; page=$(timeout 15 curl -fsS "$URL/status.json" 2>/dev/null | python3 -c "import json,sys
s=json.load(sys.stdin); eta=s['eta_seconds']
print(f\"{s['phase']}: {s['done']:,}/{s['total']:,} docs, {s['docs_per_second']}/s, ETA {eta//3600 if eta else '-'} h, \${s['cost']}\" + (f\"; error: {s['error']}\" if s['error'] else ''))" 2>/dev/null)
    echo "part $p: instance $INSTANCE ${status:-gone}; ${page:-page not answering}; $URL"
  done
  log "credit: \$$(credit)"
}

# While the laptop is on: copy each machine's live counts to the local progress page.
cmd_watch() {
  while ls "$STATE"/part*.env > /dev/null 2>&1; do
    for f in "$STATE"/part*.env; do
      local p; p=$(basename "$f" .env); p=${p#part}
      load_state "$p"
      local line; line=$(timeout 15 curl -fsS "$URL/status.json" 2>/dev/null | python3 -c "import json,sys
s=json.load(sys.stdin); e=s['eta_seconds']
eta=f'{e//3600} h {e%3600//60} min' if e else '-'
print(s['done'], s['total'], f\"{s['phase']} · {s['docs_per_second']} docs/s · ETA {eta} · \${s['cost']} · {s['gpu']}\")" 2>/dev/null)
      if [ -n "$line" ]; then
        local done total note; read -r done total note <<< "$line"
        wf progress "Part $p" "$done" "$total" --unit documents --note "$note"
      fi
    done
    sleep "${WATCH_SECONDS:-120}"
  done
}

case "${1:-}" in
  launch) cmd_launch ;;
  fetch) cmd_fetch ;;
  status) cmd_status ;;
  watch) cmd_watch ;;
  *) echo "usage: $0 launch|fetch|status|watch"; exit 1 ;;
esac
