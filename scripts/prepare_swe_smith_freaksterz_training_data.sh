#!/usr/bin/env bash
# Build on-policy SWE-smith training data with a running Freaksterz vLLM server,
# then feed it into the existing Speculators hidden-state collection pipeline.
set -Eeuo pipefail

ROOT=${ROOT:-/data/Projects/qwen3.8-27b-smoothquant-w8a8-int8-dflash2-vllm}
SPECULATORS_DIR=${SPECULATORS_DIR:-"$ROOT/../speculators"}
PYTHON=${PYTHON:-"$SPECULATORS_DIR/.venv/bin/python"}
COLLECT_SCRIPT=${COLLECT_SCRIPT:-"$ROOT/scripts/collect_freaksterz_dflash2_data.sh"}

# vLLM used to GENERATE Freaksterz responses.
GEN_ENDPOINT=${GEN_ENDPOINT:-http://127.0.0.1:8000/v1}
GEN_MODEL=${GEN_MODEL:-}  # empty = auto-detect from /v1/models
API_KEY=${API_KEY:-}

# vLLM used by Speculators to EXTRACT verifier hidden states.
# This should be your target-only hidden-state extraction server.
EXTRACT_ENDPOINT=${EXTRACT_ENDPOINT:-http://127.0.0.1:8001/v1}

DATASET_NAME=${DATASET_NAME:-SWE-bench/SWE-smith-trajectories}
DATASET_SPLIT=${DATASET_SPLIT:-tool}
MAX_SAMPLES=${MAX_SAMPLES:-5000}
SEQ_LENGTH=${SEQ_LENGTH:-8192}
CONCURRENCY=${CONCURRENCY:-4}
MAX_TOKENS=${MAX_TOKENS:-3072}
TEMPERATURE=${TEMPERATURE:-0.2}
TOP_P=${TOP_P:-0.95}
SEED=${SEED:-42}
MAX_CONTEXT_CHARS=${MAX_CONTEXT_CHARS:-12000}
MAX_OBSERVATION_CHARS=${MAX_OBSERVATION_CHARS:-4000}
MIN_RESPONSE_CHARS=${MIN_RESPONSE_CHARS:-40}

# auto | 0 | 1. "auto" leaves the server/model default untouched.
ENABLE_THINKING=${ENABLE_THINKING:-auto}

# 1 = after regeneration, run prepare_data + hidden-state extraction + bridge
# through the existing collect_freaksterz_dflash2_data.sh.
COLLECT_STATES=${COLLECT_STATES:-1}

RUN_DIR=${RUN_DIR:-"$ROOT/runs/swe-smith-freaksterz-$(date -u +%Y%m%dT%H%M%SZ)"}
SOURCE_DIR="$RUN_DIR/source"
TASKS_FILE="$SOURCE_DIR/swe-smith-tasks.jsonl"
ON_POLICY_FILE="$RUN_DIR/freaksterz-on-policy.jsonl"
ERROR_FILE="$RUN_DIR/freaksterz-on-policy.errors.jsonl"
STATS_FILE="$RUN_DIR/regeneration-stats.json"

mkdir -p "$RUN_DIR/logs" "$SOURCE_DIR"
exec > >(tee -a "$RUN_DIR/logs/swe-smith-regeneration.log") 2>&1

[[ -x "$PYTHON" ]] || { echo "ERROR: Python not found/executable: $PYTHON" >&2; exit 1; }
[[ -d "$SPECULATORS_DIR" ]] || { echo "ERROR: Speculators checkout not found: $SPECULATORS_DIR" >&2; exit 1; }

export GEN_ENDPOINT GEN_MODEL API_KEY DATASET_NAME DATASET_SPLIT MAX_SAMPLES CONCURRENCY
export MAX_TOKENS TEMPERATURE TOP_P SEED MAX_CONTEXT_CHARS MAX_OBSERVATION_CHARS
export MIN_RESPONSE_CHARS ENABLE_THINKING TASKS_FILE ON_POLICY_FILE ERROR_FILE STATS_FILE

echo "=== SWE-smith -> Freaksterz on-policy dataset ==="
echo "run_dir:           $RUN_DIR"
echo "dataset:           $DATASET_NAME [$DATASET_SPLIT]"
echo "samples:           $MAX_SAMPLES"
echo "generation vLLM:   $GEN_ENDPOINT"
echo "concurrency:        $CONCURRENCY"
echo "max output tokens:  $MAX_TOKENS"
echo "thinking override:  $ENABLE_THINKING"
echo

"$PYTHON" - <<'PY'
import concurrent.futures as cf
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from datasets import load_dataset

GEN_ENDPOINT = os.environ["GEN_ENDPOINT"].rstrip("/")
GEN_MODEL = os.environ.get("GEN_MODEL", "").strip()
API_KEY = os.environ.get("API_KEY", "").strip()
DATASET_NAME = os.environ["DATASET_NAME"]
DATASET_SPLIT = os.environ["DATASET_SPLIT"]
MAX_SAMPLES = int(os.environ["MAX_SAMPLES"])
CONCURRENCY = int(os.environ["CONCURRENCY"])
MAX_TOKENS = int(os.environ["MAX_TOKENS"])
TEMPERATURE = float(os.environ["TEMPERATURE"])
TOP_P = float(os.environ["TOP_P"])
SEED = int(os.environ["SEED"])
MAX_CONTEXT_CHARS = int(os.environ["MAX_CONTEXT_CHARS"])
MAX_OBSERVATION_CHARS = int(os.environ["MAX_OBSERVATION_CHARS"])
MIN_RESPONSE_CHARS = int(os.environ["MIN_RESPONSE_CHARS"])
ENABLE_THINKING = os.environ["ENABLE_THINKING"].lower()
TASKS_FILE = Path(os.environ["TASKS_FILE"])
ON_POLICY_FILE = Path(os.environ["ON_POLICY_FILE"])
ERROR_FILE = Path(os.environ["ERROR_FILE"])
STATS_FILE = Path(os.environ["STATS_FILE"])

if GEN_ENDPOINT.endswith("/chat/completions"):
    CHAT_URL = GEN_ENDPOINT
    API_BASE = GEN_ENDPOINT[: -len("/chat/completions")]
else:
    API_BASE = GEN_ENDPOINT
    CHAT_URL = f"{GEN_ENDPOINT}/chat/completions"
MODELS_URL = f"{API_BASE}/models"

SYSTEM_PROMPT = (
    "You are an expert software-engineering coding agent. Work on the user's task using "
    "the repository observations supplied in the prompt. Give a concrete, technically "
    "useful response. Do not claim to have executed commands that are not present in the "
    "observations."
)

print_lock = threading.Lock()


def headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def request_json(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def detect_model():
    global GEN_MODEL
    if GEN_MODEL:
        return GEN_MODEL
    obj = request_json(MODELS_URL, timeout=30)
    models = obj.get("data") or []
    if not models:
        raise RuntimeError(f"No served models returned by {MODELS_URL}")
    GEN_MODEL = models[0]["id"]
    return GEN_MODEL


def flatten_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, dict):
        typ = content.get("type")
        if typ == "text" and isinstance(content.get("text"), str):
            return content["text"]
        if typ in {"tool_result", "tool_response"}:
            return flatten_content(content.get("content"))
        if isinstance(content.get("content"), (str, list, dict)):
            return flatten_content(content.get("content"))
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    if isinstance(content, list):
        parts = [flatten_content(x) for x in content]
        return "\n".join(x for x in parts if x)
    return str(content)


def parse_messages(raw):
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        return []
    return [m for m in raw if isinstance(m, dict)]


def clean_task(text):
    # /testbed is meaningful to SWE-agent, but not to a plain remote chat-completions call.
    text = re.sub(r"<uploaded_files>.*?</uploaded_files>\s*", "", text, flags=re.S | re.I)
    return text.strip()


def extract_task_and_observations(messages):
    task = ""
    observations = []

    for m in messages:
        role = str(m.get("role", "")).lower()
        msg_type = str(m.get("message_type", "")).lower()
        content = flatten_content(m.get("content"))

        if not task and role == "user":
            # If the user content contains tool_result parts, prefer the natural-language text.
            task = clean_task(content)

        is_observation = role == "tool" or msg_type in {"observation", "tool_result", "tool_response"}
        if is_observation and content:
            content = content.strip()
            if len(content) > MAX_OBSERVATION_CHARS:
                content = content[:MAX_OBSERVATION_CHARS] + "\n...[observation truncated]"
            observations.append(content)

    # Some SWE-agent serializations put tool results inside user content parts.
    if not observations:
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"tool_result", "tool_response"}:
                    txt = flatten_content(part).strip()
                    if txt:
                        observations.append(txt[:MAX_OBSERVATION_CHARS])

    obs_text = ""
    if observations:
        chunks = []
        used = 0
        for i, obs in enumerate(observations, 1):
            chunk = f"\n--- repository observation {i} ---\n{obs}\n"
            if used + len(chunk) > MAX_CONTEXT_CHARS:
                remain = MAX_CONTEXT_CHARS - used
                if remain > 300:
                    chunks.append(chunk[:remain] + "\n...[context truncated]")
                break
            chunks.append(chunk)
            used += len(chunk)
        obs_text = "".join(chunks)

    return task, obs_text


def build_user_prompt(task, observations):
    if observations:
        return (
            f"Software engineering task:\n\n{task}\n\n"
            "Repository observations from an existing SWE-agent trajectory follow. "
            "Use them as read-only context; do not assume you can access /testbed directly.\n"
            f"{observations}"
        )
    return (
        f"Software engineering task:\n\n{task}\n\n"
        "The repository itself is not attached to this chat. Work from the task statement and "
        "state any repository assumptions explicitly."
    )


def load_tasks():
    print(f"Downloading/loading {DATASET_NAME} split={DATASET_SPLIT} via datasets ...", flush=True)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    # Deterministic shuffle prevents selecting only one repository/model cluster from the head.
    ds = ds.shuffle(seed=SEED)

    by_instance = {}
    scanned = 0
    for row in ds:
        scanned += 1
        iid = str(row.get("instance_id") or row.get("traj_id") or f"row-{scanned}")
        if iid in by_instance:
            continue
        try:
            messages = parse_messages(row.get("messages"))
            task, observations = extract_task_and_observations(messages)
        except Exception:
            continue
        if len(task) < 40:
            continue

        by_instance[iid] = {
            "id": iid,
            "instance_id": iid,
            "traj_id": str(row.get("traj_id") or ""),
            "source_model": str(row.get("model") or ""),
            "source_resolved": bool(row.get("resolved", False)),
            "task": task,
            "observations": observations,
            "user_prompt": build_user_prompt(task, observations),
        }
        if len(by_instance) >= MAX_SAMPLES:
            break

    tasks = list(by_instance.values())
    random.Random(SEED).shuffle(tasks)
    tasks = tasks[:MAX_SAMPLES]
    TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TASKS_FILE.open("w", encoding="utf-8") as f:
        for x in tasks:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"Scanned rows: {scanned}")
    print(f"Unique usable tasks: {len(tasks)}")
    print(f"Saved source tasks: {TASKS_FILE}")
    return tasks


def completed_ids():
    done = set()
    if not ON_POLICY_FILE.exists():
        return done
    with ON_POLICY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                iid = obj.get("instance_id") or obj.get("id")
                if iid:
                    done.add(str(iid))
            except Exception:
                pass
    return done


def merge_reasoning(message):
    content = flatten_content(message.get("content")).strip()
    reasoning = flatten_content(
        message.get("reasoning_content")
        if message.get("reasoning_content") is not None
        else message.get("reasoning")
    ).strip()
    if reasoning and "<think>" not in content[:100]:
        content = f"<think>\n{reasoning}\n</think>\n{content}".strip()
    return content, reasoning


def generate_one(item, ordinal):
    payload = {
        "model": GEN_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["user_prompt"]},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "seed": SEED + ordinal,
        "stream": False,
    }
    if ENABLE_THINKING in {"0", "1"}:
        payload["chat_template_kwargs"] = {"enable_thinking": ENABLE_THINKING == "1"}

    last_err = None
    for attempt in range(1, 5):
        try:
            try:
                obj = request_json(CHAT_URL, payload=payload, timeout=900)
            except urllib.error.HTTPError as e:
                # Some older vLLM versions reject chat_template_kwargs. Retry once without it.
                if e.code == 400 and "chat_template_kwargs" in payload:
                    payload.pop("chat_template_kwargs", None)
                    obj = request_json(CHAT_URL, payload=payload, timeout=900)
                else:
                    raise

            choice = (obj.get("choices") or [None])[0]
            if not choice:
                raise RuntimeError(f"No choices in response: {obj}")
            finish = str(choice.get("finish_reason") or "")
            msg = choice.get("message") or {}
            text, reasoning = merge_reasoning(msg)
            if finish == "length":
                raise RuntimeError("generation hit max_tokens (finish_reason=length)")
            if len(text) < MIN_RESPONSE_CHARS:
                raise RuntimeError(f"response too short ({len(text)} chars)")

            row = {
                "id": f"swe-smith-{item['instance_id']}",
                "instance_id": item["instance_id"],
                "traj_id": item["traj_id"],
                "source": DATASET_NAME,
                "target_model": GEN_MODEL,
                "finish_reason": finish,
                "conversations": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["user_prompt"]},
                    {"role": "assistant", "content": text},
                ],
                "metadata": {
                    "source_model": item["source_model"],
                    "source_resolved": item["source_resolved"],
                    "reasoning_chars": len(reasoning),
                    "usage": obj.get("usage") or {},
                },
            }
            return True, row
        except Exception as e:
            last_err = repr(e)
            if attempt < 4:
                time.sleep(min(2 ** (attempt - 1), 8))

    return False, {
        "id": item["instance_id"],
        "traj_id": item["traj_id"],
        "error": last_err,
    }


def main():
    model = detect_model()
    print(f"Detected generation model: {model}")
    # Basic health check before a potentially large HF download.
    request_json(MODELS_URL, timeout=30)

    tasks = load_tasks()
    done = completed_ids()
    pending = [x for x in tasks if x["instance_id"] not in done]
    print(f"Already complete: {len(done)}")
    print(f"Pending generation: {len(pending)}")

    ON_POLICY_FILE.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = 0
    started = time.time()

    with ON_POLICY_FILE.open("a", encoding="utf-8") as out, ERROR_FILE.open("a", encoding="utf-8") as err:
        with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {
                pool.submit(generate_one, item, idx): item
                for idx, item in enumerate(pending)
            }
            total = len(futures)
            for n, fut in enumerate(cf.as_completed(futures), 1):
                ok, row = fut.result()
                if ok:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    successes += 1
                else:
                    err.write(json.dumps(row, ensure_ascii=False) + "\n")
                    err.flush()
                    failures += 1
                if n == 1 or n % 25 == 0 or n == total:
                    elapsed = max(time.time() - started, 0.001)
                    with print_lock:
                        print(
                            f"[{n}/{total}] new_ok={successes} new_failed={failures} "
                            f"rate={n/elapsed:.2f} samples/s",
                            flush=True,
                        )

    final_done = completed_ids()
    stats = {
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "target_model": GEN_MODEL,
        "requested_tasks": len(tasks),
        "completed_total": len(final_done),
        "generated_this_run": successes,
        "failed_this_run": failures,
        "generation_endpoint": CHAT_URL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "enable_thinking": ENABLE_THINKING,
    }
    STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if not final_done:
        raise SystemExit("No successful generations; refusing to continue to preprocessing.")


if __name__ == "__main__":
    main()
PY

echo
echo "=== On-policy generation complete ==="
echo "tasks:      $TASKS_FILE"
echo "dataset:    $ON_POLICY_FILE"
echo "errors:     $ERROR_FILE"
echo "stats:      $STATS_FILE"

if [[ "$COLLECT_STATES" == "1" ]]; then
  [[ -x "$COLLECT_SCRIPT" || -f "$COLLECT_SCRIPT" ]] || {
    echo "ERROR: collection script not found: $COLLECT_SCRIPT" >&2
    exit 1
  }

  echo
  echo "=== Speculators preprocessing + verifier hidden states + bridge ==="
  echo "collector:          $COLLECT_SCRIPT"
  echo "extract endpoint:   $EXTRACT_ENDPOINT"

  DATASET="$ON_POLICY_FILE" \
  RUN_DIR="$RUN_DIR" \
  MAX_SAMPLES="$MAX_SAMPLES" \
  SEQ_LENGTH="$SEQ_LENGTH" \
  CONCURRENCY="$CONCURRENCY" \
  EXTRACT_ENDPOINT="$EXTRACT_ENDPOINT" \
    bash "$COLLECT_SCRIPT"
else
  echo
  echo "COLLECT_STATES=0: stopping after on-policy JSONL generation."
  echo "To collect verifier hidden states later:"
  echo "  DATASET='$ON_POLICY_FILE' RUN_DIR='$RUN_DIR' EXTRACT_ENDPOINT='$EXTRACT_ENDPOINT' bash '$COLLECT_SCRIPT'"
fi

echo
echo "=== Done ==="
echo "Training run directory: $RUN_DIR"
