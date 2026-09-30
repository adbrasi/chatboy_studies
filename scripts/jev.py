"""Minimal Jev (TypeSafe System One) client via OpenRouter, with disk cache and thread-pool batching.

Endpoint: POST https://openrouter.ai/api/v1/systemone   model: "typesafe/jev-1.13"
Body: {"state": ..., "model": ..., "questions": {id: {type, instructions, criteria}}}

Helpers build questions:
  noul(instr, true=None, false=None)
  choice(instr, {option: description})
  score(instr, [level0, level1, ...])
"""
import hashlib, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = "https://openrouter.ai/api/v1/systemone"
MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.13")
ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "processed", "jev_cache.jsonl")


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k and os.path.exists(os.path.join(ROOT, ".env")):
        for line in open(os.path.join(ROOT, ".env")):
            if line.startswith("OPENROUTER_API_KEY="):
                k = line.split("=", 1)[1].strip()
    return k


def noul(instructions, true=None, false=None):
    q = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {"true": true or "Yes", "false": false or "No"}
    return q


def choice(instructions, options):
    return {"type": "choice", "instructions": instructions, "criteria": options}


def score(instructions, levels):
    return {"type": "score", "instructions": instructions, "criteria": levels}


_lock = threading.Lock()
_cache = None
stats = {"calls": 0, "cached": 0, "input_tokens": 0, "cost": 0.0, "latency_s": []}


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(CACHE):
            for line in open(CACHE, encoding="utf-8"):
                try:
                    d = json.loads(line)
                    _cache[d["k"]] = d["r"]
                except Exception:
                    pass
    return _cache


def _hash(state, questions):
    return hashlib.sha256(json.dumps([MODEL, state, questions], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ask(state, questions, retries=6):
    """One Jev call. Returns {question_id: answer_dict}. Cached on disk."""
    k = _hash(state, questions)
    c = _load_cache()
    if k in c:
        stats["cached"] += 1
        return c[k]
    body = {"state": state, "model": MODEL, "questions": questions}
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = requests.post(URL, json=body, timeout=60,
                              headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
            if r.status_code in (429, 500, 502, 503, 529):
                raise RuntimeError(f"retryable {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            d = r.json()
            if "answers" not in d:
                raise RuntimeError(f"bad response: {str(d)[:300]}")
            with _lock:
                stats["calls"] += 1
                stats["latency_s"].append(time.time() - t0)
                u = d.get("usage", {})
                stats["input_tokens"] += u.get("input_tokens", 0)
                stats["cost"] += u.get("cost", 0) or 0
                c[k] = d["answers"]
                with open(CACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "r": d["answers"]}, ensure_ascii=False) + "\n")
            return d["answers"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(min(30, 2 ** attempt + random.random()))


def ask_many(items, workers=12, on_error="skip"):
    """items: list of (state, questions). Returns list of answers (None on error if on_error=='skip')."""
    def run(it):
        try:
            return ask(*it)
        except Exception as e:
            if on_error == "skip":
                print("jev error:", str(e)[:200])
                return None
            raise
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(run, items))


def top(ans):
    """Scalar view of an answer: noul -> p, score -> expected level, choice -> label."""
    t = ans["type"]
    return ans["noul"] if t == "noul" else ans["score"] if t == "score" else ans["choice"]


def summary():
    lat = sorted(stats["latency_s"])
    p50 = lat[len(lat) // 2] if lat else None
    return {k: v for k, v in stats.items() if k != "latency_s"} | {"latency_p50_s": p50, "n_lat": len(lat)}
