"""a6: shared helpers for the Jev-as-infrastructure experiments.

raw_ask(): one uncached-by-content call straight to the endpoint (timed), with its own
per-experiment cache keyed by (tag, rep, state, questions) so that re-running the analysis
does not re-spend, while repeated reps are real, independent calls.
"""
import hashlib, json, os, random, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from jev import _key, URL, MODEL, noul, choice, score  # noqa
import annotate_base as AB  # noqa  (main() is guarded)

SCRATCH = os.environ.get("A6_SCRATCH", "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad")
CACHE = os.path.join(SCRATCH, "a6_cache.jsonl")
OUT = os.path.join(ROOT, "analysis", "data")
PROC = os.path.join(ROOT, "data", "processed")
_lock = threading.Lock()
_cache = None
NEW_CALLS = [0]


def _load():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(CACHE):
            for l in open(CACHE, encoding="utf-8"):
                d = json.loads(l)
                _cache[d["k"]] = d
    return _cache


def raw_ask(state, questions, tag="x", rep=0, retries=6):
    """Returns dict(answers, latency_s, usage, model, cached)."""
    k = hashlib.sha256(json.dumps([tag, rep, MODEL, state, questions], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    c = _load()
    if k in c:
        return c[k] | {"cached": True}
    body = {"state": state, "model": MODEL, "questions": questions}
    for a in range(retries):
        t0 = time.time()
        try:
            r = requests.post(URL, json=body, timeout=90,
                              headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
            lat = time.time() - t0
            if r.status_code in (429, 500, 502, 503, 529):
                raise RuntimeError(f"retryable {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            d = r.json()
            if "answers" not in d:
                raise RuntimeError(f"bad response {str(d)[:300]}")
            rec = {"k": k, "tag": tag, "answers": d["answers"], "latency_s": lat, "usage": d.get("usage", {}),
                   "model": d.get("model"), "t0": t0, "t1": t0 + lat, "attempt": a}
            with _lock:
                c[k] = rec
                NEW_CALLS[0] += 1
                with open(CACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec | {"cached": False}
        except Exception as e:
            if a == retries - 1:
                print("jev error", str(e)[:200], flush=True)
                return None
            time.sleep(min(20, 2 ** a + random.random()))


def run_many(items, workers=4):
    """items: list of dicts with state, questions, tag, rep."""
    workers = min(workers, 4)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda it: raw_ask(it["state"], it["questions"], it.get("tag", "x"), it.get("rep", 0)), items))


def val(a):
    """scalar of an answer"""
    t = a["type"]
    return a["noul"] if t == "noul" else a["score"] if t == "score" else a["choice"]


# ---------------------------------------------------------------- data
def load_turns(corpora=("maichat", "whatsapp_nl")):
    by = defaultdict(list)
    for l in open(os.path.join(PROC, "turns.jsonl"), encoding="utf-8"):
        if not any(f'"corpus": "{c}"' in l[:40] for c in corpora):
            continue
        t = json.loads(l)
        if t["corpus"] in corpora:
            by[(t["corpus"], t["conv_id"])].append(t)
    for v in by.values():
        v.sort(key=lambda t: t["turn_idx"])
    return by


def load_base():
    base = {}
    for l in open(os.path.join(PROC, "jev_base.jsonl"), encoding="utf-8"):
        d = json.loads(l)
        base[(d["corpus"], d["conv_id"], d["turn_idx"])] = d
    return base


def context(conv, t, n):
    i = conv.index(t)
    if n == 0:
        return []
    return [x for x in conv[max(0, i - n):i] if x["session"] == t["session"]]


def d_state(conv, t, n=8, label_map=None):
    ctx = [AB.fmt_turn(x, label=(label_map or {}).get(x["speaker"])) for x in context(conv, t, n)]
    return {"previous_turns": ctx or "(this is the first turn of the conversation)",
            "current_turn": AB.fmt_turn(t, label=(label_map or {}).get(t["speaker"]))}


def sample_turns(by, corpus, n, min_ctx=2, seed=6, min_chars=0, in_base=None):
    rnd = random.Random(seed)
    cands = []
    for (c, cid), conv in by.items():
        if c != corpus:
            continue
        for t in conv:
            if len(context(conv, t, min_ctx)) < min_ctx or t["total_chars"] < min_chars:
                continue
            if in_base is not None and (c, cid, t["turn_idx"]) not in in_base:
                continue
            cands.append((conv, t))
    return rnd.sample(cands, min(n, len(cands)))


def dump(name, obj):
    p = os.path.join(OUT, f"a6_{name}")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    return p
