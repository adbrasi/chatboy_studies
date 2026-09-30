"""Tiny OpenRouter chat-completions client with disk cache (for generating LLM replies to compare with humans)."""
import hashlib, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor

import requests

sys_path = os.path.dirname(__file__)
from jev import _key  # same OpenRouter key

URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE = os.path.join(sys_path, "..", "data", "processed", "llm_cache.jsonl")
DEFAULT_MODEL = "google/gemini-3.5-flash-lite"   # fast/cheap "mouth" model
_lock, _cache = threading.Lock(), None
stats = {"calls": 0, "cached": 0, "cost": 0.0}


def _load():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(CACHE):
            for l in open(CACHE, encoding="utf-8"):
                try:
                    d = json.loads(l); _cache[d["k"]] = d["r"]
                except Exception:
                    pass
    return _cache


def chat(messages, model=DEFAULT_MODEL, temperature=0.8, max_tokens=300, seed=0, retries=5):
    """messages: [{"role": "system"|"user"|"assistant", "content": str}]. Returns reply text. Cached."""
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "seed": seed}
    k = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    c = _load()
    if k in c:
        stats["cached"] += 1
        return c[k]
    for a in range(retries):
        try:
            r = requests.post(URL, json=body, timeout=120, headers={"Authorization": f"Bearer {_key()}"})
            if r.status_code in (429, 500, 502, 503, 529):
                raise RuntimeError(f"retryable {r.status_code}")
            r.raise_for_status()
            d = r.json()
            txt = d["choices"][0]["message"]["content"] or ""
            with _lock:
                stats["calls"] += 1
                stats["cost"] += d.get("usage", {}).get("cost", 0) or 0
                c[k] = txt
                with open(CACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "r": txt}, ensure_ascii=False) + "\n")
            return txt
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(min(30, 2 ** a + random.random()))


def chat_many(items, workers=4):
    """items: list of dicts with chat() kwargs. Returns list of texts (None on error)."""
    def run(kw):
        try:
            return chat(**kw)
        except Exception as e:
            print("llm error:", str(e)[:200]); return None
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(run, items))
