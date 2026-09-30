"""b1: OpenRouter chat client with reasoning-effort control, disk cache and cost stats.
Allowed models this round: inception/mercury-2.5, ~deepseek/deepseek-flash-latest, openai/gpt-6-luna, google/gemini-3.5-flash-lite."""
import hashlib, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests
from b1_common import SCR
from jev import _key

URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE = os.path.join(SCR, "llm_cache.jsonl")
ALLOWED = {"inception/mercury-2.5", "~deepseek/deepseek-flash-latest", "openai/gpt-6-luna", "google/gemini-3.5-flash-lite"}
_lock, _cache = threading.Lock(), None
stats = {"calls": 0, "cached": 0, "cost": 0.0, "latency_s": []}


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


def chat(messages, model="openai/gpt-6-luna", temperature=0.8, max_tokens=700, seed=0, effort="low", retries=5, json_mode=False):
    assert model in ALLOWED, model
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "seed": seed}
    if effort:
        body["reasoning"] = {"effort": effort}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    k = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    c = _load()
    if k in c:
        stats["cached"] += 1
        return c[k]
    for a in range(retries):
        try:
            t0 = time.time()
            r = requests.post(URL, json=body, timeout=180, headers={"Authorization": f"Bearer {_key()}"})
            if r.status_code in (429, 500, 502, 503, 529):
                raise RuntimeError(f"retryable {r.status_code}")
            r.raise_for_status()
            d = r.json()
            txt = d["choices"][0]["message"]["content"] or ""
            if not txt.strip():
                raise RuntimeError("empty content")
            with _lock:
                stats["calls"] += 1
                stats["latency_s"].append(time.time() - t0)
                stats["cost"] += d.get("usage", {}).get("cost", 0) or 0
                c[k] = txt
                with open(CACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "r": txt}, ensure_ascii=False) + "\n")
            return txt
        except Exception as e:
            if a == retries - 1:
                raise
            time.sleep(min(20, 2 ** a + random.random()))


def chat_many(items, workers=4):
    def run(kw):
        try:
            return chat(**kw)
        except Exception as e:
            print("llm error:", str(e)[:200]); return None
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(run, items))
