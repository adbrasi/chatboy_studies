"""b2: cliente LLM com parâmetros extras (reasoning effort) e cache próprio (os modelos com raciocínio oculto,
ex. mercury-2.5, estouram max_tokens no scripts/llm.py). Só os 4 modelos autorizados."""
import hashlib, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jev import _key

ALLOWED = {"inception/mercury-2.5", "~deepseek/deepseek-flash-latest", "openai/gpt-6-luna", "google/gemini-3.5-flash-lite"}
CACHE = "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad/b2/llm_cache.jsonl"
_lock, _cache = threading.Lock(), None
stats = {"calls": 0, "cached": 0, "cost": 0.0, "errors": 0}


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


def chat(messages, model, temperature=0.3, max_tokens=1500, seed=0, effort="low", retries=4):
    assert model in ALLOWED, model
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "seed": seed}
    if effort:
        body["reasoning"] = {"effort": effort}
    k = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    c = _load()
    if k in c:
        stats["cached"] += 1
        return c[k]
    for a in range(retries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=120,
                              headers={"Authorization": f"Bearer {_key()}"})
            if r.status_code in (429, 500, 502, 503, 529):
                raise RuntimeError(f"retryable {r.status_code}")
            r.raise_for_status()
            d = r.json()
            txt = d["choices"][0]["message"].get("content") or ""
            with _lock:
                stats["calls"] += 1
                stats["cost"] += d.get("usage", {}).get("cost", 0) or 0
                if txt:
                    c[k] = txt
                    with open(CACHE, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"k": k, "r": txt}, ensure_ascii=False) + "\n")
            if not txt:
                raise RuntimeError("empty content")
            return txt
        except Exception:
            if a == retries - 1:
                stats["errors"] += 1
                return None
            time.sleep(min(20, 2 ** a + random.random()))


def chat_many(items, workers=4):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda kw: chat(**kw), items))
