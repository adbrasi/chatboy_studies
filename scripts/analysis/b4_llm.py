"""b4 — cliente de chat do OpenRouter com TODOS os parâmetros de controle (stop, top_p, logit_bias, prefill,
reasoning) e retorno rico (texto, finish_reason, tokens, custo, latência). Cache próprio em disco
(data/processed/b4_llm_cache.jsonl) para não interferir no cache de outros agentes.

Modelos permitidos nesta rodada (regra do usuário): flash-lite, gpt-6-luna, deepseek-flash-latest, mercury-2.5."""
import hashlib, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from jev import _key  # noqa

URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE = os.path.join(ROOT, "data", "processed", "b4_llm_cache.jsonl")
MODELS = {"lite": "google/gemini-3.5-flash-lite", "luna": "openai/gpt-6-luna",
          "deepseek": "~deepseek/deepseek-flash-latest", "mercury": "inception/mercury-2.5"}
# o que cada provedor declara aceitar (GET /models, 2026-09-30); o resto é descartado aqui para o cache ser honesto
SUPPORTED = {
    "google/gemini-3.5-flash-lite": {"stop", "temperature", "top_p", "seed", "reasoning"},
    "openai/gpt-6-luna": {"seed", "reasoning"},
    "~deepseek/deepseek-flash-latest": {"stop", "temperature", "top_p", "seed", "logit_bias", "reasoning",
                                        "frequency_penalty", "presence_penalty"},
    "inception/mercury-2.5": {"stop", "temperature", "reasoning"},
}
REASONING = {  # desligar/minimizar o raciocínio (os 4 aceitam o parâmetro unificado)
    "google/gemini-3.5-flash-lite": {"effort": "none"},  # será conferido em b4_probe
    "openai/gpt-6-luna": {"effort": "none"},
    "~deepseek/deepseek-flash-latest": {"enabled": False},
    "inception/mercury-2.5": {"effort": "none"},
}
_lock, _cache = threading.Lock(), None
stats = {"calls": 0, "cached": 0, "cost": 0.0, "by_model": {}}


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


def chat(messages, model="google/gemini-3.5-flash-lite", temperature=0.8, max_tokens=300, seed=0, stop=None,
         top_p=None, logit_bias=None, prefill=None, reasoning="default", retries=5, extra=None, tag=""):
    """Retorna dict {text, finish, usage, cost, latency, cached}. prefill = texto do início da resposta (mensagem
    assistant final); o texto devolvido já vem com o prefill concatenado."""
    model = MODELS.get(model, model)
    sup = SUPPORTED.get(model, set())
    body = {"model": model, "messages": list(messages), "max_tokens": max_tokens}
    if "temperature" in sup and temperature is not None:
        body["temperature"] = temperature
    if "seed" in sup:
        body["seed"] = seed
    else:
        body["_seed"] = seed  # só entra na chave de cache (amostras distintas)
    if stop and "stop" in sup:
        body["stop"] = stop
    if top_p is not None and "top_p" in sup:
        body["top_p"] = top_p
    if logit_bias and "logit_bias" in sup:
        body["logit_bias"] = logit_bias
    if reasoning == "default":
        reasoning = REASONING.get(model)
    if reasoning:
        body["reasoning"] = reasoning
    if prefill:
        body["messages"].append({"role": "assistant", "content": prefill})
    if extra:
        body.update(extra)
    k = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    c = _load()
    if k in c:
        with _lock:
            stats["cached"] += 1
        return dict(c[k], cached=True)
    send = {kk: v for kk, v in body.items() if not kk.startswith("_")}
    for a in range(retries):
        t0 = time.time()
        try:
            r = requests.post(URL, json=send, timeout=120, headers={"Authorization": f"Bearer {_key()}"})
            if r.status_code in (429, 500, 502, 503, 529):
                raise RuntimeError(f"retryable {r.status_code} {r.text[:200]}")
            if r.status_code >= 400:
                raise RuntimeError(f"http {r.status_code} {r.text[:300]}")
            d = r.json()
            if "choices" not in d:
                raise RuntimeError(f"bad {str(d)[:300]}")
            ch = d["choices"][0]
            txt = ch["message"].get("content") or ""
            if prefill and not txt.startswith(prefill):
                txt = prefill + txt
            u = d.get("usage", {}) or {}
            out = {"text": txt, "finish": ch.get("finish_reason") or ch.get("native_finish_reason"),
                   "usage": {"in": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
                             "reason": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")},
                   "cost": u.get("cost", 0) or 0, "latency": time.time() - t0, "tag": tag}
            with _lock:
                stats["calls"] += 1
                stats["cost"] += out["cost"]
                bm = stats["by_model"].setdefault(model, {"calls": 0, "cost": 0.0})
                bm["calls"] += 1; bm["cost"] += out["cost"]
                c[k] = out
                with open(CACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "r": out}, ensure_ascii=False) + "\n")
            return dict(out, cached=False)
        except Exception as e:
            if a == retries - 1:
                print("llm error:", model, str(e)[:300], flush=True)
                return {"text": None, "finish": "error", "usage": {}, "cost": 0, "latency": None, "cached": False,
                        "error": str(e)[:300]}
            time.sleep(min(20, 2 ** a + random.random()))


def chat_many(items, workers=4):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda kw: chat(**kw), items))
