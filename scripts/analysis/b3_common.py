"""b3 — arquiteturas Jev para "o que escrever" (movimento, tom, elemento) + Briefing v2.
Utilidades: pontos de decisão (todo o maichat), divisão dev/teste por conversa, wrappers de Jev/LLM com cache,
bootstrap por conversa, taxonomia de movimentos (a mesma da 1ª rodada, a9) e famílias.

Arquivos grandes (histórico completo dos pontos) ficam em data/processed/b3_*.jsonl (fora do git);
as saídas pequenas ficam em analysis/data/b3_*."""
import hashlib, json, math, os, random, re, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
import jev  # noqa: E402
import llm  # noqa: E402
from jev import noul, choice, score  # noqa: E402,F401
from a9_common import (load_mai, turn_text, style_fingerprint, feats, words, bubbles, clean_reply,  # noqa: E402,F401
                       BOT, USER, PERSONA, LAUGH, EMOJI)
from a9_brief import Q as Q9  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
ADATA = os.path.join(ROOT, "analysis", "data")
ALL = os.path.join(PROC, "b3_points_all.jsonl")      # todos os pontos de decisão do maichat (+ ouro)
A9P = os.path.join(ADATA, "a9_points.jsonl")
CTX = 8

MOVE_OPTS = Q9["move"]["criteria"]
TONE_OPTS = Q9["tone"]["criteria"]
MOVES = list(MOVE_OPTS)
TONES = list(TONE_OPTS)
FAMILY = {  # 7 famílias (hierárquico)
    "react_only": "react", "agree": "react",
    "answer": "respond", "plan": "respond",
    "share_own": "contribute", "new_topic": "contribute",
    "ask_follow_up": "ask",
    "tease_back": "play", "joke_riff": "play", "flirt_back": "play", "disagree": "play",
    "empathize": "support", "reassure": "support", "compliment": "support",
    "greet_back": "ritual", "goodbye": "ritual",
}
FAMILIES = ["react", "respond", "contribute", "ask", "play", "support", "ritual"]
FAMILY_OPTS = {
    "react": "a short reaction or simple agreement, adding nothing new ('lol', 'fair', 'true', 'same', 'oh no', 'nice')",
    "respond": f"answer what {USER} asked, or confirm/propose a plan",
    "contribute": f"say something of {BOT}'s own: an opinion, own experience, news, or a new topic",
    "ask": f"ask {USER} a follow-up question about what they said",
    "play": f"banter: tease {USER} back, riff on the joke, flirt back, or playfully push back",
    "support": f"show understanding, reassure, encourage or compliment {USER}",
    "ritual": "greet back or say goodbye",
}
MODELS = {"flash": "google/gemini-3.5-flash-lite", "luna": "openai/gpt-6-luna",
          "mercury": "inception/mercury-2.5", "deepseek": "~deepseek/deepseek-flash-latest"}
# parâmetros de ator (latência de chat: sem raciocínio longo)
ACTOR_EXTRA = {"flash": {}, "luna": {"reasoning": {"effort": "minimal"}},
               "mercury": {"reasoning": {"enabled": False}}, "deepseek": {"reasoning": {"enabled": False}}}


# ------------------------------------------------------------------ io
def jl_load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def jl_save(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jdump(name, obj):
    json.dump(obj, open(os.path.join(ADATA, name), "w"), indent=1, ensure_ascii=False)


# ------------------------------------------------------------------ estado padrão (igual ao a9)
def base_state(p, ctx=CTX):
    h = p["history"][-ctx:]
    return {"conversation": [{"from": x["who"], "text": x["text"]} for x in h[:-1]],
            "last_message": {"from": USER, "text": h[-1]["text"]}}


def last_msg(p):
    return p["history"][-1]["text"]


# ------------------------------------------------------------------ Jev
def jask_many(items, workers=4):
    """items: [(state, questions)] -> answers (None em erro)."""
    return jev.ask_many(items, workers=workers)


def compact(a):
    out = {}
    for k, v in (a or {}).items():
        if v["type"] == "noul":
            out[k] = v["noul"]
        elif v["type"] == "score":
            out[k] = v["score"]; out[k + "_p"] = v["probabilities"]; out[k + "_conf"] = v["confidence"]
        else:
            out[k] = v["choice"]; out[k + "_p"] = v["probabilities"]; out[k + "_conf"] = v["confidence"]
    return out


# ------------------------------------------------------------------ LLM (4 modelos permitidos) com cache próprio
LCACHE = os.path.join(PROC, "b3_llm_cache.jsonl")
_llock, _lc = threading.Lock(), None
lstats = {"calls": 0, "cached": 0, "cost": 0.0, "by_model": {}}


def _lload():
    global _lc
    if _lc is None:
        _lc = {}
        if os.path.exists(LCACHE):
            for l in open(LCACHE, encoding="utf-8"):
                try:
                    d = json.loads(l); _lc[d["k"]] = d["r"]
                except Exception:
                    pass
    return _lc


def lchat(messages, model="flash", temperature=0.8, max_tokens=300, seed=0, extra=None, retries=5):
    """model: chave de MODELS. flash sem extra usa o llm.chat original (mesmo cache do a9)."""
    mid = MODELS[model]
    extra = ACTOR_EXTRA[model] if extra is None else extra
    if model == "flash" and not extra:
        c0 = llm.stats["cost"]
        out = llm.chat(messages, model=mid, temperature=temperature, max_tokens=max_tokens, seed=seed)
        with _llock:
            lstats["cost"] += llm.stats["cost"] - c0
            lstats["by_model"][model] = lstats["by_model"].get(model, 0) + (llm.stats["cost"] - c0)
        return out
    body = {"model": mid, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
            **extra}
    k = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    c = _lload()
    if k in c:
        lstats["cached"] += 1
        return c[k]
    for a in range(retries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=120,
                              headers={"Authorization": f"Bearer {jev._key()}"})
            if r.status_code in (429, 500, 502, 503, 529):
                raise RuntimeError(f"retryable {r.status_code}")
            r.raise_for_status()
            d = r.json()
            txt = d["choices"][0]["message"].get("content") or ""
            cost = d.get("usage", {}).get("cost", 0) or 0
            with _llock:
                lstats["calls"] += 1
                lstats["cost"] += cost
                lstats["by_model"][model] = lstats["by_model"].get(model, 0) + cost
                c[k] = txt
                with open(LCACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "r": txt}, ensure_ascii=False) + "\n")
            return txt
        except Exception as e:
            if a == retries - 1:
                print("llm error", model, str(e)[:200])
                return None
            time.sleep(min(30, 2 ** a + random.random()))


def lchat_many(items, workers=4):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda kw: lchat(**kw), items))


def chat_msgs(p, system):
    m = [{"role": "system", "content": system}]
    for h in p["history"]:
        m.append({"role": "assistant" if h["who"] == BOT else "user", "content": h["text"]})
    return m


# ------------------------------------------------------------------ bootstrap por conversa
RNG = np.random.default_rng(3)


def cboot(vals, groups, stat=np.mean, iters=2000):
    keep = [k for k, v in enumerate(vals) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not keep:
        return None
    vals = np.asarray([vals[k] for k in keep], float)
    groups = np.asarray([groups[k] for k in keep])
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    bs = []
    for _ in range(iters):
        s = RNG.choice(ug, len(ug))
        v = np.concatenate([vals[idx[g]] for g in s])
        bs.append(stat(v))
    return [round(float(stat(vals)), 4), round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4), int(len(vals))]


def cboot_diff(a, b, groups, iters=2000):
    """IC da diferença pareada média(a-b) por cluster."""
    d = [None if (x is None or y is None) else x - y for x, y in zip(a, b)]
    return cboot(d, groups, iters=iters)


def fmt(ci, pct=True, nd=1):
    if not ci:
        return "–"
    m = 100 if pct else 1
    return f"{ci[0] * m:.{nd}f} [{ci[1] * m:.{nd}f}–{ci[2] * m:.{nd}f}]"


# ------------------------------------------------------------------ elementos da última mensagem (especificidade)
STOP = set("""a an the and or but so to of in on at for with from by as is are was were be been being am i im i'm me my
mine you u ur your youre you're yours he she it its it's we us our they them their this that these those there here
what which who whom whose when where why how do does did doing done have has had having will would can could should
shall may might must not no yes yeah yea ya yep ok okay oh ah um uh hmm lol lmao haha hahaha hehe omg idk tbh rn just
like really very too also then than if because cause cos cuz about into out up down over again still even only
dont don't didnt didn't cant can't wont won't isnt isn't wasnt wasn't thats that's whats what's its gonna wanna gotta
all any some more most much many lot lots get got go going went one thing things something anything nothing
""".split())


def elements(text, max_words=12, max_phr=10):
    """Divide a última mensagem em elementos numerados: palavras de conteúdo (mensagens curtas) ou frases (longas)."""
    toks = re.findall(r"[A-Za-z][A-Za-z']*|\d+", text or "")
    cont, seen = [], set()
    for t in toks:
        tl = t.lower().strip("'")
        if (tl in STOP or len(tl) < 2 or LAUGH.fullmatch(tl or "x")
                or re.fullmatch(r"(h+m+|a+h+|o+h+|u+h+|u+m+|h+a+|o+k+a*y*|y+e+a*h*|n+o+|y+e+s+|w+o+w+|l+o+l+)", tl)):
            continue
        if tl not in seen:
            seen.add(tl); cont.append(t)
    if len(cont) <= max_words:
        return cont
    phr = [s.strip() for s in re.split(r"[\n,.!?;:]+|\s(?:and|but|because|so)\s", text) if len(s.strip()) > 1]
    return phr[:max_phr]


# ------------------------------------------------------------------ saídas por id (um arquivo por etapa, sem corrida)
def kv_path(name):
    return os.path.join(PROC, f"b3_{name}.json")


def kv_load(name):
    p = kv_path(name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def kv_save(name, d):
    tmp = kv_path(name) + ".tmp"
    json.dump(d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, kv_path(name))


def load_points(eval_only=False):
    pts = jl_load(ALL)
    extra = {n: kv_load(n) for n in ("gold_jev", "gold_elem", "gold_luna")}
    for p in pts:
        p["elements"] = elements(last_msg(p))
        if p["id"] in extra["gold_jev"]:
            p["gold"] = extra["gold_jev"][p["id"]]
        if p["id"] in extra["gold_elem"]:
            p["gold_elem"] = extra["gold_elem"][p["id"]]
        if p["id"] in extra["gold_luna"]:
            p["gold_luna"] = extra["gold_luna"][p["id"]]
    return [p for p in pts if p["eval"]] if eval_only else pts
