"""a9 — experimento A/B: LLM pura x LLM + briefing do Jev x humano real (maichat).
Utilidades compartilhadas: carga dos turnos anotados, formatação de contexto, impressão digital de estilo,
métricas de código e wrappers com medição de latência/custo."""
import json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import jev  # noqa: E402
import llm  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
ADATA = os.path.join(ROOT, "analysis", "data")
POINTS = os.path.join(ADATA, "a9_points.jsonl")
BOT, USER = "Sam", "Alex"
GEN_MODEL = "google/gemini-3.5-flash-lite"
STRONG_MODEL = "anthropic/claude-haiku-4.5"
JUDGE_MODEL = "openai/gpt-4o-mini"
PERSONA = f"You are {BOT}, chatting with {USER} on a messaging app. Reply as {BOT}."


# ---------------------------------------------------------------- dados
def load_mai():
    rows = []
    for l in open(os.path.join(PROC, "jev_base.jsonl"), encoding="utf-8"):
        if not l.startswith('{"corpus": "maichat"'):
            continue
        rows.append(json.loads(l))
    rows.sort(key=lambda r: (r["conv_id"], r["turn_idx"]))
    return rows


def turn_text(texts):
    return "\n".join(t.strip() for t in texts if t and t.strip())


# ---------------------------------------------------------------- features de código
LAUGH = re.compile(r"(?i)(?<![a-z])(?:a?(?:ha){2,}h*|a?(?:he){2,}|lo+l+|lmf?a+o+|rofl)(?![a-z])|😂|🤣|😭|💀")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]")
EMOTICON = re.compile(r"(?:[:;=]-?[)(DPp3/])|<3|\bxd\b", re.I)
# Lista a priori de marcas "LLM-ish" (assistente/rebuscado). O travessão conta à parte.
LLMISH = ["absolutely", "totally", "definitely", "honestly", "amazing", "awesome", "fantastic", "wonderful",
          "sounds like", "sounds (?:great|fun|good|amazing|awesome|lovely|perfect)", "that sounds", "i'd love", "i would love",
          "aww+", "journey", "i hear you", "i'm here for you", "i'm so glad", "no worries", "vibe", "vibes", "vibing",
          "that's so", "super", "for sure", "love that", "i love that", "such a", "oh my gosh", "hope you", "feel free",
          "glad to hear", "exciting", "adorable", "energy", "haha,? (?:that's|you're|i)", "tell me (?:more|everything)"]
LLMISH_RE = [(w, re.compile(r"(?i)(?<![a-z])" + w + r"(?![a-z])")) for w in LLMISH]
DASH = re.compile("[—–]")  # travessão / meia-risca
# gíria "de internet" (para medir o extremo oposto: gíria forçada / caricata)
SLANG = ["fr", "ngl", "tbh", "tbf", "icl", "lowkey", "highkey", "no cap", "bestie", "slay", "periodt", "bruh", "omg",
         "istg", "deadass", "bet", "sus", "vibe", "vibes", "vibing", "fam", "yass+", "af", "rn", "imo", "idk", "lmk", "ikr"]
SLANG_RE = [(w, re.compile(r"(?i)(?<![a-z'])" + w + r"(?![a-z'])")) for w in SLANG]


def words(t):
    return re.findall(r"[A-Za-z0-9']+", t or "")


def bubbles(t):
    return [b for b in re.split(r"\n+", (t or "").strip()) if b.strip()]


def feats(t):
    t = t or ""
    bs = bubbles(t)
    llm_hits = [w for w, rx in LLMISH_RE if rx.search(t)]
    slang_hits = [w for w, rx in SLANG_RE if rx.search(t)]
    letters = [b.strip()[0] for b in bs if b.strip() and b.strip()[0].isalpha()]
    return {
        "n_words": len(words(t)), "n_chars": len(t), "n_bubbles": max(1, len(bs)),
        "has_q": "?" in t, "laugh": bool(LAUGH.search(t)), "emoji": bool(EMOJI.search(t) or EMOTICON.search(t)),
        "n_excl": t.count("!"), "excl": "!" in t, "dash": bool(DASH.search(t)),
        "llmish": len(llm_hits) > 0 or bool(DASH.search(t)), "llmish_n": len(llm_hits) + len(DASH.findall(t)),
        "llmish_hits": llm_hits, "slang_n": len(slang_hits), "slang_hits": slang_hits,
        "starts_lower": bool(letters) and all(c.islower() for c in letters),
        "ends_period": bool(bs) and bs[-1].rstrip().endswith("."),
        "apostrophe_ok": "'" in t or "’" in t,
    }


def style_fingerprint(own_texts, other_texts):
    """Impressão digital de estilo do BOT a partir das próprias mensagens anteriores (e do parceiro, como reserva)."""
    msgs = [b for t in own_texts for b in bubbles(t)]
    omsgs = [b for t in other_texts for b in bubbles(t)]
    fp = {"n_msgs": len(msgs)}
    if not msgs:
        msgs = omsgs
    import statistics as st
    lens = [len(words(m)) for m in msgs] or [5]
    letters = [m.strip()[0] for m in msgs if m.strip() and m.strip()[0].isalpha()]
    fp["median_words_per_bubble"] = st.median(lens)
    fp["lower_frac"] = sum(c.islower() for c in letters) / max(1, len(letters))
    fp["period_frac"] = sum(m.rstrip().endswith(".") for m in msgs) / max(1, len(msgs))
    fp["emoji_frac"] = sum(bool(EMOJI.search(m)) for m in msgs) / max(1, len(msgs))
    fp["bubbles_per_turn"] = (len(msgs) / max(1, len(own_texts))) if own_texts else 1.0
    lt = {}
    for m in msgs + omsgs:
        for x in LAUGH.findall(m):
            lt[x.lower()] = lt.get(x.lower(), 0) + (2 if m in msgs else 1)
    fp["laugh_token"] = max(lt, key=lt.get) if lt else None
    sl = {}
    for m in msgs:
        for w, rx in SLANG_RE:
            if rx.search(m):
                sl[w] = sl.get(w, 0) + 1
    fp["own_slang"] = sorted(sl, key=sl.get, reverse=True)[:3]
    return fp


def clean_reply(t):
    """Remove prefixos tipo 'Sam:' e aspas envolventes que o modelo às vezes coloca."""
    t = (t or "").strip()
    t = re.sub(rf"^\s*\**{BOT}\**\s*:\s*", "", t)
    if len(t) > 2 and t[0] == t[-1] == '"':
        t = t[1:-1]
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


# ---------------------------------------------------------------- chamadas medidas
_tl = threading.Lock()


def timed_chat(kw):
    t0 = time.time()
    try:
        out = llm.chat(**kw)
    except Exception as e:  # noqa
        print("llm error:", str(e)[:200])
        return None, None
    dt = time.time() - t0
    return out, (dt if dt > 0.05 else None)  # None = veio do cache


def chat_many_timed(items, workers=4):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(timed_chat, items))


def timed_ask(it):
    t0 = time.time()
    try:
        out = jev.ask(*it)
    except Exception as e:  # noqa
        print("jev error:", str(e)[:200])
        return None, None
    dt = time.time() - t0
    return out, (dt if dt > 0.05 else None)


def ask_many_timed(items, workers=4):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(timed_ask, items))


def load_points():
    return [json.loads(l) for l in open(POINTS, encoding="utf-8")]


def save_points(pts):
    with open(POINTS, "w", encoding="utf-8") as f:
        for p in pts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def merge_json(path, upd):
    """Atualiza um json de metadados (latência/custo) sem perder medições anteriores."""
    d = json.load(open(path)) if os.path.exists(path) else {}
    for k, v in upd.items():
        if isinstance(v, dict):
            d.setdefault(k, {})
            for kk, vv in v.items():
                if vv is not None:
                    d[k][kk] = vv
        elif v is not None:
            d[k] = v
    json.dump(d, open(path, "w"), indent=1, ensure_ascii=False)
    return d
