"""b4 — controle de saída da LLM: utilidades comuns.
Pontos: os 179 pontos de decisão do a9 (maichat; dev = 60 pontos em 6 conversas, teste = 119 em 35 conversas).
Aqui ficam: montagem das mensagens, orçamento por ponto (congelado, vindo das leituras do Jev do a9 + estilo da persona),
métricas de código (vícios), normalizador de código, segmentação em unidades, bootstrap por conversa."""
import hashlib, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
ADATA = os.path.join(ROOT, "analysis", "data")
SCR = "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad/b4"
os.makedirs(SCR, exist_ok=True)
GEN = os.path.join(SCR, "gen.jsonl")        # todas as gerações (grande; fica no scratchpad)
BOT, USER = "Sam", "Alex"
PERSONA = f"You are {BOT}, chatting with {USER} on a messaging app. Reply as {BOT}."
STATIC = (PERSONA + "\n\nText like a real person on a messaging app: casual and short, lowercase is fine, "
          "no assistant-like phrasing, match the other person's energy and style.")
MODELS = ["lite", "luna", "deepseek", "mercury"]
TH = json.load(open(os.path.join(ADATA, "a9_thresholds.json")))


def load_points():
    return [json.loads(l) for l in open(os.path.join(ADATA, "a9_points.jsonl"), encoding="utf-8")]


def msgs(p, system):
    m = [{"role": "system", "content": system}]
    for h in p["history"]:
        m.append({"role": "assistant" if h["who"] == BOT else "user", "content": h["text"]})
    return m


def last_user(p):
    return p["history"][-1]["text"]


def h01(*xs):
    """número pseudo-aleatório determinístico em [0,1) a partir de strings (sorteios reprodutíveis)."""
    return int(hashlib.md5("|".join(map(str, xs)).encode()).hexdigest()[:8], 16) / 2 ** 32


# ------------------------------------------------------------------ orçamento por ponto (código; congelado)
LAUGH = re.compile(r"(?i)(?<![a-z])(?:a?(?:ha){2,}h*|a?(?:he){2,}|lo+l+|lmf?a+o+|rofl)(?![a-z])|😂|🤣|😭|💀")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]")
EMOTICON = re.compile(r"(?:[:;=]-?[)(DPp3/])|<3|\bxd\b", re.I)


def len_words(j):
    xs, ys = TH["len_map"]
    return float(np.interp(j["length"], xs, ys))


def budget(p):
    """Alvos do momento a partir da leitura do Jev do a9 (feita sem ver a resposta) + impressão digital da persona.
    Tudo em código; limiares do a9 (calibrados no dev e congelados)."""
    j, fp = p["jev"], p["fp"]
    own = max(2.0, fp["median_words_per_bubble"] * max(1.0, fp["bubbles_per_turn"]))
    tw = int(round(max(2, 0.6 * len_words(j) + 0.4 * own))) + 1  # +1: o a9 mostrou que o alvo puro fica curto
    if j["move"] == "react_only":
        tw = min(tw, 4)
    lu = last_user(p)
    serious = j["seriousness"] >= 1.8 or (j["vulnerable"] * (1 - j["teasing"]) >= 0.55)
    # pergunta: limiar do a9 (fraco, AUC 0,57) + sorteio de base, para a taxa total ficar perto da humana (~12%)
    q = (j["p_question"] >= TH["q"] and j["move"] != "react_only") or h01(p["id"], "q") < 0.07
    laugh = (bool(LAUGH.search(lu)) and not serious) or (j["p_laugh"] >= TH["laugh"] and not serious)
    emoji = j["p_emoji"] >= TH["emoji"] and fp["emoji_frac"] >= 0.05
    excl = ("!" in lu and h01(p["id"], "excl") < 0.5) or h01(p["id"], "excl2") < 0.05
    return {"words": tw, "q": q, "laugh": laugh, "emoji": emoji, "excl": excl, "serious": serious,
            "lower": fp["lower_frac"] > 0.7, "noperiod": fp["period_frac"] < 0.2,
            "laugh_token": fp.get("laugh_token") or "lol", "bubbles": 2 if (fp["bubbles_per_turn"] >= 1.5 and tw >= 7) else 1}


# ------------------------------------------------------------------ métricas de código (vícios)
LLMISH = ["absolutely", "totally", "definitely", "honestly", "amazing", "awesome", "fantastic", "wonderful",
          "sounds like", "sounds (?:great|fun|good|amazing|awesome|lovely|perfect)", "that sounds", "i'd love", "i would love",
          "aww+", "journey", "i hear you", "i'm here for you", "i'm so glad", "no worries", "vibe", "vibes", "vibing",
          "that's so", "super", "for sure", "love that", "i love that", "such a", "oh my gosh", "hope you", "feel free",
          "glad to hear", "exciting", "adorable", "energy", "haha,? (?:that's|you're|i)", "tell me (?:more|everything)",
          "sorry to hear", "sorry you're", "i totally get", "can't wait", "let me know", "so much fun", "a blast",
          "right\\?", "it's (?:okay|ok|normal|valid) to", "proud of you", "that's so sweet"]
LLMISH_RE = [(w, re.compile(r"(?i)(?<![a-z])" + w + r"(?![a-z])")) for w in LLMISH]
DASH = re.compile("[—–]")
PERF = re.compile(r"(?i)^\W*(omg|wait|ooh+|oo+h|lol|haha\w*|lmf?ao+|hey!|aww+|ugh|honestly|stop|bro|literally|oh no|yay+|"
                  r"yesss+|nooo+|wow|oh wow|no way|ahh+|yess+)\b")
REC = re.compile(r"(?i)(what about you|wbu|how about you|and you\?|\bu\?|you\?$|how's your day|how was your day|"
                 r"what's up|whats up|how are you|what are you up to|what are u up to|hbu|how's it going|how you doing)")
ART = re.compile(r"</?p\b|\*[a-z][^*]{2,}\*|\bAlex:|\bSam:|<\|")
REACT = re.compile(r"(?i)^\W*(?:oh|aw+|wow|omg|haha\w*|lol|yay|ugh|no way|wait|hey|hi|yes+|yeah|totally|absolutely|definitely|"
                   r"i totally|i know|i'm (?:so )?sorry|that's|that is|that sounds|sounds|i get|i understand|same|right|nice|"
                   r"great|amazing|awesome|congrat)")
STOP = set("""a an the and or but so to of in on at for with is are was were be been am i you he she it we they me my your
his her its our their this that these those do does did not no yes just like oh ok okay im i'm its it's dont don't
what how why when where who u ur r lol haha yeah ya yea be have has had will would can could should there here
then than too very really also if about up out get got go going one all some any more""".split())
OPEN_END = re.compile(r"(?i)\b(the|a|an|to|and|or|but|of|my|your|with|for|in|on|at|i'm|im|is|are|was|so|that|if|just|"
                      r"like|when|because|about|this|it's|its|you're|we|i)\s*$")


def words(t):
    return re.findall(r"[A-Za-z0-9']+", t or "")


def sentences(t):
    parts = re.split(r"(?<=[.!?…])\s+|\n+", (t or "").strip())
    return [s for s in parts if re.search(r"\w", s)]


def content_words(t):
    return {w for w in re.findall(r"[a-z']+", (t or "").lower()) if w not in STOP and len(w) > 2}


def feats(t, p=None, finish=None):
    t = (t or "").strip()
    s = sentences(t)
    lu = last_user(p) if p else ""
    cw, pw = content_words(t), content_words(lu)
    llm_hits = [w for w, rx in LLMISH_RE if rx.search(t)]
    first = s[0] if s else t
    return {
        "words": len(words(t)), "chars": len(t), "sents": len(s), "ge3": len(s) >= 3,
        "q": "?" in t, "ends_q": t.rstrip().endswith("?"),
        "excl": "!" in t, "emoji": bool(EMOJI.search(t) or EMOTICON.search(t)), "laugh": bool(LAUGH.search(t)),
        "llmish": bool(llm_hits) or bool(DASH.search(t)), "llmish_hits": llm_hits,
        "template": len(s) >= 3 and bool(REACT.search(s[0])) and s[-1].rstrip().endswith("?"),
        "react_q": len(s) >= 2 and bool(REACT.search(s[0])) and s[-1].rstrip().endswith("?"),
        "echo2": len(cw & pw) >= 2, "perf": bool(PERF.search(first)), "recip": bool(REC.search(t)),
        "artefact": bool(ART.search(t)), "empty": not bool(words(t)),
        "broken": bool(t) and (finish == "length" or bool(OPEN_END.search(t))),
        "nl": t.count("\n"),
    }


RATE_KEYS = ["q", "excl", "emoji", "laugh", "llmish", "ge3", "template", "echo2", "perf", "recip"]


def lenerr(tw, hw):
    return abs(np.log2((tw + 1) / (hw + 1)))


# ------------------------------------------------------------------ normalizador de código (sem LLM, sem Jev)
OPENER_STRIP = re.compile(r"(?i)^\W*(aww+|omg+|oh wow|wow|ooh+|haha+h*|hahah\w*|lol|lmao+|yay+|totally|absolutely|"
                          r"definitely|honestly|oh my gosh|no way|that's (?:so )?(?:amazing|awesome|great|cool)|"
                          r"that sounds (?:amazing|awesome|great|fun)|wait)\b[\s,.!…]*")


def normalize(t, b, strip_openers=True):
    """C1: correções de superfície em código. b = orçamento do ponto."""
    t = (t or "").strip()
    t = re.sub(r"</?p[^>]*>", " ", t)
    t = re.sub(r"(?m)^\s*\**(Sam|Alex)\**\s*:\s*", "", t)
    t = re.sub(r"\*[a-z][^*]{2,}\*", "", t)
    t = re.sub(r"<\|[^|]*\|>", "", t)
    t = DASH.sub(", ", t)
    t = re.sub(r"(?i),?\s*\balex\b\s*,?", " ", t) if re.search(r"(?i)\balex\b", t) else t
    if not b["excl"]:
        t = re.sub(r"!+", lambda m: "", t)
    else:
        t = re.sub(r"!{2,}", "!", t)
    if not b["emoji"]:
        t = EMOJI.sub("", t).replace("‍", "").replace("️", "")
    else:  # no máximo 1 emoji
        es = EMOJI.findall(t)
        if len(es) > 1:
            first = True
            def keep1(m):
                nonlocal first
                if first:
                    first = False
                    return m.group(0)
                return ""
            t = EMOJI.sub(keep1, t)
    if strip_openers:
        m = OPENER_STRIP.match(t)
        if m and len(words(t[m.end():])) >= 2:
            if not (b["laugh"] and LAUGH.match(m.group(0).strip())):
                t = t[m.end():]
    if not b["laugh"]:
        t2 = LAUGH.sub("", t)
        if len(words(t2)) >= 1:
            t = t2
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\s+([,.?])", r"\1", t)
    t = re.sub(r"^[\s,.]+", "", t)
    t = re.sub(r"\n\s*\n+", "\n", t).strip()
    if b["lower"] and t:
        t = "\n".join((ln[:1].lower() + ln[1:]) if ln and not ln[:2].isupper() else ln for ln in t.split("\n"))
    if b["noperiod"]:
        t = re.sub(r"(?<!\.)\.\s*$", "", t)
        t = re.sub(r"(?<!\.)\.\s*\n", "\n", t)
    return t.strip(" ,")


# ------------------------------------------------------------------ segmentação em unidades (frases/cláusulas)
def units(t):
    """Quebra a resposta em unidades: linhas, depois frases; separa a pergunta final se vier colada por vírgula."""
    out = []
    for ln in re.split(r"\n+", (t or "").strip()):
        for s in re.split(r"(?<=[.!?…])\s+", ln.strip()):
            s = s.strip()
            if not re.search(r"\w", s):
                if out and s:  # emoji solto: gruda na anterior
                    out[-1] = out[-1] + " " + s
                continue
            out.append(s)
    return out


def join_units(us):
    return " ".join(u.strip() for u in us if u.strip())


# ------------------------------------------------------------------ estatística
RNG = np.random.default_rng(4)


def cboot(vals, groups, stat=np.mean, iters=2000):
    keep = [k for k, v in enumerate(vals) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not keep:
        return None
    v0 = np.asarray([vals[k] for k in keep], float)
    g0 = np.asarray([groups[k] for k in keep])
    ug = np.unique(g0)
    idx = {g: np.where(g0 == g)[0] for g in ug}
    bs = []
    for _ in range(iters):
        s = RNG.choice(ug, len(ug))
        bs.append(stat(np.concatenate([v0[idx[g]] for g in s])))
    return [round(float(stat(v0)), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), int(len(v0))]


def load_gen():
    """{(model, cond, point_id): rec}. rec = {text, finish, cost, latency, calls, ...}"""
    out = {}
    if os.path.exists(GEN):
        for l in open(GEN, encoding="utf-8"):
            d = json.loads(l)
            out[(d["model"], d["cond"], d["pid"])] = d
    return out


def append_gen(recs):
    with open(GEN, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
