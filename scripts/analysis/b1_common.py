"""b1: shared helpers for the 'LLM-face' judge architecture lab.

Unit = one candidate reply (human or LLM) at a real decision point.
Sources:
  a8: 250 contexts (200 maichat + 50 empathetic) x {human, base_gemini, base_gpt4omini, base_llama70b, styled_gemini}
      + new (b1_gen.py): base_luna, styled_luna, base_deepseek, styled_deepseek   (Maya/Jordan naming)
  a9: 179 maichat points x {human, A (gemini base), S (gemini static style), B (gemini + Jev brief), D (haiku-4.5)}
      (Sam/Alex naming)
Split: dev/test by conversation (maichat conv_id; each empathetic conv is its own cluster).
"""
import json, math, os, pickle, random, re, sys, time
from collections import defaultdict, Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "analysis", "data")
SCRATCH = "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad"
SCR = os.path.join(SCRATCH, "b1")
os.makedirs(SCR, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
import a8_common as C8  # noqa: E402

A8CTX = os.path.join(SCRATCH, "a8", "contexts.json")
A9PTS = os.path.join(OUT, "a9_points.jsonl")

A8_CONDS = ["base_gemini", "base_gpt4omini", "base_llama70b", "styled_gemini",
            "base_luna", "styled_luna", "base_deepseek", "styled_deepseek"]
A9_CONDS = ["A", "S", "B", "D"]
FAMILY = {"base_gemini": "gemini", "styled_gemini": "gemini", "A": "gemini", "S": "gemini", "B": "gemini",
          "base_gpt4omini": "gpt4omini", "base_llama70b": "llama70b", "D": "haiku45",
          "base_luna": "gpt6luna", "styled_luna": "gpt6luna", "base_deepseek": "deepseek", "styled_deepseek": "deepseek"}
INSTRUCTED = {"styled_gemini", "styled_luna", "styled_deepseek", "S", "B"}
STRONG = {"base_luna", "styled_luna", "base_deepseek", "styled_deepseek", "D"}


def norm_text(t):
    t = (t or "").replace("\r", "")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def split_of(conv):
    """Deterministic dev/test split by conversation (≈50/50)."""
    h = int(__import__("hashlib").md5(("b1split:" + conv).encode()).hexdigest(), 16)
    return "dev" if h % 2 == 0 else "test"


def _turns_maichat():
    return pickle.load(open(os.path.join(SCRATCH, "a8", "turns_maichat.pkl"), "rb"))


def load_units(include_new=True):
    """Returns (contexts, units). contexts[ckey] = {history[(name,text)], self, other, conv, src, split, turn_idx, speaker};
    units = list of dicts {uid, ckey, cond, label(1=LLM), family, text}."""
    ctxs, units = {}, []
    X = json.load(open(A8CTX))
    gens = {}
    for cn in A8_CONDS:
        p = os.path.join(SCRATCH, "a8", f"gen_{cn}.json") if not cn.endswith(("luna", "deepseek")) else os.path.join(SCR, f"gen_{cn}.json")
        if os.path.exists(p):
            gens[cn] = json.load(open(p))
    for c in X:
        ck = "a8:" + c["id"]
        hist = []
        for h in c["history"][-8:]:
            who = "Maya" if h["who"] == "bot" else "Jordan"
            txt = "\n".join(t.strip() for t in h["texts"] if t.strip())
            if txt:
                hist.append((who, txt))
        conv = c["conv"] if c["src"] == "maichat" else "ed_" + str(c["conv"])
        tidx = int(c["id"].split("_")[-1]) if c["src"] == "maichat" else None
        ctxs[ck] = {"history": hist, "self": "Maya", "other": "Jordan", "conv": conv, "src": "a8_" + c["src"],
                    "split": split_of(conv), "turn_idx": tidx, "speaker": c.get("bot_speaker"), "bucket": c["bucket"]}
        units.append({"uid": ck + "|human", "ckey": ck, "cond": "human", "label": 0, "family": "human",
                      "text": norm_text("\n".join(t.strip() for t in c["human"] if t.strip()))})
        for cn, G in gens.items():
            if G.get(c["id"]):
                units.append({"uid": ck + "|" + cn, "ckey": ck, "cond": cn, "label": 1, "family": FAMILY[cn], "text": norm_text(G[c["id"]])})
    T = None
    for l in open(A9PTS):
        p = json.loads(l)
        ck = "a9:" + p["id"]
        hist = [(h["who"], norm_text(h["text"])) for h in p["history"][-8:] if norm_text(h["text"])]
        if T is None:
            T = {(t["conv_id"], t["turn_idx"]): t for t in _turns_maichat()}
        spk = T.get((p["conv_id"], p["turn_idx"]), {}).get("speaker")
        ctxs[ck] = {"history": hist, "self": "Sam", "other": "Alex", "conv": p["conv_id"], "src": "a9", "split": split_of(p["conv_id"]),
                    "turn_idx": p["turn_idx"], "speaker": spk, "bucket": p["stratum"]}
        units.append({"uid": ck + "|human", "ckey": ck, "cond": "human", "label": 0, "family": "human", "text": norm_text(p["human"])})
        for cn in A9_CONDS:
            if cn in ("B",) and p.get("fallback"):
                continue
            t = p.get("gen", {}).get(cn)
            if t:
                units.append({"uid": ck + "|" + cn, "ckey": ck, "cond": cn, "label": 1, "family": FAMILY[cn], "text": norm_text(t)})
    units = [u for u in units if u["text"]]
    return ctxs, units


def last_other(ctx):
    """Text of the other person's last turn (the message being replied to)."""
    for who, t in reversed(ctx["history"]):
        if who == ctx["other"]:
            return t
    return ctx["history"][-1][1] if ctx["history"] else ""


def convo_lines(ctx, max_turns=8):
    return [{"from": w, "text": t} for w, t in ctx["history"][-max_turns:]]


def base_state(ctx, text, extra=None):
    """Canonical state for per-candidate questions (JSON object, named fields)."""
    s = {"setting": f"Casual one-to-one text chat on a messaging app between {ctx['other']} and {ctx['self']}.",
         "conversation_so_far": convo_lines(ctx),
         "candidate_message": {"from": ctx["self"], "text": text}}
    if extra:
        s = {**extra, **s} if extra.get("_first") else {**s, **extra}
        s.pop("_first", None)
    return s


# ---------------------------------------------------------------- code features
PERF_OPEN = re.compile(r"^\W*(?:omg+|oh+ (?:my|wow|no)|wow+|wait+|ooh+|aw+|haha\w*|lol|lmao|honestly|ugh+|yay+|hey!|stop|literally|okay so|ok so|oh+,)\b", re.I)
GEN_RECIP = re.compile(r"\b(?:what|how) about you\b|\band you\??\s*$|\bhbu\b|\bwbu\b|\byou\?\s*$|how(?:'s| is| was) (?:your|ur) (?:day|night|week|evening|morning)|what(?:'s| is) up\b|what are you up to|how are you\b|hbu\??", re.I)
REACT = re.compile(r"^\W*(?:oh|aw+|wow|omg|haha\w*|lol|yay|ugh|no way|wait|hey|hi|yes+|yeah|totally|absolutely|definitely|i totally|i know|i'm (?:so )?sorry|that's|that is|that sounds|sounds|i get|i understand|same|right|nice|great|amazing|awesome|congrat)", re.I)
VALID = re.compile(r"\bthat(?:'|’)?s (?:so |really |absolutely )?(?:amazing|awesome|great|wonderful|fantastic|incredible|exciting|sweet|valid|fair|tough|rough|hard)|\bsounds (?:like|amazing|fun|great|awesome|good|lovely|perfect|exciting|intense|rough|tough|hard)\b|\bsorry to hear\b|\bsorry you(?:'|’)?re\b|\bi (?:totally |completely )?(?:get|understand) (?:that|it|you)\b|\bit(?:'|’)?s (?:totally |completely )?(?:okay|ok|normal|understandable|valid) to\b|\bhere for you\b|\bproud of you\b|\byou(?:'|’)?ve got this\b|\byou got this\b|\bso happy for you\b|\bcongrat", re.I)
NAMES_RX = re.compile(r"\b(?:jordan|alex)\b", re.I)


def code_feats(text, other_text=""):
    f = C8.style_feats(text, other_text)
    t = text.strip()
    s = C8.sentences(t)
    out = {
        "log_chars": math.log1p(f["chars"]), "words": f["words"], "sentences": f["sentences"], "lines": f["lines"],
        "ends_q": int(f["ends_q"]), "any_q": int(f["any_q"]), "n_q": t.count("?"), "n_excl": min(5, f["n_excl"]),
        "n_emoji": min(5, f["n_emoji"]), "laugh": int(f["laugh"]), "final_punct": int(f["final_punct"]),
        "final_period": int(f["final_period"]), "starts_lower": int(f["starts_lower"]), "all_lower": int(f["all_lower"]),
        "elongation": int(f["elongation"]), "no_apostrophe": int(f["no_apostrophe"]), "le3_words": int(f["le3_words"]),
        "echo_ratio": f["echo_ratio"], "echo_any2": int(f["echo_any2"]),
        "llmish_hits": sum(v for k, v in f.items() if k.startswith("llm:")),
        "perf_open": int(bool(PERF_OPEN.search(t))), "gen_recip_q": int(bool(GEN_RECIP.search(t))),
        "template3": int(len(s) >= 3 and bool(REACT.search(s[0])) and s[-1].rstrip().endswith("?")),
        "react_then_q": int(len(s) >= 2 and bool(REACT.search(s[0])) and s[-1].rstrip().endswith("?")),
        "validation_rx": int(bool(VALID.search(t))), "vocative": int(bool(NAMES_RX.search(t))),
        "n_commas": t.count(","), "apostrophe": int("'" in t or "’" in t), "caps_start": int(bool(t) and t[0].isupper()),
        "len_ratio_other": math.log2((len(t) + 5) / (len(other_text or "") + 5)),
        "mean_word_len": float(np.mean([len(w) for w in t.split()])) if t.split() else 0.0,
    }
    return out


CODE_COLS = ["log_chars", "words", "sentences", "lines", "ends_q", "any_q", "n_q", "n_excl", "n_emoji", "laugh", "final_punct",
             "final_period", "starts_lower", "all_lower", "elongation", "no_apostrophe", "le3_words", "echo_ratio", "echo_any2",
             "llmish_hits", "perf_open", "gen_recip_q", "template3", "react_then_q", "validation_rx", "vocative", "n_commas",
             "apostrophe", "caps_start", "len_ratio_other", "mean_word_len"]
LEN_COLS = ["log_chars", "words", "sentences", "lines", "le3_words", "len_ratio_other"]


# ---------------------------------------------------------------- metrics
def auc(pos, neg):
    """P(score_pos > score_neg), ties 0.5. pos = LLM scores, neg = human scores (higher = more LLM)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    vals, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


LEN_EDGES = [0, 12, 25, 45, 80, 10 ** 9]  # chars; fixed bins (human median ~30, LLM ~80)


def len_bin(n):
    for i in range(len(LEN_EDGES) - 1):
        if LEN_EDGES[i] <= n < LEN_EDGES[i + 1]:
            return i
    return len(LEN_EDGES) - 2


def strat_auc(scores, labels, lens, min_n=5):
    """Length-controlled AUC: AUC within fixed char-length bins, averaged with weights n_pos*n_neg."""
    scores, labels, lens = np.asarray(scores, float), np.asarray(labels), np.asarray(lens)
    b = np.array([len_bin(x) for x in lens])
    num = den = 0.0
    for k in np.unique(b):
        m = b == k
        p, n = scores[m & (labels == 1)], scores[m & (labels == 0)]
        if len(p) >= min_n and len(n) >= min_n:
            w = len(p) * len(n)
            num += auc(p, n) * w; den += w
    return num / den if den else float("nan")


def plain_auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    return auc(scores[labels == 1], scores[labels == 0])


def cluster_boot(fn, clusters, n=1000, seed=0):
    """fn(idx_array) -> stat; resample clusters with replacement. Returns (lo, hi)."""
    clusters = np.asarray(clusters)
    uc = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uc}
    r = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = r.choice(uc, len(uc))
        idx = np.concatenate([idx_by[c] for c in pick])
        v = fn(idx)
        if v == v:
            vals.append(v)
    return (round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)) if vals else (None, None)


def eval_scores(scores, labels, lens, clusters, boot=500):
    scores, labels, lens, clusters = map(np.asarray, (scores, labels, lens, clusters))
    a = plain_auc(scores, labels)
    s = strat_auc(scores, labels, lens)
    lo, hi = cluster_boot(lambda i: plain_auc(scores[i], labels[i]), clusters, boot)
    slo, shi = cluster_boot(lambda i: strat_auc(scores[i], labels[i], lens[i]), clusters, boot, seed=1)
    return {"auc": round(a, 3), "ci": [lo, hi], "auc_len": round(s, 3), "ci_len": [slo, shi], "n_llm": int((labels == 1).sum()), "n_hum": int((labels == 0).sum())}


def save(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("saved", p, os.path.getsize(p))


# ---------------------------------------------------------------- Jev bookkeeping per architecture
import jev  # noqa: E402

COSTLOG = os.path.join(SCR, "costs.jsonl")


def run_jev(arch, items, workers=4):
    """items: list of (state, questions). Logs calls/cost/latency for this batch under `arch`."""
    s0 = dict(calls=jev.stats["calls"], cost=jev.stats["cost"], tok=jev.stats["input_tokens"], nlat=len(jev.stats["latency_s"]))
    t0 = time.time()
    res = jev.ask_many(items, workers=workers)
    lat = jev.stats["latency_s"][s0["nlat"]:]
    rec = {"arch": arch, "n_items": len(items), "new_calls": jev.stats["calls"] - s0["calls"], "cost": jev.stats["cost"] - s0["cost"],
           "input_tokens": jev.stats["input_tokens"] - s0["tok"], "lat_p50": float(np.median(lat)) if lat else None,
           "lat_p90": float(np.percentile(lat, 90)) if lat else None, "n_questions": len(items[0][1]) if items else 0,
           "wall_s": round(time.time() - t0, 1), "errors": sum(r is None for r in res)}
    with open(COSTLOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("JEV", rec)
    return res


def val(a):
    if a is None:
        return float("nan")
    t = a["type"]
    return a["noul"] if t == "noul" else a["score"] if t == "score" else a["choice"]


CORE_A8 = ["human", "base_luna", "styled_luna", "styled_deepseek"]
CORE_A9 = ["human", "B"]


def core_units(ctxs, units):
    """Subset used by the state-variant architectures (cost control): maichat only."""
    out = []
    has_b = {u["ckey"] for u in units if u["cond"] == "B"}
    for u in units:
        c = ctxs[u["ckey"]]
        if c["src"] == "a8_maichat" and u["cond"] in CORE_A8:
            out.append(u)
        elif c["src"] == "a9" and u["cond"] in CORE_A9 and u["ckey"] in has_b:
            out.append(u)
    return out
