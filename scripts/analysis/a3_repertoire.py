"""a3: REPERTÓRIO real por momento emocional — o que as pessoas escrevem (aberturas mais comuns, tamanho, estrutura)
ao consolar, ao reagir a notícia boa (EmpatheticDialogues, 24,8k conversas) e ao reagir a piada / momento sério
(maichat + whatsapp_nl, jev_base). Também: qual abertura faz A continuar falando mais (A2 mais longo).
Saída: analysis/data/a3_repertoire.json"""
import json, os, re, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from a3_common import *
R = {}
MOMENTS = {"consolar_tristeza": {"sadness"}, "consolar_medo": {"fear_anxiety"}, "raiva_indignacao": {"anger_disgust"},
           "vergonha_culpa": {"shame_guilt"}, "noticia_boa": {"joy_pride"}, "expectativa_esperanca": {"hope_confidence"},
           "nostalgia_ternura": {"warm_nostalgic"}}

def norm(t):
    t = t.replace("_comma_", ",").strip()
    return re.sub(r"\s+", " ", t)

def opener(t, n):
    w = re.findall(r"[a-z']+", t.lower())
    return " ".join(w[:n])

def first_sentence(t):
    m = re.match(r"^(.+?[.!?]+)(\s|$)", t)
    return m.group(1) if m else t

INTERJ = r"^(oh+|wow+|aw+|ah+|ooh+|yikes|omg|oh no|oh my|ugh|uh oh|yay|nice|cool|awesome|great|congrat\w*|that'?s|i'?m so|sorry|really|haha\w*|lol|yes|yeah|no way|damn|geez|jeez|whoa|good)"
convs = [json.loads(l) for l in open(os.path.join(SCR, "empathetic_convs.jsonl"))]
rows = []
for c in convs:
    u = c["utts"]
    if len(u) < 3 or u[1]["speaker"] != "B": continue
    b = norm(u[1]["text"]); a2 = norm(u[2]["text"])
    fs = first_sentence(b)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", b) if s.strip()]
    rows.append({"fam": FAM_OF[c["gold"]], "gold": c["gold"], "b": b, "a1": norm(u[0]["text"]),
                 "o1": opener(b, 1), "o2": opener(b, 2), "o3": opener(b, 3),
                 "words": len(b.split()), "n_sent": len(sents), "q": "?" in b, "ends_q": b.rstrip().endswith("?"),
                 "fs_short_react": len(fs.split()) <= 5 and not fs.endswith("?") and len(sents) > 1,
                 "starts_interj": bool(re.match(INTERJ, b.lower())),
                 "a2_words": len(a2.split())})
X = pd.DataFrame(rows)
X["template_react_then_q"] = X.fs_short_react & X.ends_q
X["only_question"] = (X.n_sent == 1) & X.ends_q
R["n"] = len(X)
rep = {}
for m, fams in MOMENTS.items():
    g = X[X.fam.isin(fams)]
    rep[m] = {"n": len(g), "words_median": float(g.words.median()), "words_p25_p75": g.words.quantile([.25, .75]).tolist(),
              "n_sent_median": float(g.n_sent.median()), "has_q": round(g.q.mean(), 3), "ends_with_q": round(g.ends_q.mean(), 3),
              "template_short_reaction_then_question": round(g.template_react_then_q.mean(), 3),
              "only_a_question": round(g.only_question.mean(), 3), "starts_with_interjection": round(g.starts_interj.mean(), 3),
              "top_first_word": [f"{k} {v/len(g):.1%}" for k, v in Counter(g.o1).most_common(12)],
              "top_first_2words": [f"{k} {v/len(g):.1%}" for k, v in Counter(g.o2).most_common(15)],
              "top_first_3words": [f"{k} {v/len(g):.1%}" for k, v in Counter(g.o3).most_common(12)],
              "examples_typical": g[(g.words.between(g.words.quantile(.4), g.words.quantile(.6)))].sample(6, random_state=3).b.str.slice(0, 120).tolist()}
R["ed_repertoire"] = rep
# padrões de frase (regex) por momento e efeito no engajamento de A (palavras em A2, relativo à média do momento)
PHR = {
    "oh no": r"^oh no\b", "i'm so sorry / sorry to hear": r"\b(i'?m so sorry|sorry to hear|i'?m sorry)\b",
    "that sucks / that's terrible": r"\b(that sucks|that'?s (terrible|awful|horrible|too bad|so sad|rough|a shame))\b",
    "i can imagine / must be": r"\b(i can imagine|must (have been|be))\b",
    "i understand": r"\bi (can )?understand\b", "don't worry / it'll be ok": r"\b(don'?t worry|it'?ll be (ok|okay|fine|alright))\b",
    "you should / maybe you": r"\b(you should|maybe you should|have you (tried|thought|considered))\b",
    "me too / same": r"\b(me too|same here|i know the feeling|i know how you feel)\b",
    "hope": r"\b(i hope|hope (it|you|everything|that))\b",
    "congrats": r"\bcongrat", "that's awesome/great": r"\bthat'?s (awesome|great|amazing|wonderful|fantastic|so cool|cool|nice|good)\b",
    "wow": r"^wow\b", "good for you": r"\bgood for you\b", "proud of you": r"\bproud of you\b",
    "what happened?": r"\bwhat happened\b", "how did/do you feel": r"\bhow (did|do) you feel\b",
    "why": r"^why\b|\bwhy (did|do|would|is|was)\b",
}
eng = {}
for m, fams in MOMENTS.items():
    g = X[X.fam.isin(fams)].copy()
    base = g.a2_words.mean()
    d = {}
    for k, p in PHR.items():
        hit = g.b.str.lower().str.contains(p.replace("(", "(?:").replace("(?:?:", "(?:"), regex=True)
        if hit.sum() >= 30:
            d[k] = {"freq": round(hit.mean(), 3), "n": int(hit.sum()), "A2_words_mean": round(g[hit].a2_words.mean(), 1),
                    "A2_vs_base": round(g[hit].a2_words.mean() / base, 2)}
    d["_base_A2_words_mean"] = round(base, 1)
    for k, col in (("ends_with_question", "ends_q"), ("no_question", None), ("template_react_then_q", "template_react_then_q")):
        hit = g[col] if col else ~g.q
        d[k] = {"freq": round(hit.mean(), 3), "A2_words_mean": round(g[hit].a2_words.mean(), 1), "A2_vs_base": round(g[hit].a2_words.mean() / base, 2)}
    eng[m] = d
R["ed_phrases_and_engagement"] = eng
# engajamento por estratégia Jev (amostra 608)
Jr = [json.loads(l) for l in open(os.path.join(SCR, "listener_raw.jsonl"))]
cid2a2 = {c["conv_id"]: len(norm(c["utts"][2]["text"]).split()) for c in convs if len(c["utts"]) > 2}
J = pd.DataFrame([{"main": d["r"]["main"]["choice"], "first": d["r"]["first"]["choice"], "pol": polarity(d["gold"]),
                   "a2": cid2a2.get(d["conv_id"])} for d in Jr]).dropna()
R["jev_strategy_A2_words"] = J.groupby(["pol", "main"]).a2.agg(["mean", "count"]).round(1).reset_index().to_dict("records")

# ---------- reagir a piada / momento sério (jev_base) ----------
D = pd.read_pickle(os.path.join(SCR, "D.pkl"))
LA = re.compile(r"\b(a?ha(?:ha)+h?|he(?:he)+|hi(?:hi)+|lo+l+|lmf?ao+|haa+|hah+)\b", re.I)
def react_shape(t):
    t0 = t.strip()
    m = LA.search(t0)
    rest = LA.sub("", t0); rest = re.sub(r"[\W_]+", " ", rest).strip()
    has_emo = any(ord(ch) > 0x2000 for ch in t0)
    if m and not rest: return "só riso"
    if not m and not rest and has_emo: return "só emoji"
    if m: return "riso + comentário"
    return "sem riso"
out = {}
for corp in ("maichat", "whatsapp_nl"):
    g = D[(D.corpus == corp) & (D.prev_joke == True)].copy()
    g["rshape"] = g.text.map(react_shape)
    g["words"] = g.text.str.split().str.len()
    c = {"n_after_partner_joke": len(g), "shape_share": g.rshape.value_counts(normalize=True).round(3).to_dict(),
         "words_median": float(g.words.median()), "playful_back(>=.5)": round((g.D_playful >= .5).mean(), 3),
         "laugh_tokens": [f"{k} {v}" for k, v in Counter(m.group(0).lower() for t in g.text for m in LA.finditer(t)).most_common(12)],
         "top_openers": [f"{k} {v/len(g):.1%}" for k, v in Counter(opener(t, 1) for t in g.text).most_common(12)],
         "examples": g[g.D_playful >= .5].sample(min(8, len(g)), random_state=1).apply(lambda r: f"{str(r.prev_text)[:80]} -> {r.text[:90]}", axis=1).tolist()}
    s = D[(D.corpus == corp) & (D.prev_serious == True)].copy()
    s["words"] = s.text.str.split().str.len()
    c["after_serious"] = {"n": len(s), "words_median": float(s.words.median()), "intent_share": s.D_intent.value_counts(normalize=True).head(6).round(3).to_dict(),
                          "top_openers": [f"{k} {v/len(s):.1%}" for k, v in Counter(opener(t, 1) for t in s.text).most_common(10)],
                          "examples": s.sample(min(8, len(s)), random_state=4).apply(lambda r: f"{str(r.prev_text)[:90]} -> {r.text[:100]}", axis=1).tolist()}
    ss = D[(D.corpus == corp) & (D.prev_D_seeks_support >= .5)]
    c["after_seeks_support_examples"] = ss.sample(min(8, len(ss)), random_state=2).apply(lambda r: f"{str(r.prev_text)[:90]} -> {r.text[:100]}", axis=1).tolist()
    out[corp] = c
R["chat_reactions"] = out
json.dump(R, open(os.path.join(OUT, "a3_repertoire.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False)[:30000])
