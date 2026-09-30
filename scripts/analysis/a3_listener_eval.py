"""a3: estratégias do ouvinte (B) x emoção-ouro. (1) Jev na amostra de 608; (2) regex em TODAS as ~24,8k conversas.
Saída: analysis/data/a3_listener.json"""
import json, os, re, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from a3_common import *
R = {}
# ---------- Jev ----------
rows = [json.loads(l) for l in open(os.path.join(SCR, "listener_raw.jsonl"))]
J = pd.DataFrame([{"gold": d["gold"], "pol": polarity(d["gold"]), "fam": FAM_OF[d["gold"]],
                   "main": d["r"]["main"]["choice"], "first": d["r"]["first"]["choice"],
                   "b_chars": len(d["B1"]), "b_q": "?" in d["B1"], "A1": d["A1"], "B1": d["B1"],
                   **{k: d["r"][k]["noul"] for k in d["r"] if k.startswith("n_")}} for d in rows])
R["n_jev"] = len(J)
R["main_by_pol"] = pd.crosstab(J["main"], J.pol, normalize="columns").round(3).to_dict()
R["first_by_pol"] = pd.crosstab(J["first"], J.pol, normalize="columns").round(3).to_dict()
R["main_by_fam"] = pd.crosstab(J["main"], J.fam, normalize="columns").round(2).to_dict()
nl = [c for c in J if c.startswith("n_")]
R["noul_rate_by_pol"] = J.groupby("pol")[nl].apply(lambda g: (g >= .5).mean()).round(3).to_dict("index")
R["noul_rate_by_fam"] = J.groupby("fam")[nl].apply(lambda g: (g >= .5).mean()).round(2).to_dict("index")
R["n_strategies_per_reply"] = J.groupby("pol")[nl].apply(lambda g: (g >= .5).sum(axis=1).mean()).round(2).to_dict()
R["chars_by_main"] = J.groupby("main").b_chars.agg(["median", "mean", "count"]).round(1).to_dict("index")
print(pd.DataFrame(R["main_by_pol"])); print(pd.DataFrame(R["first_by_pol"])); print(pd.DataFrame(R["noul_rate_by_pol"])); print(R["n_strategies_per_reply"])
print(pd.DataFrame(R["noul_rate_by_fam"])); print(pd.DataFrame(R["chars_by_main"]))
# exemplos
ex = {}
for s in J["main"].unique():
    x = J[J["main"] == s].head(3)
    ex[s] = [f"[{r.gold}] A: {r.A1[:110]} || B: {r.B1[:130]}" for r in x.itertuples()]
R["examples"] = ex
# ---------- regex em todo o corpus ----------
PAT = {
    "sorry": r"\b(so sorry|i'?m sorry|sorry to hear|that'?s too bad|that sucks|oh no|how awful|how terrible)\b",
    "congrats": r"\b(congrat\w*|good for you|that'?s (awesome|great|amazing|wonderful)|well done|proud of you)\b",
    "me_too": r"\b(me too|i know the feeling|same (here|thing)|i (had|have had|did) (that|the same)|happened to me)\b",
    "advice": r"\b(you should|maybe you|have you (tried|thought|considered)|try to|i would\b|i'?d )",
    "reassure": r"\b(don'?t worry|it'?ll be (ok|okay|fine|alright)|i'?m sure (it|you|they|he|she)|you'?ll (be|do) (fine|great|ok|okay))\b",
    "understand": r"\b(i (can )?understand|i can imagine|must (have been|be)|i bet|i can see why)\b",
    "wish_luck": r"\b(good luck|i hope|hope (it|you|everything|that))\b",
    "laugh": r"\b(haha\w*|lol|lmao|hehe)\b",
    "wow_opener": r"^(wow|oh|ooh|aw+|oh no|yikes|omg|really|nice|cool)\b",
}
convs = [json.loads(l) for l in open(os.path.join(SCR, "empathetic_convs.jsonl"))]
rows = []
for c in convs:
    u = c["utts"]
    if len(u) < 2 or u[1]["speaker"] != "B": continue
    b = u[1]["text"].replace("_comma_", ","); bl = b.lower()
    r = {"gold": c["gold"], "pol": polarity(c["gold"]), "fam": FAM_OF[c["gold"]], "b_chars": len(b), "b_words": len(b.split()),
         "b_q": "?" in b, "a_chars": len(u[0]["text"]),
         "b2_q": ("?" in u[3]["text"]) if len(u) > 3 else np.nan, "b2_chars": len(u[3]["text"]) if len(u) > 3 else np.nan,
         "a2_chars": len(u[2]["text"]) if len(u) > 2 else np.nan, "a2_q": ("?" in u[2]["text"]) if len(u) > 2 else np.nan}
    for k, p in PAT.items():
        r[k] = bool(re.search(p, bl))
    # posição da 1ª pergunta
    r["q_first_sentence"] = bool(re.match(r"^[^.!]*\?", b))
    rows.append(r)
X = pd.DataFrame(rows)
R["n_regex"] = len(X)
agg = X.groupby("pol").agg(n=("b_q", "size"), b_q=("b_q", "mean"), q_first=("q_first_sentence", "mean"), b_chars_med=("b_chars", "median"),
                            b_chars_mean=("b_chars", "mean"), b_words_med=("b_words", "median"),
                            b2_q=("b2_q", "mean"), b2_chars_med=("b2_chars", "median"), a2_q=("a2_q", "mean"),
                            **{k: (k, "mean") for k in PAT}).round(3)
R["regex_by_pol"] = agg.to_dict("index")
print(agg.T)
R["regex_by_fam"] = X.groupby("fam").agg(n=("b_q", "size"), b_q=("b_q", "mean"), b_chars_med=("b_chars", "median"),
                                         **{k: (k, "mean") for k in PAT}).round(3).to_dict("index")
print(pd.DataFrame(R["regex_by_fam"]))
R["regex_by_emotion_q"] = X.groupby("gold").b_q.mean().round(3).sort_values().to_dict()
# teste simples de proporção (neg x pos) para pergunta
from math import sqrt
a = X[X.pol == "negative"].b_q; b = X[X.pol == "positive"].b_q
p = (a.sum() + b.sum()) / (len(a) + len(b)); z = (a.mean() - b.mean()) / sqrt(p * (1 - p) * (1 / len(a) + 1 / len(b)))
R["q_neg_vs_pos_z"] = round(z, 2)
# comprimento de B vs comprimento de A (espelhamento)
R["corr_a_b_chars"] = round(X.a_chars.corr(X.b_chars, method="spearman"), 3)
json.dump(R, open(os.path.join(OUT, "a3_listener.json"), "w"), indent=1, ensure_ascii=False)
print("z", R["q_neg_vs_pos_z"], "corrAB", R["corr_a_b_chars"])
print(json.dumps(R["examples"], indent=0, ensure_ascii=False)[:4000])
