"""E1: repeat consistency, paraphrase robustness, Noul x Choice x Score agreement."""
import json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import a6_common as C
from a6_common import noul, choice, score

by = C.load_turns(); base = C.load_base()
S = C.sample_turns(by, "maichat", 40, seed=61, in_base=base) + C.sample_turns(by, "whatsapp_nl", 20, seed=62, in_base=base)
W = "the speaker of `current_turn`"
QD = C.AB.q_descriptive()

# ---- part A: repeats of the exact base call
items = []
for conv, t in S:
    st = C.d_state(conv, t, 8)
    items += [dict(state=st, questions=QD, tag="e1_ident", rep=0), dict(state=st, questions=QD, tag="e1_ident", rep=1)]
    for r in (1, 2):
        items.append(dict(state={"run_id": f"r{r}-{random.Random(r).randint(1000,9999)}"} | st, questions=QD, tag="e1_runid", rep=r))

# ---- part B: paraphrases / types
emo = dict(C.AB.EMOTIONS)
keys = list(emo); rnd = random.Random(3); shuf = keys[:]; rnd.shuffle(shuf)
QB = {
    "anx_n1": QD["anxious"],
    "anx_n2": noul(f"Does {W} sound nervous or insecure?"),
    "anx_n3": noul(f"{W[0].upper()+W[1:]} is feeling worried."),
    "anx_n4": noul(f"Is {W} anxious, worried or insecure?", true="The speaker shows worry, nervousness or insecurity", false="The speaker shows no sign of worry, nervousness or insecurity"),
    "anx_neg": noul(f"Is {W} calm and free of worry?"),
    "anx_c": choice(f"Is {W} anxious, worried or insecure?", {"yes": None, "no": None}),
    "anx_s": score(f"How anxious, worried or insecure is {W}?", ["not at all", "slightly", "clearly", "very"]),
    "pla_n1": QD["playful"],
    "pla_n2": noul(f"Is {W} joking around?"),
    "pla_n3": noul(f"{W[0].upper()+W[1:]} is in a playful, teasing mood."),
    "pla_c": choice(f"Is {W} being playful, joking or teasing?", {"yes": None, "no": None}),
    "pla_s": score(f"How playful, joking or teasing is {W}?", ["not at all", "slightly", "clearly", "very"]),
    "ser_s1": QD["seriousness"],
    "ser_s2": score("How heavy or emotional is the conversation at this point?", ["light and fun", "casual", "somewhat serious", "very serious / emotional"]),
    "ser_rev": score("How light-hearted is this moment of the conversation?", ["very serious / emotional", "somewhat serious", "casual", "playful banter"]),
    "ser_n": noul("Is this a serious moment in the conversation?"),
    "ser_c": choice("How serious is this moment of the conversation?", {"playful banter": None, "casual": None, "somewhat serious": None, "very serious / emotional": None}),
    "emo_c1": QD["emotion"],
    "emo_shuf": choice(f"Which emotion best describes {W}?", {k: emo[k] for k in shuf}),
    "emo_nodesc": choice(f"Which emotion best describes {W}?", {k: None for k in keys}),
    "val_s1": QD["valence"],
    "val_s10": score(f"How positive or negative is {W} feeling?", ["extremely negative", "very negative", "negative", "somewhat negative", "slightly negative", "slightly positive", "somewhat positive", "positive", "very positive", "extremely positive"]),
}
for conv, t in S:
    items.append(dict(state=C.d_state(conv, t, 8), questions=QB, tag="e1_para", rep=0))

res = C.run_many(items, 4)
print("new calls", C.NEW_CALLS[0])

# ---- analysis
def cmp(a, b):
    """distance between two answers of the same question"""
    if a["type"] == "noul": return abs(a["noul"] - b["noul"])
    if a["type"] == "score": return abs(a["score"] - b["score"])
    return float(a["choice"] != b["choice"])

out = {"n_turns": len(S), "per_question": {}}
k = 0
A = []
nS = len(S)
for j, (conv, t) in enumerate(S):
    r_i0, r_i1, r_r1, r_r2 = res[4 * j:4 * j + 4]; r_p = res[4 * nS + j]
    b = base[(t["corpus"], t["conv_id"], t["turn_idx"])]["D"]
    A.append(dict(corpus=t["corpus"], base=b, i0=r_i0 and r_i0["answers"], i1=r_i1 and r_i1["answers"],
                  r1=r_r1 and r_r1["answers"], r2=r_r2 and r_r2["answers"], p=r_p and r_p["answers"], text=" / ".join(t["texts"])[:120]))
A = [a for a in A if all(a[x] for x in ("i0", "i1", "r1", "r2", "p"))]
for q in QD:
    row = {}
    for name, (x, y) in {"ident_now": ("i0", "i1"), "ident_vs_base": ("base", "i0"), "runid_vs_ident": ("i0", "r1"), "runid_vs_runid": ("r1", "r2")}.items():
        d = [cmp(a[x][q], a[y][q]) for a in A]
        row[name] = {"mean": float(np.mean(d)), "p95": float(np.percentile(d, 95)), "max": float(np.max(d)), "frac_exact": float(np.mean([v == 0 for v in d]))}
    row["type"] = QD[q]["type"]
    out["per_question"][q] = row
# flips of a noul across 0.5 under run_id
flips = []
for q in QD:
    if QD[q]["type"] == "noul":
        flips += [(a["i0"][q]["noul"] > .5) != (a["r1"][q]["noul"] > .5) for a in A]
out["noul_flip_rate_runid"] = float(np.mean(flips))
flips = []
for q in QD:
    if QD[q]["type"] == "noul":
        flips += [(a["i0"][q]["noul"] > .5) != (a["i1"][q]["noul"] > .5) for a in A]
out["noul_flip_rate_ident"] = float(np.mean(flips))

# part B
def arr(q, f=None):
    return np.array([(f or C.val)(a["p"][q]) for a in A])
def pr(q):  # choice yes prob
    return np.array([a["p"][q]["probabilities"]["yes"] for a in A])
corr = lambda x, y: float(np.corrcoef(x, y)[0, 1])
B = {}
for fam, ns in {"anx": ["anx_n1", "anx_n2", "anx_n3", "anx_n4"], "pla": ["pla_n1", "pla_n2", "pla_n3"]}.items():
    M = {n: arr(n) for n in ns}
    M[f"{fam}_c_yes"] = pr(f"{fam}_c"); M[f"{fam}_s"] = arr(f"{fam}_s")
    if fam == "anx": M["anx_neg(1-p)"] = 1 - arr("anx_neg")
    names = list(M)
    B[fam] = {"means": {n: float(M[n].mean()) for n in names},
              "corr_vs_n1": {n: corr(M[ns[0]], M[n]) for n in names},
              "mean_abs_vs_n1": {n: float(np.mean(np.abs(M[ns[0]] - M[n]))) for n in names if not n.endswith("_s")},
              "agree_binary_vs_n1": {n: float(np.mean((M[ns[0]] > .5) == (M[n] > (1.5 if n.endswith('_s') else .5)))) for n in names}}
ser = {"s1": arr("ser_s1"), "s2": arr("ser_s2"), "rev(3-x)": 3 - arr("ser_rev"), "noul": arr("ser_n"),
       "choice_exp": np.array([sum(i * a["p"]["ser_c"]["probabilities"][l] for i, l in enumerate(["playful banter", "casual", "somewhat serious", "very serious / emotional"])) for a in A])}
B["ser"] = {"means": {n: float(v.mean()) for n, v in ser.items()}, "corr_vs_s1": {n: corr(ser["s1"], v) for n, v in ser.items()},
            "mean_abs_vs_s1": {n: float(np.mean(np.abs(ser["s1"] - v))) for n, v in ser.items() if n != "noul"}}
e1, es, en = arr("emo_c1"), arr("emo_shuf"), arr("emo_nodesc")
B["emo"] = {"agree_shuffled_order": float(np.mean(e1 == es)), "agree_no_descriptions": float(np.mean(e1 == en)),
            "agree_same_q_in_other_call(base-vs-para-call)": float(np.mean([a["p"]["emo_c1"]["choice"] == a["i0"]["emotion"]["choice"] for a in A]))}
v1, v10 = arr("val_s1"), arr("val_s10")
B["val"] = {"corr_5lvl_vs_10lvl": corr(v1, v10), "mean_5lvl_norm": float((v1 / 4).mean()), "mean_10lvl_norm": float((v10 / 9).mean())}
# same question alone in a different call (isolation): anx_n1 in para call vs base D call i0
B["isolation_same_question_other_questions"] = {q: float(np.mean([abs(a["p"][q2]["noul"] - a["i0"][q]["noul"]) for a in A]))
                                                 for q, q2 in (("anxious", "anx_n1"), ("playful", "pla_n1"))}
out["partB"] = B
# examples of largest paraphrase disagreement
ex = sorted(A, key=lambda a: -abs(a["p"]["anx_n1"]["noul"] - a["p"]["anx_n3"]["noul"]))[:4]
out["examples_anx_disagree"] = [{"text": a["text"], "n1": a["p"]["anx_n1"]["noul"], "n2": a["p"]["anx_n2"]["noul"], "n3": a["p"]["anx_n3"]["noul"], "score": a["p"]["anx_s"]["score"], "choice_yes": a["p"]["anx_c"]["probabilities"]["yes"]} for a in ex]
ex = sorted(A, key=lambda a: -abs(a["p"]["pla_n1"]["noul"] - a["p"]["pla_n2"]["noul"]))[:4]
out["examples_pla_disagree"] = [{"text": a["text"], "n1": a["p"]["pla_n1"]["noul"], "n2": a["p"]["pla_n2"]["noul"], "n3": a["p"]["pla_n3"]["noul"]} for a in ex]
# by corpus
out["by_corpus_runid_noul_meanabs"] = {c: float(np.mean([cmp(a["i0"][q], a["r1"][q]) for a in A if a["corpus"] == c for q in QD if QD[q]["type"] == "noul"])) for c in ("maichat", "whatsapp_nl")}
out["latency"] = [r["latency_s"] for r in res if r]
lat = np.array(out.pop("latency")); out["latency_17q_p50_p90"] = [float(np.median(lat[:240])), float(np.percentile(lat[:240], 90))]
print(C.dump("e1_consistency.json", out))
print(json.dumps(out, indent=1)[:6000])
