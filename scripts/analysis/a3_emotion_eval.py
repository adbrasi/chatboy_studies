"""a3: avalia o detector de emoção (plano x hierárquico x contexto) contra o ouro do EmpatheticDialogues.
Entrada: SCR/emo_raw.jsonl. Saída: analysis/data/a3_emotion_eval.json"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from a3_common import *

rows = [json.loads(l) for l in open(os.path.join(SCR, "emo_raw.jsonl"))]

def soft(r):
    pf = r["family"]["probabilities"]
    p = {}
    for f, es in FAMILIES.items():
        pe = r["fine_" + f]["probabilities"]
        for e in es:
            p[e] = pf.get(f, 0) * pe.get(e, 0)
    return p

def topk(p, k):
    return [e for e, _ in sorted(p.items(), key=lambda x: -x[1])[:k]]

recs = []
for d in rows:
    r, g = d["r"], d["gold"]
    flat = r["flat"]["probabilities"]
    fam = r["family"]["choice"]
    casc = r["fine_" + fam]["choice"]
    sp = soft(r)
    ens = {e: (flat.get(e, 0) + sp.get(e, 0)) / 2 for e in EMO32}
    pol_gold = polarity(g)
    recs.append({
        "conv_id": d["conv_id"], "gold": g, "cond": d["cond"], "gfam": FAM_OF[g], "gpol": pol_gold,
        "flat_top1": r["flat"]["choice"] == g, "flat_top3": g in topk(flat, 3),
        "flat_conf": r["flat"]["confidence"], "flat_maxp": max(flat.values()), "flat_pred": r["flat"]["choice"],
        "flat_fam_ok": FAM_OF[r["flat"]["choice"]] == FAM_OF[g],
        "fam_ok": fam == FAM_OF[g], "fam_conf": r["family"]["confidence"], "fam_pred": fam,
        "casc_top1": casc == g, "casc_pred": casc,
        "oracle_fine": r["fine_" + FAM_OF[g]]["choice"] == g,
        "soft_top1": topk(sp, 1)[0] == g, "soft_top3": g in topk(sp, 3),
        "ens_top1": topk(ens, 1)[0] == g, "ens_top3": g in topk(ens, 3),
        "pol_pred": r["polarity"]["choice"], "pol_conf": r["polarity"]["confidence"],
        "flat_pol": polarity(r["flat"]["choice"]),
    })
E = pd.DataFrame(recs)
R = {"n_calls": len(E), "n_by_cond": E.cond.value_counts().to_dict()}

def ci(p, n):
    se = np.sqrt(p * (1 - p) / n); return [round(p - 1.96 * se, 3), round(p + 1.96 * se, 3)]

metrics = ["flat_top1", "flat_top3", "casc_top1", "soft_top1", "soft_top3", "ens_top1", "ens_top3", "fam_ok", "flat_fam_ok", "oracle_fine"]
R["accuracy"] = {}
full_ids = set(E[E.cond == "FULL"].conv_id)
for cond, g in E.groupby("cond"):
    R["accuracy"][cond] = {m: {"acc": round(g[m].mean(), 3), "ci95": ci(g[m].mean(), len(g))} for m in metrics} | {"n": len(g)}
    # polaridade (sem 'surprised')
    gp = g[g.gpol != "ambiguous"]
    R["accuracy"][cond]["polarity_direct"] = round((gp.pol_pred == gp.gpol).mean(), 3)
    R["accuracy"][cond]["polarity_from_flat"] = round((gp.flat_pol == gp.gpol).mean(), 3)
# comparação pareada na subamostra FULL
sub = E[E.conv_id.isin(full_ids)]
R["accuracy_subset256"] = sub.groupby("cond")[metrics].mean().round(3).to_dict("index")

# McNemar pareado
def mcnemar(a, b):
    from math import comb
    n01 = int(((~a) & b).sum()); n10 = int((a & ~b).sum()); n = n01 + n10
    k = min(n01, n10)
    p = min(1, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1
    return {"a_only": n10, "b_only": n01, "p": round(p, 4)}
W = E.pivot_table(index="conv_id", columns="cond", values=["flat_top1", "casc_top1", "soft_top1", "fam_ok", "ens_top1"], aggfunc="first")
tests = {}
for m in ["flat_top1", "casc_top1", "soft_top1", "fam_ok", "ens_top1"]:
    x = W[m].dropna(subset=["A1", "A1B1A2"])
    tests[f"{m}: A1 vs A1B1A2"] = mcnemar(x["A1"].astype(bool), x["A1B1A2"].astype(bool))
    y = W[m].dropna(subset=["A1B1A2", "FULL"])
    tests[f"{m}: A1B1A2 vs FULL"] = mcnemar(y["A1B1A2"].astype(bool), y["FULL"].astype(bool))
for cond in ("A1", "A1B1A2"):
    g = E[E.cond == cond].set_index("conv_id")
    tests[f"{cond}: flat vs cascade(hard)"] = mcnemar(g.flat_top1, g.casc_top1)
    tests[f"{cond}: flat vs soft"] = mcnemar(g.flat_top1, g.soft_top1)
    tests[f"{cond}: flat vs ensemble"] = mcnemar(g.flat_top1, g.ens_top1)
    tests[f"{cond}: family direct vs family-of-flat"] = mcnemar(g.flat_fam_ok, g.fam_ok)
R["mcnemar"] = tests

# calibração (flat e família) por faixa de confiança
def calib(g, conf, ok):
    bins = [0, .2, .4, .6, .8, 1.01]
    c = pd.cut(g[conf], bins, right=False)
    t = g.groupby(c, observed=True).agg(n=(ok, "size"), acc=(ok, "mean"), conf=(conf, "mean"))
    ece = float((t.n * (t.acc - t.conf).abs()).sum() / t.n.sum())
    return {"bins": [{"bin": str(i), "n": int(r.n), "acc": round(r.acc, 3), "mean_conf": round(r.conf, 3)} for i, r in t.iterrows()],
            "ece": round(ece, 3)}
R["calibration"] = {}
for cond, g in E.groupby("cond"):
    R["calibration"][cond] = {"flat_conf": calib(g, "flat_conf", "flat_top1"), "flat_maxp": calib(g, "flat_maxp", "flat_top1"),
                              "family_conf": calib(g, "fam_conf", "fam_ok")}
# seletiva: agir só com confiança alta
R["selective"] = {}
for cond, g in E.groupby("cond"):
    d = {}
    for th in (.5, .7, .9):
        s = g[g.flat_conf >= th]; f = g[g.fam_conf >= th]
        d[str(th)] = {"flat_coverage": round(len(s) / len(g), 3), "flat_acc": round(s.flat_top1.mean(), 3),
                      "fam_coverage": round(len(f) / len(g), 3), "fam_acc": round(f.fam_ok.mean(), 3)}
    R["selective"][cond] = d
# por emoção e confusões (A1B1A2)
g = E[E.cond == "A1B1A2"]
R["per_emotion_A1B1A2"] = g.groupby("gold")[["flat_top1", "flat_top3", "fam_ok"]].mean().round(2).sort_values("flat_top1").to_dict("index")
conf = Counter(zip(g.gold, g.flat_pred))
R["top_confusions_A1B1A2"] = [f"{a}->{b}: {n}" for (a, b), n in conf.most_common(60) if a != b][:20]
R["pred_distribution_A1B1A2"] = g.flat_pred.value_counts().head(12).to_dict()
R["per_family_A1B1A2"] = g.groupby("gfam")[["fam_ok", "flat_fam_ok", "oracle_fine", "flat_top1"]].mean().round(2).to_dict("index")
R["family_confusions_A1B1A2"] = [f"{a}->{b}: {n}" for (a, b), n in Counter(zip(g.gfam, g.fam_pred)).most_common(30) if a != b][:10]
json.dump(R, open(os.path.join(OUT, "a3_emotion_eval.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: R[k] for k in ("n_by_cond", "accuracy", "accuracy_subset256", "mcnemar", "selective")}, indent=1))
print(json.dumps(R["calibration"], indent=0)[:3000])
print(R["per_family_A1B1A2"]); print(R["top_confusions_A1B1A2"]); print(R["family_confusions_A1B1A2"]); print(R["pred_distribution_A1B1A2"])
print({k: v for k, v in R["per_emotion_A1B1A2"].items()})
