"""E7: EmpatheticDialogues gold emotion (32 classes). Flat 32-way Choice vs hierarchical (8 groups -> fine, greedy and beam-2);
context size (first utterance / first 2 utterances / whole dialogue); confidence calibration and thresholds."""
import json, random, sys, os
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import a6_common as C
from a6_common import choice

GROUPS = {
    "happy_excited": (["excited", "joyful", "content", "anticipating", "hopeful"], "happy, excited, looking forward to something, hopeful, satisfied"),
    "proud_confident": (["proud", "confident", "impressed", "prepared"], "pride, self-assurance, admiration of someone, feeling ready"),
    "warm_caring": (["grateful", "caring", "trusting", "faithful", "sentimental", "nostalgic"], "gratitude, care for others, trust, loyalty, fond memories of the past"),
    "surprised": (["surprised"], "caught off guard by something unexpected"),
    "sad_down": (["sad", "lonely", "disappointed", "devastated"], "sadness, loneliness, disappointment, grief"),
    "fear_worry": (["afraid", "terrified", "anxious", "apprehensive"], "fear, worry, nervousness about something"),
    "anger_disgust": (["angry", "annoyed", "furious", "disgusted", "jealous"], "anger, irritation, disgust, jealousy"),
    "self_conscious": (["guilty", "ashamed", "embarrassed"], "guilt, shame, embarrassment about oneself"),
}
G_OF = {e: g for g, (es, _) in GROUPS.items() for e in es}
ALL = sorted(G_OF)
INSTR = "Which emotion best describes how speaker A feels about the situation A is talking about?"
QFLAT = {"flat": choice(INSTR, {e: None for e in ALL}),
         "group": choice("Which family of emotions best describes how speaker A feels about the situation A is talking about?", {g: d for g, (_, d) in GROUPS.items()})}

conv = defaultdict(list)
for l in open(os.path.join(C.PROC, "messages.jsonl"), encoding="utf-8"):
    if '"empathetic"' not in l[:40] or '"split": "test"' not in l:
        continue
    d = json.loads(l)
    conv[d["conv_id"]].append(d)
byemo = defaultdict(list)
for cid, ms in conv.items():
    ms.sort(key=lambda m: m["idx"])
    if len(ms) >= 4:
        byemo[ms[0]["gold_emotion"]].append(cid)
rnd = random.Random(77)
SEL = [c for e in ALL for c in rnd.sample(byemo[e], 6)]
def st(ms, k=None):
    return {"conversation": [{"speaker": m["speaker"], "text": m["text"]} for m in (ms[:k] if k else ms)]}

items = []
for cid in SEL:
    ms = conv[cid]
    items += [dict(state=st(ms), questions=QFLAT, tag="e7_full"), dict(state=st(ms, 1), questions=QFLAT, tag="e7_first"),
              dict(state=st(ms, 2), questions=QFLAT, tag="e7_two")]
res = C.run_many(items, 4)
print("new calls stage1", C.NEW_CALLS[0])
R = {cid: {"full": res[3 * i]["answers"], "first": res[3 * i + 1]["answers"], "two": res[3 * i + 2]["answers"]} for i, cid in enumerate(SEL)}
# stage 2: fine choice within top group (greedy) and within top-2 groups (beam 2), full dialogue
items2 = []
for cid in SEL:
    gp = R[cid]["full"]["group"]["probabilities"]
    top2 = sorted(gp, key=lambda g: -gp[g])[:2]
    q = {"fine1": choice(INSTR, {e: None for e in GROUPS[top2[0]][0]}) if len(GROUPS[top2[0]][0]) > 1 else None,
         "fine2": choice(INSTR, {e: None for g in top2 for e in GROUPS[g][0]})}
    q = {k: v for k, v in q.items() if v}
    items2.append(dict(state=st(conv[cid]), questions=q, tag="e7_fine"))
res2 = C.run_many(items2, 4)
print("new calls total", C.NEW_CALLS[0])
for cid, r in zip(SEL, res2):
    R[cid]["fine"] = r["answers"]

gold = {cid: conv[cid][0]["gold_emotion"] for cid in SEL}
out = {"n": len(SEL), "per_class": 6, "chance_fine": 1 / 32, "chance_group": 1 / 8}
def topk(p, k): return sorted(p, key=lambda x: -p[x])[:k]
for cond in ("first", "two", "full"):
    f = [R[c][cond]["flat"] for c in SEL]; g = [R[c][cond]["group"] for c in SEL]
    out[f"flat_{cond}"] = {"acc": float(np.mean([a["choice"] == gold[c] for a, c in zip(f, SEL)])),
                           "top3": float(np.mean([gold[c] in topk(a["probabilities"], 3) for a, c in zip(f, SEL)])),
                           "group_acc_from_flat": float(np.mean([G_OF[a["choice"]] == G_OF[gold[c]] for a, c in zip(f, SEL)])),
                           "conf_mean": float(np.mean([a["confidence"] for a in f])),
                           "p_gold_mean": float(np.mean([a["probabilities"][gold[c]] for a, c in zip(f, SEL)]))}
    out[f"group_{cond}"] = {"acc": float(np.mean([a["choice"] == G_OF[gold[c]] for a, c in zip(g, SEL)])),
                            "conf_mean": float(np.mean([a["confidence"] for a in g]))}
# hierarchical
def hier(c, beam):
    gch = R[c]["full"]["group"]["choice"]
    if beam == 1:
        return R[c]["fine"]["fine1"]["choice"] if "fine1" in R[c]["fine"] else GROUPS[gch][0][0]
    return R[c]["fine"]["fine2"]["choice"]
out["hier_greedy_full"] = {"acc": float(np.mean([hier(c, 1) == gold[c] for c in SEL]))}
out["hier_beam2_full"] = {"acc": float(np.mean([hier(c, 2) == gold[c] for c in SEL]))}
# product-rule combination: P(e) = P(group) * P(e | group) using flat probs renormalized within group
def prod(c):
    gp = R[c]["full"]["group"]["probabilities"]; fp = R[c]["full"]["flat"]["probabilities"]
    sc = {}
    for e in ALL:
        g = G_OF[e]; z = sum(fp[x] for x in GROUPS[g][0]) or 1e-9
        sc[e] = gp[g] * fp[e] / z
    return max(sc, key=sc.get)
out["group_x_flat_product_full"] = {"acc": float(np.mean([prod(c) == gold[c] for c in SEL]))}
# calibration of flat (full): confidence bins and max-prob bins
def calib(key, vals, correct, bins):
    rows = []
    for lo, hi in bins:
        m = (vals >= lo) & (vals < hi)
        if m.sum(): rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": int(m.sum()), "acc": float(correct[m].mean()), "mean_val": float(vals[m].mean())})
    return rows
for cond in ("full", "first"):
    conf = np.array([R[c][cond]["flat"]["confidence"] for c in SEL]); pmax = np.array([max(R[c][cond]["flat"]["probabilities"].values()) for c in SEL])
    corr = np.array([R[c][cond]["flat"]["choice"] == gold[c] for c in SEL])
    gcorr = np.array([G_OF[R[c][cond]["flat"]["choice"]] == G_OF[gold[c]] for c in SEL])
    B = [(0, .3), (.3, .5), (.5, .7), (.7, .85), (.85, 1.01)]
    out[f"calib_flat_{cond}_by_conf"] = calib("conf", conf, corr, B)
    out[f"calib_flat_{cond}_by_pmax"] = calib("pmax", pmax, corr, B)
    out[f"calib_flat_{cond}_groupacc_by_conf"] = calib("conf", conf, gcorr, B)
    # ECE on pmax
    ece = 0
    for lo, hi in [(i / 10, (i + 1) / 10 + (1e-9 if i == 9 else 0)) for i in range(10)]:
        m = (pmax >= lo) & (pmax < hi)
        if m.sum(): ece += m.mean() * abs(corr[m].mean() - pmax[m].mean())
    out[f"ece_flat_{cond}"] = float(ece)
    # coverage/accuracy curve for thresholds
    out[f"threshold_curve_flat_{cond}"] = [{"thr": t, "coverage": float((conf >= t).mean()), "acc": float(corr[conf >= t].mean()) if (conf >= t).any() else None,
                                              "group_acc": float(gcorr[conf >= t].mean()) if (conf >= t).any() else None} for t in (0, .3, .4, .5, .6, .7, .8, .9)]
gconf = np.array([R[c]["full"]["group"]["confidence"] for c in SEL]); gc = np.array([R[c]["full"]["group"]["choice"] == G_OF[gold[c]] for c in SEL])
out["threshold_curve_group_full"] = [{"thr": t, "coverage": float((gconf >= t).mean()), "acc": float(gc[gconf >= t].mean()) if (gconf >= t).any() else None} for t in (0, .3, .5, .7, .8, .9)]
# per-group accuracy & confusions
out["group_acc_by_gold_group(full)"] = {g: float(np.mean([R[c]["full"]["group"]["choice"] == g for c in SEL if G_OF[gold[c]] == g])) for g in GROUPS}
out["fine_acc_by_gold(full,flat)"] = {e: float(np.mean([R[c]["full"]["flat"]["choice"] == e for c in SEL if gold[c] == e])) for e in ALL}
conf_pairs = Counter((gold[c], R[c]["full"]["flat"]["choice"]) for c in SEL if gold[c] != R[c]["full"]["flat"]["choice"])
out["top_confusions(full)"] = conf_pairs.most_common(12)
out["examples_lowconf_wrong"] = [{"gold": gold[c], "pred": R[c]["full"]["flat"]["choice"], "conf": R[c]["full"]["flat"]["confidence"], "first": conv[c][0]["text"][:140]}
                                 for c in SEL if R[c]["full"]["flat"]["confidence"] < .4][:5]
out["tokens"] = {t: float(np.mean([r["usage"]["input_tokens"] for r in res + res2 if r["tag"] == t])) for t in ("e7_full", "e7_first", "e7_fine")}
print(C.dump("e7_empathetic.json", out))
for k, v in out.items():
    print(k, json.dumps(v, ensure_ascii=False)[:1500])
