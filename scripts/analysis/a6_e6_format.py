"""E6: state format. JSON vs plain text; speaker labels A/B vs user/assistant; focus explicit (current_turn) vs implicit (last turn).
Stability = answer change under an irrelevant perturbation (run_id) within each format; agreement with the base format;
focus validity = probes whose truth is known from code (does the focal turn contain a question mark / laughter)."""
import json, re, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import a6_common as C
from a6_common import noul

by = C.load_turns(); base = C.load_base()
S = C.sample_turns(by, "maichat", 30, min_ctx=4, seed=71, in_base=base) + C.sample_turns(by, "whatsapp_nl", 20, min_ctx=4, seed=72, in_base=base)
QD = C.AB.q_descriptive()
QD["probe_q"] = noul("Does the speaker of `current_turn` ask a question in `current_turn`?")
QD["probe_laugh"] = noul("Does `current_turn` contain laughter (haha, lol, kkk, 😂)?")


def rewrite(qs, cur, prev):
    out = {}
    for k, q in qs.items():
        s = json.dumps(q, ensure_ascii=False)
        s = s.replace("`current_turn`", cur).replace("`previous_turns`", prev)
        out[k] = json.loads(s)
    return out


def fmt_text(ctx, t, lab, explicit):
    L = lambda x: f"{lab.get(x['speaker'], x['speaker'])}: {x['text']}" + (f"  ({x['replied']})" if x.get("replied") else "")
    lines = [L(C.AB.fmt_turn(x)) for x in ctx]
    cur = L(C.AB.fmt_turn(t))
    if explicit:
        return "PREVIOUS TURNS:\n" + ("\n".join(lines) or "(none)") + "\n\nCURRENT TURN:\n" + cur
    return "\n".join(lines + [cur])


def formats(conv, t):
    ctx = C.context(conv, t, 8)
    ab = {}
    ua = {t["speaker"]: "user", ("B" if t["speaker"] == "A" else "A"): "assistant"}
    F = {}
    F["json_AB_explicit"] = (C.d_state(conv, t, 8), QD)
    F["json_userassistant_explicit"] = (C.d_state(conv, t, 8, label_map=ua), QD)
    F["json_AB_implicit"] = ({"conversation": [C.AB.fmt_turn(x) for x in ctx] + [C.AB.fmt_turn(t)]},
                             rewrite(QD, "the last turn in `conversation`", "the earlier turns in `conversation`"))
    F["text_AB_explicit"] = (fmt_text(ctx, t, ab, True), rewrite(QD, "the CURRENT TURN", "the PREVIOUS TURNS"))
    F["text_AB_implicit"] = (fmt_text(ctx, t, ab, False), rewrite(QD, "the last message of the chat", "the earlier messages"))
    return F

FN = list(formats(*S[0]))
items = []
for j, (conv, t) in enumerate(S):
    F = formats(conv, t)
    for f in FN:
        items.append(dict(state=F[f][0], questions=F[f][1], tag=f"e6_{f}", rep=0))
for j, (conv, t) in enumerate(S[:30]):
    F = formats(conv, t)
    for f in FN:
        st = F[f][0]
        st = ({"run_id": "p1"} | st) if isinstance(st, dict) else ("[run p1]\n" + st)
        items.append(dict(state=st, questions=F[f][1], tag=f"e6_{f}_pert", rep=1))
res = C.run_many(items, 4)
print("new calls", C.NEW_CALLS[0])
n = len(S); nf = len(FN)
R = {f: [res[j * nf + i]["answers"] for j in range(n)] for i, f in enumerate(FN)}
P = {f: [res[n * nf + j * nf + i]["answers"] for j in range(30)] for i, f in enumerate(FN)}
gold_q = np.array([t["has_q"] for _, t in S]); gold_l = np.array([t["laugh"] for _, t in S])

def d(a, b):
    if a["type"] == "noul": return abs(a["noul"] - b["noul"])
    if a["type"] == "score": return abs(a["score"] - b["score"])
    return float(a["choice"] != b["choice"])

def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0: return None
    return float(np.mean([(p > q) + .5 * (p == q) for p in pos for q in neg]))

qs = [k for k in QD if not k.startswith("probe")]
out = {"n": n, "n_pert": 30, "formats": {}}
for f in FN:
    row = {}
    row["perturb_noul_meanabs"] = float(np.mean([d(R[f][j][k], P[f][j][k]) for j in range(30) for k in qs if QD[k]["type"] == "noul"]))
    row["perturb_score_meanabs"] = float(np.mean([d(R[f][j][k], P[f][j][k]) for j in range(30) for k in qs if QD[k]["type"] == "score"]))
    row["perturb_choice_flip"] = float(np.mean([d(R[f][j][k], P[f][j][k]) for j in range(30) for k in qs if QD[k]["type"] == "choice"]))
    ref = "json_AB_explicit"
    row["vs_json_AB_explicit_noul_meanabs"] = float(np.mean([d(R[f][j][k], R[ref][j][k]) for j in range(n) for k in qs if QD[k]["type"] == "noul"]))
    row["vs_json_AB_explicit_choice_flip"] = float(np.mean([d(R[f][j][k], R[ref][j][k]) for j in range(n) for k in qs if QD[k]["type"] == "choice"]))
    row["vs_json_AB_explicit_score_meanabs"] = float(np.mean([d(R[f][j][k], R[ref][j][k]) for j in range(n) for k in qs if QD[k]["type"] == "score"]))
    row["choice_conf_mean"] = float(np.mean([R[f][j][k]["confidence"] for j in range(n) for k in qs if QD[k]["type"] == "choice"]))
    row["focus_probe_q_auc"] = auc(gold_q, [R[f][j]["probe_q"]["noul"] for j in range(n)])
    row["focus_probe_q_acc@.5"] = float(np.mean((np.array([R[f][j]["probe_q"]["noul"] for j in range(n)]) > .5) == gold_q))
    row["focus_probe_laugh_acc@.5"] = float(np.mean((np.array([R[f][j]["probe_laugh"]["noul"] for j in range(n)]) > .5) == gold_l))
    row["hook_vs_hasq_auc"] = auc(gold_q, [R[f][j]["hook"]["noul"] for j in range(n)])
    row["playful_vs_laugh_auc"] = auc(gold_l, [R[f][j]["playful"]["noul"] for j in range(n)])
    row["mean_seriousness"] = float(np.mean([R[f][j]["seriousness"]["score"] for j in range(n)]))
    row["mean_topic_shift"] = float(np.mean([R[f][j]["topic_shift"]["noul"] for j in range(n)]))
    row["tokens"] = float(np.mean([r["usage"]["input_tokens"] for r in res[:n * nf] if r["tag"] == f"e6_{f}"]))
    out["formats"][f] = row
out["base_rate"] = {"has_q": float(gold_q.mean()), "laugh": float(gold_l.mean())}
# relationship choice agreement user/assistant vs A/B
from collections import Counter
out["relationship_dist"] = {f: Counter(R[f][j]["relationship"]["choice"] for j in range(n)).most_common(4) for f in FN}
print(C.dump("e6_format.json", out))
for f, r in out["formats"].items():
    print(f"{f:28s}", " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
print(out["base_rate"], out["relationship_dist"])
