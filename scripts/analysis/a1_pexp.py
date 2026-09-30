"""a1: new Jev experiment. Base P pass showed context with bubbles JOINED ('a / b'), so Jev could not see how
people fragment. Here we re-ask on a sample (250 turns/corpus) with (V1) bubbles shown as separate messages and
(V2) the same + a code-computed line with the speaker's own history of multi-bubble turns."""
import json, os, sys, random
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a1_load import enriched
from jev import ask_many, choice, noul, score, summary

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "data")
T, M = enriched()
T = T.sort_values(["conv_id", "turn_idx"]).reset_index(drop=True)
by = {c: g.reset_index(drop=True) for c, g in T.groupby("conv_id")}
cand = T[(T.has_P == True)]
S = pd.concat([cand[cand.corpus == c].sample(250, random_state=5) for c in ["maichat", "whatsapp_nl"]])

def gap(sec):
    if sec is None or np.isnan(sec): return None
    for lim, n in ((60, "same minute"), (120, "1 min later"), (900, "a few minutes later"), (3600, "within the hour")):
        if sec < lim: return n
    return "hours later"

def fmt(t):
    d = {"speaker": t.speaker, "messages": [x.strip()[:250] for x in t.texts]}
    if t.corpus != "maichat" and gap(t.response_latency_s):
        d["replied"] = gap(t.response_latency_s)
    return d

Q = {"multi": noul("Will `next_speaker` split their next turn into two or more separate chat messages (bubbles) sent in a row?"),
     "n": choice("In how many separate chat messages (bubbles) will `next_speaker` send their next turn?", {
         "1": "one single message", "2": "two messages in a row", "3": "three messages in a row", "4+": "four or more quick messages in a row"})}
items, meta = [], []
for _, t in S.iterrows():
    conv = by[t.conv_id]
    i = conv.index[conv.turn_idx == t.turn_idx][0]
    ctx = conv.iloc[max(0, i - 8):i]
    ctx = ctx[ctx.session == t.session]
    past = conv.iloc[:i]; past = past[past.speaker == t.speaker]
    st1 = {"conversation_so_far": [fmt(x) for x in ctx.itertuples()], "next_speaker": t.speaker,
           "note": "Each turn lists the separate chat messages (bubbles) the speaker sent in a row."}
    rate = (past.n_msgs > 1).mean() if len(past) else None
    st2 = dict(st1)
    st2["next_speaker_history"] = ("no previous turns" if rate is None else
                                   f"In this chat so far, {t.speaker} split {int((past.n_msgs > 1).sum())} of {len(past)} turns into 2+ messages "
                                   f"({rate:.0%}); average {past.n_msgs.mean():.1f} messages per turn.")
    items += [(st1, Q), (st2, Q)]
    meta.append({"corpus": t.corpus, "conv_id": t.conv_id, "turn_idx": t.turn_idx, "n_msgs": t.n_msgs, "P_nm_1": t.P_nm_1,
                 "own_rate": rate})
print("calls", len(items), flush=True)
ans = ask_many(items, workers=4)
print(summary())
R = pd.DataFrame(meta)
for v, off in (("v1", 0), ("v2", 1)):
    R[f"{v}_multi"] = [None if a is None else a["multi"]["noul"] for a in ans[off::2]]
    R[f"{v}_n1"] = [None if a is None else a["n"]["probabilities"]["1"] for a in ans[off::2]]
    R[f"{v}_nexp"] = [None if a is None else sum(a["n"]["probabilities"][k] * w for k, w in (("1", 1), ("2", 2), ("3", 3), ("4+", 4.5))) for a in ans[off::2]]
R.to_csv(os.path.join(OUT, "a1_pexp.csv"), index=False)
res = {}
for c, d in R.groupby("corpus"):
    d = d.dropna(subset=["v1_multi", "v2_multi"])
    y = (d.n_msgs > 1).astype(int)
    rr = {"n": len(d), "rate_multi": round(y.mean(), 3)}
    for name, p in {"base_P(joined)": 1 - d.P_nm_1, "v1_noul": d.v1_multi, "v1_choice": 1 - d.v1_n1, "v2_noul": d.v2_multi,
                    "v2_choice": 1 - d.v2_n1, "code_own_rate": d.own_rate.fillna(y.mean())}.items():
        rr[name] = {"auc": round(roc_auc_score(y, p), 3), "brier": round(brier_score_loss(y, p.clip(0, 1)), 4),
                    "mean_pred": round(float(p.mean()), 3), "rho_n": round(stats.spearmanr(p, d.n_msgs)[0], 3)}
    rr["brier_base_rate"] = round(brier_score_loss(y, np.full(len(d), y.mean())), 4)
    res[c] = rr
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(OUT, "a1_pexp.json"), "w"), indent=1)
