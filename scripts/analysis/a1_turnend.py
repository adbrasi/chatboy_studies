"""a1: end-of-turn detection (is the user done, or is another bubble coming?). Code features on every message,
hazard of a next bubble vs silence time (maichat), and a Jev noul on a sample of 250 messages/corpus."""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a1_load import load
from a1_heur import heur
from jev import ask_many, noul, summary

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "data")
_, M, _ = load()
M = M.copy()
nsp = M.groupby("conv_id").speaker.nunique(); M = M[M.conv_id.map(nsp) == 2]
M = M.sort_values(["conv_id", "idx"]).reset_index(drop=True)
M["ts_p"] = pd.to_datetime(M.ts.astype(str).str.replace(r"[\[\]]", "", regex=True), utc=True, format="mixed", errors="coerce")
g = M.groupby("conv_id")
M["cont"] = (g.speaker.shift(-1) == M.speaker).astype(int)
M["next_gap"] = (g.ts_p.shift(-1) - M.ts_p).dt.total_seconds()
M["first_in_turn"] = (g.speaker.shift(1) != M.speaker)
M["pos"] = M.groupby((M.speaker != g.speaker.shift(1)).cumsum()).cumcount()
M["func"] = [heur(t, r) for t, r in zip(M.text, M.to_dict("records"))]
M["ends_q"] = M.text.str.strip().str.endswith("?")
res = {}
for c, d in M.groupby("corpus"):
    d = d[d.next_gap.notna() & (d.next_gap >= 0)]
    r = {"n": len(d), "P(cont)": round(d.cont.mean(), 3)}
    r["P(cont)|func"] = d.groupby("func").cont.agg(["size", "mean"]).round(3).to_dict(orient="index")
    r["P(cont)|ends_q"] = d.groupby("ends_q").cont.mean().round(3).rename(index=str).to_dict()
    r["P(cont)|ends_punct"] = d.groupby("f_ends_punct").cont.mean().round(3).rename(index=str).to_dict()
    r["P(cont)|ellipsis"] = d.groupby("f_ellipsis").cont.mean().round(3).rename(index=str).to_dict()
    r["P(cont)|first_in_turn"] = d.groupby("first_in_turn").cont.mean().round(3).rename(index=str).to_dict()
    r["P(cont)|pos"] = d.assign(p=d.pos.clip(upper=4)).groupby("p").cont.mean().round(3).to_dict()
    r["P(cont)|len"] = d.assign(b=pd.cut(d.f_n_chars, [0, 5, 15, 30, 60, 5000])).groupby("b", observed=True).cont.mean().round(3).rename(index=str).to_dict()
    # hazard (maichat): given t seconds of silence after a bubble, P(next bubble from same speaker)
    if c == "maichat":
        hz = {}
        for t in [0, 3, 5, 10, 15, 20, 30, 45, 60]:
            x = d[d.next_gap > t]
            hz[t] = {"n": len(x), "P(same speaker next)": round(x.cont.mean(), 3)}
        r["hazard_after_silence_s"] = hz
        r["share_of_continuations_within_s"] = {t: round(float((d[d.cont == 1].next_gap <= t).mean()), 3) for t in [3, 5, 10, 15, 20, 30, 45, 60]}
    else:
        r["share_of_continuations_within_s"] = {t: round(float((d[d.cont == 1].next_gap <= t).mean()), 3) for t in [0, 60, 120, 300, 600]}
        hz = {}
        for t in [0, 60, 120, 300, 600]:
            x = d[d.next_gap > t]
            hz[t] = {"n": len(x), "P(same speaker next)": round(x.cont.mean(), 3)}
        r["hazard_after_silence_s"] = hz
    # code logistic
    X = pd.get_dummies(d[["func"]]).assign(q=d.ends_q.astype(int), p=d.f_ends_punct.astype(int), el=d.f_ellipsis.astype(int),
                                            lc=np.log1p(d.f_n_chars), pos=d.pos.clip(upper=4), sl=d.f_starts_lower.astype(int)).astype(float)
    p = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X, d.cont, d.conv_id):
        p[te] = LogisticRegression(max_iter=2000).fit(X.values[tr], d.cont.values[tr]).predict_proba(X.values[te])[:, 1]
    r["code_logit_auc"] = round(roc_auc_score(d.cont, p), 3)
    d = d.assign(p_code=p)
    res[c] = r
    M.loc[d.index, "p_code"] = p
print(json.dumps(res, indent=1, default=str))

# ---- Jev noul on a sample
S = pd.concat([M[(M.corpus == c) & M.next_gap.notna() & M.p_code.notna()].sample(250, random_state=8) for c in ["maichat", "whatsapp_nl"]])
Mi = {c: d for c, d in M.groupby("conv_id")}
items = []
for _, m in S.iterrows():
    d = Mi[m.conv_id]
    prev = d[(d.idx < m.idx) & (d.idx >= m.idx - 12)]
    turns, cur = [], None
    for x in prev.itertuples():
        if cur and cur["speaker"] == x.speaker: cur["messages"].append(x.text.strip()[:200])
        else:
            cur = {"speaker": x.speaker, "messages": [x.text.strip()[:200]]}; turns.append(cur)
    if turns and turns[-1]["speaker"] == m.speaker:
        so_far = turns.pop()["messages"] + [m.text.strip()[:200]]
    else:
        so_far = [m.text.strip()[:200]]
    st = {"conversation_before": turns[-6:], "current_speaker": m.speaker, "current_speaker_messages_so_far": so_far,
          "note": "current_speaker has just sent the last message in current_speaker_messages_so_far."}
    items.append((st, {"done": noul("Is `current_speaker` done with their turn, i.e. will they now wait for the other person to reply instead of sending another message right away?")}))
ans = ask_many(items, workers=4)
print(summary())
S["p_jev_cont"] = [None if a is None else 1 - a["done"]["noul"] for a in ans]
S = S.dropna(subset=["p_jev_cont"])
jr = {}
for c, d in S.groupby("corpus"):
    jr[c] = {"n": len(d), "rate_cont": round(d.cont.mean(), 3), "auc_jev": round(roc_auc_score(d.cont, d.p_jev_cont), 3),
             "auc_code": round(roc_auc_score(d.cont, d.p_code), 3), "auc_avg": round(roc_auc_score(d.cont, (d.p_code + d.p_jev_cont) / 2), 3),
             "brier_jev": round(brier_score_loss(d.cont, d.p_jev_cont), 4), "brier_code": round(brier_score_loss(d.cont, d.p_code), 4),
             "mean_p_jev": round(d.p_jev_cont.mean(), 3)}
res["jev_sample"] = jr
print(json.dumps(jr, indent=1))
S[["corpus", "conv_id", "idx", "cont", "p_code", "p_jev_cont"]].to_csv(os.path.join(OUT, "a1_turnend_sample.csv"), index=False)
json.dump(res, open(os.path.join(OUT, "a1_turnend.json"), "w"), indent=1, default=str)
