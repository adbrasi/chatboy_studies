"""a2: does P.p_end (jev_base) predict the end? AUC per corpus and horizon, vs. code baselines and a combined model."""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(__file__))
import a2_load as L
from a2_closings import auc, boot_auc
j = pd.read_pickle(f"{L.SCR}/closing.pkl")
j = j[j.has_P].copy()  # turns with predictive pass (not first of session)
j["closing_T"] = j.bye | j.closing_intent
out = {}
for c in ["maichat", "whatsapp_nl"]:
    x = j[j.corpus == c].copy()
    res = {}
    targets = {"T_is_last": x.to_end == 0, "T_closing(bye|intent)": x.closing_T, "end_within_3": x.to_end <= 2}
    feats = {"P_p_end": x.P_p_end, "-prev_chars": -x.prev_chars, "prev_bye": x.prev_bye.astype(float),
             "prev_winding(D)": x.prev_winding.astype(float), "-prev_engagement(D)": -x.prev_eng,
             "turn_in_session": x.turn_in_session.astype(float), "-P_p_length": -x.P_p_length, "-P_p_question": -x.P_p_question}
    for tn, y in targets.items():
        res[tn] = {"n": int(len(y)), "pos": int(y.sum())}
        for fn, s in feats.items():
            a = auc(y, s)
            res[tn][fn] = round(a, 3)
        lo, hi = boot_auc(y.values, x.P_p_end.values)
        res[tn]["P_p_end_CI"] = [round(lo, 3), round(hi, 3)]
        # combined model (grouped CV by conversation)
        X = pd.DataFrame({"pend": x.P_p_end, "lpc": np.log1p(x.prev_chars.fillna(x.prev_chars.median())),
                          "pb": x.prev_bye.fillna(0).astype(float), "pw": x.prev_winding.fillna(0).astype(float),
                          "pe": x.prev_eng.fillna(2), "tis": np.log1p(x.turn_in_session), "pl": x.P_p_length,
                          "pq": x.P_p_question}).fillna(0)
        pred = np.zeros(len(X))
        for tr, te in GroupKFold(5).split(X, y, x.conv_id):
            mdl = LogisticRegression(max_iter=1000).fit(X.iloc[tr], y.iloc[tr])
            pred[te] = mdl.predict_proba(X.iloc[te])[:, 1]
        res[tn]["combined_cv"] = round(auc(y, pred), 3)
    # horizon: mean p_end by to_end and AUC of (to_end==k) vs (to_end>=8)
    hz = {}
    for k in range(0, 7):
        y = x[(x.to_end == k) | (x.to_end >= 8)]
        hz[k] = {"mean_p_end": round(x[x.to_end == k].P_p_end.mean(), 3),
                 "auc_vs_far": round(auc(y.to_end == k, y.P_p_end), 3), "n": int((x.to_end == k).sum())}
    hz["far(>=8)"] = {"mean_p_end": round(x[x.to_end >= 8].P_p_end.mean(), 3), "n": int((x.to_end >= 8).sum())}
    res["horizon"] = hz
    # precision at threshold
    thr = {}
    for t in [0.3, 0.5, 0.7]:
        sel = x.P_p_end >= t
        thr[t] = {"share_flagged": round(sel.mean(), 3), "prec_closing_T": round(x[sel].closing_T.mean(), 3) if sel.any() else None,
                  "prec_end_within_3": round((x[sel].to_end <= 2).mean(), 3) if sel.any() else None,
                  "recall_closing_T": round(x[sel].closing_T.sum() / max(1, x.closing_T.sum()), 3)}
    res["thresholds"] = thr
    out[c] = res
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../../analysis/data/a2_pend_auc.json"), "w"), indent=1)
