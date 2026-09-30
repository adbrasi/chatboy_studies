"""a1: validation of Jev's predictive pass (p_n_msgs, p_length) against the real next turn, vs code baselines;
and whether combining Jev P with code features helps (grouped CV logistic / linear regression)."""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score, f1_score, r2_score
sys.path.insert(0, os.path.dirname(__file__))
from a1_load import enriched

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "data")
T, M = enriched()
wmap = dict(zip(zip(M.conv_id, M.idx), M.f_n_words))
T["n_words"] = [sum(wmap[(c, i)] for i in ix) for c, ix in zip(T.conv_id, T.msg_idxs)]
T = T.sort_values(["conv_id", "turn_idx"])
g = T.groupby("spk")
T["own_mean_nm_past"] = g.n_msgs.transform(lambda s: s.shift(1).expanding().mean())
T["own_multi_rate_past"] = g.n_msgs.transform(lambda s: (s.shift(1) > 1).expanding().mean())
T["own_mean_logch_past"] = g.total_chars.transform(lambda s: np.log(s).shift(1).expanding().mean())
T["own_n_past"] = g.cumcount()
A = T[T.has_P == True].copy()
A = A[A.own_n_past >= 1]
cls = lambda n: np.clip(n, 1, 4).astype(int)
A["y"] = cls(A.n_msgs)
A["y_multi"] = (A.n_msgs > 1).astype(int)
A["P_cls"] = A.P_p_n_msgs.map({"1": 1, "2": 2, "3": 3, "4+": 4})
A["P_exp"] = A.P_nm_1 * 1 + A.P_nm_2 * 2 + A.P_nm_3 * 3 + A["P_nm_4+"] * 4.5
A["P_multi"] = 1 - A.P_nm_1
def lenlvl(w):
    return np.select([w <= 3, w <= 8, w <= 25, w <= 60], [0, 1, 2, 3], 4)
A["y_len"] = lenlvl(A.n_words)
res = {}
for c in ["maichat", "whatsapp_nl"]:
    d = A[A.corpus == c].dropna(subset=["prev_own_nmsgs", "own_mean_nm_past"]).copy()
    d["prev_partner_nmsgs"] = d.prev_partner_nmsgs.fillna(1)
    d["prev_partner_chars"] = d.prev_partner_chars.fillna(d.prev_partner_chars.median())
    preds = {"always_1": np.ones(len(d), int), "jev_argmax": d.P_cls.values, "prev_own": cls(d.prev_own_nmsgs.values),
             "mirror_partner": cls(d.prev_partner_nmsgs.values), "own_running_mean(rounded)": cls(np.round(d.own_mean_nm_past.values))}
    scores_rank = {"jev_expected": d.P_exp, "prev_own": d.prev_own_nmsgs, "mirror_partner": d.prev_partner_nmsgs,
                   "own_running_mean": d.own_mean_nm_past, "partner_chars": d.prev_partner_chars}
    r = {"n": len(d), "actual_dist": d.y.value_counts(normalize=True).sort_index().round(3).to_dict(),
         "jev_pred_dist": d.P_cls.value_counts(normalize=True).sort_index().round(3).to_dict()}
    r["accuracy"] = {k: round(accuracy_score(d.y, v), 3) for k, v in preds.items()}
    r["macroF1"] = {k: round(f1_score(d.y, v, average="macro"), 3) for k, v in preds.items()}
    r["spearman_vs_n_msgs"] = {k: round(stats.spearmanr(v, d.n_msgs)[0], 3) for k, v in scores_rank.items()}
    r["auc_multi"] = {k: round(roc_auc_score(d.y_multi, v), 3) for k, v in scores_rank.items()}
    r["brier_multi"] = {"jev_P_multi": round(brier_score_loss(d.y_multi, d.P_multi), 4),
                        "base_rate": round(brier_score_loss(d.y_multi, np.full(len(d), d.y_multi.mean())), 4),
                        "own_past_rate": round(brier_score_loss(d.y_multi, d.own_multi_rate_past.fillna(d.y_multi.mean())), 4)}
    cal = d.assign(b=pd.cut(d.P_multi, [-.01, .05, .15, .3, .5, .7, 1.0])).groupby("b", observed=True).agg(n=("y_multi", "size"), pred=("P_multi", "mean"), obs=("y_multi", "mean")).round(3)
    r["calibration_multi"] = cal.rename(index=str).to_dict(orient="index")
    r["mean_P_multi_vs_rate"] = [round(d.P_multi.mean(), 3), round(d.y_multi.mean(), 3)]
    # confidence gating
    hi = d[d.P_nm_conf >= .8]
    r["high_conf(>=.8)"] = {"share": round(len(hi) / len(d), 3), "acc_jev": round(accuracy_score(hi.y, hi.P_cls), 3), "acc_always1": round((hi.y == 1).mean(), 3)}
    # length
    lr = {"n": len(d), "actual_len_dist": pd.Series(d.y_len).value_counts(normalize=True).sort_index().round(3).to_dict(),
          "spearman_vs_chars": {"jev_p_length": round(stats.spearmanr(d.P_p_length, d.total_chars)[0], 3),
                                "partner_prev_chars": round(stats.spearmanr(d.prev_partner_chars, d.total_chars)[0], 3),
                                "own_prev_chars": round(stats.spearmanr(d.prev_own_chars, d.total_chars)[0], 3),
                                "own_running_mean": round(stats.spearmanr(d.own_mean_logch_past, d.total_chars)[0], 3)},
          "acc_level(round)": {"jev": round(accuracy_score(d.y_len, np.round(d.P_p_length).astype(int)), 3),
                               "majority": round(pd.Series(d.y_len).value_counts(normalize=True).max(), 3)},
          "mean_abs_level_err": {"jev": round(float(np.mean(np.abs(d.y_len - d.P_p_length))), 3),
                                 "const_median": round(float(np.mean(np.abs(d.y_len - np.median(d.y_len)))), 3)},
          "bias(mean pred - mean actual level)": round(float(d.P_p_length.mean() - d.y_len.mean()), 3)}
    # combined models, grouped CV by conversation
    feats_code = ["prev_own_nmsgs", "prev_partner_nmsgs", "own_mean_nm_past", "own_multi_rate_past", "lp", "lpp", "ltis"]
    feats_jev = ["P_nm_1", "P_nm_2", "P_nm_3", "P_nm_4+", "P_p_length", "P_nm_conf"]
    d["lp"] = np.log(d.prev_partner_chars); d["lpp"] = np.log(d.prev_own_chars.fillna(20)); d["ltis"] = np.log1p(d.turn_in_session)
    d["own_multi_rate_past"] = d.own_multi_rate_past.fillna(d.y_multi.mean())
    gkf = GroupKFold(5)
    cv = {}
    for name, fs in {"code": feats_code, "jev": feats_jev, "code+jev": feats_code + feats_jev}.items():
        p = np.zeros(len(d)); pl = np.zeros(len(d))
        X = d[fs].values; Xl = d[fs + []].values
        for tr, te in gkf.split(X, d.y_multi, d.conv_id):
            m = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], d.y_multi.values[tr]); p[te] = m.predict_proba(X[te])[:, 1]
            ml = LinearRegression().fit(X[tr], np.log(d.total_chars.values[tr])); pl[te] = ml.predict(X[te])
        cv[name] = {"auc_multi": round(roc_auc_score(d.y_multi, p), 3), "logloss": round(log_loss(d.y_multi, p), 4),
                    "brier": round(brier_score_loss(d.y_multi, p), 4), "r2_logchars": round(r2_score(np.log(d.total_chars), pl), 3),
                    "spearman_len": round(stats.spearmanr(pl, d.total_chars)[0], 3)}
    cv["jev_raw_P_multi"] = {"auc_multi": round(roc_auc_score(d.y_multi, d.P_multi), 3), "logloss": round(log_loss(d.y_multi, d.P_multi.clip(.01, .99)), 4)}
    r["length"] = lr
    r["cv_combined"] = cv
    res[c] = r
    print("\n==", c); print(json.dumps(r, indent=1, default=str))
json.dump(res, open(os.path.join(OUT, "a1_pvalid.json"), "w"), indent=1, default=str)
