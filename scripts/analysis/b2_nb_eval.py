"""b2 Parte B: avaliação das arquiteturas de nº de bolhas (cenários T e P) + latência.
Escolha no DEV (RPS = ranked probability score, a métrica própria para distribuição ordinal), relato no TESTE com IC
por bootstrap de conversas. Modelos de código e T7/P7 (Jev como features + regressão ordinal) treinados só no dev.
Saída: analysis/data/b2_nb_results.json, b2_nb_preds_test.csv.gz"""
import os, json, re, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from statsmodels.miscmodels.ordinal_model import OrderedModel
from b2_nb_common import *
from b2_common import boot_ci, dump, OUT

warnings.filterwarnings("ignore")
KS = np.arange(1, K + 1)

# ---------------- code features for every turn (full table) ----------------
T = full_T().sort_values(["conv_id", "turn_idx"]).reset_index(drop=True)
T["text"] = [" ".join(str(x).strip() for x in tx) for tx in T.y_texts]
T["chars"] = T.text.str.len()
T["lchars"] = np.log1p(T.chars)
T["n_sent"] = [n_sentences(x) for x in T.text]
T["caps_mid"] = [len(re.findall(r"(?<![.!?]) [A-Z]", x)) for x in T.text]
T["n_punct"] = [len(re.findall(r"[.!?,;:]", x)) for x in T.text]
T["yc"] = T.y.clip(upper=5)
base = T[T.split == "dev"].groupby("corpus").yc.apply(lambda s: (s > 1).mean()).to_dict()
basemean = T[T.split == "dev"].groupby("corpus").yc.mean().to_dict()
g = T.groupby(["conv_id", "speaker"])
nb = g.cumcount()
multi_before = g.yc.transform(lambda s: (s > 1).astype(int).cumsum().shift(1).fillna(0))
sum_before = g.yc.transform(lambda s: s.cumsum().shift(1).fillna(0))
T["own_n"] = nb
T["own_rmulti"] = (multi_before + 5 * T.corpus.map(base)) / (nb + 5)
T["own_mean"] = (sum_before + 5 * T.corpus.map(basemean)) / (nb + 5)
T["own_last"] = g.yc.shift(1).fillna(1)
T["own_lchars"] = g.lchars.transform(lambda s: s.expanding().mean().shift(1)).fillna(T.lchars.mean())
gc = T.groupby("conv_id")
same = gc.session.shift(1) == T.session
T["pt_n"] = gc.yc.shift(1).where(same).fillna(1)
T["pt_lchars"] = np.log1p(gc.chars.shift(1).where(same).fillna(20))
T["pt_q"] = gc.has_q.shift(1).where(same).fillna(False).astype(int)
T["pt_lat"] = np.log1p(gc.response_latency_s.shift(1).where(same))
T["own_lat_med"] = T.groupby(["conv_id", "speaker"]).response_latency_s.transform(lambda s: s.expanding().median().shift(1))
T["own_llat"] = np.log1p(T.own_lat_med)
T["wa"] = (T.corpus == "whatsapp_nl").astype(int)
T["lat"] = T.response_latency_s
T["slow"] = np.where(T.corpus == "maichat", T.lat > 20, T.lat >= 300).astype(float)
T.loc[T.lat.isna(), "slow"] = np.nan
TI = T.set_index(["conv_id", "turn_idx"])
CODE_T = ["lchars", "n_sent", "caps_mid", "n_punct", "own_rmulti", "own_mean", "pt_n", "pt_lchars", "wa"]
CODE_P = ["own_rmulti", "own_mean", "own_last", "own_lchars", "pt_n", "pt_lchars", "pt_q", "wa"]


# ---------------- metrics ----------------
def onehot(y):
    Y = np.zeros((len(y), K)); Y[np.arange(len(y)), np.clip(y, 1, K) - 1] = 1; return Y


def rps(P, y):
    return float(np.mean(np.sum((np.cumsum(P, 1) - np.cumsum(onehot(y), 1)) ** 2, 1) / (K - 1)))


def metr(P, y):
    P = np.asarray(P, float); y = np.asarray(y, int)
    emp = onehot(y).mean(0); pm = P.mean(0)
    pm2 = 1 - P[:, 0]; ym = (y >= 2).astype(int)
    q = pd.qcut(pd.Series(pm2).rank(method="first"), 5, labels=False)
    ece = float(np.mean([abs(pm2[q == b].mean() - ym[q == b].mean()) for b in range(5)]))
    return {"rps": rps(P, y), "acc": float(np.mean(P.argmax(1) + 1 == y)),
            "rho": float(stats.spearmanr(P @ KS, y)[0]) if np.std(P @ KS) > 0 else 0.0,
            "auc2": float(roc_auc_score(ym, pm2)) if np.std(pm2) > 0 else 0.5,
            "brier2": float(np.mean((pm2 - ym) ** 2)), "ece2": ece,
            "tvd": float(0.5 * np.abs(pm - emp).sum()), "mean_p_multi": float(pm2.mean()), "real_multi": float(ym.mean()),
            "pred_dist": [round(float(x), 3) for x in pm], "real_dist": [round(float(x), 3) for x in emp],
            "argmax_dist": [round(float(np.mean(P.argmax(1) == k)), 3) for k in range(K)]}


def ci_block(df, P, name):
    d = df.assign(**{f"p{k}": P[:, k] for k in range(K)})
    def f(key):
        return lambda x: metr(x[[f"p{k}" for k in range(K)]].values, x.y.values)[key]
    out = {}
    for key in ("rps", "acc", "rho", "auc2", "tvd"):
        pt, lo, hi = boot_ci(d, f(key), n=400)
        out[key] = [round(pt, 3), round(lo, 3), round(hi, 3)]
    return out


def fit_ordinal(X, y):
    Xs = (X - X.mean()) / (X.std() + 1e-9)
    m = OrderedModel(y, Xs, distr="logit")
    r = m.fit(method="bfgs", disp=False, maxiter=2000)
    return r, X.mean(), X.std() + 1e-9


def pred_ordinal(fit, X):
    r, mu, sd = fit
    return np.asarray(r.model.predict(r.params, exog=((X - mu) / sd)))


def ordinal_probs(Xtr, ytr, Xte):
    # classes present in training may be < 5 -> map back
    fit = fit_ordinal(Xtr, pd.Series(pd.Categorical(ytr, categories=sorted(set(ytr)), ordered=True)))
    P = pred_ordinal(fit, Xte)
    full = np.zeros((len(Xte), K))
    for j, c in enumerate(sorted(set(ytr))):
        full[:, c - 1] = P[:, j]
    return full


def smooth(P, eps=0.01):
    P = np.asarray(P, float) + eps
    return P / P.sum(1, keepdims=True)


# ---------------- load raw ----------------
rawT = pd.read_pickle(os.path.join(SCR, "nb_T_raw.pkl"))
R, out = rawT["R"], rawT["out"]
R = R.reset_index(drop=True)
R["y"] = R.y.clip(upper=5).astype(int)
for c in set(CODE_T + CODE_P + ["lat", "slow", "pt_lat", "own_llat"]):
    R[c] = [TI.loc[k, c] for k in R.key]
ok = np.array([all(out[a][i] is not None for a in ("T1", "T2", "A", "T3", "T4", "T5")) for i in range(len(R))])
R = R[ok].reset_index(drop=True)
idx = np.where(ok)[0]
PT = {}
for a in ("T1", "T2", "T3", "T4"):
    PT[a] = np.array([dist_from_choice(out[a][i], "n") for i in idx])
# T5 cuts
cuts_p = []
for i in idx:
    ans = out["T5"][i]
    cuts_p.append({int(k[1:]): v["noul"] for k, v in ans.items() if k.startswith("c")})
R["cuts_p"] = cuts_p
PT["T5"] = np.array([poisson_binomial(list(c.values())) for c in cuts_p])
PT["T5_thr"] = np.array([onehot([min(1 + sum(p > .5 for p in c.values()), K)])[0] for c in cuts_p])
# T6 cascade
ps6 = [out["T6"][i] for i in idx]
devmask = (R.split == "dev").values
mean_pk = {k: np.mean([p[k] for p, dm in zip(ps6, devmask) if dm and k in p] or [0.3]) for k in (2, 3, 4, 5)}
P6, P6thr = [], []
for p in ps6:
    surv, dist = 1.0, []
    for k in (2, 3, 4, 5):
        pk = p.get(k, mean_pk[k] if k > 2 else 0.5)
        dist.append(surv * (1 - pk)); surv *= pk
    dist.append(surv)
    P6.append(dist)
    n = 1
    for k in (2, 3, 4, 5):
        if p.get(k, 0) >= 0.5: n = k
        else: break
    P6thr.append(onehot([n])[0])
PT["T6"] = np.array(P6); PT["T6_thr"] = np.array(P6thr)

# Jev features for T7
A = [out["A"][i] for i in idx]
R["j_energy"] = [a["energy"]["score"] for a in A]; R["j_amount"] = [a["amount"]["score"] for a in A]
R["j_serious"] = [a["serious"]["score"] for a in A]
R["j_points"] = [sum(a["points"]["probabilities"].get(k, 0) * w for k, w in (("1", 1), ("2", 2), ("3", 3), ("4+", 4.5))) for a in A]
for k in ("anxious", "playful", "tension", "excited", "defensive", "thinking_aloud", "story", "opens_reaction", "ends_question"):
    R["j_" + k] = [a[k]["noul"] for a in A]
R["j_T2exp"] = PT["T2"] @ KS
R["j_T5sum"] = [sum(c.values()) for c in cuts_p]
R["j_T3exp"] = PT["T3"] @ KS
JEV_T = ["j_energy", "j_amount", "j_serious", "j_points", "j_playful", "j_tension", "j_opens_reaction", "j_ends_question", "j_thinking_aloud", "j_T5sum", "j_T2exp"]

# ---- code baselines for T ----
Tdev = T[T.split == "dev"]
lb_edges = [0, 20, 40, 80, 160, 1e9]
Tdev_lb = pd.cut(Tdev.chars, lb_edges)
tab = {(c, b): onehot(x.yc.values).mean(0) for (c, b), x in Tdev.groupby(["corpus", Tdev_lb], observed=True)}
R["lbin"] = pd.cut(R.chars, lb_edges)
PT["C_len_table"] = np.array([tab[(c, b)] for c, b in zip(R.corpus, R.lbin)])
PT["C_base_rate"] = np.array([onehot(Tdev[Tdev.corpus == c].yc.values).mean(0) for c in R.corpus])
# own-history mixture: own distribution (smoothed with corpus base, weight 5)
def own_dist(key, corpus):
    conv, ti = key
    past = T[(T.conv_id == conv) & (T.turn_idx < ti) & (T.speaker == TI.loc[key, "speaker"])]
    return (onehot(past.yc.values).sum(0) + 5 * PT_base[corpus]) / (len(past) + 5)
PT_base = {c: onehot(Tdev[Tdev.corpus == c].yc.values).mean(0) for c in ("maichat", "whatsapp_nl")}
PT["C_own_history"] = np.array([own_dist(k, c) for k, c in zip(R.key, R.corpus)])
PT["C_mirror_partner"] = np.array([0.5 * onehot([int(n)])[0] + 0.5 * PT_base[c] for n, c in zip(R.pt_n, R.corpus)])
# ordinal code: trained on ALL dev turns / on dev sample only
Tdev_s = Tdev.sample(min(len(Tdev), 12000), random_state=0)
PT["C_ordinal_alldev"] = smooth(ordinal_probs(Tdev_s[CODE_T].astype(float), Tdev_s.yc.values, R[CODE_T].astype(float)))
Rd = R[R.split == "dev"]
PT["C_ordinal_devsample"] = smooth(ordinal_probs(Rd[CODE_T].astype(float), Rd.y.values, R[CODE_T].astype(float)))
PT["T7_ordinal_code+jev"] = smooth(ordinal_probs(Rd[CODE_T + JEV_T].astype(float), Rd.y.values, R[CODE_T + JEV_T].astype(float)))
PT["T7_ordinal_jev_only"] = smooth(ordinal_probs(Rd[JEV_T + ["lchars"]].astype(float), Rd.y.values, R[JEV_T + ["lchars"]].astype(float)))
# T8: report-01 recipe (length table × Jev modulators, code) — shift mass by arousal/playful/serious/tension
def recipe(r, P):
    shift = 0.2 * (r.j_energy - 1.5) + (0.3 if r.j_playful > .6 else 0) - (0.3 if r.j_serious >= 2 else 0) - (0.3 if r.j_tension > .6 else 0)
    shift += 0.15 if r.own_rmulti > .5 else (-0.15 if r.own_rmulti < .15 else 0)
    e = P @ KS + shift
    # re-center distribution by moving mass: mixture between P and P shifted by one class
    up = np.r_[0, P[:-1]]; up[-1] += P[-1]; dn = np.r_[P[1:], 0]; dn[0] += P[0]
    return (1 - abs(shift)) * P + abs(shift) * (up if shift > 0 else dn) if abs(shift) <= 1 else P
PT["T8_recipe_report01"] = np.array([recipe(r, P) for r, P in zip(R.itertuples(), PT["C_len_table"])])
PT["T8_mix_T2+len"] = 0.5 * PT["T2"] + 0.5 * PT["C_len_table"]

# LLM splitter (T9)
llm_path = os.path.join(SCR, "nb_llm_raw.pkl")
if os.path.exists(llm_path):
    L = pd.read_pickle(llm_path)
    kpos = {k: i for i, k in enumerate(L["keys"])}
    for m, txts in L["res"].items():
        ns = []
        for k in R.key:
            t = txts[kpos[k]]
            n = len([l for l in (t or "").split("\n") if l.strip() and not re.fullmatch(r"[-*`\"]+", l.strip())]) if t else 1
            ns.append(max(1, min(n, K)))
        PT["T9_llm_" + m.split("/")[-1]] = smooth(onehot(ns), 0.02)
        R["llm_n_" + m.split("/")[-1]] = ns

# ---------------- evaluate T ----------------
res = {"T": {}, "P": {}, "lat": {}}
for name, P in PT.items():
    P = smooth(P, 1e-3)
    blk = {}
    for s in ("dev", "test"):
        m = (R.split == s).values
        blk[s] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metr(P[m], R.y.values[m]).items()}
        for c in ("maichat", "whatsapp_nl"):
            mc = m & (R.corpus == c).values
            blk[f"{s}_{c}"] = {k: round(v, 4) for k, v in metr(P[mc], R.y.values[mc]).items() if k in ("rps", "acc", "rho", "auc2", "tvd", "mean_p_multi", "real_multi")}
    mt = (R.split == "test").values
    blk["test_ci"] = ci_block(R[mt].reset_index(drop=True), P[mt], name)
    res["T"][name] = blk
    print("T", name, blk["dev"]["rps"], blk["test"]["rps"], blk["test"]["acc"], blk["test"]["rho"], blk["test"]["auc2"], blk["test"]["tvd"], flush=True)

# boundary accuracy for T5 and LLM
def f1_cuts(pred, true):
    pred, true = set(pred), set(true)
    if not pred and not true: return 1.0
    tp = len(pred & true); p = tp / len(pred) if pred else 0; r = tp / len(true) if true else 0
    return 2 * p * r / (p + r) if p + r else 0.0
mt = (R.split == "test")
res["T"]["T5"]["cut_f1_test_multi_only"] = float(np.mean([f1_cuts([g for g, p in c.items() if p > .5], tc) for c, tc, y in zip(R.cuts_p[mt], R.true_cuts[mt], R.y[mt]) if y > 1]))
res["T"]["T5"]["cut_precision_at_true_n"] = float(np.mean([len(set(sorted(c, key=c.get, reverse=True)[:y - 1]) & set(tc)) / (y - 1) for c, tc, y in zip(R.cuts_p[mt], R.true_cuts[mt], R.y[mt]) if y > 1 and c]))
# chance: random candidates at true n
res["T"]["T5"]["cut_precision_chance"] = float(np.mean([(y - 1) / max(len(c), 1) for c, y in zip(R.cuts_p[mt], R.y[mt]) if y > 1 and c]))
R.to_pickle(os.path.join(SCR, "nb_T_eval_R.pkl"))
pd.to_pickle(PT, os.path.join(SCR, "nb_T_eval_PT.pkl"))

# ---------------- evaluate P ----------------
rawP = pd.read_pickle(os.path.join(SCR, "nb_P_raw.pkl"))
RP, outP = rawP["R"].reset_index(drop=True), rawP["out"]
RP["y"] = RP.y.clip(upper=5).astype(int)
for c in set(CODE_P + ["lat", "slow", "pt_lat", "own_llat"]):
    RP[c] = [TI.loc[k, c] for k in RP.key]
okP = np.array([all(outP[a][i] is not None for a in ("P1", "P2", "A", "P3")) for i in range(len(RP))])
idxP = np.where(okP)[0]; RP = RP[okP].reset_index(drop=True)
PP = {a: np.array([dist_from_choice(outP[a][i], "n") for i in idxP]) for a in ("P1", "P2", "P3")}
AP = [outP["A"][i] for i in idxP]
for k in ("anxious", "excited", "playful", "tension", "challenged", "story", "asked", "ending", "needs_thought"):
    RP["j_" + k] = [a[k]["noul"] for a in AP]
for k in ("energy", "amount", "serious"):
    RP["j_" + k] = [a[k]["score"] for a in AP]
RP["j_P2exp"] = PP["P2"] @ KS
JEV_P = ["j_amount", "j_energy", "j_serious", "j_anxious", "j_excited", "j_playful", "j_tension", "j_challenged", "j_story", "j_asked", "j_P2exp"]
PP["C_base_rate"] = np.array([PT_base[c] for c in RP.corpus])
PP["C_own_history"] = np.array([own_dist(k, c) for k, c in zip(RP.key, RP.corpus)])
PP["C_mirror_partner"] = np.array([0.5 * onehot([int(n)])[0] + 0.5 * PT_base[c] for n, c in zip(RP.pt_n, RP.corpus)])
PP["C_ordinal_alldev"] = smooth(ordinal_probs(Tdev_s[CODE_P].astype(float), Tdev_s.yc.values, RP[CODE_P].astype(float)))
RPd = RP[RP.split == "dev"]
PP["C_ordinal_devsample"] = smooth(ordinal_probs(RPd[CODE_P].astype(float), RPd.y.values, RP[CODE_P].astype(float)))
PP["P7_ordinal_code+jev"] = smooth(ordinal_probs(RPd[CODE_P + JEV_P].astype(float), RPd.y.values, RP[CODE_P + JEV_P].astype(float)))
PP["P8_mix_P1+own"] = 0.5 * PP["P1"] + 0.5 * PP["C_own_history"]
for name, P in PP.items():
    P = smooth(P, 1e-3)
    blk = {}
    for s in ("dev", "test"):
        m = (RP.split == s).values
        blk[s] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metr(P[m], RP.y.values[m]).items()}
    mt = (RP.split == "test").values
    blk["test_ci"] = ci_block(RP[mt].reset_index(drop=True), P[mt], name)
    res["P"][name] = blk
    print("P", name, blk["dev"]["rps"], blk["test"]["rps"], blk["test"]["acc"], blk["test"]["rho"], blk["test"]["auc2"], blk["test"]["tvd"], flush=True)

# ---------------- latency (P position) ----------------
for a in ("P1", "P2", "P3"):
    RP[f"lat_{a}"] = [outP[a][i]["lat"]["score"] for i in idxP]
    RP[f"fast_{a}"] = [outP[a][i]["fast"]["noul"] for i in idxP]
RP["pt_lat_f"] = RP.pt_lat.fillna(RP.groupby("corpus").pt_lat.transform("median"))
RP["own_llat_f"] = RP.own_llat.fillna(RP.groupby("corpus").own_llat.transform("median"))
LC = ["pt_lat_f", "own_llat_f", "pt_q", "pt_lchars", "wa"]
LJ = ["lat_P2", "fast_P2", "j_needs_thought", "j_ending", "j_asked", "j_amount", "j_serious"]
dl = RP.dropna(subset=["slow"])
for c in ("maichat", "whatsapp_nl"):
    dc = dl[dl.corpus == c]
    dd, dt = dc[dc.split == "dev"], dc[dc.split == "test"]
    blk = {"n_test": len(dt), "slow_rate_test": round(float(dt.slow.mean()), 3), "definition": "slow = latency > 20 s" if c == "maichat" else "slow = latency >= 5 min"}
    preds = {f"{a}_lat_score": dt[f"lat_{a}"] for a in ("P1", "P2", "P3")}
    preds |= {f"{a}_not_fast": 1 - dt[f"fast_{a}"] for a in ("P1", "P2", "P3")}
    preds["code_partner_latency(mirror)"] = dt.pt_lat_f
    preds["code_own_median_latency"] = dt.own_llat_f
    for nm, cols in (("code_logit", LC[:-1]), ("jev_logit", LJ), ("code+jev_logit", LC[:-1] + LJ)):
        m = LogisticRegression(max_iter=2000).fit(dd[cols], dd.slow)
        preds[nm] = pd.Series(m.predict_proba(dt[cols])[:, 1], index=dt.index)
    for nm, p in preds.items():
        d = dt.assign(p=p.values)
        pt, lo, hi = boot_ci(d, lambda x: roc_auc_score(x.slow, x.p) if x.slow.nunique() > 1 else np.nan, n=400)
        rho = stats.spearmanr(d.p, d.lat)[0]
        blk[nm] = {"auc_slow": [round(pt, 3), round(lo, 3), round(hi, 3)], "rho_latency": round(float(rho), 3)}
    res["lat"][c] = blk
    print("LAT", c, {k: v for k, v in blk.items() if isinstance(v, dict)}, flush=True)

# multi-bubble "fast reply" link: in maichat, fast replies are more fragmented (report 01) -> check on sample
dump(res, "b2_nb_results.json")
keep = ["corpus", "split", "conv_id", "y", "chars", "text"]
pt = R[R.split == "test"][keep].copy()
for k in ("T1", "T2", "T3", "T4", "T5", "T6", "C_len_table", "T7_ordinal_code+jev", "C_ordinal_alldev"):
    pt[k + "_exp"] = (PT[k][(R.split == "test").values] @ KS).round(2)
pt.to_csv(os.path.join(OUT, "b2_nb_preds_test.csv.gz"), index=False)
print("done")
