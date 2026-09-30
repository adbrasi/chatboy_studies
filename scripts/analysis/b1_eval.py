"""b1 evaluation: every architecture, dev/test split by conversation, cluster-bootstrap CIs, length-controlled AUC.
Output: analysis/data/b1_results.json, b1_question_auc.csv"""
import csv, json, math, os, re, sys, warnings
from collections import defaultdict, Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold
import b1_common as B
import b1_questions as Q
import a8_common as C8

warnings.filterwarnings("ignore")
R = {}
J = lambda n: json.load(open(os.path.join(B.SCR, n))) if os.path.exists(os.path.join(B.SCR, n)) else {}

ctxs, units = B.load_units()
bank = J("bank.json")
U = [u for u in units if u["uid"] in bank]
for u in U:
    c = ctxs[u["ckey"]]
    u["split"], u["conv"], u["src"] = c["split"], c["conv"], c["src"]
    u["chars"] = len(u["text"])
    u["cf"] = B.code_feats(u["text"], B.last_other(c))
    sf = C8.style_feats(u["text"], B.last_other(c))
    u["cf"]["slang_forced"] = int(sf["llm:gíria forçada (bestie/slay/no cap/fr fr/vibing/elite/iconic/lowkey)"] or sf["llm:💀/😭 (gíria emoji)"])
HUMLEN = {u["ckey"]: u["chars"] for u in U if u["cond"] == "human"}
NUMQ = [k for k in Q.BANK if k != "v_main"]
BYUID = {u["uid"]: u for u in U}
print("units with bank", len(U), Counter(u["cond"] for u in U))


def arr(us, key):
    return np.array([key(u) for u in us], float)


def group_of(cond):
    if cond == "human":
        return "human"
    return "instructed" if cond in B.INSTRUCTED else "base"


def evaluate(us, scores, boot=400):
    """Overall + per-condition test metrics. scores: dict uid->score (higher = more LLM)."""
    us = [u for u in us if u["uid"] in scores and scores[u["uid"]] == scores[u["uid"]]]
    lab = arr(us, lambda u: u["label"]); s = arr(us, lambda u: scores[u["uid"]]); L = arr(us, lambda u: u["chars"])
    cl = [u["conv"] for u in us]
    out = {"all": B.eval_scores(s, lab, L, cl, boot)}
    for grp in ("base", "instructed"):
        m = [(u["label"] == 0 or group_of(u["cond"]) == grp) for u in us]
        idx = np.where(m)[0]
        out[grp] = B.eval_scores(s[idx], lab[idx], L[idx], [cl[i] for i in idx], boot)
    per = {}
    for cn in sorted({u["cond"] for u in us} - {"human"}):
        src = "a9" if cn in B.A9_CONDS else "a8"
        idx = [i for i, u in enumerate(us) if u["cond"] == cn or (u["cond"] == "human" and u["src"].startswith(src))]
        per[cn] = B.eval_scores(s[idx], lab[idx], L[idx], [cl[i] for i in idx], 0)
    out["per_cond"] = {k: {"auc": v["auc"], "auc_len": v["auc_len"]} for k, v in per.items()}
    return out


# ------------------------------------------------------------------ learned combinations
def fit_select(Xd, yd, gd, kind="lr"):
    """Pick hyperparameter by GroupKFold AUC on dev; return fitted model and cv auc."""
    grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0] if kind == "lr" else [(1, 100), (2, 100), (2, 250), (3, 150)] if kind == "gb" else [2, 3, 4]
    gkf = GroupKFold(5)
    best = None
    for h in grid:
        aucs = []
        for tr, va in gkf.split(Xd, yd, gd):
            m = make_model(kind, h); m.fit(Xd[tr], yd[tr])
            aucs.append(B.plain_auc(m.predict_proba(Xd[va])[:, 1], yd[va]))
        a = float(np.mean(aucs))
        if best is None or a > best[0]:
            best = (a, h)
    m = make_model(kind, best[1]); m.fit(Xd, yd)
    return m, best


def make_model(kind, h):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=h, max_iter=3000))
    if kind == "gb":
        return GradientBoostingClassifier(max_depth=h[0], n_estimators=h[1], learning_rate=0.05, subsample=0.8, random_state=0)
    return DecisionTreeClassifier(max_depth=h, min_samples_leaf=20, random_state=0)


def featmat(us, cols, src):
    X = np.array([[src(u, c) for c in cols] for u in us], float)
    return X


def fsrc(u, c):
    if c.startswith("code:"):
        return u["cf"][c[5:]]
    v = bank[u["uid"]].get(c)
    return float("nan") if v is None else float(v)


def learned(name, cols, us=None, kind="lr", src=fsrc, store=None):
    us = us or U
    dev = [u for u in us if u["split"] == "dev"]; test = [u for u in us if u["split"] == "test"]
    Xd, Xt = featmat(dev, cols, src), featmat(test, cols, src)
    mu = np.nanmean(Xd, 0); mu = np.where(np.isnan(mu), 0, mu)
    Xd = np.where(np.isnan(Xd), mu, Xd); Xt = np.where(np.isnan(Xt), mu, Xt)
    yd = arr(dev, lambda u: u["label"]); gd = [u["conv"] for u in dev]
    m, (cv, h) = fit_select(Xd, yd, gd, kind)
    pt = m.predict_proba(Xt)[:, 1]
    sc = {u["uid"]: p for u, p in zip(test, pt)}
    res = {"cv_dev_auc": round(cv, 3), "hyper": str(h), "n_features": len(cols), "test": evaluate(test, sc)}
    if kind == "lr":
        coef = m[-1].coef_[0]
        res["top_coef"] = sorted([(c, round(float(w), 3)) for c, w in zip(cols, coef)], key=lambda x: -abs(x[1]))[:15]
    if kind == "tree":
        res["tree"] = export_text(m, feature_names=cols, max_depth=3)
    if store is not None:
        store[name] = {"model": m, "cols": cols, "mu": mu, "scores_test": sc,
                       "scores_dev_oof": oof(dev, Xd, yd, gd, kind, h)}
    print(f"{name:40s} cv={cv:.3f} test auc={res['test']['all']['auc']} len={res['test']['all']['auc_len']} inst={res['test']['instructed']['auc']}/{res['test']['instructed']['auc_len']}")
    return res


def oof(dev, Xd, yd, gd, kind, h):
    p = np.zeros(len(dev))
    for tr, va in GroupKFold(5).split(Xd, yd, gd):
        m = make_model(kind, h); m.fit(Xd[tr], yd[tr]); p[va] = m.predict_proba(Xd[va])[:, 1]
    return {u["uid"]: float(x) for u, x in zip(dev, p)}


MODELS = {}
CODE = ["code:" + c for c in B.CODE_COLS]
CODE_NOLEN = ["code:" + c for c in B.CODE_COLS if c not in B.LEN_COLS]
ATOM = list(Q.ATOMIC)
res_l = {}
res_l["M0_length_only"] = learned("M0_length_only", ["code:log_chars"], store=MODELS)
res_l["M1_code_lr"] = learned("M1_code_lr", CODE, store=MODELS)
res_l["M1b_code_nolen_lr"] = learned("M1b_code_nolen_lr", CODE_NOLEN, store=MODELS)
res_l["M1c_code_gb"] = learned("M1c_code_gb", CODE, kind="gb", store=MODELS)
res_l["M2_jev_atomic_lr"] = learned("M2_jev_atomic_lr", ATOM, store=MODELS)
res_l["M3_jev_bank_lr"] = learned("M3_jev_bank_lr", NUMQ, store=MODELS)
res_l["M3b_jev_bank_gb"] = learned("M3b_jev_bank_gb", NUMQ, kind="gb", store=MODELS)
res_l["M3c_jev_tree3"] = learned("M3c_jev_tree3", NUMQ, kind="tree", store=MODELS)
res_l["M4_code+jev_lr"] = learned("M4_code+jev_lr", CODE + NUMQ, store=MODELS)
res_l["M4b_code+jev_gb"] = learned("M4b_code+jev_gb", CODE + NUMQ, kind="gb", store=MODELS)
res_l["M4c_codenolen+jev_lr"] = learned("M4c_codenolen+jev_lr", CODE_NOLEN + NUMQ, store=MODELS)
res_l["M4d_len+jev_lr"] = learned("M4d_len+jev_lr", ["code:log_chars", "code:len_ratio_other", "code:sentences"] + NUMQ, store=MODELS)
R["learned"] = res_l

# a priori (not fitted) combos, evaluated on test
test = [u for u in U if u["split"] == "test"]; dev = [u for u in U if u["split"] == "dev"]


def a8_code_score(u):
    f = u["cf"]
    return -(-f["log_chars"] - 0.5 * f["sentences"] - 0.3 * min(3, f["n_excl"]) - 0.3 * min(3, f["n_emoji"]) - 0.5 * f["ends_q"] - 0.5 * f["llmish_hits"])


apriori = {
    "code_a8_formula": {u["uid"]: a8_code_score(u) for u in U},
    "jev_atomic_mean_signed": {u["uid"]: float(np.nanmean([Q.SIGN[k] * bank[u["uid"]][k] for k in ATOM])) for u in U},
    "jev_a8_gate_v2": {u["uid"]: bank[u["uid"]]["t_paraphrase"] + bank[u["uid"]]["t_overvalidation"] + bank[u["uid"]]["t_forced"] + bank[u["uid"]]["p_too_much_1"] + 0.5 * abs(bank[u["uid"]]["s_length"] - 2) for u in U},
    "jev_vice_max": {u["uid"]: max(bank[u["uid"]][v[0]] for v in Q.VICES.values()) for u in U},
    "jev_vmain_not_nothing": {u["uid"]: 1 - bank[u["uid"]]["v_main_p"]["nothing"] for u in U},
}
R["apriori"] = {k: evaluate(test, v) for k, v in apriori.items()}
for k, v in R["apriori"].items():
    print(f"{k:40s} test auc={v['all']['auc']} len={v['all']['auc_len']}")

# ------------------------------------------------------------------ per-question AUC (dev and test; oriented LLM-positive by SIGN)
rows = []
for q in NUMQ:
    sg = Q.SIGN.get(q, 1)
    r = {"q": q, "sign": sg}
    for sp, us in (("dev", dev), ("test", test)):
        lab = arr(us, lambda u: u["label"]); s = sg * arr(us, lambda u: bank[u["uid"]][q]); L = arr(us, lambda u: u["chars"])
        r[f"{sp}_auc"] = round(B.plain_auc(s, lab), 3); r[f"{sp}_auc_len"] = round(B.strat_auc(s, lab, L), 3)
        ins = [i for i, u in enumerate(us) if u["label"] == 0 or u["cond"] in B.INSTRUCTED]
        r[f"{sp}_auc_instructed_len"] = round(B.strat_auc(s[ins], lab[ins], L[ins]), 3)
        r[f"{sp}_mean_h"] = round(float(np.nanmean(s[lab == 0] * sg)), 3); r[f"{sp}_mean_llm"] = round(float(np.nanmean(s[lab == 1] * sg)), 3)
    rows.append(r)
with open(os.path.join(B.OUT, "b1_question_auc.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
R["question_auc_top"] = sorted(rows, key=lambda r: -r["dev_auc_len"])[:25]
R["question_auc_bottom"] = sorted(rows, key=lambda r: r["dev_auc_len"])[:10]

# ------------------------------------------------------------------ A9 paraphrase ensembles, A10 anchored vs noul, A8 counterfactual
def q_auc(us, fn):
    lab = arr(us, lambda u: u["label"]); s = arr(us, fn); L = arr(us, lambda u: u["chars"])
    return round(B.plain_auc(s, lab), 3), round(B.strat_auc(s, lab, L), 3)


ens = {}
for concept, lst in Q.PARA.items():
    ids = [lst[0]] + [f"p_{concept}_{j}" for j in range(1, len(lst))]
    singles = {i: q_auc(test, lambda u, i=i: bank[u["uid"]][i]) for i in ids}
    mean4 = q_auc(test, lambda u: float(np.mean([bank[u["uid"]][i] for i in ids])))
    dev_best = max(ids, key=lambda i: q_auc(dev, lambda u, i=i: bank[u["uid"]][i])[1])
    ens[concept] = {"singles_test": singles, "mean_of_4_test": mean4, "dev_best_single": dev_best, "dev_best_single_test": singles[dev_best],
                    "single_sd_across_wordings_len": round(float(np.std([v[1] for v in singles.values()])), 3)}
R["A9_paraphrase_ensemble"] = ens
anch = {}
for a, bl in [("s_friend_vs_assistant", ["t_assistant", "p_assistant_1", "h_ai", "t_formal"]), ("s_validation", ["t_validation_formula", "t_overvalidation"]),
              ("s_energy_vs_other", ["t_more_excited"]), ("s_length", ["cf_friend_shorter", "t_longer_than_other"]), ("s_moves", ["t_multi_acts", "p_too_much_1"]),
              ("s_polish", ["t_tidy_punct", "t_full_sentences"])]:
    anch[a] = {"score": q_auc(test, lambda u: bank[u["uid"]][a]), **{b: q_auc(test, lambda u, b=b: bank[u["uid"]][b]) for b in bl}}
R["A10_anchored_vs_noul"] = anch
R["A8_counterfactual"] = {k: q_auc(test, lambda u, k=k: Q.SIGN[k] * bank[u["uid"]][k]) for k in Q.COUNTERFACTUAL}
R["holistic"] = {k: q_auc(test, lambda u, k=k: Q.SIGN[k] * bank[u["uid"]][k]) for k in ["h_human", "h_ai", "cf_suspect_ai", "p_assistant_1"]}

# ------------------------------------------------------------------ A11 vice diagnosis vs code labels
def vice_gold(u):
    f = u["cf"]
    hl = HUMLEN.get(u["ckey"])
    return {
        "filler_question": f["gen_recip_q"], "any_end_question": f["ends_q"], "enthusiasm": int(f["n_excl"] > 0 or f["n_emoji"] > 0),
        "validation": f["validation_rx"], "echo": f["echo_any2"], "template": f["react_then_q"],
        "too_long": int(hl is not None and u["chars"] > 2 * max(hl, 10)), "forced_casual": f["slang_forced"], "perf_opener": f["perf_open"],
    }


VICE_Q = {"filler_question": ["t_ends_generic_q", "p_filler_q_1", "t_mirror_question"], "any_end_question": ["t_ends_q"],
          "enthusiasm": ["t_more_excited", "s_energy_vs_other", "t_emoji_decor"], "validation": ["t_validation_formula", "s_validation", "t_overvalidation"],
          "echo": ["t_paraphrase", "t_repeats_words", "p_echo_2"], "template": ["t_template", "t_multi_acts"], "too_long": ["t_keeps_going", "cf_friend_shorter", "s_length"],
          "forced_casual": ["t_forced", "t_trendy_slang"], "perf_opener": ["t_perf_opener"]}
vd = {}
llm_units = [u for u in U if u["label"] == 1]
for v, qs in VICE_Q.items():
    g = np.array([vice_gold(u)[v] for u in U]); gl = np.array([vice_gold(u)[v] for u in llm_units])
    vd[v] = {"prevalence_llm": round(float(gl.mean()), 3), "prevalence_human": round(float(np.mean([vice_gold(u)[v] for u in U if u["label"] == 0])), 3)}
    for q in qs:
        s = arr(U, lambda u: bank[u["uid"]][q]); sl = arr(llm_units, lambda u: bank[u["uid"]][q])
        thr = 0.5 if not q.startswith("s_") else 2.0
        pred = sl > thr
        tp = int((pred & (gl == 1)).sum()); fp = int((pred & (gl == 0)).sum()); fn = int((~pred & (gl == 1)).sum())
        vd[v][q] = {"auc_vs_code_all": round(B.auc(s[g == 1], s[g == 0]), 3), "auc_vs_code_llm_only": round(B.auc(sl[gl == 1], sl[gl == 0]), 3),
                    "precision@thr": round(tp / (tp + fp), 3) if tp + fp else None, "recall@thr": round(tp / (tp + fn), 3) if tp + fn else None}
R["A11_vice_vs_code"] = vd
# main-vice Choice vs code: when the reply has exactly one code-flagged vice among the Choice's options
MAP = {"filler_question": "filler_question", "enthusiasm": "enthusiasm", "validation": "validation", "echo": "echo", "template": "template",
       "too_long": "too_long", "forced_casual": "forced_casual"}
hits = tot = 0; top2 = 0; conf = Counter()
for u in llm_units:
    gv = [MAP[k] for k, x in vice_gold(u).items() if x and k in MAP]
    if len(gv) == 1:
        p = bank[u["uid"]]["v_main_p"]; order = sorted(p, key=lambda k: -p[k])
        tot += 1; hits += order[0] == gv[0]; top2 += gv[0] in order[:2]; conf[(gv[0], order[0])] += 1
R["A11_main_vice_choice"] = {"n_single_vice": tot, "top1_acc": round(hits / tot, 3) if tot else None, "top2_acc": round(top2 / tot, 3) if tot else None,
                             "chance_top1": round(1 / 9, 3), "confusions_top": [[a, b, n] for (a, b), n in conf.most_common(12)],
                             "vmain_dist_llm": Counter(bank[u["uid"]]["v_main"] for u in llm_units).most_common(),
                             "vmain_dist_human": Counter(bank[u["uid"]]["v_main"] for u in U if u["label"] == 0).most_common()}

json.dump(R, open(os.path.join(B.SCR, "eval_bank_partial.json"), "w"), default=str, indent=1)
import pickle
pickle.dump({"MODELS": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in MODELS.items()}}, open(os.path.join(B.SCR, "models_scores.pkl"), "wb"))
pickle.dump({k: v["model"] for k, v in MODELS.items()}, open(os.path.join(B.SCR, "models.pkl"), "wb"))
print("bank part done")
