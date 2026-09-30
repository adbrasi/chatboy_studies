# b1_eval part 2 (exec'd inside b1_eval.py namespace): state variants, cascade, chaining, pairwise, sentence-level,
# autoresearch, lean bank, costs. Everything: fit/select on dev, report on test.
import random as _rnd

core = [u for u in B.core_units(ctxs, U)]
CORE_Q = Q.CORE


def lr_on(name, us, getter, cols, kind="lr"):
    """Fit on dev units of `us`, report test; getter(u, col) -> value."""
    return learned(name, cols, us=us, kind=kind, src=getter)


def paired_delta(us, fa, fb, len_ctrl=True, n=400):
    """AUC(fa) - AUC(fb) on the same units, with cluster-bootstrap CI."""
    lab = arr(us, lambda u: u["label"]); a = arr(us, fa); b = arr(us, fb); L = arr(us, lambda u: u["chars"])
    cl = np.array([u["conv"] for u in us])
    f = (lambda i: B.strat_auc(a[i], lab[i], L[i]) - B.strat_auc(b[i], lab[i], L[i])) if len_ctrl else (lambda i: B.plain_auc(a[i], lab[i]) - B.plain_auc(b[i], lab[i]))
    d = f(np.arange(len(us)))
    return round(float(d), 3), B.cluster_boot(f, cl, n)


# ------------------------------------------------------------------ A2 / A3a / A3b : state variants with the same CORE questions
VAR = {}
for name in ("rubric", "fewshot", "realshot"):
    V = J(f"var_{name}.json")
    us = [u for u in core if u["uid"] in V]
    if not us:
        continue
    tus = [u for u in us if u["split"] == "test"]
    per_q = {}
    for q in CORE_Q:
        sg = Q.SIGN[q]
        per_q[q] = {"bank": q_auc(tus, lambda u, q=q: sg * bank[u["uid"]][q]), "variant": q_auc(tus, lambda u, q=q: sg * V[u["uid"]][q]),
                    "delta_len": paired_delta(tus, lambda u, q=q: sg * V[u["uid"]][q], lambda u, q=q: sg * bank[u["uid"]][q])}
    ins = [u for u in tus if u["label"] == 0 or u["cond"] in B.INSTRUCTED]
    per_q_ins = {q: {"bank": q_auc(ins, lambda u, q=q: Q.SIGN[q] * bank[u["uid"]][q]), "variant": q_auc(ins, lambda u, q=q: Q.SIGN[q] * V[u["uid"]][q])} for q in ["h_ai", "h_human", "s_friend_vs_assistant", "cf_suspect_ai", "t_assistant", "t_validation_formula", "t_more_excited"]}
    VAR[name] = {"n_units": len(us), "per_q_test": per_q, "per_q_test_instructed": per_q_ins,
                 "lr_core_variant": lr_on(f"{name}_coreLR", us, lambda u, c, V=V: V[u["uid"]][c], CORE_Q),
                 "lr_core_bank_same_units": lr_on(f"{name}_bankcoreLR", us, lambda u, c: bank[u["uid"]][c], CORE_Q),
                 "lr_code+variant": lr_on(f"{name}_code+coreLR", us, lambda u, c, V=V: u["cf"][c[5:]] if c.startswith("code:") else V[u["uid"]][c], CODE + CORE_Q),
                 "lr_code+bankcore": lr_on(f"{name}_code+bankcoreLR", us, lambda u, c: u["cf"][c[5:]] if c.startswith("code:") else bank[u["uid"]][c], CODE + CORE_Q)}
R["A2_A3_state_variants"] = VAR

# ------------------------------------------------------------------ A4 speaker reference
from b1_variants import REL_SIGN
V = J("var_speaker.json")
us = [u for u in core if u["uid"] in V]
if us:
    tus = [u for u in us if u["split"] == "test"]
    RELQ = list(REL_SIGN)
    R["A4_speaker"] = {"n_units": len(us),
                       "per_q_test": {q: q_auc(tus, lambda u, q=q: REL_SIGN[q] * V[u["uid"]][q]) for q in RELQ},
                       "bank_same_q_test": {q: q_auc(tus, lambda u, q=q: Q.SIGN[q] * bank[u["uid"]][q]) for q in ["t_forced", "t_formal", "t_more_excited", "h_ai"]},
                       "lr_rel": lr_on("speaker_relLR", us, lambda u, c: V[u["uid"]][c], RELQ),
                       "lr_bankcore_same_units": lr_on("speaker_bankcoreLR", us, lambda u, c: bank[u["uid"]][c], CORE_Q),
                       "lr_code+bank": lr_on("speaker_code+bankLR", us, lambda u, c: u["cf"][c[5:]] if c.startswith("code:") else bank[u["uid"]][c], CODE + NUMQ),
                       "lr_code+bank+rel": lr_on("speaker_code+bank+relLR", us, lambda u, c: u["cf"][c[5:]] if c.startswith("code:") else (V[u["uid"]][c] if c in V[u["uid"]] else bank[u["uid"]][c]), CODE + NUMQ + ["r_" + x[2:] if False else x for x in RELQ if x.startswith("r_")])}

# ------------------------------------------------------------------ A5 cascade
V = J("var_cascade.json"); M = J("var_moment.json")
us = [u for u in core if u["uid"] in V]
if us:
    tus = [u for u in us if u["split"] == "test"]
    cas = lambda u: float(np.mean([V[u["uid"]][f"m{j}"] for j in range(5)]))
    get = lambda u, c: {"mean5": cas(u), "m_ai": V[u["uid"]]["m_ai_pattern"], "m_fits": V[u["uid"]]["m_fits"],
                        "conf": M[u["ckey"]]["moment_conf"]}.get(c)
    by_m = {}
    for m in sorted({M[u["ckey"]]["moment"] for u in tus}):
        mu_ = [u for u in tus if M[u["ckey"]]["moment"] == m]
        if sum(u["label"] == 0 for u in mu_) >= 5:
            by_m[m] = {"n": len(mu_), "cascade_mean5": q_auc(mu_, cas), "bank_atomic_mean": q_auc(mu_, lambda u: apriori["jev_atomic_mean_signed"][u["uid"]])}
    R["A5_cascade"] = {"n_units": len(us), "moment_dist": Counter(M[k]["moment"] for k in {u["ckey"] for u in us}).most_common(),
                       "moment_conf_mean": round(float(np.mean([M[k]["moment_conf"] for k in {u["ckey"] for u in us}])), 3),
                       "test_mean5": q_auc(tus, cas), "test_m_ai": q_auc(tus, lambda u: V[u["uid"]]["m_ai_pattern"]),
                       "test_m_fits_neg": q_auc(tus, lambda u: -V[u["uid"]]["m_fits"]),
                       "bank_atomic_mean_same_units": q_auc(tus, lambda u: apriori["jev_atomic_mean_signed"][u["uid"]]),
                       "by_moment": by_m,
                       "lr_cascade": lr_on("cascade_LR", us, get, ["mean5", "m_ai", "m_fits"]),
                       "lr_code+bank": lr_on("cascade_code+bankLR", us, lambda u, c: u["cf"][c[5:]] if c.startswith("code:") else bank[u["uid"]][c], CODE + NUMQ),
                       "lr_code+bank+cascade": lr_on("cascade_code+bank+casLR", us, lambda u, c: u["cf"][c[5:]] if c.startswith("code:") else (get(u, c) if c in ("mean5", "m_ai", "m_fits") else bank[u["uid"]][c]), CODE + NUMQ + ["mean5", "m_ai", "m_fits"])}

# ------------------------------------------------------------------ A6 chaining
from b1_variants import CHAIN_TRAITS
CH = {}
for name in ("chain", "chain_nocode"):
    V = J(f"var_{name}.json")
    us = [u for u in core if u["uid"] in V]
    if not us:
        continue
    tus = [u for u in us if u["split"] == "test"]
    cols = CHAIN_TRAITS + (["code:log_chars", "code:len_ratio_other", "code:sentences", "code:ends_q", "code:n_excl", "code:n_emoji", "code:llmish_hits"] if name == "chain" else [])
    CH[name] = {"n_units": len(us), "c_ai": q_auc(tus, lambda u, V=V: V[u["uid"]]["c_ai"]), "c_artificial": q_auc(tus, lambda u, V=V: V[u["uid"]]["c_artificial"]),
                "c_main_not_nothing": q_auc(tus, lambda u, V=V: 1 - V[u["uid"]]["c_main_p"]["nothing"]),
                "same_inputs_mean_readings": q_auc(tus, lambda u: float(np.mean([bank[u["uid"]][t] * Q.SIGN[t] for t in CHAIN_TRAITS]))),
                "same_inputs_LR_in_code": lr_on(f"{name}_sameinputsLR", us, lambda u, c: u["cf"][c[5:]] if c.startswith("code:") else bank[u["uid"]][c], cols),
                "h_ai_stage1_same_units": q_auc(tus, lambda u: bank[u["uid"]]["h_ai"])}
R["A6_chaining"] = CH

# ------------------------------------------------------------------ A7 pairwise decomposed
P = J("pairs.json")
if P:
    from b1_pairs import PW
    TR = list(PW)
    agg = defaultdict(lambda: defaultdict(list))
    posA = defaultdict(list)
    for pid, d in P.items():
        key = d["llm"]
        llm_is = "B" if d["order"] == "HL" else "A"; hum_is = "A" if llm_is == "B" else "B"
        for q in TR + ["pw_real"]:
            p = d[q]
            agg[key][q].append((p.get(llm_is, 0) - p.get(hum_is, 0)) if q != "pw_real" else p.get(hum_is, 0))
            posA[q].append(p.get("A", 0))
        agg[key]["_h"] = d["human"]
    rows = []
    for lu, dd in agg.items():
        if len(dd["pw_real"]) != 2 or lu not in BYUID:
            continue
        u = BYUID[lu]
        rows.append({"llm": lu, "human": dd["_h"], "split": u["split"], "conv": u["conv"], "cond": u["cond"],
                     **{q: float(np.mean(dd[q])) for q in TR}, "p_real_human": float(np.mean(dd["pw_real"])),
                     "real_orders_agree": int((dd["pw_real"][0] > 0.5) == (dd["pw_real"][1] > 0.5))})
    trd = [r for r in rows if r["split"] == "dev"]; te = [r for r in rows if r["split"] == "test"]
    Xd = np.array([[r[q] for q in TR] for r in trd]); Xt = np.array([[r[q] for q in TR] for r in te])
    Xs = np.vstack([Xd, -Xd]); ys = np.r_[np.ones(len(Xd)), np.zeros(len(Xd))]
    best = None
    for Cc in [0.01, 0.1, 1.0]:
        gk = GroupKFold(5); accs = []
        g = np.array([r["conv"] for r in trd])
        for tr, va in gk.split(Xd, groups=g):
            m = LogisticRegression(C=Cc, fit_intercept=False, max_iter=2000).fit(np.vstack([Xd[tr], -Xd[tr]]), np.r_[np.ones(len(tr)), np.zeros(len(tr))])
            accs.append(float(np.mean(Xd[va] @ m.coef_[0] > 0)))
        if best is None or np.mean(accs) > best[0]:
            best = (float(np.mean(accs)), Cc)
    mpw = LogisticRegression(C=best[1], fit_intercept=False, max_iter=2000).fit(Xs, ys)
    sign_prior = np.array([1, 0, 1, 1, 1, 1, 1, 1, -1])  # a-priori directions (generic excluded: inverted in a8)

    def pacc(rs, fn):
        v = np.array([fn(r) for r in rs], float)
        cl = np.array([r["conv"] for r in rs])
        return round(float(np.mean(v)), 3), B.cluster_boot(lambda i: float(np.mean(v[i])), cl, 400)

    MS = pickle.load(open(os.path.join(B.SCR, "models_scores.pkl"), "rb"))["MODELS"]
    def model_pair(mname):
        sc = MS[mname]["scores_test"]
        return lambda r: float(sc.get(r["llm"], np.nan) > sc.get(r["human"], np.nan))
    R["A7_pairwise"] = {
        "n_pairs_test": len(te), "n_pairs_dev": len(trd),
        "holistic_choice_acc": pacc(te, lambda r: r["p_real_human"] > 0.5),
        "holistic_orders_agree": round(float(np.mean([r["real_orders_agree"] for r in te])), 3),
        "position_bias_pct_A": {q: round(float(np.mean(v)), 3) for q, v in posA.items()},
        "decomposed_apriori_acc": pacc(te, lambda r: float(np.array([r[q] for q in TR]) @ sign_prior > 0)),
        "decomposed_LR_acc": pacc(te, lambda r: float(np.array([r[q] for q in TR]) @ mpw.coef_[0] > 0)), "decomposed_LR_cv_dev": round(best[0], 3),
        "decomposed_LR_coef": dict(zip(TR, np.round(mpw.coef_[0], 2).tolist())),
        "per_trait_acc": {q: pacc(te, lambda r, q=q: (r[q] * (-1 if q == "pw_friend" else 1)) > 0)[0] for q in TR},
        "bank_code+jev_pair_acc": pacc(te, model_pair("M4_code+jev_lr")), "code_only_pair_acc": pacc(te, model_pair("M1_code_lr")),
        "jev_bank_only_pair_acc": pacc(te, model_pair("M3_jev_bank_lr")),
        "by_cond": {cn: {"holistic": pacc([r for r in te if r["cond"] == cn], lambda r: r["p_real_human"] > 0.5)[0],
                         "decomposed_LR": pacc([r for r in te if r["cond"] == cn], lambda r: float(np.array([r[q] for q in TR]) @ mpw.coef_[0] > 0))[0],
                         "code+jev_bank": pacc([r for r in te if r["cond"] == cn], model_pair("M4_code+jev_lr"))[0],
                         "code_only": pacc([r for r in te if r["cond"] == cn], model_pair("M1_code_lr"))[0]} for cn in sorted({r["cond"] for r in te})},
    }

# ------------------------------------------------------------------ A12 sentence level + A11 stage 3 (culprit) + cut experiment
S = J("sent.json")
if S:
    us = [u for u in core if u["uid"] in S]
    tus = [u for u in us if u["split"] == "test"]
    def smin(u):
        d = S[u["uid"]]; n = len(d["sents"])
        return 1 - min(d[f"friend_S{i + 1}"] for i in range(n))
    def smeanneed(u):
        d = S[u["uid"]]; n = len(d["sents"])
        return 1 - float(np.mean([d[f"needed_S{i + 1}"] for i in range(n)]))
    def nflag(u):
        d = S[u["uid"]]; n = len(d["sents"])
        return sum(d[f"friend_S{i + 1}"] < 0.5 for i in range(n))
    sent_res = {"n_units": len(us), "detect_min_friend": q_auc(tus, smin), "detect_mean_not_needed": q_auc(tus, smeanneed),
                "detect_n_flagged": q_auc(tus, nflag),
                "detect_cs_ai_not_none(>=2 sents)": q_auc([u for u in tus if "cs_ai_p" in S[u["uid"]]], lambda u: 1 - S[u["uid"]]["cs_ai_p"]["none"]),
                "lr_sent+code": lr_on("sent_code+sentLR", us, lambda u, c: u["cf"][c[5:]] if c.startswith("code:") else {"smin": smin(u), "sneed": smeanneed(u), "nflag": nflag(u)}[c], CODE + ["smin", "sneed", "nflag"])}
    # culprit accuracy vs code gold
    def gold_sent(sents, rx):
        idx = [i for i, s in enumerate(sents) if rx(s)]
        return idx
    qrx = lambda s: s.rstrip().endswith("?")
    vrx = lambda s: bool(B.VALID.search(s)) or any(r.search(s) for k, r in C8.LLMISH.items() if k in ("absolutely", "totally", "definitely", "that's so sweet / aww", "so proud of you", "i'm here for you / here if you need", "i'm (so) sorry to hear / sorry you're"))
    cul = {}
    for nm, q, rx in (("question", "cs_filler_q", qrx), ("validation", "cs_validation", vrx)):
        one = both_none = picked_none_when_absent = absent = 0; hit = 0
        for u in us:
            d = S[u["uid"]]
            if "cs_filler_q" not in d:
                continue
            g = gold_sent(d["sents"], rx)
            ch = d[q]
            if len(g) == 1:
                one += 1; hit += ch == f"S{g[0] + 1}"
            elif len(g) == 0:
                absent += 1; picked_none_when_absent += ch == "none"
        cul[nm] = {"n_with_exactly_one_gold": one, "acc_pick_gold_sentence": round(hit / one, 3) if one else None,
                   "n_without_gold": absent, "says_none_when_absent": round(picked_none_when_absent / absent, 3) if absent else None}
    # which sentence is cut: agreement with 'last sentence' rule
    multi = [u for u in us if "cs_cut" in S[u["uid"]] and u["label"] == 1]
    cul["cs_cut_is_last_sentence"] = round(float(np.mean([S[u["uid"]]["cs_cut"] == f"S{len(S[u['uid']]['sents'])}" for u in multi])), 3) if multi else None
    cul["cs_cut_is_first_sentence"] = round(float(np.mean([S[u["uid"]]["cs_cut"] == "S1" for u in multi])), 3) if multi else None
    sent_res["culprit_vs_code"] = cul
    # cut experiment (LLM replies with >=2 sentences, TEST): remove one sentence by different rules, re-score with the CODE-only detector
    import pickle as _pk
    MM = _pk.load(open(os.path.join(B.SCR, "models.pkl"), "rb")) if os.path.exists(os.path.join(B.SCR, "models.pkl")) else None
    code_model = MODELS["M1_code_lr"]["model"]; mu1 = MODELS["M1_code_lr"]["mu"]
    def code_p(text, u):
        f = B.code_feats(text, B.last_other(ctxs[u["ckey"]]))
        x = np.array([[f[c] for c in B.CODE_COLS]], float)
        return float(code_model.predict_proba(x)[0, 1])
    rng = _rnd.Random(0)
    cuts = defaultdict(list); lenerr = defaultdict(list)
    tmulti = [u for u in multi if u["split"] == "test"]
    for u in tmulti:
        d = S[u["uid"]]; ss = d["sents"]; n = len(ss)
        hl = HUMLEN.get(u["ckey"], 30)
        amin = int(np.argmin([d[f"friend_S{i + 1}"] for i in range(n)]))
        ai = d["cs_ai"] if d["cs_ai"] != "none" else d["cs_cut"]
        rules = {"none": None, "jev_cs_cut": int(d["cs_cut"][1:]) - 1, "jev_cs_ai": int(ai[1:]) - 1, "jev_min_friend": amin, "last": n - 1, "first": 0, "random": rng.randrange(n)}
        for rname, k in rules.items():
            t = " ".join(s for i, s in enumerate(ss) if i != k) if k is not None else " ".join(ss)
            cuts[rname].append(code_p(t, u)); lenerr[rname].append(abs(math.log2((len(t) + 5) / (hl + 5))))
    sent_res["cut_experiment_test"] = {"n": len(tmulti), **{r: {"code_detector_p_llm": round(float(np.mean(v)), 3), "len_err_log2": round(float(np.mean(lenerr[r])), 3)} for r, v in cuts.items()}}
    R["A12_sentence"] = sent_res

# ------------------------------------------------------------------ A13 autoresearch
AUTO = {}
for rnd in (1, 2):
    A = J(f"auto_a{rnd}.json")
    if not A:
        continue
    prevA = {}
    for r2 in range(1, rnd + 1):
        prevA.update({uid: {**prevA.get(uid, {}), **v} for uid, v in J(f"auto_a{r2}.json").items()})
    us = [u for u in U if u["uid"] in prevA]
    acols = sorted({k for v in prevA.values() for k in v})
    getA = lambda u, c: u["cf"][c[5:]] if c.startswith("code:") else (prevA[u["uid"]].get(c, np.nan) if c in acols else bank[u["uid"]][c])
    tus = [u for u in us if u["split"] == "test"]; dus = [u for u in us if u["split"] == "dev"]
    AUTO[f"round{rnd}"] = {"n_new_q": len(acols), "per_q_dev_len": sorted([(c, q_auc(dus, lambda u, c=c: prevA[u["uid"]].get(c, np.nan))[1]) for c in acols], key=lambda x: -abs(x[1] - 0.5))[:15],
                           "lr_code+bank+auto": lr_on(f"auto{rnd}_code+bank+autoLR", us, getA, CODE + NUMQ + acols),
                           "lr_code+bank_same_units": lr_on(f"auto{rnd}_code+bankLR", us, getA, CODE + NUMQ),
                           "lr_auto_only": lr_on(f"auto{rnd}_autoLR", us, getA, acols)}
R["A13_autoresearch"] = AUTO

# ------------------------------------------------------------------ lean bank: L1 selection on dev (cost matters: price ~ number of questions)
dev_u = [u for u in U if u["split"] == "dev"]
Xd = featmat(dev_u, CODE + NUMQ, fsrc); mu = np.nanmean(Xd, 0); Xd = np.where(np.isnan(Xd), mu, Xd)
yd = arr(dev_u, lambda u: u["label"])
lean = {}
for Cc in [0.02, 0.05, 0.1]:
    m = make_pipeline(StandardScaler(), LogisticRegression(C=Cc, penalty="l1", solver="liblinear", max_iter=3000)).fit(Xd, yd)
    sel = [c for c, w in zip(CODE + NUMQ, m[-1].coef_[0]) if abs(w) > 1e-6 and not c.startswith("code:")]
    lean[str(Cc)] = {"n_jev_q": len(sel), "jev_q": sel, "res": learned(f"lean_{Cc}", CODE + sel)}
R["lean_bank"] = lean

# ------------------------------------------------------------------ costs / latency per architecture
costs = defaultdict(lambda: {"calls": 0, "cost": 0.0, "tok": 0, "items": 0, "lat": [], "nq": 0})
for l in open(os.path.join(B.SCR, "costs.jsonl")):
    d = json.loads(l)
    c = costs[d["arch"]]
    c["calls"] += d["new_calls"]; c["cost"] += d["cost"]; c["tok"] += d["input_tokens"]; c["items"] += d["n_items"]; c["nq"] = max(c["nq"], d["n_questions"])
    if d["lat_p50"]:
        c["lat"].append(d["lat_p50"])
R["costs"] = {k: {"calls": v["calls"], "cost_usd": round(v["cost"], 4), "usd_per_call": round(v["cost"] / v["calls"], 6) if v["calls"] else None,
                  "tokens_per_call": round(v["tok"] / v["calls"]) if v["calls"] else None, "lat_p50_s": round(float(np.median(v["lat"])), 3) if v["lat"] else None,
                  "n_questions_max": v["nq"]} for k, v in costs.items()}
R["costs_total"] = {"jev_calls": sum(v["calls"] for v in costs.values()), "jev_usd": round(sum(v["cost"] for v in costs.values()), 3)}

B.save("b1_results.json", R)
