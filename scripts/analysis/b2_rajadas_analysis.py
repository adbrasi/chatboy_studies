"""b2 Parte A: análise da leitura do Jev sobre as rajadas (lift de cada estado, forma por estado, reação do outro,
detector "defensivo / pego no flagra" em 4 variantes + checagem com 2 LLMs anotadoras, exemplos).
Saída: analysis/data/b2_rajadas.json, b2_rajadas_examples.json"""
import os, json
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from b2_common import OUT, SCR, dump
import b2_llm

R = pd.read_csv(os.path.join(OUT, "b2_rajadas_items.csv.gz"))
F = pd.read_pickle(os.path.join(SCR, "b2_turns_feat.pkl")).set_index(["conv_id", "turn_idx"])
for c in ("max60", "n_msgs", "bub_med", "ends_punct", "gap_med", "del_any", "cps", "abandoned", "total_chars", "texts",
          "response_latency_s", "next_partner_latency", "excl_bub", "q_bub", "laugh_bub", "caps_bub", "selfcorr", "idle_med"):
    if c in F.columns:
        R[c] = [F.loc[(a, b), c] for a, b in zip(R.conv_id, R.turn_idx)]
R["R3"] = (R.max60 >= 3).astype(int)
R["R5"] = (R.max60 >= 5).astype(int)
R["lc"] = np.log1p(R.total_chars.astype(float))
# detector variants
R["def_single"] = (R.p_defends_self >= .5).astype(int)
R["def_composite"] = ((R.p_partner_challenges >= .5) & ((R.p_defends_self >= .5) | (R.p_denies >= .5))).astype(int)
R["def_caught"] = (R.p_caught_out >= .5).astype(int)
R["defense_type"] = R.defense_type.fillna("not_defensive(no cascade)")
R["def_cascade_serious"] = R.defense_type.isin(["caught_in_lie", "caught_in_mistake_or_forgetting", "criticized_for_behavior"]).astype(int)
R["def_cascade_playful"] = (R.defense_type == "playful_accusation").astype(int)
R["def_cascade_any"] = R.defense_type.isin(["caught_in_lie", "caught_in_mistake_or_forgetting", "criticized_for_behavior", "playful_accusation"]).astype(int)

LABELS = [c for c in R.columns if c.startswith("p_")] + ["def_single", "def_composite", "def_caught", "def_cascade_serious", "def_cascade_playful", "def_cascade_any"]


def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w)) if np.sum(w) > 0 else np.nan


def lift_table(d, target):
    base = wmean(d[target], d.w)
    out = {}
    for lab in LABELS:
        s = (d[lab] >= .5).astype(int)
        if s.sum() < 5:
            continue
        pr = wmean(s, d.w)
        p_t = wmean(d[target][s == 1], d.w[s == 1])
        # length-controlled OR (weighted GLM)
        try:
            X = sm.add_constant(pd.DataFrame({"s": s, "lc": d.lc}))
            m = sm.GLM(d[target], X, family=sm.families.Binomial(), var_weights=d.w / d.w.mean()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.conv_id)[0]})
            orc, pv = float(np.exp(m.params["s"])), float(m.pvalues["s"])
            lo, hi = np.exp(m.conf_int().loc["s"]).tolist()
        except Exception:
            orc, pv, lo, hi = np.nan, np.nan, np.nan, np.nan
        # bootstrap lift CI by conversation
        convs = d.conv_id.unique(); rng = np.random.default_rng(0); L = []
        gi = {c: np.where(d.conv_id.values == c)[0] for c in convs}
        for _ in range(300):
            ii = np.concatenate([gi[c] for c in rng.choice(convs, len(convs))])
            dd, ss = d.iloc[ii], s.iloc[ii]
            b = wmean(dd[target], dd.w)
            if ss.sum() and b > 0:
                L.append(wmean(dd[target][ss == 1], dd.w[ss == 1]) / b)
        out[lab] = {"prev_all_pct": round(100 * pr, 1), "n_state": int(s.sum()), f"P({target}|state)_pct": round(100 * p_t, 2),
                    "lift": round(p_t / base, 2), "lift_ci": [round(float(np.percentile(L, 2.5)), 2), round(float(np.percentile(L, 97.5)), 2)] if L else None,
                    "OR_len_ctrl": round(orc, 2), "OR_ci": [round(lo, 2), round(hi, 2)], "p_len_ctrl": pv}
    return base, out


res = {}
for c, d in R.groupby("corpus"):
    d = d.reset_index(drop=True)
    r = {"n_items": len(d), "by_stratum": d.stratum.value_counts().to_dict()}
    for tgt in ("R3", "R5"):
        base, tab = lift_table(d, tgt)
        r[f"base_{tgt}_pct"] = round(100 * base, 2)
        r[f"lift_{tgt}"] = dict(sorted(tab.items(), key=lambda kv: -kv[1]["lift"]))
    # main context distribution by stratum (weighted within stratum = plain)
    r["main_context_by_stratum"] = {s: (x.main_context.value_counts(normalize=True).round(3) * 100).to_dict() for s, x in d.groupby("stratum")}
    r["main_context_population"] = {k: round(100 * wmean((d.main_context == k).astype(int), d.w), 1) for k in d.main_context.unique()}
    r["defense_type_by_stratum"] = {s: (x.defense_type.value_counts(normalize=True).round(3) * 100).to_dict() for s, x in d.groupby("stratum")}
    # shape of rajadas per main context (R3 items)
    rj = d[d.R3 == 1]
    shape = {}
    for k, x in list(rj.groupby("main_context")) + [("ALL_R3", rj)] + [("DEFENSIVE(composite)", rj[rj.def_composite == 1]), ("DEFENSIVE(cascade serious)", rj[rj.def_cascade_serious == 1]),
                                                                        ("EXCITED(p_excited_positive)", rj[rj.p_excited_positive >= .5]), ("ANXIOUS", rj[rj.p_anxiety_insecurity >= .5]),
                                                                        ("ANGRY", rj[rj.p_anger >= .5]), ("STORY", rj[rj.p_storytelling >= .5]), ("GOSSIP", rj[rj.p_gossip_third_party >= .5]),
                                                                        ("BANTER", rj[rj.p_banter_joking >= .5])]:
        if len(x) < 8:
            continue
        s = {"n": int(len(x)), "bubbles_mean": round(float(x.max60.mean()), 2), "bub_chars_med": float(x.bub_med.median()),
             "total_chars_med": float(x.total_chars.median()), "ends_punct_pct": round(100 * x.ends_punct.mean(), 1),
             "excl_pct": round(100 * x.excl_bub.mean(), 1), "q_pct": round(100 * x.q_bub.mean(), 1), "arousal": round(float(x.arousal.mean()), 2)}
        if c == "maichat":
            s |= {"gap_med_s": round(float(x.gap_med.median()), 1), "idle_med_s": round(float(x.idle_med.median()), 1), "del_any_pct": round(100 * float(x.del_any.mean()), 1),
                  "cps_med": round(float(x.cps.median()), 2), "abandoned_pct": round(100 * float(x.abandoned.mean()), 1)}
        s["partner_next_latency_med"] = float(x.next_partner_latency.median())
        shape[k] = s
    r["shape_by_state"] = shape
    # partner reaction: R3 vs control (none, >40 chars)
    pr = d[d.partner_reaction.notna()]
    r["partner_reaction"] = {"R3": (pr[pr.R3 == 1].partner_reaction.value_counts(normalize=True).round(3) * 100).to_dict(),
                             "R5": (pr[pr.R5 == 1].partner_reaction.value_counts(normalize=True).round(3) * 100).to_dict(),
                             "control_no_rajada_>40chars": (pr[(pr.R3 == 0)].partner_reaction.value_counts(normalize=True).round(3) * 100).to_dict(),
                             "n": [int((pr.R3 == 1).sum()), int((pr.R5 == 1).sum()), int((pr.R3 == 0).sum())]}
    for nm, m in (("R3_defensive_composite", (pr.R3 == 1) & (pr.def_composite == 1)), ("R3_excited", (pr.R3 == 1) & (pr.p_excited_positive >= .5)),
                  ("R3_story", (pr.R3 == 1) & (pr.p_storytelling >= .5)), ("R3_anxious", (pr.R3 == 1) & (pr.p_anxiety_insecurity >= .5))):
        if m.sum() >= 8:
            r["partner_reaction"][nm] = (pr[m].partner_reaction.value_counts(normalize=True).round(3) * 100).to_dict() | {"n": int(m.sum())}
    res[c] = r

# detector agreement among variants
agree = {}
for a in ("def_single", "def_composite", "def_caught", "def_cascade_serious", "def_cascade_any"):
    for b in ("def_single", "def_composite", "def_caught", "def_cascade_serious", "def_cascade_any"):
        if a < b:
            agree[f"{a}~{b}"] = round(float(cohen_kappa_score(R[a], R[b])), 2)
res["detector_kappa"] = agree
res["detector_prevalence_raw_pct"] = {v: round(100 * R[v].mean(), 1) for v in ("def_single", "def_composite", "def_caught", "def_cascade_serious", "def_cascade_playful")}

# ---- LLM check of the defensive detector (2 annotators, stratified by Jev composite score) ----
R["def_score"] = R[["p_defends_self", "p_caught_out", "p_denies"]].max(1) * (0.5 + 0.5 * R.p_partner_challenges)
R["def_bin"] = pd.qcut(R.def_score.rank(method="first"), [0, .5, .8, .95, 1], labels=False)
chk = pd.concat([x.sample(min(len(x), 60), random_state=0) for _, x in R.groupby("def_bin")])
PROMPT = ("Chat excerpt (the other person's last turn, then the target turn).\n\nOTHER: {prev}\nTARGET: {text}\n\n"
          "Question: In TARGET, is the speaker defending or justifying themselves because the other person accused, criticized, "
          "confronted or caught them (for a lie, a mistake, something they did or forgot)? Count it even if light, but NOT if it is "
          "obviously a joke both are enjoying. Answer with exactly one word: YES or NO.")
items = [{"messages": [{"role": "user", "content": PROMPT.format(prev=r.partner_last, text=r.text)}], "model": m, "temperature": 0.0, "max_tokens": 1000, "effort": "low"}
         for m in ("openai/gpt-6-luna", "~deepseek/deepseek-flash-latest") for r in chk.itertuples()]
outs = b2_llm.chat_many(items, workers=4)
n = len(chk)
lab = lambda t: np.nan if not t else (1.0 if "YES" in t.upper()[:10] else 0.0 if "NO" in t.upper()[:10] else np.nan)
chk["llm_luna"] = [lab(t) for t in outs[:n]]
chk["llm_deepseek"] = [lab(t) for t in outs[n:]]
cc = chk.dropna(subset=["llm_luna", "llm_deepseek"])
cc_agree = cc[cc.llm_luna == cc.llm_deepseek]
val = {"n_checked": len(cc), "llm_llm_kappa": round(float(cohen_kappa_score(cc.llm_luna, cc.llm_deepseek)), 2), "n_llms_agree": len(cc_agree),
       "llm_yes_rate_agree_pct": round(100 * float(cc_agree.llm_luna.mean()), 1)}
for v in ("def_single", "def_composite", "def_caught", "def_cascade_serious", "def_cascade_any"):
    yv = cc_agree.llm_luna
    tp = ((cc_agree[v] == 1) & (yv == 1)).sum(); fp = ((cc_agree[v] == 1) & (yv == 0)).sum(); fn = ((cc_agree[v] == 0) & (yv == 1)).sum()
    val[v] = {"precision": round(tp / max(tp + fp, 1), 2), "recall": round(tp / max(tp + fn, 1), 2), "kappa_vs_llms": round(float(cohen_kappa_score(cc_agree[v], yv)), 2)}
for v in ("p_defends_self", "p_caught_out", "p_denies", "p_partner_challenges", "def_score"):
    val[v + "_auc"] = round(float(roc_auc_score(cc_agree.llm_luna, cc_agree[v])), 3)
val["note"] = "amostra estratificada pelo escore do Jev (sobre-representa os altos); 'verdade' = as 2 LLMs concordando"
res["defensive_llm_check"] = val
print(b2_llm.stats)

# ---- examples ----
ex = {}
def pick(mask, k=4, by="w"):
    x = R[mask].sort_values("max60", ascending=False).head(40)
    return [{"corpus": r.corpus, "bubbles_in_60s": int(r.max60), "context": r.main_context, "partner_before": r.partner_last,
             "rajada": json.loads(json.dumps(list(eval(r.texts)) if isinstance(r.texts, str) else list(r.texts))),
             "defense_type": r.defense_type, "partner_after": r.next, "partner_reaction": r.partner_reaction,
             "gap_med_s": None if pd.isna(r.gap_med) else round(float(r.gap_med), 1)} for r in x.sample(min(k, len(x)), random_state=1).itertuples()]
ex["defensive_serious_R3"] = pick((R.R3 == 1) & (R.def_cascade_serious == 1), 6)
ex["playful_accusation_R3"] = pick((R.R3 == 1) & (R.def_cascade_playful == 1), 3)
for ctx in ("excitement_news", "storytelling", "gossip_third_party", "anxiety_insecurity", "arguing_accusing", "banter_joking", "logistics_planning", "venting_complaining", "self_correction_clarifying", "apologizing"):
    ex[ctx + "_R5"] = pick((R.R5 == 1) & (R.main_context == ctx), 3) or pick((R.R3 == 1) & (R.main_context == ctx), 2)
dump(res, "b2_rajadas.json")
dump(ex, "b2_rajadas_examples.json")
print(json.dumps(res, indent=1, ensure_ascii=False, default=str)[:6000])
