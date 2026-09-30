"""b2 Parte A (código): frequência, distribuição e forma das rajadas (>=3 / >=5 bolhas em <60 s) e o que o outro faz depois.
Saída: analysis/data/b2_bursts_stats.json"""
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from b2_common import load, dump, boot_ci

T, M = load()
Mi = M.set_index(["conv_id", "idx"])
T = T.sort_values(["conv_id", "turn_idx"]).reset_index(drop=True)
g = T.groupby("conv_id")
same_next = g.session.shift(-1) == T.session
T["next_partner_nmsgs"] = g.n_msgs.shift(-1).where(same_next)
T["next_partner_max60"] = g.max60.shift(-1).where(same_next)
T["next_partner_q"] = g.has_q.shift(-1).where(same_next)
T["next_partner_laugh"] = g.laugh.shift(-1).where(same_next)
T["next_partner_texts"] = g.texts.shift(-1).where(same_next)
T["R3"] = T.max60 >= 3
T["R5"] = T.max60 >= 5
T["R3s"] = T.max60_strict >= 3
T["R5s"] = T.max60_strict >= 5
T["lenbin"] = pd.cut(T.total_chars, [0, 20, 40, 80, 160, 320, 1e9], labels=["1-20", "21-40", "41-80", "81-160", "161-320", ">320"])

# message-level features per turn
feat = {}
for r in T.itertuples():
    ms = [Mi.loc[(r.conv_id, i)] for i in r.msg_idxs]
    feat[r.Index] = {
        "bub_med": float(np.median([m.f_n_chars for m in ms])),
        "first_bub": float(ms[0].f_n_chars),
        "ends_punct": float(np.mean([bool(m.f_ends_punct) for m in ms])),
        "q_bub": float(np.mean([bool(m.f_has_q) for m in ms])),
        "excl_bub": float(np.mean([m.f_n_excl > 0 for m in ms])),
        "laugh_bub": float(np.mean([bool(m.f_laugh) for m in ms])),
        "caps_bub": float(np.mean([bool(m.f_caps_word) for m in ms])),
        "elong_bub": float(np.mean([bool(m.f_elongation) for m in ms])),
        "emoji_bub": float(np.mean([m.f_n_emoji > 0 for m in ms])),
        "selfcorr": float(any(bool(m.f_self_correction) for m in ms)),
        "del_any": float(np.mean([(m.ty_n_deletion_events or 0) > 0 for m in ms])) if r.corpus == "maichat" else np.nan,
        "cps": float(np.nanmedian([m.f_n_chars / m.ty_compose_s if (m.ty_compose_s and m.ty_compose_s > 0.3) else np.nan for m in ms])) if r.corpus == "maichat" else np.nan,
        "abandoned": float(np.mean([(m.ty_peak_len or 0) - m.f_n_chars >= 3 for m in ms])) if r.corpus == "maichat" else np.nan,
        "idle_med": float(np.nanmedian([m.ty_idle_before_typing_s for m in ms[1:]])) if (r.corpus == "maichat" and len(ms) > 1) else np.nan,
    }
F = pd.DataFrame.from_dict(feat, orient="index")
T = T.join(F)
T.to_pickle("/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad/b2/b2_turns_feat.pkl")

res = {}
for c, d in T.groupby("corpus"):
    r = {"n_turns": len(d), "n_msgs": int(d.n_msgs.sum())}
    for k in ("R3", "R5", "R3s", "R5s"):
        r[f"pct_turns_{k}"] = round(100 * d[k].mean(), 2)
        r[f"n_{k}"] = int(d[k].sum())
        r[f"pct_msgs_in_{k}"] = round(100 * d.loc[d[k], "n_msgs"].sum() / d.n_msgs.sum(), 2)
    r["dist_max60"] = d.max60.clip(upper=8).value_counts(normalize=True).sort_index().round(4).to_dict()
    # how many >=5 turns are "5 bubbles in <60s" vs longer turns
    r["R5_span_median_s"] = float(d.loc[d.R5, "burst_span_s"].median())
    # per speaker
    sp = d.groupby("spk").agg(n=("R3", "size"), r3=("R3", "mean"), r5=("R5", "mean"), n5=("R5", "sum"), multi=("n_msgs", lambda x: (x > 1).mean()))
    sp = sp[sp.n >= 30]
    r["speakers_ge30"] = len(sp)
    r["speaker_r3_p10_p50_p90"] = [round(100 * sp.r3.quantile(q), 1) for q in (.1, .5, .9)]
    r["speaker_r5_p10_p50_p90"] = [round(100 * sp.r5.quantile(q), 2) for q in (.1, .5, .9)]
    r["pct_speakers_with_any_R5"] = round(100 * (sp.n5 > 0).mean(), 1)
    tot5 = sp.n5.sum()
    r["share_R5_from_top10pct_speakers"] = round(100 * sp.n5.sort_values(ascending=False).head(max(1, len(sp) // 10)).sum() / max(tot5, 1), 1)
    r["rho_speaker_multi_rate_vs_r3"] = round(float(sp[["multi", "r3"]].corr("spearman").iloc[0, 1]), 3)
    # ICC-ish: share of variance of R3 between speakers (one-way ANOVA estimate)
    dd = d[d.spk.isin(sp.index)]
    grand = dd.R3.mean(); k = dd.groupby("spk").R3
    msb = (k.size() * (k.mean() - grand) ** 2).sum() / (k.ngroups - 1)
    msw = ((dd.R3 - k.transform("mean")) ** 2).sum() / (len(dd) - k.ngroups)
    n0 = k.size().mean()
    r["icc_R3_speaker"] = round(float((msb - msw) / (msb + (n0 - 1) * msw)), 3)
    # per conversation / session
    cv = d.groupby("conv_id").agg(n=("R3", "size"), r3=("R3", "mean"), n5=("R5", "sum"))
    r["conv_r3_p10_p50_p90"] = [round(100 * cv.r3.quantile(q), 1) for q in (.1, .5, .9)]
    r["pct_convs_with_any_R5"] = round(100 * (cv.n5 > 0).mean(), 1)
    ss = d.groupby(["conv_id", "session"]).agg(n=("R3", "size"), n5=("R5", "sum"), n3=("R3", "sum"))
    ss10 = ss[ss.n >= 10]
    r["sessions_ge10_turns"] = len(ss10)
    r["pct_sessions_ge10_with_R5"] = round(100 * (ss10.n5 > 0).mean(), 1)
    r["pct_sessions_ge10_with_R3"] = round(100 * (ss10.n3 > 0).mean(), 1)
    if c == "maichat":
        dur = d.groupby("conv_id").apply(lambda x: (pd.to_datetime(x.ts_end.max()) - pd.to_datetime(x.ts_start.min())).total_seconds() / 3600)
        r["R5_per_conv_hour"] = round(float(cv.n5.sum() / dur.sum()), 2)
        r["R3_per_conv_hour"] = round(float(d.R3.sum() / dur.sum()), 2)
        r["median_conv_minutes"] = round(float(dur.median() * 60), 1)
    else:
        days = d.assign(day=pd.to_datetime(d.ts_start.astype(str).str[:10], errors="coerce")).groupby("spk").day.nunique()
        r["R5_per_active_day_per_person"] = round(float(d.groupby("spk").R5.sum().sum() / days.sum()), 3)
        r["R3_per_active_day_per_person"] = round(float(d.groupby("spk").R3.sum().sum() / days.sum()), 3)
        r["pct_person_days_with_R5"] = round(100 * float(d.assign(day=pd.to_datetime(d.ts_start.astype(str).str[:10], errors="coerce")).groupby(["spk", "day"]).R5.any().mean()), 2)
    # by length
    r["by_len"] = {str(k): {"n": int(len(x)), "R3": round(100 * x.R3.mean(), 1), "R5": round(100 * x.R5.mean(), 1)} for k, x in d.groupby("lenbin", observed=True)}
    r["share_R5_with_total_gt160"] = round(100 * (d.loc[d.R5, "total_chars"] > 160).mean(), 1)
    # by position in session
    slen = d.groupby(["conv_id", "session"]).turn_in_session.transform("max") + 1
    pos = np.where(d.turn_in_session < 3, "first3", np.where(d.turn_in_session >= slen - 3, "last3", "middle"))
    r["by_session_pos"] = {p: {"n": int((pos == p).sum()), "R3": round(100 * d.R3[pos == p].mean(), 1), "R5": round(100 * d.R5[pos == p].mean(), 2)} for p in ("first3", "middle", "last3")}
    # by Jev D phase / intent / emotion (turns with D labels) — D saw bubbles joined by " / "
    dj = d[d.D_phase.notna()]
    for col in ("D_phase", "D_intent", "D_emotion", "D_relationship"):
        tab = dj.groupby(col).agg(n=("R3", "size"), R3=("R3", "mean"), R5=("R5", "mean"), chars=("total_chars", "median"))
        tab = tab[tab.n >= 15]
        r[f"by_{col}"] = {k: {"n": int(v.n), "R3": round(100 * v.R3, 1), "R5": round(100 * v.R5, 2), "lift_R3": round(v.R3 / dj.R3.mean(), 2), "med_chars": float(v.chars)} for k, v in tab.iterrows()}
    # length-controlled effect of D states on R3 (logit, cluster by conv)
    dj = dj.assign(y=dj.R3.astype(int), lc=np.log1p(dj.total_chars))
    eff = {}
    for v in ("D_anxious", "D_arousal", "D_tension", "D_playful", "D_flirting", "D_vulnerable", "D_seeks_support", "D_seriousness", "D_engagement", "D_valence"):
        try:
            m0 = smf.logit(f"y ~ {v}", dj).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(dj.conv_id)[0]})
            m1 = smf.logit(f"y ~ {v} + lc", dj).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(dj.conv_id)[0]})
            eff[v] = {"OR_raw": round(float(np.exp(m0.params[v])), 2), "p_raw": float(m0.pvalues[v]),
                      "OR_len_ctrl": round(float(np.exp(m1.params[v])), 2), "p_len_ctrl": float(m1.pvalues[v])}
        except Exception as e:
            eff[v] = str(e)[:80]
    r["D_effects_on_R3"] = eff
    # shape: R3 vs other multi-bubble turns vs singles
    grp = np.where(d.R3, "R3", np.where(d.n_msgs > 1, "multi_not_R3", "single"))
    shape = {}
    for gname in ("R3", "multi_not_R3", "single"):
        x = d[grp == gname]
        s = {"n": int(len(x)), "n_msgs_mean": round(x.n_msgs.mean(), 2), "total_chars_med": float(x.total_chars.median()),
             "bub_med": float(x.bub_med.median()), "first_bub_med": float(x.first_bub.median()),
             "ends_punct": round(100 * x.ends_punct.mean(), 1), "q_bub": round(100 * x.q_bub.mean(), 1),
             "excl_bub": round(100 * x.excl_bub.mean(), 1), "laugh_bub": round(100 * x.laugh_bub.mean(), 1),
             "caps_bub": round(100 * x.caps_bub.mean(), 1), "elong_bub": round(100 * x.elong_bub.mean(), 1),
             "emoji_bub": round(100 * x.emoji_bub.mean(), 1), "selfcorr_turn": round(100 * x.selfcorr.mean(), 1),
             "latency_med": float(x.response_latency_s.median())}
        if c == "maichat":
            gaps = np.concatenate([np.array(v) for v in x.gaps if len(v)]) if (x.n_msgs > 1).any() else np.array([])
            s |= {"gap_med": float(np.median(gaps)) if len(gaps) else None, "gap_p25_p75": [float(np.percentile(gaps, 25)), float(np.percentile(gaps, 75))] if len(gaps) else None,
                  "idle_med": float(x.idle_med.median()), "cps_med": float(x.cps.median()), "del_any": round(100 * x.del_any.mean(), 1),
                  "abandoned": round(100 * x.abandoned.mean(), 1)}
        shape[gname] = s
    r["shape"] = shape
    # partner after: rajada vs non-rajada, length-matched by lenbin (weights = rajada lenbin distribution)
    after = {}
    for k in ("R3", "R5"):
        a = d[d[k]]; b = d[~d[k] & (d.n_msgs > 0)]
        w = a.lenbin.value_counts(normalize=True)
        def matched(col, fn=np.nanmean):
            vals = []
            for lb, wt in w.items():
                bb = b[b.lenbin == lb][col].dropna()
                if len(bb): vals.append((fn(bb.astype(float)), wt))
            return float(sum(v * wt for v, wt in vals) / sum(wt for _, wt in vals))
        after[k] = {"next_latency_med": [float(a.next_partner_latency.median()), matched("next_partner_latency", np.nanmedian)],
                    "next_chars_med": [float(a.next_partner_chars.median()), matched("next_partner_chars", np.nanmedian)],
                    "next_nmsgs_mean": [round(a.next_partner_nmsgs.mean(), 2), round(matched("next_partner_nmsgs"), 2)],
                    "next_R3_pct": [round(100 * (a.next_partner_max60 >= 3).mean(), 1), round(100 * matched("next_partner_max60", lambda s: (s >= 3).mean()), 1)],
                    "next_q_pct": [round(100 * a.next_partner_q.astype(float).mean(), 1), round(100 * matched("next_partner_q"), 1)],
                    "next_laugh_pct": [round(100 * a.next_partner_laugh.astype(float).mean(), 1), round(100 * matched("next_partner_laugh"), 1)],
                    "session_ends_pct": [round(100 * a.is_last_in_session.mean(), 1), round(100 * matched("is_last_in_session"), 1)],
                    "_note": "[rajada, controle sem rajada com a mesma distribuição de tamanho]"}
    r["partner_after"] = after
    res[c] = r
fn = dump(res, "b2_bursts_stats.json")
import json; print(json.dumps(res, indent=1, default=str)[:12000])
