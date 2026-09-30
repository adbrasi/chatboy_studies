"""a1: timing inside bursts and response latency (maichat = live seconds; whatsapp_nl = minute resolution)."""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from a1_load import enriched

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "data")
T, M = enriched()
res = {}
q = lambda s: {"n": int(s.notna().sum()), "p10": round(float(s.quantile(.1)), 1), "p25": round(float(s.quantile(.25)), 1),
               "med": round(float(s.median()), 1), "p75": round(float(s.quantile(.75)), 1), "p90": round(float(s.quantile(.9)), 1)}

# ---- intra-burst gaps
mc = T[T.corpus == "maichat"]
Mm = M[M.corpus == "maichat"].set_index(["conv_id", "idx"])
rows = []
for _, t in mc[mc.n_msgs > 1].iterrows():
    for k in range(1, len(t.msg_idxs)):
        m = Mm.loc[(t.conv_id, t.msg_idxs[k])]
        rows.append({"gap": t.gaps[k - 1] if k - 1 < len(t.gaps) else np.nan, "chars": m.f_n_chars, "compose": m.ty_compose_s,
                     "idle": m.ty_idle_before_typing_s, "pos": k, "prev_chars": Mm.loc[(t.conv_id, t.msg_idxs[k - 1])].f_n_chars})
G = pd.DataFrame(rows).dropna(subset=["gap"])
res["maichat_intra_gap"] = q(G.gap) | {
    "rho_gap_vs_next_chars": round(stats.spearmanr(G.gap, G.chars)[0], 3),
    "idle_between_bubbles": q(G.idle),
    "share_started_next_before_sending_prev(idle<0)": round(float((G.idle < 0).mean()), 3),
    "share_idle_under_1s": round(float((G.idle < 1).mean()), 3),
    "gap_by_next_chars": G.assign(b=pd.cut(G.chars, [0, 5, 15, 30, 60, 1000])).groupby("b", observed=True).gap.median().round(1).rename(index=str).to_dict()}
# gaps vs response latency (same people): intra-burst gap is shorter than response latency?
lat = mc.response_latency_s.dropna()
res["maichat_latency"] = q(lat)
# first-bubble vs later
print(json.dumps(res, indent=1))

wa = T[T.corpus == "whatsapp_nl"]
wg = np.concatenate([g for g in wa.gaps if len(g)])
res["whatsapp_intra_gap"] = {"n": len(wg), "same_minute": round(float((wg == 0).mean()), 3), "1min": round(float((wg == 60).mean()), 3),
                             "2-9min": round(float(((wg >= 120) & (wg < 600)).mean()), 3), ">=10min": round(float((wg >= 600).mean()), 3),
                             ">=60min": round(float((wg >= 3600).mean()), 3)}
lw = wa.response_latency_s.dropna()
res["whatsapp_latency"] = q(lw) | {"same_minute": round(float((lw == 0).mean()), 3), "<=2min": round(float((lw <= 120).mean()), 3),
                                   ">=1h": round(float((lw >= 3600).mean()), 3)}
print(json.dumps({k: res[k] for k in ["whatsapp_intra_gap", "whatsapp_latency"]}, indent=1))

# ---- late add-ons ("ah, e..."): bubbles sent >=2 min after previous own bubble with no partner reply in between
late = []
for _, t in wa[wa.n_msgs > 1].iterrows():
    for k, g in enumerate(t.gaps):
        if g >= 120 and k + 1 < len(t.texts):
            late.append({"gap": g, "text": t.texts[k + 1].strip(), "prev": t.texts[k].strip(), "prev_has_q": "?" in t.texts[k]})
L = pd.DataFrame(late)
res["late_addons_whatsapp"] = {"n": len(L), "share_of_turns_with_late_addon": round(len(L) / len(wa), 3),
                               "share_q": round(L.text.str.contains(r"\?").mean(), 3),
                               "share_prev_was_q": round(L.prev_has_q.mean(), 3),
                               "gap": q(L.gap),
                               "examples": L.sample(12, random_state=2)[["prev", "text", "gap"]].values.tolist()}
# baseline: share q among all non-first bubbles
nonfirst = [x for t in wa[wa.n_msgs > 1].texts for x in t[1:]]
res["late_addons_whatsapp"]["baseline_share_q_nonfirst"] = round(np.mean(["?" in x for x in nonfirst]), 3)
# double text when previous own ended with a question and no answer came
print(json.dumps(res["late_addons_whatsapp"], indent=1, ensure_ascii=False))

# ---- latency by D (the replier's own emotion / seriousness) and by partner's previous turn
A = T[T.D_emotion.notna()].copy()
# D of the partner's previous turn
Dcols = ["D_seriousness", "D_vulnerable", "D_anxious", "D_seeks_support", "D_hook", "D_playful", "D_engagement", "D_emotion", "D_tension"]
A = A.sort_values(["conv_id", "turn_idx"])
for c in Dcols:
    A["prevD_" + c[2:]] = A.groupby("conv_id")[c].shift(1).where(A.groupby("conv_id").turn_idx.shift(1) == A.turn_idx - 1)
lat_tab = {}
for c in ["maichat", "whatsapp_nl"]:
    g = A[A.corpus == c]
    out = {}
    out["by_emotion"] = g.groupby("D_emotion").response_latency_s.agg(["size", "median", lambda s: s.quantile(.25), lambda s: s.quantile(.75)]).round(1).set_axis(["n", "med", "p25", "p75"], axis=1).query("n>=25").sort_values("med").reset_index().to_dict(orient="records")
    g2 = g.assign(ser=g.D_seriousness.round().clip(0, 3))
    out["by_seriousness"] = g2.groupby("ser").response_latency_s.agg(["size", "median"]).round(1).reset_index().to_dict(orient="records")
    corr = {}
    for v in ["D_engagement", "D_seriousness", "D_vulnerable", "D_anxious", "D_arousal", "D_playful", "D_valence",
              "prevD_seriousness", "prevD_vulnerable", "prevD_seeks_support", "prevD_hook", "prevD_engagement", "prevD_tension", "prev_partner_chars", "prev_partner_latency"]:
        x = g[[v, "response_latency_s", "spk"]].dropna()
        r, p = stats.spearmanr(x[v], x.response_latency_s)
        xr = x.assign(a=x[v].rank(), b=x.response_latency_s.rank())
        xr["a"] -= xr.groupby("spk").a.transform("mean"); xr["b"] -= xr.groupby("spk").b.transform("mean")
        rw = stats.pearsonr(xr.a, xr.b)
        corr[v] = {"rho": round(r, 3), "p": float(f"{p:.2g}"), "rho_within_spk": round(rw[0], 3), "p_w": float(f"{rw[1]:.2g}"), "n": len(x)}
    out["corr_latency"] = corr
    # latency after partner question vs not
    x = g.dropna(subset=["prev_partner_q", "response_latency_s"])
    out["lat_after_partner_q"] = {"q": round(float(x[x.prev_partner_q == True].response_latency_s.median()), 1),
                                  "no_q": round(float(x[x.prev_partner_q == False].response_latency_s.median()), 1),
                                  "mwu_p": float(f"{stats.mannwhitneyu(x[x.prev_partner_q == True].response_latency_s, x[x.prev_partner_q == False].response_latency_s).pvalue:.2g}")}
    x = g.dropna(subset=["prevD_vulnerable", "response_latency_s"])
    out["lat_after_partner_vulnerable>=.5"] = {"vuln": q(x[x.prevD_vulnerable >= .5].response_latency_s), "not": q(x[x.prevD_vulnerable < .2].response_latency_s)}
    # does replying fast get a fast/longer reply back? my latency vs partner's next latency and next length
    y = T[T.corpus == c].dropna(subset=["response_latency_s", "next_partner_latency"])
    out["reciprocity_rho(my_lat, partner_next_lat)"] = round(stats.spearmanr(y.response_latency_s, y.next_partner_latency)[0], 3)
    yy = y.assign(a=y.response_latency_s.rank(), b=y.next_partner_latency.rank())
    for col in "ab":
        yy[col] -= yy.groupby("conv_id")[col].transform("mean")
    out["reciprocity_within_conv"] = round(stats.pearsonr(yy.a, yy.b)[0], 3)
    y2 = T[T.corpus == c].dropna(subset=["response_latency_s", "next_partner_chars"])
    out["rho(my_lat, partner_next_chars)"] = round(stats.spearmanr(y2.response_latency_s, y2.next_partner_chars)[0], 3)
    # latency of turn that ends the session vs others (whatsapp) / conversation
    lat_tab[c] = out
    print("\n==", c); print(json.dumps(out, indent=1, default=str))
res["latency_D"] = lat_tab

# ---- latency x session continuation (whatsapp): slow reply -> does the conversation die sooner?
wa = T[T.corpus == "whatsapp_nl"].copy()
wa["lat_b"] = pd.cut(wa.response_latency_s, [-1, 0, 60, 300, 1800, 3600, 11000], labels=["<1m", "1m", "2-5m", "6-30m", "31-60m", "1-3h"])
res["whatsapp_lat_vs_session_end"] = wa.groupby("lat_b", observed=True).agg(n=("n_msgs", "size"), p_last=("is_last_in_session", "mean"),
                                                                           next_partner_lat_med=("next_partner_latency", "median"),
                                                                           nm=("n_msgs", "mean"), chars=("total_chars", "median")).round(3).reset_index().astype(str).to_dict(orient="records")
print(json.dumps(res["whatsapp_lat_vs_session_end"], indent=1))
json.dump(res, open(os.path.join(OUT, "a1_timing.json"), "w"), indent=1, ensure_ascii=False, default=str)
