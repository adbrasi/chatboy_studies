"""a1: what explains the number of bubbles per turn; tests the user's 'anxious -> more/shorter/faster bubbles' hypothesis."""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
from a1_load import enriched

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "data")
T, M = enriched()
res = {}

def dist(c):
    g = T[T.corpus == c]
    vc = g.n_msgs.clip(upper=5).value_counts(normalize=True).sort_index().round(3)
    return {"n_turns": len(g), "mean": round(g.n_msgs.mean(), 2), "share_multi": round((g.n_msgs > 1).mean(), 3),
            "dist(5=5+)": vc.to_dict(),
            "share_msgs_in_multi": round(g.loc[g.n_msgs > 1, "n_msgs"].sum() / g.n_msgs.sum(), 3)}

res["dist"] = {c: dist(c) for c in ["maichat", "whatsapp_nl"]}
print(json.dumps(res["dist"], indent=1))

# speaker-level variance (ICC-ish): share of variance in n_msgs/mean_chars explained by speaker identity
def icc(c, col):
    g = T[T.corpus == c].dropna(subset=[col])
    md = smf.mixedlm(f"{col} ~ 1", g, groups=g["spk"]).fit()
    v = md.cov_re.iloc[0, 0]
    return round(v / (v + md.scale), 3)
T["log_nm"] = np.log(T.n_msgs)
T["log_mean_chars"] = np.log(T.mean_chars)
T["multi"] = (T.n_msgs > 1).astype(int)
res["icc"] = {c: {col: icc(c, col) for col in ["log_nm", "log_mean_chars", "multi"]} for c in ["maichat", "whatsapp_nl"]}
spk = T.groupby(["corpus", "spk"]).agg(n=("n_msgs", "size"), mean_nm=("n_msgs", "mean"), multi=("multi", "mean"))
spk = spk[spk.n >= 20]
res["speaker_spread"] = {c: {"n_speakers": int(len(s)), "multi_p10": round(s.multi.quantile(.1), 2),
                             "multi_p50": round(s.multi.median(), 2), "multi_p90": round(s.multi.quantile(.9), 2),
                             "mean_nm_min": round(s.mean_nm.min(), 2), "mean_nm_max": round(s.mean_nm.max(), 2)}
                         for c, s in spk.groupby(level=0)}
print("ICC", res["icc"], res["speaker_spread"])

# correlations D-vars x outcomes, pooled and within-speaker (speaker-demeaned ranks)
A = T[T.D_emotion.notna()].copy()
A["lat"] = A.response_latency_s
A["log_lat"] = np.log1p(A.lat)
A["log_total"] = np.log(A.total_chars)
dvars = ["D_anxious", "D_arousal", "D_seriousness", "D_playful", "D_valence", "D_engagement", "D_vulnerable",
         "D_tension", "D_seeks_support", "D_flirting"]
outs = ["n_msgs", "mean_chars", "total_chars", "gap_med", "lat"]
corr = {}
for c in ["maichat", "whatsapp_nl"]:
    g = A[A.corpus == c]
    corr[c] = {}
    for d in dvars:
        row = {}
        for o in outs:
            x = g[[d, o, "spk"]].dropna()
            if len(x) < 30:
                continue
            r, p = stats.spearmanr(x[d], x[o])
            # within-speaker: rank-transform then demean per speaker
            xr = x.assign(a=x[d].rank(), b=x[o].rank())
            xr["a"] -= xr.groupby("spk").a.transform("mean")
            xr["b"] -= xr.groupby("spk").b.transform("mean")
            rw, pw = stats.pearsonr(xr.a, xr.b)
            row[o] = {"rho": round(r, 3), "p": float(f"{p:.2g}"), "rho_within": round(rw, 3), "p_within": float(f"{pw:.2g}"), "n": len(x)}
        corr[c][d] = row
res["corr"] = corr
for c in corr:
    print("\n==", c)
    print(pd.DataFrame({d: {o: f"{v['rho']:+.2f}/{v['rho_within']:+.2f}" for o, v in r.items()} for d, r in corr[c].items()}).T)

# direct hypothesis test: anxious / high-arousal vs calm
hyp = {}
for c in ["maichat", "whatsapp_nl"]:
    g = A[A.corpus == c]
    grp = {"anxious(>=.5)": g[g.D_anxious >= .5], "not_anxious(<.2)": g[g.D_anxious < .2],
           "arousal>=3": g[g.D_arousal >= 3], "arousal<=1.5": g[g.D_arousal <= 1.5],
           "serious>=2": g[g.D_seriousness >= 2], "serious<=1": g[g.D_seriousness <= 1],
           "playful>=.5": g[g.D_playful >= .5]}
    hyp[c] = {k: {"n": len(x), "mean_n_msgs": round(x.n_msgs.mean(), 2), "share_multi": round(x.multi.mean(), 3),
                  "med_bubble_chars": float(x.mean_chars.median()), "med_total_chars": float(x.total_chars.median()),
                  "med_gap_s": None if x.gap_med.notna().sum() < 5 else round(x.gap_med.median(), 1),
                  "med_latency_s": None if x.lat.notna().sum() < 5 else round(x.lat.median(), 1)}
              for k, x in grp.items()}
    a, b = grp["anxious(>=.5)"], grp["not_anxious(<.2)"]
    hyp[c]["MWU_anx_vs_not"] = {o: float(f"{stats.mannwhitneyu(a[o].dropna(), b[o].dropna()).pvalue:.2g}")
                                for o in ["n_msgs", "mean_chars", "total_chars", "gap_med", "lat"] if a[o].notna().sum() > 5}
res["hypothesis"] = hyp
print(json.dumps(hyp, indent=1))

# regression with speaker FE: log(n_msgs) ~ D-vars + log_total + context
reg = {}
for c in ["maichat", "whatsapp_nl"]:
    g = A[A.corpus == c].dropna(subset=["prev_partner_chars"]).copy()
    g["log_pp"] = np.log(g.prev_partner_chars)
    g["pos"] = np.log1p(g.turn_in_session)
    for name, f in {"no_len": "multi ~ D_anxious + D_arousal + D_seriousness + D_playful + D_valence + D_engagement + log_pp + pos + C(spk)",
                    "with_len": "multi ~ D_anxious + D_arousal + D_seriousness + D_playful + D_valence + D_engagement + log_total + log_pp + pos + C(spk)",
                    "bubble_len": "log_mean_chars ~ D_anxious + D_arousal + D_seriousness + D_playful + D_valence + D_engagement + log_pp + pos + C(spk)"}.items():
        m = smf.ols(f, g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.conv_id)[0]})
        keep = [k for k in m.params.index if not k.startswith("C(") and k != "Intercept"]
        reg[f"{c}:{name}"] = {k: {"b": round(m.params[k], 3), "p": float(f"{m.pvalues[k]:.2g}")} for k in keep} | {"n": int(m.nobs), "r2": round(m.rsquared, 3)}
    # R2 decomposition: speaker FE only vs FE + D + length
    base = smf.ols("multi ~ C(spk)", g).fit().rsquared
    dv = smf.ols("multi ~ D_anxious + D_arousal + D_seriousness + D_playful + D_valence + D_engagement + log_pp + pos", g).fit().rsquared
    ln = smf.ols("multi ~ log_total", g).fit().rsquared
    full = smf.ols("multi ~ C(spk) + D_anxious + D_arousal + D_seriousness + D_playful + D_valence + D_engagement + log_pp + pos + log_total", g).fit().rsquared
    reg[f"{c}:R2"] = {"speaker_only": round(base, 3), "Dvars+ctx_only": round(dv, 3), "log_total_only": round(ln, 3), "all": round(full, 3)}
res["regression"] = reg
print(json.dumps(reg, indent=1))

# categorical: emotion, intent, phase, relationship
cat = {}
for c in ["maichat", "whatsapp_nl"]:
    g = A[A.corpus == c]
    for v in ["D_emotion", "D_intent", "D_phase", "D_relationship"]:
        t = g.groupby(v).agg(n=("n_msgs", "size"), mean_nm=("n_msgs", "mean"), multi=("multi", "mean"),
                             med_bubble=("mean_chars", "median"), med_total=("total_chars", "median"),
                             med_lat=("lat", "median")).round(2)
        t = t[t.n >= 25].sort_values("mean_nm", ascending=False)
        cat[f"{c}:{v}"] = t.reset_index().to_dict(orient="records")
        print(f"\n{c} {v}\n", t)
res["categorical"] = cat

# partner length / partner bubbles / position
ctx = {}
for c in ["maichat", "whatsapp_nl"]:
    g = T[T.corpus == c]
    x = g.dropna(subset=["prev_partner_nmsgs"])
    ctx[c] = {
        "rho_partner_nmsgs_vs_nmsgs": round(stats.spearmanr(x.prev_partner_nmsgs, x.n_msgs)[0], 3),
        "rho_partner_chars_vs_total": round(stats.spearmanr(x.prev_partner_chars, x.total_chars)[0], 3),
        "rho_partner_chars_vs_nmsgs": round(stats.spearmanr(x.prev_partner_chars, x.n_msgs)[0], 3),
        "rho_prev_own_nmsgs_vs_nmsgs": round(stats.spearmanr(*g[["prev_own_nmsgs", "n_msgs"]].dropna().T.values)[0], 3),
        "P(multi|partner multi)": round(x[x.prev_partner_nmsgs > 1].multi.mean(), 3) if "multi" in x else None,
        "P(multi|partner single)": round(x[x.prev_partner_nmsgs == 1].multi.mean(), 3),
        "rho_position_vs_nmsgs": round(stats.spearmanr(g.turn_in_session, g.n_msgs)[0], 3),
        "multi_by_position": g.assign(b=pd.cut(g.turn_in_session, [-1, 0, 2, 10, 30, 1e9], labels=["0", "1-2", "3-10", "11-30", "31+"]))
                          .groupby("b", observed=True).multi.mean().round(3).to_dict(),
        "multi_by_latency": g.assign(b=pd.cut(g.response_latency_s, [-1, 0.5, 60.5, 600, 3600, 1e9] if c == "whatsapp_nl" else [-1, 5, 15, 30, 60, 1e9]))
                          .groupby("b", observed=True).agg(n=("multi", "size"), multi=("multi", "mean"), nm=("n_msgs", "mean")).round(3).reset_index().astype(str).to_dict(orient="records"),
    }
res["context"] = ctx
print(json.dumps(ctx, indent=1))
json.dump(res, open(os.path.join(OUT, "a1_bursts.json"), "w"), indent=1, ensure_ascii=False, default=str)
