"""a1: typing dynamics in MaiChat from the raw keystroke-state logs (speed, composition time, pauses,
deletions, abandoned drafts, 'typed and gave up'), by device and by Jev D labels of the turn.
Derives an empirical formula for how long 'typing...' should be shown for N characters."""
import glob, json, os, sys
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
from a1_load import enriched

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "analysis", "data")
iso = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))

rows, drafts = [], []
for f in sorted(glob.glob(os.path.join(ROOT, "data/raw/maichat/maichat/*.json"))):
    d = json.load(open(f)); cid = os.path.basename(f)[:-5]
    first = d["firstId"]["$oid"]; prev_submit = None; prev_spk = None
    for i, m in enumerate(d["messages"]):
        spk = "A" if m["ofUser"]["$oid"] == first else "B"
        submit = iso(m["time"]["$date"]); final = m["content"]
        logs = sorted([(iso(l["time"]["$date"]), l["message"]) for l in m["logs"]], key=lambda x: x[0])
        r = {"conv_id": cid, "idx": i, "speaker": spk, "device": m.get("deviceType"), "final_len": len(final), "n_logs": len(logs)}
        if logs:
            t0 = logs[0][0]
            iv = [(b[0] - a[0]).total_seconds() for a, b in zip(logs, logs[1:])] + [(submit - logs[-1][0]).total_seconds()]
            L = [len(t) for _, t in logs]
            # resets: text emptied (or cut to <=2 chars) after having >=8 chars, then typing resumed
            resets, peak_since = [], 0
            for k, (tt, tx) in enumerate(logs):
                if len(tx) <= 1 and peak_since >= 8 and k < len(logs) - 1:
                    best = max((x for _, x in logs[:k]), key=len)
                    resets.append(best)
                    peak_since = 0
                else:
                    peak_since = max(peak_since, len(tx))
            deleted = sum(max(0, a - b) for a, b in zip(L, L[1:]))
            r.update({"compose_s": (submit - t0).total_seconds(),
                      "active_s": sum(x for x in iv if x <= 2.0),
                      "n_pause2": sum(x > 2 for x in iv), "n_pause5": sum(x > 5 for x in iv), "max_pause": max(iv),
                      "pause_time_s": sum(x for x in iv if x > 2.0),
                      "deleted": deleted, "peak": max(L), "abandoned": max(0, max(L) - len(final)),
                      "n_resets": len(resets),
                      "start_rel_prev_s": (t0 - prev_submit).total_seconds() if prev_submit else None,
                      "prev_is_partner": prev_spk is not None and prev_spk != spk})
            for x in resets:
                drafts.append({"conv_id": cid, "idx": i, "draft": x, "final": final})
        rows.append(r)
        prev_submit, prev_spk = submit, spk
K = pd.DataFrame(rows)
print(K.device.value_counts(), len(K))
K = K[K.compose_s.notna() & (K.final_len > 0)]
K["cps"] = K.final_len / K.compose_s.clip(lower=0.2)
K["cps_active"] = K.final_len / K.active_s.clip(lower=0.2)
K["del_ratio"] = K.deleted / K.final_len

# link messages -> turns -> D labels
T, M = enriched()
mc = T[T.corpus == "maichat"]
m2t = {}
for _, t in mc.iterrows():
    for k, i in enumerate(t.msg_idxs):
        m2t[(t.conv_id, i)] = (t.turn_idx, k, t.n_msgs)
K["turn_idx"] = [m2t.get((c, i), (None,))[0] for c, i in zip(K.conv_id, K.idx)]
K["pos_in_burst"] = [m2t.get((c, i), (None, None))[1] for c, i in zip(K.conv_id, K.idx)]
K = K.merge(mc[["conv_id", "turn_idx", "D_emotion", "D_seriousness", "D_vulnerable", "D_anxious", "D_arousal", "D_tension",
                "D_playful", "D_engagement", "D_seeks_support", "D_intent", "spk"]], on=["conv_id", "turn_idx"], how="left")
res = {}
q = lambda s: {"n": int(s.notna().sum()), "p25": round(float(s.quantile(.25)), 2), "med": round(float(s.median()), 2), "p75": round(float(s.quantile(.75)), 2)}
res["overall"] = {"compose_s": q(K.compose_s), "cps": q(K.cps), "cps_active(pauses>2s removed)": q(K.cps_active),
                  "share_with_deletion": round(float((K.deleted > 0).mean()), 3), "share_abandoned>=3chars": round(float((K.abandoned >= 3).mean()), 3),
                  "share_reset(draft wiped & rewritten)": round(float((K.n_resets > 0).mean()), 3),
                  "share_pause>2s": round(float((K.n_pause2 > 0).mean()), 3),
                  "share_started_before_partner_msg_arrived": round(float(((K.start_rel_prev_s < 0) & K.prev_is_partner).sum() / K.prev_is_partner.sum()), 3)}
res["by_device"] = {d: {"n": len(g), "cps_med": round(g.cps.median(), 2), "cps_active_med": round(g.cps_active.median(), 2),
                        "compose_med": round(g.compose_s.median(), 1), "del_share": round((g.deleted > 0).mean(), 3)} for d, g in K.groupby("device")}
print(json.dumps(res, indent=1))

# ---- formula: compose_s ~ a + b*chars (median regression), by device; and log-log
fm = {}
for d, g in [("all", K)] + list(K.groupby("device")):
    g = g[(g.compose_s > 0) & (g.compose_s < 300)]
    qr = smf.quantreg("compose_s ~ final_len", g).fit(q=.5)
    q25 = smf.quantreg("compose_s ~ final_len", g).fit(q=.25); q75 = smf.quantreg("compose_s ~ final_len", g).fit(q=.75)
    ll = smf.ols("np.log(compose_s) ~ np.log(final_len)", g[g.compose_s > 0.1]).fit()
    fm[d] = {"median: a_s": round(qr.params.Intercept, 2), "median: s_per_char": round(qr.params.final_len, 3),
             "p25: a,b": [round(q25.params.Intercept, 2), round(q25.params.final_len, 3)],
             "p75: a,b": [round(q75.params.Intercept, 2), round(q75.params.final_len, 3)],
             "loglog: exp(a)": round(float(np.exp(ll.params.iloc[0])), 2), "loglog: exponent": round(ll.params.iloc[1], 3), "loglog_r2": round(ll.rsquared, 3), "n": len(g)}
res["formula"] = fm
# add moment effects to the log model (with device + length control)
g = K[(K.compose_s > 0.1) & (K.compose_s < 300) & K.D_seriousness.notna()].copy()
g["ser2"] = (g.D_seriousness >= 1.5).astype(int)
mm = smf.ols("np.log(compose_s) ~ np.log(final_len) + C(device) + D_seriousness + D_vulnerable + D_anxious + D_arousal + D_tension + D_playful + D_engagement", g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.conv_id)[0]})
res["log_compose_model"] = {k: {"b": round(v, 3), "p": float(f"{mm.pvalues[k]:.2g}"), "x_factor": round(float(np.exp(v)), 3)} for k, v in mm.params.items()} | {"r2": round(mm.rsquared, 3), "n": int(mm.nobs)}
print(json.dumps(fm, indent=1)); print(json.dumps(res["log_compose_model"], indent=1))

# ---- hesitation by moment
def summ(g):
    return pd.Series({"n": len(g), "cps_med": g.cps.median(), "del_share": (g.deleted > 0).mean(), "del_ratio_mean": g.del_ratio.clip(upper=3).mean(),
                      "aband_share": (g.abandoned >= 3).mean(), "reset_share": (g.n_resets > 0).mean(), "pause2_share": (g.n_pause2 > 0).mean(),
                      "pause5_share": (g.n_pause5 > 0).mean(), "med_chars": g.final_len.median(),
                      "sec_per_char_med": (g.compose_s / g.final_len).median()})
hes = {}
K["vuln_b"] = np.where(K.D_vulnerable >= .5, "vulnerable>=.5", np.where(K.D_vulnerable < .2, "vulnerable<.2", "mid"))
K["ser_b"] = pd.cut(K.D_seriousness, [-.1, .5, 1.5, 3.1], labels=["banter(0)", "casual(1)", "serious(2-3)"])
K["anx_b"] = np.where(K.D_anxious >= .5, "anxious>=.5", np.where(K.D_anxious < .2, "anxious<.2", "mid"))
K["ten_b"] = np.where(K.D_tension >= .5, "tension>=.5", np.where(K.D_tension < .2, "tension<.2", "mid"))
K["len_b"] = pd.cut(K.final_len, [0, 15, 40, 1000], labels=["<=15", "16-40", ">40"])
for v in ["vuln_b", "ser_b", "anx_b", "ten_b", "D_emotion"]:
    t = K.groupby(v, observed=True).apply(summ).round(3)
    t = t[t.n >= 30]
    hes[v] = t.reset_index().to_dict(orient="records")
    print("\n", v, "\n", t)
# length-controlled comparison: vulnerable vs not within length buckets
lc = K[K.vuln_b != "mid"].groupby(["len_b", "vuln_b"], observed=True).apply(summ).round(3)
print(lc); hes["vuln_by_len"] = lc.reset_index().astype(str).to_dict(orient="records")
# logistic: deletion / abandonment ~ vulnerable + length + device
for y in ["deleted", "abandoned"]:
    g = K.dropna(subset=["D_vulnerable"]).assign(yy=lambda x: (x[y] >= (1 if y == "deleted" else 3)).astype(int))
    lm = smf.logit("yy ~ np.log(final_len) + C(device) + D_vulnerable + D_seriousness + D_anxious + D_tension + D_arousal", g).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(g.conv_id)[0]})
    hes[f"logit_{y}"] = {k: {"OR": round(float(np.exp(v)), 3), "p": float(f"{lm.pvalues[k]:.2g}")} for k, v in lm.params.items()}
    print(y, hes[f"logit_{y}"])
res["hesitation"] = hes

# ---- drafts wiped and rewritten: examples + how different was the final
D = pd.DataFrame(drafts)
res["resets"] = {"n": len(D), "examples": D.sample(min(15, len(D)), random_state=4)[["draft", "final"]].values.tolist()}
ab = K[K.abandoned >= 10].sort_values("abandoned", ascending=False)
print("resets", len(D)); print(D.sample(min(15, len(D)), random_state=4)[["draft", "final"]].to_string())

# ---- first bubble of a burst vs later bubbles: speed/length
K["first_of_multi"] = (K.pos_in_burst == 0) & K.idx.notna()
res["burst_position"] = K.dropna(subset=["pos_in_burst"]).assign(p=lambda x: x.pos_in_burst.clip(upper=2)).groupby("p").agg(
    n=("cps", "size"), med_chars=("final_len", "median"), med_compose=("compose_s", "median"), cps=("cps", "median")).round(2).reset_index().to_dict(orient="records")
print(res["burst_position"])
K.drop(columns=[]).to_csv(os.path.join(OUT, "a1_typing_messages.csv.gz"), index=False, compression="gzip")
json.dump(res, open(os.path.join(OUT, "a1_typing.json"), "w"), indent=1, ensure_ascii=False, default=str)
