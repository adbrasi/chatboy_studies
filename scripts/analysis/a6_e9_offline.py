"""E9 (no new Jev calls): latency/retry stats over all a6 calls; calibration of the base P-pass (predictive) against what
the speaker actually did; within-conversation stability of `relationship` in the base layer; E7 significance."""
import json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import a6_common as C

out = {}
# ---- A. latency over all a6 calls
recs = [json.loads(l) for l in open(C.CACHE, encoding="utf-8")]
lat = np.array([r["latency_s"] for r in recs]); tok = np.array([r["usage"].get("input_tokens", 0) for r in recs])
out["all_calls"] = {"n": len(recs), "lat_p50": float(np.median(lat)), "lat_p90": float(np.percentile(lat, 90)), "lat_p99": float(np.percentile(lat, 99)),
                    "lat_max": float(lat.max()), "frac_over_1s": float((lat > 1).mean()), "frac_over_1_5s": float((lat > 1.5).mean()),
                    "retries(attempt>0)": int(sum(r.get("attempt", 0) > 0 for r in recs)), "tokens_total": int(tok.sum()),
                    "cost_total_usd": float(sum(r["usage"].get("cost", 0) for r in recs)),
                    "usd_per_1k_tokens": float(sum(r["usage"].get("cost", 0) for r in recs) / tok.sum() * 1000),
                    "models": Counter(r["model"] for r in recs).most_common(3)}
bins = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 30000)]
out["lat_by_tokens"] = [{"tokens": f"{a}-{b}", "n": int(((tok >= a) & (tok < b)).sum()), "p50": float(np.median(lat[(tok >= a) & (tok < b)])),
                         "p90": float(np.percentile(lat[(tok >= a) & (tok < b)], 90))} for a, b in bins if ((tok >= a) & (tok < b)).sum()]
# ---- B. calibration of the P pass (base layer) vs actual outcome
base = [json.loads(l) for l in open(os.path.join(C.PROC, "jev_base.jsonl"), encoding="utf-8")]
def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float); o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # ties: average ranks
    from collections import defaultdict as dd
    g = dd(list)
    for i, v in enumerate(s): g[v].append(i)
    for v, ix in g.items(): r[ix] = r[ix].mean()
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def reliab(p, y, edges=(0, .1, .2, .35, .5, .65, .8, 1.01)):
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() >= 10: rows.append({"bin": f"[{a:.2f},{b:.2f})", "n": int(m.sum()), "mean_p": round(float(p[m].mean()), 3), "freq": round(float(y[m].mean()), 3)})
    return rows
B = {}
for corpus in ("maichat", "whatsapp_nl"):
    R = [r for r in base if r["P"] and r["corpus"] == corpus]
    res = {"n": len(R)}
    for q, yf in {"p_laugh": lambda r: r["laugh"], "p_question": lambda r: r["has_q"], "p_emoji": lambda r: r["n_emoji"] > 0,
                  "p_end": lambda r: r["farewell"]}.items():
        p = np.array([r["P"][q]["noul"] for r in R]); y = np.array([bool(yf(r)) for r in R])
        ece = sum(((p >= a) & (p < b)).mean() * abs(y[(p >= a) & (p < b)].mean() - p[(p >= a) & (p < b)].mean())
                  for a, b in zip(np.arange(0, 1, .1), np.arange(.1, 1.1, .1)) if ((p >= a) & (p < b)).sum())
        res[q] = {"base_rate": float(y.mean()), "mean_p": float(p.mean()), "auc": auc(y, p), "brier": float(np.mean((p - y) ** 2)),
                  "brier_baseline": float(y.mean() * (1 - y.mean())), "ece": float(ece), "reliability": reliab(p, y)}
    # p_n_msgs choice: accuracy by confidence
    lab = lambda n: "1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4+"
    ch = [r["P"]["p_n_msgs"] for r in R]; y = [lab(r["n_msgs"]) for r in R]
    conf = np.array([c["confidence"] for c in ch]); corr = np.array([c["choice"] == t for c, t in zip(ch, y)])
    res["p_n_msgs"] = {"acc": float(corr.mean()), "majority_baseline": Counter(y).most_common(1)[0][1] / len(y), "pred_dist": Counter(c["choice"] for c in ch).most_common(),
                       "true_dist": Counter(y).most_common(),
                       "by_conf": [{"conf": f"[{a},{b})", "n": int(((conf >= a) & (conf < b)).sum()), "acc": float(corr[(conf >= a) & (conf < b)].mean())}
                                   for a, b in ((0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)) if ((conf >= a) & (conf < b)).sum() >= 10],
                       "p1_auc_single_vs_multi": auc(np.array([t == "1" for t in y]), np.array([c["probabilities"]["1"] for c in ch]))}
    B[corpus] = res
out["P_pass_calibration"] = B
# D-pass choice confidence vs agreement under perturbation (E1 runid): does low confidence predict instability?
e1 = [r for r in recs if r["tag"] in ("e1_ident", "e1_runid")]
# ---- C. relationship stability within conversation (base, n=8 context)
st = {}
for corpus in ("maichat", "whatsapp_nl"):
    byc = defaultdict(list)
    for r in base:
        if r["corpus"] == corpus: byc[r["conv_id"]].append(r)
    agree, confs, n_labels = [], [], []
    for cid, rs in byc.items():
        labs = [r["D"]["relationship"]["choice"] for r in rs]
        mode = Counter(labs).most_common(1)[0][0]
        agree.append(np.mean([l == mode for l in labs])); n_labels.append(len(set(labs)))
        # aggregated: sum of probabilities over the conversation
    # agreement of the argmax of summed probabilities using only the first 10 annotated turns vs the whole conversation
    agg_first10 = []
    for cid, rs in byc.items():
        rs = sorted(rs, key=lambda r: r["turn_idx"])
        def agg(rr):
            s = Counter()
            for r in rr: s.update(r["D"]["relationship"]["probabilities"])
            return s.most_common(1)[0][0]
        agg_first10.append(agg(rs[:10]) == agg(rs))
    st[corpus] = {"n_convs": len(byc), "mean_frac_turns_equal_mode": float(np.mean(agree)), "mean_distinct_labels_per_conv": float(np.mean(n_labels)),
                  "agg_first10_turns_equals_agg_all": float(np.mean(agg_first10))}
out["relationship_within_conv"] = st
# ---- D. E7 paired tests
try:
    e7 = [r for r in recs if r["tag"] in ("e7_full", "e7_first")]
except Exception:
    pass
print(C.dump("e9_offline.json", out))
print(json.dumps(out, indent=1, default=str)[:9000])
