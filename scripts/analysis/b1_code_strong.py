"""b1: code-only view of the stronger actors (gpt-6-luna, deepseek-flash) — no Jev calls (bank not run on them: budget stop).
Applies the dev-trained code LR (same features as M1) to the new replies; surface stats vs humans."""
import json, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import b1_common as B
ctxs, units = B.load_units()
for u in units:
    c = ctxs[u["ckey"]]; u["split"], u["conv"], u["src"] = c["split"], c["conv"], c["src"]
    u["cf"] = B.code_feats(u["text"], B.last_other(c)); u["chars"] = len(u["text"])
old = [u for u in units if u["cond"] not in ("base_luna", "styled_luna", "base_deepseek", "styled_deepseek")]
dev = [u for u in old if u["split"] == "dev"]
X = lambda us: np.array([[u["cf"][c] for c in B.CODE_COLS] for u in us]); y = lambda us: np.array([u["label"] for u in us])
m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X(dev), y(dev))
out = {}
hum = [u for u in units if u["cond"] == "human" and u["src"].startswith("a8") and u["split"] == "test"]
for cn in ["base_gemini", "styled_gemini", "base_luna", "styled_luna", "base_deepseek", "styled_deepseek"]:
    us = hum + [u for u in units if u["cond"] == cn and u["split"] == "test"]
    p = m.predict_proba(X(us))[:, 1]
    ev = B.eval_scores(p, y(us), [u["chars"] for u in us], [u["conv"] for u in us], 300)
    allc = [u for u in units if u["cond"] == cn]
    f = lambda k: round(100 * float(np.mean([u["cf"][k] > 0 for u in allc])), 1)
    out[cn] = {"code_lr_test": ev, "median_chars": float(np.median([u["chars"] for u in allc])), "pct_ends_q": f("ends_q"), "pct_excl": f("n_excl"),
               "pct_emoji": f("n_emoji"), "pct_validation_rx": f("validation_rx"), "pct_perf_open": f("perf_open"), "pct_gen_recip_q": f("gen_recip_q"),
               "pct_3plus_sent": round(100 * float(np.mean([u["cf"]["sentences"] >= 3 for u in allc])), 1), "pct_llmish": f("llmish_hits")}
h8 = [u for u in units if u["cond"] == "human" and u["src"].startswith("a8")]
out["human_a8"] = {"median_chars": float(np.median([u["chars"] for u in h8])), **{k: round(100 * float(np.mean([u["cf"][kk] > 0 for u in h8])), 1) for k, kk in
                   [("pct_ends_q", "ends_q"), ("pct_excl", "n_excl"), ("pct_emoji", "n_emoji"), ("pct_validation_rx", "validation_rx"), ("pct_perf_open", "perf_open"), ("pct_gen_recip_q", "gen_recip_q"), ("pct_llmish", "llmish_hits")]},
                   "pct_3plus_sent": round(100 * float(np.mean([u["cf"]["sentences"] >= 3 for u in h8])), 1)}
B.save("b1_strong_actors_code.json", out)
for k, v in out.items():
    print(k, {kk: (vv if not isinstance(vv, dict) else (vv["auc"], vv["auc_len"], vv["ci"])) for kk, vv in v.items()})
