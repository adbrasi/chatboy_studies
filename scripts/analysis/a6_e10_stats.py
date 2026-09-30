"""E10 (no new calls): extra statistics over the a6 cache: does low confidence predict instability (E1)?
paired tests and bootstrap CIs for E7; per-token cost model from E4."""
import json, os, sys, random
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import a6_common as C

recs = [json.loads(l) for l in open(C.CACHE, encoding="utf-8")]
out = {}
# --- E1: pair identical-state calls with their run_id twins via state content
def key(st):
    st = dict(st); st.pop("run_id", None); return json.dumps(st, sort_keys=True, ensure_ascii=False)
g = defaultdict(list)
for r in recs:
    if r["tag"] in ("e1_ident", "e1_runid", "e6_json_AB_explicit", "e6_json_AB_explicit_pert"):
        pass
# we did not store the state in the cache; recompute E1 pairs from the analysis file instead
e1 = [r for r in recs if r["tag"] in ("e1_ident", "e1_runid")]
# order in cache = completion order; group by answers' question set and closeness is not possible -> rebuild via the E1 script's sampling
sys.argv = ["x"]
by = C.load_turns(); base = C.load_base()
S = C.sample_turns(by, "maichat", 40, seed=61, in_base=base) + C.sample_turns(by, "whatsapp_nl", 20, seed=62, in_base=base)
QD = C.AB.q_descriptive()
import random as _r
rows = []
for conv, t in S:
    st = C.d_state(conv, t, 8)
    a = C.raw_ask(st, QD, "e1_ident", 0); b = C.raw_ask(st, QD, "e1_ident", 1)
    runs = [C.raw_ask({"run_id": f"r{r}-{_r.Random(r).randint(1000,9999)}"} | st, QD, "e1_runid", r) for r in (1, 2)]
    if not (a and b and all(runs)): continue
    for q in QD:
        if QD[q]["type"] == "noul": continue
        answers = [a["answers"][q], b["answers"][q]] + [x["answers"][q] for x in runs]
        conf = answers[0]["confidence"]
        if QD[q]["type"] == "choice":
            unstable = len({x["choice"] for x in answers}) > 1
        else:
            unstable = (max(x["score"] for x in answers) - min(x["score"] for x in answers)) > 0.15
        rows.append((QD[q]["type"], conf, unstable))
assert C.NEW_CALLS[0] == 0, "should be cached"
res = {}
for typ in ("choice", "score"):
    rr = [(c, u) for t, c, u in rows if t == typ]
    c = np.array([x[0] for x in rr]); u = np.array([x[1] for x in rr])
    res[typ] = [{"conf": f"[{a},{b})", "n": int(((c >= a) & (c < b)).sum()), "frac_unstable": float(u[(c >= a) & (c < b)].mean())}
                for a, b in ((0, .5), (.5, .7), (.7, .9), (.9, 1.01)) if ((c >= a) & (c < b)).sum()]
out["E1_confidence_vs_instability(4 calls: 2 identical + 2 run_id)"] = res

# --- E7 paired comparisons
e7 = json.load(open(os.path.join(C.OUT, "a6_e7_empathetic.json")))
# rebuild per-item correctness from cache via the E7 script objects
import importlib.util
spec = importlib.util.spec_from_file_location("e7", os.path.join(os.path.dirname(__file__), "a6_e7_empathetic.py"))
# running the module again is cache-only (all calls cached) and prints its summary; capture its globals
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert C.NEW_CALLS[0] == 0
SEL, R, gold, G_OF, hier = m.SEL, m.R, m.gold, m.G_OF, m.hier
def corr(f): return np.array([f(c) for c in SEL], float)
V = {"flat_full": corr(lambda c: R[c]["full"]["flat"]["choice"] == gold[c]),
     "flat_first": corr(lambda c: R[c]["first"]["flat"]["choice"] == gold[c]),
     "flat_two": corr(lambda c: R[c]["two"]["flat"]["choice"] == gold[c]),
     "hier_greedy": corr(lambda c: hier(c, 1) == gold[c]),
     "hier_beam2": corr(lambda c: hier(c, 2) == gold[c]),
     "group_direct_full": corr(lambda c: R[c]["full"]["group"]["choice"] == G_OF[gold[c]]),
     "group_from_flat_full": corr(lambda c: G_OF[R[c]["full"]["flat"]["choice"]] == G_OF[gold[c]])}
rng = np.random.default_rng(0)
def boot(v):
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
def mcnemar(a, b):
    from math import comb
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum()); n = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"a_only": n01, "b_only": n10, "p_exact": p}
out["E7_acc_ci95"] = {k: {"acc": float(v.mean()), "ci95": boot(v)} for k, v in V.items()}
out["E7_paired"] = {"flat_full_vs_first": mcnemar(V["flat_full"], V["flat_first"]),
                    "flat_full_vs_hier_greedy": mcnemar(V["flat_full"], V["hier_greedy"]),
                    "flat_full_vs_hier_beam2": mcnemar(V["flat_full"], V["hier_beam2"]),
                    "group_from_flat_vs_group_direct": mcnemar(V["group_from_flat_full"], V["group_direct_full"])}
# --- E4 cost model: tokens ~ a + b * n_questions (pool order) with the 8-turn state
e4 = json.load(open(os.path.join(C.OUT, "a6_e4_fanout.json")))
xs = np.array([int(k) for k in e4["sizes"]]); ys = np.array([v["tokens_mean"] for v in e4["sizes"].values()])
b, a = np.polyfit(xs, ys, 1)
out["E4_token_model"] = {"intercept_tokens(state+overhead)": float(a), "tokens_per_question": float(b)}
print(C.dump("e10_stats.json", out)); print(json.dumps(out, indent=1))
