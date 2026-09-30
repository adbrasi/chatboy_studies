"""E4: many questions per call (1/10/40/100/all), isolation check, latency & cost; parallel split of a full briefing."""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import a6_common as C
from a6_pool import pool
from concurrent.futures import ThreadPoolExecutor

by = C.load_turns(); base = C.load_base()
S = C.sample_turns(by, "maichat", 8, seed=41, in_base=base) + C.sample_turns(by, "whatsapp_nl", 4, seed=42, in_base=base)
P = pool(); keys = list(P); NQ = len(keys)
SIZES = [1, 10, 40, 100, NQ]
sub = lambda ks: {k: P[k] for k in ks}

# A) nested sizes, 2 reps
items = [dict(state=C.d_state(conv, t, 8), questions=sub(keys[:n]), tag="e4_size", rep=r) for conv, t in S for n in SIZES for r in (0, 1)]
# B) each of the 10 core questions alone
items += [dict(state=C.d_state(conv, t, 8), questions=sub([k]), tag="e4_single", rep=0) for conv, t in S for k in keys[:10]]
res = C.run_many(items, 4)
print("new calls A+B", C.NEW_CALLS[0])
nA = len(S) * len(SIZES) * 2
A = {}
for j, (conv, t) in enumerate(S):
    for i, n in enumerate(SIZES):
        for r in (0, 1):
            A[(j, n, r)] = res[j * len(SIZES) * 2 + i * 2 + r]
Bsingle = {(j, k): res[nA + j * 10 + i] for j in range(len(S)) for i, k in enumerate(keys[:10])}

def d(a, b):
    if a["type"] == "noul": return abs(a["noul"] - b["noul"])
    if a["type"] == "score": return abs(a["score"] - b["score"])
    return float(a["choice"] != b["choice"])

out = {"n_turns": len(S), "n_questions_pool": NQ, "sizes": {}}
for n in SIZES:
    lat = [A[(j, n, r)]["latency_s"] for j in range(len(S)) for r in (0, 1)]
    tok = [A[(j, n, 0)]["usage"]["input_tokens"] for j in range(len(S))]
    cost = [A[(j, n, 0)]["usage"].get("cost", 0) for j in range(len(S))]
    out["sizes"][n] = {"lat_p50": float(np.median(lat)), "lat_p90": float(np.percentile(lat, 90)), "lat_max": float(max(lat)),
                       "tokens_mean": float(np.mean(tok)), "cost_mean_usd": float(np.mean(cost)), "cost_per_question_usd": float(np.mean(cost)) / n}
# answer drift of the shared questions vs the smallest size containing them, compared with rep-to-rep noise
drift = {}
for n in SIZES[1:]:
    ks = keys[:10]
    dd = [d(A[(j, n, 0)]["answers"][k], A[(j, 10, 0)]["answers"][k]) for j in range(len(S)) for k in ks if A[(j, n, 0)]["answers"][k]["type"] == "noul"]
    rep = [d(A[(j, n, 0)]["answers"][k], A[(j, n, 1)]["answers"][k]) for j in range(len(S)) for k in ks if A[(j, n, 0)]["answers"][k]["type"] == "noul"]
    ch = [d(A[(j, n, 0)]["answers"][k], A[(j, 10, 0)]["answers"][k]) for j in range(len(S)) for k in ks if A[(j, n, 0)]["answers"][k]["type"] == "choice"]
    sc = [d(A[(j, n, 0)]["answers"][k], A[(j, 10, 0)]["answers"][k]) for j in range(len(S)) for k in ks if A[(j, n, 0)]["answers"][k]["type"] == "score"]
    drift[n] = {"noul_meanabs_vs_size10": float(np.mean(dd)), "noul_meanabs_rep_noise": float(np.mean(rep)), "choice_flip_vs10": float(np.mean(ch)), "score_meanabs_vs10": float(np.mean(sc))}
single = [d(Bsingle[(j, k)]["answers"][k], A[(j, NQ, 0)]["answers"][k]) for j in range(len(S)) for k in keys[:10] if P[k]["type"] == "noul"]
single_c = [d(Bsingle[(j, k)]["answers"][k], A[(j, NQ, 0)]["answers"][k]) for j in range(len(S)) for k in keys[:10] if P[k]["type"] == "choice"]
single_s = [d(Bsingle[(j, k)]["answers"][k], A[(j, NQ, 0)]["answers"][k]) for j in range(len(S)) for k in keys[:10] if P[k]["type"] == "score"]
drift["single_vs_full"] = {"noul_meanabs": float(np.mean(single)), "noul_max": float(np.max(single)), "choice_flip": float(np.mean(single_c)), "score_meanabs": float(np.mean(single_s))}
lat1 = [Bsingle[(j, k)]["latency_s"] for j in range(len(S)) for k in keys[:10]]
cost1 = [Bsingle[(j, k)]["usage"].get("cost", 0) for j in range(len(S)) for k in keys[:10]]
out["ten_separate_calls"] = {"lat_p50_each": float(np.median(lat1)), "cost_sum_per_turn": float(np.sum(cost1)) / len(S),
                             "vs_one_call_of_10_cost": out["sizes"][10]["cost_mean_usd"]}
out["drift"] = drift

# C) parallel split of the full briefing: 1xNQ, 2xNQ/2, 4xNQ/4 launched simultaneously; wall-clock
def split(ks, m):
    return [ks[i::m] for i in range(m)]
wall = {1: [], 2: [], 4: []}
calls_lat = {1: [], 2: [], 4: []}
for rep in (0, 1):
    for j, (conv, t) in enumerate(S):
        st = C.d_state(conv, t, 8)
        for m in (1, 2, 4):
            its = [dict(state=st, questions=sub(g), tag=f"e4_par{m}", rep=rep) for g in split(keys, m)]
            rr = C.run_many(its, 4)
            if not all(rr): continue
            wall[m].append(max(r["t1"] for r in rr) - min(r["t0"] for r in rr))
            calls_lat[m] += [r["latency_s"] for r in rr]
out["parallel_full_briefing"] = {m: {"wall_p50": float(np.median(w)), "wall_p90": float(np.percentile(w, 90)), "wall_max": float(max(w)),
                                     "call_lat_p50": float(np.median(calls_lat[m])), "n": len(w)} for m, w in wall.items()}
print("new calls after C", C.NEW_CALLS[0])

# D) the counterfactual: every question its own call (one turn), 4 workers
conv, t = S[0]; st = C.d_state(conv, t, 8)
its = [dict(state=st, questions=sub([k]), tag="e4_each", rep=0) for k in keys]
t0 = time.time(); rr = C.run_many(its, 4); rr = [r for r in rr if r]
out["one_call_per_question"] = {"n_calls": len(rr), "wall_s_sequential_4workers": float(max(r["t1"] for r in rr) - min(r["t0"] for r in rr)),
                                "cost_total": float(sum(r["usage"].get("cost", 0) for r in rr)), "lat_p50": float(np.median([r["latency_s"] for r in rr])),
                                "cost_one_call_all": float(A[(0, NQ, 0)]["usage"].get("cost", 0)),
                                "noul_meanabs_vs_full_call": float(np.mean([d(r["answers"][k], A[(0, NQ, 0)]["answers"][k]) for r, k in zip(rr, keys) if P[k]["type"] == "noul"]))}
print("new calls total", C.NEW_CALLS[0])
print(C.dump("e4_fanout.json", out))
print(json.dumps(out, indent=1))
