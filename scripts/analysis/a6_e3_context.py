"""E3: how many previous turns should the state carry? n = 0, 2, 8 (base), 20."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import a6_common as C

by = C.load_turns(); base = C.load_base()
S = C.sample_turns(by, "maichat", 60, min_ctx=20, seed=31, in_base=base) + C.sample_turns(by, "whatsapp_nl", 40, min_ctx=20, seed=32, in_base=base)
QD = C.AB.q_descriptive()
NS = (0, 2, 20)
items = [dict(state=C.d_state(conv, t, n), questions=QD, tag="e3_ctx", rep=0) for conv, t in S for n in NS]
res = C.run_many(items, 4)
print("new calls", C.NEW_CALLS[0], "turns", len(S))
R = []
for j, (conv, t) in enumerate(S):
    rr = res[3 * j:3 * j + 3]
    if not all(rr): continue
    d = {0: rr[0]["answers"], 2: rr[1]["answers"], 8: base[(t["corpus"], t["conv_id"], t["turn_idx"])]["D"], 20: rr[2]["answers"]}
    R.append(dict(corpus=t["corpus"], A=d, text=" / ".join(t["texts"])[:100], tok={n: rr[i]["usage"].get("input_tokens") for i, n in enumerate(NS)},
                  lat={n: rr[i]["latency_s"] for i, n in enumerate(NS)}))
out = {"n": len(R), "by_corpus_n": {c: sum(r["corpus"] == c for r in R) for c in ("maichat", "whatsapp_nl")}}
def dist(a, b):
    if a["type"] == "noul": return abs(a["noul"] - b["noul"])
    if a["type"] == "score": return abs(a["score"] - b["score"])
    return float(a["choice"] != b["choice"])
for corpus in ("all", "maichat", "whatsapp_nl"):
    RR = [r for r in R if corpus == "all" or r["corpus"] == corpus]
    tab = {}
    for q in QD:
        row = {}
        for n in (0, 2, 20):
            row[f"vs8_n{n}"] = float(np.mean([dist(r["A"][n][q], r["A"][8][q]) for r in RR]))
        row["n20_vs_n2"] = float(np.mean([dist(r["A"][20][q], r["A"][2][q]) for r in RR]))
        row["mean_by_n"] = {n: (float(np.mean([C.val(r["A"][n][q]) for r in RR])) if QD[q]["type"] != "choice" else None) for n in (0, 2, 8, 20)}
        if QD[q]["type"] != "noul":
            row["conf_by_n"] = {n: float(np.mean([r["A"][n][q]["confidence"] for r in RR])) for n in (0, 2, 8, 20)}
        if QD[q]["type"] == "choice":
            from collections import Counter
            row["top_by_n"] = {n: Counter(r["A"][n][q]["choice"] for r in RR).most_common(4) for n in (0, 8, 20)}
        tab[q] = row
    out[corpus] = tab
out["input_tokens_mean"] = {n: float(np.mean([r["tok"][n] for r in R])) for n in NS}
out["latency_p50"] = {n: float(np.median([r["lat"][n] for r in R])) for n in NS}
# relationship examples where n changes the answer
out["examples_relationship"] = [{"text": r["text"], **{f"n{n}": r["A"][n]["relationship"]["choice"] for n in (0, 2, 8, 20)}} for r in R if len({r["A"][n]["relationship"]["choice"] for n in (0, 2, 8, 20)}) > 2][:6]
out["examples_emotion"] = [{"text": r["text"], **{f"n{n}": r["A"][n]["emotion"]["choice"] for n in (0, 2, 8, 20)}} for r in R if r["A"][0]["emotion"]["choice"] != r["A"][8]["emotion"]["choice"]][:8]
print(C.dump("e3_context.json", out))
for corpus in ("all", "maichat", "whatsapp_nl"):
    print("==", corpus)
    for q, row in out[corpus].items():
        print(f"{q:14s}", " ".join(f"{k}:{v:.3f}" for k, v in row.items() if isinstance(v, float)), "| mean", {k: (round(v, 2) if v is not None else None) for k, v in row["mean_by_n"].items()}, "| conf", {k: round(v, 2) for k, v in row.get("conf_by_n", {}).items()})
for k in ("input_tokens_mean", "latency_p50", "examples_relationship", "examples_emotion"):
    print(k, out[k])
for q in ("relationship", "phase", "emotion"):
    print(q, out["all"][q]["top_by_n"])
