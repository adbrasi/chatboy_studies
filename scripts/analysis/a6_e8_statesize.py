"""E8: latency/cost vs state size (8 / 40 / 150 / 400 previous turns, whatsapp_nl, sessions ignored) with the 17 base questions."""
import json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import a6_common as C

by = C.load_turns(("whatsapp_nl",))
QD = C.AB.q_descriptive()
rnd = random.Random(8)
convs = [v for v in by.values() if len(v) > 450]
S = [(c, c[rnd.randrange(420, len(c))]) for c in rnd.sample(convs, min(12, len(convs)))]
NS = (8, 40, 150, 400)
def st(conv, t, n):
    i = conv.index(t)
    return {"previous_turns": [C.AB.fmt_turn(x) for x in conv[i - n:i]], "current_turn": C.AB.fmt_turn(t)}
items = [dict(state=st(c, t, n), questions=QD, tag="e8_size") for c, t in S for n in NS]
res = C.run_many(items, 4)
print("new calls", C.NEW_CALLS[0], "turns", len(S))
out = {"n": len(S), "sizes": {}}
def d(a, b):
    if a["type"] == "noul": return abs(a["noul"] - b["noul"])
    if a["type"] == "score": return abs(a["score"] - b["score"])
    return float(a["choice"] != b["choice"])
for i, n in enumerate(NS):
    rr = [res[j * len(NS) + i] for j in range(len(S))]
    r8 = [res[j * len(NS)] for j in range(len(S))]
    ok = [(a, b) for a, b in zip(rr, r8) if a and b]
    out["sizes"][n] = {"tokens_mean": float(np.mean([a["usage"]["input_tokens"] for a, _ in ok])),
                       "lat_p50": float(np.median([a["latency_s"] for a, _ in ok])), "lat_max": float(max(a["latency_s"] for a, _ in ok)),
                       "cost_mean": float(np.mean([a["usage"].get("cost", 0) for a, _ in ok])),
                       "noul_meanabs_vs8": float(np.mean([d(a["answers"][k], b["answers"][k]) for a, b in ok for k in QD if QD[k]["type"] == "noul"])),
                       "choice_flip_vs8": float(np.mean([d(a["answers"][k], b["answers"][k]) for a, b in ok for k in QD if QD[k]["type"] == "choice"])),
                       "topic_shift_mean": float(np.mean([a["answers"]["topic_shift"]["noul"] for a, _ in ok])),
                       "emotion_conf": float(np.mean([a["answers"]["emotion"]["confidence"] for a, _ in ok]))}
print(C.dump("e8_statesize.json", out)); print(json.dumps(out, indent=1))
