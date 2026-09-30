"""c2 — checagem manual da postura (~40 casos). Amostra estratificada (sem olhar o juiz): 40 pares persona × mensagem,
cada um com as respostas A (sem briefing) e B (postura do Jev) de um ator sorteado. Imprime para eu rotular; os rótulos
(escritos à mão) ficam em analysis/data/c2_posture_hand_labels.json no formato {"<key>": {"A": [fiel, vicio], "B": [...]}}
e a comparação com os Nouls do Jev em analysis/data/c2_posture_hand.json.
Uso: python3 c2_posture_hand.py sample | compare"""
import json, random, sys
import numpy as np
from c2_common import ADATA, PROC, MODELS, jl_load, jdump
from c2_posture import PERSONAS, MSGS

SAMPLE = f"{ADATA}/c2_posture_hand_sample.json"
LAB = f"{ADATA}/c2_posture_hand_labels.json"


def sample():
    rnd = random.Random(5)
    G = {(g["actor"], g["pk"], g["mi"], g["cond"]): g["reply"] for g in jl_load(f"{PROC}/c2_posture_gen.jsonl")}
    D = json.load(open(f"{ADATA}/c2_posture_decisions.json"))
    cells = [(pk, mi) for pk in PERSONAS for mi in range(len(MSGS))]
    pick = rnd.sample(cells, 40)
    out = []
    for pk, mi in pick:
        a = rnd.choice(list(MODELS))
        out.append({"key": f"{a}|{pk}|{mi}", "actor": a, "pk": pk, "mi": mi, "type": MSGS[mi][0], "msg": MSGS[mi][1],
                    "posture": D[f"{pk}|{mi}"]["posture"], "A": G.get((a, pk, mi, "A")), "B": G.get((a, pk, mi, "B"))})
    jdump(out, SAMPLE)
    for o in out:
        print(f"--- {o['key']} [{o['type']}] {PERSONAS[o['pk']]['name']} <- \"{o['msg']}\"  (Jev: {o['posture']})")
        print("  A:", (o["A"] or "").replace("\n", " / ")[:260])
        print("  B:", (o["B"] or "").replace("\n", " / ")[:260])


def compare():
    S = json.load(open(SAMPLE)); L = json.load(open(LAB))
    J = {(d["actor"], d["pk"], d["mi"], d["cond"]): d["j"] for d in jl_load(f"{PROC}/c2_posture_judge.jsonl")}
    res = {"n": 0}
    rows = []
    for o in S:
        if o["key"] not in L:
            continue
        for c in "AB":
            me = L[o["key"]][c]
            j = J.get((o["actor"], o["pk"], o["mi"], c))
            if j is None:
                continue
            rows.append({"cond": c, "me_f": me[0], "me_v": me[1], "j_f": j["faithful"], "j_v": j["assistant_vice"]})
    res["n"] = len(rows)
    for c in "AB":
        rs = [r for r in rows if r["cond"] == c]
        res[c] = {"me_faithful": round(float(np.mean([r["me_f"] for r in rs])), 3),
                  "jev_faithful": round(float(np.mean([r["j_f"] >= 0.5 for r in rs])), 3),
                  "me_vice": round(float(np.mean([r["me_v"] for r in rs])), 3),
                  "jev_vice": round(float(np.mean([r["j_v"] >= 0.5 for r in rs])), 3)}
    for k, (a, b) in {"faithful": ("me_f", "j_f"), "vice": ("me_v", "j_v")}.items():
        ya = np.array([r[a] for r in rows]); yb = np.array([r[b] >= 0.5 for r in rows]).astype(int)
        po = float(np.mean(ya == yb)); pa, pb = ya.mean(), yb.mean(); pe = pa * pb + (1 - pa) * (1 - pb)
        res[f"agree_{k}"] = {"acc": round(po, 3), "kappa": round(float((po - pe) / (1 - pe)), 3) if pe < 1 else None}
    jdump(res, f"{ADATA}/c2_posture_hand.json")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"sample": sample, "compare": compare}[sys.argv[1]]()
