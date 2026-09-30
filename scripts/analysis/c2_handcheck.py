"""c2 — amostra para checagem manual (~60 casos). Estratificada SEM usar o ouro do luna (para não enviesar):
30 turnos com sinal forte de evento de relação no Jev sem estado (desculpa, cancelamento, insulto, desdém, defensivo,
afeto, vulnerabilidade, ciúme) + 30 aleatórios. Imprime contexto (4 turnos) e a mensagem; os meus rótulos ficam em
analysis/data/c2_handcheck_labels.json (escritos à mão) e a comparação em analysis/data/c2_handcheck.json.
Uso: python3 c2_handcheck.py sample | compare"""
import json, random, sys
import numpy as np
from c2_common import PROC, ADATA, DIMS, jl_load, jdump

T = {json.loads(l)["id"]: json.loads(l) for l in open(f"{PROC}/c2_real_turns.jsonl")}
NS = {d["id"]: d for d in jl_load(f"{PROC}/c2_jev_real_nostate.jsonl")}
SIG = ["apology", "cancel_or_broken_promise", "insult_criticism", "dismissive", "defensive", "affection_expr",
       "vulnerability", "other_person_jealousy", "hurtful_joke", "compliment", "practical_care"]
SAMPLE = f"{ADATA}/c2_handcheck_sample.json"


def sample():
    rnd = random.Random(11)
    ids = [i for i in T if i in NS]
    strong = [i for i in ids if max(NS[i]["a1"]["e_" + e]["noul"] for e in SIG) >= 0.8]
    s1 = rnd.sample(strong, 30)
    rest = [i for i in ids if i not in s1]
    s2 = rnd.sample(rest, 30)
    out = []
    for i in s1 + s2:
        t = T[i]
        out.append({"id": i, "stratum": "signal" if i in s1 else "random", "corpus": t["corpus"], "U": t["U"], "C": t["C"],
                    "gap_h": t["gap_hours"], "ctx": [f"{h['from']}: {h['text']}" for h in t["ctx"][-4:]], "msg": t["text"]})
    rnd.shuffle(out)
    jdump(out, SAMPLE)
    for k, o in enumerate(out):
        print(f"--- {k} {o['id']} U={o['U']} C={o['C']} gap={o['gap_h']}")
        for c in o["ctx"]:
            print("   ", c[:160].replace("\n", " / "))
        print("  >>", o["msg"][:300].replace("\n", " / "))


def lab_set(dims):
    return {f"{d}_{dr}" for d, (dr, lv) in dims.items() if dr != "none" and int(lv) >= 1}


def compare():
    S = json.load(open(SAMPLE))
    M = json.load(open(f"{ADATA}/c2_handcheck_labels.json"))
    G = {d["id"]: d["gold"] for d in jl_load(f"{PROC}/c2_gold_luna.jsonl") if d.get("gold")}
    ST = {d["id"]: d for d in jl_load(f"{PROC}/c2_jev_real_state.jsonl")}
    DD = [f"{d}_{dr}" for d in DIMS for dr in ("up", "down")]
    THR = json.load(open(f"{ADATA}/c2_eval_real.json"))["arch"]["DIM|com_estado"]["thr_dev"]
    rows = []
    for o in S:
        i = o["id"]
        if i not in G or str(i) not in M:
            continue
        me = set(M[str(i)])
        lu = lab_set(G[i]["dims"])
        jv = {dd for dd in DD if NS[i]["a1"][f"d_{dd}"]["noul"] >= 0.5}
        js = {dd for dd in DD if ST[i]["a1"][f"d_{dd}"]["noul"] >= 0.5}
        jt = {dd for dd in DD if ST[i]["a1"][f"d_{dd}"]["noul"] >= THR}
        rows.append((me, lu, jv, js, o["stratum"], jt))

    def agree(a_idx, b_idx, rs):
        # concordância por célula (turno x dimensão-direção) e kappa de Cohen
        ya = [dd in r[a_idx] for r in rs for dd in DD]; yb = [dd in r[b_idx] for r in rs for dd in DD]
        ya, yb = np.array(ya), np.array(yb)
        po = float(np.mean(ya == yb)); pa, pb = ya.mean(), yb.mean()
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        # só células "movidas" por alguém (evita inflar com os muitos zeros)
        mv = (ya | yb)
        jacc = float(np.mean((ya & yb)[mv])) if mv.any() else float("nan")
        return {"kappa": round(float(kappa), 3), "jaccard_moved": round(jacc, 3), "rate_a": round(float(pa), 3), "rate_b": round(float(pb), 3)}
    names = ["eu", "luna", "jev_sem_estado", "jev_com_estado", "stratum", f"jev_com_estado@{THR}"]
    out = {"n": len(rows)}
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (0, 5), (1, 5)]:
        out[f"{names[a]}~{names[b]}"] = agree(a, b, rows)
        out[f"{names[a]}~{names[b]}|signal"] = agree(a, b, [r for r in rows if r[4] == "signal"])
    jdump(out, f"{ADATA}/c2_handcheck.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"sample": sample, "compare": compare}[sys.argv[1]]()
