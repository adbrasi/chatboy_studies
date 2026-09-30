"""c2 — limiares por célula (dimensão × direção) do Jev 1, calibrados no DEV dos dados reais por casamento de taxa:
o menor limiar >= 0,5 tal que a taxa de disparo do Jev (com estado) não passe da taxa do ouro (luna). Células sem
positivos no ouro do dev ficam em 0,5. Motivo: o Jev dispara demais nas dimensões positivas "de rotina" (conforto↑ 82%
dos turnos vs 13% no ouro) e na taxa certa nas negativas (ressentimento↑ 1,5% vs 1,3%).
Saída: analysis/data/c2_cell_thr.json"""
import json
import numpy as np
from c2_common import PROC, ADATA, DIMS, jl_load, jdump

T = {json.loads(l)["id"]: json.loads(l) for l in open(f"{PROC}/c2_real_turns.jsonl")}
G = {d["id"]: d["gold"] for d in jl_load(f"{PROC}/c2_gold_luna.jsonl") if d.get("gold")}
S = {d["id"]: d for d in jl_load(f"{PROC}/c2_jev_real_state.jsonl")}
out, rep = {}, {}
for split in ("dev", "test"):
    ids = [i for i in T if T[i]["split"] == split and i in G and i in S]
    for d in DIMS:
        for dr in ("up", "down"):
            dd = f"{d}_{dr}"
            g = float(np.mean([G[i]["dims"][d][0] == dr and int(G[i]["dims"][d][1]) >= 1 for i in ids]))
            p = np.array([S[i]["a1"]["d_" + dd]["noul"] for i in ids])
            if split == "dev":
                th = 0.5
                if g > 0:
                    for cand in np.arange(0.5, 0.951, 0.01):
                        th = round(float(cand), 2)
                        if (p >= cand).mean() <= g:
                            break
                out[dd] = th
            rep.setdefault(dd, {})[split] = {"gold_rate": round(g, 4), "jev_rate@thr": round(float((p >= out[dd]).mean()), 4),
                                             "jev_rate@0.5": round(float((p >= 0.5).mean()), 4)}
jdump({"cell_thr": out, "rates": rep}, f"{ADATA}/c2_cell_thr.json")
for dd in out:
    print(dd.ljust(20), out[dd], rep[dd])
