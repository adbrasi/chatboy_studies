"""c2 — validação ponta a ponta nos dados reais: reexecuta a física sobre cada janela real com as detecções
(a) do OURO (rótulos do luna convertidos em respostas no formato do Jev) e (b) do Jev (a1/a2 da rodada com estado),
para várias especificações da física (v0 e variantes anti-deriva) e limiares de detecção.
Métricas (dev para escolher, teste para reportar):
  - taxa de movimentos aplicados por dimensão (Jev × ouro);
  - correlação de Spearman, entre personagens-janela, do estado final (Jev × ouro) por dimensão;
  - correlação por turno dos deltas (Jev × ouro);
  - saturação (fração de turnos com alguma dimensão > 0,95) e deriva (|final − base| médio);
Saída: analysis/data/c2_replay.json"""
import json, copy, itertools
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, pearsonr
from c2_common import (PROC, ADATA, DIMS, EVENTS, SPEC, RelState, DEFAULT_BASE, physics_step, jdump, jl_load, boot_fn)
from c2_jev_real import REL_BASE

T = [json.loads(l) for l in open(f"{PROC}/c2_real_turns.jsonl")]
G = {d["id"]: d["gold"] for d in jl_load(f"{PROC}/c2_gold_luna.jsonl") if d.get("gold")}
ST = {d["id"]: d for d in jl_load(f"{PROC}/c2_jev_real_state.jsonl")}


def gold_as_jev(g):
    a1, a2 = {}, {}
    evs = set(g.get("events") or [])
    for d in DIMS:
        dr, lv = g["dims"].get(d, ["none", 0])
        try:
            lv = int(lv)
        except Exception:
            lv = 0
        for x in ("up", "down"):
            on = dr == x and lv >= 1
            a1[f"d_{d}_{x}"] = {"type": "noul", "noul": 1.0 if on else 0.0}
            if on:
                a2[f"s_{d}_{x}"] = {"type": "score", "score": float(lv)}
                a2[f"n_{d}_{x}"] = {"type": "choice", "choice": str(min(100, 10 * lv if lv < 3 else 20 * lv))}
    for e in EVENTS:
        a1["e_" + e] = {"type": "noul", "noul": 1.0 if e in evs else 0.0}
    a1["e_left_waiting"] = {"type": "noul", "noul": 1.0}
    rep = 1.0 if ("apology" in evs or "promise_kept" in evs) else 0.0
    for i in range(5):
        a1[f"u_addr_{i}"] = {"type": "noul", "noul": rep}
        a1[f"p_kept_{i}"] = {"type": "noul", "noul": 1.0 if "promise_kept" in evs else 0.0}
        a1[f"p_broke_{i}"] = {"type": "noul", "noul": 1.0 if "cancel_or_broken_promise" in evs else 0.0}
    return a1, a2


def replay(rows, source, spec, thr_map=None):
    """rows: turnos de UMA janela, em ordem. source: 'gold' | 'jev'. thr_map: {dd: limiar} para o Jev (sobrescreve Nouls)."""
    base = dict(DEFAULT_BASE); base.update(REL_BASE.get(rows[0]["D_rel"], {}))
    rels = {"A": RelState(base, spec=spec), "B": RelState(base, spec=spec)}
    traj = []
    for r in rows:
        if source == "gold":
            a1, a2 = gold_as_jev(G[r["id"]])
        else:
            a1, a2 = ST[r["id"]]["a1"], ST[r["id"]]["a2"]
            if thr_map:
                a1 = dict(a1)
                for dd, th in thr_map.items():
                    k = f"d_{dd}"
                    if k in a1:  # re-escala: Noul >= th vira >= limiar da física
                        p = a1[k]["noul"]
                        a1[k] = {"type": "noul", "noul": 1.0 if p >= th else 0.0}
        rel = rels[r["C"]]
        rec = physics_step(rel, r["C"], r["U"], r["text"], a1, a2, r["t_hours"], gap_hours=r["gap_hours"],
                           char_waiting=r["char_waiting"], mag="score", spec=spec)
        traj.append({"id": r["id"], "C": r["C"], "before": rec["before"], "after": rec["after"], "mode": rec["mode"],
                     "n_applied": len(rec["applied"]), "unres": len(rec["unresolved"])})
    return traj, base


def windows(split):
    by = defaultdict(list)
    for r in T:
        if r["split"] == split and r["id"] in G and r["id"] in ST:
            by[(r["conv"], r["win"])].append(r)
    for k in by:
        by[k].sort(key=lambda r: r["pos"])
    return by


def compare(split, spec, thr_map=None):
    W = windows(split)
    finals_j, finals_g, dj, dg, sat_j, sat_g, drift_j, drift_g, modes_eq, conv_of = [], [], [], [], [], [], [], [], [], []
    moves_j = moves_g = 0
    per_turn = []
    for k, rows in W.items():
        tj, base = replay(rows, "jev", spec, thr_map)
        tg, _ = replay(rows, "gold", spec)
        for C in ("A", "B"):
            xj = [x for x in tj if x["C"] == C]; xg = [x for x in tg if x["C"] == C]
            if not xj:
                continue
            finals_j.append([xj[-1]["after"][d] for d in DIMS]); finals_g.append([xg[-1]["after"][d] for d in DIMS])
            conv_of.append(k[0])
            drift_j.append(np.mean([abs(xj[-1]["after"][d] - base[d]) for d in DIMS]))
            drift_g.append(np.mean([abs(xg[-1]["after"][d] - base[d]) for d in DIMS]))
        for a, b in zip(tj, tg):
            for d in DIMS:
                dj.append(a["after"][d] - a["before"][d]); dg.append(b["after"][d] - b["before"][d])
            sat_j.append(any(a["after"][d] > 0.95 or a["after"][d] < 0.05 for d in DIMS))
            sat_g.append(any(b["after"][d] > 0.95 or b["after"][d] < 0.05 for d in DIMS))
            modes_eq.append(a["mode"] == b["mode"])
            moves_j += a["n_applied"]; moves_g += b["n_applied"]
            per_turn.append({"conv": k[0], "eq": a["mode"] == b["mode"]})
    FJ, FG = np.array(finals_j), np.array(finals_g)
    rho = {}
    for i, d in enumerate(DIMS):
        if np.std(FJ[:, i]) > 1e-6 and np.std(FG[:, i]) > 1e-6:
            rho[d] = round(float(spearmanr(FJ[:, i], FG[:, i]).correlation), 3)
        else:
            rho[d] = None
    vals = [v for v in rho.values() if v is not None]
    return {"n_chars": len(FJ), "final_rho": rho, "final_rho_mean": round(float(np.mean(vals)), 3) if vals else None,
            "final_mae": round(float(np.mean(np.abs(FJ - FG))), 4),
            "delta_r": round(float(pearsonr(dj, dg)[0]), 3), "mode_agree": round(float(np.mean(modes_eq)), 3),
            "mode_agree_ci": boot_fn(lambda rs: float(np.mean([x["eq"] for x in rs])), per_turn, [x["conv"] for x in per_turn], iters=300),
            "sat_jev": round(float(np.mean(sat_j)), 3), "sat_gold": round(float(np.mean(sat_g)), 3),
            "drift_jev": round(float(np.mean(drift_j)), 4), "drift_gold": round(float(np.mean(drift_g)), 4),
            "moves_per_turn_jev": round(moves_j / len(sat_j), 3), "moves_per_turn_gold": round(moves_g / len(sat_g), 3)}


def variants():
    v = {}
    v["v0"] = copy.deepcopy(SPEC)
    s = copy.deepcopy(SPEC); s["habituation"] = True; v["v0+hab"] = s
    s = copy.deepcopy(SPEC); s["habituation"] = True; s["mood_decay_msg"] = 0.1; v["v0+hab+mood"] = s
    s = copy.deepcopy(SPEC); s["habituation"] = True; s["mood_decay_msg"] = 0.1; s["routine_gate"] = True; v["v1"] = s
    s = copy.deepcopy(v["v1"]); s["routine_thr"] = 0.7; v["v1+thr0.7"] = s
    s = copy.deepcopy(v["v1"]); s["routine_thr"] = 0.8; s["detect_thr"] = 0.6; v["v1+thr0.8"] = s
    return v


if __name__ == "__main__":
    import sys
    out = {"dev": {}, "test": {}}
    V = variants()
    for name, spec in V.items():
        out["dev"][name] = compare("dev", spec)
        print("dev", name, {k: out["dev"][name][k] for k in ("final_rho_mean", "final_mae", "delta_r", "mode_agree",
                                                            "sat_jev", "sat_gold", "moves_per_turn_jev", "moves_per_turn_gold")}, flush=True)
    # escolha no dev: maior correlação média do estado final com o ouro, penalizando saturação acima da do ouro
    def crit(r):
        return (r["final_rho_mean"] or 0) + r["mode_agree"] - max(0, r["sat_jev"] - r["sat_gold"])
    best = max(out["dev"], key=lambda n: crit(out["dev"][n]))
    out["chosen_on_dev"] = best
    for name in V:
        out["test"][name] = compare("test", V[name])
        print("test", name, {k: out["test"][name][k] for k in ("final_rho_mean", "final_mae", "delta_r", "mode_agree",
                                                              "sat_jev", "sat_gold")}, flush=True)
    jdump(out, f"{ADATA}/c2_replay.json")
    print("chosen", best)
