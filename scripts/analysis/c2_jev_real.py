"""c2 — roda o Jev 1 (e a cascata completa) nos turnos reais.
  nostate: state = 8 turnos + mensagem (+ intervalo de tempo), banco Jev 1 completo; paralelo.
  state:   cascata sequencial por janela: RelState por personagem (A->B e B->A), Jev 1 com estado + pendências +
           promessas, Jev 2 (Score descritivo + Choice de números) para as dimensões com Noul >= 0,3, física v0.
Saídas: data/processed/c2_jev_real_{nostate,state}.jsonl"""
import json, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from c2_common import jev  # noqa
from c2_common import (PROC, DIMS, jev1_bank, jev2_questions, make_state, RelState, DEFAULT_BASE, physics_step,
                       active_dims, jl_load, jl_append)

TURNS = [json.loads(l) for l in open(f"{PROC}/c2_real_turns.jsonl")]
REL_BASE = {"romantic_partners": {"affection": 0.75, "comfort": 0.75, "trust": 0.7},
            "flirting_or_crush": {"affection": 0.65}, "family": {"comfort": 0.7, "trust": 0.7, "protectiveness": 0.6},
            "close_friends": {"comfort": 0.7, "trust": 0.65, "playfulness": 0.6}}


def ctx_turns(r):
    return [{"from": h["from"], "text": h["text"]} for h in r["ctx"]]


def run_nostate():
    out = f"{PROC}/c2_jev_real_nostate.jsonl"
    done = {d["id"] for d in jl_load(out)}
    todo = [r for r in TURNS if r["id"] not in done]
    print("nostate todo", len(todo), flush=True)

    def one(r):
        st = make_state(r["C"], r["U"], ctx_turns(r), r["text"], with_state=False, gap_hours=r["gap_hours"])
        t0 = time.time()
        a = jev.ask(st, jev1_bank(r["C"], r["U"], 0, 0))
        return {"id": r["id"], "a1": a, "lat": round(time.time() - t0, 3)}
    B = 100
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(todo), B):
            recs = [x for x in ex.map(lambda r: _safe(one, r), todo[i:i + B]) if x]
            jl_append(out, recs)
            print("nostate", i + B, jev.summary(), flush=True)


def _safe(f, r):
    try:
        return f(r)
    except Exception as e:
        print("err", r["id"], str(e)[:200], flush=True)
        return None


def run_window(key):
    out = f"{PROC}/c2_jev_real_state.jsonl"
    rs = [r for r in TURNS if (r["conv"], r["win"]) == key]
    rs.sort(key=lambda r: r["pos"])
    base = dict(DEFAULT_BASE); base.update(REL_BASE.get(rs[0]["D_rel"], {}))
    rels = {"A": RelState(base), "B": RelState(base)}
    recs = []
    for r in rs:
        rel = rels[r["C"]]
        rel.decay(r["t_hours"])
        snap = {"v": dict(rel.v), "unresolved": [u["text"] for u in rel.unresolved],
                "promises": [p["text"] for p in rel.promises], "mode": rel.mode}
        st = make_state(r["C"], r["U"], ctx_turns(r), r["text"], rel=rel, with_state=True, gap_hours=r["gap_hours"])
        t0 = time.time()
        a1 = jev.ask(st, jev1_bank(r["C"], r["U"], len(rel.unresolved), len(rel.promises)))
        l1 = time.time() - t0
        act = active_dims(a1, 0.3)
        a2, l2 = None, 0.0
        if act:
            st2 = dict(st)
            st2["detected_effects"] = [f"user_message may {'raise' if dr == 'up' else 'lower'} {r['C']}'s {d} toward {r['U']}"
                                       for d, dr in act]
            t0 = time.time()
            a2 = jev.ask(st2, jev2_questions(r["C"], r["U"], act))
            l2 = time.time() - t0
        rec = physics_step(rel, r["C"], r["U"], r["text"], a1, a2, r["t_hours"], gap_hours=r["gap_hours"],
                           char_waiting=r["char_waiting"], mag="score")
        recs.append({"id": r["id"], "a1": a1, "a2": a2, "active": act, "snap": snap, "phys": rec,
                     "lat1": round(l1, 3), "lat2": round(l2, 3)})
    jl_append(out, recs)
    return key, len(recs)


def run_state():
    out = f"{PROC}/c2_jev_real_state.jsonl"
    done_ids = {d["id"] for d in jl_load(out)}
    keys = sorted({(r["conv"], r["win"]) for r in TURNS})
    keys = [k for k in keys if not all(r["id"] in done_ids for r in TURNS if (r["conv"], r["win"]) == k)]
    print("state windows todo", len(keys), flush=True)
    with ThreadPoolExecutor(4) as ex:
        for k, n in ex.map(lambda k: _safe_w(k), keys):
            print("window", k, n, jev.summary(), flush=True)


def _safe_w(k):
    try:
        return run_window(k)
    except Exception as e:
        print("err window", k, str(e)[:300], flush=True)
        return k, 0


if __name__ == "__main__":
    which = sys.argv[1]
    run_nostate() if which == "nostate" else run_state()
