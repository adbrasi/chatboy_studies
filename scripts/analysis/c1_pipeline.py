"""c1 — (5) Pipeline combinado, escolhido no DEV e validado no TESTE.
Condições comparadas (4 atores):
  A        a LLM pura (persona mínima + histórico; = schema free|0 do exp. 1)
  NCT+N    o melhor pipeline do relatório 13 (só medido no dev lá): 4× briefing T (seeds 0–3) → escolhe em código a de menor
           violação do orçamento → normalizador. Regerado aqui com o MESMO corpo de chamada do b4 (o dev sai do cache).
  P:<nome> os pipelines deste laboratório (configuração em CONFIGS): cabeçalho de RP + ficha situacional, nota do diretor
           (posição), schema, N variações com probabilidade verbalizada (VS) ou seeds, estratégia de escolha (filtros de
           código + banco atômico do Jev) e normalizador de código.
  humano   a resposta real.
Custo e latência por resposta: LLM (medido) + Jev (leitura do a9, ~1 chamada, + 1 chamada do banco por candidata, em
paralelo) + código.
Uso: python3 c1_pipeline.py gen <split> [configs] | jev <split> [configs] | analyze <split> [configs]"""
import json, re, sys
import numpy as np
import c1_common as C
import c1_vs as V
import c1_header as H
from b4_common import STATIC
from b4_gen import target_brief

EXP = "pipe"
# gerações (uma chamada por registro). "hdr_end#k" = cabeçalho de RP + ficha situacional + nota do diretor no FIM, seed k
# (a seed 0 é o próprio H1e do experimento 4); "hdr_end_vs5" = idem, com a instrução de VS-5 junto da nota no fim.
GENS = {"hdr_end": dict(n=None), "hdr_end_vs5": dict(n=5), "hdr0_vs5": dict(n=5, note=False)}
CONFIGS = {
    # nome: geração, nº de seeds (se não for VS), estratégia de escolha
    "P1_hdr": dict(gen="hdr_end", k=1, strat="first"),
    "P4_hdr_seed4_viol": dict(gen="hdr_end", k=4, strat="viol"),
    "P4_hdr_seed4_bank": dict(gen="hdr_end", k=4, strat="bank_pass"),
    "P5_hdr_vs5_top": dict(gen="hdr_end_vs5", strat="top"),
    "P5_hdr_vs5_low": dict(gen="hdr_end_vs5", strat="lowpass"),
    "P5_hdr_vs5_viol": dict(gen="hdr_end_vs5", strat="viol"),
    "P5_hdr_vs5_bank": dict(gen="hdr_end_vs5", strat="bank_pass"),
    # sem nota do diretor (o orçamento do Jev só entra nos filtros de código e no normalizador)
    "P6_hdr0_vs5_low": dict(gen="hdr0_vs5", strat="lowpass"),
    "P6_hdr0_vs5_viol": dict(gen="hdr0_vs5", strat="viol"),
    "P6_hdr0_vs5_bank": dict(gen="hdr0_vs5", strat="bank_pass"),
}


def brief_of(p):
    return target_brief(p, C.budget(p))


def messages(p, gname):
    g = GENS[gname]
    br = brief_of(p)
    sysm = H.header_system(p) if g.get("header", True) else C.PERSONA
    m = C.schema_messages(p, "free", None, system_override=sysm)
    tail = f"DIRECTOR NOTE (this turn only):\n{br}" if g.get("note", True) else ""
    if g["n"]:
        tail += "\n" + V.VS_FREE.format(n=g["n"], u=C.USER)
    tail = tail.strip()
    m.append({"role": "system", "content": tail})
    return m


def spec(p, model, cond):
    """cond = 'T#k' (briefing T do b4, seed k) | 'hdr_end#k' | 'hdr_end_vs5'."""
    if cond.startswith("T#"):
        k = int(cond[2:])
        b = C.budget(p)
        msgs = C.B4.msgs(p, STATIC + "\n\nFor your next message:\n" + target_brief(p, b))
        return dict(messages=msgs, model=model, temperature=0.8, max_tokens=300, seed=k, tag="c1:b4T"), {"schema": "free"}
    gname, _, k = cond.partition("#")
    g = GENS[gname]
    if g["n"]:
        return dict(messages=messages(p, gname), model=model, temperature=0.8, max_tokens=250 + 120 * g["n"], seed=0,
                    tag=f"c1:{gname}"), {"schema": "free", "n_req": g["n"]}
    return dict(messages=messages(p, gname), model=model, temperature=0.8, max_tokens=300, seed=int(k), tag=f"c1:{gname}"), \
        {"schema": "free"}


def post(p, model, cond, r, meta):
    if "n_req" in meta:
        cands, ok = V.parse_vs(r.get("text"))
        cl = []
        for t, pr in cands:
            rec = C.parsed_record(meta["schema"], t, p)
            cl.append({"text": rec["text"], "prob": pr, "invented_user": rec["invented_user"], "broken": rec["broken"],
                       "plain": rec["plain"], "n_bubbles": rec["n_bubbles"], "prop_latency": rec["prop_latency"]})
        return {"cands": cl, "json_ok": ok, "n_cands": len(cl)}
    return C.parsed_record(meta["schema"], r.get("text"), p)


def gen_conds(configs):
    out = ["T#0", "T#1", "T#2", "T#3"]
    for c in configs:
        cf = CONFIGS[c]
        if GENS[cf["gen"]]["n"]:
            out.append(cf["gen"])
        else:
            out += [f"{cf['gen']}#{k}" for k in range(1, cf["k"])]  # a seed 0 vem do H1e
    return list(dict.fromkeys(out))


def gen(split, configs):
    C.run_gen(EXP, C.split_pts(split), C.MODELS, gen_conds(configs), spec, post,
              empty_fn=lambda r: not (r.get("text") or "").strip())


def candidates(p, model, cfgname, G):
    cf = CONFIGS[cfgname]
    if GENS[cf["gen"]]["n"]:
        r = G.get((EXP, model, cf["gen"], p["id"]))
        if not r:
            return None, None
        return [c for c in r.get("cands", []) if (c["text"] or "").strip()], [r]
    rs = [G.get(("header", model, "H1e", p["id"]))] + [G.get((EXP, model, f"{cf['gen']}#{k}", p["id"])) for k in range(1, cf["k"])]
    if not all(rs):
        return None, None
    return [{"text": r.get("text") or "", "prob": None, "invented_user": r.get("invented_user")} for r in rs
            if (r.get("text") or "").strip()], rs


def final(p, model, cond, G, stage="final"):
    """-> (texto final, registro de custo/latência) para A, NCT+N ou uma config P."""
    b = C.budget(p)
    if cond == "A":
        r = G.get(("schema", model, "free|0", p["id"]))
        return (r.get("text") or "") if r else None, r
    if cond in ("NCT", "NCT+N"):
        rs = [G.get((EXP, model, f"T#{k}", p["id"])) for k in range(4)]
        if not all(rs):
            return None, None
        import b4_post
        ok = [r for r in rs if (r.get("text") or "").strip()] or rs
        best = min(ok, key=lambda r: b4_post.violations(r.get("text") or "", p, b))
        t = best.get("text") or ""
        rec = {"cost": sum(r.get("cost") or 0 for r in rs), "latency": max(r.get("latency") or 0 for r in rs),
               "llm_calls": sum(r.get("llm_calls", 1) for r in rs), "jev_calls": 1,
               "invented_user": any(r.get("invented_user") for r in rs)}
        return (C.normalize(t, b) if cond == "NCT+N" else t), rec
    name = cond.replace("+N", "").replace("-N", "")
    cands, rs = candidates(p, model, name, G)
    if cands is None:
        return None, None
    rec = {"cost": sum(r.get("cost") or 0 for r in rs), "latency": max(r.get("latency") or 0 for r in rs),
           "llm_calls": sum(r.get("llm_calls", 1) for r in rs), "jev_calls": 1 + len(cands),
           "fallback": not cands, "invented_user": any(c.get("invented_user") for c in cands)}
    if not cands:  # nada utilizável: cai na LLM pura
        r = G.get(("schema", model, "free|0", p["id"])) or {}
        return C.normalize(r.get("text") or "", b), rec
    st = CONFIGS[name]["strat"]
    ch = cands[0] if st == "first" else V.choose(p, model, name, cands, st)
    t = ch["text"]
    if stage == "pre":
        return t, rec
    return (t if cond.endswith("-N") else C.normalize(t, b)), rec


def all_texts(split, conds):
    pts = C.split_pts(split)
    G = C.load_gen()
    pairs = [(p, p["human"]) for p in pts]
    for m in C.MODELS:
        for c in conds:
            for p in pts:
                name = c.replace("+N", "").replace("-N", "")
                if name in CONFIGS:
                    cands, _ = candidates(p, m, name, G)
                    pairs += [(p, x["text"]) for x in (cands or [])]
    return pts, G, pairs


def jev(split, configs, finals=None):
    pts, G, pairs = all_texts(split, configs)
    C.jev_eval(pairs, workers=4, tag=f"pipe-cands-{split}")  # 1º: candidatas (a escolha pelo banco depende delas)
    G = C.load_gen()
    fin = []
    finals = finals or (["A", "NCT+N"] + [x + "+N" for x in configs] + [x + "-N" for x in configs])
    for m in C.MODELS:
        for c in finals:
            for p in pts:
                t, _ = final(p, m, c, G)
                if t and t.strip():
                    fin.append((p, t))
    C.jev_eval(fin, workers=4, tag=f"pipe-final-{split}")


def analyze(split, configs):
    import c1_metrics as M
    pts = C.split_pts(split)
    G = C.load_gen()
    jl = [x for x in C.jev.stats["latency_s"]] or [0.55]
    res = {"split": split, "human": M.human_block(pts), "models": {}}
    conds = ["A", "NCT+N"] + [x + "-N" for x in configs] + [x + "+N" for x in configs]
    for m in C.MODELS:
        rows = []
        for c in conds:
            items = []
            for p in pts:
                t, rec = final(p, m, c, G)
                if t is None:
                    continue
                items.append({"p": p, "text": t, "rec": rec})
            if len(items) < 0.9 * len(pts):
                continue
            o = M.summarize(items, boot=True)
            o["jev_calls"] = float(np.mean([(it["rec"] or {}).get("jev_calls", 0) for it in items]))
            res["models"].setdefault(m, {})[c] = o
            rows.append((c, o))
        print("==", m)
        M.table(rows, keys=("D", "lenerr", "words_med", "q", "excl", "emoji", "laugh", "llmish", "echo2", "multi", "fallback",
                            "move_match", "move_entropy", "coh", "bank", "cost_per_resp", "lat_p50"))
    C.jdump(f"c1_pipeline_{split}.json", res)
    return res


if __name__ == "__main__":
    a = sys.argv
    cfgs = a[3].split(",") if len(a) > 3 else list(CONFIGS)
    if a[1] == "jev":
        jev(a[2], cfgs, a[4].split(",") if len(a) > 4 else None)
    else:
        {"gen": gen, "analyze": analyze}[a[1]](a[2], cfgs)
