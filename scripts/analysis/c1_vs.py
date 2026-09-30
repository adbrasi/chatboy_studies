"""c1 — (2) Verbalized Sampling (técnica A), no formato do paper (Zhang et al. 2025, arXiv 2510.01171, §G.3,
prompt de "Dialogue Simulation", versão human-decide: "Generate N plausible responses… Return responses as a JSON object
with the key "responses"… text… probability…").
Condições (todas com a mesma persona; |1 = com a nota do diretor, |0 = sem):
  VS3|1, VS5|1      N = 3 e 5 variações com probabilidade verbalizada, numa chamada
  SEED4|1           4 gerações independentes (seeds 0–3) da mesma condição livre+nota (o "gerar 4" do relatório 13)
  VS5|0, SEED4|0    idem sem nota (efeito "puro" do VS)
  VS5x|1            VS-5 dentro do melhor schema do experimento 1 (x = schema)
Estratégias de escolha (código): top (maior probabilidade), probw (sorteio ∝ probabilidade, o do paper), rand,
  filt_top (maior probabilidade entre as que passam nos filtros de código), lowpass (probabilidade média/baixa que passa
  nos filtros), viol (menor violação do orçamento, a regra do NCT do relatório 13), bank (menor P(LLM) do banco do Jev),
  bank_pass (menor P(LLM) entre as que passam nos filtros).
Uso: python3 c1_vs.py gen <split> [conds] | jev <split> [conds] | analyze <split>"""
import json, re, sys
from collections import Counter
import numpy as np
import c1_common as C
from b4_gen import target_brief

EXP = "vs"
VS_FREE = ("\n\nGenerate {n} plausible responses that you would naturally give to {u} as your next message, based on the chat "
           "history. If a response is made of several short messages, put one message per line inside the text.\n"
           "Return responses as a JSON object with the key \"responses\" (a list of dictionaries). Each dictionary must include:\n"
           "- text: the response string only (no explanation or extra text).\n"
           "- probability: the probability representing how likely each response would be (0.0 to 1.0).\n"
           "Give ONLY the JSON object, with no explanations or extra text.")
VS_LOG = ("\n\nGenerate {n} plausible continuations that you would naturally write next. Each continuation is {b}'s next "
          "message(s) only, written as log lines in exactly the same format as the log (one line per message).\n"
          "Return them as a JSON object with the key \"responses\" (a list of dictionaries). Each dictionary must include:\n"
          "- text: the continuation only (no explanation or extra text).\n"
          "- probability: the probability representing how likely each continuation would be (0.0 to 1.0).\n"
          "Give ONLY the JSON object, with no explanations or extra text.")


def brief_of(p):
    return target_brief(p, C.budget(p))


def cond_parts(cond):
    """'VS5|1' -> (5, 'free', True); 'VS5wa_time|1' -> (5, 'wa_time', True); 'seed2|1' -> (None, 'free', True)"""
    name, note = cond.split("|")
    m = re.match(r"VS(\d)(.*)", name)
    if m:
        return int(m.group(1)), (m.group(2) or "free"), note == "1"
    return None, "free", note == "1"


def spec(p, model, cond):
    n, schema, note = cond_parts(cond)
    br = brief_of(p) if note else None
    if n is None:  # seedK
        k = int(re.match(r"seed(\d)", cond).group(1))
        return dict(messages=C.schema_messages(p, "free", br), model=model, temperature=0.8, max_tokens=300, seed=k,
                    tag=f"c1:{cond}"), {"schema": "free", "note": int(note), "seed": k}
    if schema == "free":
        sysm = C.PERSONA + (f"\n\nDirector's note for your next message:\n{br}" if br else "") + VS_FREE.format(n=n, u=C.USER)
        msgs = C.schema_messages(p, "free", None, system_override=sysm)
    else:
        sysm = C.LOG_SYS.format(platform=C.PLATFORM[schema]) + VS_LOG.format(n=n, b=C.BOT)
        msgs = C.schema_messages(p, schema, br, system_override=sysm)
    return dict(messages=msgs, model=model, temperature=0.8, max_tokens=250 + 120 * n, seed=0, tag=f"c1:{cond}"), \
        {"schema": schema, "note": int(note), "n_req": n}


def parse_vs(raw):
    """-> [(text, prob)] tolerante: JSON inteiro, senão pares "text"/"probability" por regex."""
    t = C.FENCE.sub("", raw or "")
    m = re.search(r"\{.*\}", t, re.S)
    out = []
    if m:
        try:
            d = json.loads(m.group(0))
            for x in d.get("responses", []):
                if isinstance(x, dict) and x.get("text") is not None:
                    try:
                        pr = float(str(x.get("probability", "nan")).strip("%"))
                    except Exception:
                        pr = float("nan")
                    out.append((str(x["text"]), pr))
            if out:
                return out, True
        except Exception:
            pass
    for mm in re.finditer(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"probability"\s*:\s*"?([0-9.]+)', t, re.S):
        try:
            out.append((json.loads('"' + mm.group(1) + '"'), float(mm.group(2))))
        except Exception:
            pass
    return out, False


def post(p, model, cond, r, meta):
    n, schema, note = cond_parts(cond)
    if n is None:
        return C.parsed_record("free", r.get("text"), p)
    cands, ok = parse_vs(r.get("text"))
    cl = []
    for t, pr in cands:
        rec = C.parsed_record(schema, t, p)
        cl.append({"text": rec["text"], "prob": pr, "invented_user": rec["invented_user"], "broken": rec["broken"],
                   "plain": rec["plain"], "n_bubbles": rec["n_bubbles"], "prop_latency": rec["prop_latency"]})
    return {"cands": cl, "json_ok": ok, "n_cands": len(cl)}


def gen(split, conds, models=None):
    pts = C.split_pts(split)
    C.run_gen(EXP, pts, models or C.MODELS, conds, spec, post,
              empty_fn=lambda r: not (r.get("text") or "").strip())


# ---------------------------------------------------------------------- candidatas por (ponto, modelo, condição)
def candidates(p, model, cond, G):
    """Lista de candidatas {text, prob, ...}. SEED4 junta free|n do exp1 (seed 0) + seed1..3 deste exp."""
    n, schema, note = cond_parts(cond)
    if cond.startswith("SEED4"):
        nt = cond.split("|")[1]
        out = []
        r0 = G.get(("schema", model, f"free|{nt}", p["id"]))
        if r0:
            out.append({"text": r0.get("text") or "", "prob": None, "invented_user": r0.get("invented_user")})
        for k in (1, 2, 3):
            r = G.get((EXP, model, f"seed{k}|{nt}", p["id"]))
            if r:
                out.append({"text": r.get("text") or "", "prob": None, "invented_user": r.get("invented_user")})
        return out if len(out) == 4 else None
    r = G.get((EXP, model, cond, p["id"]))
    if not r:
        return None
    return [c for c in r.get("cands", []) if (c["text"] or "").strip()] or []


def violations(t, p, b):
    f = C.feats(t, p)
    return (C.lenerr(f["words"], b["words"]) + (f["q"] and not b["q"]) + (f["excl"] and not b["excl"]) +
            (f["emoji"] and not b["emoji"]) + (f["laugh"] and not b["laugh"]) + 0.5 * len(f["llmish_hits"]) + f["perf"] +
            f["recip"] + f["template"] + f["artefact"] + 2 * f["empty"])


def passes(t, p, b):
    f = C.feats(t, p)
    return not (f["empty"] or (f["q"] and not b["q"]) or (f["excl"] and not b["excl"]) or (f["emoji"] and not b["emoji"]) or
                (f["laugh"] and not b["laugh"]) or f["llmish"] or f["perf"] or f["recip"] or f["template"] or
                f["words"] > 2 * b["words"] + 2)


STRATS = ["top", "probw", "rand", "filt_top", "lowpass", "viol", "bank", "bank_pass"]


def choose(p, model, cond, cands, strat):
    if not cands:
        return None
    b = C.budget(p)
    probs = [c["prob"] if (c.get("prob") is not None and c["prob"] == c["prob"]) else None for c in cands]
    has_p = any(x is not None for x in probs)
    u = C.h01(p["id"], model, cond, strat)
    ok = [i for i, c in enumerate(cands) if passes(c["text"], p, b)]
    if strat == "top":
        return cands[int(np.argmax([x if x is not None else -1 for x in probs]))] if has_p else cands[0]
    if strat == "probw":
        if not has_p:
            return cands[int(u * len(cands))]
        w = np.array([max(x or 0, 1e-6) for x in probs]); w = w / w.sum()
        return cands[int(np.searchsorted(np.cumsum(w), u * 0.999999))]
    if strat == "rand":
        return cands[int(u * len(cands))]
    if strat in ("filt_top", "lowpass"):
        if not ok:
            return min(cands, key=lambda c: violations(c["text"], p, b))
        if strat == "filt_top":
            return cands[max(ok, key=lambda i: probs[i] if probs[i] is not None else -1)] if has_p else cands[ok[0]]
        if len(ok) == 1:
            return cands[ok[0]]
        if has_p:
            top = max(ok, key=lambda i: probs[i] if probs[i] is not None else -1)
            rest = [i for i in ok if i != top]
        else:
            rest = ok
        return cands[rest[int(u * len(rest))]]
    if strat == "viol":
        return min(cands, key=lambda c: violations(c["text"], p, b))
    if strat in ("bank", "bank_pass"):
        pool = [cands[i] for i in ok] if (strat == "bank_pass" and ok) else cands
        sc = [(C.bank_score(p, c["text"]), k) for k, c in enumerate(pool)]
        sc = [(s if s is not None else 9, k) for s, k in sc]
        return pool[min(sc)[1]]
    raise ValueError(strat)


def norm_txt(t):
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def jev(split, conds):
    pts = C.split_pts(split)
    G = C.load_gen()
    pairs = [(p, p["human"]) for p in pts]
    for m in C.MODELS:
        for c in conds:
            for p in pts:
                cs = candidates(p, m, c, G)
                for x in cs or []:
                    pairs.append((p, x["text"]))
    print("pairs", len(pairs))
    C.jev_eval(pairs, workers=4, tag=f"vs-{split}")


def analyze(split, conds):
    import c1_metrics as M
    pts = C.split_pts(split)
    G = C.load_gen()
    res = {"split": split, "human": M.human_block(pts), "models": {}}
    for m in C.MODELS:
        rows = []
        for c in conds:
            C_ = {p["id"]: candidates(p, m, c, G) for p in pts}
            if sum(v is not None for v in C_.values()) < 0.9 * len(pts):
                continue
            # diversidade e formato das candidatas
            nc, dist, dmov, jok, invc, probs_top, psum = [], [], [], [], [], [], []
            for p in pts:
                cs = C_[p["id"]]
                if cs is None:
                    continue
                nc.append(len(cs))
                if cs:
                    dist.append(len({norm_txt(x["text"]) for x in cs}) / len(cs))
                    mv = [(C.jev_get(p, x["text"]) or {}).get("e_move") for x in cs]
                    mv = [x for x in mv if x]
                    if mv:
                        dmov.append(len(set(mv)))
                    invc.append(np.mean([bool(x.get("invented_user")) for x in cs]))
                    pr = [x["prob"] for x in cs if x.get("prob") is not None and x["prob"] == x["prob"]]
                    if pr:
                        probs_top.append(max(pr)); psum.append(sum(pr))
                r = G.get((EXP, m, c, p["id"]))
                if r and "json_ok" in r:
                    jok.append(r["json_ok"])
            div = {"n_cands": float(np.mean(nc)), "distinct_frac": float(np.mean(dist)) if dist else None,
                   "distinct_moves": float(np.mean(dmov)) if dmov else None, "json_ok": float(np.mean(jok)) if jok else None,
                   "cand_invented": float(np.mean(invc)) if invc else None,
                   "p_top_mean": float(np.mean(probs_top)) if probs_top else None, "p_sum_mean": float(np.mean(psum)) if psum else None}
            res["models"].setdefault(m, {})[c] = {"diversity": div, "strats": {}}
            for s in STRATS:
                if c.startswith("SEED4") and s in ("top", "filt_top", "probw"):
                    continue
                items = []
                for p in pts:
                    cs = C_[p["id"]]
                    if cs is None:
                        continue
                    ch = choose(p, m, c, cs, s) if cs else None
                    r = G.get((EXP, m, c, p["id"])) or G.get(("schema", m, "free|" + c.split("|")[1], p["id"])) or {}
                    rec = {"cost": sum(((G.get(("schema", m, "free|" + c.split("|")[1], p["id"])) or {}).get("cost") or 0,
                                        *[((G.get((EXP, m, f"seed{k}|" + c.split("|")[1], p["id"])) or {}).get("cost") or 0) for k in (1, 2, 3)]))
                           if c.startswith("SEED4") else (r.get("cost") or 0),
                           "latency": r.get("latency"), "llm_calls": 4 if c.startswith("SEED4") else r.get("llm_calls", 1),
                           "fallback": ch is None}
                    t = ch["text"] if ch else ((G.get(("schema", m, "free|" + c.split("|")[1], p["id"])) or {}).get("text") or "")
                    items.append({"p": p, "text": t, "rec": rec})
                o = M.summarize(items, boot=True)
                res["models"][m][c]["strats"][s] = o
                rows.append((f"{c}:{s}", o))
        print("==", m)
        M.table(rows, keys=("D", "lenerr", "words_med", "q", "excl", "emoji", "laugh", "llmish", "echo2", "multi", "fallback",
                            "move_match", "move_entropy", "coh", "bank"))
        for c, d in res["models"].get(m, {}).items():
            print("  div", c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d["diversity"].items()})
    C.jdump(f"c1_vs_{split}.json", res)
    return res


if __name__ == "__main__":
    a = sys.argv
    conds = a[3].split(",") if len(a) > 3 else ["VS3|1", "VS5|1", "seed1|1", "seed2|1", "seed3|1", "VS5|0", "seed1|0", "seed2|0", "seed3|0"]
    if a[1] == "gen":
        gen(a[2], conds, a[4].split(",") if len(a) > 4 else None)
    elif a[1] == "jev":
        jev(a[2], conds)
    elif a[1] == "analyze":
        analyze(a[2], conds)
