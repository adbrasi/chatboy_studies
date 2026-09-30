"""b4 — mecanismo 2: PODA sentence-level da resposta livre (A), guiada pelo Jev × por regra; + normalizador de código.
A LLM gera livre (condição A). O texto é quebrado em unidades (frases/linhas). O Jev responde, numa chamada por
ponto, 4 Nouls por unidade + 1 Choice "se só pudesse mandar uma" (opções embaralhadas contra viés de posição).
O código monta a resposta final. Saídas (condições derivadas, gravadas em gen.jsonl):
  N        A + normalizador de código (sem poda)
  R1..R5   poda por regra (1ª unidade; última; tira perguntas; 1ªs unidades sem pergunta até o orçamento; idem + 1 pergunta se o orçamento pede)
  J1       só a unidade escolhida pelo Choice do Jev
  J2       unidades "essenciais" e sem vício pelos Nouls (limiar th), cortadas pelo orçamento
  J3       J2 + política de pergunta do orçamento (no máx. 1 pergunta, e só se o orçamento pede e ela não é genérica)
Todas as podas passam também pelo normalizador (sufixo +N na análise).
Uso: python3 b4_prune.py <split> <modelos> [th]"""
import json, sys
import jev
from jev import noul, choice
from b4_common import (load_points, load_gen, append_gen, budget, units, join_units, words, normalize, h01, BOT, USER)
from a9_common import ask_many_timed

CTX = 6


def jstate(p, us):
    h = p["history"][-CTX:]
    return {"conversation": [{"from": x["who"], "text": x["text"]} for x in h[:-1]],
            "last_message": {"from": USER, "text": h[-1]["text"]},
            "draft_reply": {f"u{i+1}": u for i, u in enumerate(us)}}


def jquestions(p, us):
    Q = {}
    for i, u in enumerate(us):
        k = f"u{i+1}"
        Q[f"ess_{k}"] = noul(f"Is `draft_reply.{k}` {BOT}'s actual response to `last_message` (the reaction, answer or comment "
                             f"itself), rather than an add-on?")
        Q[f"gq_{k}"] = noul(f"Is `draft_reply.{k}` a follow-up question whose main purpose is to keep the conversation going "
                            f"(for example 'what about you?', 'how was your day?', 'what are you up to?', 'any plans?')?")
        Q[f"val_{k}"] = noul(f"Does `draft_reply.{k}` repeat or paraphrase what {USER} said, or validate, praise, reassure or "
                             f"sympathize with {USER}?")
        Q[f"ent_{k}"] = noul(f"Is `draft_reply.{k}` mostly enthusiasm or filler (like 'That's amazing!', 'Haha love that', "
                             f"'Omg yes') without adding new content?")
    order = sorted(range(len(us)), key=lambda i: h01(p["id"], "ord", i))  # embaralha as opções do Choice
    Q["keep1"] = choice(f"If {BOT} could send only ONE of the parts in `draft_reply`, which one would a real friend texting "
                        f"casually send?", {f"u{i+1}": us[i] for i in order})
    return Q


def cap(us, b, factor=1.5):
    out, n = [], 0
    lim = max(b["words"] * factor, 3)
    for u in us:
        w = len(words(u))
        if out and n + w > lim:
            break
        out.append(u); n += w
    return out


def isq(u):
    return "?" in u


def rules(us, b):
    nq = [u for u in us if not isq(u)] or us[:1]
    r4 = cap(nq, b)
    r5 = list(r4)
    if b["q"]:
        qs = [u for u in us if isq(u)]
        if qs and qs[-1] not in r5:
            r5 = r5 + [qs[-1]]
    return {"R1": us[:1], "R2": us[-1:], "R3": nq, "R4": r4, "R5": r5}


def jev_prune(us, a, b, th=0.5):
    g = lambda q, i: a[f"{q}_u{i+1}"]["noul"]
    k1 = a["keep1"]["choice"]
    i1 = int(k1[1:]) - 1
    J1 = [us[i1]]
    good = [i for i in range(len(us)) if g("ess", i) >= th and g("gq", i) < th and g("val", i) < th and g("ent", i) < th]
    if not good:
        good = [i1]
    J2 = cap([us[i] for i in good], b)
    # J3: política de pergunta do orçamento
    keep = [i for i in good if not isq(us[i])] or [i1]
    j3 = cap([us[i] for i in keep], b)
    if b["q"]:
        qs = [i for i in range(len(us)) if isq(us[i]) and g("gq", i) < th]
        if qs and us[qs[-1]] not in j3:
            j3 = j3 + [us[qs[-1]]]
    return {"J1": J1, "J2": J2, "J3": j3}


def main(split, models, th=0.5, src="A"):
    pts = [p for p in load_points() if split == "all" or p["split"] == split]
    G = load_gen()
    for model in models:
        todo = [p for p in pts if (model, src, p["id"]) in G and (model, f"J2@{th}" if th != 0.5 else "J2", p["id"]) not in G]
        items, meta = [], []
        for p in todo:
            us = units(G[(model, src, p["id"])]["text"])
            meta.append(us)
            items.append((jstate(p, us), jquestions(p, us)) if len(us) >= 2 else None)
        res = ask_many_timed([it for it in items if it], workers=4)
        it_res = iter(res)
        recs = []
        for p, us, it in zip(todo, meta, items):
            b = budget(p)
            a, dt = next(it_res) if it else (None, None)
            src_rec = G[(model, src, p["id"])]
            base = {"model": model, "pid": p["id"], "split": p["split"], "cost": src_rec["cost"],
                    "latency": src_rec["latency"], "llm_calls": src_rec["llm_calls"], "n_units": len(us)}
            if th == 0.5:
                recs.append(dict(base, cond="N", text=normalize(src_rec["text"], b), jev_calls=0))
                for k, v in rules(us, b).items():
                    recs.append(dict(base, cond=k, text=join_units(v), jev_calls=0))
            if a is not None:
                jp = jev_prune(us, a, b, th)
                jinfo = {"keep1": a["keep1"]["choice"], "keep1_conf": a["keep1"]["confidence"],
                         "nouls": {k: round(v["noul"], 3) for k, v in a.items() if v["type"] == "noul"}}
                jl = (dt or 0.55)
            else:
                jp = {"J1": us, "J2": us, "J3": us if (b["q"] or not any(isq(u) for u in us)) else ([u for u in us if not isq(u)] or us)}
                jinfo, jl = {}, 0
            for k, v in jp.items():
                if th != 0.5 and k == "J1":
                    continue
                name = k if th == 0.5 else f"{k}@{th}"
                recs.append(dict(base, cond=name, text=join_units(v), jev_calls=1 if a is not None else 0,
                                 latency=(src_rec["latency"] or 0) + jl, jev=jinfo))
        append_gen(recs)
        print(model, split, "points", len(todo), "jev calls", sum(1 for x in items if x), "jev cost", round(jev.stats["cost"], 5), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), float(sys.argv[3]) if len(sys.argv) > 3 else 0.5)
