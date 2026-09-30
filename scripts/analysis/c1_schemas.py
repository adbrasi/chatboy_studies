"""c1 — (1) Benchmark de schemas de chat (técnica C).
A mesma conversa real (histórico até o turno do parceiro, cada bolha com o seu timestamp real) é apresentada em 7 formatos,
com e sem a nota do diretor (o briefing de alvos T do relatório 13, calculado em código a partir da leitura do Jev do a9)
posta ANTES do log. A LLM completa o próximo turno de Sam; o código corta na 1ª linha de outro falante.
Condições: <schema>|0 (sem nota) e <schema>|1 (com nota), schema ∈ free, wa_full, wa_time, messenger, snapchat, irc, sms.
Uso: python3 c1_schemas.py gen <split> [modelos] [schemas]  |  python3 c1_schemas.py jev <split>"""
import sys
import c1_common as C
from b4_gen import target_brief

EXP = "schema"


def brief_of(p):
    return target_brief(p, C.budget(p))


def spec(p, model, cond):
    schema, note = cond.split("|")
    msgs = C.schema_messages(p, schema, brief_of(p) if note == "1" else None)
    return dict(messages=msgs, model=model, temperature=0.8, max_tokens=300 if schema == "free" else 400, seed=0,
                tag=f"c1:{cond}"), {"schema": schema, "note": int(note)}


def post(p, model, cond, r, meta):
    return C.parsed_record(meta["schema"], r.get("text"), p)


def conds_for(schemas):
    return [f"{s}|{n}" for s in schemas for n in (0, 1)]


def gen(split, models=None, schemas=None):
    pts = C.split_pts(split)
    C.run_gen(EXP, pts, models or C.MODELS, conds_for(schemas or C.SCHEMAS), spec, post)


def jev(split, conds=None):
    pts = {p["id"]: p for p in C.split_pts(split)}
    G = C.load_gen()
    pairs = [(p, p["human"]) for p in pts.values()]
    for (e, m, c, pid), r in G.items():
        if e == EXP and pid in pts and (conds is None or c in conds) and r.get("text"):
            pairs.append((pts[pid], r["text"]))
    print("pairs", len(pairs))
    C.jev_eval(pairs, workers=4, tag=f"{EXP}-{split}")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "gen":
        gen(a[2], a[3].split(",") if len(a) > 3 and a[3] != "all" else None, a[4].split(",") if len(a) > 4 else None)
    elif a[1] == "jev":
        jev(a[2], a[3].split(",") if len(a) > 3 else None)


def analyze(split, jev_on=True):
    import c1_metrics as M
    pts = C.split_pts(split)
    G = C.load_gen()
    res = {"split": split, "human": M.human_block(pts), "models": {}}
    print("HUMAN", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["human"].items() if not isinstance(v, (dict, list))})
    for m in C.MODELS:
        rows = []
        for c in conds_for(C.SCHEMAS):
            items = []
            for p in pts:
                r = G.get((EXP, m, c, p["id"]))
                if not r:
                    continue
                r = dict(r)
                t = r.get("text") or ""
                if not t.strip():  # quebrado/vazio: o sistema cairia no modo livre (fallback medido à parte)
                    fb = G.get((EXP, m, "free|0", p["id"])) or {}
                    t = fb.get("text") or ""
                    r["fallback"] = True
                items.append({"p": p, "text": t, "rec": r})
            if len(items) < 0.9 * len(pts):
                continue
            o = M.summarize(items, boot=True, jev=jev_on)
            res["models"].setdefault(m, {})[c] = o
            rows.append((c, o))
        print("==", m)
        M.table(rows)
    C.jdump(f"c1_schema_{split}.json", res)
    return res


if __name__ == "__main__" and sys.argv[1] == "analyze":
    analyze(sys.argv[2], jev_on=(len(sys.argv) < 4 or sys.argv[3] != "nojev"))
