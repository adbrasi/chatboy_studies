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
