"""b3 — rótulos-ouro da resposta HUMANA (o Jev vê a resposta real; é descrição, não previsão).
(1) Jev, todos os ~2,8 mil pontos: movimento (16, mesma taxonomia do a9), família (7), tom (7), subtexto,
    pega palavra específica, responde diretamente. Serve de ouro e de base de casos do retrieval.
(2) Jev, pontos de avaliação: qual ELEMENTO numerado da última mensagem a resposta pega.
(3) 2º rotulador independente (openai/gpt-6-luna), pontos de avaliação: movimento principal, TODOS os movimentos
    presentes (multi-rótulo), tom e elemento. Mede o teto (concordância entre rotuladores).
Uso: python3 b3_gold.py jev|elem|luna"""
import json, os, re, sys
from b3_common import (ALL, jl_load, jl_save, kv_load, kv_save, load_points, base_state, jask_many, compact, noul, choice, MOVE_OPTS, TONE_OPTS,
                       FAMILY_OPTS, MOVES, TONES, BOT, USER, elements, last_msg, lchat_many, lstats, PROC)
import jev

GQ = {
    "g_move": choice(f"Which move does `reply` make, as {BOT}'s reply to `last_message`?", MOVE_OPTS),
    "g_family": choice(f"What kind of move is `reply`, as {BOT}'s reply to `last_message`?", FAMILY_OPTS),
    "g_tone": choice("Which tone does `reply` have?", TONE_OPTS),
    "g_subtext": noul(f"Does `reply` leave its main point implicit (hinted or understated) instead of saying it outright?"),
    "g_specific": noul(f"Does `reply` pick up a specific word or detail from `last_message` (rather than the general topic)?"),
    "g_direct": noul(f"Does `reply` respond directly to `last_message`?"),
}


def gold_state(p):
    s = base_state(p)
    s["reply"] = {"from": BOT, "text": p["human"]}
    return s


def elem_q(els):
    opts = {f"e{i + 1}": f"the word/phrase '{e}'" for i, e in enumerate(els)}
    opts["whole"] = "the message as a whole or its general topic, no specific word"
    opts["other"] = "something that is not in `last_message` (earlier context or a new thing)"
    return opts


def run_jev():
    pts = jl_load(ALL)
    G = kv_load("gold_jev")
    todo = [p for p in pts if p["id"] not in G]
    todo.sort(key=lambda p: not p["eval"])  # pontos de avaliação primeiro
    print("todo", len(todo))
    for s in range(0, len(todo), 400):
        chunk = todo[s:s + 400]
        res = jask_many([(gold_state(p), GQ) for p in chunk], workers=4)
        for p, a in zip(chunk, res):
            if a:
                G[p["id"]] = compact(a)
        kv_save("gold_jev", G)
        print(s + len(chunk), jev.summary(), flush=True)


def run_elem():
    pts = load_points(eval_only=True)
    G = kv_load("gold_elem")
    todo = [p for p in pts if p["id"] not in G]
    items, idx = [], []
    for p in todo:
        els = p["elements"]
        if len(els) < 2:
            G[p["id"]] = None
            continue
        st = gold_state(p)
        st["last_message_elements"] = {f"e{i + 1}": e for i, e in enumerate(els)}
        items.append((st, {"g_elem": choice(f"Which part of `last_message` does `reply` react to most specifically?",
                                            elem_q(els))}))
        idx.append(p)
    res = jask_many(items, workers=4)
    for p, a in zip(idx, res):
        if a:
            G[p["id"]] = compact(a)
    kv_save("gold_elem", G)
    print("elem", len(items), jev.summary())


LUNA_SYS = """You annotate chat replies. You will see a short chat between Alex and Sam, and Sam's actual reply.
Label what Sam's reply does. Moves (use these exact keys):
""" + "\n".join(f"- {k}: {v}" for k, v in MOVE_OPTS.items()) + """
Tones: """ + ", ".join(f"{k} ({v})" for k, v in TONE_OPTS.items()) + """
Return ONLY a JSON object: {"primary": <the single main move>, "all": [<every move the reply clearly makes>],
"tone": <tone>, "element": <number of the element of Alex's last message that the reply reacts to most specifically,
0 if it reacts to the message as a whole/general topic, -1 if it is about something not in the last message>}"""


def luna_prompt(p):
    ctx = "\n".join(f"{h['who']}: {h['text']}" for h in p["history"][-CTXL:])
    els = p.get("elements") or elements(last_msg(p))
    el = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(els)) or "(no content words)"
    u = (f"CHAT (last message is Alex's):\n{ctx}\n\nELEMENTS of Alex's last message:\n{el}\n\n"
         f"SAM'S ACTUAL REPLY:\n{p['human']}")
    return [{"role": "system", "content": LUNA_SYS}, {"role": "user", "content": u}]


CTXL = 8


def parse_json(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def run_luna():
    pts = load_points(eval_only=True)
    G = kv_load("gold_luna")
    todo = [p for p in pts if p["id"] not in G]
    res = lchat_many([dict(messages=luna_prompt(p), model="luna", temperature=0, max_tokens=1200, seed=0,
                           extra={"reasoning": {"effort": "low"}}) for p in todo], workers=4)
    bad = 0
    for p, r in zip(todo, res):
        d = parse_json(r)
        if not d or d.get("primary") not in MOVES:
            bad += 1
            continue
        d["all"] = [m for m in d.get("all", []) if m in MOVES] or [d["primary"]]
        G[p["id"]] = d
    kv_save("gold_luna", G)
    print("luna", len(todo), "bad", bad, lstats)


if __name__ == "__main__":
    {"jev": run_jev, "elem": run_elem, "luna": run_luna}[sys.argv[1]]()
