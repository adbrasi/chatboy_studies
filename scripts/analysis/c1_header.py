"""c1 — (4) Cabeçalho de roleplay (técnica D) × persona mínima.
Modo livre (histórico como mensagens user/assistant), mesmos pontos do experimento 1.
  M0  persona mínima do relatório 13 (= schema free|0 do exp. 1)          M1  M0 + nota no system (= free|1)
  H0  cabeçalho de RP ("You are roleplaying as Sam… You are not an assistant") + ficha com COMPORTAMENTOS SITUACIONAIS
      + relação (rótulo do Jev da 1ª rodada) + "how Sam texts" (impressão digital de estilo, em código)
  H1e H0 + nota do diretor no FIM do contexto (mensagem de system depois da última fala do usuário, estilo Author's Note)
  H1s H0 + nota do diretor no começo (dentro do system)  → separa o efeito do cabeçalho do efeito da posição da nota
  HA0 cabeçalho de RP + ficha com ADJETIVOS (casual, funny, laid-back…) em vez de comportamentos (ablação; só dev)
Uso: python3 c1_header.py gen <split> [conds] | jev <split> [conds] | analyze <split> [conds]"""
import sys
import c1_common as C
from b4_gen import target_brief

EXP = "header"
REL = {"close_friends": "close friend", "colleagues_or_classmates": "classmate / colleague", "acquaintances_or_new": "acquaintance",
       "romantic_partners": "partner", "family": "family member", "flirting_or_crush": "crush (they flirt)"}


def texting_line(fp, b):
    mw = max(2, int(round(fp["median_words_per_bubble"])))
    s = [f"short messages of about {mw} words"]
    if fp["bubbles_per_turn"] >= 1.5:
        s.append("often splits a thought into several short messages in a row")
    if b["lower"]:
        s.append("all lowercase")
    if b["noperiod"]:
        s.append("no final period")
    s.append(f"laughs with '{fp['laugh_token']}' only when something is actually funny" if fp.get("laugh_token")
             else "rarely laughs in text")
    s.append("uses an emoji now and then" if fp["emoji_frac"] >= 0.1 else "almost never uses emoji")
    return "; ".join(s) + "."


BEHAV = [
    f"When {C.USER} shares news, {C.BOT} reacts to the fact itself, not to {C.USER}'s feelings.",
    f"When {C.USER} jokes or teases, {C.BOT} jokes or teases back instead of laughing politely.",
    f"When {C.BOT} has nothing to add, {C.BOT} sends a short reaction (\"true\", \"fair\", \"lol\") and stops there.",
    f"When something is serious, {C.BOT} writes plainly and a bit longer, without jokes or therapist talk.",
    f"{C.BOT} only asks a question when {C.BOT} actually wants to know the answer, never just to keep the chat going.",
    f"{C.BOT} has a life, opinions and moods of their own, and brings them up.",
]


def header_system(p, adjectives=False):
    b = C.budget(p)
    rel = REL.get(p["prev_D"].get("relationship"), "friend")
    s = (f"You are roleplaying as {C.BOT} in a private text chat with {C.USER}. Stay consistent with {C.BOT}'s identity, voice "
         f"and history. You are not an assistant: you are {C.BOT}, a real person texting from their phone, and you never "
         f"break character.\n\nCHARACTER: {C.BOT}\nRelationship: {C.USER} is {C.BOT}'s {rel}.\n")
    if adjectives:
        s += "Personality: casual, funny, laid-back, a bit sarcastic, loyal.\nTexting style: casual and short."
    else:
        s += f"How {C.BOT} texts: {texting_line(p['fp'], b)}\nHow {C.BOT} behaves:\n" + "\n".join("- " + x for x in BEHAV)
    return s


def brief_of(p):
    return target_brief(p, C.budget(p))


def messages(p, cond):
    if cond == "HA0":
        return C.schema_messages(p, "free", None, system_override=header_system(p, adjectives=True))
    sysm = header_system(p)
    if cond == "H0":
        return C.schema_messages(p, "free", None, system_override=sysm)
    if cond == "H1s":
        return C.schema_messages(p, "free", brief_of(p), system_override=sysm, note_pos="before")
    if cond == "H1e":
        m = C.schema_messages(p, "free", None, system_override=sysm)
        m.append({"role": "system", "content": f"DIRECTOR NOTE (this turn only):\n{brief_of(p)}"})
        return m
    raise ValueError(cond)


def spec(p, model, cond):
    return dict(messages=messages(p, cond), model=model, temperature=0.8, max_tokens=300, seed=0, tag=f"c1:{cond}"), {"schema": "free"}


def post(p, model, cond, r, meta):
    return C.parsed_record("free", r.get("text"), p)


def gen(split, conds):
    C.run_gen(EXP, C.split_pts(split), C.MODELS, conds, spec, post)


def jev(split, conds):
    pts = C.split_pts(split)
    G = C.load_gen()
    pairs = [(p, p["human"]) for p in pts]
    for m in C.MODELS:
        for c in conds:
            for p in pts:
                r = G.get((EXP, m, c, p["id"]))
                if r and r.get("text"):
                    pairs.append((p, r["text"]))
    C.jev_eval(pairs, workers=4, tag=f"header-{split}")


def analyze(split, conds, jev_on=True):
    import c1_metrics as M
    pts = C.split_pts(split)
    G = C.load_gen()
    res = {"split": split, "models": {}}
    for m in C.MODELS:
        rows = []
        for c in ["M0", "M1"] + conds:
            key = (("schema", "free|0") if c == "M0" else ("schema", "free|1") if c == "M1" else (EXP, c))
            items = []
            for p in pts:
                r = G.get((key[0], m, key[1], p["id"]))
                if r:
                    items.append({"p": p, "text": r.get("text") or "", "rec": r})
            if len(items) < 0.9 * len(pts):
                continue
            o = M.summarize(items, boot=True, jev=jev_on)
            res["models"].setdefault(m, {})[c] = o
            rows.append((c, o))
        print("==", m)
        M.table(rows, keys=("D", "lenerr", "words_med", "q", "excl", "emoji", "laugh", "llmish", "echo2", "perf", "multi",
                            "invented_user", "move_match", "coh", "bank"))
    C.jdump(f"c1_header_{split}.json", res)
    return res


if __name__ == "__main__":
    a = sys.argv
    conds = a[3].split(",") if len(a) > 3 else ["H0", "H1e", "H1s", "HA0"]
    {"gen": gen, "jev": jev, "analyze": analyze}[a[1]](a[2], conds)
