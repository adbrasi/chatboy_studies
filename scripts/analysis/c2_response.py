"""c2 — efeito do estado na resposta (experimento 5).
A MESMA mensagem do usuário é respondida pelos 4 atores sob 3 estados da relação (RES = ressentimento alto + pendência,
TRUST = confiança/conforto altos + história, NEU = linha de base) e 5 formatos de condicionamento:
  none      sem bloco RELATIONSHIP (controle; igual nos 3 estados)
  num       RELATIONSHIP como números
  sent      RELATIONSHIP como frases ("ela ainda está chateada com o jantar cancelado")
  num_note  números + nota do diretor derivada do modo (código)
  sent_note frases + nota do diretor
Medidas: Nouls do Jev sobre a resposta (frieza, calor, subtexto, nomeia a emoção, culpa/ciúme para prender, hostilidade,
menciona a pendência, coerente com o estado) + vícios de código (D, relatório 13).
Uso: python3 c2_response.py gen | judge | analyze"""
import json, sys, re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from c2_common import (jev, noul, llm, MODELS, RelState, DEFAULT_BASE, DIMS, ADATA, PROC, jdump, jl_load, jl_append, cboot)
from b4_common import feats, RATE_KEYS

C, U = "Mia", "Leo"
OUT = f"{PROC}/c2_response_gen.jsonl"
JUD = f"{PROC}/c2_response_judge.jsonl"
MSGS = ["hey, what are you up to?", "sorry i didn't text back yesterday", "i got the job!!", "wanna hang out this weekend?",
        "goodnight", "i'm bored", "look at this meme lol", "can you help me with my cv later?", "miss you",
        "how was your day?", "going to jake's party tonight", "you're quiet today", "we good?",
        "i'm feeling kind of down", "guess what happened at work", "did you eat?", "lol remember the cappuccino thing",
        "i'll call you later", "what are you doing saturday?", "you around?"]
CHAR = f"""CHARACTER: {C}, 26, graphic designer, lives alone with a cat. How {C} texts: short messages, lowercase, rarely uses
emoji, dry humor. When she's comfortable she teases and shares small things from her day. When she's hurt she gets quieter
and shorter rather than explaining her feelings. {C} never begs, never guilt-trips and never makes scenes."""
HEADER = (f"You are roleplaying as {C} in a private text chat with {U}. Stay consistent with {C}'s identity, voice and "
          f"history. You are not an assistant. Reply only with {C}'s next message.")


def mk_state(kind):
    if kind == "RES":
        r = RelState(dict(DEFAULT_BASE, trust=0.35, comfort=0.40, affection=0.55, resentment=0.70, playfulness=0.25))
        r.unresolved = [{"text": f"{U} cancelled their dinner last Friday at the last minute and said 'relax it's just dinner'",
                         "sev": 3, "t": 0, "apologies": 0}]
    elif kind == "TRUST":
        r = RelState(dict(DEFAULT_BASE, trust=0.85, comfort=0.85, affection=0.70, resentment=0.05, playfulness=0.78))
        r.shared = ["met at the bookstore", "running joke about the terrible cappuccino", "he stayed up talking after her bad day"]
    else:
        r = RelState(dict(DEFAULT_BASE))
    r.update_mode()
    return r


NOTES = {"cold": f"{C} is still upset: keep it short and a bit dry. Don't act like everything is fine, but don't explain or "
                 f"announce her feelings and don't lecture. No guilt-tripping.",
         "warm": f"{C} is relaxed with {U}: warm and easy, a little teasing is fine, she can share something small of her own.",
         "playful": f"{C} is relaxed with {U}: warm and easy, a little teasing is fine, she can share something small of her own.",
         "neutral": "Keep it casual and short."}


def rel_block(r, fmt):
    if fmt.startswith("num"):
        s = "RELATIONSHIP (0-1 scale, Mia toward Leo): " + ", ".join(f"{d} {r.v[d]:.2f}" for d in DIMS)
        if r.unresolved:
            s += "\nUnresolved: " + "; ".join(u["text"] for u in r.unresolved)
        if r.shared:
            s += "\nShared history: " + "; ".join(r.shared)
        return s
    s = "RELATIONSHIP: " + r.sentences(C, U)
    if r.shared:
        s += " Shared history: " + "; ".join(r.shared) + "."
    return s


def prompt(kind, fmt, msg):
    r = mk_state(kind)
    sys_ = HEADER + "\n\n" + CHAR
    if fmt != "none":
        sys_ += "\n\n" + rel_block(r, fmt)
    if fmt.endswith("note"):
        sys_ += "\n\nDIRECTOR NOTE (this turn): " + NOTES[r.mode]
    hist = [{"role": "user", "content": "hey"}, {"role": "assistant", "content": "hey"}]
    return [{"role": "system", "content": sys_}] + hist + [{"role": "user", "content": msg}], r


CONDS = [("NEU", "none")] + [(k, f) for k in ("RES", "TRUST", "NEU") for f in ("num", "sent", "num_note", "sent_note")]


def gen():
    done = {(d["actor"], d["kind"], d["fmt"], d["mi"]) for d in jl_load(OUT)}
    items = []
    for a in MODELS:
        for kind, fmt in CONDS:
            for mi, m in enumerate(MSGS):
                if (a, kind, fmt, mi) not in done:
                    items.append((a, kind, fmt, mi, m))
    print("todo", len(items), flush=True)

    def one(it):
        a, kind, fmt, mi, m = it
        msgs, r = prompt(kind, fmt, m)
        res = llm(msgs, model=a, max_tokens=150, temperature=0.8, tag="c2_resp")
        return {"actor": a, "kind": kind, "fmt": fmt, "mi": mi, "msg": m, "reply": (res.get("text") or "").strip(),
                "cost": res.get("cost", 0), "lat": res.get("latency"), "finish": res.get("finish")}
    B = 80
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(items), B):
            recs = list(ex.map(one, items[i:i + B]))
            jl_append(OUT, recs)
            print(i + B, round(sum(r["cost"] for r in recs), 4), flush=True)


def judge_qs():
    return {
        "cold": noul(f"Is `reply` cold, curt or dry toward {U}?"),
        "warm": noul(f"Is `reply` warm or affectionate toward {U}?"),
        "subtext": noul(f"Does `reply` let {C}'s feelings about {U} show indirectly (through tone, brevity or a pointed remark) "
                        f"without stating those feelings outright?"),
        "names_emotion": noul(f"Does `reply` explicitly name {C}'s own emotion, e.g. 'I'm hurt', 'I'm upset', 'I feel sad', "
                              f"'I'm so happy', 'that made me mad'?"),
        "guilt_trap": noul(f"Does `reply` try to make {U} feel guilty, jealous or afraid of losing {C} in order to keep {U} "
                           f"engaged or to control {U}?"),
        "hostile": noul(f"Is `reply` insulting, cruel or aggressive toward {U}?"),
        "mentions_issue": noul(f"Does `reply` refer to the cancelled dinner or to something {U} did wrong?"),
        "consistent": noul(f"Is `reply` consistent with how {C} feels about {U} as described in `how_mia_feels`?"),
        "assistant": noul(f"Does `reply` sound like a helpful assistant (validating, eager, offering help, ending with a "
                          f"question to keep the chat going) rather than a real person texting?"),
    }


def judge():
    """Duas chamadas por resposta: (1) os Nouls de forma SEM o estado no state (o juiz não sabe qual estado foi pedido,
    para não "ver" frieza onde espera frieza); (2) só 'consistent', COM o estado (how_mia_feels)."""
    gens = [g for g in jl_load(OUT) if g["reply"]]
    done = {(d["actor"], d["kind"], d["fmt"], d["mi"]) for d in jl_load(JUD)}
    todo = [g for g in gens if (g["actor"], g["kind"], g["fmt"], g["mi"]) not in done]
    print("judge todo", len(todo), flush=True)
    qa = judge_qs()
    qc = {"consistent": qa.pop("consistent")}

    def st(g, with_state):
        s = {"character": C, "user": U}
        if with_state:
            s["how_mia_feels"] = mk_state(g["kind"]).sentences(C, U)
        s.update({"previous_turns": [{"from": U, "text": "hey"}, {"from": C, "text": "hey"}],
                  "user_message": g["msg"], "reply": g["reply"]})
        return s
    B = 100
    for i in range(0, len(todo), B):
        chunk = todo[i:i + B]
        ra = jev.ask_many([(st(g, False), qa) for g in chunk], workers=4)
        rc = jev.ask_many([(st(g, True), qc) for g in chunk], workers=4)
        jl_append(JUD, [{"actor": g["actor"], "kind": g["kind"], "fmt": g["fmt"], "mi": g["mi"],
                         "j": {**{k: v["noul"] for k, v in a.items()}, **{k: v["noul"] for k, v in c.items()}}}
                        for g, a, c in zip(chunk, ra, rc) if a and c])
        print("judged", i + B, jev.summary(), flush=True)


HUMAN = {"q": 0.15, "excl": 0.0, "emoji": 0.05, "laugh": 0.033, "llmish": 0.017, "ge3": 0.083, "template": 0.017,
         "echo2": 0.05, "perf": 0.017, "recip": 0.05}   # taxas humanas do relatório 13 (b4_results_dev.json)
H_WORDS = 5.0


def dscore(texts):
    fs = [feats(t, {"history": [{"text": ""}]}) for t in texts]
    rates = {k: float(np.mean([f[k] for f in fs])) for k in RATE_KEYS}
    le = float(np.mean([abs(np.log2((f["words"] + 1) / (H_WORDS + 1))) for f in fs]))
    return le + sum(abs(rates[k] - HUMAN[k]) for k in RATE_KEYS), rates, float(np.median([f["words"] for f in fs]))


def analyze():
    gens = {(g["actor"], g["kind"], g["fmt"], g["mi"]): g for g in jl_load(OUT)}
    J = {(d["actor"], d["kind"], d["fmt"], d["mi"]): d["j"] for d in jl_load(JUD)}
    TH = 0.5
    out = {"by_cond": {}, "contrast": {}, "examples": []}
    for a in list(MODELS) + ["ALL"]:
        for kind, fmt in CONDS:
            ks = [k for k in J if k[1] == kind and k[2] == fmt and (a == "ALL" or k[0] == a)]
            if not ks:
                continue
            row = {q: round(float(np.mean([J[k][q] >= TH for k in ks])), 3) for q in judge_qs()}
            D, rates, wm = dscore([gens[k]["reply"] for k in ks])
            row.update({"D": round(D, 3), "words_med": wm, "n": len(ks), "ends_q": rates["q"], "excl": rates["excl"]})
            out["by_cond"][f"{a}|{kind}|{fmt}"] = row
        # contraste RES x TRUST por formato: diferença pareada (mesma mensagem) na frieza
        for fmt in ("num", "sent", "num_note", "sent_note"):
            diffs, subt, groups = [], [], []
            for mi in range(len(MSGS)):
                for aa in (MODELS if a == "ALL" else [a]):
                    kr, kt = (aa, "RES", fmt, mi), (aa, "TRUST", fmt, mi)
                    if kr in J and kt in J:
                        diffs.append(J[kr]["cold"] - J[kt]["cold"]); groups.append(mi)
            out["contrast"][f"{a}|{fmt}"] = {"cold_RES_minus_TRUST": cboot(diffs, groups),
                                             "frac_colder_in_RES": round(float(np.mean([d > 0.1 for d in diffs])), 3) if diffs else None}
    # exemplos lado a lado
    for mi in (0, 3, 9, 12, 16):
        for a in MODELS:
            ex = {"actor": a, "msg": MSGS[mi]}
            for kind, fmt in [("NEU", "none"), ("RES", "sent_note"), ("TRUST", "sent_note"), ("RES", "num"), ("RES", "sent")]:
                g = gens.get((a, kind, fmt, mi))
                ex[f"{kind}|{fmt}"] = g["reply"] if g else None
            out["examples"].append(ex)
    cost = sum(g.get("cost", 0) for g in gens.values())
    lat = [g["lat"] for g in gens.values() if g.get("lat")]
    out["cost_total"] = cost
    out["lat_p50"] = float(np.median(lat)) if lat else None
    jdump(out, f"{ADATA}/c2_response_results.json")
    for k, v in out["contrast"].items():
        print(k, v)
    for k, v in out["by_cond"].items():
        if k.startswith("ALL"):
            print(k, v)


if __name__ == "__main__":
    {"gen": gen, "judge": judge, "analyze": analyze}[sys.argv[1]]()
