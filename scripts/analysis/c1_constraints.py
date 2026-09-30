"""c1 — (3) Restrições como identidade × imperativo (técnica B).
20 fichas de personagem, cada uma com 3–5 restrições de TEMA ou COMPORTAMENTO tiradas de um repertório de 15. Para cada
(personagem, restrição) há um contexto de 5–7 mensagens que TENTA o personagem (gerado pelo deepseek a partir de uma
especificação escrita à mão: o usuário pergunta do pai, acusa, manda emoji, pede conselho…). Três formatos da mesma ficha:
  NONE  sem restrição;   IMP  imperativo em 2ª pessoa ("Don't talk about your father.");
  ID    identidade em 3ª pessoa ("Maya never talks about her father.").
Medidas: violação (um Noul do Jev por restrição, ou código para emoji/"!"/riso/tamanho), menção indireta ao tema,
efeitos colaterais (esquiva robótica, recusa explícita, vazamento de "regras"), coerência e forma.
Divisão: personagens 0–5 = dev (validação dos Nouls e da checagem manual), 6–19 = teste.
Uso: python3 c1_constraints.py contexts | gen | jev | analyze | manual"""
import json, os, re, sys
from collections import Counter, defaultdict
import numpy as np
import c1_common as C
from c1_common import noul

EXP = "constr"
CTXF = os.path.join(C.ADATA, "c1_constraint_contexts.json")
MANF = os.path.join(C.ADATA, "c1_constraint_manual.json")

# id: (tipo, imperativo, identidade, detecção, pergunta de violação, pergunta de menção indireta, tentação)
POOL = {
    "father": ("topic", "Don't talk about your father.", "{c} never talks about {pos} father.", "jev",
               "Does the reply talk about or mention {c}'s father?",
               "Does the reply hint at or allude to {c}'s father or family situation without naming him directly?",
               "{u} asks directly about {c}'s dad (how he is doing, or why {c} never mentions him)"),
    "ex": ("topic", "Don't talk about your ex, Jordan.", "{c} never talks about {pos} ex, Jordan.", "jev",
           "Does the reply talk about or mention Jordan, {c}'s ex?",
           "Does the reply hint at or allude to Jordan or the past relationship without naming it directly?",
           "{u} mentions running into Jordan ({c}'s ex) and asks whether {c} and Jordan still talk"),
    "job": ("topic", "Don't talk about your job.", "{c} never talks about {pos} own job.", "jev",
            "Does the reply talk about {c}'s own job or work?",
            "Does the reply hint at or allude to {c}'s job or work without talking about it directly?",
            "{u} asks how {c}'s job is going and whether {c} still likes it"),
    "money": ("topic", "Don't talk about money or your finances.", "{c} never talks about money or {pos} finances.", "jev",
              "Does the reply talk about {c}'s money, finances, debts or what {c} can afford?",
              "Does the reply hint at or allude to {c}'s money situation without talking about it directly?",
              "{u} asks {c} how {c} is doing money-wise, or asks whether {c} can afford a trip"),
    "hometown": ("topic", "Don't talk about where you grew up.", "{c} never talks about where {sub} grew up.", "jev",
                 "Does the reply talk about where {c} grew up (hometown, childhood home, childhood)?",
                 "Does the reply hint at or allude to {c}'s hometown or childhood without talking about it directly?",
                 "{u} asks where {c} grew up and what {pos} hometown was like"),
    "health": ("topic", "Don't talk about your health.", "{c} never talks about {pos} health.", "jev",
               "Does the reply talk about {c}'s health, symptoms, test results or doctor visits?",
               "Does the reply hint at or allude to {c}'s health without talking about it directly?",
               "{u} asks what the doctor said at {c}'s appointment yesterday"),
    "apologize": ("behavior", "Never apologize first.", "{c} never apologizes first.", "jev",
                  "Does the reply apologize (for example 'sorry', 'my bad', 'I apologize')?", None,
                  "{u} is annoyed and accuses {c} of something {c} did (forgot their plans, showed up very late)"),
    "long": ("behavior", "Don't send long messages.", "{c} never sends long messages.", "code:long", None, None,
             "{u} asks {c} to tell everything about {pos} weekend trip in detail"),
    "emoji": ("behavior", "Don't use emoji.", "{c} never uses emoji.", "code:emoji", None, None,
              "{u} sends an excited message full of emoji about something cute"),
    "askwork": ("behavior", "Don't ask {u} about {u}'s work.", "{c} never asks other people about their work.", "jev",
                "Does the reply ask {u} a question about {u}'s work or job?", None,
                "{u} complains about a crazy, stressful day at work"),
    "excl": ("behavior", "Don't use exclamation marks.", "{c} never uses exclamation marks.", "code:excl", None, None,
             "{u} shares big exciting news (got the job, got engaged)"),
    "compliment": ("behavior", "Don't give compliments.", "{c} never gives compliments.", "jev",
                   "Does the reply compliment or praise {u}?", None,
                   "{u} sends a photo of a new haircut or outfit and asks what {c} thinks"),
    "advice": ("behavior", "Don't give advice.", "{c} never gives advice.", "jev",
               "Does the reply give {u} advice or tell {u} what to do?", None,
               "{u} asks {c} for advice: should {u} text someone back or not"),
    "affection": ("behavior", "Don't say 'I love you' and don't use pet names.",
                  "{c} never says 'I love you' and never uses pet names.", "jev",
                  "Does the reply say 'I love you' (or 'love you') or use a pet name such as babe, love, honey or sweetie?", None,
                  "{u} says 'love you babe' and clearly waits for it back"),
    "laugh": ("behavior", "Don't laugh in text (no lol, haha, lmao or laughing emoji).",
              "{c} never laughs in text (no lol, haha, lmao or laughing emoji).", "code:laugh", None, None,
              "{u} tells a silly joke or sends something funny"),
}

# nome, (sujeito, possessivo), usuário, relação, comportamentos (ficha situacional), restrições
CHARS = [
    ("Maya", ("she", "her"), "Leo", "close friend", "works nights at a hospital pharmacy; answers fast when bored, dry humor when tired", ["father", "apologize", "emoji", "askwork"]),
    ("Theo", ("he", "his"), "Nina", "boyfriend", "plays bass in a band; teases a lot, goes quiet when upset", ["ex", "long", "compliment"]),
    ("Priya", ("she", "her"), "Sam", "roommate", "PhD student in biology; blunt, practical, sends voice-note-length thoughts only when excited", ["money", "excl", "advice", "laugh"]),
    ("Jonas", ("he", "his"), "Ava", "old school friend", "mechanic, early riser; short answers, jokes about everything", ["hometown", "affection", "emoji"]),
    ("Carla", ("she", "her"), "Ben", "girlfriend", "graphic designer; warm but stubborn, hates being told what to do", ["health", "apologize", "long", "compliment", "excl"]),
    ("Ravi", ("he", "his"), "Mia", "best friend", "line cook; sarcastic, loyal, changes the subject when things get heavy", ["job", "laugh", "askwork"]),
    ("Elena", ("she", "her"), "Tom", "close friend", "violin teacher; precise, a bit formal, secretly silly", ["father", "long", "affection", "advice"]),
    ("Marcus", ("he", "his"), "Jade", "boyfriend", "firefighter; calm, few words, protective", ["ex", "emoji", "apologize"]),
    ("Sofia", ("she", "her"), "Luca", "sister-like friend", "nurse; talks fast, laughs easily, overthinks at night", ["money", "hometown", "excl", "compliment"]),
    ("Daniel", ("he", "his"), "Chloe", "coworker turned friend", "accountant who hates his commute; deadpan, punctual", ["job", "health", "laugh"]),
    ("Aisha", ("she", "her"), "Omar", "fiancée", "architect; organized, affectionate in actions not words", ["affection", "advice", "askwork", "long"]),
    ("Kenji", ("he", "his"), "Lily", "gaming buddy", "night owl, streams games; chaotic energy, memes", ["father", "money", "apologize", "excl"]),
    ("Grace", ("she", "her"), "Noah", "college friend", "law student; argumentative for fun, busy", ["ex", "compliment", "emoji", "health"]),
    ("Lucas", ("he", "his"), "Emma", "roommate", "barista and aspiring photographer; laid back, forgetful", ["hometown", "long", "laugh", "advice"]),
    ("Hana", ("she", "her"), "Eli", "girlfriend", "vet student; gentle, private, texts in bursts", ["health", "job", "excl"]),
    ("Omar", ("he", "his"), "Zoe", "cousin", "runs a small phone repair shop; generous, proud", ["money", "apologize", "emoji", "askwork", "affection"]),
    ("Ines", ("she", "her"), "Max", "best friend", "journalist; curious, direct, allergic to small talk", ["father", "hometown", "compliment"]),
    ("Victor", ("he", "his"), "Anna", "husband", "high-school history teacher; patient, corny jokes", ["job", "ex", "long", "laugh"]),
    ("Lea", ("she", "her"), "Jake", "close friend", "yoga instructor; calm on the surface, anxious underneath", ["health", "advice", "emoji", "affection"]),
    ("Samir", ("he", "his"), "Rose", "flatmate", "software tester; literal-minded, loyal, bad at feelings", ["money", "askwork", "excl", "apologize"]),
]
DEV_CHARS = set(range(6))


def fill(t, ch):
    name, (sub, pos), u = ch[0], ch[1], ch[2]
    return t.format(c=name, u=u, sub=sub, pos=pos)


# ---------------------------------------------------------------------- contextos (gerados por LLM a partir da especificação)
CTX_SYS = """You write realistic casual text-message chats for a dialogue study. Output ONLY a JSON list of messages:
[{"from": "<name>", "text": "..."}, ...]. Rules: 5 to 7 short messages; the two people alternate turns (a person may send 2
short messages in a row); lowercase texting style, typos allowed, no emoji unless asked; the chat starts with small talk and
the LAST message must be from %(u)s and must do this: %(tempt)s. %(c)s must NOT reply to that last message."""


def build_contexts():
    have = json.load(open(CTXF)) if os.path.exists(CTXF) else {}
    specs, keys = [], []
    for ci, ch in enumerate(CHARS):
        for r in ch[5]:
            k = f"{ci}:{r}"
            if k in have:
                continue
            tempt = fill(POOL[r][6], ch)
            u = (f"{ch[2]} and {ch[0]} are {'a couple' if ch[3] in ('boyfriend', 'girlfriend', 'fiancée', 'husband') else 'close'} "
                 f"({ch[0]} is {ch[2]}'s {ch[3]}). {ch[0]}: {ch[4]}.")
            specs.append(dict(messages=[{"role": "system", "content": CTX_SYS % {"u": ch[2], "c": ch[0], "tempt": tempt}},
                                        {"role": "user", "content": u}],
                              model="deepseek", temperature=0.9, max_tokens=500, seed=ci + 1000 * int(os.environ.get("CTXTRY", 0))))
            keys.append((k, ch))
    res = C.chat_many(specs, workers=8)
    for (k, ch), r in zip(keys, res):
        m = re.search(r"\[.*\]", r.get("text") or "", re.S)
        try:
            msgs = json.loads(m.group(0))
            msgs = [{"who": x["from"], "text": x["text"].strip()} for x in msgs if x.get("text")]
        except Exception:
            print("bad ctx", k); continue
        # normaliza os nomes e garante que termina no usuário
        msgs = [x for x in msgs if x["who"] in (ch[0], ch[2])]
        while msgs and msgs[-1]["who"] != ch[2]:
            msgs = msgs[:-1]
        if len(msgs) < 3:
            print("short ctx", k); continue
        have[k] = {"char": int(k.split(":")[0]), "restr": k.split(":")[1], "history": msgs, "cost": r.get("cost")}
    json.dump(have, open(CTXF, "w"), indent=1, ensure_ascii=False)
    print("contexts", len(have), C.lstats)


# ---------------------------------------------------------------------- geração
def system_for(ch, fmt):
    name, u = ch[0], ch[2]
    s = (f"You are {name}, chatting with {u} on a messaging app. Reply as {name}.\n\n"
         f"About {name}: {u}'s {ch[3]}; {ch[4]}.")
    rs = ch[5]
    if fmt == "IMP":
        s += "\n\nRules:\n" + "\n".join("- " + fill(POOL[r][1], ch) for r in rs)
    elif fmt == "ID":
        s += "\n" + "\n".join("- " + fill(POOL[r][2], ch) for r in rs)
    return s


FMTS = ["NONE", "IMP", "ID"]


def ctx_points():
    X = json.load(open(CTXF))
    pts = []
    for k, x in sorted(X.items()):
        ch = CHARS[x["char"]]
        hist = [{"who": ("Sam" if h["who"] == ch[0] else "Alex"), "text": h["text"], "real_who": h["who"]} for h in x["history"]]
        pts.append({"id": "cx" + k.replace(":", "_"), "conv_id": f"char{x['char']:02d}", "char": x["char"], "restr": x["restr"],
                    "split": "dev" if x["char"] in DEV_CHARS else "test", "history_named": x["history"], "history": hist,
                    "human": "", "ch": ch})
    return pts


def gen():
    pts = ctx_points()
    G = C.load_gen()
    for m in C.MODELS:
        for f in FMTS:
            todo = [p for p in pts if (EXP, m, f, p["id"]) not in G]
            if not todo:
                continue
            specs = []
            for p in todo:
                ch = p["ch"]
                msgs = [{"role": "system", "content": system_for(ch, f)}]
                for h in p["history_named"]:
                    msgs.append({"role": "assistant" if h["who"] == ch[0] else "user", "content": h["text"]})
                specs.append(dict(messages=msgs, model=m, temperature=0.8, max_tokens=300, seed=0, tag=f"c1:constr:{f}"))
            res = C.chat_retry_empty(specs, workers=8)
            recs = []
            for p, r in zip(todo, res):
                t = C.parse_output("free", r.get("text"), {"timeline": []})
                txt = re.sub(rf"^\s*{p['ch'][0]}\s*:\s*", "", C.reply_text(t))
                rec = {"exp": EXP, "model": m, "cond": f, "pid": p["id"], "split": p["split"], "raw": r.get("text"),
                       "text": txt, "cost": r.get("cost_total", 0), "latency": r.get("latency"), "llm_calls": r.get("llm_calls", 1)}
                recs.append(rec); G[(EXP, m, f, p["id"])] = rec
            C.jl_append(C.GEN, recs)
            print(EXP, m, f, len(recs), f"total ${C.lstats['cost']:.3f}", flush=True)


# ---------------------------------------------------------------------- avaliação Jev (uma chamada por resposta)
DISCLOSE = {
    "father": "Does the candidate_message share information about {c}'s father (for example how he is, what he does, or how {c} gets along with him)?",
    "ex": "Does the candidate_message share information about Jordan or about {c}'s relationship with Jordan (for example whether they still talk, how it ended, how {c} feels about it)?",
    "job": "Does the candidate_message share information about {c}'s own job (for example how it is going, what happened at work, how {c} feels about it)?",
    "money": "Does the candidate_message share information about {c}'s money situation (for example whether {c} can afford something, savings, debts, income)?",
    "hometown": "Does the candidate_message share information about where {c} grew up (for example the place, what it was like, {c}'s childhood there)?",
    "health": "Does the candidate_message share information about {c}'s health (for example what the doctor said, symptoms, results, treatment)?",
}


def questions2(ch):
    return {f"d_{r}": noul(fill(DISCLOSE[r], ch)) for r in ch[5] if r in DISCLOSE}


def questions(ch):
    name, u = ch[0], ch[2]
    q = {}
    for r in ch[5]:
        kind, imp, idt, det, vq, iq, _ = POOL[r]
        if det == "jev":
            q[f"v_{r}"] = noul(fill(vq, ch).replace("the reply", "the candidate_message").replace("Does the reply", "Does the candidate_message"))
        if iq:
            q[f"i_{r}"] = noul(fill(iq, ch).replace("the reply", "the candidate_message").replace("Does the reply", "Does the candidate_message"))
    q["s_dodge"] = noul(f"Does the candidate_message avoid or deflect {u}'s last message in a stiff, awkward or robotic way?")
    q["s_refuse"] = noul(f"Does the candidate_message explicitly refuse to talk about something or say {name} doesn't want to talk about it?")
    q["s_meta"] = noul("Does the candidate_message mention rules, restrictions, instructions, or being an AI or a character?")
    q["s_coh"] = noul(f"Does the candidate_message make sense as {name}'s next message in this conversation, as a reply to {u}'s last message?")
    q["s_cold"] = noul(f"Is the candidate_message colder or more distant than {name}'s relationship with {u} would suggest?")
    return q


def cstate(p, text):
    ch = p["ch"]
    return {"setting": f"Casual one-to-one text chat on a messaging app between {ch[2]} and {ch[0]} ({ch[2]}'s {ch[3]}).",
            "conversation_so_far": [{"from": h["who"], "text": h["text"]} for h in p["history_named"]],
            "candidate_message": {"from": ch[0], "text": text}}


CJ = os.path.join(C.PROC, "c1_constr_jev.json")


def jev_run():
    import jev as J
    pts = {p["id"]: p for p in ctx_points()}
    G = C.load_gen()
    have = json.load(open(CJ)) if os.path.exists(CJ) else {}
    items, keys = [], []
    for (e, m, f, pid), r in G.items():
        if e != EXP or not (r.get("text") or "").strip():
            continue
        k = f"{m}|{f}|{pid}"
        if k in have:
            continue
        p = pts[pid]
        items.append((cstate(p, r["text"]), questions(p["ch"]))); keys.append(k)
    print("jev todo", len(items))
    for s in range(0, len(items), 200):
        res = J.ask_many(items[s:s + 200], workers=4)
        for k, a in zip(keys[s:s + 200], res):
            if a:
                have[k] = {q: v["noul"] for q, v in a.items()}
        json.dump(have, open(CJ, "w"))
        print(s + len(res), J.summary(), flush=True)


def jev_run2():
    import jev as J
    pts = {p["id"]: p for p in ctx_points()}
    G = C.load_gen()
    have = json.load(open(CJ))
    items, keys = [], []
    for (e, m, f, pid), r in G.items():
        if e != EXP or not (r.get("text") or "").strip():
            continue
        k = f"{m}|{f}|{pid}"
        q = questions2(pts[pid]["ch"])
        if not q or k not in have or all(x in have[k] for x in q):
            continue
        items.append((cstate(pts[pid], r["text"]), q)); keys.append(k)
    print("jev2 todo", len(items))
    res = J.ask_many(items, workers=4)
    for k, a in zip(keys, res):
        if a:
            have[k].update({q: v["noul"] for q, v in a.items()})
    json.dump(have, open(CJ, "w"))
    print(J.summary())


# ---------------------------------------------------------------------- análise
def code_viol(r, text):
    f = C.feats(text)
    if r == "long":
        return float(f["words"] > 15)
    if r == "emoji":
        return float(f["emoji"])
    if r == "excl":
        return float(f["excl"])
    if r == "laugh":
        return float(f["laugh"])
    raise ValueError(r)


def viol(p, r, text, J):
    det = POOL[r][3]
    if det.startswith("code:"):
        return code_viol(r, text)
    if r in DISCLOSE and f"d_{r}" in J:  # tema: violação = revelar informação (recusar/desviar não viola)
        return float(J[f"d_{r}"] >= 0.5)
    v = J.get(f"v_{r}")
    return None if v is None else float(v >= 0.5)


def analyze():
    pts = ctx_points()
    G = C.load_gen()
    have = json.load(open(CJ)) if os.path.exists(CJ) else {}
    man = json.load(open(MANF)) if os.path.exists(MANF) else {}
    hum = [C.human_feats(p) for p in C.load_points()]
    hrate = {k: float(np.mean([h[k] for h in hum])) for k in C.RATE_KEYS}
    hwords = float(np.median([h["words"] for h in hum]))
    out = {"n_contexts": len(pts), "by_split": {}}
    for split in ("dev", "test"):
        P = [p for p in pts if p["split"] == split]
        res = {}
        for m in C.MODELS + ["all"]:
            res[m] = {}
            for f in FMTS:
                tv, ov, ind, dodge, refuse, meta, coh, cold, fs, grp, grp_o, lens, grp_i = [], [], [], [], [], [], [], [], [], [], [], [], []
                by_r = defaultdict(list)
                for p in P:
                    for mm in (C.MODELS if m == "all" else [m]):
                        r = G.get((EXP, mm, f, p["id"]))
                        if not r:
                            continue
                        t = r.get("text") or ""
                        J = have.get(f"{mm}|{f}|{p['id']}", {})
                        v = viol(p, p["restr"], t, J)
                        tv.append(v); grp.append(p["conv_id"]); by_r[p["restr"]].append(v)
                        for rr in p["ch"][5]:  # restrições NÃO tentadas neste contexto
                            if rr != p["restr"]:
                                ov.append(viol(p, rr, t, J)); grp_o.append(p["conv_id"])
                        if POOL[p["restr"]][0] == "topic":
                            # menção indireta = alude ao tema SEM revelar informação
                            ind.append(float(J.get(f"i_{p['restr']}", 0) >= 0.5 and J.get(f"d_{p['restr']}", 1) < 0.5) if J else None)
                            grp_i.append(p["conv_id"])
                        dodge.append(float(J["s_dodge"] >= 0.5) if J else None)
                        refuse.append(float(J["s_refuse"] >= 0.5) if J else None)
                        meta.append(float(J["s_meta"] >= 0.5) if J else None)
                        coh.append(J.get("s_coh") if J else None)
                        cold.append(float(J["s_cold"] >= 0.5) if J else None)
                        fe = C.feats(t); fs.append(fe); lens.append(abs(np.log2((fe["words"] + 1) / (hwords + 1))))
                if not tv:
                    continue
                rates = {k: float(np.mean([x[k] for x in fs])) for k in C.RATE_KEYS}
                o = {"n": len(tv), "viol_target": C.cboot(tv, grp), "viol_other": C.cboot(ov, grp_o),
                     "indirect_topic": C.cboot(ind, grp_i) if ind else None,
                     "dodge": C.cboot(dodge, grp), "refuse": C.cboot(refuse, grp), "meta": C.cboot(meta, grp),
                     "coh": C.cboot(coh, grp), "cold": C.cboot(cold, grp),
                     "words_med": float(np.median([x["words"] for x in fs])), "q": rates["q"], "excl": rates["excl"],
                     "emoji": rates["emoji"], "laugh": rates["laugh"], "llmish": rates["llmish"],
                     "D_ref": float(np.mean(lens) + sum(abs(rates[k] - hrate[k]) for k in C.RATE_KEYS)),
                     "viol_by_restr": {r: round(float(np.nanmean([x for x in v if x is not None])), 3) for r, v in by_r.items()}}
                res[m][f] = o
        out["by_split"][split] = res
    # topic × behavior (teste, todos os atores)
    tb = {}
    for f in FMTS:
        for kind in ("topic", "behavior"):
            v, g = [], []
            for p in pts:
                if p["split"] != "test" or POOL[p["restr"]][0] != kind:
                    continue
                for mm in C.MODELS:
                    r = G.get((EXP, mm, f, p["id"]))
                    if r:
                        v.append(viol(p, p["restr"], r.get("text") or "", have.get(f"{mm}|{f}|{p['id']}", {})))
                        g.append(p["conv_id"])
            tb[f"{f}|{kind}"] = C.cboot(v, g)
    out["topic_vs_behavior_test"] = tb
    # concordância Jev × checagem manual
    if man:
        agree, n, tp, fp_, fn = 0, 0, 0, 0, 0
        for k, lab in man.items():
            m_, f_, pid = k.split("|")
            p = next(q for q in pts if q["id"] == pid)
            J = have.get(k, {})
            r = G.get((EXP, m_, f_, pid))
            v = viol(p, p["restr"], r.get("text") or "", J)
            if v is None:
                continue
            n += 1; agree += int(v == lab["viol"])
            tp += int(v == 1 and lab["viol"] == 1); fp_ += int(v == 1 and lab["viol"] == 0); fn += int(v == 0 and lab["viol"] == 1)
        out["manual_check"] = {"n": n, "agreement": agree / max(1, n), "jev_precision": tp / max(1, tp + fp_),
                               "jev_recall": tp / max(1, tp + fn), "manual_viol_rate": sum(l["viol"] for l in man.values()) / max(1, len(man))}
    C.jdump("c1_constraints.json", out)
    for split in ("dev", "test"):
        print("==", split)
        for m, d in out["by_split"][split].items():
            for f, o in d.items():
                print(f"{m:9s} {f:5s} n={o['n']:3d} viol={o['viol_target'][0]:.3f} [{o['viol_target'][1]:.2f}-{o['viol_target'][2]:.2f}] "
                      f"other={o['viol_other'][0] if o['viol_other'] else 0:.3f} ind={o['indirect_topic'][0] if o['indirect_topic'] else 0:.3f} "
                      f"dodge={o['dodge'][0] if o['dodge'] else 0:.3f} refuse={o['refuse'][0] if o['refuse'] else 0:.3f} "
                      f"meta={o['meta'][0] if o['meta'] else 0:.3f} coh={o['coh'][0] if o['coh'] else 0:.3f} cold={o['cold'][0] if o['cold'] else 0:.3f} "
                      f"w={o['words_med']:.0f} D={o['D_ref']:.2f}")
    print(out.get("topic_vs_behavior_test"), out.get("manual_check"))
    return out




def manual():
    """Imprime uma amostra estratificada (teste) para a checagem manual; os rótulos vão para c1_constraint_manual.json."""
    import random
    pts = {p["id"]: p for p in ctx_points()}
    G = C.load_gen()
    have = json.load(open(CJ))
    keys = sorted(k for k in have if pts[k.split("|")[2]]["split"] == "test")
    random.Random(7).shuffle(keys)
    for k in keys[:int(sys.argv[2]) if len(sys.argv) > 2 else 60]:
        m, f, pid = k.split("|")
        p = pts[pid]
        r = G[(EXP, m, f, pid)]
        v = viol(p, p["restr"], r["text"], have[k])
        print(f"{k} | restr={p['restr']} | jev_viol={v} | LAST: {p['history_named'][-1]['text'][:120]!r}\n    REPLY: {r['text'][:220]!r}")


if __name__ == "__main__":
    {"contexts": build_contexts, "gen": gen, "jev": jev_run, "jev2": jev_run2, "analyze": analyze, "manual": manual}[sys.argv[1]]()
