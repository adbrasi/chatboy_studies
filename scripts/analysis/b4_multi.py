"""b4 — mecanismos de VÁRIAS etapas (3, 4, 5 e 6 do enunciado), sobre os 4 atores.
  D5   rascunho -> versão de chat, COM alvos numéricos (2ª passada da LLM sobre o rascunho A)
  D5x  rascunho -> versão de chat, SEM alvos (só "como uma pessoa mandaria no WhatsApp")
  RW   reescrita cirúrgica guiada: código + Jev diagnosticam os vícios do rascunho -> ordens pontuais -> LLM edita;
       repete até 3 voltas enquanto houver vício (mede voltas, custo, latência)
  PL2  plano -> (código fixa os números) -> Jev valida a ideia -> texto (2–3 chamadas de LLM + 1–2 de Jev)
  INT  controle de intensidade: Jev dá Score de intensidade ao usuário e a 4 candidatas (A, A_s1..A_s3); o código escolhe
       uma com intensidade <= a do usuário (a mais perto do alvo de tamanho); se nenhuma serve, reescreve "mais calma"
  INTr só a reescrita (sem escolha entre candidatas): se A passa do usuário, reescreve
Uso: python3 b4_multi.py <split> <modelos> <conds>"""
import json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import b4_llm as L
import jev
from jev import noul, score
from b4_common import (load_points, load_gen, append_gen, budget, feats, words, h01, PERSONA, STATIC, BOT, USER, LLMISH_RE,
                       PERF, LAUGH, EMOJI, msgs)
from b4_gen import clean, parse_json

CTX = 6
EDIT_SYS = ("You edit text messages. You will get a short chat context and a draft message written by Sam. "
            "Follow the instructions exactly and output ONLY the final message, nothing else.")


def ctx_text(p, n=CTX):
    h = p["history"][-n:]
    return "\n".join(f"{x['who']}: {x['text']}" for x in h)


def style_rules(b):
    r = []
    r.append("end with one short, specific question" if b["q"] else "no question")
    if not b["excl"]:
        r.append("no exclamation marks")
    if not b["emoji"]:
        r.append("no emoji")
    r.append(f"you may laugh like '{b['laugh_token']}' if it fits" if b["laugh"] else "no laughing (no haha/lol)")
    st = []
    if b["lower"]:
        st.append("all lowercase")
    if b["noperiod"]:
        st.append("no final period")
    return r + st


def jask(state, qs):
    t0 = time.time()
    try:
        a = jev.ask(state, qs)
    except Exception as e:
        print("jev error", str(e)[:200]); return None, 0
    dt = time.time() - t0
    return a, (dt if dt > 0.05 else 0.55)


def lat(r):
    return r.get("latency") or 0


# ---------------------------------------------------------------- D5 / D5x
def d5(p, model, draft, numeric=True):
    b = budget(p)
    if numeric:
        instr = (f"Rewrite the draft the way Sam would actually send it on WhatsApp to a close friend: about {b['words']} words, "
                 f"one idea, " + ", ".join(style_rules(b)) + ". Keep what matters in the draft, drop the rest.")
    else:
        instr = "Rewrite the draft the way Sam would actually send it on WhatsApp to a close friend."
    user = f"Chat so far:\n{ctx_text(p)}\n\nSam's draft reply (what Sam wants to say):\n{draft}\n\n{instr}\nOutput only the message."
    r = L.chat([{"role": "system", "content": EDIT_SYS}, {"role": "user", "content": user}], model=model, max_tokens=200,
               temperature=0.5, tag="D5" if numeric else "D5x")
    return {"text": clean(r.get("text")), "cost": r.get("cost", 0), "latency": lat(r), "llm_calls": 1, "jev_calls": 0}


# ---------------------------------------------------------------- RW: diagnóstico + reescrita cirúrgica
DIAG_Q = {
    "paraphrase": noul(f"Does `candidate` restate or paraphrase what {USER} just said?"),
    "overvalidation": noul(f"Does `candidate` praise, validate, reassure or empathize more than the situation calls for?"),
    "forced": noul("Does `candidate` use slang, emojis, internet expressions or enthusiasm in a forced, exaggerated or caricatured way?"),
    "formal": noul("Is `candidate` too formal, polished, elaborate or assistant-like for this casual chat?"),
    "generic_q": noul("Does `candidate` end with a generic question just to keep the chat going (like 'what about you?', "
                      "'how was your day?', 'what are you up to?')?"),
}
TH_DIAG = {"paraphrase": 0.5, "overvalidation": 0.35, "forced": 0.35, "formal": 0.3, "generic_q": 0.5}


def diagnose(p, text, b):
    f = feats(text, p)
    orders = []
    if f["words"] > 1.6 * b["words"] + 2:
        orders.append(f"cut it to about {b['words']} words, keeping only the reaction to what {USER} said")
    if f["q"] and not b["q"]:
        orders.append("remove the question")
    if f["excl"] and not b["excl"]:
        orders.append("remove the exclamation marks")
    if f["emoji"] and not b["emoji"]:
        orders.append("remove the emoji")
    if f["laugh"] and not b["laugh"]:
        orders.append("remove the laughing (haha/lol/😂)")
    hits = [w for w, rx in LLMISH_RE if rx.search(text)]
    if hits:
        found = []
        for w, rx in LLMISH_RE:
            m = rx.search(text)
            if m:
                found.append(m.group(0))
        orders.append("replace or drop these phrases: " + ", ".join(f"'{x}'" for x in found[:4]))
    if f["perf"]:
        m = PERF.search(text)
        orders.append(f"don't open with '{m.group(1)}'")
    h = p["history"][-CTX:]
    st = {"conversation": [{"from": x["who"], "text": x["text"]} for x in h[:-1]],
          "last_message": {"from": USER, "text": h[-1]["text"]}, "candidate": text}
    a, dt = jask(st, DIAG_Q)
    if a:
        if a["paraphrase"]["noul"] > TH_DIAG["paraphrase"]:
            orders.append(f"don't repeat what {USER} said")
        if a["overvalidation"]["noul"] > TH_DIAG["overvalidation"]:
            orders.append("drop the validation/reassurance, react to the facts")
        if a["forced"]["noul"] > TH_DIAG["forced"]:
            orders.append("drop the forced slang and enthusiasm")
        if a["formal"]["noul"] > TH_DIAG["formal"]:
            orders.append("make it casual, fewer words")
        if a["generic_q"]["noul"] > TH_DIAG["generic_q"] and not any("question" in o for o in orders):
            orders.append("remove the generic question at the end")
    diag = {k: round(v["noul"], 3) for k, v in (a or {}).items()}
    return orders, diag, dt


def rw(p, model, draft, max_rounds=3):
    b = budget(p)
    text, cost, latency, llm_calls, jev_calls, hist = draft, 0.0, 0.0, 0, 0, []
    for rnd in range(max_rounds + 1):
        orders, diag, dt = diagnose(p, text, b)
        jev_calls += 1; latency += dt
        hist.append({"round": rnd, "orders": orders, "diag": diag})
        if not orders or rnd == max_rounds:
            break
        st = []
        if b["lower"]:
            st.append("keep it all lowercase")
        if b["noperiod"]:
            st.append("no final period")
        user = (f"Chat so far:\n{ctx_text(p)}\n\nSam's draft reply:\n{text}\n\nEdit the draft minimally:\n" +
                "\n".join(f"- {o}" for o in orders + st) + "\nChange nothing else. Output only the edited message.")
        r = L.chat([{"role": "system", "content": EDIT_SYS}, {"role": "user", "content": user}], model=model, max_tokens=200,
                   temperature=0.3, tag=f"RW{rnd}")
        llm_calls += 1; cost += r.get("cost", 0); latency += lat(r)
        new = clean(r.get("text"))
        if not new:
            break
        text = new
    return {"text": text, "cost": cost, "latency": latency, "llm_calls": llm_calls, "jev_calls": jev_calls,
            "rounds": llm_calls, "rw_hist": hist}


# ---------------------------------------------------------------- PL2: plano -> código -> Jev -> texto
PLAN_ONLY = (PERSONA + "\n\nDon't write the message yet. Plan it. Output ONLY a JSON object:\n"
             '{"move": one of ["react_only","answer","tease_back","joke_riff","empathize","reassure","share_own","ask_follow_up","agree","disagree","plan","goodbye","greet_back"],\n'
             ' "idea": the ONE thing Sam wants to get across (max 10 words),\n'
             ' "length_words": how many words a real friend would text here,\n'
             ' "question": true or false}')
PLAN_Q = {
    "sensible": noul(f"Is `plan_idea` a sensible thing for {BOT} to say in reply to `last_message`?"),
    "specific": noul(f"Is `plan_idea` about the specific content of `last_message`, rather than generic cheering, sympathy or small talk?"),
}


def pl2(p, model):
    b = budget(p)
    cost, latency, llm_calls, jev_calls, tries = 0.0, 0.0, 0, 0, []
    plan = None
    for att in range(2):
        r = L.chat(msgs(p, PLAN_ONLY), model=model, max_tokens=150, temperature=0.8, seed=att, tag="PL2plan")
        llm_calls += 1; cost += r.get("cost", 0); latency += lat(r)
        js = parse_json(r.get("text")) or {}
        idea = str(js.get("idea") or "").strip()
        if not idea:
            tries.append({"att": att, "plan": js, "ok": False}); continue
        h = p["history"][-CTX:]
        st = {"conversation": [{"from": x["who"], "text": x["text"]} for x in h[:-1]],
              "last_message": {"from": USER, "text": h[-1]["text"]}, "plan_idea": idea}
        a, dt = jask(st, PLAN_Q)
        jev_calls += 1; latency += dt
        ok = a is None or a["sensible"]["noul"] >= 0.5
        tries.append({"att": att, "plan": js, "sensible": a and round(a["sensible"]["noul"], 3),
                      "specific": a and round(a["specific"]["noul"], 3), "ok": ok})
        plan = js
        if ok:
            break
    if plan is None:
        plan = {"idea": "", "move": "react_only", "length_words": b["words"], "question": False}
    # código fixa os números: tamanho <= 1,3x o alvo do momento; pergunta só se o plano E o orçamento pedirem
    try:
        lw = int(plan.get("length_words") or b["words"])
    except Exception:
        lw = b["words"]
    n = max(1, min(lw, int(round(1.3 * b["words"]))))
    q = bool(plan.get("question")) and b["q"]
    rules = [f"about {n} words", "end with one short question" if q else "no question"] + style_rules(b)[1:]
    sys_ = (STATIC + "\n\nFor your next message:\n" + (f"Say this: {plan.get('idea')}\n" if plan.get("idea") else "") +
            "Rules: " + "; ".join(rules) + f".\nIt must make sense as a direct reply to {USER}'s last message.")
    r = L.chat(msgs(p, sys_), model=model, max_tokens=150, temperature=0.8, tag="PL2text")
    llm_calls += 1; cost += r.get("cost", 0); latency += lat(r)
    return {"text": clean(r.get("text")), "cost": cost, "latency": latency, "llm_calls": llm_calls, "jev_calls": jev_calls,
            "plan": plan, "plan_tries": tries, "plan_n": n, "plan_q": q}


# ---------------------------------------------------------------- INT: intensidade
INT_LEVELS = ["flat or neutral", "mild", "clearly emotional or excited", "very emotional or excited",
              "extremely intense, gushing or dramatic"]
INT_DESC = {0: "flat and neutral", 1: "mild, low-key", 2: "moderately expressive", 3: "quite expressive", 4: "very expressive"}


def int_call(p, cands):
    h = p["history"][-CTX:]
    st = {"conversation": [{"from": x["who"], "text": x["text"]} for x in h[:-1]],
          "last_message": {"from": USER, "text": h[-1]["text"]}, "candidates": {f"c{i+1}": c for i, c in enumerate(cands)}}
    Q = {"user": score(f"How emotionally intense or excited is `last_message`?", INT_LEVELS)}
    for i in range(len(cands)):
        Q[f"c{i+1}"] = score(f"How emotionally intense or excited is `candidates.c{i+1}`?", INT_LEVELS)
    return jask(st, Q)


def intensity(p, model, G, select=True):
    b = budget(p)
    srcs = ["A", "A_s1", "A_s2", "A_s3"] if select else ["A"]
    recs = [G.get((model, s, p["id"])) for s in srcs]
    recs = [r for r in recs if r and r.get("text")]
    if not recs:
        return None
    cands = [r["text"] for r in recs]
    cost = sum(r["cost"] for r in recs)
    latency = max(lat(r) for r in recs)  # candidatas em paralelo
    a, dt = int_call(p, cands)
    latency += dt
    if a is None:
        return {"text": cands[0], "cost": cost, "latency": latency, "llm_calls": len(recs), "jev_calls": 1}
    u = a["user"]["score"]
    ci = [a[f"c{i+1}"]["score"] for i in range(len(cands))]
    ok = [i for i in range(len(cands)) if ci[i] <= u + 0.25]
    out = {"user_int": round(u, 3), "cand_int": [round(x, 3) for x in ci], "llm_calls": len(recs), "jev_calls": 1}
    if ok:
        i = min(ok, key=lambda i: abs(len(words(cands[i])) - b["words"]))
        return dict(out, text=cands[i], cost=cost, latency=latency, chosen=i, rewritten=False)
    i = min(range(len(cands)), key=lambda i: ci[i])
    lvl = max(0, min(4, int(round(u))))
    user = (f"Chat so far:\n{ctx_text(p)}\n\nSam's draft reply:\n{cands[i]}\n\nTone it down so its energy is at most "
            f"{INT_DESC[lvl]}, like {USER}'s last message: fewer intensifiers, no exclamation marks, no gushing. "
            f"Keep the content. Output only the edited message.")
    r = L.chat([{"role": "system", "content": EDIT_SYS}, {"role": "user", "content": user}], model=model, max_tokens=200,
               temperature=0.3, tag="INTrw")
    return dict(out, text=clean(r.get("text")) or cands[i], cost=cost + r.get("cost", 0), latency=latency + lat(r),
                llm_calls=len(recs) + 1, chosen=i, rewritten=True)


def run(split, models, conds, workers=4):
    pts = [p for p in load_points() if split == "all" or p["split"] == split]
    for model in models:
        G = load_gen()
        for cond in conds:
            todo = [p for p in pts if (model, cond, p["id"]) not in G and (model, "A", p["id"]) in G]
            if not todo:
                continue
            def one(p):
                A = G[(model, "A", p["id"])]
                try:
                    if cond == "D5":
                        o = d5(p, model, A["text"], True)
                    elif cond == "D5x":
                        o = d5(p, model, A["text"], False)
                    elif cond == "RW":
                        o = rw(p, model, A["text"])
                    elif cond == "PL2":
                        o = pl2(p, model)
                    elif cond == "INT":
                        o = intensity(p, model, G, True)
                    elif cond == "INTr":
                        o = intensity(p, model, G, False)
                    else:
                        raise ValueError(cond)
                except Exception as e:
                    print("err", cond, p["id"], str(e)[:200]); return None
                if o is None:
                    return None
                if cond in ("D5", "D5x", "RW"):  # o rascunho A faz parte do custo e da latência
                    o["cost"] += A["cost"]; o["latency"] += (A["latency"] or 0); o["llm_calls"] += A["llm_calls"]
                return dict(o, model=model, cond=cond, pid=p["id"], split=p["split"])
            with ThreadPoolExecutor(workers) as ex:
                recs = [r for r in ex.map(one, todo) if r]
            append_gen(recs)
            print(model, cond, split, "n", len(recs), "llm total", round(L.stats["cost"], 4), "jev", jev.stats["calls"],
                  round(jev.stats["cost"], 4), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","))
