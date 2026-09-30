"""b4 — geração das condições de CONTROLE NA GERAÇÃO (mecanismo 1) e dos baselines, para os 4 atores.
Condições de 1 chamada (o texto final sai direto da LLM):
  A        persona pura (baseline);  A_s1..A_s3 = outras seeds (para "gerar N e cortar" e para o controle de intensidade)
  S        prompt de estilo estático (a9)
  B9       briefing curto do a9 (1ª rodada: "Max N words", "No question" etc.)
  T        briefing de ALVOS (2ª versão: "about N words", pergunta/"!" sorteados com taxa humana, linha positiva de abertura)
           T_s1..T_s3 = outras seeds
  F1       instrução de formato simples: "uma frase, no máximo N palavras" (sem briefing)
  MT       A + max_tokens justo por momento (1,5*N+4 tokens, N = alvo de palavras do ponto)
  MT12     A + max_tokens fixo de 12 tokens (corte cego)
  STnl     A + stop ["\\n"] (só a 1ª linha/bolha)        [luna não aceita stop: emulado em código na análise]
  STq      A + stop ["?"] (para na 1ª interrogação)
  TLO      A + temperatura 0,3 ; THI = A + temperatura 1,2 e top_p 0,9 (onde o provedor aceita)
  PFX      "prefill" emulado por instrução: "Begin your reply with the word 'X'" (X sorteado das aberturas humanas)
  PL1      plano -> texto numa chamada só (JSON com move/idea/length/question e depois text)
  LB       (deepseek) A + logit_bias -100 em todos os tokens com "!" e em totally/amazing/definitely/absolutely/honestly/😂
Uso: python3 b4_gen.py <split> <modelos,...> <conds,...>"""
import json, os, re, sys
import b4_llm as L
from b4_common import (load_points, msgs, budget, PERSONA, STATIC, BOT, USER, h01, load_gen, append_gen, SCR)
from a9_brief import build_brief, moment_of, MOMENT_LBL, MOVE_TXT, TONE_TXT

OPENERS = [("i", 13), ("oh", 10), ("so", 7), ("yes", 6), ("ok", 5), ("same", 4), ("yeah", 4), ("wait", 1), ("no", 2)]


def target_brief(p, b):
    """Briefing de ALVOS (T). Ordens curtas e imperativas; números como alvo com folga, não teto."""
    j = p["jev"]
    L_ = []
    m = moment_of(j)
    L_.append(f"Moment: {MOMENT_LBL[m]}.")
    if j.get("move_conf", 0) >= 0.4:
        L_.append(MOVE_TXT[j["move"]])
    L_.append(f"Length: about {b['words']} words" + (", as 2 short messages, one per line." if b["bubbles"] > 1 else ", one message.")
              + " One idea only.")
    L_.append("End with one short, specific question about what they said." if b["q"] else "No question.")
    if b["laugh"]:
        L_.append(f"You may laugh the way you usually do ('{b['laugh_token']}') if it fits.")
    else:
        L_.append("No laughing.")
    if not b["emoji"]:
        L_.append("No emoji.")
    if not b["excl"]:
        L_.append("No exclamation marks.")
    L_.append("Start plainly (like 'oh', 'ok', 'yeah', 'i', 'so') or go straight to the content. React to what they said, not to the emotion.")
    st = []
    if b["lower"]:
        st.append("all lowercase")
    if b["noperiod"]:
        st.append("no final period")
    if st:
        L_.append("Style: " + "; ".join(st) + ".")
    L_.append(f"It must make sense as a direct reply to {USER}'s last message.")
    return "\n".join(L_)


def pick_opener(p):
    tot = sum(w for _, w in OPENERS)
    x = h01(p["id"], "pfx") * tot
    for o, w in OPENERS:
        x -= w
        if x < 0:
            return o
    return "i"


PLAN_SYS = (PERSONA + "\n\nBefore writing, plan your next message. Output ONLY a JSON object with these keys, in this order:\n"
            '{"move": one of ["react_only","answer","tease_back","joke_riff","empathize","reassure","share_own","ask_follow_up","agree","disagree","plan","goodbye","greet_back"],\n'
            ' "idea": the ONE thing you want to get across (max 10 words),\n'
            ' "length_words": how many words a real friend would text here (typical chat replies are 2-12 words),\n'
            ' "question": true or false (real friends end with a question in only ~1 of 8 messages),\n'
            ' "text": the message itself, exactly as you would send it}')


def parse_json(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def ds_bias():
    return json.load(open(os.path.join(SCR, "..", "b4_ds_logit_bias.json")))


def spec(p, cond, model):
    """-> (kwargs de L.chat, info)"""
    b = budget(p)
    base = dict(model=model, temperature=0.8, max_tokens=300, seed=0, tag=cond)
    seed_of = {"A_s1": 1, "A_s2": 2, "A_s3": 3, "T_s1": 1, "T_s2": 2, "T_s3": 3}
    if cond in ("A", "A_s1", "A_s2", "A_s3"):
        return dict(base, messages=msgs(p, PERSONA), seed=seed_of.get(cond, 0)), {}
    if cond == "S":
        return dict(base, messages=msgs(p, STATIC)), {}
    if cond == "B9":
        return dict(base, messages=msgs(p, STATIC + "\n\nFor your next message:\n" + build_brief(p["jev"], p["fp"], "short"))), {}
    if cond in ("T", "T_s1", "T_s2", "T_s3"):
        return dict(base, messages=msgs(p, STATIC + "\n\nFor your next message:\n" + target_brief(p, b)),
                    seed=seed_of.get(cond, 0)), {}
    if cond == "F1":
        return dict(base, messages=msgs(p, PERSONA + f"\n\nReply in ONE short sentence of at most {b['words']} words.")), {}
    if cond == "MT":
        return dict(base, messages=msgs(p, PERSONA), max_tokens=int(1.5 * b["words"] + 4)), {"max_tokens": int(1.5 * b["words"] + 4)}
    if cond == "MT12":
        return dict(base, messages=msgs(p, PERSONA), max_tokens=12), {}
    if cond == "STnl":
        return dict(base, messages=msgs(p, PERSONA), stop=["\n"]), {}
    if cond == "STq":
        return dict(base, messages=msgs(p, PERSONA), stop=["?"]), {}
    if cond == "TLO":
        return dict(base, messages=msgs(p, PERSONA), temperature=0.3), {}
    if cond == "THI":
        return dict(base, messages=msgs(p, PERSONA), temperature=1.2, top_p=0.9), {}
    if cond == "PFX":
        o = pick_opener(p)
        return dict(base, messages=msgs(p, PERSONA + f"\n\nBegin your reply with the word '{o}'.")), {"opener": o}
    if cond == "PL1":
        return dict(base, messages=msgs(p, PLAN_SYS)), {}
    if cond == "LB":
        return dict(base, messages=msgs(p, PERSONA), logit_bias=ds_bias()), {}
    raise ValueError(cond)


def post(cond, out):
    t = out.get("text")
    info = {}
    if cond == "PL1":
        js = parse_json(t)
        info["plan"] = {k: v for k, v in (js or {}).items() if k != "text"}
        t = (js or {}).get("text") if js else None
        info["plan_ok"] = js is not None
    return t, info


def clean(t):
    t = (t or "").strip()
    t = re.sub(rf"^\s*\**{BOT}\**\s*:\s*", "", t)
    if len(t) > 2 and t[0] == t[-1] == '"':
        t = t[1:-1]
    return re.sub(r"\n{2,}", "\n", t).strip()


def run(split, models, conds, workers=6):
    pts = [p for p in load_points() if split == "all" or p["split"] == split]
    done = load_gen()
    for model in models:
        for cond in conds:
            if cond == "LB" and model != "deepseek":
                continue
            if model == "luna" and cond in ("STnl", "STq", "TLO", "THI"):
                continue  # o gpt-6-luna não aceita stop/temperatura/top_p (descartados pelo OpenRouter): emulado em código
            todo = [p for p in pts if (model, cond, p["id"]) not in done]
            if not todo:
                continue
            specs = [spec(p, cond, model) for p in todo]
            res = L.chat_many([s for s, _ in specs], workers=workers)
            # vazio/erro: 1 nova tentativa com outra seed (a Gemini às vezes devolve vazio)
            for att in (1, 2, 3):
                bad = [i for i, r in enumerate(res) if not (r.get("text") or "").strip()]
                if not bad:
                    break
                def alt(s0):
                    s1 = dict(s0, seed=s0["seed"] + 100 * att)
                    if att >= 2:
                        s1["temperature"] = 1.0
                    if att == 3:  # erro determinístico da Gemini: muda 1 caractere do system
                        s1["messages"] = [dict(s1["messages"][0], content=s1["messages"][0]["content"] + " ")] + s1["messages"][1:]
                    return s1
                rr = L.chat_many([alt(specs[i][0]) for i in bad], workers=workers)
                for i, r in zip(bad, rr):
                    r["retried"] = att
                    res[i] = r
            recs = []
            for p, (s, inf), r in zip(todo, specs, res):
                t, info2 = post(cond, r)
                recs.append({"model": model, "cond": cond, "pid": p["id"], "split": p["split"], "text": clean(t),
                             "raw": r.get("text"), "finish": r.get("finish"), "cost": r.get("cost", 0),
                             "latency": r.get("latency"), "llm_calls": 1 + (r.get("retried") or 0), "jev_calls": 0,
                             "usage": r.get("usage"), **inf, **info2})
            append_gen(recs)
            nb = sum(1 for r in recs if not r["text"])
            print(model, cond, split, "n", len(recs), "empty", nb, "cost", round(sum(r["cost"] for r in recs), 5),
                  "| total", round(L.stats["cost"], 4), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","))
