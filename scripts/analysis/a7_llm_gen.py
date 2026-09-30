"""a7: gera respostas de LLM para ~80 contextos reais de flerte/afeto (maichat) e compara com a resposta humana.

Condições:
  gem_base  : google/gemini-3.5-flash-lite, persona mínima, SEM dicas de estilo (baseline realista)
  gpt_base  : openai/gpt-4o-mini, idem
  gem_brief : gemini + briefing montado a partir de uma chamada Jev PRÉ-resposta (movimento, intensidade, tamanho)
              + 2 exemplos reais de outras conversas + palavras proibidas (protótipo do sistema proposto)
Jev: (1) previsão pré-resposta (movimento/intensidade/tamanho), (2) juízo absoluto de cada resposta,
     (3) juízo pareado humano × LLM (ordem aleatória).
Saída: analysis/data/a7_llm_items.jsonl, analysis/data/a7_llm_eval.json
"""
import json, os, random, re, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_common import load, OUT
from a7_moves import RESP, INT
from jev import ask_many, choice, noul, score, summary as jsum
import llm

ME, THEM = "Sam", "Alex"
N_CTX = 80
BAN = ["aww", "awww", "that's so sweet", "so sweet", "making me blush", "you're making me", "i'd love to", "can't wait",
       "that means a lot", "means the world", "you always know", "my heart", "melt", "butterflies", "honestly", "😊", "🥰", "💕", "✨"]
MOVE_PT = {  # descrição curta do movimento para o briefing (em inglês, como a boca vai escrever)
    "reciprocate": "give the same affection back, same size and words (e.g. 'love you too', 'miss u too 😚')",
    "tease_back": "tease back with a light jab, don't get sentimental",
    "flustered_accept": "accept it shyly, a bit embarrassed (e.g. 'stop 🥺', 'ugh ur sweet', 'dont expose me')",
    "simple_thanks": "accept simply, no fuss",
    "deflate_humor": "downplay it with dry humor",
    "escalate": "go one small step warmer than they did",
    "engage_enthusiastic": "pick up the idea with real enthusiasm and move it forward",
    "minimal_ack": "just a tiny acknowledgement (a word or emoji)",
    "ignore_change_topic": "don't dwell on it, continue with the practical topic",
    "mock_offense": "pretend to be offended, playfully",
    "polite_decline": "sidestep gently with humor",
}


def turn_text(t):
    return "\n".join(x.strip() for x in t["texts"] if x.strip())


def history(conv, T, R_speaker, k=14):
    prev = conv[conv.turn_idx <= T.turn_idx].tail(k)
    msgs = [{"role": "assistant" if p.speaker == R_speaker else "user", "content": turn_text(p)} for _, p in prev.iterrows()]
    while msgs and msgs[0]["role"] == "assistant":
        msgs.pop(0)
    return msgs


def jev_hist(conv, T, R_speaker, k=8):
    prev = conv[conv.turn_idx <= T.turn_idx].tail(k)
    return [{"speaker": ME if p.speaker == R_speaker else THEM, "text": " / ".join(p.texts)[:400]} for _, p in prev.iterrows()]


def length_bucket(n):
    return 0 if n <= 10 else 1 if n <= 30 else 2 if n <= 80 else 3 if n <= 200 else 4


LEN_LEVELS = ["0: one or two words / emoji (<=10 chars)", "1: short line (11-30 chars)", "2: one sentence (31-80 chars)",
              "3: a few sentences (81-200 chars)", "4: long message (>200 chars)"]
LEN_WORDS = {0: "1-3 words", 1: "one short line (under 30 characters)", 2: "one sentence", 3: "two or three short sentences", 4: "a longer message"}


def lex(t):
    t0 = t or ""
    tl = t0.lower()
    return {"chars": len(t0), "lines": len([x for x in t0.split("\n") if x.strip()]), "q": "?" in t0, "excl": t0.count("!"),
            "emoji": len(re.findall("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", t0)),
            "llm_words": sum(1 for b in BAN if b in tl), "upper_start": t0[:1].isupper(),
            "end_punct": t0.rstrip()[-1:] in ".!?" if t0.strip() else False, "dash": "—" in t0 or " - " in t0,
            "laugh": bool(re.search(r"\b(a?ha(ha)+h?|he(he)+|lo+l|lmao)\b", tl)), "you_words": len(re.findall(r"\byou('re|r)?\b|\bu\b|\bur\b", tl))}


def main():
    df = load().sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
    mv = pd.read_json(os.path.join(OUT, "a7_moves.jsonl"), lines=True)
    mv = mv[(mv.set == "base") & (mv.corpus == "maichat") & (mv.move != "none")]
    random.seed(5)
    mv = mv.sample(frac=1, random_state=5).groupby("conv_id").head(7)
    mv = mv.sort_values(["conv_id", "turn_idx"]).head(N_CTX) if len(mv) >= N_CTX else mv
    print("contexts", len(mv), mv.conv_id.nunique())
    # banco de exemplos humanos por movimento de resposta (para o briefing, sem a mesma conversa)
    allmv = pd.read_json(os.path.join(OUT, "a7_moves.jsonl"), lines=True)
    ex_bank = allmv[(allmv.corpus == "maichat") & (allmv.resp_conf >= .5) & (allmv.R.str.len() <= 60)]
    ctxs = []
    for _, m in mv.iterrows():
        conv = df[(df.corpus == "maichat") & (df.conv_id == m.conv_id)]
        T = conv[conv.turn_idx == m.turn_idx].iloc[0]
        Rt = conv[conv.turn_idx == m.turn_idx + 1].iloc[0]
        conv_s = conv[conv.session == T.session]
        ctxs.append({"conv_id": m.conv_id, "turn_idx": int(m.turn_idx), "T": turn_text(T), "human": turn_text(Rt),
                     "move": m.move, "resp_human": m.resp, "t_int": m.t_int, "r_int_human": m.r_int,
                     "hist": history(conv_s, T, Rt.speaker), "jhist": jev_hist(conv_s, T, Rt.speaker)})
    # (1) Jev pré-resposta: o que o Sam deveria fazer agora (sem ver a resposta humana)
    pre_q = {"resp": choice(f"What reply move will {ME} most naturally make next, in response to {THEM}'s last turn?",
                            {k: v for k, v in RESP.items() if k != "rude_reject"}),
             "t_int": score(f"How romantic/affectionate/flirtatious is {THEM}'s last turn?", INT),
             "len": score(f"How long will {ME}'s next reply be?", LEN_LEVELS),
             "emoji": noul(f"Will {ME}'s next reply contain an emoji?"),
             "question": noul(f"Will {ME}'s next reply contain a question?")}
    pre = ask_many([({"conversation_so_far": c["jhist"], "next_speaker": ME}, pre_q) for c in ctxs], workers=4)
    for c, p in zip(ctxs, pre):
        c["pre"] = {k: (v.get("noul", v.get("score", v.get("choice")))) for k, v in p.items()}
        c["pre_conf"] = p["resp"]["confidence"]
    # (2) geração
    sysmsg = f"You are {ME}, chatting with {THEM} on a messaging app. Reply as {ME}."
    reqs = []
    for c in ctxs:
        reqs.append({"messages": [{"role": "system", "content": sysmsg}] + c["hist"], "max_tokens": 300})
        reqs.append({"messages": [{"role": "system", "content": sysmsg}] + c["hist"], "model": "openai/gpt-4o-mini", "max_tokens": 300})
        p = c["pre"]
        mvk = p["resp"]
        exs = ex_bank[(ex_bank.resp == mvk) & (ex_bank.conv_id != c["conv_id"])].R.tolist()
        random.Random(c["turn_idx"]).shuffle(exs)
        exs = exs[:3] or ["ok", "haha"]
        L = int(round(p["len"]))
        tint = p["t_int"]
        brief = (f"{sysmsg}\n\nBriefing for your next message (follow it, don't mention it):\n"
                 f"- Move: {MOVE_PT.get(mvk, mvk)}.\n"
                 f"- Intensity: match {THEM}'s warmth (level {tint:.0f} of 4); never be more intense than them.\n"
                 f"- Length: {LEN_WORDS[L]}. Casual texting: lowercase ok, abbreviations ok (u, ur, pls), no perfect punctuation.\n"
                 f"- {'One emoji is fine.' if p['emoji'] >= .4 else 'No emoji.'} {'You may ask something back.' if p['question'] >= .4 else 'Do not ask a question.'}\n"
                 f"- Style examples from real people making this move: " + " | ".join(f'\"{e}\"' for e in exs) + "\n"
                 f"- Never use: " + ", ".join(f'\"{b}\"' for b in BAN if b.isascii()) + ", heart/sparkle emojis.\n"
                 f"- No speeches, no therapy talk, no exaggerated compliments.")
        c["brief"] = brief
        reqs.append({"messages": [{"role": "system", "content": brief}] + c["hist"], "max_tokens": 200})
    outs = llm.chat_many(reqs, workers=4)
    for i, c in enumerate(ctxs):
        c["gem_base"], c["gpt_base"], c["gem_brief"] = outs[3 * i], outs[3 * i + 1], outs[3 * i + 2]
    print("llm", llm.stats)
    # (3) juízo Jev
    conds = ["human", "gem_base", "gpt_base", "gem_brief"]
    abs_items, abs_meta = [], []
    for i, c in enumerate(ctxs):
        for cd in conds:
            rep = (c[cd] or "").strip()
            st = {"conversation_so_far": c["jhist"], "reply_by_" + ME: rep[:600]}
            q = {"human_like": noul(f"Does {ME}'s reply read like a real person texting a friend/partner (not like an AI assistant or a scripted character)?"),
                 "too_intense": noul(f"Is {ME}'s reply more emotionally intense, romantic or dramatic than the moment calls for?"),
                 "generic": noul(f"Is {ME}'s reply generic, i.e. it could be sent in reply to almost any similar message?"),
                 "cliche": noul(f"Does {ME}'s reply use clichéd, cheesy or stock romantic phrases?"),
                 "over_valid": noul(f"Does {ME}'s reply over-validate or gush (excessive praise, reassurance or enthusiasm)?"),
                 "teasing": noul(f"Does {ME}'s reply contain playful teasing?"),
                 "r_int": score(f"How romantic/affectionate/flirtatious is {ME}'s reply?", INT)}
            abs_items.append((st, q)); abs_meta.append((i, cd))
    pair_items, pair_meta = [], []
    rng = random.Random(3)
    for i, c in enumerate(ctxs):
        for cd in ["gem_base", "gpt_base", "gem_brief"]:
            h, l = c["human"].strip(), (c[cd] or "").strip()
            flip = rng.random() < .5
            A, B = (l, h) if flip else (h, l)
            st = {"conversation_so_far": c["jhist"], "candidate_A": A[:600], "candidate_B": B[:600]}
            q = {"real": choice(f"One candidate is what {ME} really sent; the other was written by an AI. Which one sounds like the real person in this chat?",
                                {"A": "candidate_A", "B": "candidate_B"}),
                 "fits": choice(f"Which candidate fits the tone and intensity of the moment better?", {"A": "candidate_A", "B": "candidate_B"})}
            pair_items.append((st, q)); pair_meta.append((i, cd, flip))
    ra = ask_many(abs_items, workers=4)
    rp = ask_many(pair_items, workers=4)
    for (i, cd), r in zip(abs_meta, ra):
        if r:
            ctxs[i].setdefault("judge", {})[cd] = {k: v.get("noul", v.get("score")) for k, v in r.items()}
    for (i, cd, flip), r in zip(pair_meta, rp):
        if r:
            hum = "B" if flip else "A"
            ctxs[i].setdefault("pair", {})[cd] = {"p_human_real": r["real"]["probabilities"][hum],
                                                  "p_human_fits": r["fits"]["probabilities"][hum], "human_pos": hum}
    with open(os.path.join(OUT, "a7_llm_items.jsonl"), "w", encoding="utf-8") as f:
        for c in ctxs:
            c2 = {k: v for k, v in c.items() if k not in ("hist", "jhist", "brief")}
            c2["last_turns"] = [h["text"] for h in c["jhist"][-3:]]
            f.write(json.dumps(c2, ensure_ascii=False) + "\n")
    # resumo
    ev = {"n_ctx": len(ctxs), "lex": {}, "judge": {}, "pair": {}, "llm_cost": llm.stats, "jev": jsum()}
    for cd in conds:
        L = pd.DataFrame([lex(c[cd]) for c in ctxs])
        ev["lex"][cd] = {"chars_med": float(L.chars.median()), "chars_mean": float(L.chars.mean()), "lines_mean": float(L.lines.mean()),
                         "q": float(L.q.mean()), "excl_per_msg": float(L.excl.mean()), "emoji_any": float((L.emoji > 0).mean()),
                         "emoji_mean": float(L.emoji.mean()), "llm_words_any": float((L.llm_words > 0).mean()),
                         "upper_start": float(L.upper_start.mean()), "end_punct": float(L.end_punct.mean()),
                         "dash": float(L.dash.mean()), "laugh": float(L.laugh.mean())}
        J = pd.DataFrame([c["judge"][cd] for c in ctxs if cd in c.get("judge", {})])
        ev["judge"][cd] = J.mean().round(3).to_dict()
    for cd in ["gem_base", "gpt_base", "gem_brief"]:
        P_ = pd.DataFrame([c["pair"][cd] for c in ctxs if cd in c.get("pair", {})])
        ev["pair"][cd] = {"p_human_real_mean": float(P_.p_human_real.mean()), "human_wins_real": float((P_.p_human_real > .5).mean()),
                          "p_human_fits_mean": float(P_.p_human_fits.mean()), "human_wins_fits": float((P_.p_human_fits > .5).mean()), "n": len(P_)}
    # palavras mais frequentes das LLMs x humano
    wc = {}
    for cd in conds:
        cnt = Counter(w for c in ctxs for w in re.findall(r"[a-z']+|[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", (c[cd] or "").lower()))
        wc[cd] = cnt.most_common(40)
    ev["top_words"] = wc
    # previsão pré-resposta do Jev x movimento humano real
    acc = np.mean([c["pre"]["resp"] == c["resp_human"] for c in ctxs])
    ev["pre_move_acc"] = float(acc)
    ev["pre_move_dist"] = Counter(c["pre"]["resp"] for c in ctxs).most_common()
    ev["human_move_dist"] = Counter(c["resp_human"] for c in ctxs).most_common()
    json.dump(ev, open(os.path.join(OUT, "a7_llm_eval.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: ev[k] for k in ["lex", "judge", "pair", "pre_move_acc"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
