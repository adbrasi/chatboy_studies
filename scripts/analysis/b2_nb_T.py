"""b2 Parte B, cenário T (texto conhecido): "em quantas bolhas uma pessoa mandaria ESTE texto?"
O texto é a rajada inicial humana, JUNTADA por espaço (o Jev não vê onde estavam os cortes).

Arquiteturas (todas rodam em dev e teste; a escolha é feita no dev):
 T1 bare      : 1 chamada. state = contexto (8 turnos, bolhas visíveis) + reply_text. Choice 1..5+ com opções secas.
 T2 guide     : 1 chamada. + speaker_history (código) + reply_length (código, em palavras) + GUIA com as taxas do dev.
                Choice com critérios descritivos por opção.
 T3 chain     : 2 chamadas. A = leitura do texto (13 rótulos: energia, ansioso, zoeira, tensão, seriedade, empolgação,
                defensivo, quanto tem a dizer, nº de pontos, pensar em voz alta, história, abre com reação, termina em
                pergunta, momento). B = state do T2 + `reply_reading` (rótulos de A em palavras) -> Choice.
 T4 router    : A (mesma do T3) -> momento -> B = Choice com guia ESPECÍFICO do momento (taxas medidas no dev por
                momento × faixa de tamanho).
 T5 cuts      : 1 chamada. state = mensagem anterior + histórico + `reply_words` (lista). Um Noul por fronteira
                candidata "começaria uma nova bolha em reply_words[i]?". Código: distribuição de n = Poisson-binomial.
 T6 cascade   : Noul ">1 bolha?" -> se sim, nova chamada com a decisão anterior no state -> ">2?" -> ">3?" -> ">4?".
 (T7 ordinal e T8 código ficam em b2_nb_eval.py)
Saída bruta: scratchpad/b2/nb_T_raw.pkl"""
import os, sys, json, re
import numpy as np, pandas as pd
from b2_nb_common import *
from jev import ask_many, choice, noul, score, summary

S = build_sample()
T = full_T()
by = {c: g.reset_index(drop=True) for c, g in T.groupby("conv_id")}
W = "`reply_speaker`"

rows = []
for t in S.itertuples():
    conv = by[t.conv_id]; i = int(t.pos)
    ctx, _ = context(conv, i)
    hist, htxt = history(conv, i, t.speaker)
    text = " ".join(str(x).strip() for x in t.y_texts)
    words = text.split()
    rows.append({"key": (t.conv_id, t.turn_idx), "corpus": t.corpus, "split": t.split, "conv_id": t.conv_id, "y": t.y,
                 "text": text, "words": words, "true_cuts": list(np.cumsum([len(str(x).split()) for x in t.y_texts])[:-1]),
                 "ctx": ctx, "hist": hist, "htxt": htxt, "speaker": t.speaker, "chars": len(text)})
R = pd.DataFrame(rows)
print(len(R), flush=True)


def st_bare(r):
    return {"conversation_so_far": r.ctx, "reply_speaker": r.speaker, "reply_text": r.text}


def st_guide(r):
    return st_bare(r) | {"speaker_history": r.htxt,
                         "reply_length": f"{len_bucket(r.chars)}, {n_sentences(r.text)} sentence(s)",
                         "bubble_guide": GUIDE_T}


QN = f"In how many separate chat messages (bubbles), sent in a row, would {W} send `reply_text`?"
Q_T1 = {"n": choice(QN, CRIT_BARE)}
Q_T2 = {"n": choice(QN + " Use `bubble_guide`, `speaker_history` and `reply_length`.", CRIT)}
Q_A = {
    "energy": score(f"How energetic or excited is {W} in `reply_text`?", ["very calm", "calm", "moderate", "energetic", "highly agitated or excited"]),
    "anxious": noul(f"Is {W} anxious, worried or insecure in `reply_text`?"),
    "playful": noul(f"Is {W} joking, teasing or playing around in `reply_text`?"),
    "tension": noul("Is there tension, annoyance or conflict between the two people right now?"),
    "serious": score("How serious or emotional is `reply_text`?", ["playful banter", "casual", "somewhat serious", "very serious or emotional"]),
    "excited": noul(f"Is {W} excited or enthusiastic about something in `reply_text`?"),
    "defensive": noul(f"Is {W} defending or justifying themselves in `reply_text`?"),
    "amount": score(f"How much does {W} have to say in `reply_text`?", ["almost nothing (a reaction)", "one small point", "one full point", "several points", "a lot: many points or a long story"]),
    "points": choice("How many separate points or ideas does `reply_text` contain?", {"1": "one idea", "2": "two ideas", "3": "three ideas", "4+": "four or more ideas"}),
    "thinking_aloud": noul(f"Is {W} thinking out loud, adding thoughts one after another as they come?"),
    "story": noul(f"Is {W} telling a story or recounting events in `reply_text`?"),
    "opens_reaction": noul("Does `reply_text` begin with a short reaction or interjection (like haha, wait, omg, oh, yes, true, wow) before the main content?"),
    "ends_question": noul("Does `reply_text` end with a question to the other person?"),
    "moment": choice(f"Which best describes the moment of {W}'s reply?", MOMENTS),
}


def reading_words(a):
    lv = ["very calm", "calm", "moderate", "energetic", "highly agitated or excited"]
    sv = ["playful banter", "casual", "somewhat serious", "very serious or emotional"]
    av = ["almost nothing", "one small point", "one full point", "several points", "a lot"]
    return {"energy": lv[int(round(a["energy"]["score"]))], "seriousness": sv[int(round(a["serious"]["score"]))],
            "how_much_to_say": av[int(round(a["amount"]["score"]))], "separate_points": a["points"]["choice"],
            "anxious": words_of(a["anxious"]["noul"]), "joking": words_of(a["playful"]["noul"]),
            "tension": words_of(a["tension"]["noul"]), "excited": words_of(a["excited"]["noul"]),
            "defensive": words_of(a["defensive"]["noul"]), "thinking_out_loud": words_of(a["thinking_aloud"]["noul"]),
            "telling_a_story": words_of(a["story"]["noul"]), "opens_with_quick_reaction": words_of(a["opens_reaction"]["noul"]),
            "ends_with_question": words_of(a["ends_question"]["noul"]), "moment": a["moment"]["choice"]}


def cut_candidates(words, cap=40):
    n = len(words)
    if n <= 1:
        return []
    gaps = list(range(1, n))
    if len(gaps) <= cap:
        return gaps
    conj = {"but", "so", "and", "lol", "haha", "hahaha", "maar", "en", "dus", "want", "omg", "wait", "also", "ok", "oke", "oké", "ja", "nee"}
    sc = []
    for g in gaps:
        prev, nxt = words[g - 1], words[g]
        s = 0
        if re.search(r"[.!?,:;)]$", prev): s += 3
        if nxt[:1].isupper(): s += 2
        if nxt.lower().strip(".,!?") in conj: s += 2
        if re.search(r"[\U0001F300-\U0001FAFF]", prev): s += 2
        sc.append((s, -abs(g - n / 2) * 1e-3, g))
    return sorted(g for _, _, g in sorted(sc, reverse=True)[:cap])


def run(items, label):
    print(label, len(items), flush=True)
    return ask_many(items, workers=4)


if __name__ == "__main__":
    out = {}
    # T1, T2, A (independent) -> run together
    it = [(st_bare(r), Q_T1) for r in R.itertuples()]
    out["T1"] = run(it, "T1")
    it = [(st_guide(r), Q_T2) for r in R.itertuples()]
    out["T2"] = run(it, "T2")
    it = [(st_bare(r), Q_A) for r in R.itertuples()]
    out["A"] = run(it, "A")
    pd.to_pickle({"R": R, "out": out}, os.path.join(SCR, "nb_T_raw.pkl"))
    # T3 chain
    it = []
    for r, a in zip(R.itertuples(), out["A"]):
        st = st_guide(r) | {"reply_reading": reading_words(a) if a else {}}
        it.append((st, {"n": choice(QN + " Use `bubble_guide`, `speaker_history`, `reply_length` and `reply_reading` (a reading of the reply made by another model).", CRIT)}))
    out["T3"] = run(it, "T3")
    # T4 router: moment-specific rates measured on DEV
    R["moment"] = [a["moment"]["choice"] if a else "casual_chat" for a in out["A"]]
    R["lb"] = [len_bucket(c) for c in R.chars]
    dev = R[R.split == "dev"]
    rates = {}
    for m in MOMENTS:
        dm = dev[dev.moment == m]
        rates[m] = {}
        for lb in dev.lb.unique():
            x = dm[dm.lb == lb].y.clip(upper=5)
            if len(x) >= 5:
                vc = x.value_counts(normalize=True)
                rates[m][lb] = ", ".join(f"{vc.get(k, 0):.0%} {LAB[k-1]}" for k in range(1, 6) if vc.get(k, 0) > 0) + f" (n={len(x)})"
        rates[m]["all lengths"] = ", ".join(f"{v:.0%} {LAB[k-1]}" for k, v in dm.y.clip(upper=5).value_counts(normalize=True).sort_index().items()) + f" (n={len(dm)})" if len(dm) else "no data"
    MG = {"banter": "In banter people chop replies into short quick pieces; a joke and its follow-up often go in separate messages.",
          "story_news": "Stories and news come in several messages, one piece at a time, often starting with a hook or reaction.",
          "serious_emotional": "Serious emotional messages are longer and are not chopped into tiny pieces, but a long one may still be 2-3 messages.",
          "conflict_tension": "Annoyed or tense replies are often one curt message; long self-justifications can come in several.",
          "logistics_info": "Practical answers usually fit in one or two messages; a question may be its own message.",
          "quick_reaction": "Quick reactions are almost always one message.",
          "affection_flirt": "Affectionate replies are usually one or two short messages.",
          "casual_chat": "Ordinary replies: splitting depends mostly on how much there is to say."}
    it = []
    for r in R.itertuples():
        st = st_bare(r) | {"speaker_history": r.htxt, "reply_length": f"{len_bucket(r.chars)}, {n_sentences(r.text)} sentence(s)",
                           "moment": f"{r.moment}: {MOMENTS[r.moment]}",
                           "moment_guide": {"how_people_split_in_this_moment": MG[r.moment],
                                            "measured_rates_in_this_moment_by_length": rates[r.moment]}}
        it.append((st, {"n": choice(QN + " Use `moment_guide` for this kind of moment, `speaker_history` and `reply_length`.", CRIT)}))
    out["T4"] = run(it, "T4")
    out["T4_rates"] = rates
    pd.to_pickle({"R": R, "out": out}, os.path.join(SCR, "nb_T_raw.pkl"))
    # T5 cuts (per-gap Nouls in one call)
    it, cands = [], []
    for r in R.itertuples():
        cs = cut_candidates(r.words)
        cands.append(cs)
        prev = r.ctx[-1] if r.ctx else None
        st = {"previous_message": prev, "reply_speaker": r.speaker, "speaker_history": r.htxt, "reply_words": r.words}
        qs = {f"c{g}": noul(f"Would {W}, texting `reply_words` in a chat app, send the words before `reply_words[{g}]` as one message and start a NEW separate message at `reply_words[{g}]`?") for g in cs}
        if not qs:
            qs = {"dummy": noul("Is `reply_words` a chat message?")}
        it.append((st, qs))
    out["T5"] = run(it, "T5")
    out["T5_cands"] = cands
    pd.to_pickle({"R": R, "out": out}, os.path.join(SCR, "nb_T_raw.pkl"))
    # T6 binary cascade (sequential; each step carries the previous decisions in the state)
    Qc = {2: "Would {W} split `reply_text` into more than one separate chat message (bubble) sent in a row?",
          3: "Would {W} split `reply_text` into more than two separate chat messages sent in a row?",
          4: "Would {W} split `reply_text` into more than three separate chat messages sent in a row?",
          5: "Would {W} split `reply_text` into more than four separate chat messages sent in a row?"}
    ps = {i: {} for i in range(len(R))}
    active = list(range(len(R)))
    for k in (2, 3, 4, 5):
        it = []
        for i in active:
            r = R.iloc[i]
            st = st_guide(r)
            if k > 2:
                st = st | {"earlier_decisions": [f"a first check said: more than {j-1} message{'s' if j > 2 else ''} (p={ps[i][j]:.2f})" for j in range(2, k)]}
            it.append((st, {"more": noul(Qc[k].format(W=W))}))
        ans = run(it, f"T6 step >{k-1}")
        nxt = []
        for i, a in zip(active, ans):
            if a is None:
                continue
            ps[i][k] = a["more"]["noul"]
            if ps[i][k] >= 0.5:
                nxt.append(i)
        active = nxt
        if not active:
            break
    out["T6"] = ps
    pd.to_pickle({"R": R, "out": out}, os.path.join(SCR, "nb_T_raw.pkl"))
    print(summary())
