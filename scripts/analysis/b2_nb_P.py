"""b2 Parte B, cenário P (texto AINDA não existe: a posição do "cérebro" antes de a LLM escrever).
Prever o nº de bolhas e o ritmo (latência) do próximo turno de S, vendo o contexto até o turno do parceiro.

 P1 bubbles+hist : 1 chamada. state = 8 turnos com bolhas SEPARADAS e o tempo de resposta em palavras + histórico
                   do falante (código). Choice 1..5+ (critérios descritivos) + Score de latência + Noul "responde já?".
 P2 guide        : P1 + GUIA de bolhas (sem tabela de tamanho, pois o texto não existe) + GUIA de tempo de resposta.
 P3 chain        : A = leitura do momento (ansioso, energia, empolgação, zoeira, tensão, está sendo cobrado/acusado,
                   quanto S vai ter a dizer, história em curso, parceiro perguntou, seriedade, fim de conversa) ->
                   B = P2 + `moment_reading` (em palavras) -> mesmas perguntas.
Saída bruta: scratchpad/b2/nb_P_raw.pkl"""
import os
import numpy as np, pandas as pd
from b2_nb_common import *
from jev import ask_many, choice, noul, score, summary

S = build_sample()
T = full_T()
by = {c: g.reset_index(drop=True) for c, g in T.groupby("conv_id")}
N = "`next_speaker`"
LAT = {"maichat": ["replies within about 5 seconds", "replies within 5-15 seconds", "replies within 15-30 seconds", "takes more than 30 seconds"],
       "whatsapp_nl": ["replies within the same minute", "replies within a few minutes", "replies within the hour", "replies hours later"]}

rows = []
for t in S.itertuples():
    conv = by[t.conv_id]; i = int(t.pos)
    ctx, ctxdf = context(conv, i)
    hist, htxt = history(conv, i, t.speaker)
    rows.append({"key": (t.conv_id, t.turn_idx), "corpus": t.corpus, "split": t.split, "conv_id": t.conv_id, "y": t.y,
                 "lat": t.response_latency_s, "ctx": ctx, "hist": hist, "htxt": htxt, "speaker": t.speaker})
R = pd.DataFrame(rows)


def Q(r, extra=""):
    return {"n": choice(f"In how many separate chat messages (bubbles), sent in a row, will {N} send their next turn?{extra}", CRIT),
            "lat": score(f"How long will {N} take to reply to the last message?", LAT[r.corpus]),
            "fast": noul(f"Will {N} reply right away, faster than their usual pace?")}


def st1(r):
    return {"conversation_so_far": r.ctx, "next_speaker": r.speaker, "next_speaker_history": r.htxt}


def st2(r):
    return st1(r) | {"bubble_guide": GUIDE_P, "timing_guide": GUIDE_TIMING}


QA = {"anxious": noul(f"Is {N} anxious, worried or insecure right now?"),
      "energy": score(f"How energetic or excited is {N} right now?", ["very calm", "calm", "moderate", "energetic", "highly agitated or excited"]),
      "excited": noul(f"Is {N} excited or enthusiastic about something right now?"),
      "playful": noul("Are the two people joking around or teasing right now?"),
      "tension": noul("Is there tension, annoyance or conflict between the two people right now?"),
      "challenged": noul(f"Has the other person just accused, criticized or called out {N}, so that {N} will need to defend or explain themselves?"),
      "amount": score(f"How much will {N} have to say in reply to the last message?", ["almost nothing (a reaction)", "one small point", "one full point", "several points", "a lot: many points or a long story"]),
      "story": noul(f"Is {N} in the middle of telling a story or explaining something?"),
      "asked": noul(f"Did the other person just ask {N} a question?"),
      "serious": score("How serious or emotional is this moment?", ["playful banter", "casual", "somewhat serious", "very serious or emotional"]),
      "ending": noul("Is the conversation winding down or about to end?"),
      "needs_thought": noul(f"Does the last message require {N} to think, check something or make a decision before replying?")}


def reading(a):
    lv = ["very calm", "calm", "moderate", "energetic", "highly agitated or excited"]
    sv = ["playful banter", "casual", "somewhat serious", "very serious or emotional"]
    av = ["almost nothing", "one small point", "one full point", "several points", "a lot"]
    return {"energy": lv[int(round(a["energy"]["score"]))], "seriousness": sv[int(round(a["serious"]["score"]))],
            "how_much_to_say": av[int(round(a["amount"]["score"]))],
            **{k: words_of(a[k]["noul"]) for k in ("anxious", "excited", "playful", "tension", "challenged", "story", "asked", "ending", "needs_thought")}}


if __name__ == "__main__":
    out = {}
    for name, fn, extra in (("P1", st1, " Use `next_speaker_history`."), ("P2", st2, " Use `bubble_guide` and `next_speaker_history`.")):
        it = [(fn(r), Q(r, extra)) for r in R.itertuples()]
        print(name, len(it), flush=True)
        out[name] = ask_many(it, workers=4)
    it = [(st1(r) | {"next_speaker_history": None}, QA) for r in R.itertuples()]
    print("A", len(it), flush=True)
    out["A"] = ask_many(it, workers=4)
    it = [(st2(r) | {"moment_reading": reading(a) if a else {}}, Q(r, " Use `bubble_guide`, `next_speaker_history` and `moment_reading` (a reading of the moment made by another model)."))
          for r, a in zip(R.itertuples(), out["A"])]
    print("P3", len(it), flush=True)
    out["P3"] = ask_many(it, workers=4)
    pd.to_pickle({"R": R, "out": out}, os.path.join(SCR, "nb_P_raw.pkl"))
    print(summary())
