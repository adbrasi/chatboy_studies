"""b2 Parte A (Jev): o que está acontecendo nas rajadas (>=3 / >=5 bolhas em <60 s)?

Amostra estratificada (pesos = N_estrato / n_estrato):
  maichat: todas as rajadas R3 (261) + 450 turnos sem rajada
  whatsapp_nl: 250 R5 (janela de 1 min) + 350 R3-4 + 600 sem rajada
Chamada 1 (leitura): state = 6 turnos anteriores (bolhas visíveis) + `partner_last_turn` + `current_turn` com as bolhas
  JUNTADAS por espaço (o Jev não vê a fragmentação). ~30 perguntas: Choice de contexto principal, Nouls multi-rótulo,
  arousal, e o detector "defensivo / pego no flagra" decomposto (o outro acusou? está se justificando? pego? nega?).
Chamada 2 (cascata, só se o detector acendeu): state = o mesmo + os rótulos da chamada 1 + GUIA dos tipos de defensiva
  -> Choice do tipo (mentira, erro/esquecimento, criticado pelo comportamento, acusação de brincadeira, mal-entendido, não).
Chamada 3 (reação do outro): state = contexto + turno + `partner_next_turn` -> Choice do que o outro fez.
Saída: analysis/data/b2_rajadas_items.csv.gz (+ json de resumo em b2_rajadas_analysis.py)"""
import json, os, sys
import numpy as np, pandas as pd
from b2_common import load, fmt_turn, OUT, SCR
from jev import ask_many, choice, noul, score, summary

T = pd.read_pickle(os.path.join(SCR, "b2_turns_feat.pkl"))
T = T.sort_values(["conv_id", "turn_idx"]).reset_index(drop=True)
by = {c: g.reset_index(drop=True) for c, g in T.groupby("conv_id")}
T["pos"] = T.groupby("conv_id").cumcount()
T["has_ctx"] = T.turn_in_session >= 1
T["stratum"] = np.where(T.max60 >= 5, "R5", np.where(T.max60 >= 3, "R3_4", "none"))

rng = np.random.default_rng(11)
plan = {("maichat", "R5"): None, ("maichat", "R3_4"): None, ("maichat", "none"): 450,
        ("whatsapp_nl", "R5"): 250, ("whatsapp_nl", "R3_4"): 350, ("whatsapp_nl", "none"): 600}
parts = []
for (c, s), n in plan.items():
    pool = T[(T.corpus == c) & (T.stratum == s) & T.has_ctx]
    N = int(((T.corpus == c) & (T.stratum == s)).sum())
    take = pool if n is None or n >= len(pool) else pool.sample(n, random_state=int(rng.integers(1e6)))
    parts.append(take.assign(w=N / len(take)))
S = pd.concat(parts)
print(S.groupby(["corpus", "stratum"]).size(), flush=True)

CONTEXTS = {
    "excitement_news": "sharing exciting news or reacting with excitement / enthusiasm",
    "storytelling": "telling a story or recounting something that happened to them",
    "gossip_third_party": "talking about what a third person did or said (gossip)",
    "defending_justifying": "defending or justifying themselves, explaining why they did something, making excuses",
    "arguing_accusing": "arguing with, accusing or criticizing the other person",
    "anxiety_insecurity": "anxious, worried or insecure, seeking reassurance",
    "apologizing": "apologizing to the other person",
    "flirting_affection": "flirting, being romantic or affectionate",
    "banter_joking": "joking, teasing, playing around",
    "logistics_planning": "arranging practical things: time, place, plans, errands",
    "self_correction_clarifying": "correcting or clarifying something they themselves just said",
    "venting_complaining": "venting or complaining about something or someone other than the other person",
    "comforting_supporting": "comforting, reassuring or supporting the other person",
    "opinion_discussion": "explaining a view or giving opinions in a discussion",
    "answering_info": "answering a question with information",
    "reacting_surprise": "reacting with surprise or shock to what the other said",
    "small_talk_other": "light small talk or anything else",
}
W = "the speaker of `current_turn`"
Q1 = {"main_context": choice(f"What is {W} mainly doing in `current_turn`?", CONTEXTS),
      "arousal": score(f"How emotionally activated or energetic is {W}?", ["very calm", "calm", "moderate", "energetic", "highly agitated or excited"]),
      "amount_to_say": score(f"How much does {W} have to say in `current_turn`?", ["almost nothing (a reaction)", "one small point", "one full point", "several points", "a lot: many points or a long story"]),
      "p_partner_challenges": noul(f"Does `partner_last_turn` accuse, criticize, confront or doubt {W} (their honesty, behavior or something they did or did not do)?"),
      "p_defends_self": noul(f"In `current_turn`, is {W} defending or justifying themselves, e.g. explaining why they did something, making excuses, or pushing back on blame?"),
      "p_caught_out": noul(f"Is {W} reacting to having been caught or called out, e.g. for a lie, a mistake, a contradiction, or something they did or forgot?"),
      "p_denies": noul(f"Does {W} deny or reject something they are accused of or that is said about them?"),
      "p_over_explains": noul(f"Does {W} give more explanation or detail than the situation needs (over-explaining)?"),
      "p_playful_accusation": noul("Is any accusation or criticism between the two people here clearly playful or joking rather than serious?"),
      "p_thinking_aloud": noul(f"Is {W} thinking out loud, adding thoughts one after another as they come?"),
      "p_urgent": noul(f"Is {W} in a hurry or is something urgent?"),
      "p_anger": noul(f"Is {W} angry or irritated?"),
      }
for k, v in CONTEXTS.items():
    Q1["p_" + k] = noul(f"Is {W} {v}?")
Q1["p_excited_positive"] = noul(f"Is {W} excited in a positive way?")

GUIDE = {
    "caught_in_lie": "the other person pointed out that the speaker lied, hid something or said something untrue, and the speaker reacts to that",
    "caught_in_mistake_or_forgetting": "the speaker forgot, was late, did something wrong or made a mistake and is reacting to having it pointed out",
    "criticized_for_behavior": "the other person seriously criticized or blamed the speaker's behavior, opinion or choices and the speaker pushes back or explains",
    "playful_accusation": "the 'accusation' is banter or teasing between people who are joking, and the speaker plays along or jokingly defends themselves",
    "misunderstanding_clarification": "the speaker clarifies a misunderstanding about what they meant, without real blame involved",
    "not_defensive": "the speaker is not defending or justifying themselves at all",
}
Q2 = {"defense_type": choice("Using `defense_guide`, which description best fits `current_turn`?", GUIDE)}

REACT = {
    "engages_content": "responds to the content with their own substantive comment or information",
    "short_ack": "short acknowledgment or reaction only (ok, haha, wow, true)",
    "laughs_along": "laughs along or continues the joke",
    "asks_follow_up": "asks a follow-up question about what was said",
    "reassures_comforts": "reassures, comforts or calms the speaker",
    "accepts_excuse": "accepts the explanation or apology, lets it go",
    "pushes_back": "disagrees, keeps accusing or counter-attacks",
    "matches_energy": "matches the energy with an equally excited or long reply",
    "changes_topic": "changes the topic or ignores what was said",
}
Q3 = {"partner_reaction": choice("How does the other person react in `partner_next_turn` to `current_turn`?", REACT)}


def states(t):
    conv = by[t.conv_id]
    i = int(t.pos)
    ctx = conv.iloc[max(0, i - 6):i]
    ctx = ctx[ctx.session == t.session]
    prev = [fmt_turn(x) for x in ctx.itertuples()]
    cur = fmt_turn(t, bubbles=False)
    st = {"previous_turns": prev[:-1] if len(prev) > 1 else [], "partner_last_turn": prev[-1] if prev else None,
          "current_turn": cur}
    nxt = conv.iloc[i + 1] if i + 1 < len(conv) and conv.iloc[i + 1].session == t.session else None
    return st, (fmt_turn(nxt, bubbles=False) if nxt is not None else None)


rows, items1 = [], []
for t in S.itertuples():
    st, nxt = states(t)
    items1.append((st, Q1))
    rows.append({"corpus": t.corpus, "conv_id": t.conv_id, "turn_idx": t.turn_idx, "stratum": t.stratum, "w": t.w,
                 "_st": st, "_nxt": nxt})
print("call1", len(items1), flush=True)
A1 = ask_many(items1, workers=4)
flat = []
for r, a in zip(rows, A1):
    if a is None:
        continue
    d = {k: v for k, v in r.items() if not k.startswith("_")}
    for k, v in a.items():
        d[k] = v.get("noul", v.get("score", v.get("choice")))
        if v["type"] == "choice":
            d[k + "_conf"] = v["confidence"]
            for o, p in v["probabilities"].items():
                d[f"{k}__{o}"] = p
    d["_st"], d["_nxt"] = r["_st"], r["_nxt"]
    flat.append(d)
R = pd.DataFrame(flat)
# cascade: detector fired?
fire = (R.p_defends_self >= 0.3) | (R.p_caught_out >= 0.3) | (R.p_denies >= 0.3) | (R.p_partner_challenges >= 0.5)
items2, idx2 = [], []
for i, r in R[fire].iterrows():
    st = dict(r["_st"])
    st["first_pass_labels"] = {"other_person_challenged_speaker": round(r.p_partner_challenges, 2),
                               "speaker_defends_or_justifies": round(r.p_defends_self, 2),
                               "speaker_caught_out": round(r.p_caught_out, 2), "speaker_denies": round(r.p_denies, 2),
                               "accusation_is_playful": round(r.p_playful_accusation, 2)}
    st["defense_guide"] = GUIDE
    items2.append((st, Q2)); idx2.append(i)
print("call2 (cascade)", len(items2), flush=True)
A2 = ask_many(items2, workers=4)
for i, a in zip(idx2, A2):
    if a is None:
        continue
    R.loc[i, "defense_type"] = a["defense_type"]["choice"]
    R.loc[i, "defense_type_conf"] = a["defense_type"]["confidence"]
    for o, p in a["defense_type"]["probabilities"].items():
        R.loc[i, f"defense_type__{o}"] = p
# partner reaction: rajadas + controls with >40 chars
T_ix = T.set_index(["conv_id", "turn_idx"])
R["total_chars"] = [T_ix.loc[(c, ti), "total_chars"] for c, ti in zip(R.conv_id, R.turn_idx)]
sel = R[R._nxt.notna() & ((R.stratum != "none") | (R.total_chars > 40))]
items3, idx3 = [], []
for i, r in sel.iterrows():
    st = dict(r["_st"]); st["partner_next_turn"] = r["_nxt"]
    items3.append((st, Q3)); idx3.append(i)
print("call3 (reaction)", len(items3), flush=True)
A3 = ask_many(items3, workers=4)
for i, a in zip(idx3, A3):
    if a is None:
        continue
    R.loc[i, "partner_reaction"] = a["partner_reaction"]["choice"]
    R.loc[i, "partner_reaction_conf"] = a["partner_reaction"]["confidence"]
R["text"] = [r["current_turn"]["text"][:400] for r in R._st]
R["partner_last"] = [json.dumps(r["partner_last_turn"], ensure_ascii=False)[:300] for r in R._st]
R["next"] = [json.dumps(n, ensure_ascii=False)[:300] if n else None for n in R._nxt]
R.drop(columns=["_st", "_nxt"]).to_csv(os.path.join(OUT, "b2_rajadas_items.csv.gz"), index=False)
print(summary())
