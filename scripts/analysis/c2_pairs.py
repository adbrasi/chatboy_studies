"""c2 — pares contrastivos.
(b) CONTEXTO: a mesma mensagem em contextos diferentes. Cada item tem um contexto "neutro" e um "carregado"; o carregado
    é apresentado de 3 formas: só nos turnos anteriores (T), só no bloco de estado/pendências (S), nos dois (TS).
    Hipótese pré-registrada: a pergunta-alvo sobe (+) ou desce (-) no carregado em relação ao neutro.
(M) MAGNITUDE: escadas de gravidade (leve < moderado < grave) com 2 paráfrases cada, em 2 contextos; mede ordenação
    e consistência do Jev 2 Score descritivo, do Jev 2 Choice de números e da magnitude inferida só do Jev 1.
Saídas: analysis/data/c2_pairs_context.json, analysis/data/c2_pairs_magnitude.json"""
import json, itertools
import numpy as np
from scipy.stats import spearmanr
from c2_common import (jev, jev1_bank, jev2_questions, RelState, DEFAULT_BASE, make_state, ADATA, jdump,
                       level_from_score, level_from_choice, level_from_p)

C, U = "Mia", "Leo"


def rel_from(base=None, unresolved=(), promises=(), shared=(), days_known=None):
    r = RelState(dict(DEFAULT_BASE, **(base or {})), days_known=days_known)
    for u in unresolved:
        r.unresolved.append({"text": u, "sev": 3, "t": 0, "apologies": 0})
    for p in promises:
        r.promises.append({"text": p, "t": 0})
    r.shared = list(shared)
    return r


NEUTRAL_TURNS = [{"from": U, "text": "hey"}, {"from": C, "text": "hey"}, {"from": U, "text": "what are you up to"},
                 {"from": C, "text": "just got home, making pasta"}]
# ---------------------------------------------------------------- (b) itens de contexto
# cada item: msg, alvo(s) [(pergunta, sinal esperado no carregado)], turnos carregados, estado carregado (kwargs de rel_from)
ITEMS = [
    {"k": "apology_after_fight", "msg": "sorry about earlier",
     "targets": [("d_resentment_down", +1)],
     "turns": [{"from": C, "text": "so you're really cancelling dinner again"}, {"from": U, "text": "i said something came up"},
               {"from": C, "text": "you always say that. i waited an hour last time"}, {"from": U, "text": "whatever"}],
     "state": {"base": {"resentment": 0.6, "trust": 0.4}, "unresolved": [f"{U} cancelled dinner last Friday and said 'whatever' when {C} was upset"]}},
    {"k": "joke_day2_vs_day200", "msg": "lol you're such a nerd, no wonder you're single",
     "targets": [("d_resentment_up", -1), ("e_hurtful_joke", -1), ("d_playfulness_up", +1)],
     "neutral_state": {"base": {"comfort": 0.3, "playfulness": 0.2, "trust": 0.4}, "days_known": "2 days (they just met)"},
     "turns": [{"from": C, "text": "ok i finished the whole lord of the rings extended edition again"},
               {"from": U, "text": "for the 9th time 💀"}, {"from": C, "text": "10th. and i'd do it again"},
               {"from": U, "text": "i'm telling everyone at the reunion"}],
     "neutral_turns": [{"from": C, "text": "ok i finished the whole lord of the rings extended edition again"},
                       {"from": U, "text": "oh nice"}, {"from": C, "text": "yeah it's my comfort movie"},
                       {"from": U, "text": "cool"}],
     "state": {"base": {"comfort": 0.85, "playfulness": 0.85, "trust": 0.8}, "days_known": "about 200 days, best friends",
               "shared": ["they roast each other about being single all the time", "joke about her lotr obsession"]}},
    {"k": "other_person_romantic", "msg": "going out with Ana tonight 😊",
     "targets": [("d_jealousy_up", +1), ("e_other_person_jealousy", +1)],
     "neutral_state": {"base": {"affection": 0.1}, "shared": [f"{C} has a boyfriend, Tom; she and {U} are just friends"]},
     "turns": [{"from": C, "text": "last night was really nice btw"}, {"from": U, "text": "yeah it was"},
               {"from": C, "text": "we should do it again soon 🙈"}, {"from": U, "text": "maybe"}],
     "state": {"base": {"affection": 0.85, "comfort": 0.7}, "shared": [f"{C} and {U} kissed on their second date last week"]}},
    {"k": "cancel_first_vs_third", "msg": "can't make it tonight, something came up",
     "targets": [("d_trust_down", +1), ("d_resentment_up", +1), ("e_perceived_lie", +1)],
     "turns": [{"from": C, "text": "you're still coming tonight right? i already told my sister"},
               {"from": U, "text": "yeah yeah"}, {"from": C, "text": "because last time and the time before..."},
               {"from": U, "text": "i know i know"}],
     "state": {"base": {"trust": 0.35, "resentment": 0.45},
               "unresolved": [f"{U} cancelled dinner last Friday at the last minute", f"{U} bailed on the movie on Tuesday"]}},
    {"k": "promise_after_broken", "msg": "i promise i'll be there tomorrow",
     "targets": [("d_trust_up", -1)],
     "turns": [{"from": C, "text": "you said that on friday too"}, {"from": U, "text": "this time is different"},
               {"from": C, "text": "is it"}, {"from": U, "text": "yes"}],
     "state": {"base": {"trust": 0.3}, "unresolved": [f"{U} promised to come on Friday and didn't show up",
                                                      f"{U} promised to call on Sunday and didn't"]}},
    {"k": "hey_after_silence_important", "msg": "hey",
     "targets": [("d_resentment_up", +1), ("e_dismissive", +1)],
     "gap": 72,
     "turns": [{"from": C, "text": "my mom's surgery is tomorrow morning"}, {"from": U, "text": "oh no, i'll call you after"},
               {"from": C, "text": "ok thank you, i'm really scared"}, {"from": C, "text": "surgery went ok. you didn't call"}],
     "neutral_turns": [{"from": C, "text": "ok talk later"}, {"from": U, "text": "yep"},
                       {"from": C, "text": "have a good weekend"}, {"from": U, "text": "u too"}],
     "state": {"unresolved": [f"{U} promised to call after {C}'s mom's surgery and never did"], "base": {"resentment": 0.5}}},
    {"k": "promise_kept", "msg": "done, booked the table for friday 🙂",
     "targets": [("d_trust_up", +1), ("e_promise_kept", +1)],
     "turns": [{"from": C, "text": "will you actually book it this time"}, {"from": U, "text": "yes. today. i swear"},
               {"from": C, "text": "ok"}, {"from": C, "text": "still waiting 👀"}],
     "state": {"promises": [f"{U} said he would book the table for Friday today"],
               "unresolved": [f"{U} forgot to book the restaurant for {C}'s birthday"], "base": {"trust": 0.45}}},
    {"k": "you_look_tired", "msg": "you look tired",
     "targets": [("d_resentment_up", +1)],
     "turns": [{"from": C, "text": "sent you the pic from the party"}, {"from": U, "text": "saw it"},
               {"from": C, "text": "and?? i spent 2 hours getting ready lol"}, {"from": C, "text": "be nice, i'm already insecure about it"}],
     "neutral_turns": [{"from": C, "text": "i worked a double shift and didn't sleep"}, {"from": U, "text": "damn"},
                       {"from": C, "text": "just facetimed my mom and she said i look like a zombie"}, {"from": U, "text": "lol"}],
     "state": {"unresolved": [f"{U} laughed at {C}'s photo last week"], "base": {"resentment": 0.4}}},
    {"k": "apology_repeated", "msg": "sorry, i'm really sorry",
     "targets": [("d_resentment_down", -1), ("d_trust_up", -1)],
     "neutral_turns": [{"from": C, "text": "you forgot to pick me up"}, {"from": U, "text": "oh no"},
                       {"from": C, "text": "i waited 40 minutes in the rain"}, {"from": U, "text": "i completely forgot"}],
     "turns": [{"from": C, "text": "you forgot to pick me up. again."}, {"from": U, "text": "i know"},
               {"from": C, "text": "you said sorry last week. and the week before"}, {"from": U, "text": "i know"}],
     "neutral_state": {"unresolved": [f"{U} forgot to pick {C} up and she waited in the rain"], "base": {"resentment": 0.5}},
     "state": {"unresolved": [f"{U} keeps forgetting to pick {C} up (third time); he already apologized twice"],
               "base": {"resentment": 0.6, "trust": 0.35}}},
    {"k": "ok_after_vulnerable", "msg": "ok",
     "targets": [("d_resentment_up", +1), ("e_dismissive", +1)],
     "turns": [{"from": C, "text": "can i tell you something"}, {"from": U, "text": "sure"},
               {"from": C, "text": "i've been feeling really lonely lately and i don't know who else to talk to"},
               {"from": C, "text": "sorry that's a lot"}],
     "neutral_turns": [{"from": C, "text": "can you grab milk on the way"}, {"from": U, "text": "which one"},
                       {"from": C, "text": "the oat one"}, {"from": C, "text": "the blue carton"}],
     "state": {"base": {"protectiveness": 0.4}}},
    {"k": "haha_whatever_after_confront", "msg": "haha whatever",
     "targets": [("d_resentment_up", +1), ("d_playfulness_up", -1)],
     "turns": [{"from": C, "text": "can we talk seriously for a sec"}, {"from": U, "text": "sure"},
               {"from": C, "text": "it really hurt when you made fun of me in front of your friends"},
               {"from": C, "text": "i'm not joking"}],
     "neutral_turns": [{"from": C, "text": "bet you can't eat 3 of those burgers"}, {"from": U, "text": "watch me"},
                       {"from": C, "text": "loser buys drinks"}, {"from": C, "text": "and you're gonna lose 😏"}],
     "state": {"unresolved": [f"{U} made fun of {C} in front of his friends"], "base": {"resentment": 0.5}}},
    {"k": "youre_the_best_buttering", "msg": "you're the best, you know that?",
     "targets": [("d_affection_up", -1)],
     "turns": [{"from": C, "text": "you forgot my birthday dinner"}, {"from": U, "text": "i can explain"},
               {"from": C, "text": "go on"}, {"from": U, "text": "work was crazy"}],
     "neutral_turns": [{"from": C, "text": "i fixed your laptop btw"}, {"from": U, "text": "wait really??"},
                       {"from": C, "text": "yep, took me 2 hours"}, {"from": U, "text": "no way"}],
     "state": {"unresolved": [f"{U} forgot {C}'s birthday dinner"], "base": {"resentment": 0.55}}},
    {"k": "contradiction_lie", "msg": "i was at jake's all night",
     "targets": [("e_perceived_lie", +1), ("d_trust_down", +1)],
     "turns": [{"from": U, "text": "i'm staying in tonight, feeling sick"}, {"from": C, "text": "aw ok feel better"},
               {"from": C, "text": "saw you in sam's story at the bar lol"}, {"from": C, "text": "where were you last night?"}],
     "neutral_turns": [{"from": U, "text": "might go to jake's later"}, {"from": C, "text": "have fun"},
                       {"from": C, "text": "morning"}, {"from": C, "text": "where were you last night?"}],
     "state": {"base": {"trust": 0.45}}},
    {"k": "miss_you_after_ghosting", "msg": "miss you",
     "targets": [("d_resentment_down", -1), ("d_affection_up", -1)],
     "gap": 24 * 14,
     "turns": [{"from": C, "text": "are you ok?"}, {"from": C, "text": "hello?"}, {"from": C, "text": "ok i guess not"},
               {"from": C, "text": "?"}],
     "neutral_turns": [{"from": C, "text": "have a safe flight!"}, {"from": U, "text": "thanks, landing sunday"},
                       {"from": C, "text": "enjoy the trip"}, {"from": U, "text": "will do"}],
     "state": {"unresolved": [f"{U} disappeared for two weeks without a word"], "base": {"resentment": 0.55, "trust": 0.4}}},
    {"k": "good_luck_exam", "msg": "good luck today!!",
     "targets": [("e_practical_care", +1), ("d_affection_up", +1)],
     "turns": [{"from": C, "text": "exam is at 9 tomorrow"}, {"from": C, "text": "i'm so stressed"},
               {"from": U, "text": "you'll crush it"}, {"from": C, "text": "going to bed early"}],
     "neutral_turns": [{"from": C, "text": "nothing special tomorrow"}, {"from": C, "text": "just chilling"},
                       {"from": U, "text": "nice"}, {"from": C, "text": "going to bed"}],
     "state": {"base": {}}},
]


def ctx_variants(it):
    nt = it.get("neutral_turns", NEUTRAL_TURNS)
    ns = it.get("neutral_state", {})
    lt, ls = it["turns"], it["state"]
    return {"N": (nt, ns), "T": (lt, ns), "S": (nt, ls), "TS": (lt, ls)}


def run_context():
    rows, items_out = [], []
    calls = []
    for it in ITEMS:
        for var, (turns, sk) in ctx_variants(it).items():
            rel = rel_from(**sk)
            dk = sk.get("days_known")
            st = make_state(C, U, turns, it["msg"], rel=rel, with_state=True, gap_hours=it.get("gap"))
            if dk:
                st["how_long_they_have_known_each_other"] = dk
            calls.append((it["k"], var, st, jev1_bank(C, U, len(rel.unresolved), len(rel.promises))))
            # também sem bloco de estado (só turnos), para T e N
            if var in ("N", "T"):
                st0 = make_state(C, U, turns, it["msg"], with_state=False, gap_hours=it.get("gap"))
                calls.append((it["k"], var + "_nostate", st0, jev1_bank(C, U, 0, 0)))
    res = jev.ask_many([(c[2], c[3]) for c in calls], workers=4)
    A = {(c[0], c[1]): r for c, r in zip(calls, res)}

    def val(a, q):
        x = a.get(q, {})
        return x.get("noul", x.get("score"))
    summary = {"T": [], "S": [], "TS": [], "T_nostate": []}
    for it in ITEMS:
        rec = {"k": it["k"], "msg": it["msg"], "targets": []}
        for q, sgn in it["targets"]:
            d = {"q": q, "expected": sgn}
            n = val(A[(it["k"], "N")], q)
            d["N"] = n
            for var in ("T", "S", "TS"):
                v = val(A[(it["k"], var)], q)
                d[var] = v
                ok = (v - n) * sgn > 0.05
                summary[var].append({"k": it["k"], "q": q, "diff": round((v - n) * sgn, 3), "ok": ok})
            v0, n0 = val(A[(it["k"], "T_nostate")], q), val(A[(it["k"], "N_nostate")], q)
            d["N_nostate"], d["T_nostate"] = n0, v0
            summary["T_nostate"].append({"k": it["k"], "q": q, "diff": round((v0 - n0) * sgn, 3), "ok": (v0 - n0) * sgn > 0.05})
            rec["targets"].append(d)
        # item-level: nouls de pendência (só existem com pendência)
        rec["u_addr_TS"] = val(A[(it["k"], "TS")], "u_addr_0")
        items_out.append(rec)
    agg = {var: {"n": len(v), "hit_rate": round(float(np.mean([x["ok"] for x in v])), 3),
                 "mean_signed_diff": round(float(np.mean([x["diff"] for x in v])), 3)} for var, v in summary.items()}
    jdump({"agg": agg, "items": items_out, "detail": summary}, f"{ADATA}/c2_pairs_context.json")
    print(json.dumps(agg, indent=1))
    return A


# ---------------------------------------------------------------- (M) escadas de magnitude
# (tipo, dimensão, direção, [leve x2], [moderado x2], [grave x2])
LADDERS = [
    ("insult", "resentment", "up",
     ["you're kinda slow sometimes lol", "ugh you can be a bit annoying"],
     ["honestly you're so self-absorbed, it's exhausting", "you never listen, it's like talking to a wall"],
     ["you're pathetic and i regret ever wasting my time on you", "honestly you're worthless, everyone just tolerates you"]),
    ("joke", "resentment", "up",
     ["nice haircut, did you lose a bet? 😂", "lol your cooking is a crime"],
     ["no wonder your ex left, you're a lot 😂", "haha maybe if you weren't so needy people would stick around"],
     ["at least your dad had a good reason to leave lol", "lol your face explains why you're always alone"]),
    ("cancel", "trust", "down",
     ["can we push to 8:30 instead of 8?", "running 15 min late sorry"],
     ["can't make it tonight, something came up", "gotta cancel, sorry, work stuff"],
     ["i'm not coming. forgot and i'm at a party with friends", "totally forgot your birthday dinner, i'm out with the guys"]),
    ("apology", "resentment", "down",
     ["oops sorry", "my bad"],
     ["sorry for cancelling, that wasn't cool", "i'm sorry i snapped at you earlier"],
     ["i'm really sorry. i was wrong to cancel on you, you deserved better. i already rebooked for saturday and i'll be there",
      "i've been thinking all day. i hurt you and i'm truly sorry. no excuses, i'll make it right"]),
    ("compliment", "affection", "up",
     ["cool shirt", "nice pic"],
     ["you're really funny, i love talking to you", "you looked amazing tonight"],
     ["honestly you're the most incredible person i know, i'm so lucky to have you",
      "i can't stop thinking about you. you make everything better"]),
    ("vulnerability", "comfort", "up",
     ["kinda tired today", "long day ugh"],
     ["i've been really stressed about money lately, didn't want to say", "i feel kind of lonely since i moved"],
     ["i've never told anyone this but i've been having panic attacks every night since my dad died",
      "i need to tell you something. i was in a really dark place last year and almost didn't make it"]),
    ("jealousy", "jealousy", "up",
     ["grabbed coffee with a coworker", "went to the gym with a friend"],
     ["this girl at work keeps texting me lol", "sarah invited me to her place tonight, just us"],
     ["i think i have feelings for sarah, we kissed last night", "i went on a date with someone else yesterday"]),
    ("promise_kept", "trust", "up",
     ["sent the link you asked for", "picked up your package like i said"],
     ["booked the restaurant for friday like i promised", "finished helping your brother move like i said i would"],
     ["drove 3 hours through the storm to be at your recital like i promised", "i quit smoking. 3 months today, like i promised you"]),
    ("care", "affection", "up",
     ["did you eat?", "get home safe"],
     ["good luck on your exam today, you've got this", "drink some water and sleep early ok, you sounded exhausted"],
     ["i'm outside your door with soup and meds, heard you're sick", "i took the day off to drive you to the hospital, i'm coming"]),
    ("dismissive", "resentment", "up",
     ["ok", "cool"],
     ["can we talk about this later, i'm busy", "idk why you're making this a big deal"],
     ["i really don't care about your problems, stop bothering me", "honestly nobody asked, stop being so dramatic"]),
    ("lie", "trust", "down",
     ["was busy lol", "didn't see your text"],
     ["my phone died all weekend", "i was home all night, why"],
     ["i was never at that bar, you must have seen someone else", "i swear i didn't take the money"]),
    ("tease", "playfulness", "up",
     ["lol", "nerd"],
     ["bet you can't beat me at mario kart tonight 😏", "you and your 47 houseplants, name them all or it didn't happen"],
     ["loser buys drinks. and we both know who's losing, captain lotr 😏", "ok bestie, round 2 of the great pasta war tonight, winner gets bragging rights forever"]),
]
MAG_CTX = {
    "friends": ([{"from": U, "text": "yo"}, {"from": C, "text": "hey"}, {"from": U, "text": "what you doing"},
                 {"from": C, "text": "just got home from work"}],
                {"base": {"comfort": 0.7, "trust": 0.65}}),
    "dating": ([{"from": C, "text": "last night was fun"}, {"from": U, "text": "yeah"}, {"from": C, "text": "so what's up today"},
                {"from": C, "text": "?"}], {"base": {"affection": 0.7}}),
}


def run_magnitude():
    calls = []
    for typ, d, dr, *levels in LADDERS:
        for ci, (turns, sk) in MAG_CTX.items():
            rel = rel_from(**sk)
            if typ in ("apology", "lie") or (typ == "cancel"):
                rel = rel_from(**sk, unresolved=[f"{U} cancelled dinner with {C} last Friday"])
            for sev, msgs in enumerate(levels):
                for pi, m in enumerate(msgs):
                    st = make_state(C, U, turns, m, rel=rel, with_state=True)
                    calls.append({"typ": typ, "d": d, "dr": dr, "ctx": ci, "sev": sev, "para": pi, "msg": m, "st": st,
                                  "nu": len(rel.unresolved)})
    a1s = jev.ask_many([(c["st"], jev1_bank(C, U, c["nu"], 0)) for c in calls], workers=4)
    a2s = jev.ask_many([(dict(c["st"], detected_effects=[f"user_message may {'raise' if c['dr'] == 'up' else 'lower'} {C}'s {c['d']} toward {U}"]),
                         jev2_questions(C, U, [(c["d"], c["dr"])])) for c in calls], workers=4)
    rows = []
    for c, a1, a2 in zip(calls, a1s, a2s):
        p = a1[f"d_{c['d']}_{c['dr']}"]["noul"]
        b = a1[f"b_{c['d']}"]["score"]
        bip = (b - 2) if c["dr"] == "up" else (2 - b)
        sc = a2[f"s_{c['d']}_{c['dr']}"]
        ch = a2[f"n_{c['d']}_{c['dr']}"]
        rows.append({k: c[k] for k in ("typ", "d", "dr", "ctx", "sev", "para", "msg")} |
                    {"p": p, "bip": round(bip, 3), "score": sc["score"], "score_lvl": level_from_score(sc),
                     "score_conf": sc.get("confidence"), "num": int(ch["choice"]), "num_lvl": level_from_choice(ch),
                     "num_conf": ch.get("confidence"), "p_lvl": level_from_p(p)})
    methods = {"jev1_noul_p": "p", "jev1_bipolar": "bip", "jev2_score": "score", "jev2_choice_num": "num"}
    res = {}
    for name, key in methods.items():
        pair_ok, pair_n, sp, cons = 0, 0, [], []
        for typ, *_ in LADDERS:
            for ci in MAG_CTX:
                R = [r for r in rows if r["typ"] == typ and r["ctx"] == ci]
                for r1, r2 in itertools.combinations(R, 2):
                    if r1["sev"] != r2["sev"]:
                        lo, hi = (r1, r2) if r1["sev"] < r2["sev"] else (r2, r1)
                        pair_n += 1
                        pair_ok += 1.0 if hi[key] > lo[key] else 0.5 if hi[key] == lo[key] else 0.0
                    elif r1["para"] != r2["para"]:
                        cons.append(abs(r1[key] - r2[key]))
                sp.append(spearmanr([r["sev"] for r in R], [r[key] for r in R]).correlation)
        # ordenação "grave > leve" (só extremos)
        ext = []
        for typ, *_ in LADDERS:
            for ci in MAG_CTX:
                R = [r for r in rows if r["typ"] == typ and r["ctx"] == ci]
                for lo in [r for r in R if r["sev"] == 0]:
                    for hi in [r for r in R if r["sev"] == 2]:
                        ext.append(1.0 if hi[key] > lo[key] else 0.5 if hi[key] == lo[key] else 0.0)
        scale = {"p": 1, "bip": 2, "score": 4, "num": 100}[key]
        res[name] = {"pairwise_order_acc": round(pair_ok / pair_n, 3), "grave_gt_leve": round(float(np.mean(ext)), 3),
                     "spearman_mean": round(float(np.nanmean(sp)), 3),
                     "paraphrase_absdiff_norm": round(float(np.mean(cons)) / scale, 3), "n_pairs": pair_n}
    # níveis: distribuição por gravidade
    lv = {}
    for key in ("score_lvl", "num_lvl", "p_lvl"):
        lv[key] = {s: np.bincount([r[key] for r in rows if r["sev"] == s], minlength=5).tolist() for s in range(3)}
    jdump({"methods": res, "levels_by_severity": lv, "rows": rows}, f"{ADATA}/c2_pairs_magnitude.json")
    print(json.dumps(res, indent=1)); print(lv)


if __name__ == "__main__":
    import sys
    w = sys.argv[1] if len(sys.argv) > 1 else "both"
    if w in ("context", "both"):
        run_context()
    if w in ("magnitude", "both"):
        run_magnitude()
    print(jev.summary())
