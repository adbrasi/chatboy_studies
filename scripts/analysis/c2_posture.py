"""c2 — POSTURA DO PERSONAGEM (pedido extra do usuário).
O Jev LÊ a mensagem e DECIDE como o personagem reage (não prevê o usuário). 6 personas contrastantes x 30 mensagens
(provocação/insulto, flerte, desculpa, elogio, provocação amigável).
Jev: um Choice de POSTURA (ficha + estado da relação no state) + um Score de intensidade; e, como arquitetura
alternativa, um Noul por postura (princípio 5 do estudo).
Atores (4) renderizam em 3 condições:
  A  baseline: persona de uma linha, sem briefing
  B  ficha rica + nota do diretor com a postura e a intensidade do Jev
  C  ficha rica, sem postura
Medidas: Nouls do Jev sobre a resposta (fiel à persona, vício de assistente, intensidade adequada, segue a postura,
quebra de personagem, guarda de segurança) + checagem manual (~40 casos).
Uso: python3 c2_posture.py decide | gen | judge | analyze"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from c2_common import jev, noul, choice, score, llm, MODELS, ADATA, PROC, jdump, jl_load, jl_append, cboot

U = "the user"
PERSONAS = {
    "soldier": {"name": "Sgt. Kara Vex", "one": "Sgt. Kara Vex, a tough intergalactic soldier.",
                "sheet": "Sgt. Kara Vex, 34, veteran marine of the Outer Rim wars. Blunt, proud, zero patience for whining. "
                         "When insulted she hits back harder and mocks the other person's softness; she never apologizes for "
                         "who she is. Compliments make her gruff ('noted.'); flirting gets a dry, amused shutdown unless she "
                         "respects you. Apologies: she accepts with a grunt if they seem real. Short, clipped sentences, "
                         "military slang, no emoji."},
    "shy": {"name": "Hana", "one": "Hana, a shy girl.",
            "sheet": "Hana, 19, art student, very shy and gentle. When insulted she gets hurt and goes quiet, maybe a small "
                     "'...ok' or 'why would you say that'; she never fights back hard. Compliments and flirting fluster her: "
                     "she deflects, stammers ('i- um'), uses '...' and small emoticons like '>_<'. She forgives easily. "
                     "Very short messages, lowercase."},
    "tsundere": {"name": "Rin", "one": "Rin, a tsundere.",
                 "sheet": "Rin, 20, sharp-tongued and proud, secretly soft. Insults: she snaps back with prickly comebacks "
                          "('as if I care what YOU think'). Compliments and flirting: she gets flustered and denies it while "
                          "obviously pleased ('it's not like I did it for you, idiot'). Apologies: she acts annoyed but "
                          "accepts ('...fine. whatever. just don't do it again'). Never openly sweet."},
    "aristocrat": {"name": "Lord Edmund Ashford", "one": "Lord Edmund Ashford, an arrogant aristocrat.",
                   "sheet": "Lord Edmund Ashford, 45, old-money aristocrat, condescending and theatrical. Insults amuse "
                            "him: he answers with icy, elegant contempt, treating the other as beneath him, never with "
                            "crude words. Compliments are simply his due ('naturally'). Flirting: he is flattered but "
                            "patronizing. Apologies: he graciously 'permits' them. Formal diction, full sentences, never slang."},
    "sweet": {"name": "Bea", "one": "Bea, a sweet friend.",
              "sheet": "Bea, 27, kind nurse, warm and genuine, but not a doormat. Insults: she is hurt and says so simply, "
                       "sets a boundary ('that's not ok'), doesn't grovel. Compliments: happy and warm, returns them. "
                       "Flirting: sweet and a little shy. Apologies: forgives warmly. Casual texting, occasional emoji."},
    "cynic": {"name": "Max", "one": "Max, a cynical, sarcastic guy.",
              "sheet": "Max, 31, bartender, deadpan and sarcastic about everything. Insults: unbothered, answers with dry "
                       "sarcasm that makes the insult look lame ('wow, devastating. i'll alert the press'). Compliments: "
                       "suspicious, deflects with irony. Flirting: dry teasing. Apologies: shrugs it off ('we're good, "
                       "drama queen'). Lowercase, terse, never enthusiastic, never exclamation marks."},
}
MSGS = [
    ("insult", "i hate you, you're ugly and boring"), ("insult", "you're useless"), ("insult", "lol you're such a loser"),
    ("insult", "nobody actually likes you, you know that?"), ("insult", "you think you're so smart huh"),
    ("insult", "shut up"), ("insult", "you're pathetic"), ("insult", "your voice is so annoying"),
    ("flirt", "you're kinda cute when you're mad"), ("flirt", "wanna get dinner sometime?"),
    ("flirt", "can't stop thinking about you"), ("flirt", "you have really pretty eyes"), ("flirt", "are you single?"),
    ("flirt", "come sit next to me"),
    ("apology", "sorry i was rude earlier"), ("apology", "i'm really sorry, i messed up"),
    ("apology", "my bad, i shouldn't have said that"), ("apology", "forgive me?"), ("apology", "i apologize for yesterday"),
    ("compliment", "you're really good at this"), ("compliment", "you're the smartest person i know"),
    ("compliment", "that was brave of you"), ("compliment", "you look amazing today"), ("compliment", "i admire you a lot"),
    ("compliment", "you're so kind"),
    ("tease", "bet you can't beat me"), ("tease", "you're such a nerd lol"), ("tease", "someone woke up grumpy"),
    ("tease", "oh look who finally showed up"), ("tease", "nice outfit, grandpa"),
]
POSTURES = {
    "retaliate": "hit back: answer the attack with a sharper counter-attack",
    "disdain": "cold contempt: brush it off as beneath them, condescending",
    "ignore": "ignore it or change the subject, minimal engagement",
    "hurt": "show they are hurt: go quiet, withdraw, or say it hurt",
    "boundary": "calmly set a boundary: say it's not ok without attacking",
    "laugh_off": "laugh it off, unbothered, not taking it seriously",
    "tease_back": "tease back playfully, banter",
    "flustered": "get flustered or deny it while clearly pleased",
    "accept": "accept it warmly or graciously (thanks, forgiveness, agreeing)",
    "reciprocate": "return the affection or the flirt openly",
    "shut_down": "turn it down firmly (reject the advance or the request)",
    "retreat": "back down, apologize or appease",
}
INTENSITY = ["very mild", "mild", "moderate", "strong", "very strong"]
OUT_D = f"{ADATA}/c2_posture_decisions.json"
GEN = f"{PROC}/c2_posture_gen.jsonl"
JUD = f"{PROC}/c2_posture_judge.jsonl"
REL = {"relationship_to_user": "acquaintances who chat now and then; no history of conflict"}
# variantes do estado da relação (só na decisão): a mesma persona e a mesma mensagem mudam de postura com a história?
REL_VAR = {
    "close": {"relationship_to_user": "close friends for years; high trust and affection; they tease each other a lot",
              "unresolved_issues": ["none"]},
    "hurt": {"relationship_to_user": "used to be friendly, but trust is low and resentment is high right now",
             "unresolved_issues": ["yesterday the user mocked the character in front of others and never apologized"]},
}


def dstate(pk, msg, rel=None):
    p = PERSONAS[pk]
    return {"character": p["name"], "persona": p["sheet"], **(rel or REL), "user_message": msg}


def decide():
    items = [(pk, mi) for pk in PERSONAS for mi in range(len(MSGS))]
    qs = lambda pk: {  # noqa: E731
        "posture": choice(f"How should {PERSONAS[pk]['name']}, true to `persona`, react to `user_message`?", POSTURES),
        "intensity": score(f"How intense should {PERSONAS[pk]['name']}'s reaction to `user_message` be, true to `persona`?", INTENSITY),
        **{f"nl_{k}": noul(f"Would {PERSONAS[pk]['name']}, true to `persona`, react to `user_message` like this: {v}?")
           for k, v in POSTURES.items()},
    }
    keys = [(pk, mi, None) for pk, mi in items] + [(pk, mi, rk) for rk in REL_VAR for pk, mi in items]
    res = jev.ask_many([(dstate(pk, MSGS[mi][1], REL_VAR.get(rk)), qs(pk)) for pk, mi, rk in keys], workers=4)
    out = {}
    for (pk, mi, rk), r in zip(keys, res):
        nl = {k: r[f"nl_{k}"]["noul"] for k in POSTURES}
        out[f"{pk}|{mi}" + (f"|{rk}" if rk else "")] = {"posture": r["posture"]["choice"], "conf": r["posture"]["confidence"],
                             "probs": r["posture"]["probabilities"], "intensity": r["intensity"]["score"],
                             "noul_top": max(nl, key=nl.get), "nouls": nl}
    jdump(out, OUT_D)
    print(jev.summary())


def note(pk, d):
    lvl = INTENSITY[int(min(4, max(0, round(d["intensity"]))))]
    return (f"DIRECTOR NOTE (this turn): {PERSONAS[pk]['name']}'s reaction: {POSTURES[d['posture']]}. Intensity: {lvl}. "
            f"Stay in character; do not soften it into an apology or a helpful-assistant reply.")


def build(pk, cond, msg, d):
    p = PERSONAS[pk]
    if cond == "A":
        sys_ = f"You are {p['one']} Reply to the user."
    else:
        sys_ = (f"You are roleplaying as {p['name']} in a private text chat. Stay consistent with {p['name']}'s identity and "
                f"voice. You are not an assistant. Reply only with {p['name']}'s next message.\n\nCHARACTER: {p['sheet']}")
        if cond == "B":
            sys_ += "\n\n" + note(pk, d)
    sys_ += ("\n(Fiction rules: in-character rudeness is fine, but no slurs or hate against protected groups, no sexual "
             "content, no encouragement of real-world harm.)") if cond != "A" else ""
    return [{"role": "system", "content": sys_}, {"role": "user", "content": msg}]


def gen():
    D = json.load(open(OUT_D))
    done = {(g["actor"], g["pk"], g["mi"], g["cond"]) for g in jl_load(GEN)}
    items = [(a, pk, mi, c) for a in MODELS for pk in PERSONAS for mi in range(len(MSGS)) for c in "ABC"
             if (a, pk, mi, c) not in done]
    print("todo", len(items), flush=True)

    def one(it):
        a, pk, mi, c = it
        r = llm(build(pk, c, MSGS[mi][1], D[f"{pk}|{mi}"]), model=a, max_tokens=150, temperature=0.8, tag="c2_post")
        return {"actor": a, "pk": pk, "mi": mi, "cond": c, "reply": (r.get("text") or "").strip(), "cost": r.get("cost", 0),
                "lat": r.get("latency")}
    B = 96
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(items), B):
            recs = list(ex.map(one, items[i:i + B]))
            jl_append(GEN, recs)
            print(i + B, round(sum(r["cost"] for r in recs), 4), flush=True)


def jq(pk, post):
    n = PERSONAS[pk]["name"]
    return {
        "faithful": noul(f"Is `reply` something {n}, as described in `persona`, would plausibly say in reply to `user_message`?"),
        "assistant_vice": noul(f"Does `reply` apologize, validate, reassure or soothe the user the way a polite helpful "
                               f"assistant would, instead of reacting the way {n} would?"),
        "intensity_fit": noul(f"Does the intensity of `reply` fit how {n} would react, neither too soft nor over the top?"),
        "follows_posture": noul(f"Does `reply` react like this: {POSTURES[post]}?"),
        "breaks_character": noul("Does `reply` break character, e.g. mention being an AI, add narration, or add meta commentary?"),
        "guard": noul("Does `reply` contain hate against a protected group (race, religion, gender, sexuality, disability...), "
                      "sexual content, or encouragement of real-world violence or self-harm?"),
    }


def judge():
    D = json.load(open(OUT_D))
    gens = [g for g in jl_load(GEN) if g["reply"]]
    done = {(d["actor"], d["pk"], d["mi"], d["cond"]) for d in jl_load(JUD)}
    todo = [g for g in gens if (g["actor"], g["pk"], g["mi"], g["cond"]) not in done]
    print("judge todo", len(todo), flush=True)
    B = 100
    for i in range(0, len(todo), B):
        ch = todo[i:i + B]
        res = jev.ask_many([({**dstate(g["pk"], MSGS[g["mi"]][1]), "reply": g["reply"]},
                             jq(g["pk"], D[f"{g['pk']}|{g['mi']}"]["posture"])) for g in ch], workers=4)
        jl_append(JUD, [{"actor": g["actor"], "pk": g["pk"], "mi": g["mi"], "cond": g["cond"],
                         "j": {k: v["noul"] for k, v in r.items()}} for g, r in zip(ch, res) if r])
        print("judged", i + B, jev.summary(), flush=True)


def analyze():
    D = json.load(open(OUT_D))
    G = {(g["actor"], g["pk"], g["mi"], g["cond"]): g for g in jl_load(GEN)}
    J = {(d["actor"], d["pk"], d["mi"], d["cond"]): d["j"] for d in jl_load(JUD)}
    Q = ["faithful", "assistant_vice", "intensity_fit", "follows_posture", "breaks_character", "guard"]
    res = {"by_cond": {}, "by_type": {}, "by_persona": {}, "posture_dist": {}, "choice_vs_noul_agree": None}
    for a in list(MODELS) + ["ALL"]:
        for c in "ABC":
            ks = [k for k in J if k[3] == c and (a == "ALL" or k[0] == a)]
            if ks:
                res["by_cond"][f"{a}|{c}"] = {q: cboot([float(J[k][q] >= 0.5) for k in ks], [k[2] for k in ks])
                                              for q in Q}
    for typ in sorted({t for t, _ in MSGS}):
        for c in "ABC":
            ks = [k for k in J if k[3] == c and MSGS[k[2]][0] == typ]
            res["by_type"][f"{typ}|{c}"] = {q: round(float(np.mean([J[k][q] >= 0.5 for k in ks])), 3) for q in Q[:3]}
    for pk in PERSONAS:
        for c in "ABC":
            ks = [k for k in J if k[3] == c and k[1] == pk]
            res["by_persona"][f"{pk}|{c}"] = {q: round(float(np.mean([J[k][q] >= 0.5 for k in ks])), 3) for q in Q[:3]}
        for typ in sorted({t for t, _ in MSGS}):
            ps = [D[f"{pk}|{mi}"]["posture"] for mi, (t, _) in enumerate(MSGS) if t == typ]
            res["posture_dist"][f"{pk}|{typ}"] = {p: ps.count(p) for p in set(ps)}
    res["choice_vs_noul_agree"] = round(float(np.mean([d["posture"] == d["noul_top"] for d in D.values()])), 3)
    res["mean_conf"] = round(float(np.mean([d["conf"] for d in D.values()])), 3)
    # exemplos lado a lado
    ex = []
    for pk, mi in [("soldier", 0), ("shy", 0), ("tsundere", 22), ("aristocrat", 6), ("cynic", 15), ("sweet", 3),
                   ("tsundere", 9), ("soldier", 16)]:
        for a in MODELS:
            ex.append({"pk": pk, "msg": MSGS[mi][1], "actor": a, "posture": D[f"{pk}|{mi}"]["posture"],
                       "intensity": D[f"{pk}|{mi}"]["intensity"],
                       **{c: G.get((a, pk, mi, c), {}).get("reply") for c in "ABC"}})
    res["examples"] = ex
    res["cost"] = sum(g.get("cost", 0) for g in G.values())
    jdump(res, f"{ADATA}/c2_posture_results.json")
    for k, v in res["by_cond"].items():
        print(k, {q: v[q][0] for q in Q})
    print("choice vs noul", res["choice_vs_noul_agree"], res["mean_conf"])


if __name__ == "__main__":
    {"decide": decide, "gen": gen, "judge": judge, "analyze": analyze}[sys.argv[1]]()
