"""a3: classifica a 1ª resposta do ouvinte (B) no EmpatheticDialogues com o Jev (mesma amostra estratificada
de 608 conversas do a3_emotion_jev). Choice da estratégia principal e da PRIMEIRA estratégia + Nouls multi-rótulo.
Saída: SCR/listener_raw.jsonl"""
import json, os, random, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from a3_common import *
from jev import ask_many, choice, noul, summary

STRAT = {
    "exploratory_question": "asks A for more details, feelings or what happened next",
    "validation_empathy": "acknowledges or validates A's feelings, expresses sympathy or understanding (I'm sorry, that must be hard, I understand)",
    "congratulate_share_joy": "congratulates, praises or shares A's happiness (congrats!, that's awesome!)",
    "advice": "suggests what A should do",
    "self_disclosure": "shares a similar experience or feeling of their own (me too, I had that happen)",
    "reassure_minimize": "reassures that it will be fine or downplays the problem (don't worry, it's not a big deal)",
    "opinion_judgment": "gives an evaluation or opinion about the situation or the people involved (that's so rude of him)",
    "humor": "jokes or lightens the mood with humor",
    "generic_reaction": "a brief reaction with little content (oh wow, cool, I see, nice)",
}
NOULS = {
    "n_question": "Does B ask A a question?",
    "n_validation": "Does B acknowledge or validate A's feelings or express sympathy?",
    "n_congrats": "Does B congratulate A, praise A or celebrate with A?",
    "n_advice": "Does B give advice or suggest what A should do?",
    "n_selfdisc": "Does B share a similar experience or feeling of their own?",
    "n_minimize": "Does B downplay A's problem or say not to worry?",
    "n_humor": "Does B joke or use humor?",
    "n_wish": "Does B express a wish or hope for A (I hope..., good luck)?",
}

def sample():
    convs = [json.loads(l) for l in open(os.path.join(SCR, "empathetic_convs.jsonl"))]
    by = defaultdict(list)
    for c in convs:
        if len(c["utts"]) >= 4:
            by[c["gold"]].append(c)
    rnd = random.Random(33)
    out = []
    for e in sorted(by):
        out += rnd.sample(by[e], 19)
    return out

def main():
    Q = {"main": choice("What is the main thing listener B does in their reply to A?", STRAT),
         "first": choice("What is the FIRST thing listener B does in their reply (the opening words)?", STRAT)}
    Q |= {k: noul(v) for k, v in NOULS.items()}
    S = sample()
    items = [({"A_tells": c["utts"][0]["text"], "B_replies": c["utts"][1]["text"]}, Q) for c in S]
    if len(sys.argv) > 1: items, S = items[:2], S[:2]
    res = ask_many(items, workers=4)
    with open(os.path.join(SCR, "listener_raw.jsonl"), "w") as fo:
        for c, r in zip(S, res):
            if r is not None:
                fo.write(json.dumps({"conv_id": c["conv_id"], "gold": c["gold"], "A1": c["utts"][0]["text"],
                                     "B1": c["utts"][1]["text"], "r": r}) + "\n")
    print(summary())

if __name__ == "__main__":
    main()
