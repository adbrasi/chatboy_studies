"""a3: valida o detector de emoção do Jev contra o ouro do EmpatheticDialogues (32 classes).
Condições de contexto: A1 (1ª fala de A), A1B1A2 (duas primeiras falas de A com a resposta de B no meio),
FULL (conversa inteira, subamostra). Em cada chamada: Choice plano (32), Choice de polaridade, Choice de família (7)
e 7 Choices finos (um por família; a cascata usa o da família escolhida — perguntas são isoladas, então é
equivalente a rodar em sequência, só que numa única chamada).
Saída: SCR/emo_raw.jsonl (respostas) — análise em a3_emotion_eval.py"""
import json, os, random, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from a3_common import *
from jev import ask_many, choice, summary

N_PER, N_FULL = 19, 8
WHO = "speaker A"

def questions():
    q = {
        "flat": choice(f"Which emotion is {WHO} feeling about the situation they are telling?", EMO32),
        "polarity": choice(f"Is {WHO}'s feeling about the situation positive or negative?", {
            "positive": "pleasant feeling", "negative": "unpleasant feeling", "neutral_or_mixed": "neither, or both"}),
        "family": choice(f"Which family of emotions best describes what {WHO} is feeling about the situation?", FAMILY_DESC),
    }
    for f, es in FAMILIES.items():
        q["fine_" + f] = choice(f"Which of these emotions is {WHO} feeling about the situation?", {e: EMO32[e] for e in es})
    return q

def state(c, cond):
    u = c["utts"]
    if cond == "A1":
        return {"speaker_A_message": u[0]["text"]}
    sel = u[:3] if cond == "A1B1A2" else u
    return {"conversation": [{"speaker": x["speaker"], "text": x["text"]} for x in sel]}

def main():
    convs = [json.loads(l) for l in open(os.path.join(SCR, "empathetic_convs.jsonl"))]
    by = defaultdict(list)
    for c in convs:
        if len(c["utts"]) >= 4:
            by[c["gold"]].append(c)
    rnd = random.Random(33)
    sample = []
    for e in sorted(by):
        s = rnd.sample(by[e], N_PER)
        for i, c in enumerate(s):
            sample.append((c, i < N_FULL))
    items, meta = [], []
    Q = questions()
    for c, full in sample:
        for cond in (["A1", "A1B1A2"] + (["FULL"] if full else [])):
            items.append((state(c, cond), Q)); meta.append((c["conv_id"], c["gold"], cond))
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        items, meta = items[:2], meta[:2]
    print("calls", len(items), flush=True)
    res = ask_many(items, workers=4)
    with open(os.path.join(SCR, "emo_raw.jsonl"), "w") as fo:
        for (cid, g, cond), r in zip(meta, res):
            if r is not None:
                fo.write(json.dumps({"conv_id": cid, "gold": g, "cond": cond, "r": r}) + "\n")
    print(summary())

if __name__ == "__main__":
    main()
