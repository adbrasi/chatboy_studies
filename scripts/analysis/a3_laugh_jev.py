"""a3: classifica a FUNÇÃO de cada risada (turnos do jev_base com riso) com o Jev.
Riso = flag `laugh` das features OU emojis SoftBank antigos do WhatsApp ( 😂,  😁,  😄).
State: até 4 turnos anteriores da mesma sessão + o turno atual (mensagens juntadas).
Controle: 150 turnos SEM riso recebem só a pergunta de 'face threat' (taxa-base).
Saída: SCR/laugh_raw.jsonl"""
import json, os, random, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from a3_common import *
from jev import ask_many, choice, noul, summary

P = os.path.join(ROOT, "data", "processed")
SB = ""
FUNC = {
    "reaction_to_humor": "laughing at something funny the other person said or sent",
    "own_joke_marker": "laughing along with their own joke or tease, to signal it is meant as a joke",
    "amused_own_story": "laughing about a funny or absurd situation they themselves are telling",
    "softener": "softening something that is not really funny (a complaint, request, criticism, refusal, correction, bad news or awkward admission) to keep it light",
    "nervous_awkward": "nervous or embarrassed laughter",
    "friendly_backchannel": "friendly acknowledgment or warmth; nothing particularly funny",
}
FT = "Does `current_turn` contain a complaint, criticism, request, refusal, correction or an awkward/embarrassing admission?"

def is_laugh(t):
    return t["laugh"] or any(ch in " ".join(t["texts"]) for ch in SB)

def fmt(t):
    x = " / ".join(t["texts"])
    return {"speaker": t["speaker"], "text": x[:500]}

def main():
    base = [json.loads(l) for l in open(f"{P}/jev_base.jsonl")]
    want = {(b["corpus"], b["conv_id"]) for b in base}
    turns = defaultdict(dict)
    for l in open(f"{P}/turns.jsonl"):
        if '"maichat"' not in l[:40] and '"whatsapp_nl"' not in l[:40]:
            continue
        t = json.loads(l)
        if (t["corpus"], t["conv_id"]) in want:
            turns[(t["corpus"], t["conv_id"])][t["turn_idx"]] = t
    Qfull = {"function": choice("Why does the speaker of `current_turn` laugh (haha, lol, laughing emoji)?", FUNC),
             "prev_funny": noul("Did the other person's most recent turn contain something funny (a joke, tease, funny story or funny picture)?"),
             "cur_joke": noul("Is the speaker of `current_turn` making a joke or teasing in this turn (apart from the laughter itself)?"),
             "softening": noul("Is the laughter in `current_turn` softening something potentially uncomfortable (a complaint, criticism, request, refusal, correction or embarrassing admission)?"),
             "face_threat": noul(FT)}
    Qctl = {"face_threat": noul(FT)}
    rnd = random.Random(5)
    lau = [b for b in base if is_laugh(b)]
    non = rnd.sample([b for b in base if not is_laugh(b)], 150)
    items, meta = [], []
    for grp, lst, Q in (("laugh", lau, Qfull), ("control", non, Qctl)):
        for b in lst:
            conv = turns[(b["corpus"], b["conv_id"])]
            prev = [conv[i] for i in range(b["turn_idx"] - 4, b["turn_idx"]) if i in conv and conv[i]["session"] == b["session"]]
            st = {"previous_turns": [fmt(t) for t in prev] or "(start of conversation)", "current_turn": fmt(b)}
            items.append((st, Q)); meta.append((grp, b["corpus"], b["conv_id"], b["turn_idx"]))
    if len(sys.argv) > 1: items, meta = items[:2], meta[:2]
    print("calls", len(items), flush=True)
    res = ask_many(items, workers=4)
    with open(os.path.join(SCR, "laugh_raw.jsonl"), "w") as fo:
        for (g, c, cid, ti), r in zip(meta, res):
            if r is not None:
                fo.write(json.dumps({"grp": g, "corpus": c, "conv_id": cid, "turn_idx": ti, "r": r}, ensure_ascii=False) + "\n")
    print(summary())

if __name__ == "__main__":
    main()
