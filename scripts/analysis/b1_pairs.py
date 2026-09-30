"""b1 step 4 (A7): pairwise comparison decomposed by trait, both orders (position-bias check).
Pairs = same context, human reply vs one LLM reply: a8 maichat x {base_luna, styled_luna} + a9 x {B}.
State: conversation + option_A + option_B. 9 trait Choices (A / B / same) + the round-1 holistic Choice (A / B).
Output: scratch b1/pairs.json {pair_id: {"order": "HL"|"LH", answers...}}"""
import json, os
import b1_common as B
from jev import choice

OPT3 = {"A": "option_A", "B": "option_B", "same": "neither, or about the same"}
PW = {
    "pw_enthusiastic": "Which option is more enthusiastic or excited?",
    "pw_generic": "Which option could more easily be sent in reply to many other messages?",
    "pw_polished": "Which option is more polished (full sentences, standard punctuation)?",
    "pw_more_moves": "Which option does more things at once (react, comment, explain, ask)?",
    "pw_validating": "Which option validates, sympathizes or praises more?",
    "pw_assistant": "Which option sounds more like a helpful assistant?",
    "pw_question": "Which option asks the other person more questions?",
    "pw_echo": "Which option repeats more of what the other person just said?",
    "pw_friend": "Which option sounds more like a quick text from a friend?",
}
QS = {k: choice(v, OPT3) for k, v in PW.items()}
QS["pw_real"] = choice("One of the two options is the message the real person actually typed next in this chat; the other was written by an AI chatbot. Which option was typed by the real person?",
                       {"A": "option_A was typed by the real person", "B": "option_B was typed by the real person"})
PAIR_CONDS = {"a8_maichat": ["base_luna", "styled_luna"], "a9": ["B"]}


def pairs(ctxs, units):
    by = {}
    for u in units:
        by.setdefault(u["ckey"], {})[u["cond"]] = u
    out = []
    for ck, d in by.items():
        src = ctxs[ck]["src"]
        if "human" not in d:
            continue
        for cn in PAIR_CONDS.get(src, []):
            if cn in d:
                out.append((ck, d["human"], d[cn]))
    return out


if __name__ == "__main__":
    ctxs, units = B.load_units()
    P = pairs(ctxs, units)
    outp = os.path.join(B.SCR, "pairs.json")
    done = json.load(open(outp)) if os.path.exists(outp) else {}
    items, keys = [], []
    for ck, h, l in P:
        c = ctxs[ck]
        for order in ("HL", "LH"):
            pid = f"{l['uid']}|{order}"
            if pid in done:
                continue
            a, b = (h, l) if order == "HL" else (l, h)
            st = {"setting": B.base_state(c, "")["setting"], "conversation_so_far": B.convo_lines(c),
                  "option_A": {"from": c["self"], "text": a["text"]}, "option_B": {"from": c["self"], "text": b["text"]}}
            items.append((st, QS)); keys.append((pid, order, h["uid"], l["uid"]))
    print("pairs", len(P), "calls todo", len(items))
    for i in range(0, len(items), 300):
        res = B.run_jev("A7_pairs", items[i:i + 300])
        for (pid, order, hu, lu), r in zip(keys[i:i + 300], res):
            if r:
                done[pid] = {"order": order, "human": hu, "llm": lu, **{q: a["probabilities"] for q, a in r.items()}}
        json.dump(done, open(outp, "w"))
    print("saved", len(done))
