"""b1 step 5 (A11 stage 3 + A12): sentence-level questions. The candidate is split (in code) into numbered sentences.
Per sentence: 'would a real friend write this sentence here?' and 'does it add something needed?' (Nouls).
Per reply with >=2 sentences: Choices over the sentence ids (+ 'none'): which is the filler question, which carries the stock
validation, which one a friend would leave out, which makes it sound most like an AI.
Output: scratch b1/sent.json {uid: {"sents": [...], answers}}"""
import json, os
import b1_common as B
import a8_common as C8
from jev import noul, choice

MAXS = 6


def split_sents(text):
    s = C8.sentences(text)
    if len(s) > MAXS:
        s = s[:MAXS - 1] + [" ".join(s[MAXS - 1:])]
    return s


def questions(n):
    ids = [f"S{i + 1}" for i in range(n)]
    qs = {}
    for sid in ids:
        qs[f"friend_{sid}"] = noul(f"A real friend casually texting in this chat would write sentence {sid} of candidate_sentences as part of this reply.")
        qs[f"needed_{sid}"] = noul(f"Sentence {sid} of candidate_sentences adds something the reply would miss without it.")
    if n >= 2:
        opts = {sid: f"sentence {sid}" for sid in ids}
        qs["cs_filler_q"] = choice("Which sentence of candidate_sentences is a follow-up question that is only there to keep the conversation going?", {**opts, "none": "no such sentence"})
        qs["cs_validation"] = choice("Which sentence of candidate_sentences contains stock validation, sympathy or praise (like 'that's amazing', 'sorry to hear that', 'so sweet')?", {**opts, "none": "no such sentence"})
        qs["cs_cut"] = choice("A friend texting quickly would send only part of this reply. Which one sentence of candidate_sentences would they most likely leave out?", opts)
        qs["cs_ai"] = choice("Which sentence of candidate_sentences makes the reply sound most like an AI chatbot?", {**opts, "none": "none, it all reads like a friend's text"})
    return qs


if __name__ == "__main__":
    ctxs, units = B.load_units()
    core = B.core_units(ctxs, units)
    outp = os.path.join(B.SCR, "sent.json")
    done = json.load(open(outp)) if os.path.exists(outp) else {}
    items, keys = [], []
    for u in core:
        if u["uid"] in done:
            continue
        s = split_sents(u["text"])
        if not s:
            continue
        c = ctxs[u["ckey"]]
        st = {"setting": B.base_state(c, "")["setting"], "conversation_so_far": B.convo_lines(c),
              "candidate_message_from": c["self"], "candidate_sentences": {f"S{i + 1}": x for i, x in enumerate(s)}}
        items.append((st, questions(len(s)))); keys.append((u["uid"], s))
    print("todo", len(items))
    for i in range(0, len(items), 300):
        res = B.run_jev("A12_sentences", items[i:i + 300])
        for (uid, s), r in zip(keys[i:i + 300], res):
            if r:
                d = {"sents": s}
                for q, a in r.items():
                    d[q] = B.val(a)
                    if a["type"] == "choice":
                        d[q + "_p"] = a["probabilities"]
                done[uid] = d
        json.dump(done, open(outp, "w"))
    print("saved", len(done))
