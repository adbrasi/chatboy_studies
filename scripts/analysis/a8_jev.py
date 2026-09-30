"""a8 step 6: Jev naturalness detectors on human vs LLM replies (same contexts).
  single: 11 isolated questions about ONE candidate reply (human or LLM)
  pair:   Choice "which one was typed by a real person?" (human vs LLM, random order)
Usage: python3 a8_jev.py single <cond...> | pair <cond...> | test
Output: scratch a8/jev_single_<cond>.json, a8/jev_pair_<cond>.json
"""
import json, os, random, sys
import a8_common as C
from jev import ask_many, noul, choice, score, stats, summary

BOT, USER = "Maya", "Jordan"

SINGLE_Q = {
    "human": noul("Was the CANDIDATE MESSAGE typed by a real person in this chat, rather than written by an AI chatbot?",
                  "Typed by a real person", "Written by an AI chatbot"),
    "formal": noul("The CANDIDATE MESSAGE uses vocabulary, phrasing or structure that is too formal, polished, elaborate or assistant-like for this casual chat."),
    "forced": noul("The CANDIDATE MESSAGE uses slang, emojis, internet expressions or enthusiasm in a forced, exaggerated or caricatured way."),
    "generic": noul("The CANDIDATE MESSAGE is generic: it could be sent in reply to many different messages and does not depend on the specifics of what was just said."),
    "direct": noul("The CANDIDATE MESSAGE responds directly to the last thing the other person said."),
    "register": noul("The CANDIDATE MESSAGE matches the other person's register: similar length, formality, punctuation and energy."),
    "overvalidation": noul("The CANDIDATE MESSAGE praises, validates, reassures or empathizes more than the situation calls for."),
    "paraphrase": noul("The CANDIDATE MESSAGE restates or paraphrases what the other person just said."),
    "too_much": noul("The CANDIDATE MESSAGE tries to do too much at once (for example react, comment and ask a follow-up question) where a real person would just reply to one thing."),
    "invented": noul("The CANDIDATE MESSAGE mentions facts, events or shared memories that are not supported by the conversation."),
    "length": score("Compared with what a real person would text at this point in this chat, the CANDIDATE MESSAGE's length is:",
                    ["much too short", "a bit short", "about right", "a bit too long", "much too long"]),
}
PAIR_Q = {"which_human": choice("One of the two options is the message the real person actually typed next in this chat; the other was written by an AI chatbot. Which option was typed by the real person?",
                                {"A": "Option A was typed by the real person", "B": "Option B was typed by the real person"})}


def convo(c, max_turns=8):
    lines = []
    for h in c["history"][-max_turns:]:
        who = BOT if h["who"] == "bot" else USER
        for t in h["texts"]:
            if t.strip():
                lines.append(f"{who}: {t.strip()}")
    return "\n".join(lines)


def single_state(c, text):
    return (f"Casual one-to-one text chat on a messaging app between {USER} and {BOT}.\n\nCONVERSATION SO FAR:\n{convo(c)}\n\n"
            f"CANDIDATE MESSAGE (next message from {BOT}):\n{text.strip()}")


def pair_state(c, a, b):
    return (f"Casual one-to-one text chat on a messaging app between {USER} and {BOT}.\n\nCONVERSATION SO FAR:\n{convo(c)}\n\n"
            f"Two options for {BOT}'s next message:\n\nOPTION A:\n{a.strip()}\n\nOPTION B:\n{b.strip()}")


def texts_for(cond, X):
    if cond == "human":
        return {c["id"]: "\n".join(t.strip() for t in c["human"] if t.strip()) for c in X}
    return json.load(open(f"{C.SCR}/gen_{cond}.json"))


def run_single(conds, X):
    for cond in conds:
        T = texts_for(cond, X)
        ids = [c["id"] for c in X if T.get(c["id"])]
        res = ask_many([(single_state(c, T[c["id"]]), SINGLE_Q) for c in X if c["id"] in ids], workers=4)
        json.dump(dict(zip(ids, res)), open(f"{C.SCR}/jev_single_{cond}.json", "w"))
        print(cond, "done", sum(r is None for r in res), "errors", summary())


def run_pair(conds, X):
    H = texts_for("human", X)
    for cond in conds:
        T = texts_for(cond, X)
        rnd = random.Random(hash(cond) % 1000 + 8)
        items, meta = [], []
        for c in X:
            if not T.get(c["id"]):
                continue
            human_first = rnd.random() < 0.5
            a, b = (H[c["id"]], T[c["id"]]) if human_first else (T[c["id"]], H[c["id"]])
            items.append((pair_state(c, a, b), PAIR_Q)); meta.append((c["id"], "A" if human_first else "B"))
        res = ask_many(items, workers=4)
        json.dump({i: {"human_is": h, "ans": r} for (i, h), r in zip(meta, res)}, open(f"{C.SCR}/jev_pair_{cond}.json", "w"))
        print(cond, "pair done", summary())


if __name__ == "__main__":
    X = json.load(open(f"{C.SCR}/contexts.json"))
    mode = sys.argv[1]
    if mode == "test":
        T = texts_for("base_gemini", X)
        r = ask_many([(single_state(X[1], "\n".join(X[1]["human"])), SINGLE_Q), (single_state(X[1], T[X[1]["id"]]), SINGLE_Q)], workers=2)
        print(json.dumps(r, indent=0)[:3000]); print(summary())
    elif mode == "single":
        run_single(sys.argv[2:], X)
    elif mode == "pair":
        run_pair(sys.argv[2:], X)
