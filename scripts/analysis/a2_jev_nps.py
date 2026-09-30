"""a2: NEW Jev experiment on NPS chatroom first posts (each user's first public post in a room).
State = 6 previous posts in the room + the post. Output: $A2_SCRATCH/jev_nps.pkl"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import a2_load as L
from jev import ask_many, choice, noul, score, summary

TYPES = {
    "greet_room": "a greeting to the whole room (hi all, hello everyone)",
    "greet_person": "a greeting to a specific person",
    "greet_bare": "a bare greeting not clearly addressed to anyone",
    "demographics_asl": "asks or states age/sex/location",
    "open_invitation": "looks for someone to chat with or asks people to PM (private message)",
    "compliment_or_flirt": "compliment, pet name or flirty remark",
    "sexual_proposition": "explicit sexual request or remark",
    "joins_ongoing": "reacts to or joins the ongoing conversation in the room",
    "question_to_room": "asks the room a question about something",
    "self_statement": "a statement about themselves or their mood",
    "other": "anything else (spam, commands, noise)",
}


def main():
    d = pd.read_pickle(os.path.join(L.SCR, "nps_posts.pkl"))
    f = d[d.first_post]
    items = []
    for _, r in f.iterrows():
        prev = d[(d.room == r.room) & (d.pos < r.pos) & (d.pos >= r.pos - 6)]
        state = {"context": "Public online chat room (2006), many strangers. Usernames look like 11-08-20sUser12.",
                 "recent_room_messages": [{"user": x.speaker, "text": x.text[:200]} for _, x in prev.iterrows()] or "(none)",
                 "new_post": {"user": r.speaker, "text": r.text[:300], "note": "first public message of this user"}}
        items.append((state, {
            "open_type": choice("What is `new_post` doing?", TYPES),
            "flirting": noul("Is `new_post` flirting, romantic or sexual?"),
            "addressed": noul("Is `new_post` addressed to one specific person?"),
            "easy_reply": noul("Does `new_post` give others an easy, natural way to reply?"),
            "pushy_creepy": noul("Would most people find `new_post` pushy, creepy or off-putting?"),
            "energy": score("How much energy/enthusiasm does `new_post` show?", ["flat", "mild", "lively", "very excited"]),
        }))
    print("calls", len(items), flush=True)
    res = ask_many(items, workers=4)
    out = []
    for (i, r), a in zip(f.iterrows(), res):
        if a is None:
            continue
        rec = {"idx": i}
        for k, v in a.items():
            rec["j_" + k] = v.get("noul", v.get("score", v.get("choice")))
        out.append(rec)
    pd.DataFrame(out).to_pickle(os.path.join(L.SCR, "jev_nps.pkl"))
    print(summary())


if __name__ == "__main__":
    main()
