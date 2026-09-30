"""a2: NEW Jev experiments on WhatsApp session openings (typology, gap acknowledgement, refers back) + chat-level relationship.
Sample: up to 30 sessions per dyadic chat (random, seed 11). One call per opening, one call per chat for relationship.
Output: $A2_SCRATCH/jev_open.pkl, $A2_SCRATCH/jev_rel.pkl
"""
import os, sys, random, json
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import a2_load as L
from jev import ask_many, choice, noul, score, summary

N = int(os.environ.get("A2_N", 10**9))
CAP = 30
OPEN_TYPES = {
    "greeting_only": "only a greeting (hi, hey, good morning), possibly with a name or emoji, nothing else",
    "greeting_how_are_you": "a greeting and/or asking how the other is doing, with nothing else substantial",
    "greeting_plus_topic": "a greeting followed by a concrete topic, question, request or news",
    "availability_check": "checks whether the other is there/awake/free/able to talk or call",
    "late_reply": "answers or reacts to the last thing the other said before the silence (a delayed reply)",
    "follow_up_earlier": "comes back to an earlier topic or event (e.g. asks how something went), not a direct reply",
    "new_news_or_story": "starts directly (no greeting) with new news, a story, a feeling or a comment",
    "new_question_or_request": "starts directly (no greeting) with a new question or request",
    "logistics_update": "practical coordination: times, places, plans, arriving/leaving, errands",
    "media_or_link": "shares a photo, video or link (possibly with a caption)",
    "wishes": "congratulations, birthday wishes, good luck, get well, good night wishes",
}
REL = {
    "romantic_partners": "dating / couple",
    "parent_child": "parent and child",
    "siblings_or_other_family": "siblings, cousins, other relatives",
    "close_friends": "close friends",
    "casual_friends_or_acquaintances": "friends who are not very close, acquaintances",
    "classmates_or_colleagues": "fellow students, teammates, colleagues (mostly practical contact)",
}


def gap_txt(h):
    if h is None or h != h:
        return None
    for lim, s in ((6, "3 to 6 hours"), (12, "6 to 12 hours"), (24, "12 to 24 hours"), (72, "1 to 3 days"),
                   (168, "3 to 7 days")):
        if h < lim:
            return s
    return "more than a week"


def questions():
    return {
        "open_type": choice("What best describes the `new_message` (the first message after the silence)?", OPEN_TYPES),
        "acknowledges_gap": noul("Does `new_message` explicitly acknowledge the silence or delay (e.g. apologizes for replying late, 'long time no talk', 'sorry I fell asleep', 'are you still awake')?"),
        "refers_back": noul("Does `new_message` refer back to something mentioned in `earlier_messages`?"),
        "has_greeting": noul("Does `new_message` contain a greeting (hi, hey, hello, good morning, or similar)?"),
        "invites_reply": noul("Does `new_message` give the other person a clear reason to reply (a question, request, or invitation)?"),
        "warmth": score("How warm/affectionate is `new_message`?", ["cold or curt", "neutral", "friendly", "very warm / affectionate"]),
    }


def main():
    turns = L.turns()
    turns = turns[turns.corpus == "whatsapp_nl"]
    sess = pd.read_pickle(os.path.join(L.SCR, "sessions.pkl"))
    w = sess[sess.corpus == "whatsapp_nl"].copy()
    rnd = random.Random(11)
    samp = []
    for cid, g in w.groupby("conv_id"):
        idx = list(g.index)
        rnd.shuffle(idx)
        samp += idx[:CAP]
    samp = sorted(samp)[:N] if N < 10**9 else sorted(samp)
    tg = {k: g.sort_values("turn_idx") for k, g in turns.groupby("conv_id")}
    items, keys = [], []
    for i in samp:
        r = w.loc[i]
        ct = tg[r.conv_id]
        prev = ct[ct.session == r.session - 1].tail(4)
        earlier = [{"speaker": x.speaker, "text": " / ".join(x.texts)[:300]} for _, x in prev.iterrows()]
        ts = r.ts0
        state = {
            "context": "Dutch WhatsApp chat between two people (A and B). A silence of at least 3 hours preceded `new_message`.",
            "earlier_messages": earlier or "(none: this is the start of the chat log)",
            "silence_before_new_message": gap_txt(r.gap_h) or "unknown",
            "new_message_time": ts.strftime("%A %H:%M") if pd.notna(ts) else "unknown",
            "new_message": {"speaker": r.opener, "text": r.open_text[:500]},
        }
        items.append((state, questions()))
        keys.append(i)
    # chat-level relationship
    rel_items, rel_keys = [], []
    for cid, ct in tg.items():
        if cid not in set(w.conv_id):
            continue
        rs = random.Random(5)
        sids = sorted(ct.session.unique())
        pick = sorted(rs.sample(sids, min(8, len(sids))))
        ex = []
        for s in pick:
            for _, x in ct[ct.session == s].head(5).iterrows():
                ex.append({"speaker": x.speaker, "text": " / ".join(x.texts)[:200]})
        rel_items.append(({"context": "Excerpts (several separate days) from a Dutch WhatsApp chat between A and B.",
                           "excerpts": ex[:40]},
                          {"relationship": choice("What is the relationship between A and B?", REL)}))
        rel_keys.append(cid)
    print("opening calls", len(items), "relationship calls", len(rel_items), flush=True)
    res = ask_many(items + rel_items, workers=4)
    out = []
    for i, a in zip(keys, res[:len(items)]):
        if a is None:
            continue
        rec = {"sidx": i}
        for k, v in a.items():
            rec[k] = v.get("noul", v.get("score", v.get("choice")))
            if v.get("type") == "choice":
                rec[k + "_conf"] = v.get("confidence")
        out.append(rec)
    pd.DataFrame(out).to_pickle(os.path.join(L.SCR, "jev_open.pkl"))
    rel = [{"conv_id": c, "rel": a["relationship"]["choice"], "rel_conf": a["relationship"]["confidence"]}
           for c, a in zip(rel_keys, res[len(items):]) if a]
    pd.DataFrame(rel).to_pickle(os.path.join(L.SCR, "jev_rel.pkl"))
    print(summary())


if __name__ == "__main__":
    main()
