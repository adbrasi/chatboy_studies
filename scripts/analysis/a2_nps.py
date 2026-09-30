"""a2: NPS chatroom openings between strangers: who answers which openings; flirting openers.
Response = within the next K non-system posts another user mentions the opener's username (NPS anonymizes names
as <room>UserN in text too). Addressed reply = the addressee of an addressed post answers mentioning the opener.
Output: $A2_SCRATCH/nps_posts.pkl and analysis/data/a2_nps_summary.json
"""
import json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import a2_load as L

K = 15
USER = re.compile(r"\d\d-\d\d-(?:teens|20s|30s|40s|adults)User\d+")
GROUP = re.compile(r"\b(?:all|every(?:one|body)|room|ppl|people|peeps|chat|guys|ladies|girls|y'?all|folks|gang|everybody)\b", re.I)
FLIRT = {
    "asl": re.compile(r"\ba\s*/?\s*s\s*/?\s*l\b|\basl\b|\bage\b|how old|where (?:r|are) (?:u|you) from|\b(?:m|f)\s*or\s*(?:m|f)\b|\bm/f\b", re.I),
    "open_call": re.compile(r"\bany\b.{0,25}\b(?:ladies|girls|women|females|chicks|guys|men|males|single|hot|cute)\b|\b(?:wanna|want to|care to|like to) (?:chat|talk|pm|private)|\bpm me\b|\bprivate\b|\bchat with me\b", re.I),
    "compliment_petname": re.compile(r"\b(?:sexy|hot|cute|beautiful|gorgeous|pretty|handsome|babe|baby|hun|honey|sweetie|sweety|darlin|darling|sweetheart|hottie|sugar)\b", re.I),
    "sexual": re.compile(r"\b(?:fuck|sex|cyber|horny|naked|nude|kiss|lick|dick|cock|pussy|boobs?|tits?|bed|webcam|cam)\b", re.I),
}


def main():
    m = L.messages()
    n = m[m.corpus == "nps_chatroom"].sort_values(["conv_id", "idx"]).copy()
    rows = []
    for room, g in n.groupby("conv_id"):
        g = g.reset_index(drop=True)
        posts = g[g.dialogue_act != "System"].reset_index()  # 'index' = position in g
        seen_first = set()
        joined_at = {}
        greeted = {}
        for i, r in g.iterrows():
            if r.dialogue_act == "System" and r.text.startswith("JOIN"):
                joined_at[r.speaker] = i
        P = posts.to_dict("records")
        for j, p in enumerate(P):
            u = p["speaker"]
            ment = set(USER.findall(p["text"])) - {u}
            nxt = P[j + 1:j + 1 + K]
            mentioned_back = [q for q in nxt if q["speaker"] != u and u in q["text"]]
            addr_reply = None
            if ment:
                addr_reply = any(q["speaker"] in ment and u in q["text"] for q in nxt)
                addr_posts_any = any(q["speaker"] in ment for q in nxt)
            else:
                addr_posts_any = None
            own_next = sum(1 for q in nxt if q["speaker"] == u)
            first = u not in seen_first
            seen_first.add(u)
            txt = p["text"]
            rec = {"room": room, "pos": j, "speaker": u, "text": txt, "act": p["dialogue_act"], "first_post": first,
                   "addressed": bool(ment), "to_group": bool(GROUP.search(txt)) and not ment,
                   "n_words": len(txt.split()), "has_q": "?" in txt,
                   "resp_named": bool(mentioned_back),
                   "resp_lag_posts": (P.index(mentioned_back[0]) - j) if mentioned_back else None,
                   "addr_reply": addr_reply, "addr_posts_any": addr_posts_any, "own_next": own_next,
                   "later_posts_total": sum(1 for q in P[j + 1:] if q["speaker"] == u),
                   "resp_first_text": mentioned_back[0]["text"] if mentioned_back else None,
                   "resp_first_act": mentioned_back[0]["dialogue_act"] if mentioned_back else None,
                   "age_room": room.split("-")[-1]}
            for k, rx in FLIRT.items():
                rec["fl_" + k] = bool(rx.search(txt))
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_pickle(os.path.join(L.SCR, "nps_posts.pkl"))
    print(len(df), df.first_post.sum())


if __name__ == "__main__":
    main()
