"""Base Jev annotation layer over conversational turns.

Two passes per sampled turn T (speaker S):

  D (descriptive)  state = previous context turns + T.  "What is S feeling/doing in T?"
                   T's messages are JOINED into one text so Jev cannot see how S fragmented
                   the turn -> emotion labels stay independent of the burst shape we later correlate.

  P (predictive)   state = context up to the partner's last turn (T is NOT shown).
                   "What will S's next turn look like?"  -> compared with what S actually did.
                   This is exactly the position the chatbot is in when it has to reply.

Output: data/processed/jev_base.jsonl  (one line per turn, D and P answers flattened)
"""
import json, os, random, sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from jev import ask_many, choice, noul, score, summary

P = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CTX_TURNS = 8
MAX_CHARS = 500


def gap_bucket(sec):
    if sec is None:
        return None
    for lim, name in ((20, "immediately"), (120, "within 2 minutes"), (900, "a few minutes later"),
                      (3600, "within the hour"), (6 * 3600, "hours later")):
        if sec < lim:
            return name
    return "much later (next day or more)"


def fmt_turn(t, joined=True, label=None):
    txt = " / ".join(t["texts"]) if joined else t["texts"]
    if joined and len(txt) > MAX_CHARS:
        txt = txt[:MAX_CHARS] + "…"
    d = {"speaker": label or t["speaker"], "text": txt}
    g = gap_bucket(t.get("response_latency_s"))
    if g and t["corpus"] != "maichat":
        d["replied"] = g
    return d


EMOTIONS = {
    "joy_excitement": "happy, excited, enthusiastic",
    "affection": "loving, tender, missing the other, caring",
    "playful_teasing": "joking, teasing, banter, being silly",
    "amusement": "laughing at / enjoying something funny the other said",
    "curiosity_interest": "interested, wants to know more",
    "anxiety_insecurity": "worried, nervous, insecure, needing reassurance",
    "sadness": "sad, down, disappointed, lonely",
    "frustration_anger": "annoyed, irritated, angry, hurt",
    "boredom_disengaged": "bored, low effort, wants to leave",
    "gratitude": "thankful, appreciative",
    "surprise": "surprised, shocked",
    "neutral_informational": "matter-of-fact, logistics, no particular emotion",
}
INTENTS = {
    "greet": "opening / hello",
    "ask_question": "asks the other something",
    "answer": "answers a question the other asked",
    "share_story_or_info": "tells something about themselves or events",
    "share_feeling": "expresses their own feelings",
    "react_acknowledge": "short reaction/backchannel (haha, ok, wow, nice)",
    "joke_tease": "makes a joke or teases",
    "compliment_affection": "compliments or expresses affection to the other",
    "make_plans_logistics": "arranging a meeting, time, practical matters",
    "comfort_support": "consoles, reassures, supports the other",
    "disagree_complain": "disagrees, complains, criticizes",
    "change_topic": "starts a new, unrelated topic",
    "closing": "wrapping up / goodbye",
}
PHASES = {
    "opening": "the first exchanges, greetings",
    "small_talk": "light, getting going, how-are-you",
    "deep_personal": "personal, meaningful or emotional conversation",
    "playful_banter": "joking back and forth",
    "logistics": "coordinating practical matters",
    "conflict_or_repair": "tension, disagreement, apologizing",
    "winding_down": "energy dropping, heading to the end",
}
RELATION = {
    "romantic_partners": None, "flirting_or_crush": None, "close_friends": None,
    "family": None, "acquaintances_or_new": None, "colleagues_or_classmates": None,
}


def q_descriptive():
    who = "the speaker of `current_turn`"
    return {
        "emotion": choice(f"Which emotion best describes {who}?", EMOTIONS),
        "valence": score(f"How positive or negative is {who} feeling?", ["very negative", "negative", "neutral", "positive", "very positive"]),
        "arousal": score(f"How emotionally activated/energetic is {who}?", ["very calm, low energy", "calm", "moderate", "energetic", "highly agitated or excited"]),
        "anxious": noul(f"Is {who} anxious, worried or insecure?"),
        "playful": noul(f"Is {who} being playful, joking or teasing?"),
        "flirting": noul(f"Is {who} flirting or being romantic?"),
        "vulnerable": noul(f"Is {who} disclosing something personal or vulnerable?"),
        "seeks_support": noul(f"Is {who} seeking comfort, reassurance or support?"),
        "tension": noul("Is there tension, disagreement or conflict between the two people right now?"),
        "seriousness": score("How serious is this moment of the conversation?", ["playful banter", "casual", "somewhat serious", "very serious / emotional"]),
        "intent": choice(f"What is the main thing {who} is doing?", INTENTS),
        "phase": choice("Which phase is the conversation in at `current_turn`?", PHASES),
        "relationship": choice("What is the most likely relationship between the two people?", RELATION),
        "engagement": score(f"How engaged/invested in the conversation is {who}?", ["disengaged, minimal effort", "low", "moderate", "high", "very high, eager"]),
        "hook": noul(f"Does `current_turn` give the other person something easy to reply to (a question, a hook, an invitation)?"),
        "topic_shift": noul("Does `current_turn` start a new topic compared to the previous turns?"),
        "mirrors": noul(f"Does {who} mirror the style or energy of the previous turn by the other person?"),
    }


def q_predictive():
    who = "`next_speaker`"
    return {
        "p_n_msgs": choice(f"In how many separate chat messages (bubbles) will {who} send their next turn?", {
            "1": "one single message", "2": "two messages in a row", "3": "three messages in a row",
            "4+": "four or more quick messages in a row"}),
        "p_length": score(f"How long will {who}'s next turn be in total?", [
            "1-3 words", "one short sentence", "one or two full sentences", "several sentences", "a long paragraph"]),
        "p_laugh": noul(f"Will {who} laugh (haha, lol, 😂) in their next turn?"),
        "p_question": noul(f"Will {who} ask a question in their next turn?"),
        "p_emoji": noul(f"Will {who} use an emoji or emoticon in their next turn?"),
        "p_topic_shift": noul(f"Will {who} change the topic in their next turn?"),
        "p_end": noul(f"Will {who} start ending the conversation (goodbye) in their next turn?"),
        "p_tone": choice(f"What tone will {who}'s next turn most likely have?", {
            "playful": "joking, teasing, light", "warm_affectionate": "sweet, caring, loving",
            "supportive_serious": "serious, empathetic, careful", "enthusiastic": "excited, energetic",
            "neutral_informative": "matter-of-fact", "annoyed_cold": "irritated, curt, distant"}),
        "p_emotion": choice(f"Which emotion will {who} most likely express in their next turn?", EMOTIONS),
        "p_joke_welcome": noul(f"Would a joke be a good idea for {who}'s next turn?"),
    }


def sample_turns(turns, corpus, n_per_conv=None, total=None, seed=7):
    by = defaultdict(list)
    for t in turns:
        if t["corpus"] == corpus:
            by[t["conv_id"]].append(t)
    rnd = random.Random(seed)
    out = []
    for cid, ts in by.items():
        ts.sort(key=lambda t: t["turn_idx"])
        if n_per_conv and len(ts) > n_per_conv:
            # contiguous windows keep sequence analyses possible
            start = rnd.randrange(0, len(ts) - n_per_conv)
            ts_sel = ts[start:start + n_per_conv]
        else:
            ts_sel = ts
        out += [(t, ts) for t in ts_sel]
    if total and len(out) > total:
        out = rnd.sample(out, total)
    return out


def build(t, conv_turns):
    i = conv_turns.index(t)
    ctx = [x for x in conv_turns[max(0, i - CTX_TURNS):i] if x["session"] == t["session"]]
    context = [fmt_turn(x) for x in ctx]
    d_state = {"previous_turns": context or "(this is the first turn of the conversation)",
               "current_turn": fmt_turn(t)}
    items = [(d_state, q_descriptive())]
    if ctx:  # predictive only makes sense when there is something to react to
        p_state = {"conversation_so_far": context, "next_speaker": t["speaker"]}
        items.append((p_state, q_predictive()))
    return items


def main():
    turns = [json.loads(l) for l in open(f"{P}/turns.jsonl", encoding="utf-8")]
    plan = []
    plan += sample_turns(turns, "maichat")                              # all ~2.9k turns
    plan += sample_turns(turns, "whatsapp_nl", n_per_conv=60)           # contiguous windows, 57 chats
    items, index = [], []
    for t, conv in plan:
        its = build(t, conv)
        index.append((t, len(its)))
        items += its
    print("turns", len(plan), "calls", len(items), flush=True)
    res = ask_many(items, workers=int(os.environ.get("JEV_WORKERS", 16)))
    k = 0
    with open(f"{P}/jev_base.jsonl", "w", encoding="utf-8") as fo:
        for t, n in index:
            rs = res[k:k + n]
            k += n
            if any(r is None for r in rs):
                continue
            rec = {x: t[x] for x in t if x not in ("texts",)} | {"texts": t["texts"]}
            rec["D"] = rs[0]
            rec["P"] = rs[1] if n > 1 else None
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(summary())


if __name__ == "__main__":
    main()
