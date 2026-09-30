"""a6: a realistic pool of ~150 'briefing' questions about `current_turn` (the user's latest message)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from a6_common import AB, noul, choice, score

W = "the speaker of `current_turn`"


def pool():
    P = {}
    QD = AB.q_descriptive()
    core = ["anxious", "emotion", "seriousness", "playful", "valence", "intent", "flirting", "tension", "hook", "relationship"]
    for k in core:
        P[k] = QD[k]
    for k in QD:
        if k not in P:
            P[k] = QD[k]
    for e, desc in AB.EMOTIONS.items():
        P[f"emo_{e}"] = noul(f"Is {W} feeling {e.replace('_', ' ')} ({desc})?")
    for i, desc in AB.INTENTS.items():
        P[f"int_{i}"] = noul(f"In `current_turn`, does the speaker {desc.split(' / ')[0]}?")
    for p, desc in AB.PHASES.items():
        P[f"ph_{p}"] = noul(f"Is the conversation at `current_turn` in this phase: {desc}?")
    extra = {
        "d_split": noul("Would a natural reply to `current_turn` be sent as several short messages rather than one?"),
        "d_long": noul("Does `current_turn` call for a long, thoughtful reply?"),
        "d_fast": noul("Would a quick, immediate reply to `current_turn` feel natural?"),
        "d_laugh": noul("Would it be natural to laugh (haha, lol) in the reply to `current_turn`?"),
        "d_emoji": noul("Would an emoji fit in the reply to `current_turn`?"),
        "d_sticker": noul("Would a sticker or GIF be an appropriate reply to `current_turn`?"),
        "d_question": noul("Should the reply to `current_turn` include a question back?"),
        "d_joke": noul("Would a joke be welcome as a reply to `current_turn`?"),
        "d_comfort": noul("Does `current_turn` call for comfort or reassurance?"),
        "d_apology": noul("Does `current_turn` call for an apology?"),
        "d_topic": noul("Would changing the topic after `current_turn` feel rude?"),
        "d_callback": noul("Would it be natural to refer back to something said earlier in `previous_turns`?"),
        "d_tease_back": noul("Is `current_turn` teasing the other person in a way that invites teasing back?"),
        "d_ask_more": noul("Would asking a follow-up question about what `current_turn` says show interest?"),
        "d_short_ok": noul("Would a very short reply (one or two words) be enough for `current_turn`?"),
        "d_end": noul("Is the speaker of `current_turn` trying to end the conversation?"),
        "d_sarcasm": noul("Is `current_turn` sarcastic?"),
        "d_compliment": noul("Does `current_turn` contain a compliment to the other person?"),
        "d_complaint": noul("Does `current_turn` contain a complaint about the other person?"),
        "d_plan": noul("Does `current_turn` propose meeting or doing something together?"),
        "d_secret": noul("Does `current_turn` share a secret or something private?"),
        "d_bad_news": noul("Does `current_turn` report bad news?"),
        "d_good_news": noul("Does `current_turn` report good news?"),
        "d_ignored": noul("Does the speaker of `current_turn` seem to feel ignored?"),
        "d_jealous": noul("Does the speaker of `current_turn` seem jealous?"),
        "d_miss": noul("Does the speaker of `current_turn` say they miss the other person?"),
        "d_tired": noul("Does the speaker of `current_turn` seem tired or sleepy?"),
        "d_busy": noul("Does the speaker of `current_turn` seem busy or in a hurry?"),
        "d_bored": noul("Does the speaker of `current_turn` seem bored?"),
        "d_excited": noul("Does the speaker of `current_turn` seem excited about something?"),
        "d_opinion": noul("Does `current_turn` ask for the other person's opinion?"),
        "d_advice": noul("Does `current_turn` ask for advice?"),
        "d_food": noul("Is `current_turn` about food or drinks?"),
        "d_work": noul("Is `current_turn` about work or study?"),
        "d_family": noul("Is `current_turn` about family?"),
        "d_health": noul("Is `current_turn` about health or the body?"),
        "d_media": noul("Is `current_turn` about a movie, series, music or a game?"),
        "d_travel": noul("Is `current_turn` about travel or going somewhere?"),
        "d_romance": noul("Is `current_turn` about romance or dating?"),
        "d_money": noul("Is `current_turn` about money?"),
        "d_weather": noul("Is `current_turn` about the weather?"),
        "d_time_ref": noul("Does `current_turn` mention a specific time or day?"),
        "d_person_ref": noul("Does `current_turn` mention a third person by name?"),
        "d_inside_joke": noul("Does `current_turn` seem to refer to an inside joke between the two?"),
        "d_rhetorical": noul("Is the question in `current_turn` rhetorical?"),
        "d_unanswered": noul("Does `current_turn` leave a question from `previous_turns` unanswered?"),
        "d_mood_drop": noul("Is the mood of `current_turn` worse than in `previous_turns`?"),
        "d_mood_rise": noul("Is the mood of `current_turn` better than in `previous_turns`?"),
        "d_energy_mismatch": noul("Is the energy of `current_turn` very different from the other person's last turn?"),
        "d_emph": noul("Does the speaker of `current_turn` use emphasis (caps, repeated letters, !!!) to show emotion?"),
        "s_warmth": score(f"How warm is {W} toward the other person?", ["cold", "neutral", "friendly", "affectionate"]),
        "s_effort": score(f"How much effort did {W} put into `current_turn`?", ["minimal", "some", "a lot"]),
        "s_urgency": score("How urgent is it to reply to `current_turn`?", ["not urgent", "somewhat", "urgent"]),
        "s_formality": score("How formal is `current_turn`?", ["very informal / slang", "casual", "neutral", "formal"]),
        "s_humor": score("How funny is `current_turn` meant to be?", ["not at all", "a little", "clearly funny"]),
        "s_intimacy": score("How intimate is the topic of `current_turn`?", ["public / trivial", "personal", "very private"]),
        "s_reply_len": score("How long should a natural reply to `current_turn` be?", ["1-3 words", "one short sentence", "one or two sentences", "several sentences"]),
        "s_confidence": score(f"How self-confident does {W} sound?", ["insecure", "neutral", "confident"]),
        "s_curiosity": score(f"How curious about the other person is {W}?", ["not at all", "somewhat", "very"]),
        "s_openness": score(f"How open about their feelings is {W}?", ["closed", "somewhat open", "very open"]),
        "c_reply_tone": choice("What tone should the reply to `current_turn` have?", {
            "playful": "joking, teasing, light", "warm_affectionate": "sweet, caring", "supportive_serious": "serious, empathetic",
            "enthusiastic": "excited", "neutral_informative": "matter-of-fact", "calm_reassuring": "calm, soothing"}),
        "c_reply_act": choice("What is the best next move after `current_turn`?", {
            "answer": "answer what was asked", "react": "short reaction", "ask": "ask something back", "comfort": "comfort",
            "joke": "joke back", "share": "share something about oneself", "plan": "make a plan", "close": "say goodbye"}),
        "c_bubbles": choice("In how many chat bubbles should the reply to `current_turn` be sent?", {"1": "one message", "2": "two messages", "3": "three messages", "4+": "four or more"}),
        "c_delay": choice("How quickly should the reply to `current_turn` come?", {"instant": "within seconds", "short": "within a minute", "later": "a few minutes later"}),
        "c_laugh_form": choice("If the reply laughs, which form fits best?", {"haha": None, "hahaha": None, "lol": None, "emoji_laugh": "😂", "none": "no laughing"}),
    }
    P.update(extra)
    # a few paraphrase duplicates to reach ~150
    for k in list(AB.EMOTIONS)[:12]:
        P[f"emo2_{k}"] = score(f"How strongly does {W} feel {k.replace('_', ' ')}?", ["not at all", "a little", "clearly", "strongly"])
    for k in list(AB.INTENTS)[:6]:
        P[f"int2_{k}"] = noul(f"Is the main purpose of `current_turn` to {AB.INTENTS[k].split(' / ')[0]}?")
    return P


if __name__ == "__main__":
    p = pool(); print(len(p))
