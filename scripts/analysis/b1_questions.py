"""b1: question banks for the judge architectures.
All per-candidate questions refer to the state fields `conversation_so_far` and `candidate_message`.
Convention: every atomic question is phrased so that YES / high = the trait is present. The expected
direction (LLM-like = +1, human-like = -1) is in SIGN; learned combinations do not depend on it."""
from jev import noul, choice, score

CM = "the candidate_message"
LAST = "the other person's last message in conversation_so_far"

# ------------------------------------------------------------------ A1: atomic bank (concrete traits)
ATOMIC = {
    # amount / structure
    "t_multi_ideas": noul(f"{CM} contains more than one separate idea or point."),
    "t_multi_acts": noul(f"{CM} does two or more of these things: reacts, comments or explains, shares something about the sender, asks a question."),
    "t_template": noul(f"{CM} follows the pattern: first a reaction, then a comment or restatement, then a question at the end."),
    "t_longer_than_other": noul(f"{CM} is longer than {LAST}."),
    "t_answers_several": noul(f"{CM} responds to more than one thing the other person said."),
    "t_explains": noul(f"{CM} explains or justifies something that did not need explaining."),
    "t_keeps_going": noul(f"After its first reaction or answer, {CM} keeps going with extra sentences that a person texting quickly would skip."),
    "t_full_sentences": noul(f"{CM} is written in complete, well-formed sentences."),
    "t_tidy_punct": noul(f"{CM} uses standard capitalization and punctuation throughout."),
    # questions
    "t_ends_generic_q": noul(f"{CM} ends with a generic follow-up question meant to keep the conversation going, such as 'what about you?', 'how was your day?' or 'what are you up to?'."),
    "t_ends_q": noul(f"{CM} ends with a question."),
    "t_interview_q": noul(f"{CM} asks the other person about their preferences, feelings or plans, the way an interviewer would."),
    "t_q_filler": noul(f"{CM} asks a question only to keep the conversation going, not because the sender needs the answer."),
    # validation / emotion
    "t_validation_formula": noul(f"{CM} uses a stock validation or sympathy phrase, such as 'that's amazing', 'I totally get that', 'sorry to hear that', 'that's so sweet', 'that sounds tough' or 'it's okay to feel'."),
    "t_names_emotion": noul(f"{CM} names the other person's feelings (for example 'you must be so excited', 'that sounds stressful') instead of reacting to what actually happened."),
    "t_reacts_to_fact": noul(f"{CM} reacts to a specific fact or detail from {LAST}."),
    "t_overvalidation": noul(f"{CM} praises, validates, reassures or empathizes more than the situation calls for."),
    "t_reassures": noul(f"{CM} reassures or encourages the other person, for example 'you got this', 'don't worry' or 'you'll be fine'."),
    "t_praises": noul(f"{CM} compliments or praises the other person."),
    "t_thanks": noul(f"{CM} thanks the other person."),
    "t_offers_support": noul(f"{CM} offers help or availability, for example 'I'm here for you' or 'let me know if you need anything'."),
    # enthusiasm / energy
    "t_more_excited": noul(f"{CM} is more excited or enthusiastic than {LAST}."),
    "t_superlatives": noul(f"{CM} uses exaggerated or superlative words such as amazing, incredible, so much, the best, obsessed."),
    "t_perf_opener": noul(f"{CM} opens with a performative interjection such as 'Oh wow', 'Omg', 'Aww', 'Haha', 'Ooh', 'Wait', 'Honestly' or 'Ugh' before the actual content."),
    "t_laughs_unfunny": noul(f"{CM} laughs (haha, lol, 😂, 💀) at something that was not actually funny."),
    "t_emoji_decor": noul(f"{CM} uses emojis as decoration or to add emotion."),
    "t_forced": noul(f"{CM} uses slang, emojis, internet expressions or enthusiasm in a forced, exaggerated or caricatured way."),
    "t_trendy_slang": noul(f"{CM} uses trendy internet slang such as fr, lowkey, bestie, slay, no cap, vibe, iconic or elite."),
    "t_over_warm": noul(f"{CM} is warmer or more affectionate than the moment calls for."),
    # echo
    "t_paraphrase": noul(f"{CM} restates or paraphrases what the other person just said."),
    "t_repeats_words": noul(f"{CM} reuses key words or phrases from {LAST}."),
    "t_summarizes": noul(f"{CM} summarizes or restates the situation before reacting to it."),
    # register
    "t_formal": noul(f"{CM} uses vocabulary, phrasing or structure that is too formal, polished, elaborate or assistant-like for this casual chat."),
    "t_assistant": noul(f"{CM} sounds like a customer-service agent or an AI assistant talking to a user."),
    "t_polite_markers": noul(f"{CM} uses polite or hedging stock phrases such as 'definitely', 'absolutely', 'I'd love to', 'feel free', 'for sure' or 'thanks for sharing'."),
    "t_balanced": noul(f"{CM} is carefully balanced or diplomatic, avoiding any blunt opinion."),
    "t_blunt": noul(f"{CM} is blunt, dry or curt."),
    "t_teases": noul(f"{CM} teases or mocks the other person playfully."),
    "t_self_focus": noul(f"{CM} is mainly about the sender's own life, plans or opinion."),
    "t_own_thread": noul(f"{CM} ignores {LAST} and continues the sender's own topic or starts a new one."),
    "t_minimal_ack": noul(f"{CM} is a minimal acknowledgment such as 'ok', 'true', 'haha', 'same' or 'nice'."),
    "t_casual_typing": noul(f"{CM} contains typos, texting abbreviations (u, ur, idk, tbh, rn) or missing apostrophes."),
    "t_insider": noul(f"{CM} refers to something only these two people would know, without explaining it."),
    "t_invented": noul(f"{CM} mentions facts, events or shared memories that are not supported by the conversation."),
    "t_generic": noul(f"{CM} is generic: it could be sent in reply to many different messages and does not depend on the specifics of what was just said."),
    "t_direct": noul(f"{CM} responds directly to the last thing the other person said."),
    "t_register": noul(f"{CM} matches the other person's register: similar length, formality, punctuation and energy."),
    "t_performs_personality": noul(f"{CM} tries hard to sound fun, quirky or relatable."),
    "t_scripted_warmth": noul(f"{CM} sounds like a friendly script: pleasant and positive but impersonal."),
    "t_mirror_question": noul(f"{CM} answers a question and then asks the same question back."),
    "t_advice": noul(f"{CM} gives advice or suggestions that were not asked for."),
    "t_hedged_opinion": noul(f"{CM} softens its opinion with qualifiers such as 'kind of', 'honestly', 'I think', 'maybe' or 'a bit'."),
}
SIGN = {k: 1 for k in ATOMIC}
for k in ["t_reacts_to_fact", "t_blunt", "t_teases", "t_own_thread", "t_minimal_ack", "t_casual_typing", "t_insider", "t_generic",
          "t_self_focus"]:
    SIGN[k] = -1
for k in ["t_direct", "t_register"]:  # a8: these came out inverted; we keep the a-priori sign as in a8 (+1 = human)
    SIGN[k] = -1

# ------------------------------------------------------------------ A8: counterfactual questions (same state)
COUNTERFACTUAL = {
    "cf_friend_shorter": noul(f"If a real friend replied to {LAST} in a casual text chat, their reply would be shorter than {CM}."),
    "cf_stop_first": noul(f"A real friend texting quickly would have stopped after the first sentence or phrase of {CM}."),
    "cf_friend_no_q": noul(f"A real friend would not bother to ask the question that {CM} asks."),
    "cf_friend_less_emotion": noul(f"A real friend would say this with less emotion and enthusiasm than {CM}."),
    "cf_suspect_ai": noul(f"If the other person read {CM}, they would suspect it was written by an AI."),
    "cf_fits_any": noul(f"{CM} could be pasted into a different conversation and still make sense."),
    "cf_friend_same": noul(f"A real friend in this exact situation could plausibly have sent exactly {CM}."),
}
SIGN.update({k: 1 for k in COUNTERFACTUAL}); SIGN["cf_friend_same"] = -1

# ------------------------------------------------------------------ A9: paraphrase ensembles (3 extra wordings per concept)
PARA = {
    "validation": ["t_validation_formula",
                   noul(f"{CM} contains a ready-made supportive phrase, the kind a greeting card or a therapist would use."),
                   noul(f"{CM} tells the other person that their feelings or news are valid, great, understandable or sad."),
                   noul(f"{CM} includes an empathy formula like 'that must be hard' or 'so happy for you'.")],
    "echo": ["t_paraphrase",
             noul(f"{CM} repeats back the content of {LAST} in other words."),
             noul(f"Part of {CM} just mirrors what the other person already said, without adding anything new."),
             noul(f"{CM} echoes the other person's words before responding.")],
    "enthusiasm": ["t_more_excited",
                   noul(f"{CM} has more energy, excitement or exclamation than the conversation so far."),
                   noul(f"{CM} sounds over-excited for what was said."),
                   noul(f"{CM} raises the emotional volume compared with {LAST}.")],
    "too_much": ["t_multi_acts",
                 noul(f"{CM} tries to do too much at once (for example react, comment and ask a follow-up question) where a real person would just reply to one thing."),
                 noul(f"{CM} packs several moves into one reply instead of making a single point."),
                 noul(f"{CM} could lose one or more of its sentences without losing anything important.")],
    "assistant": ["t_assistant",
                  noul(f"{CM} reads like an AI chatbot's reply rather than a text from a person."),
                  noul(f"{CM} has the polished, accommodating tone of a helpful assistant."),
                  noul(f"{CM} sounds like it was written to please the reader rather than typed casually.")],
    "filler_q": ["t_ends_generic_q",
                 noul(f"{CM} ends by bouncing the conversation back with a question like 'you?' or 'what about you?'."),
                 noul(f"{CM} closes with a question whose main purpose is to keep the other person talking."),
                 noul(f"{CM} ends with a routine small-talk question.")],
}
PARA_Q = {}
for concept, lst in PARA.items():
    for j, q in enumerate(lst[1:], 1):
        PARA_Q[f"p_{concept}_{j}"] = q
SIGN.update({k: 1 for k in PARA_Q})

# ------------------------------------------------------------------ A10: anchored scores (+ holistic reference)
ANCHORED = {
    "s_friend_vs_assistant": score(f"How does {CM} read, from a friend's text to an assistant's reply?", [
        "Friend's text: short, one reaction or one piece of content, no framing",
        "Mostly a friend's text, with one small extra such as a light follow-up or a filler phrase",
        "Mixed: a reaction plus a comment, a bit polished",
        "Leaning assistant: validates or restates, then adds an explanation or a question",
        "Assistant's reply: validation plus explanation plus follow-up question, polished and enthusiastic"]),
    "s_energy_vs_other": score(f"Compared with {LAST}, the energy and enthusiasm of {CM} is:", [
        "much lower", "a bit lower", "about the same", "a bit higher", "much higher"]),
    "s_length": score(f"Compared with what a real person would text at this point in this chat, the length of {CM} is:", [
        "much too short", "a bit short", "about right", "a bit too long", "much too long"]),
    "s_polish": score(f"How polished is the writing in {CM}?", [
        "Raw texting: lowercase, fragments, abbreviations, little punctuation",
        "Casual texting with a few tidy touches",
        "Neutral: normal sentences, some casual touches",
        "Clean: full sentences, standard punctuation",
        "Edited prose: full sentences, standard punctuation, varied vocabulary, commas and transitions"]),
    "s_moves": score(f"How many different things does {CM} do (react, comment, explain, share own news, give advice, ask)?", [
        "one thing", "two things", "three things", "four or more things"]),
    "s_validation": score(f"How much validating, sympathizing or praising does {CM} do?", [
        "none", "a light touch", "a clear validating phrase", "several validating phrases", "the whole message is validation"]),
    "h_human": noul(f"Was {CM} typed by a real person in this chat, rather than written by an AI chatbot?", "Typed by a real person", "Written by an AI chatbot"),
    "h_ai": noul(f"{CM} was written by an AI chatbot, not typed by a real person."),
}
SIGN.update({"s_friend_vs_assistant": 1, "s_energy_vs_other": 1, "s_length": 1, "s_polish": 1, "s_moves": 1, "s_validation": 1,
             "h_human": -1, "h_ai": 1})

# ------------------------------------------------------------------ A11: vice hierarchy (stage 2 Choice + main-vice)
VICES = {  # vice -> (atomic id used as the multilabel Noul, short description)
    "too_long": ("t_keeps_going", "too long for the moment: keeps going after the point is made"),
    "filler_question": ("t_ends_generic_q", "ends with an unneeded follow-up question"),
    "enthusiasm": ("t_more_excited", "more enthusiasm or exclamation than the other person"),
    "validation": ("t_validation_formula", "stock validation, sympathy or praise"),
    "echo": ("t_paraphrase", "restates what the other person said"),
    "template": ("t_template", "reaction, then comment, then question"),
    "forced_casual": ("t_forced", "forced slang, emoji or laughter"),
    "assistant_tone": ("t_formal", "formal, polished, assistant-like tone"),
}
MAIN_VICE = {"v_main": choice(
    f"What is the main thing that makes {CM} sound unlike a real friend's text in this chat?",
    {**{k: d for k, (_, d) in VICES.items()}, "nothing": "nothing: it reads like a normal text from a friend"})}

BANK = {**ATOMIC, **COUNTERFACTUAL, **PARA_Q, **ANCHORED, **MAIN_VICE}

# subset used in the state-variant architectures (A2 rubric, A3 few-shot, A4 speaker ref comparisons)
CORE = ["t_multi_acts", "t_template", "t_ends_generic_q", "t_validation_formula", "t_names_emotion", "t_overvalidation",
        "t_more_excited", "t_perf_opener", "t_forced", "t_paraphrase", "t_formal", "t_assistant", "t_minimal_ack",
        "t_generic", "cf_friend_shorter", "cf_suspect_ai", "s_friend_vs_assistant", "h_human", "h_ai"]
