"""b1 step 3: architectures that change the STATE (same or relative questions), on the core subset.
  A2  rubric   : a guide 'how people text vs how AI chatbots text' with OUR measured rates in the state + CORE questions
  A3a fewshot  : 8 labeled pairs (real reply vs AI reply) from OTHER (dev) conversations in the state + CORE questions
  A3b realshot : 10 real replies from other (dev) conversations, no AI examples (unlabeled 'this is how people text') + CORE
  A4  speaker  : 10 real messages from the SAME speaker (other parts of the conversation) + relative questions
  A5  cascade  : stage 1 = moment Choice (per context); stage 2 = moment guide + moment-specific questions
  A6  chain    : bank readings (stage 1 Jev) + code measurements written into the state of a 2nd Jev -> AI? / how artificial / main vice
Usage: python3 b1_variants.py rubric|fewshot|realshot|speaker|moment|cascade|chain|chain_nocode
Output: scratch b1/var_<name>.json {uid: {qid: value}}"""
import json, os, random, sys
from collections import defaultdict
import b1_common as B
import b1_questions as Q
from jev import noul, choice, score

CM = Q.CM

GUIDE = """Measured in thousands of real casual chats between people who know each other, versus AI chatbots replying at the same points:
- Length: real replies are short (median about 30 characters; 45% have 3 words or fewer; only 13% have 3+ sentences). AI replies are 2-2.6x longer, with 3+ sentences in 42-61%.
- One idea: a real person reacts OR comments OR asks. AI does all three: reaction, then comment or restatement, then a question (20-25% of AI replies, under 2% of real ones).
- Questions: 20% of real replies end with a question, almost never after good news, flirting or a joke; AI ends with a question 44-58% of the time, often a generic one ('what about you?', 'how was your day?': 18% AI vs 1% real).
- Punctuation and energy: '!' in 7% of real replies vs 40-92% of AI; emoji 2-5% vs up to 60%; AI opens with performative interjections ('Oh wow', 'Aww', 'Omg', 'Haha', 'Wait', 'Honestly') in 37% vs 5% real. Real people mostly start plainly ('ok', 'so', 'yeah', 'i', 'but') and usually write lowercase with no final period.
- Validation: real people almost never use stock phrases ('that's amazing', 'sorry to hear that', 'I totally get that', 'that's so sweet', 'it's okay to feel'); they react to the FACT ('u did scream', 'send pic') instead of naming the feeling. AI validates, praises and reassures.
- Echo: AI restates what the other said (14-21%) vs 3% real.
- Real replies are often generic backchannel ('true', 'haha', 'fair', 'same'), sometimes ignore the last message and continue their own thread (1 in 5), tease, or are blunt. AI answers everything, is balanced and warm, adds enthusiasm the other did not have, and sometimes forces slang ('fr', 'lowkey', 'bestie', '💀') or invents shared memories."""

MOMENTS = {
    "flirt_affection": "flirting, affection, compliments, 'miss you' or 'love you'",
    "banter_joke": "teasing, joking, banter or something funny",
    "venting_bad_news": "complaining, venting, bad news, feeling down or stressed",
    "good_news": "sharing good news or excitement",
    "plans_logistics": "making plans, logistics or practical arrangements",
    "question_to_me": "asking the other person a direct question about facts, opinions or preferences",
    "greeting_how_are_you": "a greeting, 'how are you' or a small-talk opener",
    "closing": "wrapping up, saying goodbye or leaving to sleep or work",
    "story_info": "telling a story or sharing neutral information",
}
MOMENT_Q = {"moment": choice("What kind of moment is the other person's last message in conversation_so_far? Pick what they just did.", MOMENTS)}

MOMENT_GUIDE = {
    "flirt_affection": "Real friends answer affection BELOW its intensity: they often just continue the topic (32%), send 'hehe', tease back, get flustered ('stop', 'dont expose me') or return it briefly and symmetrically ('love u too'). They almost never thank, gush or ask a question (0 of 25 replies ended with a question). AI gushes ('Aww you're making me blush 🥺'), thanks, adds 'can't wait' and asks a question.",
    "banter_joke": "Real friends answer a joke with another joke or a dry reframe ('define productive', 'allegedly'), not with 'haha that's hilarious'. Questions are rare (3-6%). AI laughs, praises the joke, explains it, turns wholesome and asks a question.",
    "venting_bad_news": "Real friends open very short ('oh no', 'ugh', 'wait what') and often ask ONE concrete question ('what happened?'); explicit consolation is rare and 'sorry to hear' essentially never appears. AI writes a paragraph of validation, names the feelings, reassures, offers support and advice.",
    "good_news": "Real friends comment on the FACT ('u did scream', 'passed??', 'finally') and rarely ask a follow-up (8%); 'congrats' and 'that's amazing' are near zero. AI names the emotion ('That's amazing!! So happy for you!'), uses '!' and emojis, and asks how they will celebrate (56-64%).",
    "plans_logistics": "Real friends confirm tersely ('ok', 'sounds good', 'see you then') or add one practical detail. AI confirms with enthusiasm formulas ('Sounds perfect! Can't wait!'), restates the plan, adds extra suggestions and questions.",
    "question_to_me": "Real friends answer directly and briefly (median ~30 characters: 'yes', 'hmm', 'aries - im so firey'); they return the question only sometimes. AI answers, elaborates, adds a preamble and asks the same question back.",
    "greeting_how_are_you": "Real friends answer 'how are you' with ONE concrete detail of their day ('meh / tired but ok', 'just got home, exhausted'), about 35 characters, and ask back about a third of the time. AI replies with a generic pleasantry ('I'm doing great, thanks for asking!'), upbeat punctuation and 'how about you?' 70-90% of the time.",
    "closing": "Real friends mirror the goodbye briefly with a concrete anchor ('ok see you friday', 'night') and often don't say goodbye at all. AI thanks for the chat, adds wishes ('take care! talk soon!') and extra sentences.",
    "story_info": "Real friends react briefly to a detail, share a related bit, or just backchannel ('true', 'lol same'). AI comments on how interesting it is, restates the information, uses '!' and asks a follow-up question.",
}
MQ = {
    "flirt_affection": ["responds to the affection by thanking or gushing", "adds a question after responding to the affection",
                        "is more intense in affection than the other person", "uses cute emojis such as 🥺, 🥰 or 😊 to perform warmth",
                        "says explicitly how the other's words make the sender feel (blushing, smiling, made my day)"],
    "banter_joke": ["laughs (haha, lol, 😂) instead of joking back", "explains or comments on the joke instead of playing along",
                    "praises the joke or the other person's humor", "adds a question after reacting to the joke",
                    "breaks the playful tone with sincere or wholesome talk"],
    "venting_bad_news": ["uses a sympathy formula such as 'sorry to hear' or 'that sounds hard'", "names or validates the other person's feelings",
                         "offers encouragement or advice that was not asked for", "offers support such as 'I'm here for you'",
                         "is much longer than a quick concerned reply"],
    "good_news": ["names the emotion ('so exciting', 'so happy for you') instead of reacting to the specific fact", "congratulates with a stock formula",
                  "uses exclamation marks or celebration emojis", "asks a follow-up question about the news", "praises the other person"],
    "plans_logistics": ["confirms with an enthusiasm formula such as 'sounds perfect!' or 'can't wait!'", "adds suggestions or details beyond what is needed",
                        "asks more than one practical question", "restates the plan", "adds warm filler around the practical content"],
    "question_to_me": ["answers and then asks the same question back", "gives more detail than was asked for",
                       "adds a reaction or preamble before answering", "answers in a balanced, hedged way", "ends with a follow-up question"],
    "greeting_how_are_you": ["replies with a generic pleasantry such as 'I'm good, thanks!' without a concrete detail", "asks 'how about you?' or a similar question back",
                             "uses exclamation marks or emoji to sound upbeat", "offers a 'what's up' or 'how can I help' style opener",
                             "is more energetic than the other person's greeting"],
    "closing": ["extends the goodbye with wishes and extra sentences", "thanks the other person for the chat", "adds a question after the goodbye",
                "uses a formula such as 'take care' or 'talk soon!'", "is warmer than the other person's goodbye"],
    "story_info": ["comments on how interesting or cool the information is", "restates the information", "asks a follow-up question",
                   "adds a long related story of its own", "uses exclamation marks or emojis to show interest"],
}


def cascade_questions(m):
    qs = {f"m{j}": noul(f"{CM} {t}.") for j, t in enumerate(MQ[m])}
    qs["m_ai_pattern"] = noul(f"{CM} does what moment_guide says AI chatbots do in this moment.")
    qs["m_fits"] = noul(f"{CM} responds to this moment the way moment_guide says real friends do.")
    return qs


# ---------------------------------------------------------------- speaker reference
REL = lambda me: {
    "r_longer": noul(f"{CM} is longer than the messages {me} usually sends (see how_{me}_usually_texts)."),
    "r_more_enthusiastic": noul(f"{CM} is more enthusiastic or exclamatory than {me} usually is in how_{me}_usually_texts."),
    "r_more_formal": noul(f"{CM} is more formal or polished than the way {me} usually writes in how_{me}_usually_texts."),
    "r_punct_diff": noul(f"{CM} uses capitalization or punctuation that {me} does not usually use in how_{me}_usually_texts."),
    "r_new_words": noul(f"{CM} uses words, expressions, slang or emojis that {me} never uses in how_{me}_usually_texts."),
    "r_more_markers": noul(f"{CM} uses more slang, emoji or laughter than {me} usually does in how_{me}_usually_texts."),
    "r_more_questions": noul(f"{CM} asks more questions than {me} usually does in how_{me}_usually_texts."),
    "r_sounds_like": noul(f"{CM} sounds like the way {me} usually texts in how_{me}_usually_texts."),
    "r_ai": noul(f"{CM} was written by an AI imitating {me}, not by {me}."),
    "t_forced": Q.ATOMIC["t_forced"], "t_formal": Q.ATOMIC["t_formal"], "t_more_excited": Q.ATOMIC["t_more_excited"], "h_ai": Q.ANCHORED["h_ai"],
}
REL_SIGN = {k: 1 for k in REL("x")}; REL_SIGN["r_sounds_like"] = -1


def speaker_refs(ctxs, units):
    T = B._turns_maichat()
    by_conv = defaultdict(list)
    for t in T:
        by_conv[t["conv_id"]].append(t)
    refs = {}
    for ck, c in ctxs.items():
        if c["src"] not in ("a8_maichat", "a9") or c["turn_idx"] is None or not c.get("speaker"):
            continue
        pool = [x for t in by_conv[c["conv"]] if t["speaker"] == c["speaker"] and abs(t["turn_idx"] - c["turn_idx"]) > 10
                for x in t["texts"] if x.strip()]
        rnd = random.Random(ck)
        rnd.shuffle(pool)
        refs[ck] = pool[:10]
    return refs


# ---------------------------------------------------------------- few-shot pools (dev only, other conversations)
def example_pool(ctxs, units):
    by = defaultdict(dict)
    for u in units:
        c = ctxs[u["ckey"]]
        if c["split"] == "dev" and c["src"] == "a8_maichat":
            by[u["ckey"]][u["cond"]] = u["text"]
    pool = []
    for ck, d in by.items():
        if "human" in d:
            llm = [d[k] for k in ("base_gemini", "base_gpt4omini", "base_llama70b", "styled_gemini") if k in d]
            pool.append({"ckey": ck, "conv": ctxs[ck]["conv"], "msg": B.last_other(ctxs[ck]), "human": d["human"], "llm": llm})
    return pool


def pick_examples(pool, ck, conv, k, seed):
    rnd = random.Random(f"{ck}:{seed}")
    cand = [p for p in pool if p["conv"] != conv]
    return rnd.sample(cand, k)


CHAIN_TRAITS = ["t_keeps_going", "t_ends_generic_q", "t_more_excited", "t_validation_formula", "t_paraphrase", "t_template", "t_forced",
                "t_formal", "t_multi_acts", "t_perf_opener", "t_names_emotion", "t_minimal_ack", "cf_friend_shorter", "t_casual_typing"]


def chain_state(ctx, u, bank, with_code):
    b = bank[u["uid"]]
    readings = {Q.BANK[t]["instructions"].replace(CM, "the reply").replace(Q.LAST, "the other person's last message"): round(b[t], 2) for t in CHAIN_TRAITS}
    st = B.base_state(ctx, u["text"])
    st["trait_detector_readings (probability that each statement is true of candidate_message)"] = readings
    if with_code:
        f = B.code_feats(u["text"], B.last_other(ctx))
        st["code_measurements"] = {"characters": len(u["text"]), "characters_in_other_persons_last_message": len(B.last_other(ctx)),
                                   "sentences": f["sentences"], "ends_with_question": bool(f["ends_q"]), "exclamation_marks": f["n_excl"],
                                   "emojis": f["n_emoji"], "stock_phrases_found": f["llmish_hits"],
                                   "typical_real_reply": "about 30 characters, 1 sentence, no '!', 20% end with a question"}
    return st


CHAIN_Q = {
    "c_ai": noul(f"Taking into account conversation_so_far, {CM} and the readings in the state, {CM} was written by an AI chatbot rather than typed by a real person."),
    "c_artificial": score(f"How artificial does {CM} read, taking the readings in the state into account?", Q.ANCHORED["s_friend_vs_assistant"]["criteria"]),
    "c_main": choice(f"What is the main thing that makes {CM} sound unlike a real friend's text?", Q.MAIN_VICE["v_main"]["criteria"]),
}


def compact(ans):
    d = {}
    for q, a in (ans or {}).items():
        d[q] = B.val(a)
        if a["type"] == "choice":
            d[q + "_p"] = a["probabilities"]; d[q + "_conf"] = a.get("confidence")
    return d


def run(name):
    ctxs, units = B.load_units()
    core = B.core_units(ctxs, units)
    outp = os.path.join(B.SCR, f"var_{name}.json")
    done = json.load(open(outp)) if os.path.exists(outp) else {}
    core_q = {k: Q.BANK[k] for k in Q.CORE}
    items, keys = [], []
    if name == "moment":
        cks = sorted({u["ckey"] for u in core})
        for ck in cks:
            if ck in done:
                continue
            c = ctxs[ck]
            items.append(({"setting": B.base_state(c, "")["setting"], "conversation_so_far": B.convo_lines(c)}, MOMENT_Q)); keys.append(ck)
    else:
        pool = example_pool(ctxs, units) if name in ("fewshot", "realshot") else None
        refs = speaker_refs(ctxs, units) if name == "speaker" else None
        mom = json.load(open(os.path.join(B.SCR, "var_moment.json"))) if name == "cascade" else None
        bank = json.load(open(os.path.join(B.SCR, "bank.json"))) if name.startswith("chain") else None
        for u in core:
            if u["uid"] in done:
                continue
            c = ctxs[u["ckey"]]
            if name == "rubric":
                st = B.base_state(c, u["text"], {"_first": True, "guide_how_people_text_vs_ai_chatbots": GUIDE}); qs = core_q
            elif name == "fewshot":
                ex = pick_examples(pool, u["ckey"], c["conv"], 8, 0)
                rnd = random.Random(u["uid"])
                exs = [{"their_message": e["msg"], "real_person_reply": e["human"], "ai_chatbot_reply": rnd.choice(e["llm"])} for e in ex]
                st = B.base_state(c, u["text"], {"_first": True, "labeled_examples_from_other_chats": exs}); qs = core_q
            elif name == "realshot":
                ex = pick_examples(pool, u["ckey"], c["conv"], 10, 1)
                exs = [{"their_message": e["msg"], "reply": e["human"]} for e in ex]
                st = B.base_state(c, u["text"], {"_first": True, "how_real_people_reply_in_chats_like_this": exs}); qs = core_q
            elif name == "speaker":
                r = refs.get(u["ckey"])
                if not r or len(r) < 5:
                    continue
                st = B.base_state(c, u["text"], {"_first": True, f"how_{c['self']}_usually_texts": r}); qs = REL(c["self"])
            elif name == "cascade":
                m = mom.get(u["ckey"])
                if not m:
                    continue
                st = B.base_state(c, u["text"], {"_first": True, "moment": MOMENTS[m["moment"]], "moment_guide": MOMENT_GUIDE[m["moment"]]})
                qs = cascade_questions(m["moment"])
            elif name in ("chain", "chain_nocode"):
                if u["uid"] not in bank:
                    continue
                st = chain_state(c, u, bank, name == "chain"); qs = CHAIN_Q
            items.append((st, qs)); keys.append(u["uid"])
    print(name, "todo", len(items))
    for i in range(0, len(items), 300):
        res = B.run_jev("A_" + name, items[i:i + 300])
        for k, r in zip(keys[i:i + 300], res):
            if r:
                done[k] = compact(r)
        json.dump(done, open(outp, "w"))
    print("saved", len(done))


if __name__ == "__main__":
    run(sys.argv[1])
