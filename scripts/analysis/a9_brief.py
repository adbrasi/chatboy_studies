"""a9 — o "cérebro": perguntas do Jev sobre o momento + conversão EM CÓDIGO para um briefing curto e imperativo.
Uma chamada do Jev por ponto de decisão (todas as perguntas juntas, avaliadas em paralelo e isoladas).
Uso: python3 a9_brief.py [dev|test]  -> grava p["jev"] (respostas compactas) e p["lat"]["jev_brief"] em a9_points.jsonl
Os limiares usados em build_brief() foram calibrados no split dev (ver a9_calibrate.py) e congelados antes do teste."""
import json, os, sys
from a9_common import ask_many_timed, load_points, save_points, BOT, USER, ADATA, merge_json
import jev
from jev import noul, choice, score

CTX = 8  # turnos no state do Jev (state enxuto)

Q = {
    # --- leitura do momento / do parceiro (sobre a última mensagem) ---
    "teasing": noul(f"Is {USER} teasing, joking or being playful in `last_message`?"),
    "flirting": noul(f"Is {USER} flirting with {BOT} or being romantic in `last_message`?"),
    "affection": noul(f"Does `last_message` express affection, care, gratitude or a compliment toward {BOT}?"),
    "vulnerable": noul(f"Is {USER} sharing a worry, a problem, sadness or something personal and vulnerable in `last_message`?"),
    "good_news": noul(f"Is {USER} sharing good news or excitement in `last_message`?"),
    "complaint": noul(f"Is {USER} complaining, ranting or venting about something in `last_message`?"),
    "direct_q": noul(f"Does `last_message` ask {BOT} a direct question that expects an answer?"),
    "closing": noul(f"Is {USER} wrapping up or saying goodbye in `last_message`?"),
    "greeting": noul(f"Is `last_message` a greeting that opens the conversation?"),
    "logistics": noul(f"Is `last_message` about practical logistics such as plans, times, places or tasks?"),
    "annoyed": noul(f"Is {USER} annoyed or upset with {BOT} in `last_message`?"),
    "sarcasm": noul(f"Is `last_message` sarcastic or ironic?"),
    "story_ongoing": noul(f"Is {USER} in the middle of telling something, with more still to come?"),
    "minimal": noul(f"Is `last_message` a minimal reply (like 'ok', 'yeah', 'lol', 'true') that adds nothing new?"),
    "wants": choice(f"What does {USER} most want from {BOT}'s next message?", {
        "laughter": "to have fun, laugh, keep the banter going",
        "comfort": "comfort, sympathy or reassurance",
        "answer": "an answer or information",
        "attention": "interest and attention to what they shared",
        "validation": "agreement or validation of their opinion",
        "plan": "a decision or confirmation about a plan",
        "acknowledgment": "just a quick acknowledgment",
        "closure": "to end the conversation",
    }),
    "seriousness": score("How serious is the current moment of the conversation?",
                         ["playful banter", "casual", "somewhat serious", "very serious or emotional"]),
    "energy": score(f"How energetic or excited is {USER} in `last_message`?", ["flat, low energy", "calm", "lively", "very excited"]),
    # --- o que o próximo turno de Sam deve ter (posição do bot) ---
    "p_laugh": noul(f"In {BOT}'s next message, would {BOT} naturally laugh (e.g. 'haha', 'lol', 'lmao')?"),
    "p_question": noul(f"Will {BOT}'s next message ask {USER} a question?"),
    "p_emoji": noul(f"Will {BOT}'s next message contain an emoji?"),
    "p_joke": noul(f"Would a light joke or playful comment from {BOT} be welcome right now?"),
    "p_empathy": noul(f"Does the moment call for {BOT} to show empathy or support?"),
    "p_excitement": noul(f"Does the moment call for {BOT} to show excitement?"),
    "p_overkill": noul(f"Would an enthusiastic, gushing or very emotional reply from {BOT} feel over the top here?"),
    "p_share_own": noul(f"Would it be natural for {BOT} to share their own related experience or opinion next?"),
    "length": score(f"How long will {BOT}'s next message naturally be, as friends texting casually?",
                    ["1-3 words, a quick reaction", "one short sentence (4-8 words)", "one full sentence (9-15 words)",
                     "two or three sentences (16-30 words)", "a long message (over 30 words)"]),
    "bubbles": score(f"How will {BOT} send the next reply?",
                     ["as one single message", "split into two quick messages", "split into three or more quick messages"]),
    "move": choice(f"What is the most natural next move for {BOT}?", {
        "react_only": "a short reaction only ('lol', 'nice', 'oh no', 'fair', 'ooh')",
        "answer": f"answer {USER}'s question directly",
        "tease_back": f"tease {USER} back",
        "joke_riff": "add to the joke or riff on it",
        "empathize": "show understanding or sympathy",
        "reassure": f"reassure or encourage {USER}",
        "share_own": "share own related experience or opinion",
        "ask_follow_up": f"ask a follow-up about what {USER} said",
        "flirt_back": "flirt back",
        "compliment": f"compliment {USER}",
        "agree": "agree",
        "disagree": "disagree or push back",
        "plan": "propose or confirm a plan",
        "goodbye": "say goodbye",
        "greet_back": "greet back",
        "new_topic": "bring up a new topic",
    }),
    "tone": choice(f"Which tone fits {BOT}'s next message best?", {
        "playful": "playful, joking", "warm": "warm, affectionate", "sincere": "sincere, supportive",
        "matter_of_fact": "plain, matter-of-fact", "excited": "excited", "flirty": "flirty",
        "dry": "dry, deadpan, understated"}),
}


def state_of(p):
    h = p["history"][-CTX:]
    return {"conversation": [{"from": x["who"], "text": x["text"]} for x in h[:-1]],
            "last_message": {"from": USER, "text": h[-1]["text"]}}


def compact(a):
    out = {}
    for k, v in a.items():
        if v["type"] == "noul":
            out[k] = v["noul"]
        elif v["type"] == "score":
            out[k] = v["score"]; out[k + "_conf"] = v["confidence"]
        else:
            out[k] = v["choice"]; out[k + "_p"] = v["probabilities"]; out[k + "_conf"] = v["confidence"]
    return out


# ------------------------------------------------------------------ briefing (código)
# Limiares CONGELADOS a partir do dev (a9_calibrate.py). Os Nouls "p_*" do Jev superestimam as taxas humanas
# (ex.: humanos perguntam em 9,5% dos turnos; o Jev dá ~0,35 em média), então os cortes são altos.
TH = json.load(open(os.path.join(ADATA, "a9_thresholds.json"))) if os.path.exists(os.path.join(ADATA, "a9_thresholds.json")) else {
    "q": 0.55, "laugh": 0.5, "emoji": 0.45, "bub2": 0.6, "bub3": 1.4, "len_words": [3, 6, 10, 18, 30]}

MOVE_TXT = {
    "react_only": "Just react in 1-3 words (like 'lol', 'fair', 'oh no', 'nice'). Add nothing else.",
    "answer": f"Answer {USER}'s question directly and plainly.",
    "tease_back": f"Tease {USER} back.",
    "joke_riff": "Riff on the joke with one quick line.",
    "empathize": "Show you get it, simply, like a friend. No advice, no therapist talk.",
    "reassure": f"Reassure {USER} briefly, like a friend would.",
    "share_own": "Share your own take or experience in one line.",
    "ask_follow_up": f"Ask one specific follow-up about what {USER} said.",
    "flirt_back": "Flirt back lightly.",
    "compliment": f"Compliment {USER} casually.",
    "agree": "Just agree, casually.",
    "disagree": "Push back or disagree, casually.",
    "plan": "Confirm or propose the plan concretely.",
    "goodbye": "Say bye, short.",
    "greet_back": "Greet back casually.",
    "new_topic": "Bring up something new.",
}
MOMENT_LBL = {
    "closing": f"{USER} is wrapping up", "greeting": "start of the chat",
    "serious": f"serious moment, {USER} is sharing something personal", "annoyed": f"{USER} is annoyed with you",
    "flirting": f"{USER} is flirting with you", "affection": f"{USER} is being sweet to you",
    "complaint": f"{USER} is venting", "good_news": f"{USER} is sharing good news",
    "teasing": f"light banter, {USER} is teasing/joking", "logistics": "logistics/plans",
    "minimal": f"{USER} gave a minimal reply", "casual": "casual chat",
}


TONE_TXT = {"playful": "playful", "warm": "warm", "sincere": "sincere", "matter_of_fact": "plain",
            "excited": "excited", "flirty": "flirty", "dry": "dry, deadpan"}


def moment_of(j):
    """Rótulo do momento = argmax dos Nouls de leitura. Guardas aprendidas no dev: 'vulnerable' e 'annoyed' do Jev
    disparam em provocação de brincadeira, então são descontados pela probabilidade de 'teasing'."""
    c = {k: j[k] for k in ("closing", "greeting", "flirting", "affection", "complaint", "good_news", "teasing",
                           "logistics", "minimal")}
    c["serious"] = max(j["vulnerable"] * (1 - j["teasing"]), 1.0 if j["seriousness"] >= 1.8 else 0)
    c["annoyed"] = j["annoyed"] * (1 - j["teasing"])
    k = max(c, key=c.get)
    return k if c[k] >= 0.55 else "casual"


BAN = ["aww", "totally", "absolutely", "amazing", "sounds like", "that sounds", "I'd love", "honestly", "vibe(s)",
       "super", "definitely", "journey", "em dashes (—)", "exclamation marks"]


def len_words(j):
    import numpy as np
    xs, ys = TH["len_map"]
    return float(np.interp(j["length"], xs, ys))


def build_brief(j, fp, variant="short"):
    """j: respostas compactas do Jev; fp: impressão digital de estilo do bot. variant: short | noban | long"""
    if variant == "long":
        return build_long(j, fp)
    L = []
    moment = moment_of(j)
    L.append(f"Moment: {MOMENT_LBL[moment]}.")
    L.append(MOVE_TXT[j["move"]] + f" Tone: {TONE_TXT[j['tone']]}.")
    # tamanho: escala do Jev (quantis casados com os humanos) misturada com a "voz" da persona (mediana própria);
    # nº de bolhas: hábito da persona (o Score de bolhas do Jev não previu nada no dev: Spearman -0,16)
    own = max(2.0, fp["median_words_per_bubble"] * max(1.0, fp["bubbles_per_turn"]))
    mw = int(round(max(2, 0.6 * len_words(j) + 0.4 * own))) + 1
    if j["move"] == "react_only":
        mw = min(mw, 4)
    nb = 2 if (fp["bubbles_per_turn"] >= 1.5 and mw >= 7) else 1
    L.append(f"Max {mw} words" + (f", split into {nb} short messages, one per line." if nb > 1 else ", one message."))
    no = []
    if j["p_question"] >= TH["q"] and j["move"] != "react_only":
        L.append("You can end with one short question.")
    else:
        no.append("no question")
    tok = fp.get("laugh_token") or "lol"
    if j["p_laugh"] >= TH["laugh"] and moment != "serious":
        L.append(f"Laugh the way you usually do ('{tok}').")
    elif j["p_laugh"] < 0.35 or moment == "serious":
        no.append("no laughing")
    if j["p_emoji"] < TH["emoji"] or fp["emoji_frac"] < 0.05:
        no.append("no emoji")
    if no:
        L.append(", ".join(no).capitalize() + ".")
    if moment == "serious" and j["p_joke"] < 0.5:
        L.append("No jokes. Be simple and sincere.")
    elif j["p_overkill"] > 0.5 and j["p_excitement"] < 0.5:
        L.append("Keep it low-key, not gushing.")
    st = []
    if fp["lower_frac"] > 0.7:
        st.append("all lowercase")
    if fp["period_frac"] < 0.2:
        st.append("no final period")
    # (dev: listar as gírias da própria persona, "words you use: idk, omg", fez a LLM enfiar gíria em tudo -> removido)
    L.append("Style: " + "; ".join(st) + ".")
    L.append(f"It must make sense as a direct reply to {USER}'s last message.")
    if variant != "noban":
        L.append("Never use: " + ", ".join(BAN) + ".")
    return "\n".join(L)


def build_long(j, fp):
    """Ablação: despejar tudo o que o Jev disse, em prosa, sem transformar em ordens concretas."""
    def pct(x):
        return f"{int(round(100 * x))}%"
    wants = sorted(j["wants_p"].items(), key=lambda kv: -kv[1])[:3]
    moves = sorted(j["move_p"].items(), key=lambda kv: -kv[1])[:3]
    return (
        f"Conversation analysis. Reading of {USER}'s last message: teasing {pct(j['teasing'])}, flirting {pct(j['flirting'])}, "
        f"affection {pct(j['affection'])}, vulnerable {pct(j['vulnerable'])}, good news {pct(j['good_news'])}, complaint "
        f"{pct(j['complaint'])}, asks you a question {pct(j['direct_q'])}, closing {pct(j['closing'])}, annoyed at you "
        f"{pct(j['annoyed'])}, sarcasm {pct(j['sarcasm'])}. Seriousness {j['seriousness']:.1f}/3, energy {j['energy']:.1f}/3. "
        f"What {USER} wants: " + ", ".join(f"{k} ({pct(v)})" for k, v in wants) + ". "
        f"Recommended moves: " + ", ".join(f"{k} ({pct(v)})" for k, v in moves) + f". Tone: {j['tone']}. "
        f"Probability that your reply should laugh {pct(j['p_laugh'])}, ask a question {pct(j['p_question'])}, use emoji "
        f"{pct(j['p_emoji'])}, joke {pct(j['p_joke'])}, show empathy {pct(j['p_empathy'])}, show excitement "
        f"{pct(j['p_excitement'])}, share your own experience {pct(j['p_share_own'])}; a gushing reply would be over the top "
        f"with probability {pct(j['p_overkill'])}. Expected length level {j['length']:.1f} on a 0-4 scale (0 = 1-3 words, "
        f"4 = over 30 words); expected number of messages {1 + j['bubbles']:.1f}. Your usual style: "
        f"{int(100 * fp['lower_frac'])}% of your messages start lowercase, {int(100 * fp['period_frac'])}% end with a period, "
        f"median {fp['median_words_per_bubble']} words per message" +
        (f", you laugh with '{fp['laugh_token']}'" if fp.get("laugh_token") else "") + "."
    )


def main(split):
    pts = load_points()
    todo = [p for p in pts if p["split"] == split]
    res = ask_many_timed([(state_of(p), Q) for p in todo], workers=4)
    lat = {}
    for p, (a, dt) in zip(todo, res):
        if a is None:
            continue
        p["jev"] = compact(a)
        lat[p["id"]] = dt
    save_points(pts)
    merge_json(os.path.join(ADATA, "a9_latency.json"), {"jev_brief": lat})
    print(split, "ok", sum(1 for p in todo if "jev" in p), "/", len(todo), jev.summary())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
