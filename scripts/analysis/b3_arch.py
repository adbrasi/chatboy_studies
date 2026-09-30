"""b3 — arquiteturas de chamadas do Jev para PREVER o movimento da resposta humana (posição do bot: a resposta não
aparece no state). Cada etapa grava data/processed/b3_pred_<etapa>.json ({id: respostas compactas}).

Etapas (python3 b3_arch.py <etapa> [dev|test|all]):
  base   1 chamada, state padrão (8 turnos), fan-out de ~50 perguntas independentes:
         flat (Choice de 16 = 1ª rodada) · flat_rich (Choice com critérios ricos) · fam (Choice de 7 famílias)
         · fine_<fam> (Choice dentro de cada família) · n_<mov> (16 Nouls "um amigo responderia assim?")
         · leitura do momento (Nouls/Choices, entra no state da cascata) · subtexto · tom
  elem   1 chamada: state + elementos numerados da última msg → Choice "a que elemento reagir"
  guide  1 chamada: state + GUIA de movimentos (quando humanos fazem cada um, com taxas do dev) → Choice
  casc   2ª chamada da cascata: state + leitura do momento (saída da base, em rótulos) + guia → Choice
  cascp  idem + prior empírico por gatilho ("quando o outro faz X, pessoas responderam: …", LOCO)
  char   state + estado do personagem e da relação → Choice
  rag5/rag10/rag20  state + k situações parecidas de OUTRAS conversas (resposta real + movimento) → Choice
  ragtxt state + 10 situações parecidas SEM rótulo de movimento (só as respostas reais) → Choice
  cand   LLM (flash-lite) escreve 1 candidata por família; Jev avalia cada uma com 5 Nouls atômicos → código escolhe
  luna   referência sem Jev: openai/gpt-6-luna prevê o movimento (top-3)
"""
import json, os, random, re, sys
from collections import Counter, defaultdict
import numpy as np
from b3_common import (load_points, base_state, last_msg, jask_many, compact, noul, choice, score, kv_load, kv_save,
                       MOVE_OPTS, TONE_OPTS, MOVES, FAMILY, FAMILIES, FAMILY_OPTS, BOT, USER, lchat_many, lstats,
                       ADATA)
import jev

# ------------------------------------------------------------------ perguntas
MOVE_Q = f"What is the most natural next move for {BOT}?"  # igual à 1ª rodada
RICH = {
    "react_only": "just a short reaction and nothing else ('lol', 'nice', 'oh no', 'fair', 'true', 'same', 'haha'); "
                  "the most common reply among friends, typical after statements, jokes and minimal messages",
    "answer": f"answer the question {USER} just asked, plainly; only if {USER} asked something",
    "tease_back": f"tease {USER} back or make fun of what they said (dry, short: 'rude', 'define productive', 'allegedly')",
    "joke_riff": "continue the joke with another joke or a silly line",
    "empathize": "show understanding or sympathy in simple words ('ugh', 'that sucks'); rare as the whole reply",
    "reassure": f"reassure or encourage {USER} ('you'll be fine', 'you got this')",
    "share_own": f"share {BOT}'s own related experience, news, opinion or state ('same i ...', 'i just ...')",
    "ask_follow_up": f"ask {USER} a specific follow-up question about what they said ('what happened?', 'which one')",
    "flirt_back": "flirt back or return affection ('love u too', 'miss u more')",
    "compliment": f"compliment {USER}",
    "agree": "agree with what was said ('exactly', 'yeah true', 'fr')",
    "disagree": "disagree or push back ('no', 'not true', 'excuse me??')",
    "plan": "propose, accept or confirm a concrete plan (time, place, what to do)",
    "goodbye": "say goodbye, only when the chat is ending",
    "greet_back": "greet back, only at the start of the chat",
    "new_topic": "bring up something new that is unrelated to the last message",
}
NOUL_TXT = {
    "react_only": "just react briefly (like 'lol', 'fair', 'oh no', 'nice') and add nothing else",
    "answer": f"answering a question {USER} asked",
    "tease_back": f"teasing {USER} back",
    "joke_riff": "adding another joke to the joke",
    "empathize": "showing understanding or sympathy",
    "reassure": f"reassuring or encouraging {USER}",
    "share_own": "sharing their own related experience or opinion",
    "ask_follow_up": f"asking a follow-up question about what {USER} said",
    "flirt_back": "flirting back or returning affection",
    "compliment": f"complimenting {USER}",
    "agree": "simply agreeing",
    "disagree": "disagreeing or pushing back",
    "plan": "proposing or confirming a plan",
    "goodbye": "saying goodbye",
    "greet_back": "greeting back",
    "new_topic": "bringing up a new, unrelated topic",
}
FINE = {f: {m: MOVE_OPTS[m] for m in MOVES if FAMILY[m] == f} for f in FAMILIES}

READ = {  # leitura do momento (estágio 1 da cascata)
    "teasing": noul(f"Is {USER} teasing, joking or being playful in `last_message`?"),
    "flirting": noul(f"Is {USER} flirting with {BOT} or being romantic in `last_message`?"),
    "affection": noul(f"Does `last_message` express affection, care, gratitude or a compliment toward {BOT}?"),
    "vulnerable": noul(f"Is {USER} sharing a worry, a problem, sadness or something personal and vulnerable in `last_message`?"),
    "good_news": noul(f"Is {USER} sharing good news or excitement in `last_message`?"),
    "complaint": noul(f"Is {USER} complaining, ranting or venting about something in `last_message`?"),
    "direct_q": noul(f"Does `last_message` ask {BOT} a direct question that expects an answer?"),
    "closing": noul(f"Is {USER} wrapping up or saying goodbye in `last_message`?"),
    "greeting": noul("Is `last_message` a greeting that opens the conversation?"),
    "logistics": noul("Is `last_message` about practical logistics such as plans, times, places or tasks?"),
    "minimal": noul("Is `last_message` a minimal reply (like 'ok', 'yeah', 'lol', 'true') that adds nothing new?"),
    "annoyed": noul(f"Is {USER} annoyed or upset with {BOT} in `last_message`?"),
    "story": noul(f"Is {USER} telling {BOT} about something that happened to them in `last_message`?"),
    "opinion": noul(f"Does `last_message` state an opinion or a claim that {BOT} could agree or disagree with?"),
    "topic_done": noul("Does the current topic feel exhausted, with nothing left to say about it?"),
    "wants": choice(f"What does {USER} most want from {BOT}'s next message?", {
        "laughter": "to have fun, laugh, keep the banter going", "comfort": "comfort, sympathy or reassurance",
        "answer": "an answer or information", "attention": "interest and attention to what they shared",
        "validation": "agreement or validation of their opinion", "plan": "a decision or confirmation about a plan",
        "acknowledgment": "just a quick acknowledgment", "closure": "to end the conversation"}),
    "seriousness": score("How serious is the current moment of the conversation?",
                         ["playful banter", "casual", "somewhat serious", "very serious or emotional"]),
    "energy": score(f"How energetic or excited is {USER} in `last_message`?", ["flat, low energy", "calm", "lively", "very excited"]),
}
SUBTEXT = {
    "sub_implicit": noul(f"Should {BOT}'s reply leave something implicit (hint or understate it) rather than say it outright?"),
    "sub_how": choice(f"How should {BOT} handle the main point of the reply?", {
        "explicit": "say it plainly and directly", "hint": "hint at it or understate it, leaving it implied",
        "skip": "say nothing about it, just react or move on"}),
}
TONE = {"tone": choice(f"Which tone fits {BOT}'s next message best?", TONE_OPTS)}


def base_questions():
    q = {"flat": choice(MOVE_Q, MOVE_OPTS),
         "flat_rich": choice(f"What would {BOT} most likely do in the next message, as a close friend texting?", RICH),
         "fam": choice(f"What kind of move is {BOT}'s next message most likely to be?", FAMILY_OPTS)}
    for f, opts in FINE.items():
        if len(opts) > 1:
            q[f"fine_{f}"] = choice(f"If {BOT}'s next message is of this kind, which specific move is it?", opts)
    for m in MOVES:
        q[f"n_{m}"] = noul(f"Would a close friend reply to `last_message` by {NOUL_TXT[m]}?")
    q.update(READ); q.update(SUBTEXT); q.update(TONE)
    return q


# ------------------------------------------------------------------ guia de movimentos (taxas do DEV)
def move_guide(dev_pts):
    """Guia: quando humanos fazem cada movimento + taxa medida nas conversas de dev (rótulo-ouro do Jev)."""
    c = Counter(p["gold"]["g_move"] for p in dev_pts if "gold" in p)
    n = sum(c.values())
    when = {
        "react_only": "after statements, jokes, news told casually and minimal messages; friends often just react",
        "answer": f"when {USER} asked a direct question; the answer is plain and usually does not ask back",
        "tease_back": "in banter; people answer a joke with another dry jab rather than laughing",
        "joke_riff": "in banter, to extend the joke",
        "empathize": "after bad news or a complaint, but friends rarely console explicitly; a short 'oh no' is more common",
        "reassure": f"when {USER} is worried about something concrete",
        "share_own": f"when {USER} tells something and {BOT} has a related experience or opinion ('same, i...')",
        "ask_follow_up": "after news or a problem ('what happened?'); rare after jokes; people ask in about 1 of 8 turns",
        "flirt_back": "after affection; but the most common reply to affection is to continue the topic or say 'hehe'",
        "compliment": "rare",
        "agree": f"when {USER} states an opinion or claim",
        "disagree": "playful pushback in banter ('rude', 'no'), sometimes real disagreement",
        "plan": "when making plans or logistics",
        "goodbye": f"only when {USER} is clearly ending the chat",
        "greet_back": "only at the start of the chat",
        "new_topic": "after a minimal reply or when the topic is exhausted",
    }
    return {m: f"{when[m]} (about {round(100 * c[m] / n)}% of replies)" for m in MOVES}


def reading_labels(b):
    """Estágio 1 → rótulos legíveis (código) para o state do estágio 2."""
    names = {"teasing": f"{USER} is teasing or joking", "flirting": f"{USER} is flirting",
             "affection": f"{USER} is being affectionate or grateful", "vulnerable": f"{USER} shares something vulnerable",
             "good_news": f"{USER} shares good news", "complaint": f"{USER} is venting", "direct_q": f"{USER} asked {BOT} a direct question",
             "closing": f"{USER} is wrapping up", "greeting": f"{USER} is greeting", "logistics": "it is about plans/logistics",
             "minimal": f"{USER}'s message is minimal", "annoyed": f"{USER} is annoyed with {BOT}",
             "story": f"{USER} is telling about something that happened", "opinion": f"{USER} states an opinion",
             "topic_done": "the topic seems exhausted"}
    on = [names[k] for k in names if b.get(k, 0) >= 0.6]
    maybe = [names[k] for k in names if 0.4 <= b.get(k, 0) < 0.6]
    ser = ["playful banter", "casual", "somewhat serious", "very serious"][int(round(b["seriousness"]))]
    en = ["flat", "calm", "lively", "very excited"][int(round(b["energy"]))]
    return {"what_is_happening": on or ["nothing special, casual chat"], "maybe": maybe,
            f"what_{USER}_wants": b["wants"], "seriousness": ser, f"{USER}_energy": en}


# ------------------------------------------------------------------ retrieval (TF-IDF + rótulos D do parceiro)
class Retriever:
    def __init__(self, pts, w_cat=0.15):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.pts = [p for p in pts if "gold" in p]
        q = [self.qtext(p) for p in self.pts]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, token_pattern=r"(?u)\b\w+\b|[?!]")
        self.X = self.vec.fit_transform(q)
        self.cat = [self.cats(p) for p in self.pts]
        self.conv = np.array([p["conv_id"] for p in self.pts])
        self.w_cat = w_cat

    @staticmethod
    def qtext(p):
        h = p["history"]
        prev = h[-2]["text"] if len(h) >= 2 else ""
        return (last_msg(p) + " ") * 3 + prev

    @staticmethod
    def cats(p):
        d = p["prev_D"]
        return (d.get("intent"), d.get("emotion"), d.get("phase"))

    def query(self, p, k=10):
        v = self.vec.transform([self.qtext(p)])
        sim = (self.X @ v.T).toarray().ravel()
        c = self.cats(p)
        sim = sim + self.w_cat * np.array([sum(a == b for a, b in zip(c, cc)) for cc in self.cat]) / 3
        sim[self.conv == p["conv_id"]] = -1  # nunca a própria conversa
        top = np.argsort(-sim)[:k]
        return [(self.pts[i], float(sim[i])) for i in top]


def case_view(q, with_move=True, with_reply=True):
    h = q["history"]
    d = {"context": [f"{x['who']}: {x['text']}" for x in h[-3:]]}
    if with_reply:
        d[f"what_{BOT}_replied"] = q["human"]
    if with_move:
        d["move"] = q["gold"]["g_move"]
    return d


# ------------------------------------------------------------------ etapas
def select(split):
    pts = load_points()
    ev = [p for p in pts if p["eval"] and (split == "all" or p["split"] == split)]
    return pts, ev


def run_calls(stage, ev, build):
    P = kv_load(f"pred_{stage}")
    todo = [p for p in ev if p["id"] not in P]
    items = [build(p) for p in todo]
    for s in range(0, len(todo), 200):
        res = jask_many(items[s:s + 200], workers=4)
        for p, a in zip(todo[s:s + 200], res):
            if a:
                P[p["id"]] = compact(a)
        kv_save(f"pred_{stage}", P)
        print(stage, s + len(res), "/", len(todo), jev.summary(), flush=True)
    return P


def main(stage, split="all"):
    pts, ev = select(split)
    dev_all = [p for p in pts if p["split"] == "dev"]
    if stage == "base":
        Q = base_questions()
        run_calls(stage, ev, lambda p: (base_state(p), Q))
    elif stage == "elem":
        def b(p):
            els = p["elements"]
            if len(els) < 2:
                return (base_state(p), {"skip": noul("Is `last_message` empty?")})
            st = base_state(p)
            st["last_message_elements"] = {f"e{i + 1}": e for i, e in enumerate(els)}
            opts = {f"e{i + 1}": f"the word/phrase '{e}'" for i, e in enumerate(els)}
            opts["whole"] = "the message as a whole or its general topic, no specific word"
            opts["other"] = "something that is not in `last_message` (earlier context or a new thing)"
            return (st, {"elem": choice(f"Which part of `last_message` will {BOT}'s next message most likely react to?", opts),
                         "elem_friend": choice(f"Friends often pick up one concrete word and play with it. Which part of "
                                               f"`last_message` would a close friend pick up in the reply?", opts)})
        run_calls(stage, ev, b)
    elif stage == "guide":
        G = move_guide(dev_all)
        run_calls(stage, ev, lambda p: (dict(base_state(p), move_guide=G), {"move": choice(MOVE_Q, MOVE_OPTS)}))
    elif stage in ("casc", "cascp"):
        B = kv_load("pred_base")
        G = move_guide(dev_all)
        prior = loco_prior(pts) if stage == "cascp" else None

        def b(p):
            st = dict(base_state(p), moment_reading=reading_labels(B[p["id"]]), move_guide=G)
            if prior:
                st["how_people_usually_reply_here"] = prior(p)
            return (st, {"move": choice(MOVE_Q, MOVE_OPTS)})
        run_calls(stage, [p for p in ev if p["id"] in B], b)
    elif stage == "char":
        def b(p):
            st = base_state(p)
            od = p.get("own_D") or {}
            rel = (p.get("rel_so_far") or "close_friends").replace("_", " ")
            inti = {"romantic partners": "very close, romantic", "family": "family, close", "close friends": "close friends",
                    "flirting or crush": "flirting, getting closer", "colleagues or classmates": "friendly classmates",
                    "acquaintances or new": "still getting to know each other"}.get(rel, "close")
            eng = ["barely engaged", "low interest", "moderately engaged", "into the conversation", "very into it"]
            st["character"] = {"name": BOT, "relationship_with_" + USER: rel, "intimacy": inti,
                               "current_mood": (od.get("emotion") or "neutral").replace("_", " "),
                               "engagement": eng[int(round(od.get("engagement", 2) or 2))],
                               "last_thing_" + BOT + "_did": (od.get("intent") or "none").replace("_", " "),
                               "style": "texts like a real friend: short, casual, does not over-explain"}
            return (st, {"move": choice(MOVE_Q, MOVE_OPTS)})
        run_calls(stage, ev, b)
    elif stage.startswith("rag"):
        k = 10 if stage == "ragtxt" else int(stage[3:])
        R = Retriever(pts)

        def b(p):
            cases = [case_view(q, with_move=(stage != "ragtxt")) for q, _ in R.query(p, k)]
            st = dict(base_state(p), similar_situations_from_other_chats=cases)
            return (st, {"move": choice(f"In similar situations people replied as shown in `similar_situations_from_other_chats`. "
                                        f"{MOVE_Q}", MOVE_OPTS)})
        run_calls(stage, ev, b)
    elif stage == "cand":
        run_cand(ev)
    elif stage == "luna":
        run_luna(ev)


def loco_prior(pts):
    """P(movimento | intenção D do turno do parceiro), deixando a própria conversa de fora (estatística de corpus)."""
    tab = defaultdict(Counter)
    for p in pts:
        if "gold" in p:
            tab[(p["conv_id"], p["prev_D"].get("intent"))][p["gold"]["g_move"]] += 1
    tot = defaultdict(Counter)
    for (cv, it), c in tab.items():
        tot[it].update(c)

    def f(p):
        it = p["prev_D"].get("intent")
        c = tot[it].copy()
        c.subtract(tab[(p["conv_id"], it)])
        n = sum(c.values())
        top = [(m, v / n) for m, v in c.most_common(5) if v > 0]
        return {"situation": f"{USER}'s last message is: {str(it).replace('_', ' ')}",
                "most_common_replies": [f"{m} ({round(100 * s)}%)" for m, s in top]}
    return f


# ------------------------------------------------------------------ candidatas geradas + Nouls atômicos
CAND_SYS = (f"You are helping write {BOT}'s next text message in a chat with {USER}. Write 7 alternative replies, each "
            "doing a DIFFERENT kind of move:\n" + "\n".join(f"- {f}: {d}" for f, d in FAMILY_OPTS.items()) +
            f"\nWrite them the way {BOT} texts (casual, short). Return ONLY JSON: " +
            "{" + ", ".join(f'"{f}": "..."' for f in FAMILIES) + "}")


def cand_prompt(p):
    ctx = "\n".join(f"{h['who']}: {h['text']}" for h in p["history"][-8:])
    return [{"role": "system", "content": CAND_SYS}, {"role": "user", "content": f"CHAT:\n{ctx}\n\n{BOT}'s next message?"}]


ATOM = {
    "fits": "Does `candidates.{k}` respond to what {U} actually said in `last_message`?",
    "tone": "Does the tone of `candidates.{k}` match the mood of `last_message`?",
    "intense": "Is `candidates.{k}` more intense or emotional than `last_message`?",
    "friend": "Is `candidates.{k}` what a close friend would typically text back here?",
    "overdo": "Does `candidates.{k}` do more than the moment needs (explains, validates or asks more than needed)?",
}


def run_cand(ev):
    C = kv_load("cand_texts")
    todo = [p for p in ev if p["id"] not in C]
    res = lchat_many([dict(messages=cand_prompt(p), model="flash", temperature=0.7, max_tokens=500, seed=0) for p in todo])
    for p, r in zip(todo, res):
        m = re.search(r"\{.*\}", r or "", re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except Exception:
            d = None
        if d and all(isinstance(d.get(f), str) and d.get(f).strip() for f in FAMILIES):
            C[p["id"]] = {f: d[f].strip() for f in FAMILIES}
    kv_save("cand_texts", C)
    print("cand texts", len(C), lstats)

    def b(p):
        rnd = random.Random(p["id"])
        fams = FAMILIES[:]
        rnd.shuffle(fams)
        keymap = {f"c{i + 1}": f for i, f in enumerate(fams)}
        st = dict(base_state(p), candidates={k: C[p["id"]][f] for k, f in keymap.items()})
        qs = {f"{a}_{k}": noul(t.format(k=k, U=USER)) for k in keymap for a, t in ATOM.items()}
        return (st, qs)
    evc = [p for p in ev if p["id"] in C]
    P = run_calls("cand", evc, b)
    # desembrulha: por família
    out = {}
    for p in evc:
        a = P.get(p["id"])
        if not a:
            continue
        rnd = random.Random(p["id"])
        fams = FAMILIES[:]
        rnd.shuffle(fams)
        keymap = {f"c{i + 1}": f for i, f in enumerate(fams)}
        out[p["id"]] = {f: {at: a[f"{at}_{k}"] for at in ATOM} for k, f in keymap.items()}
    kv_save("pred_cand_unpacked", out)


LUNA_PRED = (f"You predict how a real person ({BOT}) will reply in a casual chat with a friend ({USER}). Moves:\n" +
             "\n".join(f"- {k}: {v}" for k, v in MOVE_OPTS.items()) +
             "\nReturn ONLY JSON: {\"top3\": [<most likely move>, <2nd>, <3rd>]} using the exact keys.")


def run_luna(ev):
    P = kv_load("pred_luna")
    todo = [p for p in ev if p["id"] not in P]

    def pr(p):
        ctx = "\n".join(f"{h['who']}: {h['text']}" for h in p["history"][-8:])
        return [{"role": "system", "content": LUNA_PRED}, {"role": "user", "content": f"CHAT:\n{ctx}\n\nWhat will {BOT} do next?"}]
    res = lchat_many([dict(messages=pr(p), model="luna", temperature=0, max_tokens=800, seed=0,
                           extra={"reasoning": {"effort": "low"}}) for p in todo])
    for p, r in zip(todo, res):
        m = re.search(r"\{.*\}", r or "", re.S)
        try:
            d = json.loads(m.group(0))
            t3 = [x for x in d["top3"] if x in MOVES]
            if t3:
                P[p["id"]] = t3
        except Exception:
            pass
    kv_save("pred_luna", P)
    print("luna pred", len(P), lstats)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "all")
