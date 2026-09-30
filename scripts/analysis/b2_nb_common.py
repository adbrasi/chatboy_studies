"""b2 Parte B: amostra, alvos, states e guias para as arquiteturas de "quantas bolhas" e ritmo.

Alvo y = nº de bolhas da RAJADA INICIAL do turno (bolhas até o 1º intervalo > 120 s; a bolha extra "ah, e…" é
um alvo separado). Classes 1,2,3,4,5+.
Amostra: 200 turnos por corpus × split (dev/test por conversa), com >= 2 turnos de contexto na sessão.
"""
import json, os, re
import numpy as np, pandas as pd
from b2_common import load, OUT, SCR

K = 5  # classes 1..5+
LAB = ["1", "2", "3", "4", "5+"]


def initial_burst(bub_t):
    n = 1
    for a, b in zip(bub_t, bub_t[1:]):
        if (b - a) > 120:
            break
        n += 1
    return n


def lat_words(sec, corpus):
    if sec is None or (isinstance(sec, float) and np.isnan(sec)):
        return None
    if corpus == "maichat":
        for lim, n in ((5, "right away"), (15, "after a few seconds"), (30, "after about 20 seconds"), (60, "after half a minute"), (1e9, "after more than a minute")):
            if sec < lim:
                return n
    for lim, n in ((60, "within the same minute"), (120, "about a minute later"), (900, "a few minutes later"), (3600, "within the hour"), (6 * 3600, "hours later"), (1e12, "much later")):
        if sec < lim:
            return n


def build_sample(n_per=200, seed=3):
    fn = os.path.join(SCR, "b2_nb_sample.pkl")
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    T, _ = load()
    T = T.sort_values(["conv_id", "turn_idx"]).reset_index(drop=True)
    T["pos"] = T.groupby("conv_id").cumcount()
    T["y"] = [min(initial_burst(b), K) for b in T.bub_t]
    T["y_texts"] = [list(tx[:initial_burst(b)]) for tx, b in zip(T.texts, T.bub_t)]
    T["afterthought"] = [initial_burst(b) < len(b) for b in T.bub_t]
    pool = T[(T.turn_in_session >= 2)]
    parts = []
    for (c, s), g in pool.groupby(["corpus", "split"]):
        parts.append(g.sample(n_per, random_state=seed))
    S = pd.concat(parts)
    S.to_pickle(fn)
    T.to_pickle(os.path.join(SCR, "b2_nb_T.pkl"))
    return S


def full_T():
    return pd.read_pickle(os.path.join(SCR, "b2_nb_T.pkl"))


def history(conv, i, spk):
    past = conv.iloc[:i]
    past = past[past.speaker == spk]
    if len(past) == 0:
        return None, "no previous turns from this person yet"
    ys = past.y.clip(upper=5)
    r1 = (ys == 1).mean(); r2 = (ys == 2).mean(); r3 = (ys >= 3).mean()
    style = ("almost never splits" if r1 > .9 else "rarely splits" if r1 > .75 else "sometimes splits" if r1 > .55 else
             "often splits" if r1 > .35 else "very often splits")
    txt = (f"{spk} {style} turns into several messages: in this chat so far {len(past)} turns, "
           f"{r1:.0%} as 1 message, {r2:.0%} as 2, {r3:.0%} as 3 or more.")
    return {"n": len(past), "r1": r1, "rmulti": 1 - r1, "mean": float(ys.mean()), "last": int(ys.iloc[-1]),
            "own_chars": float(past.total_chars.mean())}, txt


def fmt(t, bubbles=True):
    d = {"speaker": t.speaker}
    if bubbles:
        d["messages"] = [str(x).strip()[:300] for x in t.texts]
    else:
        d["text"] = " ".join(str(x).strip() for x in t.texts)[:600]
    lw = lat_words(t.response_latency_s, t.corpus)
    if lw:
        d["replied"] = lw
    return d


def context(conv, i, n=8):
    ctx = conv.iloc[max(0, i - n):i]
    t = conv.iloc[i]
    ctx = ctx[ctx.session == t.session]
    return [fmt(x) for x in ctx.itertuples()], ctx


def len_bucket(c):
    for lim, n in ((20, "very short (up to 20 characters)"), (40, "short (21-40 characters)"), (80, "medium (41-80 characters)"),
                   (160, "long (81-160 characters)"), (1e9, "very long (over 160 characters)")):
        if c <= lim:
            return n


def n_sentences(txt):
    return max(1, len(re.findall(r"[.!?]+(\s|$)|\n", txt)))


# ---- guides (rates measured on the DEV split, initial-burst target; not tuned on test) ----
GUIDE_T = {
    "what_this_is": "How people split one chat turn into separate messages (bubbles) sent in a row.",
    "overall": "About 59% of turns are sent as 1 message, 26% as 2, 10% as 3, 3% as 4 and 2-3% as 5 or more.",
    "by_length": {"up to 20 characters": "88% one message, 11% two",
                  "21-40 characters": "64% one message, 30% two, 6% three",
                  "41-80 characters": "39% one, 40% two, 16% three, 5% four or more",
                  "81-160 characters": "24% one, 33% two, 24% three, 19% four or more",
                  "over 160 characters": "17% one, 20% two, 14% three, 48% four or more"},
    "more_messages_when": ["the speaker is excited or agitated", "joking around: banter is chopped into short pieces",
                           "telling a story or thinking out loud, adding thoughts as they come",
                           "the turn starts with a quick reaction (haha, wait, omg, oh, true) before the content",
                           "a question at the end is sent as its own message", "several separate points"],
    "fewer_messages_when": ["short reply or a single idea", "a tense or annoyed reply: sent as one curt message",
                            "a serious emotional message: longer messages, not chopped into pieces"],
    "personal_style": "Each person has a habit (some split 7% of turns, others 60%). Follow `speaker_history`.",
}
GUIDE_P = {k: v for k, v in GUIDE_T.items() if k != "by_length"} | {
    "length_matters_most": "The more someone has to say, the more messages: a short reaction is almost always 1 message; a long answer or story is usually 2-4."}
GUIDE_TIMING = {
    "what_this_is": "How fast people reply in chat.",
    "pace_mirroring": "People reply at the pace they are being replied to: if the other replies fast, they reply fast.",
    "slower_when": ["answering a practical question that needs checking or thinking", "the conversation is winding down",
                    "a very serious or vulnerable message was received (a bit slower, then a longer reply)"],
    "not_slower_when": ["the speaker is anxious or excited (latency does not change)", "the other's message was long (no reading delay)"],
}
CRIT = {"1": "one single message: the whole reply goes in one bubble",
        "2": "two messages in a row, e.g. a quick reaction then the content, or the content then an add-on or question",
        "3": "three messages in a row", "4": "four messages in a row",
        "5+": "five or more quick messages in a row (a rapid burst)"}
CRIT_BARE = {"1": "1 message", "2": "2 messages", "3": "3 messages", "4": "4 messages", "5+": "5 or more messages"}
MOMENTS = {"banter": "joking, teasing, playing around",
           "story_news": "telling a story, news or something that happened",
           "serious_emotional": "a serious, emotional or vulnerable moment",
           "conflict_tension": "tension, annoyance, arguing or defending oneself",
           "logistics_info": "practical matters or answering with information",
           "quick_reaction": "a short reaction or acknowledgment",
           "affection_flirt": "affection or flirting",
           "casual_chat": "ordinary casual conversation"}


def words_of(p, kind="noul"):
    if kind == "noul":
        return "no" if p < .25 else "a little" if p < .5 else "yes" if p < .8 else "clearly yes"
    return p


def dist_from_choice(ans, key):
    pr = ans[key]["probabilities"]
    v = np.array([pr.get(l, 0.0) for l in LAB], float)
    return v / v.sum() if v.sum() > 0 else np.full(K, 1 / K)


def poisson_binomial(ps, K=K):
    d = np.zeros(len(ps) + 1); d[0] = 1
    for p in ps:
        d[1:] = d[1:] * (1 - p) + d[:-1] * p
        d[0] *= (1 - p)
    out = np.zeros(K)
    for k, v in enumerate(d):
        out[min(k, K - 1)] += v
    return out
