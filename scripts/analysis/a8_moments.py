"""a8 step 1: 'léxico de momento' — what humans actually write in each type of moment.

Moments are defined with the Jev base D-pass (current turn + previous turn of the partner) for maichat/whatsapp_nl,
NPS dialogue acts, and EmpatheticDialogues gold emotion (listener's first reply).
Output: analysis/data/a8_moments.json
"""
import re, statistics as st
from collections import Counter, defaultdict
import a8_common as C

HOWRU = re.compile(r"\b(how are (you|u|ya)|how r (u|you)|how('?s| is) (it going|your day|life|everything|ur day)|how was (your|ur) (day|weekend|night)|hbu|wbu|how you doing|how are things|you good\??$|u good|how ?ya doin|hoe gaat( het)?( met je)?|hoe is het|alles goed)\b", re.I)
POS_EMO = {"joy_excitement", "gratitude", "affection", "surprise"}
NEG_EMO = {"sadness", "anxiety_insecurity", "frustration_anger"}


def D(r, k):
    return r["D"][k]
def ch(r, k):
    return r["D"][k]["choice"]
def sc(r, k):
    return r["D"][k]["score"]
def nl(r, k):
    return r["D"][k]["noul"]


def moments_of(prev, cur):
    ms = []
    it = ch(cur, "intent")
    pt = " ".join(prev["texts"]) if prev else ""
    if prev:
        pint, pemo = ch(prev, "intent"), ch(prev, "emotion")
        if pint in ("share_story_or_info", "share_feeling", "answer") and sc(prev, "valence") >= 3.0 and pemo in POS_EMO:
            ms.append("reagir_noticia_boa")
        if (pemo in ("amusement", "playful_teasing") or pint == "joke_tease") and nl(prev, "playful") >= 0.6:
            ms.append("reagir_algo_engracado")
        if sc(prev, "valence") <= 1.25 and pemo in NEG_EMO and pint in ("share_feeling", "share_story_or_info", "disagree_complain", "answer"):
            ms.append("reagir_desabafo_negativo")
        if HOWRU.search(pt):
            ms.append("responder_como_vai")
        if nl(prev, "flirting") >= 0.5 or pint == "compliment_affection":
            ms.append("responder_flerte_afeto")
        if pint == "ask_question":
            ms.append("responder_pergunta")
        if pint == "make_plans_logistics":
            ms.append("responder_plano")
        if pint == "disagree_complain" and nl(prev, "tension") >= 0.5:
            ms.append("responder_provocacao_ou_bronca")
    ms.append("intent:" + it)
    return ms


def stats_for(turns):
    """turns: list of lists of bubble texts."""
    n = len(turns)
    if not n:
        return None
    first = [t[0] for t in turns]
    allb = [b for t in turns for b in t]
    tc = [sum(len(b) for b in t) for t in turns]
    forms = Counter(C.canon(b) for b in allb if len(b.split()) <= 5 and b.strip())
    fw = Counter(C.first_word(b) for b in first)
    LAUGH = re.compile(r"\b(a?ha(ha)+h?|he(he)+|lo+l+|lmf?ao+|haha\w*|hah+|hihi\w*)\b|[\U0001F602\U0001F923\U0001F62D\U0001F480]", re.I)
    return {
        "n": n,
        "median_chars_turn": st.median(tc),
        "p25_p75_chars": [sorted(tc)[n // 4], sorted(tc)[3 * n // 4]],
        "mean_bubbles": round(len(allb) / n, 2),
        "pct_first_bubble_le3words": round(100 * sum(len(b.split()) <= 3 for b in first) / n, 1),
        "pct_laugh": round(100 * sum(bool(LAUGH.search(" ".join(t))) for t in turns) / n, 1),
        "pct_question": round(100 * sum("?" in " ".join(t) for t in turns) / n, 1),
        "pct_excl": round(100 * sum("!" in " ".join(t) for t in turns) / n, 1),
        "pct_emoji": round(100 * sum(bool(re.search("[\U0001F300-\U0001FAFF☀-➿]", " ".join(t))) for t in turns) / n, 1),
        "pct_no_final_punct": round(100 * sum(not re.search(r"[.!?]\s*$", t[-1]) for t in turns) / n, 1),
        "pct_starts_lower": round(100 * sum(first_lower(t[0]) for t in turns) / n, 1),
        "top_forms": [[f, c] for f, c in forms.most_common(14) if c >= 2][:12],
        "top_first_words": fw.most_common(12),
        "examples": [" / ".join(t)[:120] for t in turns[:: max(1, n // 8)][:8]],
    }


def first_lower(b):
    s = b.lstrip()
    return bool(s) and s[0].isalpha() and s[0].islower()


def main():
    J = C.jev_base()
    out = {}
    for corp in ("maichat", "whatsapp_nl"):
        by = defaultdict(list)
        R = [r for r in J if r["corpus"] == corp]
        idx = {(r["conv_id"], r["turn_idx"]): r for r in R}
        for r in R:
            prev = idx.get((r["conv_id"], r["turn_idx"] - 1))
            if prev and (prev["session"] != r["session"] or prev["speaker"] == r["speaker"]):
                prev = None
            for m in moments_of(prev, r):
                by[m].append(r["texts"])
        out[corp] = {m: stats_for(v) for m, v in sorted(by.items())}
    # NPS dialogue acts
    M = C.messages()
    by = defaultdict(list)
    for m in M["nps_chatroom"]:
        if m.get("dialogue_act") and m["dialogue_act"] != "System":
            by[m["dialogue_act"]].append([m["text"]])
    out["nps_chatroom"] = {k: stats_for(v) for k, v in by.items()}
    # Empathetic: listener's first reply by gold emotion polarity
    POS = {"excited", "proud", "joyful", "grateful", "impressed", "hopeful", "content", "confident", "anticipating", "caring", "trusting", "faithful", "prepared"}
    NEG = {"sad", "lonely", "devastated", "disappointed", "afraid", "anxious", "terrified", "apprehensive", "angry", "annoyed", "furious", "jealous", "embarrassed", "ashamed", "guilty", "disgusted"}
    conv = defaultdict(list)
    for m in M["empathetic"]:
        conv[m["conv_id"]].append(m)
    by = defaultdict(list)
    for cid, ms in conv.items():
        ms.sort(key=lambda m: m["idx"])
        if len(ms) >= 2 and ms[1]["speaker"] != ms[0]["speaker"]:
            g = ms[0]["gold_emotion"]
            k = "B_reply_positive_news" if g in POS else "B_reply_negative_story" if g in NEG else "B_reply_other"
            by[k].append([ms[1]["text"]])
    out["empathetic"] = {k: stats_for(v) for k, v in by.items()}
    # also the first 3 words in empathetic listener replies (formula starts)
    for k, v in by.items():
        c3 = Counter(" ".join(C.canon(t[0]).split()[:3]) for t in v)
        out["empathetic"][k]["top_first3"] = c3.most_common(15)
    # NUS SMS: overall short repertoire
    sms = [m["text"] for m in M["nus_sms"]]
    short = Counter(C.canon(t) for t in sms if len(t.split()) <= 3)
    out["nus_sms"] = {"n": len(sms), "top_short_messages": short.most_common(40),
                      "top_first_words": Counter(C.first_word(t) for t in sms).most_common(30)}
    C.save("a8_moments.json", out)
    # print compact
    for corp in ("maichat", "whatsapp_nl", "nps_chatroom", "empathetic"):
        print("#####", corp)
        for k, s in out[corp].items():
            if s and s["n"] >= 10:
                print(f"{k:34s} n={s['n']:4d} med={s['median_chars_turn']:5} bub={s['mean_bubbles']} short={s['pct_first_bubble_le3words']} laugh={s['pct_laugh']} q={s['pct_question']} !={s['pct_excl']} emo={s['pct_emoji']} nopunct={s['pct_no_final_punct']} lower={s['pct_starts_lower']}")
                print("     forms:", s["top_forms"][:10])
                print("     fw:", s["top_first_words"][:10])
                if "top_first3" in s:
                    print("     f3:", s["top_first3"][:12])
    print("SMS", out["nus_sms"]["top_short_messages"][:40])


if __name__ == "__main__":
    main()
