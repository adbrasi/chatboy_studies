"""a8 step 9: formula lexicon per moment family — how often each candidate formula appears
(per 1,000 messages) in real chat corpora vs LLM replies (per 1,000 replies). Also length-budget table by Jev P.p_length.
Output: analysis/data/a8_lexicon.json"""
import json, re, statistics as st
from collections import defaultdict
import numpy as np
import a8_common as C
from a8_compare import load_all

FAM = {
 "notícia boa": {"congrats/congratulations": r"\bcongrat\w*|\bcongratz\b", "omg": r"\bomg+\b", "yay/yayy": r"\bya+y+\b", "woo/woohoo": r"\bwo+(?:ho+)?\b",
                  "nice/niceee": r"\bnic+e+\b", "wow": r"\bwo+w+\b", "no way": r"\bno wa+y+\b", "finally": r"\bfinally\b", "yess/yesss": r"\byes{2,}\b",
                  "let's go/lesgo": r"\blet'?s go+\b|\blesgo+\b", "amazing": r"\bamazing\b", "awesome": r"\bawesome\b", "great": r"\bgreat\b",
                  "cool": r"\bcoo+l\b", "proud of you": r"\bproud of (?:you|u)\b", "so happy for you": r"\bhappy for (?:you|u)\b"},
 "notícia ruim/desabafo": {"oh no/noo": r"\boh no+\b|^\s*no{2,}\b", "aw/aww": r"\baw+\b", "ugh": r"\bugh+\b", "that sucks/sucks": r"\bsucks?\b",
                  "sorry": r"\bsorry\b|\bsry\b", "damn": r"\bdamn\b", "rip": r"\brip\b", "wtf": r"\bwtf\b", "what happened": r"\bwhat happened\b",
                  "are you ok/u ok": r"\b(?:are )?(?:you|u) ok(?:ay)?\b", "hugs/*hug*": r"\bhugs?\b", "sorry to hear": r"\bsorry to hear\b",
                  "that must be": r"\bthat must be\b", "i'm here for you": r"\bhere for (?:you|u)\b", "it's okay to feel": r"\bokay to feel\b"},
 "algo engraçado": {"lol": r"\blo+l+\b", "haha": r"\bhaha\b", "hahaha+": r"\bha(?:ha){2,}h?\b", "lmao": r"\blmf?ao+\b", "hehe": r"\bhe(?:he)+\b",
                  "😂": "\U0001F602", "💀": "\U0001F480", "😭": "\U0001F62D", "dead/i'm dead": r"\bdead\b", "i'm crying/crying": r"\bcrying\b",
                  "stop/stopp": r"^\s*sto+p+\b", "rude": r"\brude\b", "that's hilarious": r"\bhilarious\b", "you're so funny": r"\bso funny\b"},
 "concordar/aceitar": {"yeah/yea/ya": r"\b(?:yeah|yea|ya)\b", "yes/yess": r"\byes+\b", "ok/okay/k": r"^\s*(?:ok+|okay|k)\b", "true": r"\btrue\b",
                  "exactly": r"\bexactly\b", "same": r"\bsame\b", "fair": r"\bfair\b", "sure": r"\bsure\b", "yep/yup": r"\by[eu]p\b",
                  "right": r"\bright\b", "for real/fr": r"\bfor real\b|\bfr\b", "agreed/i agree": r"\bagree", "absolutely": r"\babsolutely\b",
                  "totally": r"\btotally\b", "definitely": r"\bdefinitely\b", "sounds good": r"\bsounds good\b"},
 "discordar/recusar": {"no/noo": r"^\s*no+\b", "nah": r"\bnah\b", "nope": r"\bnope\b", "not really": r"\bnot really\b", "idk": r"\bidk\b",
                  "hmm": r"\bh?m{2,}\b", "i mean": r"\bi mean\b", "wdym": r"\bwdym\b", "excuse me": r"\bexcuse me\b", "but": r"^\s*but\b",
                  "i disagree": r"\bdisagree\b", "i see your point": r"\bi see (?:your|ur) point\b"},
 "despedir": {"bye/byee": r"\bbye+\b", "gn/good night/night": r"\bgn\b|\bgood ?night\b|\bnight\b|\bnite\b", "ttyl/talk later": r"\bttyl\b|\btalk (?:to (?:you|u) )?later\b",
              "see ya/cya": r"\bsee (?:ya|you|u)\b|\bcya\b", "gtg/gotta go": r"\bgtg\b|\bgotta go\b", "brb": r"\bbrb\b", "take care": r"\btake care\b",
              "xx/xoxo": r"(?:^|\s)x{2,}(?:o\w*)?\b", "love you/ly": r"\blove (?:you|u)\b|\bily\b", "sleep well": r"\bsleep well\b"},
 "como vai": {"good/gud": r"^\s*(?:good|gud|gd)\b", "fine": r"\bfine\b", "ok/okay": r"^\s*(?:ok|okay)\b", "tired": r"\btired\b", "meh": r"\bmeh\b",
              "not bad": r"\bnot bad\b", "busy": r"\bbusy\b", "u?/you?/wbu/hbu": r"\b(?:wbu|hbu|and (?:you|u)\??)|(?:^|\s)(?:you|u)\?\s*$",
              "doing well": r"\bdoing (?:well|great|good)\b", "thanks for asking": r"\bthanks for asking\b"},
}


def rate(texts, rx):
    r = re.compile(rx, re.I)
    return round(1000 * sum(bool(r.search(t)) for t in texts) / max(1, len(texts)), 1)


def main():
    M = C.messages()
    srcs = {c: [m["text"].replace("_comma_", ",") for m in M[c] if m.get("dialogue_act") != "System" and m["text"].strip()]
            for c in ("maichat", "nps_chatroom", "nus_sms", "empathetic")}
    X, R = load_all()
    ids = [c["id"] for c in X]
    srcs["LLM base (3 modelos)"] = [R[cn][i] for cn in ("base_gemini", "base_gpt4omini", "base_llama70b") for i in ids]
    srcs["LLM styled"] = [R["styled_gemini"][i] for i in ids]
    srcs["humano mesmos pontos"] = [R["human"][i] for i in ids]
    out = {}
    for fam, d in FAM.items():
        out[fam] = {k: {s: rate(T, rx) for s, T in srcs.items()} for k, rx in d.items()}
    # length budget: human total chars (maichat turns) by Jev P.p_length bucket and by D.intent/moment
    J = [r for r in C.jev_base() if r["corpus"] == "maichat" and r.get("P")]
    by = defaultdict(list)
    for r in J:
        by[int(round(r["P"]["p_length"]["score"]))].append(r["total_chars"])
    out["_length_by_p_length"] = {k: {"n": len(v), "median": st.median(v), "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))} for k, v in sorted(by.items())}
    C.save("a8_lexicon.json", out)
    cols = list(srcs)
    for fam in FAM:
        print("=====", fam)
        print(f"{'':28s}" + "".join(f"{c[:11]:>12s}" for c in cols))
        for k, v in sorted(out[fam].items(), key=lambda kv: -(kv[1]["maichat"] + kv[1]["nps_chatroom"] + kv[1]["nus_sms"])):
            print(f"{k:28s}" + "".join(f"{v[c]:>12}" for c in cols))
    print(out["_length_by_p_length"])


if __name__ == "__main__":
    main()
