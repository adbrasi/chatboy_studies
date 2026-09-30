"""a2: session table (whatsapp_nl + maichat): openings, first reply, reopen gap, closings. Rule-based typology."""
import json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import a2_load as L

GREET_W = r"(?:ho+i+|hoihoi|he+y*|he+u+j*|hé+|hei|ha+i+|ha+y+|hallo+|he+llo+|helo+|hellu+|hi+|hiya|heya|ha|yo+|jo+|joe|moi|sup|wassup|goedemorgen|goedemiddag|goedenavond|morgen|morning|good morning|gm|dag|hola|howdy|helluu+)"
GREET = re.compile(rf"^\s*{GREET_W}\b[\s!.,:;)(\-]*", re.I)
NAME = re.compile(r"\[REMOVED\]|\b(?:schat|lieverd|lief|pap|pa|mam|ma|zus|zuslief|broer|bro|dude|girl|babe|hon|love|chef|meis|meid|jongen|jongens|allemaal|all)\b", re.I)
WELL = re.compile(r"alles (?:goed|oke|ok|oké|lekker|kits)|hoe (?:gaat|is|was)(?: het| t|'t| ie|het)? ?(?:met je|met jou|daar|ermee)?\s*\??$|hoe gaat (?:het|t|ie)|gaat (?:het|t) (?:goed|een beetje)|how (?:are|r) (?:you|u|ya)|how'?s (?:it going|your day|ur day|life)|how you doing|wbu|what'?s up|whats good|wassup|hoe voel je", re.I)
PRES = re.compile(r"\b(?:ben je (?:nog |al )?(?:wakker|thuis|er|online|daar|bezig|op|vrij)|zit je (?:nog )?(?:op|thuis)|are (?:you|u) (?:there|up|awake|free|online|alive|on)|(?:you|u) (?:there|up|free|online|awake|alive|on)\??|you on yet|kan ik (?:je )?(?:even |zo )?bellen|mag ik (?:je )?bellen|heb je (?:even |nu )?tijd|(?:are you )?ignoring me|is this you)\b", re.I)
LOGI = re.compile(r"\b(?:hoe laat|om \d|\d ?uur|vanavond|vanmiddag|vannacht|morgen|straks|zo meteen|afspreken|zullen we|kom je|kun je|kan je|kunnen we|ophalen|trein|bus|station|onderweg|ik ben er|ben er over|eten|avondeten|boodschappen|weekend|zaterdag|zondag|maandag|dinsdag|woensdag|donderdag|vrijdag)\b", re.I)
PAST = re.compile(r"\b(?:hoe was|hoe ging|gister(?:en|avond)?|vannacht|afgelopen|trouwens|btw|net (?:thuis|terug|aangekomen|wakker)|ben (?:net|weer) |weer thuis)\b", re.I)
SORRY = re.compile(r"\b(?:sorry|sry|excuus|excuses|vergeten te (?:reageren|antwoorden)|late? (?:reactie|antwoord)|was (?:in )?slaap|in slaap gevallen)\b", re.I)
MEDIA = re.compile(r"weggelaten>|<media omitted>", re.I)
LAUGH0 = re.compile(r"^\s*(?:ha(?:ha)+h*|hihi+|lol|hehe+|xd|😂)", re.I)
BYE = re.compile(r"\b(?:bye+|byee*|bubye|doei+|dag+|daag+|tot (?:zo|straks|morgen|snel|gauw|dan|later|zaterdag|zondag|vanavond)|slaap lekker|welterusten|truste|goede?nacht|good ?night|night+|nite|gn|see (?:ya|you|u)|ttyl|talk later|later|cya|take care|love you|hou van je|xx+|x)\b", re.I)
ACK = re.compile(r"^\s*(?:ok+e*|oké|okay|okey|oki+|k+|top|prima|goed|is goed|oke dan|ja|jaa+|yes|yep|yup|cool|nice|thanks|dank je|dankje|bedankt|thnx|thx|alright|aight|sure|lekker|mooi|super|👍+|😘+|❤️*)[\s!.]*$", re.I)


def opening_type(texts):
    t = " / ".join(texts).strip()
    first = texts[0].strip()
    if all(MEDIA.search(x) for x in texts):
        return "media"
    has_media = any(MEDIA.search(x) for x in texts)
    g = GREET.match(first)
    rest = t
    if g:
        rest = first[g.end():]
        rest = NAME.sub("", rest).strip(" !.,?;:)(-/xX")
        rest = " / ".join([rest] + [x for x in texts[1:]]).strip(" /")
    rest_words = len(re.findall(r"\w+", rest))
    well = bool(WELL.search(t))
    if g and rest_words == 0:
        return "greet_only"
    if g and well and rest_words <= 6:
        return "greet+howareyou"
    if g and PRES.search(t) and rest_words <= 8:
        return "greet+ping"
    if g:
        return "greet+content"
    if well and len(re.findall(r"\w+", t)) <= 6:
        return "howareyou_only"
    if PRES.search(t):
        return "ping"
    if SORRY.search(t):
        return "apology_late"
    if has_media:
        return "media+text"
    if PAST.search(t) or LAUGH0.match(t):
        return "continuation"
    if LOGI.search(t):
        return "logistics"
    if "?" in t:
        return "direct_question"
    return "direct_statement"


def reply_type(texts):
    t = " / ".join(texts).strip()
    first = texts[0].strip()
    g = bool(GREET.match(first))
    well = bool(WELL.search(t))
    q = "?" in t
    elong = bool(re.search(r"([a-zA-Z])\1{2,}", t))
    if g and well:
        r = "greet+howareyou"
    elif g and q:
        r = "greet+question"
    elif g and len(re.findall(r"\w+", NAME.sub("", t))) <= 2:
        r = "greet_only"
    elif g:
        r = "greet+content"
    elif well:
        r = "howareyou"
    elif q:
        r = "question"
    else:
        r = "content"
    return r, elong


def main():
    t = L.turns()
    t = t[t.corpus.isin(["whatsapp_nl", "maichat"])].copy()
    # dyadic whatsapp only
    nsp = t.groupby("conv_id").speaker.nunique()
    t = t[t.conv_id.map(nsp) == 2]
    t["ts0"] = pd.to_datetime(t.ts_start.str[:19], errors="coerce", format="%Y-%m-%dT%H:%M:%S")
    t["ts1"] = pd.to_datetime(t.ts_end.str[:19], errors="coerce", format="%Y-%m-%dT%H:%M:%S")
    rows = []
    for (corp, cid), g in t.groupby(["corpus", "conv_id"]):
        g = g.sort_values("turn_idx")
        sess = [s for _, s in g.groupby("session")]
        prev = None
        for s in sess:
            f = s.iloc[0]
            r = {"corpus": corp, "conv_id": cid, "session": f.session, "n_turns": len(s),
                 "n_msgs": int(s.n_msgs.sum()), "n_speakers": s.speaker.nunique(),
                 "opener": f.speaker, "open_text": " / ".join(f.texts), "open_n_msgs": f.n_msgs,
                 "open_chars": f.total_chars, "open_type": opening_type(f.texts),
                 "open_sorry": bool(SORRY.search(" ".join(f.texts))),
                 "open_elong": bool(f.elongation), "open_has_q": bool(f.has_q),
                 "hour": f.ts0.hour if pd.notna(f.ts0) else None,
                 "weekday": f.ts0.weekday() if pd.notna(f.ts0) else None,
                 "ts0": f.ts0}
            if len(s) > 1:
                s2 = s.iloc[1]
                rt, el = reply_type(s2.texts)
                r.update(reply_text=" / ".join(s2.texts), reply_type=rt, reply_elong=el,
                         reply_latency_s=s2.response_latency_s, reply_chars=s2.total_chars, reply_n_msgs=s2.n_msgs,
                         reply_has_q=bool(s2.has_q))
            if prev is not None:
                r.update(gap_h=(f.ts0 - prev.iloc[-1].ts1).total_seconds() / 3600 if pd.notna(f.ts0) else None,
                         prev_last_speaker=prev.iloc[-1].speaker, prev_first_speaker=prev.iloc[0].speaker,
                         prev_n_turns=len(prev), prev_last_text=" / ".join(prev.iloc[-1].texts),
                         prev_last_farewell=bool(BYE.search(" / ".join(prev.iloc[-1].texts))),
                         prev_last_has_q=bool(prev.iloc[-1].has_q))
            # closing features
            last = s.iloc[-1]
            tail = s.iloc[-3:]
            r.update(closer=last.speaker, last_text=" / ".join(last.texts),
                     last_farewell=bool(BYE.search(" / ".join(last.texts))),
                     last_ack=bool(ACK.match(" / ".join(last.texts))),
                     last_has_q=bool(last.has_q), last_chars=last.total_chars,
                     n_bye_turns_tail=int(sum(bool(BYE.search(" / ".join(x))) for x in tail.texts)),
                     dur_min=(s.iloc[-1].ts1 - f.ts0).total_seconds() / 60 if pd.notna(f.ts0) else None)
            rows.append(r)
            prev = s
    df = pd.DataFrame(rows)
    df.to_pickle(os.path.join(L.SCR, "sessions.pkl"))
    print(df.shape)
    print(df.groupby("corpus").open_type.value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
