"""a4 — style features per message (richer + fixed version of features.py for the style study).

Why recompute: features.py's EMOJI class ends with a literal '-' so f.n_emoji counts hyphens.
Here: fixed emoji regex, laugh *forms*, end-punctuation type, abbreviations (EN + NL lists), dropped
apostrophes, lowercase 'i', etc.

Output (scratchpad, big): $A4_OUT/a4_msgs.pkl  (pandas DataFrame, one row per message)
"""
import json, os, re, sys
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
P = os.path.join(ROOT, "data", "processed")
OUT = os.environ.get("A4_OUT", "/tmp/a4")
os.makedirs(OUT, exist_ok=True)

# + U+E001-U+E53E: pre-2014 iPhone (SoftBank) emoji, frequent in whatsapp_nl; + ':shortcode:' (maichat/nps)
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF❤☺-]|:[a-z_]{3,}:")
LAUGH_EMOJI = re.compile("[\U0001F602\U0001F923\U0001F606\U0001F605\U0001F601\U0001F604\U0001F639\ue412]")
EMOTICON = re.compile(r"(?:^|\s)(?:[:;=8][-'o^]?[)(DPpOo/\\|*3\]\[]+|[xX][dDpP]+|<3+|\^\^|\^_\^|-_-|:'\()(?=\s|$|[.!?,])")
# laugh forms (token level)
LF = {
    "haha": re.compile(r"\b(?:a?ha(?:ha)+h?a*|ha{2,}h*|hah+)\b", re.I),
    "hehe": re.compile(r"\b(?:he(?:he)+h?|hihi(?:hi)*)\b", re.I),
    "lol": re.compile(r"\blo+l+(?:z|s)?\b", re.I),
    "lmao": re.compile(r"\b(?:lmf?ao+|rofl|lmfao+)\b", re.I),
    "xd": re.compile(r"(?:^|\s)[xX][dD]+(?=\s|$|[.!?])"),
    "kkk": re.compile(r"\bk{3,}\b|\bjaja+\b|\brs(?:rs)+\b", re.I),
    "laugh_emoji": LAUGH_EMOJI,
}
ELONG = re.compile(r"([a-zA-Z])\1{2,}")
LAUGH_TOKEN = re.compile(r"^(?:a?ha(?:ha)+h?a*|ha{2,}|he(?:he)+|hihi+|lo+l+|lmf?ao+|x+d+|h+m+|z{3,}|w{3,}.*)$", re.I)

EN_ABBR = set("""u ur r ya yea yeah yep yup nope idk omg tbh btw rn ngl pls plz thx thanx ty np brb gtg g2g wbu hbu wyd imo imho jk nvm smh ikr lol lmao gonna wanna gotta kinda sorta dunno cuz coz cos bc bcos k kk okie oki ok okay okies tmr tmrw 2moro 2day 2nite tonite b4 gr8 l8r luv abt wat wot dat da tho thru n y c 2 4 lor lah leh meh hor sia ah nah ya gd msg pple ppl bday lil tot shd wld cld dun nt jus juz v ard e r den oso ard haf hav wan wif 2morrow dat lot4 nite xx xoxo ily ilu tq sry sorry4 pic pics fb""".split())
EN_NO_APOS = set("im dont cant didnt wont isnt thats youre ive doesnt wasnt couldnt shouldnt wouldnt havent arent hes shes theyre whats lets aint".split())
NL_ABBR = set("""ff idd wss mss iig hvj lvj zsm gwn egt nie oke okee ok oki kk dr w8 bijv evt mn zn jwz ofz tis tuurlijk ffkes effe gezellig ben bn wrm wnr wie dx doei xx xxx thnx thanks tnx sws ah oh hmm nee jaa jup jep ja""".split()) - {"ben","wie","gezellig","ah","oh","ja","nee"}
TOK = re.compile(r"[A-Za-z0-9']+")


def laugh_forms(t):
    return [k for k, rx in LF.items() if rx.search(t)]


def end_type(t):
    s = t.rstrip()
    if not s:
        return "empty"
    if s.endswith("...") or s.endswith("…") or s.endswith(".."):
        return "ellipsis"
    if EMOTICON.search(s[-4:] if len(s) >= 4 else s) or EMOJI.search(s[-1]) or s[-1] in "️‍":
        return "emoji"
    c = s[-1]
    if c == ".":
        return "period"
    if c == "!":
        return "excl"
    if c == "?":
        return "question"
    if c.isalnum():
        # trailing laugh token counts as 'laugh' ending (a punctuation substitute)
        last = s.split()[-1]
        if LAUGH_TOKEN.match(last.strip(".,!?")) and not last.lower().startswith(("hm", "zz", "ww")):
            return "laugh"
        return "none"
    if s.endswith("]") or s.endswith(">"):
        return "anon"   # [REMOVED] / <#> anonymisation placeholders
    return "other"


def feats(t, lang):
    toks = TOK.findall(t)
    low = [w.lower() for w in toks]
    letters = [c for c in t if c.isalpha()]
    elong_words = [w for w in t.split() if ELONG.search(w) and not LAUGH_TOKEN.match(w.strip(".,!?"))
                   and not w.lower().startswith(("http", "www"))]
    lf = laugh_forms(t)
    abbr = EN_ABBR if lang == "en" else NL_ABBR
    return {
        "n_chars": len(t),
        "n_words": len(t.split()),
        "laugh": bool(lf),
        "lf": "|".join(lf),
        "emoji": bool(EMOJI.search(t)),
        "n_emoji": len(EMOJI.findall(t)),
        "emoticon": bool(EMOTICON.search(t)),
        "emo_any": bool(EMOJI.search(t) or EMOTICON.search(t)),
        "elong": bool(elong_words),
        "starts_lower": bool(letters) and t.lstrip()[:1].islower(),
        "all_lower": bool(letters) and not any(c.isupper() for c in letters) and len(letters) >= 3,
        "end": end_type(t),
        "ellipsis": ("..." in t) or ("…" in t),
        "caps_word": any(len(w) >= 3 and w.isupper() and w.isalpha() for w in t.split()),
        "multi_punct": bool(re.search(r"[!?]{2,}", t)),
        "n_excl": t.count("!"),
        "q": "?" in t,
        "abbr": any(w in abbr for w in low),
        "n_abbr": sum(w in abbr for w in low),
        "no_apos": lang == "en" and any(w in EN_NO_APOS for w in low),
        "lower_i": lang == "en" and bool(re.search(r"(?:^|\s)i(?:\s|$|')", t)),
        "self_corr": bool(re.match(r"^\s*\*\S|^\s*\S+\*\s*$", t)),
        "comma": "," in t,
    }


def main():
    rows = []
    for l in open(f"{P}/messages.jsonl", encoding="utf-8"):
        m = json.loads(l)
        t = m["text"] or ""
        if m["corpus"] == "nps_chatroom" and m.get("dialogue_act") == "System":
            continue
        if m["corpus"] == "whatsapp_nl" and ("weggelaten>" in t):
            media = True
        else:
            media = False
        r = {"corpus": m["corpus"], "conv_id": m["conv_id"], "idx": m["idx"], "speaker": m["speaker"],
             "text": t, "ts": m.get("ts"), "media": media, "da": m.get("dialogue_act"),
             "gold": m.get("gold_emotion")}
        r.update(feats(t, m.get("lang", "en")))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_pickle(f"{OUT}/a4_msgs.pkl")
    print(df.groupby("corpus").size())


if __name__ == "__main__":
    main()
