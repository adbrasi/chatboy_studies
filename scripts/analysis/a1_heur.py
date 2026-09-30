"""a1: heuristic bubble-function tagger (shared)."""
import re

LAUGH = re.compile(r"^\W*((ha|he|hi|ah){2,}h?|lo+l|lmao+|lmfao|rofl|xd+|kk+|hah+a*|jaja+|😂+|🤣+|😆+|:'?\)+|:d+|xd)\W*$", re.I)
REACT = re.compile(r"^\W*(ok(ay|e|ee|ie)?|oke+|k+|yes+|yeah+|yea|yep|ya+|ja+h?|jaa+|jep|nee+|no+|nope|nah|oh+|ohh+|ooh+|aw+h?|aa+h+|ah+|wow+|omg|oo+h|hmm+|mm+|nice|cool|true|same|right|fair|damn|echt|wauw|haha+ ?ja|top|super|leuk|lekker|prima|goed|mooi|jeetje|serieus|wtf|lol ok|sure|thanks|thank you|thx|dank je|dankje|bedankt|sorry|sws|yay|jup|klopt|inderdaad|precies|idd)\W*[!?.]*\W*$", re.I)
EMO_ONLY = re.compile(r"^[\W\s-]+$")


def heur(text, f):
    t = text.strip()
    if f["f_media"] or t.startswith("<") and "weggelaten" in t:
        return "media"
    if f["f_self_correction"] or (len(t) < 30 and (t.startswith("*") or t.endswith("*"))):
        return "correction"
    if f["f_emoji_only"] or (t and EMO_ONLY.match(t) and not any(ch.isalnum() for ch in t)):
        return "emoji"
    if LAUGH.match(t):
        return "laugh"
    if "?" in t:
        return "question"
    if REACT.match(t) or (len(t.split()) <= 2 and len(t) <= 12):
        return "reaction"
    return "content"


