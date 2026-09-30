"""a8: shared helpers (vocabulary / human tics vs LLM tics)."""
import json, os, pickle, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "analysis", "data")
SCR = "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad/a8"
os.makedirs(SCR, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _cached(name, fn):
    p = os.path.join(SCR, name + ".pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    x = fn()
    pickle.dump(x, open(p, "wb"))
    return x


def messages(corpora=("maichat", "nps_chatroom", "nus_sms", "whatsapp_nl", "empathetic"), emp_max_conv=6000):
    """Returns {corpus: [msg dicts]} (empathetic truncated to first emp_max_conv conversations)."""
    def load():
        out = {c: [] for c in corpora}
        seen_emp = set()
        for l in open(os.path.join(PROC, "messages_feat.jsonl"), encoding="utf-8"):
            m = json.loads(l)
            c = m["corpus"]
            if c not in out:
                continue
            if c == "empathetic":
                if m["conv_id"] not in seen_emp:
                    if len(seen_emp) >= emp_max_conv:
                        continue
                    seen_emp.add(m["conv_id"])
            out[c].append(m)
        return out
    return _cached("messages_" + "_".join(corpora), load)


def jev_base():
    return _cached("jev_base", lambda: [json.loads(l) for l in open(os.path.join(PROC, "jev_base.jsonl"), encoding="utf-8")])


def turns(corpus):
    def load():
        return [t for t in (json.loads(l) for l in open(os.path.join(PROC, "turns.jsonl"), encoding="utf-8")) if t["corpus"] == corpus]
    return _cached("turns_" + corpus, load)


TOK = re.compile(r"[a-z0-9']+|[^\sa-z0-9']", re.I)


def norm_form(t):
    """Normalize a short message to a 'form' for frequency tables."""
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def canon(t):
    """Canonical form: lowercase, collapse elongations (>2 -> 2), strip trailing punct runs to one char."""
    t = norm_form(t)
    t = re.sub(r"([a-z])\1{2,}", r"\1\1", t)
    t = re.sub(r"([!?.])\1+", r"\1", t)
    return t


def first_word(t):
    m = re.match(r"\s*([A-Za-z']+|[^\sA-Za-z0-9])", t)
    return m.group(1).lower() if m else ""


def save(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", p, os.path.getsize(p))


# ---------------- style / tic features (shared by human corpora and LLM outputs) ----------------
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]")
LAUGH_RE = re.compile(r"\b(?:a?ha(?:ha)+h?|he(?:he)+|hi(?:hi)+|lo+l+|lmf?ao+|rofl|hah+|haa+)\b|[\U0001F602\U0001F923]", re.I)
TICS = {  # name: regex (case-insensitive) — human discourse markers / crutches
    "like (qualquer uso)": r"\blike\b",
    "honestly": r"\bhonestly\b",
    "literally": r"\bliterally\b",
    "i mean": r"\bi mean\b",
    "tbh/ngl": r"\b(?:tbh|ngl)\b",
    "idk/dunno": r"\b(?:idk|dunno|i dont know|i don't know)\b",
    "omg": r"\bomg+\b",
    "lol/lmao": r"\b(?:lo+l+|lmf?ao+)\b",
    "haha-família": r"\b(?:a?ha(?:ha)+h?|hah+|he(?:he)+)\b",
    "anyway(s)": r"\banyways?\b",
    "wait": r"\bwait\b",
    "so (início)": r"^\s*so\b",
    "oh/ooh (início)": r"^\s*o+h+\b",
    "ok/okay/k (início)": r"^\s*(?:ok+|okay|oki|okie|k)\b",
    "yeah/yea/ya/yes": r"\b(?:yeah|yea|ya|yes+|yep|yup)\b",
    "no/nah/nope": r"\b(?:no+|nah|nope)\b",
    "hmm/umm/uh": r"\b(?:h?mm+|um+|uh+|erm)\b",
    "ugh": r"\bugh+\b",
    "same": r"^\s*same\b",
    "true/fair/exactly": r"^\s*(?:true|fair|exactly|facts)\b",
    "kinda/sorta/gonna/wanna": r"\b(?:kinda|sorta|gonna|wanna|gotta)\b",
    "u/ur/r (abreviação)": r"\b(?:u|ur|r)\b",
    "começa com and/but": r"^\s*(?:and|but)\b",
    "começa com wait": r"^\s*wait\b",
    "lol no fim (pontuação)": r"\b(?:lo+l+|lmf?ao+|haha\w*)\s*$",
}
TICS = {k: re.compile(v, re.I) for k, v in TICS.items()}

LLMISH = {  # candidate LLM-ish expressions (checked on every text source)
    "absolutely": r"\babsolutely\b",
    "totally": r"\btotally\b",
    "definitely": r"\bdefinitely\b",
    "that's/that is + amazing/awesome/great/wonderful/fantastic": r"\bthat(?:'|’)?s (?:so |really |absolutely )?(?:amazing|awesome|great|wonderful|fantastic|incredible|exciting)|\bthat is (?:so |really )?(?:amazing|awesome|great|wonderful|fantastic)",
    "sounds (like/amazing/fun/great…)": r"\bsounds (?:like|amazing|fun|great|awesome|good|lovely|perfect|exciting|intense|rough|tough)\b",
    "i'd love to / i would love": r"\bi(?:'|’)?d love\b|\bi would love\b",
    "it's okay to / it's normal to / it's understandable": r"\bit(?:'|’)?s (?:totally |completely )?(?:okay|ok|normal|understandable|valid) to\b|\bunderstandable\b",
    "i'm here for you / here if you need": r"\bi(?:'|’)?m (?:always )?here (?:for you|if you)|\bhere for you\b",
    "i'm (so) sorry to hear / sorry you're": r"\bsorry to hear\b|\bsorry you(?:'|’)?re\b",
    "so proud of you": r"\bproud of you\b",
    "that's so sweet / aww": r"\bthat(?:'|’)?s so sweet\b|\baw+\b",
    "journey": r"\bjourney\b",
    "vibe(s)": r"\bvibes?\b",
    "em dash —": r"[—–]",
    "começa com 'Oh,'": r"^\s*oh,",
    "começa com 'Haha,' / 'Lol,'": r"^\s*(?:haha+|lol),",
    "what about you / how about you / and you?": r"\b(?:what|how) about you\b|\band you\?|\bhbu\b|\bwbu\b|\byou\?\s*$",
    "let me know / feel free": r"\blet me know\b|\bfeel free\b",
    "i hope / hopefully": r"\bi hope\b|\bhopefully\b",
    "for sure": r"\bfor sure\b",
    "can't wait": r"\bcan(?:'|’)?t wait\b",
    "tell me (more/everything)": r"\btell me (?:more|everything|all)\b",
    "so much fun / a blast": r"\bso much fun\b|\ba blast\b",
    "nome do usuário (vocativo)": r"\bjordan\b",
    "right? (tag)": r"\bright\?",
    "💀/😭 (gíria emoji)": r"[\U0001F480\U0001F62D]",
    "😂/🤣": r"[\U0001F602\U0001F923]",
    "😊/🥰/❤️/✨/🎉": r"[\U0001F60A\U0001F970❤✨\U0001F389]",
    "gíria forçada (bestie/slay/no cap/fr fr/vibing/elite/iconic/lowkey)": r"\b(?:bestie|slay\w*|no cap|fr fr|vibing|elite|iconic|lowkey|highkey|periodt?|bussin)\b",
}
LLMISH = {k: re.compile(v, re.I) for k, v in LLMISH.items()}

STOP = set("""a an the and or but so to of in on at for with is are was were be been am i you he she it we they me my your
his her its our their this that these those do does did not no yes just like oh ok okay im i'm its it's dont don't
what how why when where who u ur r lol haha yeah ya yea be have has had will would can could should there here
then than too very really also if about up out get got go going one all some any more""".split())


def sentences(t):
    parts = re.split(r"(?<=[.!?…])\s+|\n+", t.strip())
    return [p for p in parts if re.search(r"\w", p)]


def content_words(t):
    return {w for w in re.findall(r"[a-z']+", t.lower()) if w not in STOP and len(w) > 2}


def style_feats(text, prev_user_text=""):
    t = text.strip()
    sents = sentences(t)
    cw, pw = content_words(t), content_words(prev_user_text)
    last = sents[-1] if sents else t
    f = {
        "chars": len(t), "words": len(t.split()), "sentences": len(sents),
        "lines": len([x for x in t.split("\n") if x.strip()]),
        "any_q": "?" in t, "ends_q": t.rstrip().endswith("?") or "?" in last,
        "n_excl": t.count("!"), "any_excl": "!" in t, "multi_excl": "!!" in t,
        "emoji": bool(EMOJI.search(t)), "n_emoji": len(EMOJI.findall(t)),
        "laugh": bool(LAUGH_RE.search(t)),
        "final_punct": bool(re.search(r"[.!?]\s*$", t)),
        "final_period": bool(re.search(r"(?<!\.)\.\s*$", t)),
        "starts_lower": bool(t) and t[0].isalpha() and t[0].islower(),
        "all_lower": t == t.lower() and any(c.isalpha() for c in t),
        "elongation": bool(re.search(r"([a-zA-Z])\1{2,}", t)),
        "no_apostrophe": bool(re.search(r"\b(?:dont|im|thats|cant|didnt|isnt|wasnt|ive|youre|ill|wont|doesnt|its gonna)\b", t, re.I)),
        "self_correction": bool(re.search(r"(?:^|\n)\s*\*\S|\S\*\s*$", t)),
        "le3_words": len(t.split()) <= 3,
        "echo_ratio": round(len(cw & pw) / len(cw), 3) if cw else 0.0,
        "echo_any2": len(cw & pw) >= 2,
    }
    for k, rx in TICS.items():
        f["tic:" + k] = bool(rx.search(t))
    for k, rx in LLMISH.items():
        f["llm:" + k] = bool(rx.search(t))
    return f
