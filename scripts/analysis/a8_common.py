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
