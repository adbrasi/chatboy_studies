"""a4 shared loaders: message features (from a4_style_feats.py) and turn-level style features."""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
P = os.path.join(ROOT, "data", "processed")
OUT = os.environ.get("A4_OUT", "/tmp/a4")
ADATA = os.path.join(ROOT, "analysis", "data")

MSG_FEATS = ["laugh", "emo_any", "elong", "starts_lower", "all_lower", "ellipsis", "caps_word",
             "multi_punct", "q", "abbr", "no_apos", "lower_i", "self_corr"]


def load_msgs():
    df = pd.read_pickle(f"{OUT}/a4_msgs.pkl")
    df["end_period"] = df.end == "period"
    df["end_none"] = df.end.isin(["none"])
    df["end_excl"] = df.end == "excl"
    df["end_emoji"] = df.end == "emoji"
    df["logc"] = np.log1p(df.n_chars)
    return df


def load_turns(corpora=("maichat", "whatsapp_nl", "empathetic", "nps_chatroom")):
    """Turns from turns.jsonl + style flags aggregated from a4 message features."""
    m = load_msgs()
    m = m[m.corpus.isin(corpora)]
    key = {(r.corpus, r.conv_id, r.idx): r for r in m.itertuples(index=False)}
    rows = []
    for l in open(f"{P}/turns.jsonl", encoding="utf-8"):
        t = json.loads(l)
        if t["corpus"] not in corpora:
            continue
        ms = [key.get((t["corpus"], t["conv_id"], i)) for i in t["msg_idxs"]]
        ms = [x for x in ms if x is not None]
        if not ms:
            continue
        txt = [x for x in ms if not x.media]
        r = {k: t[k] for k in ("corpus", "conv_id", "session", "turn_idx", "turn_in_session", "speaker",
                               "n_msgs", "total_chars", "response_latency_s")}
        r["texts"] = [x.text for x in ms]
        r["media_only"] = not txt
        base = txt or ms
        for f in ("laugh", "emo_any", "elong", "abbr", "q", "caps_word", "ellipsis", "multi_punct", "no_apos", "lower_i"):
            r[f] = any(getattr(x, f) for x in base)
        r["starts_lower"] = base[0].starts_lower
        r["all_lower"] = all(x.all_lower or not any(c.isalpha() for c in x.text) for x in base) and any(x.all_lower for x in base)
        r["end_period"] = base[-1].end == "period"
        r["any_period_end"] = any(x.end == "period" for x in base)
        r["end_none"] = base[-1].end == "none"
        r["end_excl"] = base[-1].end == "excl"
        r["logc"] = float(np.log1p(t["total_chars"]))
        r["punct_frac"] = float(np.mean([x.end in ("period", "excl", "question", "ellipsis") for x in base]))
        r["n_words"] = int(sum(x.n_words for x in base))
        rows.append(r)
    return pd.DataFrame(rows)
