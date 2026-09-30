"""Deterministic (code-computed) features per message + segmentation into turns/bursts.

A *turn* = maximal run of consecutive messages by the same speaker (the "burst").
A *session* (timestamped corpora) = a stretch of chat with no gap > SESSION_GAP.

Outputs:
  data/processed/messages_feat.jsonl
  data/processed/turns.jsonl
"""
import json, os, re
from datetime import datetime

P = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SESSION_GAP_S = 3 * 3600

EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF-]")
EMOTICON = re.compile(r"(?:^|\s)(?:[:;=xX8][-']?[)(DPpOo/\\|*3]+|<3|\^\^|\^_\^|-_-|:'\()(?=\s|$)")
LAUGH = re.compile(r"\b(?:a?ha(?:ha)+h?|he(?:he)+|hi(?:hi)+|lo+l+|lmf?ao+|rofl|haa+|hah+|jaja+|kkk+|rs(?:rs)+)\b", re.I)
LAUGH_EMOJI = re.compile("[\U0001F602\U0001F923\U0001F606\U0001F605\U0001F601\U0001F604\U0001F639]")
ELONG = re.compile(r"([a-zA-Z])\1{2,}")
SLANG = re.compile(r"\b(?:u|ur|r|ya|yea|yeah|yep|nope|idk|omg|tbh|btw|rn|ngl|pls|plz|thx|ty|np|brb|gtg|wbu|hbu|wyd|imo|imho|jk|nvm|smh|ikr|lol|lmao|gonna|wanna|gotta|kinda|sorta|dunno|cuz|coz|bc|k|kk|okie|oki|ok|okay|hmm+|mhm+|uh+|um+|ah+|oh+|ooh+|aww+|ugh+|yay+|wow+)\b", re.I)
GREET = re.compile(r"^\s*(?:hi+|hey+|hello+|helo+|hiya|yo+|sup|wassup|heya|howdy|good (?:morning|evening|afternoon)|morning|hoi|hey+|hallo|goedemorgen|hee+|ola|oi+|hai+|helluu*)\b", re.I)
BYE = re.compile(r"\b(?:bye+|byee+|good ?night|gn|nite|see (?:ya|you)|ttyl|later|cya|take care|doei+|dag|tot (?:zo|straks|morgen|snel|gauw)|slaap lekker|welterusten|xx+)\b", re.I)
MEDIA = re.compile(r"weggelaten>|<media omitted>", re.I)
URL = re.compile(r"https?://|www\.")


def msg_features(t):
    words = t.split()
    letters = [c for c in t if c.isalpha()]
    return {
        "n_chars": len(t),
        "n_words": len(words),
        "has_q": "?" in t,
        "n_excl": t.count("!"),
        "laugh": bool(LAUGH.search(t) or LAUGH_EMOJI.search(t)),
        "laugh_text": bool(LAUGH.search(t)),
        "n_emoji": len(EMOJI.findall(t)),
        "emoticon": bool(EMOTICON.search(t)),
        "emoji_only": bool(t.strip()) and not any(c.isalnum() for c in t),
        "elongation": bool(ELONG.search(t)),
        "caps_word": any(len(w) >= 3 and w.isupper() and w.isalpha() for w in words),
        "n_slang": len(SLANG.findall(t)),
        "ellipsis": "..." in t or "…" in t,
        "ends_punct": bool(re.search(r"[.!?]\s*$", t)),
        "starts_lower": bool(letters) and t.lstrip()[:1].islower(),
        "self_correction": bool(re.match(r"^\s*\*\S|^\s*\S+\*\s*$", t)),
        "greeting": bool(GREET.search(t)),
        "farewell": bool(BYE.search(t)),
        "media": bool(MEDIA.search(t)),
        "url": bool(URL.search(t)),
    }


def parse_ts(s):
    try:
        return datetime.fromisoformat(s) if s else None
    except ValueError:  # a few malformed WhatsApp export timestamps
        return None


def main():
    msgs = [json.loads(l) for l in open(f"{P}/messages.jsonl", encoding="utf-8")]
    with open(f"{P}/messages_feat.jsonl", "w", encoding="utf-8") as fo:
        for m in msgs:
            m["f"] = msg_features(m["text"])
            fo.write(json.dumps(m, ensure_ascii=False) + "\n")

    # group by conversation (skip SMS: not conversations)
    convs = {}
    for m in msgs:
        if m["corpus"] == "nus_sms":
            continue
        convs.setdefault((m["corpus"], m["conv_id"]), []).append(m)

    n = 0
    with open(f"{P}/turns.jsonl", "w", encoding="utf-8") as fo:
        for (corpus, cid), ms in convs.items():
            ms.sort(key=lambda m: m["idx"])
            if corpus == "nps_chatroom":
                ms = [m for m in ms if m.get("dialogue_act") != "System"]
            session, turns, cur = 0, [], None
            prev_ts = None
            for m in ms:
                ts = parse_ts(m["ts"])
                if ts and prev_ts and (ts - prev_ts).total_seconds() > SESSION_GAP_S:
                    session += 1
                    cur = None
                if cur is None or m["speaker"] != cur["speaker"]:
                    cur = {"corpus": corpus, "conv_id": cid, "session": session,
                           "speaker": m["speaker"], "msgs": []}
                    turns.append(cur)
                cur["msgs"].append(m)
                prev_ts = ts or prev_ts
            # summarize
            pos_in_session = {}
            for i, t in enumerate(turns):
                ms_ = t["msgs"]
                ts0, ts1 = parse_ts(ms_[0]["ts"]), parse_ts(ms_[-1]["ts"])
                prev = turns[i - 1] if i else None
                latency = None
                prev_end = parse_ts(prev["msgs"][-1]["ts"]) if prev else None
                if prev and prev["session"] == t["session"] and ts0 and prev_end:
                    latency = (ts0 - prev_end).total_seconds()
                pos_in_session[t["session"]] = pos_in_session.get(t["session"], -1) + 1
                rec = {
                    "corpus": corpus, "conv_id": cid, "session": t["session"],
                    "turn_idx": i, "turn_in_session": pos_in_session[t["session"]],
                    "speaker": t["speaker"], "msg_idxs": [m["idx"] for m in ms_],
                    "texts": [m["text"] for m in ms_],
                    "n_msgs": len(ms_),
                    "total_chars": sum(m["f"]["n_chars"] for m in ms_),
                    "mean_chars": round(sum(m["f"]["n_chars"] for m in ms_) / len(ms_), 1),
                    "ts_start": ms_[0]["ts"], "ts_end": ms_[-1]["ts"],
                    "burst_span_s": (ts1 - ts0).total_seconds() if ts0 and ts1 else None,
                    "response_latency_s": latency,
                    "laugh": any(m["f"]["laugh"] for m in ms_),
                    "n_emoji": sum(m["f"]["n_emoji"] for m in ms_),
                    "has_q": any(m["f"]["has_q"] for m in ms_),
                    "elongation": any(m["f"]["elongation"] for m in ms_),
                    "media": any(m["f"]["media"] for m in ms_),
                    "greeting": ms_[0]["f"]["greeting"],
                    "farewell": any(m["f"]["farewell"] for m in ms_),
                    "self_correction": any(m["f"]["self_correction"] for m in ms_),
                }
                if corpus == "maichat":
                    ty = [m["typing"] for m in ms_ if m.get("typing")]
                    if ty:
                        comp = sum(x["compose_s"] for x in ty)
                        rec["typing"] = {
                            "compose_s_total": round(comp, 2),
                            "chars_per_s": round(rec["total_chars"] / comp, 2) if comp > 0 else None,
                            "chars_deleted": sum(x["chars_deleted"] for x in ty),
                            "deletion_ratio": round(sum(x["chars_deleted"] for x in ty) / max(1, rec["total_chars"]), 3),
                            "n_deletion_events": sum(x["n_deletion_events"] for x in ty),
                            "max_pause_s": max(x["max_pause_s"] for x in ty),
                            "abandoned_text": sum(max(0, x["peak_len"] - len(m["text"])) for x, m in zip(ty, ms_)),
                        }
                if corpus == "empathetic":
                    rec["gold_emotion"] = ms_[0]["gold_emotion"]
                    rec["situation"] = ms_[0]["situation"]
                if corpus == "nps_chatroom":
                    rec["dialogue_acts"] = [m["dialogue_act"] for m in ms_]
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print("turns", n)


if __name__ == "__main__":
    main()
