"""a2: loader helpers (turns/messages for maichat, whatsapp_nl, nps) cached as pickle in scratch."""
import json, os, pickle
import pandas as pd
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
P = os.path.join(ROOT, "data", "processed")
SCR = os.environ.get("A2_SCRATCH", "/tmp/a2_cache")
os.makedirs(SCR, exist_ok=True)
CORP = {"maichat", "whatsapp_nl", "nps_chatroom"}

def turns():
    fn = os.path.join(SCR, "turns.pkl")
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    rows = []
    for l in open(f"{P}/turns.jsonl", encoding="utf-8"):
        if '"corpus": "empathetic"' in l[:40]:
            continue
        d = json.loads(l)
        if d["corpus"] in CORP:
            d.pop("typing", None) if d["corpus"] != "maichat" else None
            rows.append(d)
    df = pd.DataFrame(rows)
    df.to_pickle(fn)
    return df

def messages():
    fn = os.path.join(SCR, "msgs.pkl")
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    rows = []
    for l in open(f"{P}/messages_feat.jsonl", encoding="utf-8"):
        if not any(f'"corpus": "{c}"' in l[:40] for c in CORP):
            continue
        d = json.loads(l)
        f = d.pop("f")
        d.update({"f_" + k: v for k, v in f.items()})
        d.pop("typing", None)
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_pickle(fn)
    return df

def jev():
    fn = os.path.join(SCR, "jev.pkl")
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    rows = []
    def flat(prefix, ans, out):
        for k, a in (ans or {}).items():
            t = a.get("type")
            if t == "noul": out[f"{prefix}{k}"] = a["noul"]
            elif t == "score": out[f"{prefix}{k}"] = a["score"]
            else:
                out[f"{prefix}{k}"] = a.get("choice")
                out[f"{prefix}{k}_conf"] = a.get("confidence")
    for l in open(f"{P}/jev_base.jsonl", encoding="utf-8"):
        d = json.loads(l)
        r = {k: d[k] for k in ("corpus", "conv_id", "session", "turn_idx", "turn_in_session", "speaker", "n_msgs",
                               "total_chars", "response_latency_s", "ts_start", "greeting", "farewell", "has_q")}
        r["text"] = " / ".join(d["texts"])
        flat("D_", d["D"], r); flat("P_", d["P"], r)
        r["has_P"] = d["P"] is not None
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_pickle(fn)
    return df
