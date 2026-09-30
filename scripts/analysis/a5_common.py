"""a5 (engajamento/dinâmica): carregamento comum de turnos + camada Jev base, com colunas de sessão/continuidade."""
import json, os, re
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
P = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "analysis", "data")

ACK_VOCAB = set("""ok oke okee okey oké okay okido okidoki oki k kk ja jaa jaaa jep jup jip yes yeah yea ye yep yup ya nee no nope
jawel prima goed top chill leuk mooi lekker echt idd inderdaad zeker true sure same fair right cool nice wow oh ohh ah ahh aha
hmm hm mm mhm thanks thx tnx thnx bedankt dankje dank je snap ik is goed lol lmao xd rofl haha hahaha hihi hehe jaja
exactly facts good fine great yay yess yesss sweet oja ohja owja och ow alright alrighty okayy okk okie okii okiii""".split())
LAUGH_TOK = re.compile(r"^(h+a+h*)+h*$|^(h+e+)+h*$|^(h+i+){2,}h*$|^(l+o+l+)+$|^x+d+$|^(ha)+h?$|^(ja){2,}$", re.I)


def is_ack(texts):
    """Turno mínimo: só backchannel/risada/ok/emoji, sem conteúdo novo."""
    t = " ".join(texts).strip()
    if not t:
        return True
    toks = re.findall(r"[a-zà-ÿ']+", t.lower())
    if not toks:  # só emoji / pontuação / emoticon
        return "<" not in t and "[" not in t
    if len(toks) > 4:
        return False
    return all(w in ACK_VOCAB or LAUGH_TOK.match(w) for w in toks)


def is_laugh_only(texts):
    toks = re.findall(r"[a-zà-ÿ']+", " ".join(texts).lower())
    return bool(toks) and all(LAUGH_TOK.match(w) for w in toks)


def load_turns(corpora=("maichat", "whatsapp_nl")):
    rows = []
    for l in open(f"{P}/turns.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d["corpus"] in corpora:
            rows.append(d)
    df = pd.DataFrame(rows)
    df = df.sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
    g = df.groupby(["corpus", "conv_id", "session"])
    df["sess_len"] = g["turn_idx"].transform("count")
    df["turns_left"] = df["sess_len"] - df["turn_in_session"] - 1  # turnos que ainda vêm depois deste na sessão
    df["is_last"] = df["turns_left"] == 0
    df["text"] = df["texts"].apply(lambda x: " / ".join(x))
    df["ack"] = df["texts"].apply(is_ack)
    df["laugh_only"] = df["texts"].apply(is_laugh_only)
    df["ts_start"] = pd.to_datetime(df["ts_start"], utc=True, format="ISO8601", errors="coerce")
    df["ts_end"] = pd.to_datetime(df["ts_end"], utc=True, format="ISO8601", errors="coerce")
    # próximo turno na conversa (mesma sessão)
    for c in ["speaker", "total_chars", "n_msgs", "response_latency_s", "has_q", "laugh", "text", "session"]:
        df["next_" + c] = g[c].shift(-1) if c != "session" else df.groupby(["corpus", "conv_id"])[c].shift(-1)
    return df


def flat_jev(r):
    o = {}
    for k, v in (r.get("D") or {}).items():
        o["D_" + k] = v.get("noul", v.get("score", v.get("choice")))
        if v["type"] == "choice":
            o["D_" + k + "_conf"] = v.get("confidence")
    for k, v in (r.get("P") or {}).items():
        o["P_" + k] = v.get("noul", v.get("score", v.get("choice")))
    return o


def load_jev():
    rows = []
    for l in open(f"{P}/jev_base.jsonl", encoding="utf-8"):
        r = json.loads(l)
        o = {"corpus": r["corpus"], "conv_id": r["conv_id"], "turn_idx": r["turn_idx"]}
        o.update(flat_jev(r))
        rows.append(o)
    return pd.DataFrame(rows)


def load_all():
    t = load_turns()
    j = load_jev()
    return t.merge(j, on=["corpus", "conv_id", "turn_idx"], how="left")


def auc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def prop_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, c - h, c + h)
