"""a3: carrega jev_base (maichat + whatsapp_nl) num DataFrame achatado, com o turno anterior do parceiro
(t-1), o próximo turno do parceiro (t+1) e o turno anterior do próprio falante (t-2) alinhados."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from a3_common import ROOT

P = os.path.join(ROOT, "data", "processed")
SB = ""
EMOTICON_SMILE = ("xD", "XD", ":D", ":-D", ";D")

def top(a):
    if a is None: return np.nan
    return a.get("noul", a.get("score", a.get("choice")))

def load():
    rows = []
    for l in open(f"{P}/jev_base.jsonl", encoding="utf-8"):
        b = json.loads(l)
        txt = " / ".join(b["texts"])
        r = {k: b[k] for k in ("corpus", "conv_id", "session", "turn_idx", "turn_in_session", "speaker", "n_msgs",
                               "total_chars", "response_latency_s", "burst_span_s", "n_emoji", "has_q", "media", "greeting", "farewell")}
        r["text"] = txt
        r["laugh"] = bool(b["laugh"] or any(c in txt for c in SB))
        r["laugh_or_xd"] = r["laugh"] or any(e in txt for e in EMOTICON_SMILE)
        r["emoji_any"] = b["n_emoji"] > 0 or any(0xE000 <= ord(c) <= 0xF8FF for c in txt)
        for k, v in b["D"].items():
            r["D_" + k] = top(v)
            if v.get("type") == "choice": r["Dconf_" + k] = v.get("confidence")
        if b["P"]:
            for k, v in b["P"].items():
                r[k] = top(v)
                if k == "p_tone": r["p_tone_probs"] = v["probabilities"]
        rows.append(r)
    D = pd.DataFrame(rows).sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
    key = D.set_index(["corpus", "conv_id", "turn_idx"])
    cols = ["speaker", "session", "D_valence", "D_seriousness", "D_playful", "D_vulnerable", "D_seeks_support", "laugh",
            "emoji_any", "total_chars", "n_msgs", "response_latency_s", "has_q", "D_intent", "D_emotion", "text", "D_arousal", "D_anxious", "D_tension"]
    for off, pre in ((-1, "prev_"), (1, "next_"), (-2, "prev2_")):
        idx = pd.MultiIndex.from_arrays([D.corpus, D.conv_id, D.turn_idx + off])
        sub = key.reindex(idx)[cols]
        sub.columns = [pre + c for c in cols]
        D = pd.concat([D, sub.reset_index(drop=True)], axis=1)
        ok = (D[pre + "session"] == D["session"])
        if off != -2:
            ok &= D[pre + "speaker"] != D["speaker"]
        else:
            ok &= D[pre + "speaker"] == D["speaker"]
        D.loc[~ok, [pre + c for c in cols]] = np.nan
    return D

def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def boot_conv(D, fn, n=500, seed=0):
    """IC95% por bootstrap de conversas."""
    rng = np.random.default_rng(seed)
    g = [x for _, x in D.groupby(["corpus", "conv_id"])]
    vals = []
    for _ in range(n):
        s = pd.concat([g[i] for i in rng.integers(0, len(g), len(g))])
        vals.append(fn(s))
    return np.nanpercentile(vals, [2.5, 97.5])
