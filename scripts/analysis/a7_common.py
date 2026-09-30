"""a7 (flerte/afeto/intimidade): carga comum de turnos + camada Jev base, com flags de flerte/afeto."""
import json, os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
P = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "analysis", "data")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
from a5_common import load_turns, flat_jev, is_ack, is_laugh_only  # noqa

ROMANTIC = {"romantic_partners", "flirting_or_crush"}


def load_jev_full():
    rows = []
    for l in open(f"{P}/jev_base.jsonl", encoding="utf-8"):
        r = json.loads(l)
        o = {"corpus": r["corpus"], "conv_id": r["conv_id"], "turn_idx": r["turn_idx"]}
        o.update(flat_jev(r))
        D = r.get("D") or {}
        if "relationship" in D:
            pr = D["relationship"]["probabilities"]
            o["p_romantic_rel"] = pr.get("romantic_partners", 0) + pr.get("flirting_or_crush", 0)
        if "emotion" in D:
            o["p_affection"] = D["emotion"]["probabilities"].get("affection", 0)
            o["p_tease"] = D["emotion"]["probabilities"].get("playful_teasing", 0)
        if "intent" in D:
            o["p_compliment_aff"] = D["intent"]["probabilities"].get("compliment_affection", 0)
        rows.append(o)
    return pd.DataFrame(rows)


def load():
    t = load_turns()
    j = load_jev_full()
    df = t.merge(j, on=["corpus", "conv_id", "turn_idx"], how="left")
    df["annotated"] = df["D_flirting"].notna()
    return df
