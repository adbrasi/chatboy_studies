"""a1: lookup tables for the delivery layer: bubbles per turn as a function of total length, and the
arousal / seriousness modulation inside each length bucket."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from a1_load import enriched
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "data")
T, M = enriched()
T["len_b"] = pd.cut(T.total_chars, [0, 20, 40, 80, 160, 10000], labels=["1-20", "21-40", "41-80", "81-160", ">160"])
res = {}
for c, g in T.groupby("corpus"):
    t = g.groupby("len_b", observed=True).apply(lambda x: pd.Series({
        "n": len(x), "P1": (x.n_msgs == 1).mean(), "P2": (x.n_msgs == 2).mean(), "P3": (x.n_msgs == 3).mean(), "P4+": (x.n_msgs >= 4).mean(),
        "mean_nm": x.n_msgs.mean(), "med_bubble_chars_multi": x[x.n_msgs > 1].mean_chars.median()})).round(3)
    res[c] = {"by_length": t.rename(index=str).to_dict(orient="index")}
    print(c); print(t)
    a = g[g.D_arousal.notna()].copy()
    a["ar"] = pd.cut(a.D_arousal, [-.1, 1.5, 2.5, 4.1], labels=["calm<=1.5", "mid", "energetic>=2.5"])
    a["se"] = pd.cut(a.D_seriousness, [-.1, .5, 1.5, 3.1], labels=["banter", "casual", "serious"])
    a["len2"] = pd.cut(a.total_chars, [0, 40, 100, 10000], labels=["<=40", "41-100", ">100"])
    for v in ["ar", "se"]:
        x = a.groupby(["len2", v], observed=True).agg(n=("n_msgs", "size"), mean_nm=("n_msgs", "mean"), multi=("n_msgs", lambda s: (s > 1).mean()),
                                                       bubble_med=("mean_chars", "median")).round(2)
        x = x[x.n >= 15]
        print(x)
        res[c][f"by_len_x_{v}"] = x.reset_index().astype(str).to_dict(orient="records")
json.dump(res, open(os.path.join(OUT, "a1_split_table.json"), "w"), indent=1)
