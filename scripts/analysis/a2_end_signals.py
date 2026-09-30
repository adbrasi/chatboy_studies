"""a2: deterministic early-end signals on ALL whatsapp_nl/maichat session turns (not only jev windows)."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import a2_load as L
from a2_sessions import BYE
t = L.turns(); t = t[t.corpus.isin(["whatsapp_nl", "maichat"])].copy()
mx = t.groupby(["corpus", "conv_id", "session"]).turn_in_session.transform("max")
t["to_end"] = mx - t.turn_in_session
t = t[mx >= 7]
t["d"] = t.to_end.clip(upper=10)
t["lat_min"] = t.response_latency_s / 60
t["slow5"] = t.lat_min >= 5
t["bye"] = t.texts.map(lambda x: bool(BYE.search(" / ".join(x))))
out = {}
for c, g in t.groupby("corpus"):
    tab = g.groupby("d").agg(n=("n_msgs", "size"), chars_med=("total_chars", "median"), n_msgs_mean=("n_msgs", "mean"),
                             q=("has_q", "mean"), laugh=("laugh", "mean"), emoji=("n_emoji", lambda s: (s > 0).mean()),
                             lat_med_min=("lat_min", "median"), slow5=("slow5", "mean"), bye=("bye", "mean")).round(3)
    print(c); print(tab)
    out[c] = tab.reset_index().to_dict("records")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../../analysis/data/a2_end_signals.json"), "w"), indent=1)
