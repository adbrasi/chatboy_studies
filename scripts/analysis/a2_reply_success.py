"""a2: which FIRST REPLIES (2nd turn of a session) go with longer sessions? whatsapp_nl, within-chat normalized length.
Length counted AFTER the reply (n_turns - 2) to reduce the mechanical effect; still correlational."""
import os, sys, json, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import a2_load as L
from a2_sessions import GREET
s = pd.read_pickle(f"{L.SCR}/sessions.pkl")
w = s[(s.corpus == "whatsapp_nl") & s.reply_text.notna()].copy()
w["after"] = np.log1p(w.n_turns - 2)
w["rel_after"] = w.after - w.groupby("conv_id").after.transform("mean")
w["r_greet"] = w.reply_text.map(lambda t: bool(GREET.match(t)))
w["r_q"] = w.reply_has_q.astype(bool)
w["r_long"] = w.reply_chars >= 40
w["r_multi"] = w.reply_n_msgs >= 2
w["r_fast"] = w.reply_latency_s <= 60
w["o_greet"] = w.open_type.str.startswith("greet")
out = {}
for f in ["r_q", "r_greet", "r_long", "r_multi", "r_fast", "reply_elong"]:
    t = w.groupby(f).agg(n=("rel_after", "size"), rel_after=("rel_after", "mean"), med_turns=("n_turns", "median"))
    se = w.groupby(f).rel_after.sem()
    t["ci95"] = 1.96 * se
    out[f] = t.round(3).reset_index().to_dict("records")
    print(f); print(t.round(3))
g = w[w.o_greet]
print("greeting openings only (n=%d)" % len(g))
for f in ["r_q", "r_greet", "r_multi"]:
    print(f, g.groupby(f).rel_after.agg(["size", "mean"]).round(3).to_dict("index"))
    out["greet_open_" + f] = g.groupby(f).rel_after.agg(["size", "mean"]).round(3).reset_index().to_dict("records")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../../analysis/data/a2_reply_success.json"), "w"), indent=1)
