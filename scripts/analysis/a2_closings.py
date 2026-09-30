"""a2: closings. Joins jev_base with session position; features by distance to session end; P.p_end AUC."""
import json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import a2_load as L
from a2_sessions import BYE, ACK

def auc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    if y.sum() == 0 or (~y).sum() == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())

def boot_auc(y, s, B=500, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(s, float); n = len(y)
    v = [auc(y[i], s[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.nanpercentile(v, [2.5, 97.5])

def build():
    j = L.jev()
    t = L.turns()
    t = t[t.corpus.isin(["maichat", "whatsapp_nl"])]
    ns = t.groupby(["corpus", "conv_id", "session"]).turn_in_session.max().rename("last_tis")
    j = j.join(ns, on=["corpus", "conv_id", "session"])
    j["to_end"] = j.last_tis - j.turn_in_session
    j["is_last"] = j.to_end == 0
    j["bye"] = j.text.apply(lambda x: bool(BYE.search(x)))
    j["ack"] = j.text.apply(lambda x: bool(ACK.match(x)))
    j["closing_intent"] = j.D_intent == "closing"
    j["winding"] = j.D_phase == "winding_down"
    j["lat_min"] = j.response_latency_s / 60
    # previous turn features (what the bot sees before replying)
    j = j.sort_values(["corpus", "conv_id", "turn_idx"])
    g = j.groupby(["corpus", "conv_id", "session"])
    j["prev_chars"] = g.total_chars.shift(1)
    j["prev_lat_min"] = g.lat_min.shift(1)
    j["prev_bye"] = g.bye.shift(1)
    j["prev_eng"] = g.D_engagement.shift(1)
    j["prev_winding"] = g.winding.shift(1)
    j["prev_closing"] = g.closing_intent.shift(1)
    j["prev_has_q"] = g.has_q.shift(1)
    j["chars_trend"] = g.total_chars.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    return j

if __name__ == "__main__":
    j = build()
    j.to_pickle(os.path.join(L.SCR, "closing.pkl"))
    print(j.shape, j.is_last.sum())
