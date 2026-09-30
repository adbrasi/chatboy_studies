"""a7: digitação (maichat) de quem RESPONDE a um flerte/afeto × a um turno neutro. Saída: analysis/data/a7_reply_typing.json"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_common import load, OUT

df = load().sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
m = df[df.corpus == "maichat"].copy()
m["fa"] = (m.D_flirting >= .5) | (m.D_emotion == "affection") | (m.D_intent == "compliment_affection")
m["neu"] = (m.D_flirting < .15) & (m.D_emotion != "affection") & (m.D_intent != "compliment_affection")
ty = pd.json_normalize(m["typing"]).set_index(m.index)
m = m.join(ty)
g = m.groupby("conv_id")
for c in ["compose_s_total", "deletion_ratio", "n_deletion_events", "abandoned_text", "max_pause_s", "total_chars", "response_latency_s", "speaker", "session"]:
    m["R_" + c] = g[c].shift(-1)
ok = (m.R_speaker.notna()) & (m.R_speaker != m.speaker) & (m.R_session == m.session)
m["R_idle"] = m.R_response_latency_s - m.R_compose_s_total  # tempo antes de começar a digitar (aprox.)
m["R_s_per_char"] = m.R_compose_s_total / m.R_total_chars.clip(lower=1)
out = {}
for k in ["R_compose_s_total", "R_s_per_char", "R_deletion_ratio", "R_n_deletion_events", "R_abandoned_text", "R_max_pause_s", "R_response_latency_s", "R_idle", "R_total_chars"]:
    a, b = m[ok & m.fa][k].dropna(), m[ok & m.neu][k].dropna()
    out[k] = {"fa_med": round(float(a.median()), 3), "neu_med": round(float(b.median()), 3), "fa_mean": round(float(a.mean()), 3),
              "neu_mean": round(float(b.mean()), 3), "n_fa": len(a), "n_neu": len(b), "p": float(mannwhitneyu(a, b).pvalue)}
for k, lab in [("R_n_deletion_events", "any_del"), ("R_abandoned_text", "any_abandon")]:
    out[lab] = {"fa": float((m[ok & m.fa][k] > 0).mean()), "neu": float((m[ok & m.neu][k] > 0).mean())}
json.dump(out, open(os.path.join(OUT, "a7_reply_typing.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
# texto abandonado nas respostas a flerte (o que a pessoa quase disse)
