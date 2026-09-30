"""a5 — Trajetórias de engajamento (D.engagement ao longo da sessão) e eventos que precedem subidas."""
import json, warnings
import numpy as np, pandas as pd
from a5_common import load_all, OUT
warnings.filterwarnings("ignore")
df = load_all()
j = df[df.D_engagement.notna()].sort_values(["corpus", "conv_id", "turn_idx"]).copy()
res = {}
# ---------- formas
def shape(e):
    n = len(e)
    s = pd.Series(e).rolling(5, center=True, min_periods=2).mean().values
    rng = s.max() - s.min()
    third = max(1, n // 3)
    a, b, c = s[:third].mean(), s[third:2 * third].mean(), s[2 * third:].mean()
    diffs = np.sign(np.diff(e)); diffs = diffs[diffs != 0]
    alt = (np.diff(diffs) != 0).mean() if len(diffs) > 2 else 0
    raw_sd = np.std(e)
    if rng < 0.6: k = "plana"
    elif b - max(a, c) > 0.25: k = "sobe_e_cai"
    elif min(a, c) - b > 0.25: k = "cai_e_sobe"
    elif c - a > 0.4: k = "crescente"
    elif a - c > 0.4: k = "decrescente"
    else: k = "oscilante_sem_tendencia"
    return k, dict(n=n, rng=rng, a=a, b=b, c=c, alt=alt, sd=raw_sd, mean=np.mean(e))
rows = []
for (corpus, cid, sess), g in j.groupby(["corpus", "conv_id", "session"]):
    if len(g) < 12: continue
    k, f = shape(g.D_engagement.values)
    # serra: alternância alta turno a turno (pico/vale a cada turno) com amplitude relevante
    f["serra"] = f["alt"] > 0.6 and f["sd"] > 0.6
    # assimetria: engajamento médio de cada falante
    sp = g.groupby("speaker").D_engagement.mean()
    f["gap_speakers"] = sp.max() - sp.min() if len(sp) > 1 else 0
    rows.append(dict(corpus=corpus, conv_id=cid, session=sess, shape=k, **f))
sh = pd.DataFrame(rows)
print(sh.groupby("corpus").shape.value_counts(normalize=True).round(3).to_string())
print(sh.groupby("corpus")[["n", "mean", "sd", "alt", "gap_speakers"]].median().round(3).to_string())
print("serra (alt>0.6 & sd>0.6):", sh.groupby("corpus").serra.mean().round(3).to_dict())
res["shapes"] = {c: g["shape"].value_counts().to_dict() for c, g in sh.groupby("corpus")}
res["shape_stats_median"] = sh.groupby("corpus")[["n", "mean", "sd", "alt", "gap_speakers"]].median().round(3).to_dict()
res["serra_share"] = sh.groupby("corpus").serra.mean().round(3).to_dict()
# o que explica a oscilação turno a turno? engajamento dentro do falante vs alternância
g = j.groupby(["corpus", "conv_id", "session"])
j["eng_prev_other"] = g.D_engagement.shift(1)
j["spk_prev_eng"] = j.groupby(["corpus", "conv_id", "session", "speaker"]).D_engagement.shift(1)
from scipy import stats
for c in ("maichat", "whatsapp_nl"):
    x = j[j.corpus == c].dropna(subset=["eng_prev_other", "spk_prev_eng"])
    res.setdefault("autocorr", {})[c] = {"rho_eng_vs_partner_prev": round(stats.spearmanr(x.D_engagement, x.eng_prev_other)[0], 3),
                                         "rho_eng_vs_own_prev": round(stats.spearmanr(x.D_engagement, x.spk_prev_eng)[0], 3),
                                         "rho_eng_vs_logchars": round(stats.spearmanr(x.D_engagement, np.log1p(x.total_chars))[0], 3)}
print(res["autocorr"])
# engajamento por posição normalizada na sessão (curva média)
j["relpos"] = j.turn_in_session / j.sess_len
j["relbin"] = pd.cut(j.relpos, np.linspace(0, 1, 6), include_lowest=True)
curve = j[j.sess_len >= 12].groupby(["corpus", "relbin"]).D_engagement.mean().round(3).unstack(0)
print(curve.to_string()); res["mean_curve"] = curve.reset_index().astype(str).to_dict("records")
# ---------- subidas: engajamento do falante S no turno t vs o turno anterior de S; o que o parceiro fez em t-1?
j["prev_intent"] = g.D_intent.shift(1)
j["prev_spk"] = g.speaker.shift(1)
for c in ("D_playful", "D_vulnerable", "D_topic_shift", "D_flirting", "D_hook", "has_q", "laugh", "ack", "D_seriousness"):
    j["prev_" + c] = g[c].shift(1)
x = j[(j.prev_spk.notna()) & (j.prev_spk != j.speaker) & j.spk_prev_eng.notna()].copy()
x["delta"] = x.D_engagement - x.spk_prev_eng
x["rise"] = x.delta >= 1
x["fall"] = x.delta <= -1
print("n pares", len(x), "p_rise", x.rise.mean().round(3), "p_fall", x.fall.mean().round(3))
lift = x.groupby("prev_intent").agg(n=("rise", "size"), p_rise=("rise", "mean"), p_fall=("fall", "mean"), mean_delta=("delta", "mean")).round(3)
lift["lift_rise"] = (lift.p_rise / x.rise.mean()).round(2)
lift = lift[lift.n >= 30].sort_values("mean_delta")
print(lift.to_string()); res["rise_by_prev_intent"] = lift.reset_index().to_dict("records")
bf = {}
for c in ("prev_D_playful", "prev_D_vulnerable", "prev_D_topic_shift", "prev_D_flirting", "prev_D_hook"):
    hi = x[c] > .5
    bf[c] = {"n_hi": int(hi.sum()), "p_rise_hi": round(x[hi].rise.mean(), 3), "p_rise_lo": round(x[~hi].rise.mean(), 3),
             "delta_hi": round(x[hi].delta.mean(), 3), "delta_lo": round(x[~hi].delta.mean(), 3)}
for c in ("prev_has_q", "prev_laugh", "prev_ack"):
    hi = x[c].astype(bool)
    bf[c] = {"n_hi": int(hi.sum()), "p_rise_hi": round(x[hi].rise.mean(), 3), "p_rise_lo": round(x[~hi].rise.mean(), 3),
             "delta_hi": round(x[hi].delta.mean(), 3), "delta_lo": round(x[~hi].delta.mean(), 3)}
print(json.dumps(bf, indent=0)); res["rise_by_prev_flag"] = bf
# regressão à média: delta depende muito do nível anterior
res["delta_by_prev_level"] = x.groupby(x.spk_prev_eng.round()).delta.agg(["size", "mean"]).round(3).reset_index().to_dict("records")
print(res["delta_by_prev_level"])
# exemplos de subida
ex = x[(x.delta >= 1.5)].sample(6, random_state=4)
g2 = df.set_index(["corpus", "conv_id", "turn_idx"])
exs = []
for _, r in ex.iterrows():
    prev = g2.loc[(r.corpus, r.conv_id, r.turn_idx - 1)].text
    exs.append({"corpus": r.corpus, "parceiro": prev[:160], "resposta": r.text[:160], "delta": round(r.delta, 2), "prev_intent": r.prev_intent})
print(json.dumps(exs, ensure_ascii=False, indent=0)); res["rise_examples"] = exs
sh.to_csv(f"{OUT}/a5_shapes.csv", index=False)
json.dump(res, open(f"{OUT}/a5_trajectories.json", "w"), indent=1, default=str, ensure_ascii=False)
