"""a5 — Tópicos: tipos de transição (Jev, a5_jev_moves), marcadores explícitos ('btw', 'trouwens'), callbacks e seu efeito
sobre o turno seguinte do parceiro; devolução de pergunta medida pelo Jev."""
import json, re, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from a5_common import load_all, prop_ci, OUT
warnings.filterwarnings("ignore")
df = load_all().sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
mv = pd.read_json(f"{OUT}/a5_jev_moves.jsonl", lines=True)
df = df.merge(mv, on=["corpus", "conv_id", "turn_idx"], how="left")
g = df.groupby(["corpus", "conv_id", "session"])
df["logch"] = np.log1p(df.total_chars)
df["zch"] = df.groupby(["conv_id", "speaker"]).logch.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
df["zlat"] = df.groupby(["conv_id", "speaker"]).response_latency_s.transform(lambda s: (np.log1p(s) - np.log1p(s).mean()) / (np.log1p(s).std() + 1e-9))
for c in ("D_engagement", "zch", "zlat", "speaker", "has_q", "text", "is_last"):
    df["nx_" + c] = g[c].shift(-1)
for c in ("personal_q", "has_q", "speaker", "text", "D_engagement"):
    df["pv_" + c] = g[c].shift(1)
res = {}
m = df[df.transition.notna()].copy()
print("n anotados (moves):", len(m), m.corpus.value_counts().to_dict())
res["n_moves"] = m.corpus.value_counts().to_dict()
# 1) distribuição dos tipos de transição
td = m.groupby("corpus").transition.value_counts(normalize=True).round(3).unstack(0)
print(td.to_string()); res["transition_dist"] = td.to_dict()
res["callback_rate"] = m.groupby("corpus").callback.apply(lambda s: round((s > .5).mean(), 3)).to_dict()
print("callback>0.5:", res["callback_rate"])
# 2) marcadores explícitos de mudança abrupta
MK = re.compile(r"\b(btw|by the way|anyway|anyways|random(ly)? (but|question)|unrelated|on another note|speaking of|trouwens|overigens|anyhow|oh en|oh ja en|ander onderwerp|wait|zeg|hey)\b", re.I)
df["marker"] = df.text.str.contains(MK)
m["marker"] = m.text.str.contains(MK)
cm = m.groupby("transition").marker.mean().round(3)
print("P(marcador | tipo de transição):", cm.to_dict()); res["marker_by_transition"] = cm.to_dict()
res["marker_rate_all"] = df.groupby("corpus").marker.mean().round(4).to_dict()
# 3) efeito de cada transição no parceiro (turno seguinte)
m2 = m[m.nx_speaker.notna() & (m.nx_speaker != m.speaker)].copy()
eff = m2.groupby("transition").agg(n=("nx_D_engagement", "size"), eng_next=("nx_D_engagement", "mean"), zch_next=("nx_zch", "mean"),
                                   zlat_next=("nx_zlat", "mean"), p_next_q=("nx_has_q", "mean")).round(3)
print(eff.to_string()); res["transition_effect_raw"] = eff.reset_index().to_dict("records")
# continuidade whatsapp por tipo de transição
w = m[m.corpus == "whatsapp_nl"]
ct = w.groupby("transition").agg(n=("is_last", "size"), p_cont1=("is_last", lambda s: 1 - s.mean()), p_cont5=("turns_left", lambda s: (s >= 5).mean())).round(3)
print(ct.to_string()); res["transition_cont_whatsapp"] = ct.reset_index().to_dict("records")
# 4) callback: efeito ajustado (engajamento anterior do parceiro, fase, seriedade, tamanho do próprio turno)
m2["cb"] = (m2.callback > .5).astype(int)
m2["abrupt"] = (m2.transition == "abrupt_shift").astype(int)
m2["smooth"] = (m2.transition == "smooth_shift").astype(int)
m2["ret"] = (m2.transition == "returns_earlier").astype(int)
m2["react"] = (m2.transition == "reaction_only").astype(int)
m2["pv_eng_partner"] = g.D_engagement.shift(1).loc[m2.index]
out = {}
for y in ("nx_D_engagement", "nx_zch", "nx_zlat"):
    d = m2.dropna(subset=[y, "pv_eng_partner"])
    f = f"{y} ~ cb + abrupt + smooth + ret + react + pv_eng_partner + D_seriousness + logch + C(D_phase) + C(corpus)"
    r = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.conv_id)[0]})
    ci = r.conf_int()
    out[y] = {"n": int(r.nobs), **{v: [round(r.params[v], 3), round(ci.loc[v, 0], 3), round(ci.loc[v, 1], 3), round(r.pvalues[v], 4)] for v in ("cb", "abrupt", "smooth", "ret", "react")}}
    print(y, out[y])
res["callback_transition_ols"] = out
# 5) quem inicia tópicos novos (Jev: smooth/abrupt) — o que falou mais antes? o que fez a última pergunta?
# 6) exemplos
def ex(mask, k=6):
    e = m[mask].sample(min(k, int(mask.sum())), random_state=11)
    return [{"corpus": r.corpus, "antes": (r.pv_text if isinstance(r.pv_text, str) else "")[:120], "turno": r.text[:160]} for _, r in e.iterrows()]
res["ex_callback"] = ex((m.callback > .8) & (m.corpus == "maichat")) + ex((m.callback > .8) & (m.corpus == "whatsapp_nl"), 3)
res["ex_abrupt"] = ex((m.transition == "abrupt_shift") & (m.corpus == "maichat"))
res["ex_smooth"] = ex((m.transition == "smooth_shift") & (m.corpus == "maichat"))
for k in ("ex_callback", "ex_abrupt", "ex_smooth"):
    print(k); [print("  ", e) for e in res[k]]
# 7) devolução de pergunta pelo Jev: após pergunta (has_q) do parceiro, P(asks_back)
a = m[(m.pv_has_q == True) & (m.pv_speaker != m.speaker)]
for c, d in a.groupby("corpus"):
    k, n = int((d.asks_back > .5).sum()), len(d)
    p, lo, hi = prop_ci(k, n)
    res.setdefault("asks_back_after_q", {})[c] = {"n": n, "p": round(p, 3), "ci": [round(lo, 3), round(hi, 3)]}
    pp = d[d.pv_personal_q > .5]
    if len(pp) > 10:
        k2 = int((pp.asks_back > .5).sum())
        res["asks_back_after_q"][c]["after_personal_q"] = {"n": len(pp), "p": round(k2 / len(pp), 3)}
    np_ = d[d.pv_personal_q <= .5]
    if len(np_) > 10:
        res["asks_back_after_q"][c]["after_nonpersonal_q"] = {"n": len(np_), "p": round((np_.asks_back > .5).mean(), 3)}
print("asks_back:", res["asks_back_after_q"])
# efeito de devolver a pergunta sobre o parceiro
ab = m2.assign(ab=(m2.asks_back > .5).astype(int))
eab = ab[ab.pv_has_q == True].groupby("ab").agg(n=("nx_D_engagement", "size"), eng=("nx_D_engagement", "mean"), zch=("nx_zch", "mean"),
                                                   cont=("nx_is_last", lambda s: 1 - s.astype(float).mean())).round(3)
print(eab.to_string()); res["asks_back_effect_raw"] = eab.reset_index().to_dict("records")
res["ex_asks_back"] = [{"antes": (r.pv_text if isinstance(r.pv_text, str) else "")[:100], "turno": r.text[:140]} for _, r in m[(m.asks_back > .8)].sample(8, random_state=3).iterrows()]
print(res["ex_asks_back"])
# pergunta pessoal: taxa e quem pergunta
res["personal_q_rate"] = m.groupby("corpus").personal_q.apply(lambda s: round((s > .5).mean(), 3)).to_dict()
res["personal_q_share_of_questions"] = m[m.has_q].groupby("corpus").personal_q.apply(lambda s: round((s > .5).mean(), 3)).to_dict()
print(res["personal_q_rate"], res["personal_q_share_of_questions"])
json.dump(res, open(f"{OUT}/a5_topics.json", "w"), indent=1, default=str, ensure_ascii=False)
