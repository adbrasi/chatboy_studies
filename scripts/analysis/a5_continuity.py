"""a5 — O que mantém a conversa viva × o que a mata (whatsapp_nl, corpus completo + camada Jev)."""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from a5_common import load_all, prop_ci, OUT
warnings.filterwarnings("ignore")

df = load_all()
w = df[df.corpus == "whatsapp_nl"].copy()
w["nchar"] = w.total_chars
w["pos"] = pd.cut(w.turn_in_session, [-1, 2, 9, 1e9], labels=["0-2", "3-9", "10+"])
w["hour"] = w.ts_start.dt.hour
def cat(r):
    if r.farewell: return "despedida"
    if r.media and r.nchar < 40: return "midia"
    if r.laugh_only: return "so_risada"
    if r.ack: return "minimo_ok"
    if r.has_q: return "pergunta"
    if r.nchar >= 100: return "longo_sem_pergunta"
    return "afirmacao_curta_media"
w["cat"] = w.apply(cat, axis=1)
w["cont1"] = ~w.is_last
for k in (3, 5, 10):
    w[f"cont{k}"] = w.turns_left >= k
res = {}
tab = []
for c, g in w.groupby("cat"):
    row = {"cat": c, "n": len(g)}
    for k in (1, 3, 5, 10):
        p, lo, hi = prop_ci(g[f"cont{k}"].sum(), len(g))
        row[f"p_cont{k}"] = round(p, 3); row[f"ci{k}"] = f"[{lo:.2f},{hi:.2f}]"
    lat = g.next_response_latency_s.dropna()
    row["lat_parceiro_med_s"] = float(lat.median()) if len(lat) else None
    tab.append(row)
tab = pd.DataFrame(tab).sort_values("p_cont1")
print("== whatsapp_nl: continuidade por tipo de turno (todos os turnos) ==")
print(tab.to_string(index=False))
res["by_cat"] = tab.to_dict("records")
# estratificado por posição
st = w.groupby(["pos", "cat"]).agg(n=("cont1", "size"), p_cont1=("cont1", "mean"), p_cont5=("cont5", "mean")).round(3).reset_index()
print(st.to_string(index=False))
res["by_pos_cat"] = st.to_dict("records")
# logística com controles
w["cat"] = pd.Categorical(w.cat, categories=["afirmacao_curta_media"] + sorted(set(w.cat) - {"afirmacao_curta_media"}))
w["logpos"] = np.log1p(w.turn_in_session)
w["lognc"] = np.log1p(w.nchar)
m = smf.logit("cont1 ~ C(cat) + logpos + C(hour // 6) + C(conv_id)", data=w.assign(cont1=w.cont1.astype(int))).fit(disp=0, maxiter=200)
co = m.params.filter(like="cat"); ci = m.conf_int().loc[co.index]
or_tab = pd.DataFrame({"OR": np.exp(co), "lo": np.exp(ci[0]), "hi": np.exp(ci[1]), "p": m.pvalues[co.index]}).round(3)
print("\n== logit P(parceiro responde na sessão) ~ tipo + posição + hora + efeito fixo do chat ==")
print(or_tab.to_string())
res["logit_cont1_OR"] = or_tab.reset_index().to_dict("records")

# ---- camada Jev (janelas anotadas)
j = w[w.D_engagement.notna()].copy()
print("\n== Jev (whatsapp, n=%d) ==" % len(j))
for col in ["D_hook", "D_vulnerable", "D_topic_shift", "D_playful", "D_seeks_support", "D_flirting"]:
    hi = j[col] >= 0.5
    a, b = j[hi], j[~hi]
    row = {"feat": col, "n_hi": int(hi.sum()), "p_cont1_hi": a.cont1.mean(), "p_cont1_lo": b.cont1.mean(),
           "p_cont5_hi": a.cont5.mean(), "p_cont5_lo": b.cont5.mean()}
    res.setdefault("jev_feats", []).append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
# hook em faixas
j["hook_bin"] = pd.cut(j.D_hook, [0, .2, .5, .8, 1.01], include_lowest=True)
hb = j.groupby("hook_bin").agg(n=("cont1", "size"), p_cont1=("cont1", "mean"), p_cont3=("cont3", "mean"), p_cont5=("cont5", "mean")).round(3)
print(hb.to_string()); res["hook_bins"] = hb.reset_index().astype(str).to_dict("records")
ib = j.groupby("D_intent").agg(n=("cont1", "size"), p_cont1=("cont1", "mean"), p_cont3=("cont3", "mean"), p_cont5=("cont5", "mean"),
                               lat_med=("next_response_latency_s", "median")).round(3).sort_values("p_cont1")
print(ib.to_string()); res["intent"] = ib.reset_index().to_dict("records")
jm = smf.logit("cont1 ~ D_hook + D_vulnerable + D_topic_shift + D_playful + ack + logpos + C(conv_id)",
               data=j.assign(cont1=j.cont1.astype(int), ack=j.ack.astype(int))).fit(disp=0, maxiter=300, method="bfgs")
co = jm.params[["D_hook", "D_vulnerable", "D_topic_shift", "D_playful", "ack", "logpos"]]
ci = jm.conf_int().loc[co.index]
t2 = pd.DataFrame({"OR(0→1)": np.exp(co), "lo": np.exp(ci[0]), "hi": np.exp(ci[1]), "p": jm.pvalues[co.index]}).round(3)
print(t2.to_string()); res["jev_logit"] = t2.reset_index().to_dict("records")
# D_hook AUC para o fim
from a5_common import auc
res["auc_hook_cont1"] = auc(j.cont1, j.D_hook); res["auc_minus_ack"] = auc(j.cont1, 1 - j.ack.astype(float))
res["auc_hasq"] = auc(j.cont1, j.has_q.astype(float))
print("AUC hook->continua", res["auc_hook_cont1"], "AUC has_q", res["auc_hasq"], "AUC não-ack", res["auc_minus_ack"])
# exemplos: turnos que mataram (último) com hook baixo e sem pergunta
ex = j[(j.is_last) & (j.D_hook < .2)].sample(8, random_state=3)[["text", "D_intent"]]
print(ex.to_string())
ex2 = j[(~j.is_last) & (j.D_hook > .8) & (j.turns_left >= 10)].sample(8, random_state=3)[["text", "D_intent"]]
print(ex2.to_string())
# quem manda a última mensagem da sessão: tipo
last = w[w.is_last]
print("último turno da sessão, dist tipo:", last.cat.value_counts(normalize=True).round(3).to_dict())
res["last_turn_type_dist"] = last.cat.value_counts(normalize=True).round(3).to_dict()
res["base_rates"] = {"n_turns": len(w), "p_last": float(w.is_last.mean()), "n_sessions": int(w.groupby(["conv_id", "session"]).ngroups)}
json.dump(res, open(f"{OUT}/a5_continuity.json", "w"), ensure_ascii=False, indent=1, default=str)
