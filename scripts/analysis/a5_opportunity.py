"""a5 — Detector de oportunidade: efeito (observacional) de cada MOVIMENTO de um falante sobre o engajamento/continuidade
do parceiro no turno seguinte, condicionado à fase e à seriedade. Usa jev_base (D) + a5_jev_moves (callback, elogio...)."""
import json, os, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from a5_common import load_all, OUT
warnings.filterwarnings("ignore")
df = load_all().sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
mvp = f"{OUT}/a5_jev_moves.jsonl"
mv = pd.read_json(mvp, lines=True) if os.path.exists(mvp) else pd.DataFrame(columns=["corpus", "conv_id", "turn_idx"])
df = df.merge(mv.drop(columns=[c for c in mv.columns if c.endswith("_probs")]), on=["corpus", "conv_id", "turn_idx"], how="left")
g = df.groupby(["corpus", "conv_id", "session"])
df["logch"] = np.log1p(df.total_chars)
df["loglat"] = np.log1p(df.response_latency_s)
df["zlat"] = df.groupby(["conv_id", "speaker"]).loglat.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
df["zch"] = df.groupby(["conv_id", "speaker"]).logch.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
# movimentos no turno m (multi-rótulo)
df["mv_pergunta"] = (df.has_q | (df.D_intent == "ask_question")).astype(int)
df["mv_revela"] = ((df.D_vulnerable > .5) | (df.D_intent == "share_feeling")).astype(int)
df["mv_historia"] = (df.D_intent == "share_story_or_info").astype(int)
df["mv_brinca"] = (df.D_playful > .5).astype(int)
df["mv_elogia"] = ((df.D_intent == "compliment_affection") | (df.compliment > .5)).astype(int)
df["mv_muda_topico"] = (df.D_topic_shift > .5).astype(int)
df["mv_so_reage"] = (df.ack | (df.D_intent == "react_acknowledge")).astype(int)
df["mv_flerta"] = (df.D_flirting > .5).astype(int)
MOVES = ["mv_pergunta", "mv_revela", "mv_historia", "mv_brinca", "mv_elogia", "mv_muda_topico", "mv_so_reage", "mv_flerta"]
# resposta do parceiro (turno m+1) e nível anterior do parceiro (m-1)
for c in ("D_engagement", "logch", "zch", "zlat", "speaker", "has_q", "D_hook"):
    df["nx_" + c] = g[c].shift(-1)
df["pv_eng"] = g.D_engagement.shift(1)
df["pv_spk"] = g.speaker.shift(1)
df["cont1"] = (~df.is_last).astype(int); df["cont5"] = (df.turns_left >= 5).astype(int)
df["ser"] = df.D_seriousness
df["logpos"] = np.log1p(df.turn_in_session)
x = df[df.D_engagement.notna() & df.nx_D_engagement.notna() & df.pv_eng.notna() & (df.nx_speaker != df.speaker) & (df.pv_spk != df.speaker)].copy()
x["phase"] = x.D_phase
print("pares movimento→resposta:", len(x), x.corpus.value_counts().to_dict())
res = {"n_pairs": len(x)}
def fit(data, y, extra="", label=""):
    f = f"{y} ~ " + " + ".join(MOVES) + " + pv_eng + ser + C(phase) + logpos + C(conv_id)" + extra
    m = smf.ols(f, data=data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data.conv_id)[0]})
    ci = m.conf_int()
    out = {mvn: {"b": round(m.params[mvn], 3), "lo": round(ci.loc[mvn, 0], 3), "hi": round(ci.loc[mvn, 1], 3), "p": round(m.pvalues[mvn], 4),
                 "n_mv": int(data[mvn].sum())} for mvn in MOVES}
    return out, int(m.nobs)
tabs = {}
for y in ("nx_D_engagement", "nx_zch", "nx_zlat"):
    for corp in ("all", "maichat", "whatsapp_nl"):
        d = x if corp == "all" else x[x.corpus == corp]
        d = d.dropna(subset=[y])
        if y == "nx_zlat" and corp == "maichat": continue
        o, n = fit(d, y)
        tabs[f"{y}|{corp}"] = {"n": n, "coef": o}
        print(f"\n== {y} | {corp} (n={n}) ==")
        print(pd.DataFrame(o).T.to_string())
res["ols"] = tabs
# continuidade no whatsapp (logit, janelas anotadas)
w = x[x.corpus == "whatsapp_nl"]
wa_all = df[(df.corpus == "whatsapp_nl") & df.D_engagement.notna()].copy()
wa_all["phase"] = wa_all.D_phase
for y in ("cont1", "cont5"):
    f = f"{y} ~ " + " + ".join(MOVES) + " + ser + C(phase) + logpos"
    m = smf.logit(f, data=wa_all).fit(disp=0, maxiter=300)
    ci = m.conf_int()
    o = {mvn: {"OR": round(np.exp(m.params[mvn]), 3), "lo": round(np.exp(ci.loc[mvn, 0]), 3), "hi": round(np.exp(ci.loc[mvn, 1]), 3),
               "p": round(m.pvalues[mvn], 4)} for mvn in MOVES}
    res[f"logit_{y}_whatsapp"] = {"n": int(m.nobs), "coef": o}
    print(f"\n== logit {y} whatsapp (n={int(m.nobs)}) ==\n", pd.DataFrame(o).T.to_string())
# estratos: fase × seriedade -> efeito no engajamento do parceiro
strata = {}
PG = {"playful_banter": "leve_brincadeira", "small_talk": "leve_smalltalk", "opening": "leve_smalltalk", "logistics": "logistica",
      "deep_personal": "profundo_ou_tenso", "conflict_or_repair": "profundo_ou_tenso", "winding_down": "encerrando"}
x["phase_grp"] = x.phase.map(PG)
x["ser_bin"] = pd.cut(x.ser, [-1, .5, 1.0, 9], labels=["ser<0.5", "ser0.5-1", "ser>=1"]).astype(str)
for col in ("phase_grp", "ser_bin"):
    for key, d in x.groupby(col):
        if len(d) < 150: continue
        f = "nx_D_engagement ~ " + " + ".join(MOVES) + " + pv_eng + logpos + C(corpus)"
        m = smf.ols(f, data=d).fit()
        strata[f"{col}={key}"] = {"n": len(d), **{mvn: [round(m.params[mvn], 2), round(m.pvalues[mvn], 3), int(d[mvn].sum())] for mvn in MOVES}}
        f2 = "nx_zch ~ " + " + ".join(MOVES) + " + pv_eng + logpos + C(corpus)"
        m2 = smf.ols(f2, data=d.dropna(subset=["nx_zch"])).fit()
        strata[f"{col}={key}|zch"] = {"n": len(d), **{mvn: [round(m2.params[mvn], 2), round(m2.pvalues[mvn], 3), int(d[mvn].sum())] for mvn in MOVES}}
st = pd.DataFrame(strata).T
print("\n== efeito por estrato (b, p, n_mov) ==\n", st.to_string()); res["strata"] = strata
# médias brutas: engajamento do parceiro após cada movimento "puro" por fase
raw = []
for mvn in MOVES:
    for ph, d in x.groupby("phase"):
        if len(d) < 100: continue
        a, b = d[d[mvn] == 1], d[d[mvn] == 0]
        if len(a) < 20: continue
        raw.append({"move": mvn, "phase": ph, "n": len(a), "eng_after": round(a.nx_D_engagement.mean(), 2), "eng_after_others": round(b.nx_D_engagement.mean(), 2)})
res["raw_by_phase"] = raw
# ---- callbacks e outros rótulos novos (subconjunto com a5_jev_moves)
xm = x[x.callback.notna()].copy()
if len(xm) > 200:
    xm["mv_callback"] = (xm.callback > .5).astype(int)
    xm["mv_perg_pessoal"] = (xm.personal_q > .5).astype(int)
    xm["mv_historia2"] = (xm.story > .5).astype(int)
    xm["mv_elogio2"] = (xm.compliment > .5).astype(int)
    xm["mv_devolve"] = (xm.asks_back > .5).astype(int)
    NEW = ["mv_callback", "mv_perg_pessoal", "mv_historia2", "mv_elogio2", "mv_devolve", "mv_brinca", "mv_revela", "mv_muda_topico", "mv_so_reage"]
    out = {}
    for y in ("nx_D_engagement", "nx_zch", "nx_zlat"):
        d = xm.dropna(subset=[y])
        if y == "nx_zlat": d = d[d.corpus == "whatsapp_nl"]
        f = f"{y} ~ " + " + ".join(NEW) + " + pv_eng + ser + C(phase) + logpos + C(corpus)"
        m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.conv_id)[0]})
        ci = m.conf_int()
        out[y] = {"n": int(m.nobs), **{v: {"b": round(m.params[v], 3), "lo": round(ci.loc[v, 0], 3), "hi": round(ci.loc[v, 1], 3), "p": round(m.pvalues[v], 4), "n_mv": int(d[v].sum())} for v in NEW}}
        print(f"\n== subconjunto moves: {y} (n={int(m.nobs)}) ==\n", pd.DataFrame({k: v for k, v in out[y].items() if k != 'n'}).T.to_string())
    res["moves_subset"] = out
    # médias brutas do callback
    cb = xm.groupby("mv_callback").agg(n=("nx_D_engagement", "size"), eng=("nx_D_engagement", "mean"), chars_med=("nx_zch", "mean"), zlat=("nx_zlat", "mean")).round(3)
    print(cb.to_string()); res["callback_raw"] = cb.reset_index().to_dict("records")
json.dump(res, open(f"{OUT}/a5_opportunity.json", "w"), indent=1, default=str)
