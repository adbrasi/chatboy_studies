"""a5 — padrões 'invisíveis' de dinâmica: espelhamento de tamanho, quem fecha/reabre sessões, pergunta sem resposta,
desaceleração antes do fim, reabertura com callback."""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from a5_common import load_all, OUT
warnings.filterwarnings("ignore")
df = load_all().sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
res = {}
df["logch"] = np.log1p(df.total_chars)
df["zch"] = df.groupby(["conv_id", "speaker"]).logch.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
g = df.groupby(["corpus", "conv_id", "session"])
df["pv_zch"] = g.zch.shift(1); df["pv_spk"] = g.speaker.shift(1); df["pv_nmsgs"] = g.n_msgs.shift(1)
df["own_pv_zch"] = df.groupby(["corpus", "conv_id", "session", "speaker"]).zch.shift(1)
for c in ("maichat", "whatsapp_nl"):
    d = df[(df.corpus == c) & (df.pv_spk != df.speaker)].dropna(subset=["pv_zch", "own_pv_zch"])
    import statsmodels.formula.api as smf
    m = smf.ols("zch ~ pv_zch + own_pv_zch", data=d).fit()
    res.setdefault("length_mirroring", {})[c] = {"n": int(m.nobs), "b_partner_prev": round(m.params.pv_zch, 3), "b_own_prev": round(m.params.own_pv_zch, 3),
                                                 "p_partner": float(m.pvalues.pv_zch)}
    d2 = df[(df.corpus == c) & (df.pv_spk != df.speaker)].dropna(subset=["pv_nmsgs"])
    res.setdefault("bubbles_mirroring_rho", {})[c] = round(stats.spearmanr(d2.n_msgs, d2.pv_nmsgs)[0], 3)
print(res)
w = df[df.corpus == "whatsapp_nl"].copy()
ss = w.groupby(["conv_id", "session"])
first = ss.head(1).set_index(["conv_id", "session"]); last = ss.tail(1).set_index(["conv_id", "session"])
sess = pd.DataFrame({"opener": first.speaker, "closer": last.speaker, "open_q": first.has_q, "open_greet": first.greeting,
                     "close_q": last.has_q, "close_fw": last.farewell, "len": ss.size(), "open_text": first.text, "close_text": last.text}).reset_index()
sess = sess[sess.len >= 2]
res["p_opener_is_closer"] = round((sess.opener == sess.closer).mean(), 3)
res["p_session_ends_with_unanswered_q"] = round(sess.close_q.mean(), 3)
res["p_session_ends_with_farewell"] = round(sess.close_fw.mean(), 3)
res["p_session_opens_with_q"] = round(sess.open_q.mean(), 3)
res["p_session_opens_with_greeting"] = round(sess.open_greet.mean(), 3)
# quem reabre a próxima sessão: quem NÃO falou por último?
sess = sess.sort_values(["conv_id", "session"])
sess["next_opener"] = sess.groupby("conv_id").opener.shift(-1)
z = sess.dropna(subset=["next_opener"])
res["p_next_session_opened_by_who_did_not_close"] = round((z.next_opener != z.closer).mean(), 3)
z2 = z[z.close_q]
res["p_next_opened_by_other_when_closed_with_unanswered_q"] = round((z2.next_opener != z2.closer).mean(), 3)
# sessões que abrem com pergunta duram mais?
res["sess_len_median_open_q_vs_not"] = [float(sess[sess.open_q].len.median()), float(sess[~sess.open_q].len.median())]
res["sess_len_mean_open_q_vs_not"] = [round(sess[sess.open_q].len.mean(), 2), round(sess[~sess.open_q].len.mean(), 2)]
print(json.dumps(res, indent=1))
res["ex_unanswered_close"] = sess[sess.close_q].sample(6, random_state=1).close_text.str[:120].tolist()
print(res["ex_unanswered_close"])
json.dump(res, open(f"{OUT}/a5_invisible.json", "w"), indent=1, default=str, ensure_ascii=False)
