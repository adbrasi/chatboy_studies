"""a5 — Latência como sinal de interesse (whatsapp_nl, resolução de minuto; maichat em segundos como apoio)."""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from a5_common import load_all, OUT
warnings.filterwarnings("ignore")
df = load_all()
res = {}
for corpus in ("whatsapp_nl", "maichat"):
    d = df[df.corpus == corpus].copy()
    d["lat"] = d.response_latency_s
    d = d[d.lat.notna() | True]
    d["loglat"] = np.log1p(d.lat)
    # z-score within speaker (conv, speaker) -> compara a pessoa com ela mesma
    d["zlat"] = d.groupby(["conv_id", "speaker"]).loglat.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    d["logch"] = np.log1p(d.total_chars)
    d["zch"] = d.groupby(["conv_id", "speaker"]).logch.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    g = d.groupby(["conv_id", "session"])
    for c in ("zlat", "lat", "zch", "has_q", "total_chars", "D_engagement", "n_msgs", "ack"):
        d["prev_" + c] = g[c].shift(1)
        d["nxt_" + c] = g[c].shift(-1)
    x = d[d.lat.notna()]
    r = {"n": len(x)}
    if corpus == "whatsapp_nl":
        bins = [-1, 59, 179, 599, 3599, 1e9]; labels = ["<1min", "1-2min", "3-9min", "10-59min", "1-3h"]
    else:
        bins = [-1, 5, 15, 45, 120, 1e9]; labels = ["<5s", "5-15s", "15-45s", "45-120s", ">120s"]
    x["latbin"] = pd.cut(x.lat, bins, labels=labels)
    tb = x.groupby("latbin").agg(n=("lat", "size"), chars_med=("total_chars", "median"), n_msgs_mean=("n_msgs", "mean"),
                                 p_q=("has_q", "mean"), p_ack=("ack", "mean"), eng=("D_engagement", "mean"),
                                 p_parceiro_continua=("is_last", lambda s: 1 - s.mean()),
                                 nxt_lat_med=("nxt_lat", "median")).round(3)
    print(corpus, "\n", tb.to_string()); r["by_latbin"] = tb.reset_index().astype(str).to_dict("records")
    # dentro da pessoa: latência (z) × tamanho (z)
    xx = x.dropna(subset=["zlat", "zch"])
    rho = stats.spearmanr(xx.zlat, xx.zch); r["within_rho_zlat_zchars"] = [round(rho[0], 3), float(rho[1])]
    # espelhamento de ritmo: minha latência ~ latência do outro antes
    yy = x.dropna(subset=["zlat", "prev_zlat"])
    rho2 = stats.spearmanr(yy.prev_zlat, yy.zlat); r["rho_partner_prevlat_mylat"] = [round(rho2[0], 3), float(rho2[1])]
    # quando o outro demorou (prev_zlat > 1) o que muda no meu turno seguinte (o turno do outro é o anterior)
    x["other_slow"] = x.prev_zlat > 1
    x["other_fast"] = x.prev_zlat < -0.5
    comp = x.dropna(subset=["prev_zlat"]).groupby(pd.cut(x.prev_zlat, [-9, -0.5, 0.5, 1, 9])).agg(
        n=("lat", "size"), my_lat_med=("lat", "median"), my_zlat=("zlat", "mean"), my_zch=("zch", "mean"), my_p_q=("has_q", "mean"),
        my_eng=("D_engagement", "mean"), my_ack=("ack", "mean")).round(3)
    print(comp.to_string()); r["after_partner_latency"] = comp.reset_index().astype(str).to_dict("records")
    # latência do parceiro após pergunta vs sem pergunta (pergunta puxa resposta rápida?)
    z = d.dropna(subset=["nxt_lat"])
    r["partner_lat_med_after_q"] = float(z[z.has_q].nxt_lat.median()); r["partner_lat_med_after_noq"] = float(z[~z.has_q].nxt_lat.median())
    r["partner_p_fast_after_q"] = float((z[z.has_q].nxt_lat < (60 if corpus == "whatsapp_nl" else 15)).mean())
    r["partner_p_fast_after_noq"] = float((z[~z.has_q].nxt_lat < (60 if corpus == "whatsapp_nl" else 15)).mean())
    # latência × engajamento Jev (mesmo turno)
    j = x.dropna(subset=["D_engagement", "zlat"])
    if len(j) > 50:
        r["rho_zlat_engagement"] = [round(v, 4) for v in stats.spearmanr(j.zlat, j.D_engagement)]
        m = smf.ols("D_engagement ~ zlat + logch + C(conv_id)", data=j).fit()
        r["ols_eng_on_zlat_ctrl_len"] = {"b": round(m.params["zlat"], 3), "p": float(m.pvalues["zlat"]), "n": int(m.nobs)}
    # latência crescente prevê fim? últimos turnos da sessão vs outros
    if corpus == "whatsapp_nl":
        d["tl"] = d.turns_left
        e = d[d.lat.notna()]
        r["zlat_by_turns_left"] = e.groupby(pd.cut(e.tl, [-1, 0, 1, 3, 10, 1e9])).zlat.agg(["size", "mean"]).round(3).reset_index().astype(str).to_dict("records")
        print("zlat por turns_left", r["zlat_by_turns_left"])
    print(json.dumps({k: v for k, v in r.items() if k not in ("by_latbin", "after_partner_latency", "zlat_by_turns_left")}, indent=1))
    res[corpus] = r
json.dump(res, open(f"{OUT}/a5_latency.json", "w"), indent=1, default=str)
