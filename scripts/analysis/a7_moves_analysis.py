"""a7: análise dos movimentos de flerte/afeto (a7_moves.jsonl): inventário, respostas, intensidade, o que mantém vivo.
Saída: analysis/data/a7_moves_summary.json
"""
import json, os, re, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_common import load, OUT

EMO = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
W = re.compile(r"[a-zà-ÿ']+|<3|[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", re.I)


def overlap(a, b):
    A, B = set(w.lower() for w in W.findall(a)), set(w.lower() for w in W.findall(b))
    return len(A & B) / max(1, len(B))


def main():
    df = load().sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
    key = df.set_index(["corpus", "conv_id", "turn_idx"])
    mv = pd.read_json(os.path.join(OUT, "a7_moves.jsonl"), lines=True)
    mv["keep"] = (mv.set == "base") | (mv.is_flirt >= .5)
    m = mv[mv.keep & (mv.move != "none")].copy()
    out = {"n_all": len(mv), "n_expand_flirt": int(((mv.set == "expand") & (mv.is_flirt >= .5)).sum()),
           "n_expand": int((mv.set == "expand").sum()), "n_used": len(m), "none_in_base": int(((mv.set == "base") & (mv.move == "none")).sum())}
    # atributos do R e do T+2
    rows = []
    for _, r in m.iterrows():
        T = key.loc[(r.corpus, r.conv_id, r.turn_idx)]
        R = key.loc[(r.corpus, r.conv_id, r.turn_idx + 1)]
        nx = key.loc[(r.corpus, r.conv_id, r.turn_idx + 2)] if (r.corpus, r.conv_id, r.turn_idx + 2) in key.index else None
        same = nx is not None and nx.session == T.session and nx.speaker == T.speaker
        rows.append({"R_chars": R.total_chars, "T_chars": T.total_chars, "R_msgs": R.n_msgs, "R_lat": R.response_latency_s,
                     "R_emoji": EMO.search(" ".join(R.texts)) is not None, "T_emoji": EMO.search(" ".join(T.texts)) is not None,
                     "R_laugh": bool(R.laugh), "R_q": bool(R.has_q), "overlap": overlap(" ".join(T.texts), " ".join(R.texts)),
                     "same_emoji": bool(set(EMO.findall(" ".join(T.texts))) & set(EMO.findall(" ".join(R.texts)))),
                     "too": bool(re.search(r"\b(too|also|ook|jij ook|you too|u too)\b", " ".join(R.texts), re.I)),
                     "R_last": bool(R.is_last),
                     "T2_flirt": (float(nx.D_flirting >= .5 or nx.D_emotion == "affection") if same and nx.annotated else np.nan),
                     "T2_exists": bool(same), "T2_chars": nx.total_chars if same else np.nan,
                     "sess_left_after_R": R.turns_left})
    m = pd.concat([m.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    m.to_csv(os.path.join(OUT, "a7_moves_enriched.csv"), index=False)
    out["move_dist"] = {c: m[m.corpus == c].move.value_counts().to_dict() for c in ["maichat", "whatsapp_nl"]}
    out["resp_dist"] = {c: m[m.corpus == c].resp.value_counts().to_dict() for c in ["maichat", "whatsapp_nl"]}
    ct = pd.crosstab(m.move, m.resp)
    out["move_x_resp"] = {mvk: row[row > 0].sort_values(ascending=False).to_dict() for mvk, row in ct.iterrows()}
    # resposta por grupo de movimento
    agg = m.groupby("move").agg(n=("resp", "size"), R_chars_med=("R_chars", "median"), T_chars_med=("T_chars", "median"),
                                R_emoji=("R_emoji", "mean"), R_laugh=("R_laugh", "mean"), R_q=("R_q", "mean"),
                                overlap=("overlap", "mean"), too=("too", "mean"), alive=("alive", "mean"),
                                r_int=("r_int", "mean"), t_int=("t_int", "mean"), T2_flirt=("T2_flirt", "mean"),
                                R_last=("R_last", "mean")).round(3)
    out["by_move"] = agg.reset_index().to_dict("records")
    ragg = m.groupby("resp").agg(n=("move", "size"), R_chars_med=("R_chars", "median"), alive=("alive", "mean"),
                                 T2_flirt=("T2_flirt", "mean"), n_T2=("T2_flirt", "count"), T2_exists=("T2_exists", "mean"),
                                 R_last=("R_last", "mean"), r_minus_t=("r_int", lambda x: 0),
                                 lat_med=("R_lat", "median")).round(3)
    ragg["r_minus_t"] = m.assign(d=m.r_int - m.t_int).groupby("resp").d.mean().round(2)
    out["by_resp"] = ragg.reset_index().to_dict("records")
    # intensidade: resposta x provocação
    d = (m.r_int - m.t_int)
    out["intensity"] = {"spearman": float(spearmanr(m.t_int, m.r_int)[0]), "delta_mean": float(d.mean()),
                        "share_R_above_T": float((d > .5).mean()), "share_R_above_T_plus1": float((d > 1).mean()),
                        "share_R_below_T_minus1": float((d < -1).mean()), "share_within_1": float((d.abs() <= 1).mean()),
                        "by_t_level": m.assign(tl=m.t_int.round()).groupby("tl").r_int.agg(["mean", "count"]).round(2).reset_index().to_dict("records")}
    # vivo: quem responde recíproco vs deflate/ignore
    out["alive_vs_intensity_gap"] = m.assign(g=pd.cut(d, [-5, -1.5, -.5, .5, 1.5, 5])).groupby("g", observed=True).agg(
        n=("alive", "size"), alive=("alive", "mean"), T2_flirt=("T2_flirt", "mean")).round(3).reset_index().astype(str).to_dict("records")
    # comparação com respostas a turnos neutros (baseline)
    a = df[df.annotated]
    neu = a[(a.D_flirting < .15) & (a.D_emotion != "affection") & (a.next_speaker.notna()) & (a.next_speaker != a.speaker) & (a.next_session == a.session)]
    base = {}
    for c in ["maichat", "whatsapp_nl"]:
        nb = neu[neu.corpus == c]
        mm = m[m.corpus == c]
        R_neu = [key.loc[(c, r.conv_id, r.turn_idx + 1)] for r in nb.itertuples()]
        chars_neu = np.array([x.total_chars for x in R_neu]); lat_neu = np.array([x.response_latency_s for x in R_neu], dtype=float)
        emo_neu = np.mean([EMO.search(" ".join(x.texts)) is not None for x in R_neu]); q_neu = np.mean([bool(x.has_q) for x in R_neu])
        lat_fl = mm.R_lat.dropna().values
        base[c] = {"n_fa": len(mm), "n_neu": len(nb), "R_chars_med_fa": float(mm.R_chars.median()), "R_chars_med_neu": float(np.median(chars_neu)),
                   "p_chars": float(mannwhitneyu(mm.R_chars, chars_neu).pvalue),
                   "R_lat_med_fa": float(np.median(lat_fl)), "R_lat_med_neu": float(np.nanmedian(lat_neu)),
                   "p_lat": float(mannwhitneyu(lat_fl, lat_neu[~np.isnan(lat_neu)]).pvalue),
                   "R_emoji_fa": float(mm.R_emoji.mean()), "R_emoji_neu": float(emo_neu), "R_q_fa": float(mm.R_q.mean()), "R_q_neu": float(q_neu),
                   "R_msgs_mean_fa": float(mm.R_msgs.mean()), "R_msgs_mean_neu": float(np.mean([x.n_msgs for x in R_neu])),
                   "T_chars_med": float(mm.T_chars.median()), "len_ratio_med": float((mm.R_chars / mm.T_chars.clip(lower=1)).median())}
    out["reply_vs_neutral"] = base
    # emoji espelhado: quando T tem emoji, R tem?
    out["emoji_mirror"] = {"P(R_emoji|T_emoji)": float(m[m.T_emoji].R_emoji.mean()), "n_T_emoji": int(m.T_emoji.sum()),
                           "P(R_emoji|no T_emoji)": float(m[~m.T_emoji].R_emoji.mean())}
    # exemplos por movimento e por resposta
    ex = {}
    for mvk, g in m.groupby("move"):
        g = g.sort_values("move_conf", ascending=False)
        ex[mvk] = [f"{r.T[:90]}  →  {r.R[:90]}  [{r.resp}]" for r in g.head(8).itertuples()]
    out["examples"] = ex
    exr = {}
    for rk, g in m.groupby("resp"):
        g = g.sort_values("resp_conf", ascending=False)
        exr[rk] = [f"{r.T[:80]}  →  {r.R[:80]}" for r in g.head(8).itertuples()]
    out["examples_resp"] = exr
    json.dump(out, open(os.path.join(OUT, "a7_moves_summary.json"), "w"), ensure_ascii=False, indent=1, default=str)
    for k in ["n_all", "n_expand_flirt", "n_expand", "n_used", "none_in_base", "move_dist", "resp_dist", "intensity", "alive_vs_intensity_gap",
              "reply_vs_neutral", "emoji_mirror"]:
        print(k, json.dumps(out[k], ensure_ascii=False, default=str))
    print(agg.to_string()); print(ragg.to_string())
    for k, v in out["move_x_resp"].items():
        print(k, v)


if __name__ == "__main__":
    main()
