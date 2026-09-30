"""a5 — Validação de P.p_question, P.p_topic_shift, P.p_end contra o que aconteceu (AUC, acurácia × baseline)."""
import json, warnings
import numpy as np, pandas as pd
from a5_common import load_all, auc, OUT
warnings.filterwarnings("ignore")
df = load_all()
d = df[df.P_p_question.notna()].copy()
full = df.sort_values(["corpus", "conv_id", "turn_idx"])
g = full.groupby(["corpus", "conv_id"])
full["prev_farewell"] = g.farewell.shift(1).fillna(False).astype(bool)
full["prev_ack"] = g.ack.shift(1).fillna(False).astype(bool)
full["prev_hasq"] = g.has_q.shift(1).fillna(False).astype(bool)
# taxa de pergunta do falante até aqui (sem ver T)
full["spk_qrate"] = full.groupby(["corpus", "conv_id", "speaker"]).has_q.transform(lambda s: s.shift(1).expanding().mean()).fillna(0.2)
full["conv_last"] = full.turn_idx == g.turn_idx.transform("max")
d = d.merge(full[["corpus", "conv_id", "turn_idx", "prev_farewell", "prev_ack", "prev_hasq", "spk_qrate", "conv_last"]], on=["corpus", "conv_id", "turn_idx"])
res = {}
def evalp(y, p, base_scores, name):
    y = np.asarray(y).astype(bool); p = np.asarray(p, float)
    o = {"n": int(len(y)), "base_rate": round(y.mean(), 3), "mean_pred": round(p.mean(), 3), "AUC_jev": round(auc(y, p), 3),
         "acc_jev@0.5": round(((p >= .5) == y).mean(), 3), "acc_majority": round(max(y.mean(), 1 - y.mean()), 3),
         "brier_jev": round(((p - y) ** 2).mean(), 4), "brier_const": round(((y.mean() - y) ** 2).mean(), 4)}
    # melhor limiar (F1)
    best = max(((t, 2 * ((p >= t) & y).sum() / max(1, (p >= t).sum() + y.sum())) for t in np.arange(.05, .95, .05)), key=lambda z: z[1])
    o["best_F1_thr"] = [round(best[0], 2), round(best[1], 3)]
    for k, s in base_scores.items():
        o["AUC_" + k] = round(auc(y, s), 3)
    return o
for corpus in ("maichat", "whatsapp_nl", "all"):
    x = d if corpus == "all" else d[d.corpus == corpus]
    r = {}
    r["p_question"] = evalp(x.has_q, x.P_p_question, {"spk_qrate": x.spk_qrate, "prev_not_q": 1 - x.prev_hasq.astype(float)}, "q")
    r["p_topic_shift_vs_D"] = evalp(x.D_topic_shift > .5, x.P_p_topic_shift, {"prev_ack": x.prev_ack.astype(float)}, "ts")
    end_any = x.farewell | x.is_last | (x.D_intent == "closing")
    r["p_end_vs_farewell_or_last_or_closing"] = evalp(end_any, x.P_p_end, {"prev_farewell": x.prev_farewell.astype(float), "pos": x.turn_in_session}, "end")
    r["p_end_vs_farewell"] = evalp(x.farewell, x.P_p_end, {"prev_farewell": x.prev_farewell.astype(float)}, "end")
    if corpus != "maichat":
        r["p_end_vs_session_last"] = evalp(x.is_last, x.P_p_end, {"prev_farewell": x.prev_farewell.astype(float)}, "end")
        r["p_end_vs_turns_left_le2"] = evalp(x.turns_left <= 2, x.P_p_end, {"prev_farewell": x.prev_farewell.astype(float)}, "end")
    res[corpus] = r
    print(corpus); [print(" ", k, v) for k, v in r.items()]
# p_topic_shift vs rótulo novo de transição (a5_jev_moves)
try:
    mv = pd.read_json(f"{OUT}/a5_jev_moves.jsonl", lines=True)
    z = d.merge(mv, on=["corpus", "conv_id", "turn_idx"])
    y = z.transition.isin(["smooth_shift", "abrupt_shift", "returns_earlier"])
    res["p_topic_shift_vs_moves_transition"] = evalp(y, z.P_p_topic_shift, {"prev_ack": z.prev_ack.astype(float)}, "ts")
    res["p_topic_shift_vs_abrupt"] = evalp(z.transition == "abrupt_shift", z.P_p_topic_shift, {}, "ts")
    print(res["p_topic_shift_vs_moves_transition"], res["p_topic_shift_vs_abrupt"])
except Exception as e:
    print("moves not ready", e)
# calibração p_end em faixas
bins = pd.cut(d.P_p_end, [0, .05, .1, .2, .4, 1], include_lowest=True)
cal = d.assign(end=(d.farewell | d.is_last | (d.D_intent == "closing"))).groupby(bins).agg(n=("end", "size"), obs=("end", "mean"), pred=("P_p_end", "mean")).round(3)
print(cal.to_string()); res["p_end_calibration"] = cal.reset_index().astype(str).to_dict("records")
bins = pd.cut(d.P_p_question, [0, .1, .2, .4, .6, 1], include_lowest=True)
cal = d.groupby(bins).agg(n=("has_q", "size"), obs=("has_q", "mean"), pred=("P_p_question", "mean")).round(3)
print(cal.to_string()); res["p_question_calibration"] = cal.reset_index().astype(str).to_dict("records")
json.dump(res, open(f"{OUT}/a5_validate_P.json", "w"), indent=1, default=str)
