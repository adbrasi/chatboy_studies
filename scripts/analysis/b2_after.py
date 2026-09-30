"""b2 ritmo: a bolha extra "ah, e…" (whatsapp_nl): depois da rajada inicial, a MESMA pessoa manda outra bolha >= 2 min
depois, antes de o outro responder? Amostra por split: 120 positivos + 240 negativos (turnos com >= 2 de contexto).

 F1 bare   : 1 chamada, Noul direto.
 F2 guide  : 1 chamada, state + GUIA do "ah, e…" (taxas medidas) + Noul direto + 7 Nouls atômicos (fio aberto,
             faltou dizer algo, pergunta pendente do próprio falante, espera resposta, fechamento, logística pendente,
             conversa esfriando). Código: logística no dev com (Jev atômicos) e (Jev + código).
 código    : termina em pergunta, tamanho, taxa própria de "ah, e…", latência do parceiro, hora do dia.
Saída: analysis/data/b2_after.json + scratchpad/b2/after_raw.pkl"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from b2_nb_common import *
from b2_common import boot_ci, dump
from jev import ask_many, noul, summary

T = full_T()
T = T[T.corpus == "whatsapp_nl"]
by = {c: g.reset_index(drop=True) for c, g in T.groupby("conv_id")}
pool = T[T.turn_in_session >= 2]
parts = []
for s in ("dev", "test"):
    p = pool[pool.split == s]
    parts += [p[p.afterthought].sample(120, random_state=1), p[~p.afterthought].sample(240, random_state=1)]
S = pd.concat(parts)
C = "the speaker of `current_turn`"
GUIDE_A = {"what_this_is": "Sometimes, after sending a message, a person sends one more message minutes later before the other replies ('oh and…', 'or later?', 'good night xx').",
           "how_often": "In about 6% of turns; typically 3-40 minutes later (median 10 minutes).",
           "what_it_usually_is": "A question in about 30% of cases (re-opening the conversation), an extra detail, an afterthought, or a sweet add-on after a goodbye.",
           "more_likely_when": ["the turn did not end with a question, so nothing forces a reply", "a practical matter is still being arranged",
                                "the speaker is eager for a reply that is not coming"],
           "less_likely_when": ["the turn asks the other a question and waits for the answer", "the conversation has clearly ended"]}
Q1 = {"after": noul(f"After sending `current_turn`, will {C} send one more message a few minutes later, before the other person replies?")}
Q2 = Q1 | {
    "open_thread": noul(f"Does `current_turn` leave something open or unfinished that {C} might add to?"),
    "forgot": noul(f"Is there something {C} would naturally still want to add or ask after `current_turn`?"),
    "asks_q": noul("Does `current_turn` ask the other person a question?"),
    "waiting": noul(f"Is {C} now waiting for the other person's answer or decision?"),
    "closure": noul("Does `current_turn` close the conversation (goodbye, good night, see you)?"),
    "logistics": noul("Is a practical matter (time, place, plans) still being arranged?"),
    "cooling": noul("Is the conversation cooling down or winding down?"),
}
rows, it1, it2 = [], [], []
for t in S.itertuples():
    conv = by[t.conv_id]; i = int(t.pos)
    ctx, _ = context(conv, i)
    cur = {"speaker": t.speaker, "messages": [str(x).strip()[:300] for x in t.y_texts]}
    st = {"conversation_so_far": ctx, "current_turn": cur}
    it1.append((st, Q1)); it2.append((st | {"afterthought_guide": GUIDE_A}, Q2))
    past = conv.iloc[:i]; own = past[past.speaker == t.speaker]; oth = past[past.speaker != t.speaker]
    last = str(t.y_texts[-1])
    rows.append({"conv_id": t.conv_id, "split": t.split, "y": int(t.afterthought),
                 "c_q": int("?" in last), "c_lchars": np.log1p(sum(len(str(x)) for x in t.y_texts)),
                 "c_own_rate": (own.afterthought.sum() + 0.06 * 5) / (len(own) + 5),
                 "c_partner_lat": np.log1p(np.nanmedian(oth.response_latency_s.values) if oth.response_latency_s.notna().any() else 60),
                 "c_night": int(str(t.ts_start)[11:13] in ("22", "23", "00", "01")), "c_nmsgs": t.y})
R = pd.DataFrame(rows)
print("calls", len(it1) + len(it2), flush=True)
A1 = ask_many(it1, workers=4)
A2 = ask_many(it2, workers=4)
R["F1"] = [a["after"]["noul"] if a else np.nan for a in A1]
for k in Q2:
    R["j_" + k] = [a[k]["noul"] if a else np.nan for a in A2]
R = R.dropna()
pd.to_pickle(R, os.path.join(SCR, "after_raw.pkl"))
code = ["c_q", "c_lchars", "c_own_rate", "c_partner_lat", "c_night", "c_nmsgs"]
jev = ["j_" + k for k in Q2]
dev, test = R[R.split == "dev"], R[R.split == "test"]
res = {"n_dev": len(dev), "n_test": len(test), "pos_rate_sample": float(R.y.mean())}
preds = {"F1_bare_noul": test.F1, "F2_guide_noul": test.j_after}
for name, cols in (("code_logit", code), ("F2_atomic_logit", jev), ("code+F2_logit", code + jev)):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(dev[cols], dev.y)
    preds[name] = pd.Series(m.predict_proba(test[cols])[:, 1], index=test.index)
    res[name + "_dev_auc"] = round(float(roc_auc_score(dev.y, m.predict_proba(dev[cols])[:, 1])), 3)
for name, p in preds.items():
    d = test.assign(p=p.values)
    pt, lo, hi = boot_ci(d, lambda x: roc_auc_score(x.y, x.p) if x.y.nunique() > 1 else np.nan)
    res[name] = {"auc_test": round(pt, 3), "ci": [round(lo, 3), round(hi, 3)]}
res["univariate_auc_test"] = {c: round(float(roc_auc_score(test.y, test[c])), 3) for c in code + jev}
res["mean_F1_pos_neg"] = [round(float(test[test.y == 1].F1.mean()), 3), round(float(test[test.y == 0].F1.mean()), 3)]
dump(res, "b2_after.json")
print(json.dumps(res, indent=1)); print(summary())
