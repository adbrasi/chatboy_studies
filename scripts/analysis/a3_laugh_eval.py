"""a3: função das risadas (Jev) — reação x social/suavizador; quem ri; o que acontece depois.
Entrada: SCR/laugh_raw.jsonl + D.pkl (do a3_seq_analysis). Saída: analysis/data/a3_laugh.json"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from a3_common import *
R = {}
rows = [json.loads(l) for l in open(os.path.join(SCR, "laugh_raw.jsonl"))]
D = pd.read_pickle(os.path.join(SCR, "D.pkl")).set_index(["corpus", "conv_id", "turn_idx"])
L = []
for d in rows:
    r = d["r"]; key = (d["corpus"], d["conv_id"], d["turn_idx"])
    x = D.loc[key]
    rec = {"grp": d["grp"], "corpus": d["corpus"], "face_threat": r["face_threat"]["noul"], "text": x.text,
           "prev_text": x.prev_text, "next_text": x.next_text, "next_laugh": x.next_laugh, "next_playful": x.next_D_playful,
           "prev_joke": x.prev_joke, "prev_laugh": x.prev_laugh, "total_chars": x.total_chars, "D_intent": x.D_intent,
           "D_tension": x.D_tension, "next_valence": x.next_D_valence, "valence": x.D_valence}
    if d["grp"] == "laugh":
        rec |= {"function": r["function"]["choice"], "func_conf": r["function"]["confidence"],
                "func_probs": r["function"]["probabilities"],
                "prev_funny": r["prev_funny"]["noul"], "cur_joke": r["cur_joke"]["noul"], "softening": r["softening"]["noul"]}
    L.append(rec)
L = pd.DataFrame(L)
A = L[L.grp == "laugh"].copy(); C = L[L.grp == "control"]
R["n"] = {"laugh": len(A), "control": len(C), "by_corpus": A.corpus.value_counts().to_dict()}
R["function_share"] = {c: g.function.value_counts(normalize=True).round(3).to_dict() for c, g in list(A.groupby("corpus")) + [("all", A)]}
R["function_share_conf>=.6"] = {c: g[g.func_conf >= .6].function.value_counts(normalize=True).round(3).to_dict() for c, g in list(A.groupby("corpus")) + [("all", A)]}
# soma de probabilidades (soft share)
R["function_soft_share"] = {c: pd.DataFrame(g.func_probs.tolist()).mean().round(3).to_dict() for c, g in list(A.groupby("corpus")) + [("all", A)]}
# agregados: reação a humor alheio x risada 'sobre si' (piada/história própria) x social (suavizador, nervoso, backchannel)
grp = {"reaction_to_humor": "reação", "own_joke_marker": "próprio humor", "amused_own_story": "próprio humor",
       "softener": "social", "nervous_awkward": "social", "friendly_backchannel": "social"}
A["macro"] = A.function.map(grp)
R["macro_share"] = {c: g.macro.value_counts(normalize=True).round(3).to_dict() for c, g in list(A.groupby("corpus")) + [("all", A)]}
R["noul_rates"] = {c: {"prev_funny>=.5": round((g.prev_funny >= .5).mean(), 3), "cur_joke>=.5": round((g.cur_joke >= .5).mean(), 3),
                       "softening>=.5": round((g.softening >= .5).mean(), 3), "face_threat>=.5": round((g.face_threat >= .5).mean(), 3),
                       "no_prev_funny_and_no_cur_joke": round(((g.prev_funny < .5) & (g.cur_joke < .5)).mean(), 3)}
                   for c, g in list(A.groupby("corpus")) + [("all", A)]}
R["control_face_threat"] = {c: {"n": len(g), "face_threat>=.5": round((g.face_threat >= .5).mean(), 3)} for c, g in list(C.groupby("corpus")) + [("all", C)]}
# o que vem depois de cada função
R["after_by_function"] = A.groupby("function").agg(n=("text", "size"), partner_laughs=("next_laugh", "mean"),
                                                   partner_playful=("next_playful", lambda s: (s >= .5).mean()),
                                                   chars_med=("total_chars", "median")).round(3).to_dict("index")
# exemplos (preferir inglês/maichat e curtos)
ex = {}
for f, g in A.groupby("function"):
    g = g.sort_values("func_conf", ascending=False)
    pick = pd.concat([g[g.corpus == "maichat"].head(3), g[g.corpus == "whatsapp_nl"].head(2)])
    ex[f] = [f"[{r.corpus}] prev: {str(r.prev_text)[:90]} || laugh turn: {r.text[:110]}" for r in pick.itertuples()]
R["examples"] = ex
json.dump(R, open(os.path.join(OUT, "a3_laugh.json"), "w"), indent=1, ensure_ascii=False, default=str)
for k in ("n", "function_share", "function_share_conf>=.6", "function_soft_share", "macro_share", "noul_rates", "control_face_threat"):
    print(k, json.dumps(R[k], ensure_ascii=False))
print(pd.DataFrame(R["after_by_function"]))
print(json.dumps(ex, indent=0, ensure_ascii=False))
