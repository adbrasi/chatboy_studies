"""a5 — checagem de validade do Noul 'callback': concorda com um proxy lexical (palavras de conteúdo do turno que
aparecem no trecho anterior mas não nos 3 turnos recentes)?"""
import json, re
import numpy as np, pandas as pd
from a5_common import load_all, auc, OUT
df = load_all().sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
mv = pd.read_json(f"{OUT}/a5_jev_moves.jsonl", lines=True)
W = re.compile(r"[a-zà-ÿ]{4,}")
STOP = set("that this with have what just like really know yeah think about would there they been were your from will when then them also some want good well need much very even into more only here dont cant didnt thats haha hahaha maar niet heel echt goed wel even naar zijn heeft hebben kunnen moet gaan komt komen toch want doen weet voor niks iets alles morgen vandaag", "".split()) if False else set("""that this with have what just like really know yeah think about would there they been were your from will when then them also some want good well need much very even into more only here dont cant didnt thats haha hahaha hahah hahahaha maar niet heel echt goed wel even naar zijn heeft hebben kunnen moet gaan komt komen toch want doen weet voor niks iets alles morgen vandaag okay oke okee""".split())
key = df.set_index(["corpus", "conv_id"])
rows = []
for c, cid in mv[["corpus", "conv_id"]].drop_duplicates().values:
    conv = df[(df.corpus == c) & (df.conv_id == cid)].set_index("turn_idx")
    for _, r in mv[(mv.corpus == c) & (mv.conv_id == cid)].iterrows():
        ti = r.turn_idx
        prev = conv.loc[:ti - 1]
        if c == "maichat": prev = prev[prev.session == conv.loc[ti].session]
        rec, ear = prev.tail(3), prev.iloc[:-3].tail(40)
        cur = set(W.findall(conv.loc[ti].text.lower())) - STOP
        e = set(W.findall(" ".join(ear.text).lower())) - STOP
        rr = set(W.findall(" ".join(rec.text).lower())) - STOP
        rows.append({"corpus": c, "callback": r.callback, "overlap": len((cur & e) - rr), "cur_n": len(cur)})
x = pd.DataFrame(rows)
res = {}
for c, d in list(x.groupby("corpus")) + [("all", x)]:
    y = d.callback > .5
    res[c] = {"n": len(d), "AUC_overlap_for_jev_callback": round(auc(y, d.overlap), 3),
              "p_overlap>=1_if_cb": round((d[y].overlap >= 1).mean(), 3), "p_overlap>=1_if_not": round((d[~y].overlap >= 1).mean(), 3)}
print(res)
json.dump(res, open(f"{OUT}/a5_callback_check.json", "w"), indent=1)
