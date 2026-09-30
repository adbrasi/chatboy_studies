"""a8 step 8: what drives Jev's 'sounds human' judgements? (single noul + pairwise), vs code features.
Output: analysis/data/a8_judge_bias.json"""
import json
import numpy as np
import a8_common as C
from a8_compare import load_all, prev_user
from a8_eval import load_jev

X, R = load_all()
ids = [c["id"] for c in X]
prevs = {c["id"]: prev_user(c) for c in X}
out = {}
# single noul 'human': correlation with code features pooled over all conditions
rows = []
for cn in ["human", "base_gemini", "base_gpt4omini", "base_llama70b", "styled_gemini"]:
    J = load_jev(cn)
    for i in ids:
        f = C.style_feats(R[cn][i], prevs[i])
        rows.append({"cond": cn, "p_human": J[i]["human"], "is_human": cn == "human", "log_chars": np.log1p(f["chars"]), "emoji": f["emoji"],
                     "laugh": f["laugh"], "lower": f["all_lower"], "excl": f["any_excl"], "ends_q": f["ends_q"], "le3": f["le3_words"],
                     "slangy": f["tic:lol/lmao"] or f["tic:omg"] or f["llm:💀/😭 (gíria emoji)"] or f["tic:u/ur/r (abreviação)"]})
import pandas as pd
df = pd.DataFrame(rows)
corr = {k: round(float(np.corrcoef(df["p_human"], df[k].astype(float))[0, 1]), 3) for k in ["log_chars", "emoji", "laugh", "lower", "excl", "ends_q", "le3", "slangy", "is_human"]}
out["single_p_human_corr"] = corr
out["single_p_human_by_cond_and_emoji"] = df.groupby(["cond", "emoji"])["p_human"].mean().round(3).reset_index().values.tolist()
out["single_p_human_human_le3"] = df[df.cond == "human"].groupby("le3")["p_human"].agg(["mean", "count"]).round(3).reset_index().values.tolist()
# pairwise: when Jev chose LLM, what differed
pr = []
for cn in ["base_gemini", "base_gpt4omini", "styled_gemini", "bestof3_jev"]:
    P = json.load(open(f"{C.SCR}/jev_pair_{cn}.json"))
    for i, v in P.items():
        fh, fl = C.style_feats(R["human"][i]), C.style_feats(R[cn][i])
        pr.append({"cond": cn, "correct": v["ans"]["which_human"]["choice"] == v["human_is"],
                   "llm_longer": fl["chars"] > fh["chars"], "llm_emoji_only": fl["emoji"] and not fh["emoji"],
                   "llm_laugh_only": fl["laugh"] and not fh["laugh"], "human_le3": fh["le3_words"]})
pdf = pd.DataFrame(pr)
out["pair_acc_by"] = {k: pdf.groupby(["cond", k])["correct"].agg(["mean", "count"]).round(3).reset_index().values.tolist()
                      for k in ["llm_longer", "llm_emoji_only", "llm_laugh_only", "human_le3"]}
C.save("a8_judge_bias.json", out)
print(json.dumps(out, indent=0, ensure_ascii=False)[:4000])
