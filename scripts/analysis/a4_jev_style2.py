"""a4 §5b — follow-up: can Jev tell a *casual-sounding* LLM reply from the real one if asked differently?
Reuses the 60 cases/candidates of a4_jev_style.py (analysis/data/a4_jev_style_cases.csv).
X's style samples are re-drawn deterministically (same pool: X's messages in the chat, >3 turns away from T).
  Nouls (real / llm_plain / llm_styled): tryhard, more_markers, ai_imitating  + Score typicality 0-4
  Choice with explicit framing ("one of them is an AI imitating X"), both orders, vs llm_plain / llm_styled
Code baselines: 'marker excess' = (laugh+emoji+slang markers per bubble in candidate) - (same in X's samples).
Output: analysis/data/a4_jev_style2.json"""
import json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from jev import ask_many, choice, noul, score, summary
from a4_common import ADATA
from a4_style_feats import feats
import a4_jev_style as J

Q = {
    "tryhard": noul("X_reply tries harder to be witty, funny or clever than X's usual messages in style_samples_of_X."),
    "more_markers": noul("X_reply uses more slang, abbreviations, emoji or laughter than X usually does in style_samples_of_X."),
    "ai_imitating": noul("X_reply sounds like an AI imitating a young person texting (forced or trendy slang, performative jokes) rather than X's own plain way of texting."),
    "typical": score("How typical is X_reply of the way X writes in style_samples_of_X (wording, tone, effort, style)?",
                     ["not at all like X", "a little like X", "somewhat like X", "very much like X", "exactly like X"]),
}
Q_PAIR = choice("One candidate reply is the genuine message X sent; the other was written by an AI imitating X. "
                "Which one is the genuine message from X? Use X's earlier messages (style_samples_of_X) and the conversation.",
                {"A": "candidate A is X's genuine message", "B": "candidate B is X's genuine message"})
MARK = re.compile(r"\b(?:lol|lmao|haha\w*|omg|fr|frfr|bro|bestie|ngl|tbh|smh|wtf|lowkey|deadass|slay|literally|u|ur|rn|tho|omgg+)\b|[\U0001F300-\U0001FAFF]", re.I)


def marker_rate(texts):
    b = [x for x in texts if x.strip()]
    return sum(len(MARK.findall(x)) for x in b) / max(1, len(b))


def main():
    C = pd.read_csv(f"{ADATA}/a4_jev_style_cases.csv")
    cases = J.build_cases()
    assert [(c["conv_id"], c["turn_i"]) for c in cases] == list(zip(C.conv_id, C.turn_i))
    for c, (_, r) in zip(cases, C.iterrows()):
        for k in ["llm_plain", "llm_styled"]:
            c[k] = r[k] if isinstance(r[k], str) else ""
    jobs, keys = [], []
    for k, c in enumerate(cases):
        base = {"style_samples_of_X": c["samples"], "conversation_so_far": c["context"]}
        for cand in ["real", "llm_plain", "llm_styled"]:
            jobs.append((json.dumps(dict(base, X_reply=c[cand]), ensure_ascii=False), Q)); keys.append(("noul", k, cand, None))
        for other in ["llm_plain", "llm_styled"]:
            for order in (0, 1):
                a, b = (c["real"], c[other]) if order == 0 else (c[other], c["real"])
                jobs.append((json.dumps(dict(base, candidate_replies={"A": a, "B": b}), ensure_ascii=False), {"which": Q_PAIR}))
                keys.append(("pair", k, other, order))
    print("jobs", len(jobs))
    ans = ask_many(jobs, workers=4)
    rows = []
    for (kind, k, cand, order), a in zip(keys, ans):
        if a is None: continue
        if kind == "pair":
            pr = a["which"]["probabilities"]
            rows.append({"kind": kind, "case": k, "cand": cand, "order": order, "p_real": pr.get("A", 0) if order == 0 else pr.get("B", 0)})
        else:
            rows.append({"kind": kind, "case": k, "cand": cand, **{q: (a[q]["noul"] if q != "typical" else a[q]["score"]) for q in Q}})
    R = pd.DataFrame(rows)
    res = {"jev": summary(), "noul": {}, "pair": {}, "code_marker_excess": {}}
    nl = R[R.kind == "noul"]
    for q in Q:
        real = nl[nl.cand == "real"].set_index("case")[q]
        res["noul"][q] = {"real_mean": round(float(real.mean()), 3)}
        for cand in ["llm_plain", "llm_styled"]:
            s = nl[nl.cand == cand].set_index("case")[q]
            # for 'typical' the real one should be HIGHER; for the others the LLM should be higher
            a = J.auc(real.values, s.values) if q == "typical" else J.auc(s.values, real.values)
            res["noul"][q][cand] = {"mean": round(float(s.mean()), 3), "auc_detect": round(a, 3)}
    pr = R[R.kind == "pair"]
    for cand, g in pr.groupby("cand"):
        m = g.groupby("case").p_real.mean()
        res["pair"][cand] = {"acc": round(float((m > .5).mean()), 3), "mean_p_real": round(float(m.mean()), 3)}
    for cand in ["llm_plain", "llm_styled"]:
        ex_r = [marker_rate(c["real"].split("\n")) - marker_rate(c["samples"]) for c in cases]
        ex_l = [marker_rate(c[cand].split("\n")) - marker_rate(c["samples"]) for c in cases]
        res["code_marker_excess"][cand] = {"auc_detect": round(J.auc(ex_l, ex_r), 3),
                                           "mean_excess_llm": round(float(np.mean(ex_l)), 3),
                                           "mean_excess_real": round(float(np.mean(ex_r)), 3)}
    # length ratio vs samples (LLM plain tends to be longer)
    for cand in ["real", "llm_plain", "llm_styled"]:
        res.setdefault("len_ratio_vs_samples_median", {})[cand] = round(float(np.median(
            [len(c[cand]) / max(1, np.mean([len(x) for x in c["samples"]])) for c in cases])), 2)
    json.dump(res, open(f"{ADATA}/a4_jev_style2.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
