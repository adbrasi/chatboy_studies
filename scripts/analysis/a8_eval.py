"""a8 step 7: AUC of Jev detectors and of code features (human vs LLM), overall, length-stratified, per model.
Also builds best-of-3 selection (styled + 2 extra samples) by the Jev gate score.
Output: analysis/data/a8_jev_auc.json ; scratch gen_bestof3_jev.json"""
import json, math, os, statistics as st
from collections import defaultdict
import numpy as np
import a8_common as C
from a8_compare import load_all, prev_user

DET = ["human", "formal", "forced", "generic", "direct", "register", "overvalidation", "paraphrase", "too_much", "invented", "length"]
# sign: +1 if higher => more human-like expected
SIGN = {"human": 1, "formal": -1, "forced": -1, "generic": -1, "direct": 1, "register": 1, "overvalidation": -1, "paraphrase": -1,
        "too_much": -1, "invented": -1, "length": -1}


def auc(pos, neg):
    """P(score_pos > score_neg) with ties 0.5."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    vals, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(pos, neg, n=500, seed=0):
    r = np.random.default_rng(seed)
    pos, neg = np.asarray(pos), np.asarray(neg)
    v = [auc(r.choice(pos, len(pos)), r.choice(neg, len(neg))) for _ in range(n)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def val(a):
    return a["noul"] if a["type"] == "noul" else a["score"]


def load_jev(cond):
    p = f"{C.SCR}/jev_single_{cond}.json"
    if not os.path.exists(p):
        return {}
    return {k: {q: val(a) for q, a in v.items()} for k, v in json.load(open(p)).items() if v}


def gate(d):
    """Naturalness gate: combined score (higher = more human). Weights fixed a priori (not fitted)."""
    return d["human"] + 0.5 * (d["register"] + d["direct"]) - 0.5 * (d["formal"] + d["too_much"] + d["overvalidation"] + d["forced"]) - 0.25 * max(0, d["length"] - 2)


def gate_v2(d):
    """Revised gate: only the detectors that separated human x base LLM in the expected direction
    (paraphrase, overvalidation, forced, too_much) + distance of the length judgement from 'about right'."""
    return -(d["paraphrase"] + d["overvalidation"] + d["forced"] + d["too_much"]) - 0.5 * abs(d["length"] - 2)


def strat_auc(pos, neg, pos_len, neg_len, nbins=4):
    """AUC averaged over pooled length-quartile strata (weights = n_pos*n_neg)."""
    L = np.log1p(np.concatenate([pos_len, neg_len]))
    qs = np.quantile(L, np.linspace(0, 1, nbins + 1))
    num = den = 0
    per = []
    for i in range(nbins):
        lo, hi = qs[i], qs[i + 1]
        pm = [(np.log1p(l) >= lo) & ((np.log1p(l) <= hi) if i == nbins - 1 else (np.log1p(l) < hi)) for l in pos_len]
        nm = [(np.log1p(l) >= lo) & ((np.log1p(l) <= hi) if i == nbins - 1 else (np.log1p(l) < hi)) for l in neg_len]
        p = [s for s, m in zip(pos, pm) if m]; n = [s for s, m in zip(neg, nm) if m]
        if len(p) >= 5 and len(n) >= 5:
            a = auc(p, n); w = len(p) * len(n)
            num += a * w; den += w; per.append([round(a, 3), len(p), len(n)])
    return (round(num / den, 3) if den else None), per


def main():
    X, R = load_all()
    ids = [c["id"] for c in X]
    prevs = {c["id"]: prev_user(c) for c in X}
    J = {cn: load_jev(cn) for cn in ["human", "base_gemini", "base_gpt4omini", "base_llama70b", "styled_gemini", "cand_gemini_s1", "cand_gemini_s2"]}
    # best-of-3 by gate
    S1 = json.load(open(f"{C.SCR}/gen_cand_gemini_s1.json")); S2 = json.load(open(f"{C.SCR}/gen_cand_gemini_s2.json"))
    best, picks = {}, defaultdict(int)
    for i in ids:
        cands = [("styled_gemini", R["styled_gemini"][i]), ("cand_gemini_s1", S1[i]), ("cand_gemini_s2", S2[i])]
        sc = [(gate_v2(J[cn][i]), cn, t) for cn, t in cands if i in J[cn]]
        g, cn, t = max(sc, key=lambda x: x[0])
        best[i] = t; picks[cn] += 1
    json.dump(best, open(f"{C.SCR}/gen_bestof3_jev.json", "w"), ensure_ascii=False)
    J["bestof3_jev"] = {}
    for i in ids:
        for cn, T in (("styled_gemini", R["styled_gemini"]), ("cand_gemini_s1", S1), ("cand_gemini_s2", S2)):
            if T[i] == best[i]:
                J["bestof3_jev"][i] = J[cn][i]; break
    R["bestof3_jev"] = best
    n_distinct = st.mean(len({R["styled_gemini"][i], S1[i], S2[i]}) for i in ids)

    feats = {cn: {i: C.style_feats(R[cn][i], prevs[i]) for i in ids if R[cn].get(i)} for cn in R}
    out = {"picks": picks, "mean_distinct_candidates": round(n_distinct, 2), "detectors": {}, "code_features": {}, "means": {}}
    LLM = ["base_gemini", "base_gpt4omini", "base_llama70b", "styled_gemini", "bestof3_jev"]
    # Jev detector means
    for cn in ["human"] + LLM:
        out["means"][cn] = {q: round(float(np.mean([J[cn][i][q] for i in ids if i in J[cn]])), 3) for q in DET}
        out["means"][cn]["gate"] = round(float(np.mean([gate(J[cn][i]) for i in ids if i in J[cn]])), 3)
        out["means"][cn]["gate_v2"] = round(float(np.mean([gate_v2(J[cn][i]) for i in ids if i in J[cn]])), 3)
    hl = [feats["human"][i]["chars"] for i in ids]
    for q in DET + ["gate", "gate_v2"]:
        get = (lambda cn, i: gate(J[cn][i])) if q == "gate" else (lambda cn, i: gate_v2(J[cn][i])) if q == "gate_v2" else (lambda cn, i, q=q: J[cn][i][q])
        row = {}
        hs = [SIGN.get(q, 1) * get("human", i) for i in ids]
        base_all = []
        for cn in LLM:
            ls = [SIGN.get(q, 1) * get(cn, i) for i in ids]
            ll = [feats[cn][i]["chars"] for i in ids]
            sa, per = strat_auc(hs, ls, hl, ll)
            row[cn] = {"auc": round(auc(hs, ls), 3), "ci95": boot_ci(hs, ls), "auc_len_strat": sa}
            if cn.startswith("base_"):
                base_all += ls
        bl = [feats[cn][i]["chars"] for cn in LLM[:3] for i in ids]
        row["base_pooled"] = {"auc": round(auc(hs, base_all), 3), "ci95": boot_ci(hs, base_all), "auc_len_strat": strat_auc(hs, base_all, hl, bl)[0]}
        # by source
        for src, pre in (("maichat", "mc_"), ("empathetic", "ed_")):
            sid = [i for i in ids if i.startswith(pre)]
            row["base_pooled_" + src] = round(auc([SIGN.get(q, 1) * get("human", i) for i in sid], [SIGN.get(q, 1) * get(cn, i) for cn in LLM[:3] for i in sid]), 3)
        out["detectors"][q] = row
    # code features (sign chosen so that AUC>0.5 = separates in the human direction)
    CF = {"chars (menor=humano)": lambda f: -f["chars"], "sentences (menos=humano)": lambda f: -f["sentences"],
          "sem ! (humano)": lambda f: -f["n_excl"], "sem emoji": lambda f: -f["n_emoji"], "não termina em ?": lambda f: -int(f["ends_q"]),
          "sem pontuação final": lambda f: -int(f["final_punct"]), "le3_words": lambda f: int(f["le3_words"]),
          "blacklist hits (menos=humano)": lambda f: -sum(v for k, v in f.items() if k.startswith("llm:"))}
    for name, fn in CF.items():
        hs = [fn(feats["human"][i]) for i in ids]
        out["code_features"][name] = {cn: round(auc(hs, [fn(feats[cn][i]) for i in ids]), 3) for cn in LLM}
    # simple combined code score: z(-log chars) etc. (a priori weights)
    def code_score(f):
        return -math.log1p(f["chars"]) - 0.5 * f["sentences"] - 0.3 * min(3, f["n_excl"]) - 0.3 * min(3, f["n_emoji"]) - 0.5 * int(f["ends_q"]) - 0.5 * sum(v for k, v in f.items() if k.startswith("llm:"))
    hs = [code_score(feats["human"][i]) for i in ids]
    out["code_features"]["COMBINADO (código)"] = {cn: round(auc(hs, [code_score(feats[cn][i]) for i in ids]), 3) for cn in LLM}
    # Jev + code: rank-average
    def rank(v):
        v = np.asarray(v); return v.argsort().argsort() / (len(v) - 1)
    for cn in LLM:
        allc = [code_score(feats["human"][i]) for i in ids] + [code_score(feats[cn][i]) for i in ids]
        allj = [gate(J["human"][i]) for i in ids] + [gate(J[cn][i]) for i in ids]
        comb = rank(allc) + rank(allj)
        out["code_features"].setdefault("código+gate Jev (rank médio)", {})[cn] = round(auc(comb[:len(ids)], comb[len(ids):]), 3)
    # pairwise results
    out["pairwise"] = {}
    for cn in LLM:
        p = f"{C.SCR}/jev_pair_{cn}.json"
        if os.path.exists(p):
            P = json.load(open(p))
            ok = [v for v in P.values() if v["ans"]]
            acc = [v["ans"]["which_human"]["choice"] == v["human_is"] for v in ok]
            ph = [v["ans"]["which_human"]["probabilities"][v["human_is"]] for v in ok]
            posA = [v["ans"]["which_human"]["choice"] == "A" for v in ok]
            out["pairwise"][cn] = {"n": len(ok), "acc_picks_human": round(float(np.mean(acc)), 3), "mean_p_human": round(float(np.mean(ph)), 3),
                                   "pct_choose_A": round(float(np.mean(posA)), 3),
                                   "acc_maichat": round(float(np.mean([v["ans"]["which_human"]["choice"] == v["human_is"] for k, v in P.items() if v["ans"] and k.startswith("mc_")])), 3),
                                   "acc_empathetic": round(float(np.mean([v["ans"]["which_human"]["choice"] == v["human_is"] for k, v in P.items() if v["ans"] and k.startswith("ed_")])), 3)}
    C.save("a8_jev_auc.json", out)
    print("picks", dict(picks), "distinct", n_distinct)
    print(f"{'':16s}" + "".join(f"{c[-12:]:>13s}" for c in ["human"] + LLM))
    for q in DET + ["gate", "gate_v2"]:
        print(f"{q:16s}" + "".join(f"{out['means'][c][q]:>13}" for c in ["human"] + LLM))
    print("AUC (human vs LLM; >0.5 = detector separates in expected direction) [auc | len-strat]")
    for q, row in out["detectors"].items():
        print(f"{q:16s}" + "".join(f"  {row[c]['auc']:.3f}|{row[c]['auc_len_strat']}" for c in LLM) + f"  pooled={row['base_pooled']['auc']} {row['base_pooled']['ci95']} strat={row['base_pooled']['auc_len_strat']} mc={row['base_pooled_maichat']} ed={row['base_pooled_empathetic']}")
    print("CODE")
    for k, v in out["code_features"].items():
        print(f"{k:34s}", v)
    print("PAIR", out["pairwise"])


if __name__ == "__main__":
    main()
