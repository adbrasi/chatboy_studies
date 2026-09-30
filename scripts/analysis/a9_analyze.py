"""a9 — análise: taxas de "engano" (juízes Jev e LLM), métricas de código, qualidade Jev, quebras por momento,
ablações, efeito do "não use X", aderência ao briefing, latência e custo. Bootstrap por CONVERSA (cluster).
Saída: analysis/data/a9_results.json e a9_examples.json"""
import json, math, os, re
from collections import Counter, defaultdict
import numpy as np
from a9_common import load_points, feats, load_mai, turn_text, words, ADATA
from a9_brief import BAN, moment_of, TH

RNG = np.random.default_rng(9)
MAIN = ["A", "S", "B", "C", "D"]
ABL = ["A", "S", "B", "Bnoban", "Blong", "Bpure"]
B_ITERS = 2000


def cboot(vals, groups, stat=np.mean, iters=B_ITERS):
    """IC 95% bootstrap por cluster (conversa)."""
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    ok = ~np.isnan(vals)
    vals, groups = vals[ok], groups[ok]
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    bs = []
    for _ in range(iters):
        s = RNG.choice(ug, len(ug))
        v = np.concatenate([vals[idx[g]] for g in s])
        bs.append(stat(v))
    return [float(stat(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(vals))]


def main():
    pts = {p["id"]: p for p in load_points() if p["split"] == "test" and "gen" in p}
    J = {d["id"]: d for d in map(json.loads, open(os.path.join(ADATA, "a9_judge.jsonl")))}
    ids = [i for i in pts if i in J]
    grp = {i: pts[i]["conv_id"] for i in ids}
    R = {"n_points": len(ids), "n_convs": len(set(grp.values()))}
    F = {i: {c: feats(t) for c, t in pts[i]["gen"].items()} for i in ids}
    for i in ids:
        F[i]["H"] = feats(pts[i]["human"])

    def col(fn, conds, sub=None):
        out = {}
        for c in conds:
            v, g = [], []
            for i in (sub or ids):
                x = fn(i, c)
                if x is None:
                    continue
                v.append(x); g.append(grp[i])
            out[c] = cboot(v, g) if v else None
        return out

    # ---------------- (1) juiz Jev em pares (média das duas ordens) e (4) juiz LLM
    def jev_fool(i, c):
        d = J[i]["pair_jev"].get(c)
        return None if not d or len(d) < 2 else (d["0"] + d["1"]) / 2 if "0" in d else (d[0] + d[1]) / 2

    def jev_fool_hard(i, c):
        x = jev_fool(i, c)
        return None if x is None else float(x > 0.5)

    def llm_fool(i, c):
        d = J[i]["llm_judge"].get(c)
        return None if not d else d["fooled"]

    allc = MAIN + ["Bnoban", "Blong", "Bpure"]
    R["jev_fool_soft"] = col(jev_fool, allc)
    R["jev_fool_hard"] = col(jev_fool_hard, allc)
    R["llm_fool"] = col(llm_fool, allc)
    # viés de posição do juiz Jev: P(reply_1 = humano) médio; e concordância entre ordens
    p1 = []
    agree = []
    for i in ids:
        for c, d in J[i]["pair_jev"].items():
            if len(d) == 2:
                a0, a1 = d.get("0", d.get(0)), d.get("1", d.get(1))
                p1 += [1 - a0, a1]  # order0: condição é reply_2 -> P(reply_1)=1-a0 ; order1: condição é reply_1
                agree.append((a0 > 0.5) == (a1 > 0.5))
    R["jev_judge_position"] = {"mean_P_reply1_is_human": float(np.mean(p1)), "order_agreement": float(np.mean(agree))}
    lo = [J[i]["llm_judge"][c]["order"] for i in ids for c in J[i]["llm_judge"]]
    lf = [J[i]["llm_judge"][c]["fooled"] for i in ids for c in J[i]["llm_judge"]]
    lpick1 = [int((f == 1 and o == 1) or (f == 0 and o == 0)) for f, o in zip(lf, lo)]
    R["llm_judge_position"] = {"P_pick_candidate1": float(np.mean(lpick1))}
    # concordância Jev x LLM (por par)
    both = [(jev_fool_hard(i, c), llm_fool(i, c)) for i in ids for c in MAIN if jev_fool(i, c) is not None and llm_fool(i, c) is not None]
    R["judge_agreement_jev_vs_llm"] = float(np.mean([a == b for a, b in both]))

    # ---------------- (2) métricas de código
    human_rate = {}
    codem = {}
    for c in ["H"] + allc:
        sub = [i for i in ids if c in F[i]]
        g = [grp[i] for i in sub]
        m = {}
        for k in ("has_q", "laugh", "emoji", "excl", "llmish", "dash", "starts_lower", "ends_period"):
            m[k] = cboot([float(F[i][c][k]) for i in sub], g)
        m["slang_any"] = cboot([float(F[i][c]["slang_n"] > 0) for i in sub], g)
        m["words_median"] = cboot([F[i][c]["n_words"] for i in sub], g, stat=np.median)
        m["words_mean"] = cboot([F[i][c]["n_words"] for i in sub], g)
        m["bubbles_mean"] = cboot([F[i][c]["n_bubbles"] for i in sub], g)
        m["multi_bubble"] = cboot([float(F[i][c]["n_bubbles"] > 1) for i in sub], g)
        if c != "H":
            m["abs_log2_len_ratio"] = cboot([abs(math.log2((F[i][c]["n_words"] + 1) / (F[i]["H"]["n_words"] + 1))) for i in sub], g)
            m["len_ratio_median"] = cboot([(F[i][c]["n_words"] + 1) / (F[i]["H"]["n_words"] + 1) for i in sub], g, stat=np.median)
            for k in ("has_q", "laugh", "emoji", "multi"):
                kk = "n_bubbles" if k == "multi" else k
                m[f"match_{k}"] = cboot([float((F[i][c][kk] > 1 if k == "multi" else F[i][c][kk]) ==
                                               (F[i]["H"][kk] > 1 if k == "multi" else F[i]["H"][kk])) for i in sub], g)
        codem[c] = m
    R["code_metrics"] = codem

    # ---------------- (3) qualidade Jev (Nouls isolados) — inclui o humano como referência
    qk = ["fits_mood", "sounds_ai", "too_formal", "forced_slang", "coherent", "over_enth"]
    R["jev_quality"] = {c: {k: cboot([J[i]["qual_jev"][c][k] for i in ids if c in J[i]["qual_jev"]],
                                     [grp[i] for i in ids if c in J[i]["qual_jev"]]) for k in qk}
                        for c in ["H"] + allc}

    # ---------------- diferenças pareadas com IC (bootstrap por conversa)
    def paired(fn, a, b, sub=None):
        v, g = [], []
        for i in (sub or ids):
            x, y = fn(i, a), fn(i, b)
            if x is None or y is None:
                continue
            v.append(x - y); g.append(grp[i])
        return cboot(v, g)

    def mood(i, c):
        return J[i]["qual_jev"].get(c, {}).get("fits_mood")

    def aiish(i, c):
        return J[i]["qual_jev"].get(c, {}).get("sounds_ai")

    def lenerr(i, c):
        return abs(math.log2((F[i][c]["n_words"] + 1) / (F[i]["H"]["n_words"] + 1))) if c in F[i] else None

    def llmish(i, c):
        return float(F[i][c]["llmish"]) if c in F[i] else None

    diffs = {}
    for a, b in [("B", "A"), ("B", "S"), ("C", "B"), ("C", "S"), ("S", "A"), ("D", "A"), ("C", "D"), ("B", "D")]:
        diffs[f"{a}-{b}"] = {"jev_fool": paired(jev_fool, a, b), "llm_fool": paired(llm_fool, a, b),
                             "fits_mood": paired(mood, a, b), "sounds_ai": paired(aiish, a, b),
                             "abs_log2_len": paired(lenerr, a, b), "llmish": paired(llmish, a, b)}
    R["paired_diffs"] = diffs

    # ---------------- quebra por tipo de momento
    strata = sorted(set(pts[i]["stratum"] for i in ids))
    bys = {}
    for s in strata:
        sub = [i for i in ids if pts[i]["stratum"] == s]
        bys[s] = {"n": len(sub)}
        for c in MAIN:
            def mean_of(fn):
                v = [fn(i, c) for i in sub]
                v = [x for x in v if x is not None]
                return round(float(np.mean(v)), 3) if v else None
            bys[s][c] = {"jev_fool": mean_of(jev_fool), "llm_fool": mean_of(llm_fool), "fits_mood": mean_of(mood),
                         "words_med": float(np.median([F[i][c]["n_words"] for i in sub])),
                         "q": mean_of(lambda i, c: float(F[i][c]["has_q"])),
                         "llmish": mean_of(llmish)}
        bys[s]["H"] = {"words_med": float(np.median([F[i]["H"]["n_words"] for i in sub])),
                       "q": float(np.mean([F[i]["H"]["has_q"] for i in sub])),
                       "fits_mood": float(np.mean([J[i]["qual_jev"]["H"]["fits_mood"] for i in sub]))}
    R["by_stratum"] = bys
    # combinação dos dois juízes por momento com IC para B-A e C-A (média dos juízes)
    R["by_stratum_diff"] = {}
    for s in strata:
        sub = [i for i in ids if pts[i]["stratum"] == s]
        both_j = lambda i, c: (None if jev_fool(i, c) is None or llm_fool(i, c) is None else (jev_fool(i, c) + llm_fool(i, c)) / 2)
        R["by_stratum_diff"][s] = {"B-A": paired(both_j, "B", "A", sub), "B-S": paired(both_j, "B", "S", sub),
                                   "C-A": paired(both_j, "C", "A", sub)}

    # ---------------- ablações (mesmos 36 pontos)
    sub = [i for i in ids if all(c in pts[i]["gen"] for c in ABL)]
    ab = {"n": len(sub)}
    for c in ABL:
        g = [grp[i] for i in sub]
        ab[c] = {"jev_fool": cboot([jev_fool(i, c) for i in sub], g), "llm_fool": cboot([llm_fool(i, c) for i in sub], g),
                 "fits_mood": cboot([mood(i, c) for i in sub], g), "sounds_ai": cboot([aiish(i, c) for i in sub], g),
                 "coherent": cboot([J[i]["qual_jev"][c]["coherent"] for i in sub], g),
                 "words_median": float(np.median([F[i][c]["n_words"] for i in sub])),
                 "llmish": float(np.mean([F[i][c]["llmish"] for i in sub])),
                 "has_q": float(np.mean([F[i][c]["has_q"] for i in sub])),
                 "abs_log2_len": float(np.mean([lenerr(i, c) for i in sub]))}
        ab[c]["brief_words"] = (float(np.mean([len(words(pts[i]["brief"][c])) for i in sub if c in pts[i].get("brief", {})]))
                                if c in ("B", "Bnoban", "Blong", "Bpure") else 0)
    R["ablations"] = ab

    # ---------------- "não use X": taxa de cada palavra proibida por condição
    ban_rx = {"aww": r"aww+", "totally": "totally", "absolutely": "absolutely", "amazing": "amazing",
              "sounds like": "sounds like", "that sounds": "that sounds", "I'd love": r"i'd love|i would love",
              "honestly": "honestly", "vibe(s)": r"vibes?|vibing", "super": "super", "definitely": "definitely",
              "journey": "journey", "em dash": "[—–]", "exclamation": "!"}
    ban = {}
    for c in ["A", "S", "B", "C", "D"]:
        ban[c] = {w: float(np.mean([bool(re.search(rf"(?i)(?<![a-z]){rx}(?![a-z])" if w not in ("em dash", "exclamation") else rx,
                                                   pts[i]["gen"][c])) for i in ids])) for w, rx in ban_rx.items()}
        ban[c]["any_banned"] = float(np.mean([any(re.search(rf"(?i)(?<![a-z]){rx}(?![a-z])" if w not in ("em dash", "exclamation") else rx,
                                                            pts[i]["gen"][c]) for w, rx in ban_rx.items()) for i in ids]))
    for c in ["S", "B", "Bnoban", "Blong", "Bpure"]:
        ban[c + "_sub36"] = float(np.mean([any(re.search(rf"(?i)(?<![a-z]){rx}(?![a-z])" if w not in ("em dash", "exclamation") else rx,
                                                         pts[i]["gen"][c]) for w, rx in ban_rx.items()) for i in sub]))
    R["ban_words"] = ban

    # ---------------- aderência ao briefing (B) e qualidade das previsões do Jev frente ao humano
    comp = defaultdict(list)
    pred = defaultdict(list)
    for i in ids:
        p, j = pts[i], pts[i]["jev"]
        if p.get("fallback", {}).get("B"):
            continue
        br = p["brief"]["B"]
        mw = int(re.search(r"Max (\d+) words", br).group(1))
        fb = F[i]["B"]
        comp["within_max_words"].append(fb["n_words"] <= mw)
        comp["within_max_words+2"].append(fb["n_words"] <= mw + 2)
        if "no question" in br.lower():
            comp["no_question_respected"].append(not fb["has_q"])
            comp["no_question_but_question_words"].append(bool(re.match(r"(?i)\s*(what|how|why|do|did|are|is|wanna|u)\b", p["gen"]["B"].split("\n")[-1])) and not fb["has_q"])
        if "no laughing" in br.lower():
            comp["no_laugh_respected"].append(not fb["laugh"])
        if "Laugh the way" in br:
            comp["laugh_when_told"].append(fb["laugh"])
        if "no emoji" in br.lower():
            comp["no_emoji_respected"].append(not fb["emoji"])
        if "split into 2" in br:
            comp["split_when_told"].append(fb["n_bubbles"] >= 2)
        else:
            comp["single_when_told"].append(fb["n_bubbles"] == 1)
        if "all lowercase" in br:
            comp["lowercase_when_told"].append(fb["starts_lower"])
        h = F[i]["H"]
        pred["q"].append((j["p_question"], h["has_q"]))
        pred["laugh"].append((j["p_laugh"], h["laugh"]))
        pred["emoji"].append((j["p_emoji"], h["emoji"]))
        pred["len"].append((j["length"], h["n_words"]))
        pred["bub"].append((j["bubbles"], p["human_n_msgs"]))
        pred["maxw"].append((mw, h["n_words"]))
    R["brief_compliance"] = {k: [float(np.mean(v)), len(v)] for k, v in comp.items()}
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    pq = {}
    for k in ("q", "laugh", "emoji"):
        s, y = zip(*pred[k])
        thr = TH[k if k != "q" else "q"]
        pq[k] = {"auc": float(roc_auc_score(y, s)) if 0 < sum(y) < len(y) else None, "human_rate": float(np.mean(y)),
                 "mean_pred": float(np.mean(s)), "flag_rate": float(np.mean(np.array(s) >= thr)),
                 "acc_flag": float(np.mean((np.array(s) >= thr) == np.array(y)))}
    for k in ("len", "bub", "maxw"):
        s, y = zip(*pred[k])
        pq[k] = {"spearman": float(spearmanr(s, y).correlation)}
    mw_h = [(m, h) for m, h in pred["maxw"]]
    pq["human_within_maxw"] = float(np.mean([h <= m for m, h in mw_h]))
    R["jev_vs_human"] = pq
    R["moments_jev"] = dict(Counter(moment_of(pts[i]["jev"]) for i in ids))
    R["moves_jev"] = dict(Counter(pts[i]["jev"]["move"] for i in ids))
    R["gate"] = {"pick_choice": dict(Counter(pts[i]["gate"]["pick_choice"] for i in ids)),
                 "pick_noul": dict(Counter(pts[i]["gate"]["pick"] for i in ids))}
    R["fallbacks"] = dict(Counter(c for i in ids for c in pts[i].get("fallback", {})))
    # o gate escolhe melhor que o acaso? (fool LLM de C x média dos 3 candidatos não é observável: só C e B foram julgados)

    # ---------------- latência e custo
    lat = json.load(open(os.path.join(ADATA, "a9_latency.json")))
    cost = json.load(open(os.path.join(ADATA, "a9_costs.json")))

    def pct(v, q):
        v = [x for x in v if x]
        return round(float(np.percentile(v, q)), 3) if v else None
    L = {}
    for k, d in lat.items():
        v = [d.get(i) for i in ids if d.get(i)]
        L[k] = {"p50": pct(v, 50), "p90": pct(v, 90), "n": len(v)}
    pipe = {}
    for c in ["A", "S", "D"]:
        pipe[c] = [lat[f"gen_{c}"].get(i) for i in ids]
    pipe["B"] = [(lat["jev_brief"].get(i) or np.nan) + (lat["gen_B"].get(i) or np.nan) for i in ids]
    pipe["C"] = [(lat["jev_brief"].get(i) or np.nan) + max(lat["gen_B"].get(i) or np.nan, lat["gen_B1"].get(i) or np.nan,
                                                           lat["gen_B2"].get(i) or np.nan) + (lat["jev_gate"].get(i) or np.nan)
                 for i in ids]
    L["pipeline"] = {c: {"p50": pct([x for x in v if x and not np.isnan(x)], 50), "p90": pct([x for x in v if x and not np.isnan(x)], 90)}
                     for c, v in pipe.items()}
    jev_call = 0.00004 * 1.0  # ~US$0,00004 por 1k tokens de entrada (DATA.md); abaixo usamos o custo medido
    cpr = {"A": cost["gen_A_test"]["per_call"], "S": cost["gen_S_test"]["per_call"], "D": cost["gen_D_test"]["per_call"]}
    brief_cost = 0.009090942 / 118  # custo medido das chamadas de briefing no teste
    gate_cost = 0.003124884 / 122 if "jev_gate" in cost else 0
    cpr["B"] = cost["gen_B_test"]["per_call"] + brief_cost
    cpr["C"] = cost["gen_B_test"]["per_call"] + cost["gen_B1_test"]["per_call"] + cost["gen_B2_test"]["per_call"] + brief_cost + gate_cost
    R["latency"] = L
    R["cost_per_reply_usd"] = cpr
    R["cost_components"] = {"jev_brief_per_call": brief_cost, "jev_gate_per_call": gate_cost}
    json.dump(R, open(os.path.join(ADATA, "a9_results.json"), "w"), indent=1, ensure_ascii=False)

    # ---------------- vocabulário: tokens super-representados nas LLMs vs humanos (log-odds com prior)
    hum = Counter(w.lower() for r in load_mai() for w in words(turn_text(r["texts"])))
    voc = {}
    for c in ["A", "S", "B", "D"]:
        gen = Counter(w.lower() for i in ids for w in words(pts[i]["gen"][c]))
        nh, ng = sum(hum.values()), sum(gen.values())
        sc = {}
        for w, k in gen.items():
            if k < 4:
                continue
            a = (k + 0.5) / (ng + 1)
            b = (hum.get(w, 0) + 0.5) / (nh + 1)
            sc[w] = (math.log(a / b), k, hum.get(w, 0))
        voc[c] = sorted(sc.items(), key=lambda kv: -kv[1][0])[:25]
    hgen = Counter(w.lower() for i in ids for w in words(pts[i]["human"]))
    ga = Counter(w.lower() for i in ids for w in words(pts[i]["gen"]["A"]))
    under = {w: k for w, k in hum.most_common(60)}
    voc["human_words_llmA_underuses"] = sorted(
        [(w, round(math.log(((hum[w] + .5) / sum(hum.values())) / ((ga.get(w, 0) + .5) / sum(ga.values()))), 2), hum[w], ga.get(w, 0))
         for w in under], key=lambda x: -x[1])[:20]
    json.dump(voc, open(os.path.join(ADATA, "a9_vocab.json"), "w"), indent=1, ensure_ascii=False)

    # ---------------- exemplos (compactos) para escolher os mais reveladores
    ex = []
    for i in ids:
        p = pts[i]
        ex.append({"id": i, "stratum": p["stratum"], "ctx": [f"{h['who']}: {h['text']}" for h in p["history"][-3:]],
                   "H": p["human"], **{c: p["gen"].get(c) for c in ["A", "S", "B", "C", "D", "Blong", "Bnoban", "Bpure"]},
                   "brief": p.get("brief", {}).get("B"), "jev_fool": {c: jev_fool(i, c) for c in MAIN},
                   "llm_fool": {c: llm_fool(i, c) for c in MAIN}, "gate": p["gate"]["pick"]})
    json.dump(ex, open(os.path.join(ADATA, "a9_examples.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: R[k] for k in ("jev_fool_soft", "jev_fool_hard", "llm_fool")}, indent=0))


if __name__ == "__main__":
    main()
