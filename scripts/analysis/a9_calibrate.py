"""a9 — calibra (no split dev, conversas disjuntas do teste) os limiares que convertem as respostas do Jev no briefing.
Casamento de taxas: o limiar de cada Noul "p_*" é escolhido para que a fração de pontos marcados seja ~ a taxa humana
no maichat inteiro (os Nouls do Jev superestimam pergunta/riso/emoji). O tamanho usa casamento de quantis.
Saída: analysis/data/a9_thresholds.json (+ diagnósticos: AUC/Spearman no dev)."""
import json, os
import numpy as np
from a9_common import load_points, load_mai, feats, turn_text, ADATA


def auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return None
    from itertools import product
    pos, neg = s[y], s[~y]
    return float(np.mean([(a > b) + 0.5 * (a == b) for a, b in product(pos, neg)]))


def main():
    dev = [p for p in load_points() if p["split"] == "dev" and "jev" in p]
    allF = [feats(turn_text(r["texts"])) for r in load_mai() if not r["media"]]
    rate = {k: float(np.mean([f[k] for f in allF])) for k in ("has_q", "laugh", "emoji")}
    nm = np.array([r["n_msgs"] for r in load_mai()])
    out, diag = {}, {"n_dev": len(dev), "human_rates_maichat": rate}
    for key, hk in (("q", "has_q"), ("laugh", "laugh"), ("emoji", "emoji")):
        s = np.array([p["jev"]["p_" + ("question" if key == "q" else key)] for p in dev])
        y = [p["human_f"][hk] for p in dev]
        thr = float(np.quantile(s, 1 - rate[hk]))
        out[key] = round(max(thr, 0.3), 3)
        diag[key] = {"mean_pred": float(s.mean()), "dev_rate": float(np.mean(y)), "auc_dev": auc(y, s), "thr": out[key]}
    b = np.array([p["jev"]["bubbles"] for p in dev])
    p2, p3 = float(np.mean(nm >= 2)), float(np.mean(nm >= 3))
    out["bub2"] = round(float(np.quantile(b, 1 - p2)), 3)
    out["bub3"] = round(max(out["bub2"] + 0.05, float(np.quantile(b, 1 - p3))), 3)
    hb = [p["human_n_msgs"] for p in dev]
    from scipy.stats import spearmanr
    ownb = [p["fp"]["bubbles_per_turn"] for p in dev]
    diag["bubbles"] = {"spearman_dev": spearmanr(b, hb).correlation, "spearman_own_habit": spearmanr(ownb, hb).correlation,
                       "p_ge2": p2, "p_ge3": p3}
    L = np.array([p["jev"]["length"] for p in dev])
    hw = np.array([p["human_f"]["n_words"] for p in dev])
    allw = np.array([f["n_words"] for f in allF])
    # casamento de quantis: score contínuo do Jev -> nº de palavras humano (maichat inteiro)
    qs = np.linspace(0.05, 0.95, 10)
    xs = [float(np.quantile(L, q)) for q in qs]
    ys = [float(np.quantile(allw, q)) for q in qs]
    for i in range(1, len(xs)):
        xs[i] = max(xs[i], xs[i - 1] + 1e-3)
    out["len_map"] = [xs, ys]
    own = np.array([p["fp"]["median_words_per_bubble"] * max(1, p["fp"]["bubbles_per_turn"]) for p in dev])
    diag["length"] = {"spearman_jev_vs_human": spearmanr(L, hw).correlation,
                      "spearman_ownmedian_vs_human": spearmanr(own, hw).correlation,
                      "jev_mean": float(L.mean()), "human_median_words_dev": float(np.median(hw))}
    json.dump(out, open(os.path.join(ADATA, "a9_thresholds.json"), "w"), indent=1)
    json.dump(diag, open(os.path.join(ADATA, "a9_calibration_diag.json"), "w"), indent=1)
    print(json.dumps(out), "\n", json.dumps(diag, indent=1))


if __name__ == "__main__":
    main()
