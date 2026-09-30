"""b3 — avaliação das arquiteturas de "o que escrever" (dev para escolher, teste para reportar; IC bootstrap por conversa).
Saída: analysis/data/b3_arch_results.json (+ b3_arch_dev.json)."""
import json, math, os, sys
from collections import Counter, defaultdict
import numpy as np
from b3_common import (load_points, kv_load, MOVES, FAMILY, FAMILIES, TONES, cboot, cboot_diff, jdump, ADATA)
from b3_arch import Retriever, FINE

EPS = 0.03  # suavização uniforme p/ log-loss
MI = {m: i for i, m in enumerate(MOVES)}


def vec(d, keys=MOVES):
    v = np.array([float((d or {}).get(k, 0) or 0) for k in keys])
    return v / v.sum() if v.sum() > 0 else np.ones(len(keys)) / len(keys)


def smooth(v):
    return (1 - EPS) * v + EPS / len(v)


def fam_of(v):
    f = np.zeros(len(FAMILIES))
    for m, x in zip(MOVES, v):
        f[FAMILIES.index(FAMILY[m])] += x
    return f


def build(pts, all_pts):
    """Distribuições P(movimento) por arquitetura, por ponto."""
    S = {n: kv_load(f"pred_{n}") for n in ("base", "guide", "casc", "cascp", "char", "rag5", "rag10", "rag20", "ragtxt")}
    CU = kv_load("pred_cand_unpacked")
    LU = kv_load("pred_luna")
    dev = [p for p in all_pts if p["split"] == "dev" and "gold" in p]
    glob = Counter(p["gold"]["g_move"] for p in dev)
    gvec = vec(glob)
    # prior por gatilho (LOCO)
    tab = defaultdict(Counter)
    for p in all_pts:
        if "gold" in p:
            tab[(p["conv_id"], p["prev_D"].get("intent"))][p["gold"]["g_move"]] += 1
    tot = defaultdict(Counter)
    for (cv, it), c in tab.items():
        tot[it].update(c)
    R = Retriever(all_pts)
    out = {}
    for p in pts:
        i, D = p["id"], {}
        D["majority"] = gvec
        it = p["prev_D"].get("intent")
        c = tot[it].copy(); c.subtract(tab[(p["conv_id"], it)])
        D["prior_trigger"] = 0.8 * vec({k: max(0, v) for k, v in c.items()}) + 0.2 * gvec
        nb = R.query(p, 20)
        kc = Counter()
        for q, s in nb:
            kc[q["gold"]["g_move"]] += max(s, 0.01)
        D["knn20"] = 0.8 * vec(kc) + 0.2 * gvec
        b = S["base"].get(i)
        if b:
            D["flat"] = vec(b["flat_p"])
            D["flat_rich"] = vec(b["flat_rich_p"])
            fam = vec(b["fam_p"], FAMILIES)
            h = np.zeros(len(MOVES))
            for fi, f in enumerate(FAMILIES):
                opts = list(FINE[f])
                sub = vec(b[f"fine_{f}_p"], opts) if len(opts) > 1 else np.ones(1)
                for m, x in zip(opts, sub):
                    h[MI[m]] = fam[fi] * x
            D["hier"] = h / h.sum()
            nv = np.array([b[f"n_{m}"] for m in MOVES])
            D["nouls"] = nv / nv.sum()
            D["nouls_sharp"] = nv ** 3 / (nv ** 3).sum()
            D["flat_x_nouls"] = (D["flat"] + 0.02) * nv; D["flat_x_nouls"] /= D["flat_x_nouls"].sum()
        for n in ("guide", "casc", "cascp", "char", "rag5", "rag10", "rag20", "ragtxt"):
            a = S[n].get(i)
            if a:
                D[n] = vec(a["move_p"])
        if i in CU:  # candidatas: só no nível de família
            sc = np.array([CU[i][f]["friend"] + CU[i][f]["fits"] + CU[i][f]["tone"] - CU[i][f]["intense"] - CU[i][f]["overdo"]
                           for f in FAMILIES])
            e = np.exp(4 * (sc - sc.max()))
            D["cand_fam"] = e / e.sum()
            fr = np.array([CU[i][f]["friend"] for f in FAMILIES])
            e = np.exp(6 * (fr - fr.max()))
            D["cand_friend_fam"] = e / e.sum()
        if i in LU:
            v = np.zeros(len(MOVES))
            for r, m in enumerate(LU[i][:3]):
                v[MI[m]] += [0.6, 0.25, 0.15][r]
            D["luna_top3"] = v / v.sum()
        # misturas (pesos escolhidos no dev, ver choose())
        out[i] = D
    return out


def mix(D, parts, w):
    v = sum(wi * np.log(smooth(D[pn])) for pn, wi in zip(parts, w))
    e = np.exp(v - v.max())
    return e / e.sum()


def metrics(pts, dists, name, fam_only=False):
    rows = []
    for p in pts:
        D = dists.get(p["id"], {})
        if name not in D or "gold" not in p:
            continue
        v = D[name]
        g = p["gold"]["g_move"]
        gl = p.get("gold_luna")
        r = {"conv": p["conv_id"]}
        if fam_only:
            gf = FAMILIES.index(FAMILY[g])
            r["fam_top1"] = float(np.argmax(v) == gf)
            r["fam_ll"] = -math.log(smooth(v)[gf])
            if gl:
                r["fam_top1_luna"] = float(np.argmax(v) == FAMILIES.index(FAMILY[gl["primary"]]))
                r["fam_hit_luna"] = float(FAMILIES[int(np.argmax(v))] in {FAMILY[m] for m in gl["all"]})
            rows.append(r); continue
        order = np.argsort(-v)
        r["top1"] = float(MOVES[order[0]] == g)
        r["top3"] = float(g in [MOVES[k] for k in order[:3]])
        r["ll"] = -math.log(smooth(v)[MI[g]])
        r["exp_acc"] = float(v[MI[g]])  # coincidência esperada se SORTEAR de v
        f = fam_of(v)
        r["fam_top1"] = float(np.argmax(f) == FAMILIES.index(FAMILY[g]))
        r["pred"] = MOVES[order[0]]
        if gl:
            r["top1_luna"] = float(MOVES[order[0]] == gl["primary"])
            r["hit_luna"] = float(MOVES[order[0]] in gl["all"])
            r["fam_top1_luna"] = float(FAMILIES[int(np.argmax(f))] == FAMILY[gl["primary"]])
        rows.append(r)
    return rows


def summarize(rows, keys, boot=True):
    out = {"n": len(rows)}
    g = [r["conv"] for r in rows]
    for k in keys:
        v = [r.get(k) for r in rows]
        if all(x is None for x in v):
            continue
        out[k] = cboot(v, g) if boot else [round(float(np.mean([x for x in v if x is not None])), 4)]
    return out


def diversity(rows, pts):
    pr = Counter(r["pred"] for r in rows)
    n = sum(pr.values())
    H = -sum(c / n * math.log2(c / n) for c in pr.values())
    return {"entropy_argmax_bits": round(H, 2), "top_pred_share": round(pr.most_common(1)[0][1] / n, 3),
            "n_distinct": len(pr), "most_common": pr.most_common(4)}


MOVE_ARCHS = ["majority", "prior_trigger", "knn20", "flat", "flat_rich", "hier", "nouls", "nouls_sharp", "flat_x_nouls",
              "guide", "casc", "cascp", "char", "rag5", "rag10", "rag20", "ragtxt", "luna_top3"]
FAM_ARCHS = ["cand_fam", "cand_friend_fam"]
KEYS = ["top1", "top3", "ll", "exp_acc", "fam_top1", "top1_luna", "hit_luna", "fam_top1_luna"]


def main():
    all_pts = load_points()
    ev = [p for p in all_pts if p["eval"] and "gold" in p]
    dists = build(ev, all_pts)
    dev = [p for p in ev if p["split"] == "dev"]
    test = [p for p in ev if p["split"] == "test"]
    # ---------- escolha no DEV (sem bootstrap)
    devres = {a: summarize(metrics(dev, dists, a), KEYS, boot=False) for a in MOVE_ARCHS}
    for a in FAM_ARCHS:
        devres[a] = summarize(metrics(dev, dists, a, fam_only=True), ["fam_top1", "fam_ll", "fam_top1_luna", "fam_hit_luna"], boot=False)
    # misturas / ensembles: grade de pesos no dev
    cands = {}
    base_parts = [a for a in ["flat", "flat_rich", "hier", "nouls", "guide", "casc", "cascp", "char", "rag10", "rag20", "knn20",
                              "prior_trigger"]]
    for combo in [("rag20", "knn20"), ("rag10", "knn20"), ("flat", "knn20"), ("casc", "knn20"), ("cascp", "knn20"),
                  ("rag20", "flat"), ("rag20", "cascp"), ("rag20", "hier"), ("rag20", "rag10", "knn20"),
                  ("rag20", "cascp", "knn20"), ("rag20", "flat_rich", "knn20"), ("flat", "flat_rich", "hier", "casc", "cascp", "guide", "char")]:
        best = None
        grid = [(1,) * len(combo)] + ([(w, 1 - w) for w in (0.2, 0.35, 0.5, 0.65, 0.8)] if len(combo) == 2 else [])
        for w in grid:
            nm = "mix:" + "+".join(combo) + ":" + ",".join(f"{x:.2f}" for x in w)
            for p in dev + test:
                D = dists.get(p["id"], {})
                if all(c in D for c in combo):
                    D[nm] = mix(D, combo, w)
            s = summarize(metrics(dev, dists, nm), KEYS, boot=False)
            sc = s.get("top1", [0])[0] + s.get("top1_luna", [0])[0] - 0.1 * s.get("ll", [9])[0]
            if best is None or sc > best[0]:
                best = (sc, nm, s)
        cands[best[1]] = best[2]
        devres[best[1]] = best[2]
    json.dump(devres, open(os.path.join(ADATA, "b3_arch_dev.json"), "w"), indent=1)

    def dscore(s):
        return s.get("top1", [0])[0] + s.get("top1_luna", [0])[0] - 0.1 * s.get("ll", [9])[0]
    ranked = sorted([a for a in devres if "top1" in devres[a] and a != "luna_top3"], key=lambda a: -dscore(devres[a]))
    chosen = ranked[0]
    print("DEV ranking:")
    for a in ranked:
        s = devres[a]
        print(f"  {a:45s} top1 {s['top1'][0]:.3f} luna {s.get('top1_luna', [0])[0]:.3f} hit {s.get('hit_luna', [0])[0]:.3f} "
              f"top3 {s['top3'][0]:.3f} ll {s['ll'][0]:.3f} fam {s['fam_top1'][0]:.3f} n {s['n']}")
    for a in FAM_ARCHS:
        print(f"  {a:45s}", devres[a])
    # ---------- TESTE
    res = {"chosen_on_dev": chosen, "dev": devres, "test": {}}
    for a in MOVE_ARCHS + [c for c in cands]:
        rows = metrics(test, dists, a)
        if not rows:
            continue
        res["test"][a] = summarize(rows, KEYS)
        res["test"][a]["diversity"] = diversity(rows, test)
    for a in FAM_ARCHS:
        res["test"][a] = summarize(metrics(test, dists, a, fam_only=True), ["fam_top1", "fam_ll", "fam_top1_luna", "fam_hit_luna"])
    # diferenças pareadas contra a 1ª rodada (flat) e o melhor código (knn20)
    def paired(a, b, k="top1"):
        ra = {r_: None for r_ in []}
        A = {p["id"]: r for p, r in zip([q for q in test if a in dists.get(q["id"], {}) and b in dists.get(q["id"], {})], [])}
        ids = [q["id"] for q in test if a in dists.get(q["id"], {}) and b in dists.get(q["id"], {})]
        sub = [q for q in test if q["id"] in set(ids)]
        ma, mb = metrics(sub, dists, a), metrics(sub, dists, b)
        return cboot_diff([r[k] for r in ma], [r[k] for r in mb], [r["conv"] for r in ma])
    res["paired_test"] = {}
    for a in [chosen, "rag20", "cascp", "hier", "knn20", "flat_rich", "casc"]:
        for b in ["flat", "knn20", "majority"]:
            if a != b:
                res["paired_test"][f"{a} - {b}"] = {k: paired(a, b, k) for k in ("top1", "top1_luna", "ll")}
    # ---------- teto dos rótulos (concordância entre rotuladores)
    agr = [(float(p["gold"]["g_move"] == p["gold_luna"]["primary"]), float(p["gold"]["g_move"] in p["gold_luna"]["all"]),
            float(FAMILY[p["gold"]["g_move"]] == FAMILY[p["gold_luna"]["primary"]]), p["conv_id"])
           for p in test if p.get("gold_luna")]
    res["label_ceiling"] = {"jev_vs_luna_primary": cboot([a[0] for a in agr], [a[3] for a in agr]),
                            "jev_in_luna_all": cboot([a[1] for a in agr], [a[3] for a in agr]),
                            "family_agree": cboot([a[2] for a in agr], [a[3] for a in agr])}
    gm = Counter(p["gold"]["g_move"] for p in test)
    res["gold_dist_test"] = {m: round(c / len(test), 3) for m, c in gm.most_common()}
    gl = Counter(p["gold_luna"]["primary"] for p in test if p.get("gold_luna"))
    res["gold_luna_dist_test"] = {m: round(c / sum(gl.values()), 3) for m, c in gl.most_common()}
    H = -sum(c / len(test) * math.log2(c / len(test)) for c in gm.values())
    res["gold_entropy_bits"] = round(H, 2)
    # ---------- confiança como portão (arquitetura escolhida e flat): precisão por faixa
    res["gating"] = {}
    for a in [chosen, "flat", "rag20", "cascp"]:
        if a not in res["test"]:
            continue
        for split, P_ in (("dev", dev), ("test", test)):
            rows = []
            for p in P_:
                D = dists.get(p["id"], {})
                if a not in D:
                    continue
                v = D[a]; conf = float(v.max())
                rows.append((conf, float(MOVES[int(np.argmax(v))] == p["gold"]["g_move"]),
                             float(MOVES[int(np.argmax(v))] in (p.get("gold_luna") or {}).get("all", [p["gold"]["g_move"]]))))
            bins = {}
            for lo, hi in ((0, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 1.01)):
                s = [r for r in rows if lo <= r[0] < hi]
                if s:
                    bins[f"{lo}-{hi}"] = {"n": len(s), "share": round(len(s) / len(rows), 3), "top1": round(np.mean([x[1] for x in s]), 3),
                                          "hit_luna": round(np.mean([x[2] for x in s]), 3)}
            res["gating"][f"{a}:{split}"] = bins
    # ---------- tom
    tone_rows = [{"conv": p["conv_id"], "tone": float(kv_base(p, "tone") == p["gold"]["g_tone"]),
                  "tone_luna": (float(kv_base(p, "tone") == p["gold_luna"].get("tone")) if p.get("gold_luna") else None)}
                 for p in test if kv_base(p, "tone")]
    maj_tone = Counter(p["gold"]["g_tone"] for p in dev).most_common(1)[0][0]
    res["tone_test"] = {"jev_tone": summarize(tone_rows, ["tone", "tone_luna"]),
                        "majority_tone": maj_tone,
                        "majority_acc": cboot([float(p["gold"]["g_tone"] == maj_tone) for p in test], [p["conv_id"] for p in test]),
                        "tone_agree_jev_luna": cboot([float(p["gold"]["g_tone"] == p["gold_luna"].get("tone")) for p in test if p.get("gold_luna")],
                                                     [p["conv_id"] for p in test if p.get("gold_luna")])}
    # ---------- subtexto
    res["subtext"] = subtext_eval(test)
    # ---------- elemento reagido
    res["element"] = element_eval(dev, test)
    jdump("b3_arch_results.json", res)
    print(json.dumps({k: res["test"][k] for k in list(res["test"])[:3]}, indent=0)[:1500])


_B = None


def kv_base(p, k):
    global _B
    if _B is None:
        _B = kv_load("pred_base")
    return (_B.get(p["id"]) or {}).get(k)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y, s = np.array(y), np.array(s)
    return round(float(roc_auc_score(y, s)), 3) if 0 < y.sum() < len(y) else None


def subtext_eval(test):
    B = kv_load("pred_base")
    rows = [(p["gold"]["g_subtext"] >= 0.5, B[p["id"]]["sub_implicit"], B[p["id"]]["sub_how"], p["conv_id"],
             p["gold"]["g_specific"] >= 0.5) for p in test if p["id"] in B]
    y = [r[0] for r in rows]
    how = Counter((r[2], r[0]) for r in rows)
    return {"n": len(rows), "gold_subtext_rate": round(float(np.mean(y)), 3), "auc_sub_implicit": auc(y, [r[1] for r in rows]),
            "sub_how_x_gold": {f"{a}|{'implicit' if b else 'explicit'}": c for (a, b), c in how.items()},
            "gold_specific_rate": round(float(np.mean([r[4] for r in rows])), 3)}


def element_eval(dev, test):
    E = kv_load("pred_elem")
    out = {}
    for split, P_ in (("dev", dev), ("test", test)):
        rows = []
        for p in P_:
            ge = p.get("gold_elem")
            gl = (p.get("gold_luna") or {}).get("element")
            els = p["elements"]
            if not ge or len(els) < 2 or p["id"] not in E or "elem" not in E[p["id"]]:
                continue
            g = ge["g_elem"]
            pred, pf = E[p["id"]]["elem"], E[p["id"]]["elem_friend"]
            n = len(els)
            r = {"conv": p["conv_id"], "n_elems": n, "gold_is_word": float(g.startswith("e")),
                 "luna_is_word": (float(isinstance(gl, int) and gl >= 1) if gl is not None else None)}
            if g.startswith("e"):
                pw = E[p["id"]]["elem_p"]
                wv = {k: v for k, v in pw.items() if k.startswith("e")}
                r["acc_elem"] = float(max(wv, key=wv.get) == g)  # entre as palavras
                pwf = {k: v for k, v in E[p["id"]]["elem_friend_p"].items() if k.startswith("e")}
                r["acc_friend"] = float(max(pwf, key=pwf.get) == g)
                r["acc_last"] = float(g == f"e{n}")
                r["acc_first"] = float(g == "e1")
                lw = max(range(n), key=lambda k: len(els[k]))
                r["acc_longest"] = float(g == f"e{lw + 1}")
                r["acc_random"] = 1.0 / n
                r["pred_says_word"] = float(pred.startswith("e"))
            if isinstance(gl, int) and gl >= 1 and g.startswith("e"):
                r["jev_luna_agree"] = float(g == f"e{gl}")
            if isinstance(gl, int) and gl >= 1:
                pw = E[p["id"]]["elem_p"]
                wv = {k: v for k, v in pw.items() if k.startswith("e")}
                r["acc_elem_vs_luna"] = float(max(wv, key=wv.get) == f"e{gl}")
                r["acc_random_vs_luna"] = 1.0 / n
            rows.append(r)
        out[split] = summarize(rows, ["gold_is_word", "luna_is_word", "acc_elem", "acc_friend", "acc_last", "acc_first",
                                      "acc_longest", "acc_random", "jev_luna_agree", "acc_elem_vs_luna",
                                      "acc_random_vs_luna", "pred_says_word"], boot=(split == "test"))
    return out


if __name__ == "__main__":
    main()
