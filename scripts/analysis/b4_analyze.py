"""b4 — análise: distâncias às taxas humanas, coerência/perda de conteúdo (Jev), intensidade, custo, latência.
Métrica principal (menor = mais perto do humano):  D = erro de tamanho médio |log2((w+1)/(h+1))| + Σ|taxa_cond − taxa_humana|
sobre 10 vícios (pergunta, "!", emoji, riso, lista LLM-ish, 3+ frases, molde de 3 tempos, paráfrase/eco, abertura
performática, pergunta recíproca genérica). ICs 95% por bootstrap por CONVERSA.
Uso: python3 b4_analyze.py <split> [modelos] -> analysis/data/b4_results_<split>.json"""
import json, os, sys, statistics as st
import numpy as np
from b4_common import (load_points, load_gen, feats, lenerr, RATE_KEYS, cboot, budget, ADATA, SCR)
from b4_eval import load, EV, KP

ORDER = ["A", "S", "B9", "T", "F1", "MT", "MTfix", "MT12", "STnl", "STnlE", "STq", "STqE", "TLO", "THI", "PFX", "LB", "NCA",
         "SHORT", "NCT", "PL1", "PL2", "D5x", "D5", "RW", "INT", "INTr", "N", "R1", "R2", "R3", "R4", "R5", "J1", "J2", "J3"]


def per_item(p, rec, A, ev, kp, model):
    t = rec.get("text") or ""
    fb = False
    if not t.strip():
        t, fb = (A.get("text") or ""), True
    f = feats(t, p, rec.get("finish"))
    h = feats(p["human"], p)
    e = ev.get((model, p["id"], t), {})
    k = kp.get((model, p["id"], t), {})
    return {"f": f, "le": lenerr(f["words"], h["words"]), "coh": e.get("coh"), "int": e.get("int"), "uint": e.get("user_int"),
            "ans": e.get("ans"), "keep": (1.0 if t == A.get("text") else k.get("keep")), "fb": fb,
            "cost": rec.get("cost", 0), "lat": rec.get("latency"), "llm": rec.get("llm_calls", 1), "jev": rec.get("jev_calls", 0),
            "rounds": rec.get("rounds"), "words": f["words"], "hw": h["words"]}


def summarize(items, pts, hum, boot=False):
    conv = [p["conv_id"] for p in pts]
    out = {"n": len(items)}
    for k in RATE_KEYS + ["broken", "artefact", "ends_q"]:
        out[k] = float(np.mean([it["f"][k] for it in items]))
    out["dev_rates"] = {k: out[k] - hum[k] for k in RATE_KEYS}
    out["sum_abs_rates"] = float(sum(abs(v) for v in out["dev_rates"].values()))
    out["lenerr"] = float(np.mean([it["le"] for it in items]))
    out["D"] = out["lenerr"] + out["sum_abs_rates"]
    out["words_med"] = float(st.median([it["words"] for it in items]))
    out["len_ratio_med"] = float(st.median([(it["words"] + 1) / (it["hw"] + 1) for it in items]))
    for k in ("coh", "keep", "ans"):
        v = [it[k] for it in items if it[k] is not None]
        out[k] = float(np.mean(v)) if v else None
        out[k + "_lt05"] = float(np.mean([x < 0.5 for x in v])) if v else None
    gaps = [it["int"] - it["uint"] for it in items if it["int"] is not None and it["uint"] is not None]
    out["int_gap"] = float(np.mean(gaps)) if gaps else None
    out["int_over"] = float(np.mean([g > 0.5 for g in gaps])) if gaps else None
    out["fallback"] = float(np.mean([it["fb"] for it in items]))
    out["cost_per_resp"] = float(np.mean([it["cost"] for it in items]))
    lats = [it["lat"] for it in items if it["lat"]]
    out["lat_p50"] = float(np.percentile(lats, 50)) if lats else None
    out["lat_p90"] = float(np.percentile(lats, 90)) if lats else None
    out["llm_calls"] = float(np.mean([it["llm"] for it in items]))
    out["jev_calls"] = float(np.mean([it["jev"] for it in items]))
    r = [it["rounds"] for it in items if it["rounds"] is not None]
    if r:
        out["rounds_mean"] = float(np.mean(r)); out["rounds_dist"] = {str(k): r.count(k) for k in sorted(set(r))}
    if boot:
        out["ci"] = {
            "lenerr": cboot([it["le"] for it in items], conv),
            "q": cboot([float(it["f"]["q"]) for it in items], conv),
            "excl": cboot([float(it["f"]["excl"]) for it in items], conv),
            "coh": cboot([it["coh"] for it in items], conv),
            # D por item: erro de tamanho + desvio de cada vício do item em relação à taxa humana (média = D agregado aprox.)
        }
    return out


def human_rates(pts):
    H = [feats(p["human"], p) for p in pts]
    return {k: float(np.mean([h[k] for h in H])) for k in RATE_KEYS + ["broken", "ends_q"]}


def main(split, models=None):
    pts = [p for p in load_points() if split == "all" or p["split"] == split]
    G = load_gen()
    ev, kp = load(EV), load(KP)
    models = models or sorted({m for (m, c, i) in G})
    hum = human_rates(pts)
    # humano: coerência/intensidade médias pelo Jev (média das avaliações nos lotes dos 4 modelos)
    hcoh = [ev[(m, p["id"], p["human"])]["coh"] for p in pts for m in models if (m, p["id"], p["human"]) in ev]
    hint = [ev[(m, p["id"], p["human"])]["int"] - ev[(m, p["id"], p["human"])]["user_int"] for p in pts for m in models
            if (m, p["id"], p["human"]) in ev]
    hans = [ev[(m, p["id"], p["human"])]["ans"] for p in pts for m in models
            if (m, p["id"], p["human"]) in ev and "ans" in ev[(m, p["id"], p["human"])]]
    res = {"split": split, "n_points": len(pts), "human": dict(hum, coh=float(np.mean(hcoh)) if hcoh else None,
                                                                  int_gap=float(np.mean(hint)) if hint else None,
                                                                  int_over=float(np.mean([g > 0.5 for g in hint])) if hint else None,
                                                                  ans=float(np.mean(hans)) if hans else None,
                                                                  words_med=float(st.median(feats(p["human"])["words"] for p in pts))),
           "models": {}}
    for m in models:
        conds = sorted({c for (mm, c, i) in G if mm == m}, key=lambda c: (ORDER.index(c.replace("+N", "")) if c.replace("+N", "") in ORDER else 99, c))
        res["models"][m] = {}
        for c in conds:
            its, pp = [], []
            for p in pts:
                rec = G.get((m, c, p["id"]))
                if rec is None:
                    continue
                its.append(per_item(p, rec, G.get((m, "A", p["id"]), {}), ev, kp, m)); pp.append(p)
            if len(its) < 0.9 * len(pts):
                continue
            res["models"][m][c] = summarize(its, pp, hum, boot=(split == "test"))
    out = os.path.join(ADATA, f"b4_results_{split}.json")
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print("saved", out, os.path.getsize(out))
    return res


def table(res, keys=("D", "lenerr", "sum_abs_rates", "words_med", "q", "excl", "emoji", "laugh", "llmish", "template", "perf",
                     "coh", "keep", "int_gap", "broken", "cost_per_resp", "lat_p50")):
    print("human", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["human"].items()})
    for m, d in res["models"].items():
        print("==", m)
        print("cond".ljust(10), " ".join(k[:7].rjust(7) for k in keys))
        for c, s in d.items():
            print(c.ljust(10), " ".join((f"{s[k]:7.3f}" if isinstance(s.get(k), float) else str(s.get(k))[:7].rjust(7)) for k in keys))


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else None)
    table(r)
