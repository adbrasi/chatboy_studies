"""c1 — métricas por condição (reutilizadas por todos os experimentos c1).
summarize(items) com items = [{"p": ponto, "text": resposta final, "rec": registro de geração (opcional)}]:
  forma (D do relatório 13 e os 10 vícios, com IC por bootstrap por conversa), bolhas, tempos propostos × reais,
  invenção da fala do usuário, formato quebrado, e (se a avaliação Jev existir) movimento, coerência e o escore
  código + banco do Jev (P(LLM), ↓ = mais humano)."""
import math, statistics as st
from collections import Counter
import numpy as np
import c1_common as C


def human_block(pts):
    hf = [C.human_feats(p) for p in pts]
    out = {"n": len(pts), "words_med": float(st.median(f["words"] for f in hf))}
    for k in C.RATE_KEYS:
        out[k] = float(np.mean([f[k] for f in hf]))
    out["bubbles_mean"] = float(np.mean([p["human_n_msgs"] for p in pts]))
    out["multi"] = float(np.mean([p["human_n_msgs"] > 1 for p in pts]))
    lat = [p["real_latency"] for p in pts if p["real_latency"] is not None]
    out["lat_med"] = float(np.median(lat)) if lat else None
    gaps = [g for p in pts for g in p["real_gaps"]]
    out["gap_med"] = float(np.median(gaps)) if gaps else None
    J = [C.jev_get(p, p["human"]) for p in pts]
    bs = [C.bank_score(p, p["human"], j) for p, j in zip(pts, J) if j]
    out["bank"] = float(np.mean(bs)) if bs else None
    out["bank_ci"] = C.cboot([C.bank_score(p, p["human"], j) if j else None for p, j in zip(pts, J)], [p["conv_id"] for p in pts])
    coh = [j["e_coh"] for j in J if j]
    out["coh"] = float(np.mean(coh)) if coh else None
    mv = [j["e_move"] for j in J if j]
    out["move_entropy"] = C.entropy(mv) if mv else None
    out["move_dist"] = dict(Counter(mv).most_common(6))
    return out


def summarize(items, boot=True, jev=True):
    pts = [it["p"] for it in items]
    groups = [p["conv_id"] for p in pts]
    fs = [C.feats(it["text"], it["p"]) for it in items]
    hfs = [C.human_feats(p) for p in pts]
    les = [C.lenerr(f["words"], h["words"]) for f, h in zip(fs, hfs)]
    o = {"n": len(items)}
    o["D"] = C.D_of(fs, les, hfs)
    if boot:
        o["D_ci"] = C.boot_D(fs, les, hfs, groups)
    o["lenerr"] = float(np.mean(les))
    o["words_med"] = float(st.median(f["words"] for f in fs))
    for k in C.RATE_KEYS:
        o[k] = float(np.mean([f[k] for f in fs]))
    o["empty"] = float(np.mean([f["empty"] for f in fs]))
    # bolhas: nº de linhas da resposta final × nº real de bolhas do humano
    nb = [max(1, len([l for l in (it["text"] or "").split("\n") if l.strip()])) for it in items]
    hb = [p["human_n_msgs"] for p in pts]
    o["bubbles_mean"] = float(np.mean(nb))
    o["multi"] = float(np.mean([n > 1 for n in nb]))
    o["bubbles_mae"] = float(np.mean([abs(a - b) for a, b in zip(nb, hb)]))
    o["bubbles_exact"] = float(np.mean([min(a, 4) == min(b, 4) for a, b in zip(nb, hb)]))
    # registro de geração: invenção, formato, tempos propostos
    recs = [it.get("rec") or {} for it in items]
    for k in ("invented_user", "other_speaker", "broken", "plain", "deviation", "restart", "fallback"):
        v = [r.get(k) for r in recs if k in r]
        o[k] = float(np.mean([bool(x) for x in v])) if v else None
    if boot and any("invented_user" in r for r in recs):
        o["invented_ci"] = C.cboot([float(bool(r.get("invented_user"))) if "invented_user" in r else None for r in recs], groups)
        o["broken_ci"] = C.cboot([float(bool(r.get("broken"))) if "broken" in r else None for r in recs], groups)
    lat = [(r.get("prop_latency"), p["real_latency"]) for r, p in zip(recs, pts)
           if r.get("prop_latency") is not None and p["real_latency"] is not None]
    if lat:
        pl = [a for a, b in lat]
        o["prop_lat_med"] = float(np.median(pl))
        o["real_lat_med"] = float(np.median([b for a, b in lat]))
        o["prop_lat_share"] = len(lat) / len(items)
        o["prop_lat_nonpos"] = float(np.mean([a <= 0 for a in pl]))
        o["lat_logerr"] = float(np.mean([abs(math.log2((max(a, 0) + 1) / (b + 1))) for a, b in lat]))
        try:
            from scipy.stats import spearmanr
            o["lat_spearman"] = float(spearmanr([a for a, b in lat], [b for a, b in lat]).correlation)
        except Exception:
            pass
        g = [x for r in recs for x in (r.get("prop_gaps") or [])]
        o["prop_gap_med"] = float(np.median(g)) if g else None
    o["cost_per_resp"] = float(np.mean([(r.get("cost") or 0) for r in recs])) if recs and recs[0] else None
    L = [r.get("latency") for r in recs if r.get("latency")]
    o["lat_p50"] = float(np.percentile(L, 50)) if L else None
    o["llm_calls"] = float(np.mean([r.get("llm_calls", 1) for r in recs])) if recs and recs[0] else None
    if jev:
        J = [C.jev_get(it["p"], it["text"]) if (it["text"] or "").strip() else None for it in items]
        HJ = [C.jev_get(p, p["human"]) for p in pts]
        mv = [float(j["e_move"] == h["e_move"]) if (j and h) else None for j, h in zip(J, HJ)]
        fm = [float(C.FAMILY[j["e_move"]] == C.FAMILY[h["e_move"]]) if (j and h) else None for j, h in zip(J, HJ)]
        coh = [j["e_coh"] if j else None for j in J]
        bk = [C.bank_score(it["p"], it["text"], j) if j else None for it, j in zip(items, J)]
        o["jev_cov"] = float(np.mean([j is not None for j in J]))
        if o["jev_cov"] > 0.5:
            o["move_match"] = C.cboot(mv, groups) if boot else float(np.nanmean([x for x in mv if x is not None]))
            o["family_match"] = C.cboot(fm, groups) if boot else None
            o["coh"] = C.cboot(coh, groups) if boot else float(np.mean([x for x in coh if x is not None]))
            o["coh_lt05"] = float(np.mean([x < 0.5 for x in coh if x is not None]))
            o["bank"] = C.cboot(bk, groups) if boot else float(np.mean([x for x in bk if x is not None]))
            o["move_entropy"] = C.entropy([j["e_move"] for j in J if j])
            o["move_dist"] = dict(Counter(j["e_move"] for j in J if j).most_common(6))
    return o


def short(o, keys=("D", "lenerr", "words_med", "q", "excl", "emoji", "laugh", "llmish", "echo2", "bubbles_mean", "multi",
                   "invented_user", "broken", "plain", "prop_lat_med", "move_match", "coh", "bank", "cost_per_resp")):
    out = {}
    for k in keys:
        v = o.get(k)
        if isinstance(v, list):
            v = v[0]
        out[k] = None if v is None else round(float(v), 3)
    return out


def table(rows, keys=("D", "lenerr", "words_med", "q", "excl", "emoji", "laugh", "llmish", "bubbles_mean", "multi",
                      "invented_user", "broken", "plain", "prop_lat_med", "move_match", "coh", "bank")):
    print("cond".ljust(22), " ".join(k[:7].rjust(7) for k in keys))
    for name, o in rows:
        s = short(o, keys)
        print(name.ljust(22), " ".join(("   –   " if s[k] is None else f"{s[k]:7.3f}") for k in keys))
