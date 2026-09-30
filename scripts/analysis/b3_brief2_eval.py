"""b3 — avaliação do Briefing v2: métricas de código (as do relatório 09), coincidência de movimento/família/tom com o
humano (mesmo rotulador Jev do ouro, uma resposta por chamada) e juiz DECOMPOSTO (traços atômicos, Jev; sem
"qual é humana?"). IC bootstrap por conversa. Uso: python3 b3_brief2_eval.py label | analyze"""
import json, math, os, re, sys
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from b3_common import (load_points, kv_load, kv_save, jl_load, A9P, base_state, jask_many, compact, noul, choice,
                       MOVE_OPTS, TONE_OPTS, FAMILY_OPTS, MOVES, FAMILY, BOT, USER, feats, cboot, cboot_diff, jdump)
from a9_extra import PERF, PLAIN, REC, ART
import jev

TRAITS = {
    "t_direct": f"Does `reply` respond directly to `last_message`?",
    "t_specific": "Does `reply` pick up a specific word or detail from `last_message` (rather than the general topic)?",
    "t_paraphrase": f"Does `reply` repeat or paraphrase what {USER} just said?",
    "t_overvalidate": f"Does `reply` validate, praise or sympathize with {USER} more than the moment needs?",
    "t_toomuch": "Does `reply` do several things at once (for example react, then comment, then ask a question)?",
    "t_intense": "Is `reply` more emotional or enthusiastic than `last_message`?",
    "t_forced": "Does `reply` use slang, emoji or internet expressions in a forced or exaggerated way?",
    "t_assistant": "Does `reply` sound like a customer-service assistant, a coach or a therapist?",
    "t_generic": "Could `reply` be sent as an answer to almost any message?",
    "t_invents": "Does `reply` mention facts, plans or shared memories that nothing in the conversation supports?",
    "t_own": f"Does `reply` add something about {BOT} (own news, experience or opinion)?",
    "t_implicit": "Does `reply` leave its main point implicit (hinted or understated) instead of saying it outright?",
}
LQ = {"g_move": choice(f"Which move does `reply` make, as {BOT}'s reply to `last_message`?", MOVE_OPTS),
      "g_family": choice(f"What kind of move is `reply`, as {BOT}'s reply to `last_message`?", FAMILY_OPTS),
      "g_tone": choice("Which tone does `reply` have?", TONE_OPTS)}
LQ.update({k: noul(v) for k, v in TRAITS.items()})
ACTORS = ["flash", "luna"]  # rodada reduzida (créditos): mercury/deepseek não rodaram
CONDS = ["A", "B1", "N2", "B2"]  # K2/O2 construídos mas não gerados


def replies():
    a9 = {p["id"]: p for p in jl_load(A9P)}
    G = kv_load("gen2")
    R = {}
    for i, g in G.items():
        R[i] = {"H": a9[i]["human"], "D_haiku_hist": a9[i]["gen"].get("D"), "flash|B1_a9": a9[i]["gen"].get("B")}
        for k, v in g.items():
            if v:
                R[i][k] = v
    return a9, R


def run_label():
    a9, R = replies()
    L = kv_load("brief2_labels")
    items, keys = [], []
    for i, rr in R.items():
        p = a9[i]
        st0 = base_state(p)
        for c, t in rr.items():
            if not t or f"{c}" in L.get(i, {}):
                continue
            st = dict(st0, reply={"from": BOT, "text": t})
            items.append((st, LQ)); keys.append((i, c))
    print("to label", len(items))
    for s in range(0, len(items), 300):
        res = jask_many(items[s:s + 300], workers=4)
        for (i, c), a in zip(keys[s:s + 300], res):
            if a:
                L.setdefault(i, {})[c] = compact(a)
        kv_save("brief2_labels", L)
        print(s + len(res), jev.summary(), flush=True)


def code_m(t, h):
    f, fh = feats(t), feats(h)
    tl = t.strip().lower()
    return {"words": f["n_words"], "q": float(f["has_q"]), "laugh": float(f["laugh"]), "emoji": float(f["emoji"]),
            "excl": float(f["excl"]), "llmish": float(f["llmish"]), "slang": float(f["slang_n"] > 0),
            "multi": float(f["n_bubbles"] > 1), "lower": float(f["starts_lower"]),
            "perf": float(bool(re.search(PERF, tl))), "plain": float(bool(re.search(PLAIN, tl))),
            "recip": float(bool(re.search(REC, tl))), "art": float(bool(re.search(ART, t))),
            "lenerr": abs(math.log2((f["n_words"] + 1) / (fh["n_words"] + 1)))}


def analyze():
    a9, R = replies()
    L = kv_load("brief2_labels")
    B = kv_load("briefs2")
    ids = sorted(R)
    grp = [a9[i]["conv_id"] for i in ids]
    conds = ["H", "D_haiku_hist"] + [f"{a}|{c}" for a in ACTORS for c in CONDS]
    out = {"n": len(ids), "conds": {}}
    hum = {i: code_m(R[i]["H"], R[i]["H"]) for i in ids}
    hw = [hum[i]["words"] for i in ids]
    for c in conds:
        rows = {i: code_m(R[i][c], R[i]["H"]) for i in ids if R[i].get(c)}
        if not rows:
            continue
        o = {}
        for k in ["q", "laugh", "emoji", "excl", "llmish", "slang", "multi", "lower", "perf", "plain", "recip", "art", "lenerr"]:
            o[k] = cboot([rows[i][k] if i in rows else None for i in ids], grp)
        o["words_median"] = cboot([rows[i]["words"] if i in rows else None for i in ids], grp, stat=np.median)
        sub = [i for i in ids if i in rows]
        o["len_spearman"] = round(float(spearmanr([rows[i]["words"] for i in sub], [hum[i]["words"] for i in sub]).correlation), 3)
        # movimento / família / tom (mesmo rotulador; humano rotulado no mesmo formato)
        mv, fm, tn, prof, tr = [], [], [], [], {k: [] for k in TRAITS}
        for i in ids:
            lh, lc = L.get(i, {}).get("H"), L.get(i, {}).get(c)
            if not lh or not lc:
                mv.append(None); fm.append(None); tn.append(None); prof.append(None)
                for k in TRAITS:
                    tr[k].append(None)
                continue
            mv.append(float(lc["g_move"] == lh["g_move"]))
            fm.append(float(lc["g_family"] == lh["g_family"]))
            tn.append(float(lc["g_tone"] == lh["g_tone"]))
            prof.append(float(np.mean([abs(lc[k] - lh[k]) for k in TRAITS])))
            for k in TRAITS:
                tr[k].append(lc[k])
        o["move_match"] = cboot(mv, grp); o["family_match"] = cboot(fm, grp); o["tone_match"] = cboot(tn, grp)
        o["trait_profile_dist"] = cboot(prof, grp)
        o["traits_mean"] = {k: cboot(v, grp) for k, v in tr.items()}
        mvd = Counter(L[i][c]["g_move"] for i in ids if c in L.get(i, {}))
        o["move_dist"] = dict(mvd.most_common(6))
        out["conds"][c] = o
    # distância agregada dos traços ao humano (|taxa média cond - taxa média humano|, média nos 12 traços)
    H = out["conds"]["H"]["traits_mean"]
    for c, o in out["conds"].items():
        if c == "H":
            continue
        o["trait_rate_gap"] = round(float(np.mean([abs(o["traits_mean"][k][0] - H[k][0]) for k in TRAITS
                                                  if o["traits_mean"][k] and H[k]])), 4)
        o["code_gap"] = round(float(np.mean([abs(o[k][0] - out["conds"]["H"][k][0]) for k in
                                             ["q", "laugh", "emoji", "excl", "llmish", "slang", "multi", "perf", "recip"]])), 4)
    # diferenças pareadas-chave (por ator): A→N2 (código), N2→K2 (corpus), K2→B2 (Jev online), B2→O2 (teto), B1→B2
    def per_point(c, metric):
        v = []
        for i in ids:
            if metric == "lenerr":
                v.append(code_m(R[i][c], R[i]["H"])["lenerr"] if R[i].get(c) else None)
            elif metric == "prof":
                lh, lc = L.get(i, {}).get("H"), L.get(i, {}).get(c)
                v.append(float(np.mean([abs(lc[k] - lh[k]) for k in TRAITS])) if lh and lc else None)
            else:
                lh, lc = L.get(i, {}).get("H"), L.get(i, {}).get(c)
                v.append(float(lc[metric] == lh[metric]) if lh and lc else None)
        return v
    out["paired"] = {}
    for a in ACTORS:
        for x, y in [("N2", "A"), ("B2", "N2"), ("B2", "B1"), ("B2", "A"), ("B1", "A")]:
            cx, cy = f"{a}|{x}", f"{a}|{y}"
            out["paired"][f"{a}: {x}-{y}"] = {m: cboot_diff(per_point(cx, m), per_point(cy, m), grp)
                                               for m in ("g_move", "g_family", "lenerr", "prof")}
    # aderência do briefing: pedido x feito (B2)
    adh = {}
    for a in ACTORS:
        for c in ("N2", "B2"):
            q_ok, lg_ok, ex_ok = [], [], []
            for i in ids:
                meta, t = B[i].get(c + "_meta"), R[i].get(f"{a}|{c}")
                if not meta or not t:
                    continue
                f = feats(t)
                q_ok.append(float(f["has_q"] == meta["q"]))
                lg_ok.append(float(f["laugh"] == meta["laugh"]))
                ex_ok.append(float(f["excl"] == meta["excl"]))
            adh[f"{a}|{c}"] = {"q": round(float(np.mean(q_ok)), 3), "laugh": round(float(np.mean(lg_ok)), 3),
                               "excl": round(float(np.mean(ex_ok)), 3)}
    out["adherence"] = adh
    jdump("b3_brief2_results.json", out)
    # tabela curta no terminal
    for c in conds:
        o = out["conds"].get(c)
        if not o:
            continue
        print(f"{c:18s} w {o['words_median'][0]:>4} q {o['q'][0]:.2f} lg {o['laugh'][0]:.2f} em {o['emoji'][0]:.2f} ! {o['excl'][0]:.2f} "
              f"llm {o['llmish'][0]:.2f} perf {o['perf'][0]:.2f} rec {o['recip'][0]:.2f} lenerr {o['lenerr'][0]:.2f} "
              f"rho {o['len_spearman']:.2f} mv {o['move_match'][0] if o['move_match'] else 0:.2f} fam {o['family_match'][0] if o['family_match'] else 0:.2f} "
              f"tone {o['tone_match'][0] if o['tone_match'] else 0:.2f} prof {o['trait_profile_dist'][0] if o['trait_profile_dist'] else 0:.3f} "
              f"gap {o.get('trait_rate_gap', 0):.3f}")


if __name__ == "__main__":
    {"label": run_label, "analyze": analyze}[sys.argv[1]]()
