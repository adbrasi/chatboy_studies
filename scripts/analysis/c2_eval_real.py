"""c2 — avaliação do Jev 1 (detecção) e do Jev 2 (magnitude) contra o ouro do luna nos turnos reais.
Arquiteturas do Jev 1: DIM (Noul por dimensão/direção), BIP (Score bipolar), EVT (Nouls de evento -> mapa em código),
COMB (DIM+EVT: média, máximo, regressão logística ajustada no dev) — cada uma sem e com o estado no state.
Dev e teste separados por conversa; limiares e pesos escolhidos no dev, números reportados no teste, IC 95% por bootstrap
por conversa. Saída: analysis/data/c2_eval_real.json"""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from c2_common import (PROC, ADATA, DIMS, EVENTS, evt_to_dims, auc, f1, boot_fn, jdump, jl_load, level_from_choice)

T = {json.loads(l)["id"]: json.loads(l) for l in open(f"{PROC}/c2_real_turns.jsonl")}
G = {d["id"]: d["gold"] for d in jl_load(f"{PROC}/c2_gold_luna.jsonl") if d.get("gold")}
NS = {d["id"]: d for d in jl_load(f"{PROC}/c2_jev_real_nostate.jsonl")}
ST = {d["id"]: d for d in jl_load(f"{PROC}/c2_jev_real_state.jsonl")}
IDS = [i for i in T if i in G and i in NS and i in ST]
DD = [f"{d}_{dr}" for d in DIMS for dr in ("up", "down")]


def gold_y(g, dd, minlvl=1):
    d, dr = dd.rsplit("_", 1)
    x = g["dims"].get(d, ["none", 0])
    try:
        return int(x[0] == dr and int(x[1]) >= minlvl)
    except Exception:
        return 0


def gold_lvl(g, dd):
    d, dr = dd.rsplit("_", 1)
    x = g["dims"].get(d, ["none", 0])
    try:
        return int(x[1]) if x[0] == dr else 0
    except Exception:
        return 0


def scores(a1, pending):
    ev = {e: a1.get("e_" + e, {}).get("noul", 0.0) for e in EVENTS}
    evd = evt_to_dims(ev, pending=pending)
    out = {}
    for dd in DD:
        d, dr = dd.rsplit("_", 1)
        b = a1[f"b_{d}"]["score"]
        out[dd] = {"DIM": a1[f"d_{dd}"]["noul"], "BIP": (b - 2) / 2 if dr == "up" else (2 - b) / 2, "EVT": evd[dd]}
        out[dd]["COMB_mean"] = (out[dd]["DIM"] + out[dd]["EVT"]) / 2
        out[dd]["COMB_max"] = max(out[dd]["DIM"], out[dd]["EVT"])
    return out


rows = []
for i in IDS:
    t = T[i]
    pend = bool(ST[i]["snap"]["unresolved"])
    rows.append({"id": i, "conv": t["conv"], "corpus": t["corpus"], "split": t["split"], "g": G[i],
                 "ns": scores(NS[i]["a1"], True), "st": scores(ST[i]["a1"], pend),
                 "ev_ns": {e: NS[i]["a1"]["e_" + e]["noul"] for e in EVENTS},
                 "ev_st": {e: ST[i]["a1"]["e_" + e]["noul"] for e in EVENTS}})
dev = [r for r in rows if r["split"] == "dev"]
test = [r for r in rows if r["split"] == "test"]
ARCH = ["DIM", "BIP", "EVT", "COMB_mean", "COMB_max", "COMB_LR"]


# ---- COMB_LR: regressão logística agrupada (mesmos pesos para todas as dimensões + intercepto por dimensão), no dev
def lr_feats(r, var, dd):
    s = r[var][dd]
    return [s["DIM"], s["EVT"], s["BIP"]] + [1.0 if dd == x else 0.0 for x in DD]


LR = {}
for var in ("ns", "st"):
    X = [lr_feats(r, var, dd) for r in dev for dd in DD]
    y = [gold_y(r["g"], dd) for r in dev for dd in DD]
    LR[var] = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
for var in ("ns", "st"):
    for r in rows:
        ps = LR[var].predict_proba([lr_feats(r, var, dd) for dd in DD])[:, 1]
        for dd, p in zip(DD, ps):
            r[var][dd]["COMB_LR"] = float(p)


def pooled_auc(rs, var, arch, minlvl=1, dds=DD):
    y = [gold_y(r["g"], dd, minlvl) for r in rs for dd in dds]
    s = [r[var][dd][arch] for r in rs for dd in dds]
    return auc(y, s)


def macro_auc(rs, var, arch, minlvl=1, min_pos=5):
    vals = []
    for dd in DD:
        y = [gold_y(r["g"], dd, minlvl) for r in rs]
        if sum(y) >= min_pos and len(y) - sum(y) >= min_pos:
            vals.append(auc(y, [r[var][dd][arch] for r in rs]))
    return float(np.mean(vals)) if vals else float("nan")


def best_thr(rs, var, arch):
    y = [gold_y(r["g"], dd) for r in rs for dd in DD]
    s = np.array([r[var][dd][arch] for r in rs for dd in DD])
    best = (0, 0.5)
    for th in np.unique(np.round(s, 2)):
        f = f1(y, s >= th)[0]
        if f > best[0]:
            best = (f, float(th))
    return best[1]


def f1_at(rs, var, arch, th):
    y = [gold_y(r["g"], dd) for r in rs for dd in DD]
    s = np.array([r[var][dd][arch] for r in rs for dd in DD])
    return f1(y, s >= th)


res = {"n": {"dev": len(dev), "test": len(test), "test_convs": len({r["conv"] for r in test})},
       "gold_rates_test": {dd: round(float(np.mean([gold_y(r["g"], dd) for r in test])), 4) for dd in DD},
       "gold_pos_test": {dd: int(sum(gold_y(r["g"], dd) for r in test)) for dd in DD},
       "arch": {}}
groups = [r["conv"] for r in test]
for var in ("ns", "st"):
    for arch in ARCH:
        th = best_thr(dev, var, arch)
        key = f"{arch}|{'com_estado' if var == 'st' else 'sem_estado'}"
        f, p, rc = f1_at(test, var, arch, th)
        res["arch"][key] = {
            "dev_pooled_auc": round(pooled_auc(dev, var, arch), 4),
            "test_pooled_auc": boot_fn(lambda rs: pooled_auc(rs, var, arch), test, groups, iters=400),
            "test_pooled_auc_lvl2": boot_fn(lambda rs: pooled_auc(rs, var, arch, 2), test, groups, iters=400),
            "test_macro_auc": round(macro_auc(test, var, arch), 4),
            "thr_dev": th, "test_f1": boot_fn(lambda rs: f1_at(rs, var, arch, th)[0], test, groups, iters=400),
            "test_prec": round(p, 4), "test_rec": round(rc, 4)}
        print(key, res["arch"][key]["test_pooled_auc"], res["arch"][key]["test_f1"], flush=True)

# ---- por dimensão (melhor arquitetura de cada variante, e DIM)
res["per_dim_test"] = {}
for dd in DD:
    y = [gold_y(r["g"], dd) for r in test]
    row = {"pos": int(sum(y))}
    for var in ("ns", "st"):
        for arch in ("DIM", "EVT", "COMB_LR", "BIP"):
            row[f"{arch}|{var}"] = round(auc(y, [r[var][dd][arch] for r in test]), 3) if 3 <= sum(y) < len(y) else None
    res["per_dim_test"][dd] = row

# ---- diferença pareada com × sem estado (mesmos turnos), por arquitetura
res["state_minus_nostate"] = {}
for arch in ARCH:
    res["state_minus_nostate"][arch] = boot_fn(lambda rs: pooled_auc(rs, "st", arch) - pooled_auc(rs, "ns", arch), test, groups, iters=400)

# ---- eventos
res["events_test"] = {}
for e in EVENTS:
    y = [int(e in (r["g"].get("events") or [])) for r in test]
    if sum(y) >= 3:
        res["events_test"][e] = {"pos": int(sum(y)), "auc_ns": round(auc(y, [r["ev_ns"][e] for r in test]), 3),
                                 "auc_st": round(auc(y, [r["ev_st"][e] for r in test]), 3),
                                 "f1_ns@0.5": round(f1(y, [r["ev_ns"][e] >= 0.5 for r in test])[0], 3),
                                 "f1_st@0.5": round(f1(y, [r["ev_st"][e] >= 0.5 for r in test])[0], 3)}
    else:
        res["events_test"][e] = {"pos": int(sum(y))}

# ---- Jev 2 (magnitude) nos turnos reais: onde o Jev 2 foi chamado (Noul >= 0,3), magnitude estimada x nível do ouro
mag = defaultdict(list)
for r in rows:
    s = ST[r["id"]]
    if not s.get("a2"):
        continue
    for d, dr in s["active"]:
        dd = f"{d}_{dr}"
        sc = s["a2"].get(f"s_{dd}"); ch = s["a2"].get(f"n_{dd}")
        if not sc or not ch:
            continue
        mag["split"].append(r["split"]); mag["conv"].append(r["conv"])
        mag["gold"].append(gold_lvl(r["g"], dd)); mag["score"].append(sc["score"]); mag["num"].append(int(ch["choice"]))
        mag["p"].append(s["a1"][f"d_{dd}"]["noul"]); mag["bip"].append(r["st"][dd]["BIP"])
idx = [k for k, sp in enumerate(mag["split"]) if sp == "test"]
M = [{k: mag[k][j] for k in mag} for j in idx]
res["jev2_real_test"] = {"n_pairs": len(M), "gold_nonzero": int(sum(m["gold"] > 0 for m in M))}
for key in ("score", "num", "p", "bip"):
    res["jev2_real_test"][f"spearman_{key}"] = boot_fn(
        lambda ms: spearmanr([m["gold"] for m in ms], [m[key] for m in ms]).correlation, M, [m["conv"] for m in M], iters=400)
    nz = [m for m in M if m["gold"] > 0]
    res["jev2_real_test"][f"spearman_{key}_goldpos"] = round(float(spearmanr([m["gold"] for m in nz], [m[key] for m in nz]).correlation), 3) if len(nz) > 5 else None
jdump(res, f"{ADATA}/c2_eval_real.json")
print(json.dumps(res["state_minus_nostate"], indent=0))
print(json.dumps(res["jev2_real_test"], indent=0))
