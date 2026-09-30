"""c2 — avaliação OFFLINE da física v3d nos cenários (sem chamar a API: os créditos do OpenRouter acabaram no meio da
retomada). Reexecuta FULL@v3c só com o cache (as respostas do Jev 1/Jev 2 condicionadas à trajetória da v3c) e aplica a
física v3d sobre essas mesmas detecções. Aproximação declarada: na execução real, o bloco de estado do Jev 1 mudaria com
a trajetória da v3d (pendências resolvidas mais cedo). Saída: adiciona 'FULL@v3d_offline' em c2_scen_results.json."""
import json, copy
import numpy as np
import c2_scen_run as R
from c2_common import jev, physics_step, RelState, DEFAULT_BASE, jdump, ADATA

orig = jev.ask


def cache_only(st, q):
    k = jev._hash(st, q)
    c = jev._load_cache()
    if k not in c:
        raise KeyError("cache miss")
    return c[k]


def capture(name):
    seq = []

    def spy(st, q):
        a = cache_only(st, q)
        seq.append((st.get("user_message"), "e_apology" in q, a))
        return a
    jev.ask = spy
    try:
        R.run(name, "FULL@v3c")
    finally:
        jev.ask = orig
    # agrupa por mensagem: (a1, a2)
    out, cur = [], None
    for msg, is1, a in seq:
        if is1:
            cur = {"msg": msg, "a1": a, "a2": None}; out.append(cur)
        else:
            cur["a2"] = a
    return out


def run_offline(name, spec):
    det = capture(name)
    turns = R.SC[name]["turns"]
    base = dict(DEFAULT_BASE, **R.BASES.get(name, {}))
    rel = RelState(base, spec=spec)
    traj, hist, k = [], [], 0
    for i, t in enumerate(turns):
        if t["from"] == R.U:
            prev = hist[-1] if hist else None
            gap = (t["t_hours"] - prev["t_hours"]) if prev else None
            cw = bool(prev and prev["from"] == R.C)
            d = det[k]; k += 1
            assert d["msg"] == t["text"]
            rec = physics_step(rel, R.C, R.U, t["text"], d["a1"], d["a2"], t["t_hours"], gap_hours=gap, char_waiting=cw, mag="score")
            ev = {kk[2:]: round(v["noul"], 2) for kk, v in d["a1"].items() if kk.startswith("e_") and v["noul"] >= 0.5}
            traj.append({"i": i, "tag": t["tag"], "text": t["text"], "day": t["day"], "time": t["time"], "events": ev,
                         **{kk: rec[kk] for kk in ("before", "after", "applied", "mode", "mode_changed", "unresolved",
                                                   "promises", "resolved")}})
        hist.append({"from": t["from"], "text": t["text"], "t_hours": t["t_hours"]})
    return {"traj": traj, "base": base, "n_jev1": k, "n_jev2": sum(1 for d in det if d["a2"]), "lat_mean": None}


if __name__ == "__main__":
    path = f"{ADATA}/c2_scen_results.json"
    allres = json.load(open(path))
    for label, sv in (("FULL@v3c_offline_check", "v3c"), ("FULL@v3d_offline", "v3d"),
                      ("FULL@v3c+relief_offline", "v3c+relief"), ("FULL@v3e_offline", "v3e")):
        out = {}
        for n in R.SC:
            r = run_offline(n, R.SPECS[sv])
            r["checks"] = R.check(n, r)
            out[n] = r
        sc = [c["ok"] for r in out.values() for c in r["checks"] if not c["global"]]
        gl = [c["ok"] for r in out.values() for c in r["checks"] if c["global"]]
        allres[label] = {"scenarios": out, "pass_specific": round(float(np.mean(sc)), 3), "n_specific": len(sc),
                         "pass_global": round(float(np.mean(gl)), 3), "n_global": len(gl),
                         "scen_all_pass": sum(all(c["ok"] for c in r["checks"]) for r in out.values())}
        print(label, allres[label]["pass_specific"], allres[label]["pass_global"], allres[label]["scen_all_pass"])
    jdump(allres, path)
