"""c2 — resume c2_scen_results.json: taxa de critérios por configuração (todos, cenários-dev ímpares, cenários-teste pares),
critérios globais (saturação, oscilação, salto), e exemplos de trajetória. Saída: analysis/data/c2_scen_summary.json"""
import json
import numpy as np
from c2_common import ADATA, DIMS, jdump

R = json.load(open(f"{ADATA}/c2_scen_results.json"))
names = list(next(iter(R.values()))["scenarios"])
DEV = [n for n in names if int(n[1:3]) % 2 == 1]
out = {"configs": {}, "per_crit": {}}
for cfg, r in R.items():
    row = {}
    for sp, ns in (("all", names), ("dev", DEV), ("test", [n for n in names if n not in DEV])):
        spec = [c["ok"] for n in ns for c in r["scenarios"][n]["checks"] if not c["global"]]
        gl = [c["ok"] for n in ns for c in r["scenarios"][n]["checks"] if c["global"]]
        row[sp] = {"specific": round(float(np.mean(spec)), 3), "n_spec": len(spec), "global": round(float(np.mean(gl)), 3),
                   "scen_all_pass": int(sum(all(c["ok"] for c in r["scenarios"][n]["checks"]) for n in ns))}
    sw = []
    for n in names:
        tr = r["scenarios"][n]["traj"]
        sw.append(sum(1 for x in tr[1:] if x["mode_changed"]))
    row["mode_switches_mean"] = round(float(np.mean(sw)), 2)
    row["max_jump"] = round(max(abs(x["after"][d] - x["before"][d]) for n in names for x in r["scenarios"][n]["traj"] for d in DIMS), 3)
    row["n_jev1"] = sum(r["scenarios"][n]["n_jev1"] for n in names)
    row["n_jev2"] = sum(r["scenarios"][n]["n_jev2"] for n in names)
    out["configs"][cfg] = row
    for n in names:
        for c in r["scenarios"][n]["checks"]:
            out["per_crit"].setdefault(f"{n} :: {c['crit']}", {})[cfg] = c["ok"]
jdump(out, f"{ADATA}/c2_scen_summary.json")
print(f"{'cfg':12s} {'all':>6s} {'dev':>6s} {'test':>6s} {'glob':>6s} {'allpass':>7s} sw  jump  jev1 jev2")
for cfg, row in out["configs"].items():
    print(f"{cfg:12s} {row['all']['specific']:6.3f} {row['dev']['specific']:6.3f} {row['test']['specific']:6.3f} "
          f"{row['all']['global']:6.3f} {row['all']['scen_all_pass']:7d} {row['mode_switches_mean']:.2f} {row['max_jump']:.3f} "
          f"{row['n_jev1']} {row['n_jev2']}")
