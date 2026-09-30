"""c2 — roda a cascata inteira (Jev 1 com estado -> Jev 2 -> física) nos cenários roteirizados e confere critérios de
plausibilidade PRÉ-REGISTRADOS (escritos antes de rodar; ver CRIT abaixo). Ablações:
  FULL     Jev 1 com estado + Jev 2 Score descritivo + física completa
  CHOICE   igual, magnitude pelo Choice de números (10/20/30...)
  P        magnitude inferida só do Jev 1 (sem Jev 2)
  NOSTATE  Jev 1 sem bloco de estado/pendências (só turnos) + Jev 2 Score + física
  NAIVE    Jev 1 com estado + Jev 2 Score, mas física ingênua: delta cru, sem saturação/decaimento/histerese/AND
Saída: data/processed/c2_scen_results.json (trajetórias + critérios)"""
import json, sys, copy, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from c2_common import (PROC, jev, DIMS, jev1_bank, jev2_questions, make_state, RelState, DEFAULT_BASE, SPEC, physics_step,
                       active_dims, ADATA, jdump)

C, U = "Mia", "Leo"
SC = json.load(open(f"{ADATA}/c2_scenarios.json"))
BASES = {"S01_sincere_apology": {"affection": 0.65}, "S07_jealousy": {"affection": 0.75, "comfort": 0.7},
         "S14_day200_banter": {"comfort": 0.85, "playfulness": 0.8, "trust": 0.8},
         "S08_reconciliation": {"comfort": 0.7, "trust": 0.7}, "S10_constant_affection": {"affection": 0.6}}
KNOWN = {"S14_day200_banter": "about 5 years, best friends", "S03_bad_joke": "a few weeks"}

# v2 = especificação escolhida no dev dos dados reais (c2_replay.py): limiar de detecção 0,74 + humor volta 10%/msg
SPEC2 = copy.deepcopy(SPEC); SPEC2["detect_thr"] = 0.74; SPEC2["routine_thr"] = 0.74; SPEC2["mood_decay_msg"] = 0.1
# v3* = correções do diagnóstico nos cenários-DEV (ímpares); avaliadas nos cenários-TESTE (pares) e no replay real
CELL_THR = json.load(open(f"{ADATA}/c2_cell_thr.json"))["cell_thr"]
SPECS = {"v2": SPEC2}
s3 = copy.deepcopy(SPEC); s3["mood_decay_msg"] = 0.1; s3["cell_thr"] = CELL_THR; SPECS["v3a"] = s3
s3 = copy.deepcopy(s3); s3["delta_interp"] = [0.0, 0.04, 0.10, 0.18, 0.28]; SPECS["v3b"] = s3
s3 = copy.deepcopy(s3); s3["jealousy_rule"] = True
s3["modes"] = dict(s3["modes"], cold=("resentment", 0.40, 0.28, ">"), jealous=("jealousy", 0.35, 0.22, ">")); SPECS["v3c"] = s3
s3 = copy.deepcopy(s3); s3["resolve_v3"] = True; s3["resolve_relief"] = True; SPECS["v3d"] = s3
s4 = copy.deepcopy(SPECS["v3c"]); s4["resolve_relief"] = True; SPECS["v3c+relief"] = s4
s4 = copy.deepcopy(SPECS["v3d"]); s4["pattern_rule"] = True; SPECS["v3e"] = s4
DEV_SC = [k for k in SC if k[1:3].isdigit() and int(k[1:3]) % 2 == 1]
NAIVE_SPEC = copy.deepcopy(SPEC)
NAIVE_SPEC["rate_up"] = {d: 1.0 for d in DIMS}; NAIVE_SPEC["rate_down"] = {d: 1.0 for d in DIMS}
NAIVE_SPEC["half_life_h"] = {d: None for d in DIMS}


class NaiveRel(RelState):
    """física ingênua: soma o delta cru (sem saturação), sem decaimento, sem histerese, sem portões AND."""
    def _apply(self, d, sign, level, mult=1.0):
        if level <= 0:
            return 0.0
        dv = sign * self.spec["delta"][int(level)]
        self.v[d] = float(min(1.0, max(0.0, self.v[d] + dv)))
        return dv

    def update_mode(self):
        prev = self.mode
        act = []
        for m in self.spec["mode_priority"]:
            dim, enter, leave, sense = self.spec["modes"][m]
            val = self.warmth() if dim == "warmth" else self.v[dim]
            if (sense == ">" and val >= enter) or (sense == "<" and val <= enter):
                act.append(m)
        self.mode = act[0] if act else "neutral"
        return prev != self.mode


def naive_step(rel, C, U, msg, a1, a2, t_hours, gap_hours=None, char_waiting=False):
    before = dict(rel.v)
    applied = []
    for d in DIMS:
        for dr, sign in (("up", 1), ("down", -1)):
            p = a1.get(f"d_{d}_{dr}", {}).get("noul", 0.0)
            if p < SPEC["detect_thr"]:
                continue
            s = (a2 or {}).get(f"s_{d}_{dr}")
            lvl = int(min(4, max(0, (s or {}).get("score", 0) + 0.5))) if s else 1
            dv = rel._apply(d, sign, lvl)
            applied.append((d, dr, lvl, dv))
    ch = rel.update_mode()
    rec = {"t": t_hours, "before": before, "after": dict(rel.v), "applied": applied, "mode": rel.mode, "mode_changed": ch,
           "unresolved": [], "promises": [], "resolved": []}
    rel.log.append(rec)
    return rec


def run(name, cfg0):
    if "@" in cfg0:
        cfg, sv = cfg0.split("@"); spec = SPECS[sv]
    else:
        cfg = cfg0.replace("v2", "")
        spec = SPEC2 if cfg0.endswith("v2") else SPEC
    turns = SC[name]["turns"]
    base = dict(DEFAULT_BASE, **BASES.get(name, {}))
    rel = NaiveRel(base, spec=NAIVE_SPEC) if cfg == "NAIVE" else RelState(base, spec=spec)
    if name in KNOWN:
        rel.days_known = KNOWN[name]
    traj, hist, n1, n2, lat = [], [], 0, 0, []
    for i, t in enumerate(turns):
        if t["from"] == U:
            prev = hist[-1] if hist else None
            gap = (t["t_hours"] - prev["t_hours"]) if prev else None
            char_waiting = bool(prev and prev["from"] == C)
            rel.decay(t["t_hours"]) if cfg != "NAIVE" else None
            with_state = cfg != "NOSTATE"
            st = make_state(C, U, hist, t["text"], rel=rel, with_state=with_state, gap_hours=gap)
            nu, npm = (len(rel.unresolved), len(rel.promises)) if with_state else (0, 0)
            t0 = time.time()
            a1 = jev.ask(st, jev1_bank(C, U, nu, npm)); n1 += 1
            act = active_dims(a1, 0.3)
            a2 = None
            if act and cfg != "P":
                st2 = dict(st, detected_effects=[f"user_message may {'raise' if dr == 'up' else 'lower'} {C}'s {d} toward {U}" for d, dr in act])
                a2 = jev.ask(st2, jev2_questions(C, U, act)); n2 += 1
            lat.append(time.time() - t0)
            if cfg == "NAIVE":
                rec = naive_step(rel, C, U, t["text"], a1, a2, t["t_hours"])
            else:
                rec = physics_step(rel, C, U, t["text"], a1, a2, t["t_hours"], gap_hours=gap, char_waiting=char_waiting,
                                   mag={"CHOICE": "choice", "P": "p"}.get(cfg, "score"))
            ev = {k[2:]: round(v["noul"], 2) for k, v in a1.items() if k.startswith("e_") and v["noul"] >= 0.5}
            traj.append({"i": i, "tag": t["tag"], "text": t["text"], "day": t["day"], "time": t["time"], "events": ev,
                         **{k: rec[k] for k in ("before", "after", "applied", "mode", "mode_changed", "unresolved",
                                                "promises", "resolved")}})
        hist.append({"from": t["from"], "text": t["text"], "t_hours": t["t_hours"]})
    return {"traj": traj, "base": base, "n_jev1": n1, "n_jev2": n2, "lat_mean": float(np.mean(lat)) if lat else None}


# ---------------------------------------------------------------- critérios (pré-registrados)
def _at(tr, tag, key="after"):
    for x in tr:
        if x["tag"] == tag:
            return x[key]
    raise KeyError(tag)


def _x(tr, tag):
    for x in tr:
        if x["tag"] == tag:
            return x
    raise KeyError(tag)


def switches(tr):
    return sum(1 for x in tr if x["mode_changed"]) - (1 if tr and tr[0]["mode_changed"] else 0)


def peak(tr, d):
    return max(x["after"][d] for x in tr)


def drop(tr, tag, d="resentment"):
    x = _x(tr, tag)
    return x["before"][d] - x["after"][d]


CRIT = {
    "S01_sincere_apology": [
        ("ressentimento sobe >= +0,15 depois de cancelar+desdenhar", lambda tr, b: _at(tr, "dismiss")["resentment"] >= b["resentment"] + 0.15),
        ("vira pendência", lambda tr, b: len(_x(tr, "dismiss")["unresolved"]) > 0),
        ("depois da desculpa+reparação o ressentimento cai >= 0,10 do pico", lambda tr, b: _at(tr, "repair")["resentment"] <= peak(tr, "resentment") - 0.10),
        ("pendência resolvida até a promessa cumprida", lambda tr, b: len(_x(tr, "kept")["unresolved"]) == 0),
        ("confiança após cumprir >= confiança após o desdém", lambda tr, b: _at(tr, "kept")["trust"] >= _at(tr, "dismiss")["trust"]),
    ],
    "S02_repeated_apology": [
        ("confiança final <= inicial - 0,15", lambda tr, b: tr[-1]["after"]["trust"] <= b["trust"] - 0.15),
        ("desculpa repetida rende menos (queda 3a <= 2a <= 1a + 0,01)",
         lambda tr, b: drop(tr, "apology3") <= drop(tr, "apology2") + 0.01 and drop(tr, "apology2") <= drop(tr, "apology1") + 0.01),
        ("pendência continua aberta no fim", lambda tr, b: len(tr[-1]["unresolved"]) > 0),
        ("ressentimento final >= base + 0,2", lambda tr, b: tr[-1]["after"]["resentment"] >= b["resentment"] + 0.2),
    ],
    "S03_bad_joke": [
        ("ressentimento sobe >= +0,12 com a piada", lambda tr, b: _at(tr, "joke")["resentment"] >= b["resentment"] + 0.12),
        ("vira pendência", lambda tr, b: len(_x(tr, "defensive")["unresolved"]) > 0),
        ("defensividade não alivia (>= valor após a piada - 0,02)", lambda tr, b: _at(tr, "defensive")["resentment"] >= _at(tr, "joke")["resentment"] - 0.02),
        ("sem desculpa, a pendência segue no fim", lambda tr, b: len(tr[-1]["unresolved"]) > 0),
        ("esfria com o tempo mas com piso (dia 3 < pico e >= base+0,1)",
         lambda tr, b: _at(tr, "later")["resentment"] < peak(tr, "resentment") and _at(tr, "later")["resentment"] >= b["resentment"] + 0.1),
    ],
    "S04a_disappear_no_explanation": [
        ("volta sem explicar: ressentimento sobe >= +0,07", lambda tr, b: _at(tr, "return")["resentment"] >= _at(tr, "return", "before")["resentment"] + 0.07),
        ("há pendência depois da desculpa esfarrapada", lambda tr, b: len(_x(tr, "excuse")["unresolved"]) > 0),
        ("confiança cai (após a desculpa esfarrapada < antes da volta)", lambda tr, b: _at(tr, "excuse")["trust"] < _at(tr, "return", "before")["trust"]),
    ],
    "S04b_disappear_explained": [
        ("volta explicando: ressentimento <= base + 0,08", lambda tr, b: _at(tr, "return")["resentment"] <= b["resentment"] + 0.08),
        ("proteção/preocupação sobe >= +0,05", lambda tr, b: max(x["after"]["protectiveness"] for x in tr) >= b["protectiveness"] + 0.05),
        ("sem pendência no fim", lambda tr, b: len(tr[-1]["unresolved"]) == 0),
    ],
    "S05_promise_kept": [
        ("confiança final >= inicial + 0,05", lambda tr, b: _at(tr, "kept2")["trust"] >= b["trust"] + 0.05),
        ("promessa sai da lista ao ser cumprida", lambda tr, b: len(_x(tr, "kept2")["promises"]) == 0),
        ("ressentimento nunca passa de base + 0,05", lambda tr, b: max(x["after"]["resentment"] for x in tr) <= b["resentment"] + 0.05),
    ],
    "S06_promise_broken": [
        ("confiança após o furo <= inicial - 0,10", lambda tr, b: _at(tr, "broken")["trust"] <= b["trust"] - 0.10),
        ("vira pendência", lambda tr, b: len(_x(tr, "broken")["unresolved"]) > 0),
        ("ressentimento após o desdém >= base + 0,2", lambda tr, b: _at(tr, "dismiss")["resentment"] >= b["resentment"] + 0.2),
    ],
    "S07_jealousy": [
        ("ciúme >= 0,30 depois do encontro contado com entusiasmo", lambda tr, b: _at(tr, "jeal3")["jealousy"] >= 0.30),
        ("ciúme final < pico - 0,10 (tranquilizado)", lambda tr, b: tr[-1]["after"]["jealousy"] < peak(tr, "jealousy") - 0.10),
        ("afeto final >= inicial", lambda tr, b: tr[-1]["after"]["affection"] >= b["affection"]),
    ],
    "S08_reconciliation": [
        ("entra no modo frio na briga", lambda tr, b: any(x["mode"] == "cold" for x in tr[:4])),
        ("ressentimento final <= pico - 0,2", lambda tr, b: tr[-1]["after"]["resentment"] <= peak(tr, "resentment") - 0.2),
        ("sai do modo frio no fim", lambda tr, b: tr[-1]["mode"] != "cold"),
        ("<= 3 trocas de modo", lambda tr, b: switches(tr) <= 3),
    ],
    "S09_neutral_control": [
        ("nenhuma dimensão se afasta > 0,12 da base (sem deriva)", lambda tr, b: all(abs(x["after"][d] - b[d]) <= 0.12 for x in tr for d in DIMS)),
        ("nenhuma pendência", lambda tr, b: all(len(x["unresolved"]) == 0 for x in tr)),
        ("<= 1 troca de modo", lambda tr, b: switches(tr) <= 1),
    ],
    "S10_constant_affection": [
        ("afeto final >= inicial + 0,10", lambda tr, b: tr[-1]["after"]["affection"] >= b["affection"] + 0.10),
        ("afeto final <= 0,95 (não satura)", lambda tr, b: tr[-1]["after"]["affection"] <= 0.95),
        ("retornos decrescentes (3 últimos incrementos <= o 1o)",
         lambda tr, b: all((x["after"]["affection"] - x["before"]["affection"]) <= (tr[0]["after"]["affection"] - tr[0]["before"]["affection"]) + 1e-9 for x in tr[-3:])),
    ],
    "S11_vulnerability": [
        ("conforto final >= inicial + 0,05", lambda tr, b: tr[-1]["after"]["comfort"] >= b["comfort"] + 0.05),
        ("confiança final >= inicial + 0,03", lambda tr, b: tr[-1]["after"]["trust"] >= b["trust"] + 0.03),
        ("proteção sobe >= +0,05 com a revelação", lambda tr, b: _at(tr, "vuln1")["protectiveness"] >= b["protectiveness"] + 0.05),
    ],
    "S12_perceived_lie": [
        ("confiança cai >= 0,07 com a contradição", lambda tr, b: _at(tr, "lie2")["trust"] <= _at(tr, "lie", "before")["trust"] - 0.07),
        ("vira pendência", lambda tr, b: len(_x(tr, "dismiss")["unresolved"]) > 0),
        ("ressentimento final >= base + 0,1", lambda tr, b: tr[-1]["after"]["resentment"] >= b["resentment"] + 0.1),
    ],
    "S13_slow_neglect": [
        ("ressentimento final >= base + 0,2", lambda tr, b: tr[-1]["after"]["resentment"] >= b["resentment"] + 0.2),
        ("conforto final <= inicial - 0,10", lambda tr, b: tr[-1]["after"]["comfort"] <= b["comfort"] - 0.10),
        ("nenhum 'k' isolado sobe o ressentimento > 0,15", lambda tr, b: all(x["after"]["resentment"] - x["before"]["resentment"] <= 0.15 for x in tr)),
    ],
    "S14_day200_banter": [
        ("ressentimento <= base + 0,1 sempre", lambda tr, b: max(x["after"]["resentment"] for x in tr) <= b["resentment"] + 0.1),
        ("cumplicidade final >= inicial + 0,05", lambda tr, b: tr[-1]["after"]["playfulness"] >= b["playfulness"] + 0.05),
        ("nenhuma pendência", lambda tr, b: all(len(x["unresolved"]) == 0 for x in tr)),
    ],
    "S15_hot_cold": [
        ("<= 3 trocas de modo (histerese)", lambda tr, b: switches(tr) <= 3),
        ("ressentimento final >= base + 0,1 (as grosserias acumulam)", lambda tr, b: tr[-1]["after"]["resentment"] >= b["resentment"] + 0.1),
    ],
}
GLOBAL = [
    ("sem saturação: todas as dimensões em [0,03; 0,97]", lambda tr, b: all(0.03 <= x["after"][d] <= 0.97 for x in tr for d in DIMS)),
    ("sem oscilação: <= 3 trocas de modo", lambda tr, b: switches(tr) <= 3),
    ("sem salto > 0,30 numa mensagem", lambda tr, b: all(abs(x["after"][d] - x["before"][d]) <= 0.30 for x in tr for d in DIMS)),
]


def check(name, res):
    tr, b = res["traj"], res["base"]
    out = []
    for lab, f in CRIT[name] + GLOBAL:
        try:
            ok = bool(f(tr, b))
        except Exception as e:
            ok = False; lab = lab + f" [erro: {e}]"
        out.append({"crit": lab, "ok": ok, "global": (lab, f) in GLOBAL})
    return out


def main(cfgs):
    path = f"{PROC}/c2_scen_results.json"
    try:
        allres = json.load(open(path))
    except Exception:
        allres = {}
    for cfg in cfgs:
        with ThreadPoolExecutor(4) as ex:
            rs = list(ex.map(lambda n: (n, run(n, cfg)), list(SC)))
        out = {}
        for n, r in rs:
            r["checks"] = check(n, r)
            out[n] = r
        sc = [c["ok"] for r in out.values() for c in r["checks"] if not c["global"]]
        gl = [c["ok"] for r in out.values() for c in r["checks"] if c["global"]]
        split_pass = {}
        for sp, names in (("dev", DEV_SC), ("test", [k for k in SC if k not in DEV_SC])):
            split_pass[sp] = round(float(np.mean([c["ok"] for n in names for c in out[n]["checks"] if not c["global"]])), 3)
            split_pass[sp + "_global"] = round(float(np.mean([c["ok"] for n in names for c in out[n]["checks"] if c["global"]])), 3)
        allres[cfg] = {"scenarios": out, "pass_specific": round(float(np.mean(sc)), 3), "n_specific": len(sc),
                       "split": split_pass,
                       "pass_global": round(float(np.mean(gl)), 3), "n_global": len(gl),
                       "scen_all_pass": sum(all(c["ok"] for c in r["checks"]) for r in out.values())}
        jdump(allres, path)
        print(cfg, allres[cfg]["pass_specific"], allres[cfg]["pass_global"], allres[cfg]["scen_all_pass"], jev.summary(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["FULL"])
