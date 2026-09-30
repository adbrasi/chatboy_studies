"""c2 — análise só da DECISÃO de postura do Jev (a renderização pelos atores não rodou: créditos esgotados).
Distribuição de posturas por persona × tipo de mensagem, intensidade, concordância Choice × Noul-por-postura, e
sensibilidade ao estado da relação (conhecidos × amigos íntimos × magoado com pendência).
Saída: analysis/data/c2_posture_dec.json"""
import json
from collections import Counter
import numpy as np
from c2_common import ADATA, jdump
from c2_posture import PERSONAS, MSGS, POSTURES, REL_VAR

D = json.load(open(f"{ADATA}/c2_posture_decisions.json"))
TYPES = sorted({t for t, _ in MSGS})
out = {"dist": {}, "intensity": {}, "choice_vs_noul": {}, "rel_change": {}, "retreat_rate": {}, "examples": []}
for pk in PERSONAS:
    for t in TYPES:
        keys = [f"{pk}|{mi}" for mi, (tt, _) in enumerate(MSGS) if tt == t]
        out["dist"][f"{pk}|{t}"] = dict(Counter(D[k]["posture"] for k in keys).most_common())
        out["intensity"][f"{pk}|{t}"] = round(float(np.mean([D[k]["intensity"] for k in keys])), 2)
base = [k for k in D if k.count("|") == 1]
out["choice_vs_noul"]["agree_top"] = round(float(np.mean([D[k]["posture"] == D[k]["noul_top"] for k in base])), 3)
out["choice_vs_noul"]["choice_in_noul_top2"] = round(float(np.mean(
    [D[k]["posture"] in sorted(D[k]["nouls"], key=D[k]["nouls"].get)[-2:] for k in base])), 3)
out["choice_vs_noul"]["mean_conf"] = round(float(np.mean([D[k]["conf"] for k in base])), 3)
# "recuar/pedir desculpa" (o reflexo de assistente) nunca deveria ser escolhido para insulto numa persona dura
out["retreat_rate"] = {t: round(float(np.mean([D[f"{pk}|{mi}"]["posture"] in ("retreat",) for pk in PERSONAS
                                               for mi, (tt, _) in enumerate(MSGS) if tt == t])), 3) for t in TYPES}
out["assistant_like_on_insult"] = round(float(np.mean([D[f"{pk}|{mi}"]["posture"] in ("retreat", "accept")
                                                       for pk in PERSONAS for mi, (tt, _) in enumerate(MSGS) if tt == "insult"])), 3)
for rk in REL_VAR:
    ch = [(pk, mi) for pk in PERSONAS for mi in range(len(MSGS)) if D[f"{pk}|{mi}|{rk}"]["posture"] != D[f"{pk}|{mi}"]["posture"]]
    out["rel_change"][rk] = {
        "frac_changed": round(len(ch) / (len(PERSONAS) * len(MSGS)), 3),
        "intensity_delta": round(float(np.mean([D[f"{pk}|{mi}|{rk}"]["intensity"] - D[f"{pk}|{mi}"]["intensity"]
                                                for pk in PERSONAS for mi in range(len(MSGS))])), 3),
        "by_type": {t: round(float(np.mean([D[f"{pk}|{mi}|{rk}"]["posture"] != D[f"{pk}|{mi}"]["posture"]
                                            for pk in PERSONAS for mi, (tt, _) in enumerate(MSGS) if tt == t])), 3) for t in TYPES},
        "by_persona": {pk: round(float(np.mean([D[f"{pk}|{mi}|{rk}"]["posture"] != D[f"{pk}|{mi}"]["posture"]
                                               for mi in range(len(MSGS))])), 3) for pk in PERSONAS},
        "transitions": dict(Counter(f"{D[f'{pk}|{mi}']['posture']}->{D[f'{pk}|{mi}|{rk}']['posture']}" for pk, mi in ch).most_common(12)),
        "examples": [f"{PERSONAS[pk]['name']} <- \"{MSGS[mi][1]}\": {D[f'{pk}|{mi}']['posture']} -> {D[f'{pk}|{mi}|{rk}']['posture']}"
                     for pk, mi in ch][:60]}
for mi in (0, 6, 9, 15, 22, 26):
    out["examples"].append({"msg": MSGS[mi][1], **{PERSONAS[pk]["name"]: f"{D[f'{pk}|{mi}']['posture']} ({D[f'{pk}|{mi}']['intensity']:.1f})"
                                                   for pk in PERSONAS}})
jdump(out, f"{ADATA}/c2_posture_dec.json")
print(json.dumps({k: out[k] for k in ("choice_vs_noul", "retreat_rate", "assistant_like_on_insult")}, indent=1))
for rk, v in out["rel_change"].items():
    print(rk, v["frac_changed"], v["intensity_delta"], v["by_type"], v["by_persona"], v["transitions"])
    for e in v["examples"][:25]:
        print("   ", e)
for e in out["examples"]:
    print(e)
print(out["intensity"])
