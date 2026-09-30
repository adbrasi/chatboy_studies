"""c2 — análise do experimento 5 (efeito do estado na resposta) com o que existe após o fim dos créditos:
  * métricas de CÓDIGO para as 1.040 respostas dos 4 atores (palavras, termina com pergunta, "!", D do relatório 13,
    nomeia emoção por regex, menciona a pendência por regex, marcadores de calor por regex);
  * Nouls do Jev só para as 200 respostas do flash-lite que chegaram a ser julgadas (RES/TRUST × 4 formatos + controles).
Contrastes pareados pela mesma mensagem (RES − TRUST), IC 95% bootstrap por mensagem. Saída: analysis/data/c2_response_results.json"""
import json, re
import numpy as np
from c2_common import ADATA, PROC, MODELS, jl_load, jdump, cboot
from c2_response import MSGS, CONDS, dscore

G = {(g["actor"], g["kind"], g["fmt"], g["mi"]): g for g in jl_load(f"{PROC}/c2_response_gen.jsonl")}
J = {(d["actor"], d["kind"], d["fmt"], d["mi"]): d["j"] for d in jl_load(f"{PROC}/c2_response_judge.jsonl")}
EMO = re.compile(r"(?i)\b(i'?m|i am|im|i feel|i was|made me|makes me)\s+(so |really |kinda |kind of |a bit |a little |still |just |pretty |very )?"
                 r"(hurt|upset|sad|mad|angry|happy|annoyed|disappointed|excited|glad|thrilled|pissed|frustrated|lonely|jealous)\b")
ISSUE = re.compile(r"(?i)dinner|friday|cancel|bail|flake|last time|just dinner|stood me up|again\?")
WARM = re.compile(r"(?i)haha|lol|lmao|😂|❤|🥰|😊|😘|proud of you|miss you too|congrat|yay|omg")


def words(t):
    return len(re.findall(r"\w+", t))


def code_feats(t):
    t = t or ""
    return {"words": words(t), "q": float(t.strip().endswith("?")), "excl": float("!" in t), "emo": float(bool(EMO.search(t))),
            "issue": float(bool(ISSUE.search(t))), "warm": float(bool(WARM.search(t)))}


out = {"by_cond": {}, "contrast_code": {}, "contrast_jev_lite": {}, "jev_lite_by_cond": {}, "examples": []}
for a in list(MODELS) + ["ALL"]:
    for kind, fmt in CONDS:
        ks = [k for k in G if k[1] == kind and k[2] == fmt and (a == "ALL" or k[0] == a)]
        fs = [code_feats(G[k]["reply"]) for k in ks]
        D, rates, wm = dscore([G[k]["reply"] or "." for k in ks])
        out["by_cond"][f"{a}|{kind}|{fmt}"] = {"n": len(ks), "words_med": float(np.median([f["words"] for f in fs])),
                                                **{m: round(float(np.mean([f[m] for f in fs])), 3) for m in ("q", "excl", "emo", "issue", "warm")},
                                                "D": round(D, 3)}
    for fmt in ("num", "sent", "num_note", "sent_note"):
        row = {}
        for m in ("words", "issue", "warm", "emo", "q"):
            diffs, groups = [], []
            for mi in range(len(MSGS)):
                for aa in (MODELS if a == "ALL" else [a]):
                    kr, kt = (aa, "RES", fmt, mi), (aa, "TRUST", fmt, mi)
                    if kr in G and kt in G:
                        fr, ft = code_feats(G[kr]["reply"]), code_feats(G[kt]["reply"])
                        diffs.append(fr[m] - ft[m] if m != "words" else np.log2((fr[m] + 1) / (ft[m] + 1))); groups.append(mi)
            row[m] = cboot(diffs, groups)
        # respostas idênticas entre RES e TRUST (o estado não mudou nada)
        same = [G[(aa, "RES", fmt, mi)]["reply"].strip().lower() == G[(aa, "TRUST", fmt, mi)]["reply"].strip().lower()
                for mi in range(len(MSGS)) for aa in (MODELS if a == "ALL" else [a])]
        row["identical_frac"] = round(float(np.mean(same)), 3)
        out["contrast_code"][f"{a}|{fmt}"] = row
# Jev (flash-lite apenas)
Q = ["cold", "warm", "subtext", "names_emotion", "guilt_trap", "hostile", "mentions_issue", "consistent", "assistant"]
for kind, fmt in CONDS:
    ks = [k for k in J if k[0] == "lite" and k[1] == kind and k[2] == fmt]
    if ks:
        out["jev_lite_by_cond"][f"{kind}|{fmt}"] = {q: round(float(np.mean([J[k][q] >= 0.5 for k in ks])), 3) for q in Q} | {"n": len(ks)}
for fmt in ("num", "sent", "num_note", "sent_note"):
    row = {}
    for q in ("cold", "warm", "subtext", "names_emotion", "guilt_trap", "mentions_issue", "assistant"):
        diffs, groups = [], []
        for mi in range(len(MSGS)):
            kr, kt = ("lite", "RES", fmt, mi), ("lite", "TRUST", fmt, mi)
            if kr in J and kt in J:
                diffs.append(float(J[kr][q] >= 0.5) - float(J[kt][q] >= 0.5)); groups.append(mi)
        row[q] = cboot(diffs, groups) if diffs else None
    out["contrast_jev_lite"][fmt] = row
for mi in (0, 2, 3, 8, 12, 16):
    for a in MODELS:
        ex = {"actor": a, "msg": MSGS[mi]}
        for kind, fmt in [("NEU", "none"), ("RES", "num"), ("RES", "sent_note"), ("TRUST", "sent_note")]:
            ex[f"{kind}|{fmt}"] = G[(a, kind, fmt, mi)]["reply"]
        out["examples"].append(ex)
out["cost_total"] = round(sum(g.get("cost", 0) for g in G.values()), 4)
out["lat_p50"] = float(np.median([g["lat"] for g in G.values() if g.get("lat")]))
jdump(out, f"{ADATA}/c2_response_results.json")
for k, v in out["contrast_code"].items():
    print(k, {m: (v[m][:3] if isinstance(v[m], list) else v[m]) for m in v})
for k, v in out["jev_lite_by_cond"].items():
    print("lite", k, v)
for k, v in out["contrast_jev_lite"].items():
    print("lite contrast", k, {q: v[q][:3] for q in v if v[q]})
for k, v in out["by_cond"].items():
    if k.startswith("ALL"):
        print(k, v)
