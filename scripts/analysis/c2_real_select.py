"""c2 — seleciona janelas reais ricas em eventos de relação (maichat + whatsapp_nl) a partir do jev_base e de regex,
e monta um registro por turno (o turno é a "mensagem do usuário"; o personagem é o outro falante).
Saída: data/processed/c2_real_turns.jsonl ; resumo em analysis/data/c2_real_select.json"""
import json, re, hashlib
from collections import Counter, defaultdict
from datetime import datetime
from c2_common import PROC, ADATA, jdump


def P(x):
    try:
        return datetime.fromisoformat(re.sub(r"[\[\]]", "", x))
    except Exception:
        return None


APO = re.compile(r"(?i)\b(sorry|sry|soz|my bad|apolog\w*|excuus|excuses|sorrie|sorrry|spijt|vergeef)\b")
JEAL = re.compile(r"(?i)\b(jealous|jaloers|ex|date|crush)\b")
PROM = re.compile(r"(?i)\b(promise|beloof\w*|i'll|i will|ik zal)\b")
N_MAI, N_WA, CAP = 14, 16, 90

rows = [json.loads(l) for l in open(f"{PROC}/jev_base.jsonl")]
by = defaultdict(list)
for r in rows:
    by[(r["corpus"], r["conv_id"])].append(r)
wins = []
for k, rs in by.items():
    rs.sort(key=lambda r: r["turn_idx"])
    w = [[rs[0]]]
    for a, b in zip(rs, rs[1:]):
        if b["turn_idx"] != a["turn_idx"] + 1:
            w.append([])
        w[-1].append(b)
    for i, ws in enumerate(w):
        ws = ws[:CAP]
        s = Counter()
        for r in ws:
            D = r["D"]; t = " ".join(r["texts"])
            s["tension"] += D["tension"]["noul"] >= 0.5
            s["conflict"] += D["phase"]["choice"] == "conflict_or_repair"
            s["anger"] += D["emotion"]["choice"] == "frustration_anger"
            s["aff"] += D["intent"]["choice"] == "compliment_affection" or D["emotion"]["choice"] == "affection"
            s["vuln"] += D["vulnerable"]["noul"] >= 0.5
            s["apo"] += bool(APO.search(t)); s["jeal"] += bool(JEAL.search(t)); s["prom"] += bool(PROM.search(t))
        gaps = 0
        for a, b in zip(ws, ws[1:]):
            if P(b["ts_start"]) and P(a["ts_end"]):
                gaps += (P(b["ts_start"]) - P(a["ts_end"])).total_seconds() / 3600 >= 20
        s["gap20h"] = gaps
        sc = (s["tension"] + 2 * s["conflict"] + s["anger"] + 2 * s["aff"] + s["vuln"] + 3 * s["apo"] + 2 * s["jeal"]
              + 2 * s["prom"] + 2 * gaps) / len(ws) ** 0.5
        wins.append({"corpus": k[0], "conv": k[1], "win": i, "turns": ws, "score": sc, "sig": dict(s)})

sel = sorted([w for w in wins if w["corpus"] == "maichat"], key=lambda w: -w["score"])[:N_MAI] + \
      sorted([w for w in wins if w["corpus"] == "whatsapp_nl"], key=lambda w: -w["score"])[:N_WA]


def split_of(conv):
    return "dev" if int(hashlib.md5(("c2" + conv).encode()).hexdigest()[:6], 16) % 5 < 2 else "test"


out, summ = [], []
for w in sel:
    ws = w["turns"]
    t0 = P(ws[0]["ts_start"])
    hist = []
    for j, r in enumerate(ws):
        U = r["speaker"]; C = "B" if U == "A" else "A"
        text = "\n".join(r["texts"])
        ts = P(r["ts_start"]) or (P(ws[j - 1]["ts_end"]) if j else t0)
        th = (ts - t0).total_seconds() / 3600
        if j:
            pe = P(ws[j - 1]["ts_end"]) or ts
            gap = (ts - pe).total_seconds() / 3600
        else:
            gap = None
        out.append({"id": f"{w['conv']}#{r['turn_idx']}", "corpus": w["corpus"], "conv": w["conv"], "win": w["win"],
                    "turn_idx": r["turn_idx"], "pos": j, "U": U, "C": C, "text": text, "t_hours": round(th, 4),
                    "gap_hours": None if gap is None else round(gap, 3),
                    "char_waiting": bool(j and ws[j - 1]["speaker"] == C), "split": split_of(w["conv"]),
                    "ctx": [dict(h) for h in hist[-24:]], "D_rel": r["D"]["relationship"]["choice"]})
        hist.append({"from": U, "text": text, "t_hours": round(th, 3)})
    summ.append({"corpus": w["corpus"], "conv": w["conv"], "n": len(ws), "score": round(w["score"], 2), "sig": w["sig"],
                 "split": split_of(w["conv"])})
with open(f"{PROC}/c2_real_turns.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
c = Counter((r["corpus"], r["split"]) for r in out)
jdump({"windows": summ, "n_turns": len(out), "by_corpus_split": {f"{a}/{b}": n for (a, b), n in c.items()}},
      f"{ADATA}/c2_real_select.json")
print(len(out), c)
