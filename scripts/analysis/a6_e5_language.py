"""E5: language. maichat EN vs hand-translated PT-BR (EN questions, and PT questions); whatsapp NL vs hand-translated EN."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import a6_common as C
from a6_common import noul, choice, score
import a6_translations as T

SCR = C.SCRATCH
QD = C.AB.q_descriptive()
KEYS = ["emotion", "valence", "arousal", "anxious", "playful", "flirting", "vulnerable", "seeks_support", "tension", "seriousness"]
QEN = {k: QD[k] for k in KEYS}
qp = T.Q_PT; who = qp["who"]; Who = who[0].upper() + who[1:]
QPT = {
    "emotion": choice(qp["emotion"].format(who=who), qp["EMO_DESC"]),
    "valence": score(qp["valence"][0].format(who=who), qp["valence"][1]),
    "arousal": score(qp["arousal"][0].format(who=who), qp["arousal"][1]),
    "seriousness": score(qp["seriousness"][0], qp["seriousness"][1]),
    "tension": noul(qp["tension"]),
}
for k in ("anxious", "playful", "flirting", "vulnerable", "seeks_support"):
    QPT[k] = noul(qp[k].format(Who=Who))


def win_states(by, corpus, windows, trans):
    """for each window (cid, i): targets i-4..i with 2 previous turns; returns list of (orig_state, trans_state, meta)"""
    out = []
    for cid, i in windows:
        conv = by[(corpus, cid)]
        for j in range(i - 4, i + 1):
            def st(tr):
                def f(t):
                    t2 = dict(t)
                    if tr: t2["texts"] = trans[(cid, t["turn_idx"])]
                    return C.AB.fmt_turn(t2)
                return {"previous_turns": [f(x) for x in conv[j - 2:j]], "current_turn": f(conv[j])}
            out.append((st(False), st(True), {"cid": cid, "turn_idx": conv[j]["turn_idx"], "orig": " / ".join(conv[j]["texts"])[:90],
                                              "trans": " / ".join(trans[(cid, conv[j]["turn_idx"])])[:90]}))
    return out


by = C.load_turns()
PTW = json.load(open(os.path.join(SCR, "a6_pt_windows.json")))
NLW = json.load(open(os.path.join(SCR, "a6_nl_windows.json")))
pt = win_states(by, "maichat", PTW, T.PT)
nl = win_states(by, "whatsapp_nl", NLW, T.EN_FROM_NL)
items = []
for o, tr, m in pt:
    items += [dict(state=o, questions=QEN, tag="e5_en"), dict(state=tr, questions=QEN, tag="e5_pt"),
              dict(state=tr, questions=QPT, tag="e5_ptq"), dict(state={"run_id": "r7"} | o, questions=QEN, tag="e5_en_rep")]
for o, tr, m in nl:
    items += [dict(state=o, questions=QEN, tag="e5_nl"), dict(state=tr, questions=QEN, tag="e5_nl2en"),
              dict(state={"run_id": "r7"} | o, questions=QEN, tag="e5_nl_rep")]
res = C.run_many(items, 4)
print("new calls", C.NEW_CALLS[0], len(pt), len(nl))

def d(a, b):
    if a["type"] == "noul": return abs(a["noul"] - b["noul"])
    if a["type"] == "score": return abs(a["score"] - b["score"])
    return float(a["choice"] != b["choice"])

def compare(pairs, name):
    """pairs: list of (ansA, ansB)"""
    row = {}
    for k in KEYS:
        xs = [(a[k], b[k]) for a, b in pairs]
        r = {"mean_abs_or_flip": float(np.mean([d(a, b) for a, b in xs]))}
        if xs[0][0]["type"] != "choice":
            va = np.array([C.val(a) for a, _ in xs]); vb = np.array([C.val(b) for _, b in xs])
            r["corr"] = float(np.corrcoef(va, vb)[0, 1]) if va.std() > 0 and vb.std() > 0 else None
            r["mean_A"], r["mean_B"] = float(va.mean()), float(vb.mean())
            if xs[0][0]["type"] == "noul":
                r["binary_agree"] = float(np.mean((va > .5) == (vb > .5)))
        else:
            r["conf_A"] = float(np.mean([a["confidence"] for a, _ in xs])); r["conf_B"] = float(np.mean([b["confidence"] for _, b in xs]))
        row[k] = r
    return row

out = {"n_pt": len(pt), "n_nl": len(nl)}
A = res[:4 * len(pt)]; B = res[4 * len(pt):]
en = [A[4 * i]["answers"] for i in range(len(pt))]; ptA = [A[4 * i + 1]["answers"] for i in range(len(pt))]
ptq = [A[4 * i + 2]["answers"] for i in range(len(pt))]; enr = [A[4 * i + 3]["answers"] for i in range(len(pt))]
out["EN_vs_ENrep(noise)"] = compare(list(zip(en, enr)), "")
out["EN_vs_PT(state PT, questions EN)"] = compare(list(zip(en, ptA)), "")
out["EN_vs_PT(state PT, questions PT)"] = compare(list(zip(en, ptq)), "")
out["PTstate_ENq_vs_PTq"] = compare(list(zip(ptA, ptq)), "")
nlA = [B[3 * i]["answers"] for i in range(len(nl))]; nl2en = [B[3 * i + 1]["answers"] for i in range(len(nl))]; nlr = [B[3 * i + 2]["answers"] for i in range(len(nl))]
out["NL_vs_NLrep(noise)"] = compare(list(zip(nlA, nlr)), "")
out["NL_vs_EN(translated)"] = compare(list(zip(nl2en, nlA)), "")
# worst PT disagreements (examples)
ex = []
for (o, tr, m), a, b in zip(pt, en, ptA):
    dd = sum(d(a[k], b[k]) for k in ("anxious", "playful")) + d(a["seriousness"], b["seriousness"]) / 3
    ex.append((dd, m["orig"], m["trans"], {k: (C.val(a[k]), C.val(b[k])) for k in ("emotion", "anxious", "playful", "seriousness")}))
out["examples_pt_biggest_diff"] = [e[1:] for e in sorted(ex, key=lambda e: -e[0])[:6]]
ex = []
for (o, tr, m), a, b in zip(nl, nl2en, nlA):
    dd = sum(d(a[k], b[k]) for k in ("anxious", "playful")) + d(a["seriousness"], b["seriousness"]) / 3
    ex.append((dd, m["orig"], m["trans"], {k: (C.val(a[k]), C.val(b[k])) for k in ("emotion", "anxious", "playful", "seriousness")}))
out["examples_nl_biggest_diff(EN,NL)"] = [e[1:] for e in sorted(ex, key=lambda e: -e[0])[:6]]
out["tokens_mean"] = {t: float(np.mean([r["usage"]["input_tokens"] for r in res if r and r["tag"] == t])) for t in ("e5_en", "e5_pt", "e5_ptq", "e5_nl", "e5_nl2en")}
print(C.dump("e5_language.json", out))
for k, v in out.items():
    if isinstance(v, dict) and "anxious" in v:
        print("==", k)
        for q, r in v.items():
            print(f"  {q:14s}", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()})
    else:
        print(k, json.dumps(v, ensure_ascii=False)[:2500])
