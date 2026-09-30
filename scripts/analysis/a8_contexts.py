"""a8 step 2: pick ~250 real English contexts (200 maichat + 50 empathetic) where a human replied.
Output: scratch a8/contexts.json and analysis/data/a8_contexts_meta.json (small)."""
import json, random
from collections import Counter, defaultdict
import a8_common as C
from a8_moments import moments_of

BUCKETS = ["responder_como_vai", "reagir_noticia_boa", "reagir_desabafo_negativo", "responder_flerte_afeto",
           "responder_provocacao_ou_bronca", "responder_plano", "reagir_algo_engracado", "responder_pergunta", "outro"]


def main():
    random.seed(8)
    J = [r for r in C.jev_base() if r["corpus"] == "maichat"]
    idx = {(r["conv_id"], r["turn_idx"]): r for r in J}
    cands = defaultdict(list)
    for r in J:
        prev = idx.get((r["conv_id"], r["turn_idx"] - 1))
        if not prev or prev["session"] != r["session"] or prev["speaker"] == r["speaker"] or r["turn_in_session"] < 2:
            continue
        ms = [m for m in moments_of(prev, r) if not m.startswith("intent:")]
        b = next((x for x in BUCKETS if x in ms), "outro")
        # history: up to 12 previous turns of the same session
        hist = []
        k = r["turn_idx"] - 1
        while k >= 0 and len(hist) < 12:
            h = idx.get((r["conv_id"], k))
            if not h or h["session"] != r["session"]:
                break
            hist.append(h); k -= 1
        hist = hist[::-1]
        cands[b].append({"id": f"mc_{r['conv_id']}_{r['turn_idx']}", "src": "maichat", "bucket": b, "conv": r["conv_id"],
                         "bot_speaker": r["speaker"],
                         "history": [{"who": "bot" if h["speaker"] == r["speaker"] else "user", "texts": h["texts"]} for h in hist],
                         "human": r["texts"], "prev_D": {k2: prev["D"][k2].get("choice", prev["D"][k2].get("score", prev["D"][k2].get("noul"))) for k2 in ("intent", "emotion", "phase", "valence", "flirting", "playful", "seriousness")},
                         "cur_D": {k2: r["D"][k2].get("choice", r["D"][k2].get("score", r["D"][k2].get("noul"))) for k2 in ("intent", "emotion", "phase")}})
    per_conv, chosen = Counter(), []
    quota = {"responder_como_vai": 20, "reagir_noticia_boa": 25, "reagir_desabafo_negativo": 25, "responder_flerte_afeto": 25,
             "responder_provocacao_ou_bronca": 20, "responder_plano": 20, "reagir_algo_engracado": 25, "responder_pergunta": 25, "outro": 15}
    for b in BUCKETS:
        pool = cands[b][:]
        random.shuffle(pool)
        n = 0
        for c in pool:
            if n >= quota[b]:
                break
            if per_conv[c["conv"]] >= 7:
                continue
            per_conv[c["conv"]] += 1; chosen.append(c); n += 1
    print("maichat chosen", len(chosen), Counter(c["bucket"] for c in chosen), "convs", len(per_conv))
    # empathetic: A's first utterance -> B's reply
    M = C.messages()
    conv = defaultdict(list)
    for m in M["empathetic"]:
        conv[m["conv_id"]].append(m)
    POS = {"excited", "proud", "joyful", "grateful", "impressed", "hopeful", "content", "confident"}
    NEG = {"sad", "lonely", "devastated", "disappointed", "afraid", "anxious", "terrified", "apprehensive", "angry", "annoyed", "embarrassed", "ashamed", "guilty"}
    keys = sorted(conv); random.shuffle(keys)
    pos = neg = 0
    for cid in keys:
        ms = sorted(conv[cid], key=lambda m: m["idx"])
        if len(ms) < 2 or ms[0]["speaker"] == ms[1]["speaker"]:
            continue
        g = ms[0]["gold_emotion"]
        b = "emp_pos" if g in POS else "emp_neg" if g in NEG else None
        if b == "emp_pos" and pos < 25:
            pos += 1
        elif b == "emp_neg" and neg < 25:
            neg += 1
        else:
            continue
        chosen.append({"id": f"ed_{cid}", "src": "empathetic", "bucket": b, "conv": cid, "gold_emotion": g,
                       "history": [{"who": "user", "texts": [ms[0]["text"].replace("_comma_", ",")]}],
                       "human": [ms[1]["text"].replace("_comma_", ",")]})
        if pos >= 25 and neg >= 25:
            break
    json.dump(chosen, open(f"{C.SCR}/contexts.json", "w"), ensure_ascii=False)
    C.save("a8_contexts_meta.json", {"n": len(chosen), "by_bucket": Counter(c["bucket"] for c in chosen),
                                     "ids": [c["id"] for c in chosen]})


if __name__ == "__main__":
    main()
