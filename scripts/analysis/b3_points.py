"""b3 — todos os pontos de decisão do maichat (mesma definição do a9) + divisão dev/teste por conversa.
Teste = as 35 conversas de teste do a9 (para o Briefing v2 rodar nos mesmos 119 pontos); dev = as outras 7.
Amostras de avaliação: dev_eval (240 pontos das conversas de dev) e test_eval (os 119 pontos do a9 + 281 outros).
Todos os pontos (~2,7 mil) servem de base de casos para o retrieval (sempre excluindo a própria conversa).
Saída: data/processed/b3_points_all.jsonl"""
import json, random
from collections import Counter, defaultdict
from b3_common import load_mai, turn_text, style_fingerprint, feats, jl_load, jl_save, ALL, A9P, BOT, USER

HIST = 12


def dval(v):
    return v.get("choice") if v["type"] == "choice" else v.get("score", v.get("noul"))


def main():
    rows = load_mai()
    by_conv = defaultdict(list)
    for r in rows:
        by_conv[r["conv_id"]].append(r)
    a9 = {p["id"]: p for p in jl_load(A9P)}
    test_convs = {p["conv_id"] for p in a9.values() if p["split"] == "test"}
    pts = []
    for cid, turns in by_conv.items():
        rel = Counter()
        for i in range(1, len(turns)):
            t, prev = turns[i], turns[i - 1]
            rel[prev["D"]["relationship"]["choice"]] += 1
            if prev["session"] != t["session"] or prev["speaker"] == t["speaker"] or t["media"]:
                continue
            tgt = turn_text(t["texts"])
            if not tgt or not turn_text(prev["texts"]):
                continue
            sess = [x for x in turns[:i] if x["session"] == t["session"]]
            hist = sess[-HIST:]
            while hist and hist[0]["speaker"] == t["speaker"]:
                hist = hist[1:]
            if not hist:
                continue
            own_prev = [x for x in sess if x["speaker"] == t["speaker"]]
            own = [turn_text(x["texts"]) for x in own_prev]
            oth = [turn_text(x["texts"]) for x in sess if x["speaker"] != t["speaker"]]
            pid = f"{cid}_{t['turn_idx']}"
            pts.append({
                "id": pid, "conv_id": cid, "turn_idx": t["turn_idx"], "turn_in_session": t["turn_in_session"],
                "split": "test" if cid in test_convs else "dev", "a9": pid in a9 and a9[pid]["split"] == "test",
                "history": [{"who": BOT if x["speaker"] == t["speaker"] else USER, "text": turn_text(x["texts"])}
                            for x in hist],
                "human": tgt, "human_n_msgs": t["n_msgs"], "fp": style_fingerprint(own, oth),
                "prev_D": {k: dval(v) for k, v in prev["D"].items()},
                "own_D": ({k: dval(v) for k, v in own_prev[-1]["D"].items()} if own_prev else None),
                "rel_so_far": rel.most_common(1)[0][0] if rel else None,
                "human_f": feats(tgt),
                "stratum": a9[pid]["stratum"] if pid in a9 else None,
            })
    rnd = random.Random(33)
    dev = [p for p in pts if p["split"] == "dev"]
    rnd.shuffle(dev)
    for p in dev[:240]:
        p["eval"] = True
    per = Counter(p["conv_id"] for p in pts if p["a9"])
    test_pool = [p for p in pts if p["split"] == "test" and not p["a9"]]
    rnd.shuffle(test_pool)
    k = 0
    for p in pts:
        if p["a9"]:
            p["eval"] = True
    for p in test_pool:
        if k >= 281:
            break
        if per[p["conv_id"]] < 12:
            p["eval"] = True; per[p["conv_id"]] += 1; k += 1
    for p in pts:
        p.setdefault("eval", False)
    jl_save(ALL, pts)
    print("pontos", len(pts), Counter((p["split"], p["eval"]) for p in pts), "a9", sum(p["a9"] for p in pts))
    print("convs dev", len({p["conv_id"] for p in pts if p["split"] == "dev"}),
          "test", len({p["conv_id"] for p in pts if p["split"] == "test"}))


if __name__ == "__main__":
    main()
