"""a9 — amostragem estratificada dos pontos de decisão (maichat).
Ponto de decisão = turno T do falante S cujo turno anterior (T-1) é do parceiro, na mesma sessão.
O tipo de momento vem do D do turno do PARCEIRO (T-1): é o que o bot sabe na hora de responder.
Conversas de dev (calibração dos limiares e iteração do briefing) são disjuntas das de teste.
Saída: analysis/data/a9_points.jsonl"""
import random
from collections import Counter, defaultdict
from a9_common import load_mai, turn_text, style_fingerprint, save_points, feats, BOT, USER

N_PER_STRATUM, N_DEV, MAX_PER_CONV, HIST = 17, 60, 4, 12


def stratum(prev, t):
    D = prev["D"]
    ph, emo, it = D["phase"]["choice"], D["emotion"]["choice"], D["intent"]["choice"]
    if ph == "winding_down" or it == "closing" or prev["farewell"]:
        return "fechamento"
    if (ph == "opening" or it == "greet" or prev["greeting"]) and t["turn_in_session"] <= 4:
        return "abertura"
    if (D["vulnerable"]["noul"] > 0.5 or D["seeks_support"]["noul"] > 0.5 or emo in ("sadness", "anxiety_insecurity")
            or ph == "deep_personal" or D["seriousness"]["score"] >= 1.6):
        return "serio_vulneravel"
    if D["flirting"]["noul"] > 0.4 or emo == "affection" or it == "compliment_affection":
        return "flerte_afeto"
    if ph == "logistics" or it == "make_plans_logistics":
        return "logistica"
    if emo in ("playful_teasing", "amusement") and D["playful"]["noul"] > 0.6:
        return "brincadeira"
    return "casual"


def main():
    rows = load_mai()
    by_conv = defaultdict(list)
    for r in rows:
        by_conv[r["conv_id"]].append(r)
    cands = []
    for cid, turns in by_conv.items():
        for i in range(1, len(turns)):
            t, prev = turns[i], turns[i - 1]
            if prev["session"] != t["session"] or prev["speaker"] == t["speaker"] or t["media"]:
                continue
            tgt = turn_text(t["texts"])
            if not tgt or not turn_text(prev["texts"]):
                continue
            sess = [x for x in turns[:i] if x["session"] == t["session"]]
            hist = sess[-HIST:]
            while hist and hist[0]["speaker"] == t["speaker"]:  # a conversa com a LLM começa com o usuário
                hist = hist[1:]
            if not hist:
                continue
            own = [turn_text(x["texts"]) for x in sess if x["speaker"] == t["speaker"]]
            oth = [turn_text(x["texts"]) for x in sess if x["speaker"] != t["speaker"]]
            cands.append({
                "id": f"{cid}_{t['turn_idx']}", "conv_id": cid, "turn_idx": t["turn_idx"],
                "turn_in_session": t["turn_in_session"], "stratum": stratum(prev, t),
                "history": [{"who": BOT if x["speaker"] == t["speaker"] else USER, "text": turn_text(x["texts"])} for x in hist],
                "human": tgt, "human_n_msgs": t["n_msgs"], "fp": style_fingerprint(own, oth),
                "prev_D": {k: (v.get("choice") if v["type"] == "choice" else v.get("score", v.get("noul")))
                           for k, v in prev["D"].items()},
            })
    print("candidatos", len(cands), Counter(c["stratum"] for c in cands))
    rnd = random.Random(909)
    convs = sorted(by_conv)
    dev_convs = set(rnd.sample(convs, 6))
    dev = [c for c in cands if c["conv_id"] in dev_convs]
    rnd.shuffle(dev)
    dev = dev[:N_DEV]
    for c in dev:
        c["split"] = "dev"
    test, per_conv = [], Counter()
    pool = [c for c in cands if c["conv_id"] not in dev_convs]
    rnd.shuffle(pool)
    for s in ["abertura", "fechamento", "serio_vulneravel", "flerte_afeto", "logistica", "brincadeira", "casual"]:
        k = 0
        for c in pool:
            if k >= N_PER_STRATUM:
                break
            if c["stratum"] == s and per_conv[c["conv_id"]] < MAX_PER_CONV and c["turn_in_session"] >= 1:
                c["split"] = "test"
                test.append(c); per_conv[c["conv_id"]] += 1; k += 1
    print("teste", len(test), Counter(c["stratum"] for c in test), "convs", len(per_conv))
    print("dev", len(dev), Counter(c["stratum"] for c in dev))
    for c in test + dev:
        c["human_f"] = feats(c["human"])
    save_points(test + dev)


if __name__ == "__main__":
    main()
