"""a9 — condição C: o Jev escolhe, entre 3 candidatos gerados com o briefing (B, B1, B2), o mais natural/adequado.
Pergunta DIFERENTE da do juiz ("qual é o humano?") para não ensinar para a prova; mesmo assim o juiz Jev pode
favorecer C (mesmo modelo), por isso o juiz LLM independente é a checagem principal de C."""
import os
from a9_common import ask_many_timed, load_points, save_points, merge_json, ADATA, BOT, USER
from a9_brief import state_of
from jev import choice, noul
import jev

GATE = {"best": choice(
    f"Which candidate would be the most natural next message for {BOT} to send, reading like a real friend texting "
    f"and fitting the mood of the conversation?",
    {"reply_1": "`candidates.reply_1`", "reply_2": "`candidates.reply_2`", "reply_3": "`candidates.reply_3`"})}
KEYS = ["B", "B1", "B2"]
# v2 (usada como C): um Noul ABSOLUTO por candidato (o Choice acima mostrou viés de posição: 53/32/15%)
GATE_N = {f"ok_{i}": noul(f"Does `candidates.reply_{i}` read like a real friend texting, fitting the mood and making sense "
                          f"as {BOT}'s reply to `last_message`?") for i in (1, 2, 3)}


def main():
    pts = load_points()
    todo = [p for p in pts if p["split"] == "test" and all(k in p.get("gen", {}) for k in KEYS)]
    items = []
    for p in todo:
        st = state_of(p)
        st["candidates"] = {f"reply_{i + 1}": p["gen"][k] for i, k in enumerate(KEYS)}
        items.append((st, GATE))
    c0 = jev.stats["cost"]
    res = ask_many_timed(items, workers=4)
    res2 = ask_many_timed([(st, GATE_N) for st, _ in items], workers=4)
    lat = {}
    for p, (a, dt), (a2, dt2) in zip(todo, res, res2):
        if a is None or a2 is None:
            continue
        ch = a["best"]["choice"]
        k = KEYS[int(ch.split("_")[1]) - 1]
        p["gen"]["Cchoice"] = p["gen"][k]
        sc = [a2[f"ok_{i}"]["noul"] for i in (1, 2, 3)]
        k2 = KEYS[max(range(3), key=lambda i: (sc[i], -i))]
        p["gen"]["C"] = p["gen"][k2]
        p["gate"] = {"pick_choice": k, "conf": a["best"]["confidence"], "probs": a["best"]["probabilities"],
                     "pick": k2, "nouls": sc}
        lat[p["id"]] = dt2
    save_points(pts)
    merge_json(os.path.join(ADATA, "a9_latency.json"), {"jev_gate": lat})
    merge_json(os.path.join(ADATA, "a9_costs.json"), {"jev_gate": {"calls": len(lat), "cost": jev.stats["cost"] - c0}})
    from collections import Counter
    print(Counter(p.get("gate", {}).get("pick_choice") for p in todo), Counter(p.get("gate", {}).get("pick") for p in todo),
          jev.summary())


if __name__ == "__main__":
    main()
