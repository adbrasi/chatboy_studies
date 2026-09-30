"""a9 — "o que escrever": rotula o MOVIMENTO e o TOM de cada resposta (humano e condições) com o mesmo Choice usado no
briefing, para medir (i) se a condição faz o mesmo tipo de movimento que o humano fez e (ii) se o movimento previsto
pelo Jev antes de ver a resposta bate com o do humano. Uma chamada por ponto: as respostas vão no state sob chaves
embaralhadas (r1..rk) e cada pergunta aponta para uma delas (perguntas isoladas).
Saída: analysis/data/a9_moves.jsonl"""
import json, os, random
from a9_common import ask_many_timed, load_points, ADATA, BOT
from a9_brief import Q, state_of
from jev import choice
import jev

CONDS = ["H", "A", "S", "B", "C", "D", "Bnojev"]
MOVE_OPTS = Q["move"]["criteria"]
TONE_OPTS = Q["tone"]["criteria"]


def main():
    pts = [p for p in load_points() if p["split"] == "test" and "gen" in p]
    items, maps = [], []
    for p in pts:
        texts = {"H": p["human"], **{c: p["gen"][c] for c in CONDS[1:] if c in p["gen"]}}
        # respostas idênticas (C==B) compartilham a mesma chave
        uniq = []
        for c, t in texts.items():
            if t not in uniq:
                uniq.append(t)
        rnd = random.Random(p["id"])
        rnd.shuffle(uniq)
        key_of = {t: f"r{k + 1}" for k, t in enumerate(uniq)}
        st = state_of(p)
        st["candidate_replies"] = {key_of[t]: t for t in uniq}
        qs = {}
        for t, k in key_of.items():
            qs[f"{k}_move"] = choice(f"Which move does `candidate_replies.{k}` make, as {BOT}'s reply to `last_message`?", MOVE_OPTS)
            qs[f"{k}_tone"] = choice(f"Which tone does `candidate_replies.{k}` have?", TONE_OPTS)
        items.append((st, qs)); maps.append({c: key_of[t] for c, t in texts.items()})
    res = ask_many_timed(items, workers=4)
    with open(os.path.join(ADATA, "a9_moves.jsonl"), "w", encoding="utf-8") as f:
        for p, m, (a, _) in zip(pts, maps, res):
            if a is None:
                continue
            f.write(json.dumps({"id": p["id"], "move": {c: a[f"{k}_move"]["choice"] for c, k in m.items()},
                                "tone": {c: a[f"{k}_tone"]["choice"] for c, k in m.items()},
                                "move_p_H": a[f"{m['H']}_move"]["probabilities"]}, ensure_ascii=False) + "\n")
    print(jev.summary())


if __name__ == "__main__":
    main()
