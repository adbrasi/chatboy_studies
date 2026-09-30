"""a9 — avaliação cega.
(1) Jev como juiz de pares humano x condição ("qual foi escrita pelo humano real?"), nas DUAS ordens (cancela viés de posição);
(3) Jev: qualidade de cada resposta isolada (Nouls: cabe no clima, soa IA, formal demais, gíria forçada, coerente, entusiasmo demais);
(4) juiz LLM independente (gpt-4o-mini, temperatura 0) nos mesmos pares, 1 ordem balanceada por hash.
Saída: analysis/data/a9_judge.jsonl  (um registro por ponto)"""
import hashlib, json, os, re
from a9_common import (ask_many_timed, chat_many_timed, load_points, merge_json, ADATA, BOT, USER, JUDGE_MODEL)
from a9_brief import state_of
from jev import choice, noul
import jev, llm

OUT = os.path.join(ADATA, "a9_judge.jsonl")
CONDS = ["A", "S", "B", "C", "D", "Bnoban", "Blong", "Bpure", "Bnojev"]
LLM_CONDS = ["A", "S", "B", "C", "D", "Bnoban", "Blong", "Bpure"]  # Bnojev só com o Jev (orçamento de LLM esgotado)

PAIR = {"human": choice(
    f"This is a real text chat between two friends. One of `reply_1` and `reply_2` is the message {BOT} actually sent "
    f"next; the other was written by an AI imitating {BOT}. Which one did the real {BOT} send?",
    {"reply_1": "`reply_1` was sent by the real person", "reply_2": "`reply_2` was sent by the real person"})}
QUAL = {
    "fits_mood": noul(f"Does `reply` fit the tone and mood of the moment in the conversation?"),
    "sounds_ai": noul(f"Does `reply` sound like it was written by an AI assistant rather than a person texting a friend?"),
    "too_formal": noul(f"Is `reply` more formal, polished or wordy than people normally text?"),
    "forced_slang": noul(f"Does `reply` use slang, abbreviations or emojis in a forced or exaggerated way?"),
    "coherent": noul(f"Does `reply` make sense as a direct response to `last_message`?"),
    "over_enth": noul(f"Is `reply` more enthusiastic or emotional than the moment calls for?"),
}


def llm_prompt(p, r1, r2):
    conv = "\n".join(f"{h['who']}: {h['text'].replace(chr(10), ' / ')}" for h in p["history"][-8:])
    return [{"role": "user", "content":
             f"Below is a real text conversation between two friends, {USER} and {BOT}, followed by two candidate next "
             f"messages from {BOT}. One is what {BOT} actually sent; the other was written by an AI. Which one did the real "
             f"{BOT} send? (Line breaks inside a candidate are separate chat bubbles.)\n\nConversation:\n{conv}\n\n"
             f"Candidate 1:\n{r1}\n\nCandidate 2:\n{r2}\n\nAnswer with only the number 1 or 2."}]


def h01(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 2


def main():
    pts = [p for p in load_points() if p["split"] == "test" and "gen" in p]
    # ---------- (1) Jev pares, duas ordens
    items, keys = [], []
    for p in pts:
        base = state_of(p)
        for c in CONDS:
            if c not in p["gen"]:
                continue
            for order in (0, 1):
                r = [p["human"], p["gen"][c]] if order == 0 else [p["gen"][c], p["human"]]
                items.append((dict(base, reply_1=r[0], reply_2=r[1]), PAIR)); keys.append((p["id"], c, order))
    c0 = jev.stats["cost"]
    res = ask_many_timed(items, workers=4)
    pair = {}
    for (pid, c, order), (a, _) in zip(keys, res):
        if a is None:
            continue
        pr = a["human"]["probabilities"]
        # prob. de a CONDIÇÃO ser apontada como humana
        pair.setdefault(pid, {}).setdefault(c, {})[order] = pr["reply_2"] if order == 0 else pr["reply_1"]
    print("pares jev ok", jev.summary(), flush=True)
    # ---------- (3) Jev qualidade de cada resposta
    items, keys = [], []
    for p in pts:
        base = state_of(p)
        texts = {"H": p["human"], **{c: p["gen"][c] for c in CONDS if c in p["gen"]}}
        for c, t in texts.items():
            items.append((dict(base, reply={"from": BOT, "text": t}), QUAL)); keys.append((p["id"], c))
    res = ask_many_timed(items, workers=4)
    qual = {}
    for (pid, c), (a, _) in zip(keys, res):
        if a is not None:
            qual.setdefault(pid, {})[c] = {k: v["noul"] for k, v in a.items()}
    merge_json(os.path.join(ADATA, "a9_costs.json"), {"jev_judge": {"cost": jev.stats["cost"] - c0,
                                                                     "calls": jev.stats["calls"]}})
    print("qualidade jev ok", jev.summary(), flush=True)
    # ---------- (4) juiz LLM
    items, keys = [], []
    for p in pts:
        for c in LLM_CONDS:
            if c not in p["gen"]:
                continue
            o = h01(p["id"] + p["gen"][c])
            r = [p["human"], p["gen"][c]] if o == 0 else [p["gen"][c], p["human"]]
            items.append(dict(messages=llm_prompt(p, *r), model=JUDGE_MODEL, temperature=0.0, max_tokens=4, seed=0))
            keys.append((p["id"], c, o))
    l0, n0 = llm.stats["cost"], llm.stats["calls"]
    res = chat_many_timed(items, workers=4)
    lj = {}
    for (pid, c, o), (txt, _) in zip(keys, res):
        m = re.search(r"[12]", txt or "")
        if not m:
            continue
        picked = int(m.group()) - 1
        lj.setdefault(pid, {})[c] = {"fooled": int(picked != o), "order": o}  # fooled = apontou a condição como humana
    merge_json(os.path.join(ADATA, "a9_costs.json"), {"llm_judge": {"calls": llm.stats["calls"] - n0,
                                                                     "cost": llm.stats["cost"] - l0}})
    with open(OUT, "w", encoding="utf-8") as f:
        for p in pts:
            f.write(json.dumps({"id": p["id"], "pair_jev": pair.get(p["id"], {}), "qual_jev": qual.get(p["id"], {}),
                                "llm_judge": lj.get(p["id"], {})}, ensure_ascii=False) + "\n")
    print("ok", jev.summary(), llm.stats)


if __name__ == "__main__":
    main()
