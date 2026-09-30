"""a7: classifica o MOVIMENTO de resposta das LLMs (mesmas classes do humano) e mede distância de estilo ao humano.
Saída: analysis/data/a7_llm_moves.json
"""
import json, os, re, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_common import OUT
from a7_moves import RESP, INT
from a7_llm_gen import lex, length_bucket
from jev import ask_many, choice, noul, score, summary

items = [json.loads(l) for l in open(os.path.join(OUT, "a7_llm_items.jsonl"), encoding="utf-8")]
conds = ["human", "gem_base", "gpt_base", "gem_brief"]
Q, st = [], []
for c in items:
    for cd in ["gem_base", "gpt_base", "gem_brief"]:
        s = {"previous_turns": c["last_turns"][:-1], "target_turn": {"speaker": "Alex", "text": c["last_turns"][-1]},
             "reply_turn": {"speaker": "Sam", "text": (c[cd] or "")[:500]}}
        q = {"resp": choice("How does Sam respond, in reply_turn, to Alex's target_turn?", RESP),
             "r_int": score("How romantic/affectionate/flirtatious is Sam's reply_turn?", INT),
             "t_int": score("How romantic/affectionate/flirtatious is Alex's target_turn?", INT)}
        Q.append((s, q)); st.append((c, cd))
res = ask_many(Q, workers=4)
for (c, cd), r in zip(st, res):
    if r:
        c.setdefault("cls", {})[cd] = {"resp": r["resp"]["choice"], "r_int": r["r_int"]["score"], "t_int": r["t_int"]["score"]}
out = {"n": len(items)}
dist = {"human": Counter(c["resp_human"] for c in items)}
for cd in ["gem_base", "gpt_base", "gem_brief"]:
    dist[cd] = Counter(c["cls"][cd]["resp"] for c in items if cd in c.get("cls", {}))
out["resp_dist"] = {k: dict(v.most_common()) for k, v in dist.items()}
out["agree_with_human_move"] = {cd: float(np.mean([c["cls"][cd]["resp"] == c["resp_human"] for c in items])) for cd in ["gem_base", "gpt_base", "gem_brief"]}
# intensidade relativa ao turno do parceiro (t_int da mesma chamada p/ as LLMs; a7_moves p/ humano)
out["intensity"] = {"human": {"r_minus_t": float(np.mean([c["r_int_human"] - c["t_int"] for c in items])),
                              "above_t_plus1": float(np.mean([c["r_int_human"] > c["t_int"] + 1 for c in items]))}}
for cd in ["gem_base", "gpt_base", "gem_brief"]:
    d = [c["cls"][cd]["r_int"] - c["cls"][cd]["t_int"] for c in items]
    out["intensity"][cd] = {"r_minus_t": float(np.mean(d)), "above_t_plus1": float(np.mean([x > 1 for x in d])),
                            "above_t": float(np.mean([x > .5 for x in d]))}
# distância de estilo ao humano do mesmo contexto
sm = {}
for cd in ["gem_base", "gpt_base", "gem_brief"]:
    rows = []
    for c in items:
        h, l = lex(c["human"]), lex(c[cd] or "")
        rows.append({"len_bucket_eq": length_bucket(h["chars"]) == length_bucket(l["chars"]),
                     "len_ratio": l["chars"] / max(1, h["chars"]), "q_eq": h["q"] == l["q"], "emoji_eq": (h["emoji"] > 0) == (l["emoji"] > 0),
                     "two_part": "\n\n" in (c[cd] or ""), "html_leak": "</p>" in (c[cd] or ""),
                     "react_plus_q": ("?" in (c[cd] or "")) and len(c[cd] or "") > 40})
    R = pd.DataFrame(rows)
    sm[cd] = {"len_bucket_eq": float(R.len_bucket_eq.mean()), "len_ratio_med": float(R.len_ratio.median()), "q_eq": float(R.q_eq.mean()),
              "emoji_eq": float(R.emoji_eq.mean()), "two_part": float(R.two_part.mean()), "html_leak": int(R.html_leak.sum()),
              "react_plus_q": float(R.react_plus_q.mean())}
sm["human"] = {"two_part": float(np.mean(["\n\n" in c["human"] for c in items])),
               "react_plus_q": float(np.mean([("?" in c["human"]) and len(c["human"]) > 40 for c in items]))}
out["style_match"] = sm
# repetição de exemplos do briefing (cópia literal)
cnt = Counter((c["gem_brief"] or "").strip().lower() for c in items)
out["brief_repeated_replies"] = [(k, v) for k, v in cnt.most_common(10) if v > 1]
# frases típicas de LLM
PH = ["aww", "stop", "blush", "so sweet", "you're the best", "u are the best", "can't wait", "cant wait", "i'd love", "love that", "😭", "🥹", "🥰", "❤️", "💖",
      "haha", "lol", "lmao", "tho", "fr", "literally", "right?", "wait", "omg", "glad", "fun", "!"]
out["phrases"] = {cd: {p: int(sum(p in (c[cd] or "").lower() for c in items)) for p in PH} for cd in conds}
json.dump(out, open(os.path.join(OUT, "a7_llm_moves.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)); print(summary())
