"""b4 — avaliação pelo Jev (coerência, intensidade, responde à pergunta, conserva a ideia do rascunho).
Padrão "itens no state + 1 pergunta por item" (docs do Jev: iterar itens em código): até 8 respostas por chamada,
em ordem embaralhada; a resposta humana entra no mesmo lote (referência cega). Resultados por (modelo, ponto, texto)
ficam em scratchpad/b4/eval.jsonl e keep.jsonl (grandes); a análise agrega.
  coh   Noul "faz sentido como a próxima mensagem de Sam em resposta a last_message?"
  int   Score 0–4 de intensidade emocional/animação (e o do last_message do usuário, para 'resposta <= usuário')
  ans   Noul "responde ao que o usuário perguntou?" (só quando a última mensagem do usuário tem '?')
  keep  Noul "a versão ainda passa a ideia principal do rascunho A?" (só condições derivadas de A)
Validação do lote: modo 'single' avalia A/T/humano sozinhos (1 resposta por chamada) no dev para medir o efeito do lote.
Uso: python3 b4_eval.py <split> <modelos> [eval|keep|single]"""
import json, os, sys
import b4_common  # noqa (põe scripts/ no sys.path)
from jev import noul, score
import jev
from b4_common import load_points, load_gen, SCR, h01, BOT, USER
from b4_multi import INT_LEVELS
from a9_common import ask_many_timed

EV = os.path.join(SCR, "eval.jsonl")
KP = os.path.join(SCR, "keep.jsonl")
SG = os.path.join(SCR, "single.jsonl")
CTX, CH = 6, 8
DERIVED = ["N", "R1", "R2", "R3", "R4", "R5", "J1", "J2", "J3", "D5", "D5x", "RW", "MT", "MT12", "STnl", "STq"]


def load(path):
    out = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            d = json.loads(l)
            out[(d["model"], d["pid"], d["text"])] = d
    return out


def base_state(p):
    h = p["history"][-CTX:]
    return {"conversation": [{"from": x["who"], "text": x["text"]} for x in h[:-1]],
            "last_message": {"from": USER, "text": h[-1]["text"]}}


def eval_items(p, texts):
    st = dict(base_state(p), replies={f"r{i+1}": t for i, t in enumerate(texts)})
    Q = {"user_int": score("How emotionally intense or excited is `last_message`?", INT_LEVELS)}
    asks = "?" in p["history"][-1]["text"]
    for i in range(len(texts)):
        k = f"r{i+1}"
        Q[f"coh_{k}"] = noul(f"Does `replies.{k}` make sense as {BOT}'s next message, replying to `last_message`?")
        Q[f"int_{k}"] = score(f"How emotionally intense or excited is `replies.{k}`?", INT_LEVELS)
        if asks:
            Q[f"ans_{k}"] = noul(f"{USER}'s `last_message` asks {BOT} something. Does `replies.{k}` answer it?")
    return st, Q


def keep_items(p, draft, texts):
    st = dict(base_state(p), draft=draft, versions={f"v{i+1}": t for i, t in enumerate(texts)})
    Q = {f"keep_v{i+1}": noul(f"Does `versions.v{i+1}` still get across the main point of `draft`?") for i in range(len(texts))}
    return st, Q


def main(split, models, mode="eval"):
    pts = {p["id"]: p for p in load_points() if split == "all" or p["split"] == split}
    G = load_gen()
    path = {"eval": EV, "keep": KP, "single": SG}[mode]
    done = load(path)
    for model in models:
        jobs = []  # (pid, [texts])
        for pid, p in pts.items():
            if mode == "eval":
                ts = {r["text"] for (m, c, i), r in G.items() if m == model and i == pid and r.get("text")}
                ts.add(p["human"])
            elif mode == "single":
                ts = {G[(model, c, pid)]["text"] for c in ("A", "T") if (model, c, pid) in G and G[(model, c, pid)].get("text")}
                ts.add(p["human"])
            else:
                if (model, "A", pid) not in G:
                    continue
                draft = G[(model, "A", pid)]["text"]
                ts = {G[(model, c, pid)]["text"] for c in G_conds(G, model, pid) if G[(model, c, pid)].get("text")}
                ts.discard(draft)
            ts = sorted((t for t in ts if (model, pid, t) not in done), key=lambda t: h01(pid, t))
            if not ts:
                continue
            step = 1 if mode == "single" else CH
            for k in range(0, len(ts), step):
                jobs.append((pid, ts[k:k + step]))
        items = []
        for pid, ts in jobs:
            p = pts[pid]
            items.append(keep_items(p, G[(model, "A", pid)]["text"], ts) if mode == "keep" else eval_items(p, ts))
        print(model, mode, "jobs", len(items), flush=True)
        res = ask_many_timed(items, workers=4)
        with open(path, "a", encoding="utf-8") as f:
            for (pid, ts), (a, dt) in zip(jobs, res):
                if a is None:
                    continue
                for i, t in enumerate(ts):
                    k = f"r{i+1}" if mode != "keep" else f"v{i+1}"
                    if mode == "keep":
                        d = {"model": model, "pid": pid, "text": t, "keep": a[f"keep_{k}"]["noul"]}
                    else:
                        d = {"model": model, "pid": pid, "text": t, "coh": a[f"coh_{k}"]["noul"], "int": a[f"int_{k}"]["score"],
                             "user_int": a["user_int"]["score"], "batch": len(ts)}
                        if f"ans_{k}" in a:
                            d["ans"] = a[f"ans_{k}"]["noul"]
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(model, mode, "done; jev calls", jev.stats["calls"], "cost", round(jev.stats["cost"], 4), flush=True)


def G_conds(G, model, pid):
    return [c for c in DERIVED + [x for (m, x, i) in G if m == model and i == pid and x.startswith(("J2@", "J3@", "PIPE"))]
            if (model, c, pid) in G]


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), sys.argv[3] if len(sys.argv) > 3 else "eval")
