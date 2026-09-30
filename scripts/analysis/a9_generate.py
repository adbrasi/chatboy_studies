"""a9 — gera as respostas das condições.
  A       : gemini-3.5-flash-lite, persona + histórico (sem nenhuma dica de estilo)
  S       : controle "prompt de estilo estático" (dicas genéricas de texting, sem Jev)
  B       : persona + prompt de estilo estático (= S) + briefing curto do Jev (seed 0); B1/B2 = seeds 1 e 2 (gate C)
            -> B - S mede o valor MARGINAL do Jev sobre um prompt de estilo que qualquer produto já teria.
  Bpure   : ablação, persona + briefing do Jev SEM o prompt estático (o desenho "A + briefing" literal; subamostra)
  Bnoban  : ablação, briefing sem a lista "Never use" (subamostra)
  Blong   : ablação, o Jev despejado em prosa longa, sem ordens concretas (subamostra)
  Bnojev  : controle-chave, mesmo formato de briefing SEM conteúdo do Jev (taxas-base no lugar; subamostra)
  D       : claude-haiku-4.5, persona + histórico (referência "LLM mais cara", sem briefing)
Uso: python3 a9_generate.py <split> <cond1,cond2,...> [limite]"""
import os, sys
from a9_common import (chat_many_timed, load_points, save_points, clean_reply, merge_json, ADATA, PERSONA, BOT, USER,
                       GEN_MODEL, STRONG_MODEL)
from a9_brief import build_brief
import llm  # noqa: E402 (a9_common coloca scripts/ no sys.path)

STATIC = (PERSONA + "\n\nText like a real person on a messaging app: casual and short, lowercase is fine, "
          "no assistant-like phrasing, match the other person's energy and style.")
N_ABL = 36  # tamanho da subamostra das ablações


def msgs(p, system):
    m = [{"role": "system", "content": system}]
    for h in p["history"]:
        m.append({"role": "assistant" if h["who"] == BOT else "user", "content": h["text"]})
    return m


def spec(p, cond):
    if cond == "A":
        return dict(messages=msgs(p, PERSONA), model=GEN_MODEL, seed=0)
    if cond == "S":
        return dict(messages=msgs(p, STATIC), model=GEN_MODEL, seed=0)
    if cond == "D":
        return dict(messages=msgs(p, PERSONA), model=STRONG_MODEL, seed=0)
    variant = {"Bnoban": "noban", "Blong": "long", "Bnojev": "nojev"}.get(cond, "short")
    seed = {"B1": 1, "B2": 2}.get(cond, 0)
    base = PERSONA if cond == "Bpure" else STATIC
    sys_ = base + "\n\nFor your next message:\n" + build_brief(p["jev"], p["fp"], variant)
    return dict(messages=msgs(p, sys_), model=GEN_MODEL, seed=seed)


def main(split, conds, limit=None):
    pts = load_points()
    todo = [p for p in pts if p["split"] == split and "jev" in p]
    if limit:
        todo = todo[:limit]
    for cond in conds:
        sub = todo[:N_ABL] if cond in ("Bnoban", "Blong", "Bpure", "Bnojev") and split == "test" else todo
        c0, n0 = llm.stats["cost"], llm.stats["calls"]
        res = chat_many_timed([dict(spec(p, cond), temperature=0.8, max_tokens=300) for p in sub], workers=4)
        # a Gemini às vezes devolve finish_reason=error com conteúdo vazio (e o cache guarda o vazio): refaz com outra seed
        for attempt in (1, 2):
            bad = [i for i, (t, _) in enumerate(res) if not (t or "").strip()]
            if not bad:
                break
            retry = chat_many_timed([dict(spec(sub[i], cond), temperature=0.8, max_tokens=300,
                                          **{"seed": 100 * attempt + spec(sub[i], cond)["seed"]}) for i in bad], workers=4)
            for i, r in zip(bad, retry):
                res[i] = r
            merge_json(os.path.join(ADATA, "a9_costs.json"), {f"empty_{cond}_{split}_try{attempt}": len(bad)})
        lat = {}
        for p, (txt, dt) in zip(sub, res):
            if not (txt or "").strip():
                # falha persistente do provedor: fallback de produção = resposta sem briefing (A), marcada
                if cond != "A" and p.get("gen", {}).get("A"):
                    p.setdefault("fallback", {})[cond] = True
                    p["gen"][cond] = p["gen"]["A"]
                continue
            p.setdefault("gen", {})[cond] = clean_reply(txt)
            p.setdefault("brief", {})
            if cond in ("B", "Bnoban", "Blong", "Bpure", "Bnojev"):
                p["brief"][cond] = spec(p, cond)["messages"][0]["content"].split("For your next message:\n", 1)[1]
            lat[p["id"]] = dt
        dc, dn = llm.stats["cost"] - c0, llm.stats["calls"] - n0
        save_points(pts)
        merge_json(os.path.join(ADATA, "a9_latency.json"), {f"gen_{cond}": lat})
        if dn:
            merge_json(os.path.join(ADATA, "a9_costs.json"), {f"gen_{cond}_{split}": {"calls": dn, "cost": dc,
                                                                                    "per_call": dc / dn}})
        print(cond, split, "n", len(sub), "new calls", dn, "cost", round(dc, 5), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]) if len(sys.argv) > 3 else None)
