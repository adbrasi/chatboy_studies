"""b4 — condições derivadas só em CÓDIGO (sem novas chamadas):
  X+N     qualquer condição X passada pelo normalizador de código (C1)
  STnlE   stop "\\n" emulado em código sobre A (corta na 1ª quebra de linha)  -> para o luna, que não aceita stop,
  STqE    stop "?" emulado em código sobre A (corta antes da 1ª "?")           e para checar a equivalência nos outros
  MTfix   MT (max_tokens justo) + conserto do corte: volta até a última fronteira de frase/cláusula completa
  NCA     gerar N=4 (A, A_s1..A_s3) e escolher em código a de menos violações do orçamento
  NCT     gerar N=4 (T, T_s1..T_s3) e escolher em código
  SHORT   gerar N=4 (A..A_s3) e escolher a mais curta
Uso: python3 b4_post.py <split> <modelos>"""
import re, sys
from b4_common import load_points, load_gen, append_gen, budget, feats, normalize, words, lenerr, units, join_units


def violations(t, p, b):
    f = feats(t, p)
    v = lenerr(f["words"], b["words"]) + (f["q"] and not b["q"]) + (f["excl"] and not b["excl"]) + \
        (f["emoji"] and not b["emoji"]) + (f["laugh"] and not b["laugh"]) + 0.5 * len(f["llmish_hits"]) + f["perf"] + \
        f["recip"] + f["template"] + f["artefact"] + 2 * f["empty"]
    return v


def fix_cut(t):
    t = (t or "").strip()
    m = list(re.finditer(r"[.!?…](?=\s|$)|\n|,", t))
    if m and m[-1].end() >= 3:
        cut = t[:m[-1].start() + (1 if t[m[-1].start()] in ".!?…" else 0)].strip(" ,")
        if words(cut):
            return cut
    ws = t.split()
    return " ".join(ws[:-1]) if len(ws) > 2 else t


def main(split, models):
    pts = [p for p in load_points() if split == "all" or p["split"] == split]
    for model in models:
        G = load_gen()
        conds = sorted({c for (m, c, i) in G if m == model and "+N" not in c})
        recs = []
        for p in pts:
            b = budget(p)
            A = G.get((model, "A", p["id"]))
            if not A:
                continue
            base = lambda src, **kw: dict({"model": model, "pid": p["id"], "split": p["split"], "cost": src.get("cost", 0),
                                           "latency": src.get("latency"), "llm_calls": src.get("llm_calls", 1),
                                           "jev_calls": src.get("jev_calls", 0)}, **kw)
            new = {}
            t = A["text"] or ""
            new["STnlE"] = base(A, text=t.split("\n")[0].strip())
            new["STqE"] = base(A, text=(t.split("?")[0].strip() if "?" in t else t))
            if (model, "MT", p["id"]) in G:
                mt = G[(model, "MT", p["id"])]
                new["MTfix"] = base(mt, text=fix_cut(mt["text"]) if mt.get("finish") == "length" else mt["text"])
            for name, srcs in (("NCA", ["A", "A_s1", "A_s2", "A_s3"]), ("NCT", ["T", "T_s1", "T_s2", "T_s3"])):
                rs = [G.get((model, s, p["id"])) for s in srcs]
                rs = [r for r in rs if r and r.get("text")]
                if len(rs) == len(srcs):
                    best = min(rs, key=lambda r: violations(r["text"], p, b))
                    new[name] = base(best, text=best["text"], cost=sum(r["cost"] for r in rs),
                                     latency=max((r["latency"] or 0) for r in rs), llm_calls=len(rs))
                    if name == "NCA":
                        sh = min(rs, key=lambda r: len(r["text"]))
                        new["SHORT"] = base(sh, text=sh["text"], cost=sum(r["cost"] for r in rs),
                                            latency=max((r["latency"] or 0) for r in rs), llm_calls=len(rs))
            for c, r in new.items():
                if (model, c, p["id"]) not in G:
                    recs.append(dict(r, cond=c))
            for c in conds + list(new):
                src = G.get((model, c, p["id"])) or new.get(c)
                if not src or c == "N" or (model, c + "+N", p["id"]) in G:
                    continue
                txt = src.get("text") or A["text"]
                recs.append(dict(base(src, text=normalize(txt, b)), cond=c + "+N", fallback=not bool(src.get("text"))))
        append_gen(recs)
        print(model, split, "derived records", len(recs), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
