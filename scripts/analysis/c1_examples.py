"""c1 — exemplos reais lado a lado (teste): humano × LLM pura × schemas × cabeçalho × pipeline, por ator.
Uso: python3 c1_examples.py <split> <n> [config do pipeline] -> analysis/data/c1_examples_<split>.json (+ impressão em markdown)"""
import json, random, sys
import c1_common as C
import c1_pipeline as P

ROWS = [("A (LLM pura)", ("schema", "free|0")), ("WhatsApp export, sem nota", ("schema", "wa_full|0")),
        ("Messenger JSON, sem nota", ("schema", "messenger|0")), ("livre + nota", ("schema", "free|1")),
        ("cabeçalho RP (H0)", ("header", "H0")), ("cabeçalho + nota no fim (H1e)", ("header", "H1e"))]


def main(split, n, cfg=None, models=("lite", "luna", "deepseek", "mercury")):
    pts = C.split_pts(split)
    G = C.load_gen()
    rnd = random.Random(5)
    sel = rnd.sample(pts, n)
    out = []
    for p in sel:
        ex = {"pid": p["id"], "context": [f"{h['who']}: {h['text']}" for h in p["history"][-4:]], "human": p["human"], "by_model": {}}
        for m in models:
            d = {}
            for lbl, (e, c) in ROWS:
                r = G.get((e, m, c, p["id"]))
                if r:
                    d[lbl] = r.get("text") or ""
            t, _ = P.final(p, m, "NCT+N", G)
            if t is not None:
                d["NCT+N (rel. 13)"] = t
            if cfg:
                t, _ = P.final(p, m, cfg + "+N", G)
                if t is not None:
                    d[f"pipeline {cfg}+N"] = t
            ex["by_model"][m] = d
        out.append(ex)
    C.jdump(f"c1_examples_{split}.json", out)
    for ex in out:
        print(f"\n### {ex['pid']}\n```\n" + "\n".join(ex["context"]) + f"\n```\n**humano:** {ex['human']!r}")
        for m, d in ex["by_model"].items():
            print(f"- **{m}**: " + " · ".join(f"{k}: {v!r}" for k, v in d.items()))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
