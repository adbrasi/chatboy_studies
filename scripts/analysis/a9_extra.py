"""a9 — métricas extras de "o que escrever": abertura performática (omg/wait/ooh/lol/honestly...) x marcador discursivo
simples (ok/so/oh/yeah/same/i...), perguntas recíprocas genéricas, artefatos de formatação e o subconjunto de 36
(B x Bnojev: acompanhamento de tamanho). Saída: analysis/data/a9_extra.json"""
import json, os, re
import numpy as np
from scipy.stats import spearmanr
from a9_common import load_points, load_mai, turn_text, feats, ADATA

PERF = r"^(omg|wait|ooh+|oo+h|lol|haha\w*|lmf?ao+|hey!|aww+|ugh|honestly|stop|bro|literally|oh no|yay+|yesss+|nooo+)\b"
PLAIN = r"^(ok|okay|so|oh|yeah|yes|yep|same|i|im|i'm|well|no|nah|idk|and|but|true|fair|what|why|how)\b"
REC = (r"(what about you|wbu|how about you|and you\?|u\?|you\?$|how's your day|how was your day|what's up|whats up|"
       r"how are you|what are you up to|what are u up to|hbu)")
ART = r"</?p\b|\*[a-z][^*]{2,}\*|\bAlex:|\bSam:"


def main():
    allp = [p for p in load_points() if p["split"] == "test"]
    allH = [turn_text(r["texts"]) for r in load_mai()]
    rate = lambda ts, rx: round(float(np.mean([bool(re.search(rx, t.strip().lower())) for t in ts])), 3)
    out = {"H_all_maichat": {"perf_open": rate(allH, PERF), "plain_open": rate(allH, PLAIN), "recip_q": rate(allH, REC), "n": len(allH)},
           "H": {"perf_open": rate([p["human"] for p in allp], PERF), "plain_open": rate([p["human"] for p in allp], PLAIN),
                 "recip_q": rate([p["human"] for p in allp], REC), "n": len(allp)}}
    for c in ["A", "S", "B", "C", "D", "Bnojev", "Blong", "Bnoban", "Bpure"]:
        T = [p["gen"][c] for p in allp if c in p.get("gen", {})]
        out[c] = {"perf_open": rate(T, PERF), "plain_open": rate(T, PLAIN), "recip_q": rate(T, REC),
                  "artifacts": int(sum(bool(re.search(ART, t)) for t in T)), "n": len(T)}
    sub = [p for p in allp if "Bnojev" in p.get("gen", {})]
    hw = [feats(p["human"])["n_words"] for p in sub]
    out["len_spearman_sub36"] = {c: round(float(spearmanr([feats(p["gen"][c])["n_words"] for p in sub], hw).correlation), 3)
                                 for c in ["A", "S", "B", "C", "D", "Bnojev", "Bpure", "Bnoban", "Blong"]}
    json.dump(out, open(os.path.join(ADATA, "a9_extra.json"), "w"), indent=1)
    print(json.dumps(out, indent=0))


if __name__ == "__main__":
    main()
