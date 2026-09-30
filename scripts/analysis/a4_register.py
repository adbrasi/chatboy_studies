"""a4 §4 — register inventory: SMS (nus_sms), chat room (nps_chatroom), maichat vs a 'careful' baseline
(empathetic, crowdworkers). Informativeness via log-odds with informative Dirichlet prior (Monroe et al. 2008).
Also 'non-standard token' rate = tokens absent from the empathetic lexicon (freq>=5) — a crude typo/abbr proxy.
Output: analysis/data/a4_register.json"""
import json, os, re, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from a4_common import load_msgs, ADATA

TOK = re.compile(r"[a-z0-9']+|<3|[:;=][-']?[)(dp]")


def toks(t):
    t = re.sub(r"<[^>]+>|\[[^\]]+\]", " ", t.lower())
    return TOK.findall(t)


def logodds(ca, cb, prior, a0=None):
    na, nb = sum(ca.values()), sum(cb.values())
    a0 = a0 or sum(prior.values())
    out = {}
    for w in ca:
        if ca[w] < 15:
            continue
        aw = prior.get(w, 0) + .01
        la = np.log((ca[w] + aw) / (na + a0 - ca[w] - aw))
        lb = np.log((cb.get(w, 0) + aw) / (nb + a0 - cb.get(w, 0) - aw))
        var = 1 / (ca[w] + aw) + 1 / (cb.get(w, 0) + aw)
        out[w] = (la - lb) / np.sqrt(var)
    return out


def main():
    m = load_msgs()
    m = m[~m.media]
    emp = m[m.corpus == "empathetic"]
    ce = Counter(w for t in emp.text for w in toks(t))
    lex = {w for w, n in ce.items() if n >= 5}
    res = {}
    for c in ["nus_sms", "nps_chatroom", "maichat", "empathetic", "whatsapp_nl"]:
        d = m[m.corpus == c]
        cc = Counter(w for t in d.text for w in toks(t))
        prop = {k: round(float(d[k].astype(float).mean()), 4) for k in
                ["starts_lower", "all_lower", "abbr", "no_apos", "lower_i", "elong", "multi_punct", "ellipsis",
                 "laugh", "emo_any", "caps_word", "q", "self_corr", "comma"]}
        prop["end_none"] = round(float((d.end == "none").mean()), 4)
        prop["end_laugh_or_emoji"] = round(float(d.end.isin(["laugh", "emoji"]).mean()), 4)
        prop["end_period"] = round(float((d.end == "period").mean()), 4)
        prop["no_final_punct"] = round(float((~d.end.isin(["period", "excl", "question", "ellipsis"])).mean()), 4)
        prop["median_chars"] = float(d.n_chars.median())
        prop["median_words"] = float(d.n_words.median())
        out = {"n_msgs": int(len(d)), "prop": prop}
        if c in ("nus_sms", "nps_chatroom", "maichat"):
            alpha = [[w for w in toks(t) if w.isalpha() and len(w) > 1] for t in d.text]
            oov_msg = [any(w not in lex for w in ws) for ws in alpha if ws]
            n_tok = sum(len(ws) for ws in alpha); n_oov = sum(w not in lex for ws in alpha for w in ws)
            out["prop"]["msg_with_nonlexicon_token"] = round(float(np.mean(oov_msg)), 4)
            out["prop"]["nonlexicon_token_rate"] = round(n_oov / n_tok, 4)
            lo = logodds(cc, ce, {w: (cc[w] + ce[w]) * .01 for w in set(cc) | set(ce)})
            out["top_marked_vs_empathetic"] = [(w, round(z, 1), cc[w]) for w, z in sorted(lo.items(), key=lambda x: -x[1])[:80]]
            oov = Counter(w for ws in alpha for w in ws if w not in lex)
            out["top_nonlexicon"] = oov.most_common(60)
        res[c] = out
        print("==", c, out["n_msgs"], out["prop"])
        if "top_marked_vs_empathetic" in out:
            print("  marked:", " ".join(f"{w}({n})" for w, z, n in out["top_marked_vs_empathetic"][:80]))
            print("  nonlex:", " ".join(f"{w}({n})" for w, n in out["top_nonlexicon"][:60]))
    json.dump(res, open(f"{ADATA}/a4_register.json", "w"), indent=1)


if __name__ == "__main__":
    main()
