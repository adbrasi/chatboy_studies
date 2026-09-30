"""a8 step 5: deterministic comparison human vs LLM replies at the same points + log-odds (Monroe et al. 2008)
Output: analysis/data/a8_compare.json, analysis/data/a8_blacklist.csv, a8_logodds.json"""
import json, math, os, re, statistics as st
from collections import Counter, defaultdict
import a8_common as C

CONDS = ["human", "base_gemini", "base_gpt4omini", "base_llama70b", "styled_gemini", "bestof3_jev"]


def load_all():
    X = json.load(open(f"{C.SCR}/contexts.json"))
    R = {"human": {c["id"]: "\n".join(t.strip() for t in c["human"] if t.strip()) for c in X}}
    for cn in CONDS[1:]:
        p = f"{C.SCR}/gen_{cn}.json"
        if os.path.exists(p):
            R[cn] = {k: (v or "").strip() for k, v in json.load(open(p)).items()}
    return X, R


def prev_user(c):
    return "\n".join(c["history"][-1]["texts"]) if c["history"] else ""


def tokens(t):
    t = t.lower().replace("’", "'")
    return re.findall(r"[a-z0-9']+|[\U0001F300-\U0001FAFF☀-➿]|[—–]|[!?]+|\.\.\.|[.,]", t)


def ngrams(t, nmax=3):
    tk = ["<s>"] + tokens(t)
    out = []
    for n in range(1, nmax + 1):
        out += [" ".join(tk[i:i + n]) for i in range(len(tk) - n + 1) if not (n == 1 and tk[i] == "<s>")]
    return out


def logodds(A, B, prior, min_count=6):
    """A, B: Counters of doc-frequency; prior: Counter. Returns list of (term, z, fa, fb)."""
    a0, b0, p0 = sum(A.values()), sum(B.values()), sum(prior.values())
    scale = 1.0
    res = []
    for w in set(A) | set(B):
        if A[w] + B[w] < min_count:
            continue
        aw = prior[w] * scale + 0.01
        a_ = scale * p0
        la = math.log((A[w] + aw) / (a0 + a_ - A[w] - aw))
        lb = math.log((B[w] + aw) / (b0 + a_ - B[w] - aw))
        var = 1 / (A[w] + aw) + 1 / (B[w] + aw)
        res.append((w, (la - lb) / math.sqrt(var), A[w], B[w]))
    return sorted(res, key=lambda x: -x[1])


def summarize(texts, prevs):
    F = [C.style_feats(t, p) for t, p in zip(texts, prevs)]
    n = len(F)
    s = {"n": n}
    for k in ("chars", "words", "sentences", "lines", "echo_ratio"):
        v = [f[k] for f in F]
        s[k + "_median"] = st.median(v); s[k + "_mean"] = round(st.mean(v), 2)
    for k in F[0]:
        if isinstance(F[0][k], bool):
            s["pct:" + k] = round(100 * sum(f[k] for f in F) / n, 1)
    s["pct:sentences>=3"] = round(100 * sum(f["sentences"] >= 3 for f in F) / n, 1)
    s["pct:multi_paragraph"] = round(100 * sum("\n\n" in t for t in texts) / n, 1)
    return s, F


def main():
    X, R = load_all()
    ids = [c["id"] for c in X]
    prevs = {c["id"]: prev_user(c) for c in X}
    out = {"summary": {}, "by_src": {}, "by_bucket": {}}
    feats = {}
    for cn, d in R.items():
        ok = [i for i in ids if d.get(i)]
        s, F = summarize([d[i] for i in ok], [prevs[i] for i in ok])
        out["summary"][cn] = s
        feats[cn] = dict(zip(ok, F))
        for src in ("maichat", "empathetic"):
            sub = [i for i in ok if (i.startswith("mc_") if src == "maichat" else i.startswith("ed_"))]
            out["by_src"].setdefault(src, {})[cn] = summarize([d[i] for i in sub], [prevs[i] for i in sub])[0]
    # per bucket: median chars, ends_q, sentences
    for c in X:
        pass
    buckets = sorted({c["bucket"] for c in X})
    for b in buckets:
        bid = [c["id"] for c in X if c["bucket"] == b]
        out["by_bucket"][b] = {}
        for cn in R:
            fs = [feats[cn][i] for i in bid if i in feats[cn]]
            if fs:
                out["by_bucket"][b][cn] = {"n": len(fs), "chars_med": st.median(f["chars"] for f in fs),
                                           "ends_q_pct": round(100 * sum(f["ends_q"] for f in fs) / len(fs)),
                                           "sent_mean": round(st.mean(f["sentences"] for f in fs), 2),
                                           "excl_pct": round(100 * sum(f["any_excl"] for f in fs) / len(fs)),
                                           "emoji_pct": round(100 * sum(f["emoji"] for f in fs) / len(fs))}
    # paired length ratio LLM/human
    out["paired"] = {}
    for cn in R:
        if cn == "human":
            continue
        rat = [feats[cn][i]["chars"] / max(1, feats["human"][i]["chars"]) for i in ids if i in feats[cn]]
        longer = sum(feats[cn][i]["chars"] > feats["human"][i]["chars"] for i in ids if i in feats[cn])
        out["paired"][cn] = {"median_ratio_chars": round(st.median(rat), 2), "pct_llm_longer": round(100 * longer / len(rat), 1)}
    # log-odds: LLM baseline (3 models) vs human chat (maichat all turns + the 50 empathetic human replies)
    M = C.messages()
    hum_docs = [" \n".join(t["texts"]) for t in C.turns("maichat")] + [R["human"][i] for i in ids if i.startswith("ed_")]
    llm_docs = [R[cn][i] for cn in ("base_gemini", "base_gpt4omini", "base_llama70b") for i in ids if R.get(cn, {}).get(i)]
    A = Counter(g for d in llm_docs for g in set(ngrams(d)))
    B = Counter(g for d in hum_docs for g in set(ngrams(d)))
    prior = A + B
    lo = logodds(A, B, prior)
    out["logodds"] = {"n_llm_docs": len(llm_docs), "n_human_docs": len(hum_docs),
                      "llm_top": [[w, round(z, 2), a, b] for w, z, a, b in lo[:120]],
                      "human_top": [[w, round(z, 2), a, b] for w, z, a, b in lo[::-1][:80]]}
    # blacklist table: LLM-ish regexes, % of responses in each source + human corpora per 1000 msgs
    tics = json.load(open(os.path.join(C.OUT, "a8_human_tics.json")))
    rows = []
    for k in C.LLMISH:
        row = {"expr": k}
        for cn in R:
            row[cn] = out["summary"][cn].get("pct:llm:" + k)
        for corp in ("maichat", "nps_chatroom", "nus_sms", "empathetic"):
            row[corp + "_per1000msg"] = tics[corp].get("llm:" + k, 0.0)
        rows.append(row)
    out["blacklist"] = rows
    C.save("a8_compare.json", out)
    with open(os.path.join(C.OUT, "a8_blacklist.csv"), "w") as f:
        cols = list(rows[0].keys())
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f'"{r[c]}"' if c == "expr" else str(r.get(c, "")) for c in cols) + "\n")
    # print
    keys = ["n", "chars_median", "chars_mean", "words_median", "sentences_mean", "pct:sentences>=3", "lines_mean", "pct:multi_paragraph",
            "pct:any_q", "pct:ends_q", "pct:any_excl", "pct:multi_excl", "pct:emoji", "pct:laugh", "pct:final_punct", "pct:final_period",
            "pct:starts_lower", "pct:all_lower", "pct:le3_words", "pct:elongation", "pct:no_apostrophe", "echo_ratio_mean", "pct:echo_any2"]
    keys += [k for k in out["summary"]["human"] if k.startswith("pct:tic:")]
    conds = list(R)
    for src in ("ALL", "maichat", "empathetic"):
        S = out["summary"] if src == "ALL" else out["by_src"][src]
        print("=====", src)
        print(f"{'':32s}" + "".join(f"{c[-12:]:>13s}" for c in conds))
        for k in keys:
            print(f"{k:32s}" + "".join(f"{S[c].get(k, ''):>13}" for c in conds))
    print("paired", out["paired"])
    print("BLACKLIST (% responses)")
    for r in rows:
        print(f"{r['expr'][:45]:45s}" + "".join(f"{str(r.get(c)):>8s}" for c in conds) + f"  | mc/1k={r['maichat_per1000msg']} emp/1k={r['empathetic_per1000msg']}")
    print("LLM-ish n-grams:", [(w, z, a, b) for w, z, a, b in out["logodds"]["llm_top"][:80]])
    print("HUMAN-ish n-grams:", [(w, z, a, b) for w, z, a, b in out["logodds"]["human_top"][:50]])
    for b, d in out["by_bucket"].items():
        print(b, {c: (v["chars_med"], v["ends_q_pct"], v["sent_mean"]) for c, v in d.items()})


if __name__ == "__main__":
    main()
