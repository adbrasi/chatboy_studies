"""a4 §3b — does a final period sound cold?
A) real data (maichat + whatsapp_nl, Jev D labels): turns whose last bubble ends in '.' vs not, within speaker:
   tension / valence of the turn itself, and of the partner's NEXT turn.
B) Jev minimal pairs: 60 short real maichat replies (2-10 words, no final punctuation, no laugh/emoji) in context,
   5 variants: bare | '.' | '!' | ' haha' | ' :)'. Nouls: cold, warm, bot. Paired differences vs bare.
Output: analysis/data/a4_period.json"""
import json, os, random, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from jev import ask_many, noul, summary
from a4_common import OUT, ADATA
from a4_moments import load_jev


def within(d, mask, f):
    diffs = []
    for s, g in d.groupby("spk"):
        m = mask.loc[g.index]
        if m.sum() >= 2 and (~m).sum() >= 5:
            diffs.append(g.loc[m, f].mean() - g.loc[~m, f].mean())
    diffs = np.array(diffs)
    if len(diffs) < 5:
        return None
    return {"n_spk": len(diffs), "mean_diff": round(float(diffs.mean()), 3), "p": round(float(stats.ttest_1samp(diffs, 0).pvalue), 4)}


def part_a():
    T = pd.read_pickle(f"{OUT}/a4_turns.pkl")
    T = T[T.corpus.isin(["maichat", "whatsapp_nl"])]
    df = T.merge(load_jev(), on=["corpus", "conv_id", "turn_idx"], how="inner").sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
    df["spk"] = df.corpus + ":" + df.conv_id + ":" + df.speaker.astype(str)
    nxt = df.groupby(["corpus", "conv_id"]).shift(-1)
    ok = (nxt.speaker != df.speaker) & (nxt.session == df.session) & (nxt.turn_idx == df.turn_idx + 1)
    df["next_tension"] = nxt.tension.where(ok); df["next_valence"] = nxt.valence.where(ok)
    out = {}
    for c in ["maichat", "whatsapp_nl"]:
        d = df[(df.corpus == c) & ~df.media_only]
        # short turns only (1-6 words), where the period is the salient cue
        for scope, dd in [("all", d), ("short<=6w", d[d.n_words <= 6])]:
            m = dd.end_period
            r = {"n_period": int(m.sum()), "n": int(len(dd))}
            for f in ["tension", "valence", "next_tension", "next_valence"]:
                x = dd[[f]].assign(m=m).dropna()
                r[f] = {"period": round(float(x.loc[x.m, f].mean()), 3), "no_period": round(float(x.loc[~x.m, f].mean()), 3),
                        "within": within(dd.loc[x.index], x.m, f)}
            out[f"{c}/{scope}"] = r
    return out


def part_b(n=60):
    T = pd.read_pickle(f"{OUT}/a4_turns.pkl")
    T = T[T.corpus == "maichat"].sort_values(["conv_id", "turn_idx"]).reset_index(drop=True)
    cand = []
    for i in range(1, len(T)):
        t, p = T.iloc[i], T.iloc[i - 1]
        if p.conv_id != t.conv_id or p.speaker == t.speaker or t.n_msgs != 1:
            continue
        x = t.texts[0].strip()
        w = len(x.split())
        if 2 <= w <= 10 and x[-1].isalnum() and not t.laugh and not t.emo_any:
            cand.append((" / ".join(p.texts)[:300], x))
    random.Random(7).shuffle(cand)
    cand = cand[:n]
    V = {"bare": lambda s: s, "period": lambda s: s + ".", "excl": lambda s: s + "!", "haha": lambda s: s + " haha",
         "smiley": lambda s: s + " :)"}
    Q = {"cold": noul("Y's reply sounds cold, curt or annoyed."), "warm": noul("Y's reply sounds warm and friendly."),
         "bot": noul("Y's reply was written by an AI chatbot, not by a person texting a friend.")}
    jobs, keys = [], []
    for k, (prev, x) in enumerate(cand):
        for v, fn in V.items():
            st = {"conversation": [{"speaker": "X", "text": prev}, {"speaker": "Y", "text": fn(x)}]}
            jobs.append((json.dumps(st, ensure_ascii=False), Q)); keys.append((k, v))
    ans = ask_many(jobs, workers=4)
    R = pd.DataFrame([{"case": k, "v": v, **{q: a[q]["noul"] for q in Q}} for (k, v), a in zip(keys, ans) if a])
    out = {"n_cases": len(cand), "examples": [c[1] for c in cand[:8]], "variants": {}}
    bare = R[R.v == "bare"].set_index("case")
    for v, g in R.groupby("v"):
        g = g.set_index("case")
        r = {}
        for q in Q:
            r[q] = {"mean": round(float(g[q].mean()), 3)}
            if v != "bare":
                d = (g[q] - bare[q]).dropna()
                r[q]["paired_diff"] = round(float(d.mean()), 3)
                r[q]["p_wilcoxon"] = round(float(stats.wilcoxon(d).pvalue), 4) if (d != 0).any() else None
                r[q]["share_up"] = round(float((d > 0).mean()), 3)
        out["variants"][v] = r
    return out


def main():
    res = {"real_data": part_a()}
    print(json.dumps(res["real_data"], indent=1))
    res["jev_minimal_pairs"] = part_b()
    res["jev"] = summary()
    print(json.dumps(res["jev_minimal_pairs"], indent=1))
    json.dump(res, open(f"{ADATA}/a4_period.json", "w"), indent=1)


if __name__ == "__main__":
    main()
