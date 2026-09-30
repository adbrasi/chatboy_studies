"""a4 §1 — style fingerprint: between- vs within-speaker variation, split-half stability,
speaker identification by style vector, laugh-form loyalty.
Output: analysis/data/a4_fingerprint.json"""
import json, re, sys, os
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from a4_common import load_msgs, ADATA

FEATS = ["laugh", "emo_any", "elong", "starts_lower", "end_period", "end_none", "end_excl", "q", "abbr",
         "ellipsis", "caps_word", "multi_punct", "logc"]
MIN = {"maichat": 30, "whatsapp_nl": 100, "nps_chatroom": 40, "nus_sms": 60}
rng = np.random.default_rng(0)


def eta2(d, f):
    g = d.groupby("spk")[f]
    grand = d[f].astype(float).mean()
    ssb = (g.size() * (g.mean() - grand) ** 2).sum()
    sst = ((d[f].astype(float) - grand) ** 2).sum()
    return float(ssb / sst) if sst > 0 else None


def main():
    m = load_msgs()
    m = m[~m.media & m.corpus.isin(MIN)]
    m["spk"] = m.corpus + ":" + m.conv_id + ":" + m.speaker.astype(str)
    if True:  # SMS speakers are the same person across 'conv' (conv = user)
        pass
    res = {}
    for c, mn in MIN.items():
        d = m[m.corpus == c].copy()
        cnt = d.spk.value_counts()
        d = d[d.spk.isin(cnt[cnt >= mn].index)]
        d["pos"] = d.groupby("spk").cumcount()
        d["n"] = d.spk.map(d.spk.value_counts())
        d["half"] = np.where(d.pos < d.n / 2, 0, 1)          # temporal halves
        d["odd"] = d.pos % 2                                   # interleaved halves
        out = {"n_speakers": int(d.spk.nunique()), "n_msgs": int(len(d)), "feat": {}}
        spk_means = d.groupby("spk")[FEATS].mean().astype(float)
        for f in FEATS:
            h = d.groupby(["spk", "half"])[f].mean().unstack().astype(float)
            o = d.groupby(["spk", "odd"])[f].mean().unstack().astype(float)
            r_t = h[0].corr(h[1]) if h[0].std() > 0 else None
            r_o = o[0].corr(o[1]) if o[0].std() > 0 else None
            out["feat"][f] = {
                "mean": round(float(d[f].astype(float).mean()), 4),
                "spk_p10": round(float(spk_means[f].quantile(.1)), 4),
                "spk_p50": round(float(spk_means[f].median()), 4),
                "spk_p90": round(float(spk_means[f].quantile(.9)), 4),
                "eta2_between": round(eta2(d, f), 4),
                "r_halves_temporal": None if r_t is None or np.isnan(r_t) else round(float(r_t), 3),
                "r_halves_interleaved": None if r_o is None or np.isnan(r_o) else round(float(r_o), 3),
                "mean_abs_drift": round(float((h[1] - h[0]).abs().mean()), 4),
            }
        # identification: first-half vector -> nearest second-half vector (z-scored)
        h0 = d[d.half == 0].groupby("spk")[FEATS].mean().astype(float)
        h1 = d[d.half == 1].groupby("spk")[FEATS].mean().astype(float)
        mu, sd = pd.concat([h0, h1]).mean(), pd.concat([h0, h1]).std().replace(0, 1)
        A, B = ((h0 - mu) / sd).values, ((h1 - mu) / sd).values
        D = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        top1 = float((D.argmin(1) == np.arange(len(A))).mean())
        ranks = [(D[i] < D[i, i]).sum() for i in range(len(A))]
        top5 = float(np.mean([r < 5 for r in ranks]))
        out["ident"] = {"top1": round(top1, 3), "top5": round(top5, 3), "chance_top1": round(1 / len(A), 4),
                        "median_rank": float(np.median(ranks)), "n": len(A)}
        # partner distinguishability (dyadic corpora): is own 2nd half closer than partner's 2nd half?
        if c in ("maichat", "whatsapp_nl"):
            idx = {s: i for i, s in enumerate(h0.index)}
            wins = []
            for s, i in idx.items():
                cc, cv, sp = s.split(":")
                partners = [j for t, j in idx.items() if t.startswith(f"{cc}:{cv}:") and t != s]
                for j in partners:
                    wins.append(D[i, i] < D[i, j])
            out["ident"]["own_vs_partner_acc"] = round(float(np.mean(wins)), 3) if wins else None
            out["ident"]["own_vs_partner_n"] = len(wins)
        # laugh-form loyalty
        lf = d[d.laugh].copy()
        def form(t):
            t2 = t.lower()
            mm = re.search(r"\b(a?ha(?:ha)+h?a*|ha{2,}|hah+|he(?:he)+h?|hihi+|lo+l+|lmf?ao+|rofl|x+d+)\b", t2)
            if mm:
                w = mm.group(1)
                if w.startswith("lo"): return "lol"
                if w.startswith("lm") or w == "rofl": return "lmao"
                if w.startswith(("he", "hi")): return "hehe"
                if w.startswith("x"): return "xd"
                n = w.count("ha") + w.count("ah") // 2
                return "ha" + ("x2" if len(w) <= 5 else "x3+")
            return "emoji"
        lf["form"] = lf.text.map(form)
        loy = []
        for s, g in lf.groupby("spk"):
            if len(g) >= 5:
                loy.append(g.form.value_counts(normalize=True).iloc[0])
        out["laugh_form"] = {"dist": {k: round(v, 3) for k, v in lf.form.value_counts(normalize=True).items()},
                             "n_spk_5plus_laughs": len(loy),
                             "dominant_form_share_median": round(float(np.median(loy)), 3) if loy else None,
                             "dominant_form_share_mean": round(float(np.mean(loy)), 3) if loy else None}
        res[c] = out
        print(c, out["n_speakers"], out["ident"], out["laugh_form"])
        for f, v in out["feat"].items():
            print(f"  {f:13s} mean={v['mean']:.3f} p10={v['spk_p10']:.3f} p50={v['spk_p50']:.3f} p90={v['spk_p90']:.3f} eta2={v['eta2_between']:.3f} rT={v['r_halves_temporal']} rI={v['r_halves_interleaved']}")
    json.dump(res, open(f"{ADATA}/a4_fingerprint.json", "w"), indent=1)


if __name__ == "__main__":
    main()
