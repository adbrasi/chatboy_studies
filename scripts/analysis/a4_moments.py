"""a4 §3 — style x conversational moment (Jev D-pass labels from jev_base.jsonl).

Two framings:
  same  : style of turn T vs the D labels of T itself (partly circular: Jev saw T's text, incl. 'haha'/emoji)
  prev  : style of turn T vs the D labels of the PARTNER's previous turn (the bot's position: it knows the
          moment from the user's last turn and must choose its own style). Not circular.
Comparison is within speaker: per speaker (conv:speaker) with >=3 turns in and out of the moment,
diff = rate_in - rate_out; report pooled rates, mean within-speaker diff, t-test and sign share.
Output: analysis/data/a4_moments.json"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from a4_common import OUT, ADATA, P

STY = ["laugh", "emo_any", "elong", "end_period", "any_period_end", "end_excl", "punct_frac", "starts_lower",
       "abbr", "q", "ellipsis", "multi_punct", "caps_word", "logc", "n_msgs"]


def load_jev():
    rows = []
    for l in open(f"{P}/jev_base.jsonl", encoding="utf-8"):
        d = json.loads(l)
        D = d.get("D") or {}
        if not D:
            continue
        g = lambda k: D[k].get("noul") if k in D else None
        rows.append({"corpus": d["corpus"], "conv_id": d["conv_id"], "turn_idx": d["turn_idx"],
                     "seriousness": D["seriousness"]["score"], "valence": D["valence"]["score"],
                     "arousal": D["arousal"]["score"], "engagement": D["engagement"]["score"],
                     "vulnerable": g("vulnerable"), "tension": g("tension"), "flirting": g("flirting"),
                     "playful": g("playful"), "anxious": g("anxious"), "seeks_support": g("seeks_support"),
                     "phase": D["phase"]["choice"], "emotion": D["emotion"]["choice"],
                     "relationship": D["relationship"]["choice"]})
    return pd.DataFrame(rows)


MOMENTS = {
    "serious(>=2)": lambda j: j.seriousness >= 2,
    "very_serious(>=2.5)": lambda j: j.seriousness >= 2.5,
    "vulnerable": lambda j: j.vulnerable >= .5,
    "tension": lambda j: j.tension >= .5,
    "conflict_phase": lambda j: j.phase == "conflict_or_repair",
    "deep_personal_phase": lambda j: j.phase == "deep_personal",
    "sad": lambda j: j.emotion == "sadness",
    "anger": lambda j: j.emotion == "frustration_anger",
    "flirting": lambda j: j.flirting >= .5,
    "playful": lambda j: j.playful >= .5,
    "negative_valence(<1.5)": lambda j: j.valence < 1.5,
}


def within(df, mask, f, min_n=3):
    diffs = []
    for s, g in df.groupby("spk"):
        m = mask.loc[g.index]
        if m.sum() >= min_n and (~m).sum() >= min_n:
            diffs.append(g.loc[m, f].astype(float).mean() - g.loc[~m, f].astype(float).mean())
    if len(diffs) < 5:
        return None
    diffs = np.array(diffs)
    t = stats.ttest_1samp(diffs, 0)
    return {"n_spk": len(diffs), "mean_diff": round(float(diffs.mean()), 4), "p": round(float(t.pvalue), 4),
            "share_pos": round(float((diffs > 0).mean()), 3)}


def main():
    T = pd.read_pickle(f"{OUT}/a4_turns.pkl")
    T = T[T.corpus.isin(["maichat", "whatsapp_nl"])]
    J = load_jev()
    df = T.merge(J, on=["corpus", "conv_id", "turn_idx"], how="left")
    df = df.sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
    df["spk"] = df.corpus + ":" + df.conv_id + ":" + df.speaker.astype(str)
    # partner's previous turn labels (same session, different speaker, directly preceding)
    prev = df.groupby(["corpus", "conv_id"]).shift(1)
    ok = (prev.speaker != df.speaker) & (prev.session == df.session)
    for k in ["seriousness", "valence", "vulnerable", "tension", "flirting", "playful", "anxious", "phase", "emotion", "seeks_support"]:
        df["prev_" + k] = prev[k].where(ok)
    res = {}
    for c in ["maichat", "whatsapp_nl"]:
        d = df[(df.corpus == c) & ~df.media_only]
        out = {}
        for frame in ["same", "prev"]:
            if frame == "same":
                dd = d[d.seriousness.notna()].copy(); J_ = dd
            else:
                dd = d[d.prev_seriousness.notna()].copy()
                J_ = dd[[x for x in dd.columns if x.startswith("prev_")]].rename(columns=lambda x: x[5:])
            out[frame] = {"n_turns": int(len(dd))}
            for mname, fn in MOMENTS.items():
                mask = fn(J_).fillna(False).astype(bool)
                if mask.sum() < 20:
                    continue
                r = {"n_in": int(mask.sum()), "share": round(float(mask.mean()), 3), "feat": {}}
                for f in STY:
                    a = dd.loc[mask, f].astype(float).mean(); b = dd.loc[~mask, f].astype(float).mean()
                    r["feat"][f] = {"in": round(float(a), 4), "out": round(float(b), 4), "within": within(dd, mask, f)}
                out[frame][mname] = r
        res[c] = out
    json.dump(res, open(f"{ADATA}/a4_moments.json", "w"), indent=1)
    # print compact table
    for c in res:
        for frame in res[c]:
            print(f"==== {c} / {frame}  n={res[c][frame]['n_turns']}")
            for m, r in res[c][frame].items():
                if m == "n_turns": continue
                s = []
                for f in ["laugh", "emo_any", "elong", "any_period_end", "punct_frac", "end_excl", "abbr", "logc", "n_msgs", "ellipsis", "q"]:
                    v = r["feat"][f]; w = v["within"]
                    star = "" if not w else ("**" if w["p"] < .01 else "*" if w["p"] < .05 else "")
                    s.append(f"{f}={v['in']:.3f}/{v['out']:.3f}{star}")
                print(f"  {m:22s} n={r['n_in']:5d} ({r['share']:.2f}) " + " ".join(s))


if __name__ == "__main__":
    main()
