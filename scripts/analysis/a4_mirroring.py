"""a4 §2 — mirroring / accommodation between partners (turn level, dyadic chats).

For consecutive turns A(t) -> B(t+1) (different speakers, same session):
  lift_raw  = P(B_f | A_f) / P(B_f)
  lift_spk  = observed B_f among A_f pairs / expected from B's own leave-one-out rate in that conversation
              (removes "B is just a laugher" + conversation-level style)
  lag profile: P(B_f at B's k-th next turn | A_f at t) / P(B_f)   (local priming should decay with k)
  self-persistence: P(A_f at A's next turn | A_f at t) / P(A_f)
Length: corr of log(chars) A -> B, raw and demeaned per speaker-in-conversation.
Convergence: |rate_A - rate_B| first third vs last third (Wilcoxon over dyads); dyad-level style
matching vs random re-pairing.
Output: analysis/data/a4_mirroring.json"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from a4_common import OUT, ADATA

FEATS = ["laugh", "emo_any", "elong", "abbr", "q", "end_period", "any_period_end", "end_excl", "starts_lower",
         "caps_word", "ellipsis", "multi_punct", "all_lower"]
rng = np.random.default_rng(1)


def dyadic(T):
    ns = T.groupby(["corpus", "conv_id"]).speaker.nunique()
    keep = ns[ns == 2].index
    return T.set_index(["corpus", "conv_id"]).loc[keep].reset_index()


def pairs_for(d):
    """rows: A turn t, B turn t+1 (+ B's next turns for lag profile)."""
    out = []
    for (c, cv), g in d.groupby(["corpus", "conv_id"], sort=False):
        g = g.sort_values("turn_idx").reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if a.speaker == b.speaker or a.session != b.session or a.media_only or b.media_only:
                continue
            out.append((c, cv, i, i + 1))
    return out


def main():
    T = pd.read_pickle(f"{OUT}/a4_turns.pkl")
    T = dyadic(T[T.corpus.isin(["maichat", "whatsapp_nl", "empathetic"])])
    res = {}
    for c in ["maichat", "whatsapp_nl", "empathetic"]:
        d = T[T.corpus == c].sort_values(["conv_id", "turn_idx"]).reset_index(drop=True)
        d["spk"] = d.conv_id + ":" + d.speaker.astype(str)
        d["pos"] = d.groupby("conv_id").cumcount()
        # arrays for speed
        conv = d.conv_id.values; spk = d.spk.values; sess = d.session.values; mo = d.media_only.values
        ok = (conv[:-1] == conv[1:]) & (spk[:-1] != spk[1:]) & (sess[:-1] == sess[1:]) & ~mo[:-1] & ~mo[1:]
        ia = np.where(ok)[0]; ib = ia + 1
        out = {"n_pairs": int(len(ia)), "n_dyads": int(d.conv_id.nunique()), "feat": {}}
        # next turns of B after ib (B's k-th turn from t+1), for lag profile
        spk_turns = {s: np.where(spk == s)[0] for s in np.unique(spk)}
        pos_in_spk = np.zeros(len(d), int)
        for s, ix in spk_turns.items():
            pos_in_spk[ix] = np.arange(len(ix))
        for f in FEATS:
            x = d[f].values.astype(float)
            # leave-one-out speaker rate
            tot = d.groupby("spk")[f].transform("sum").values.astype(float)
            cnt = d.groupby("spk")[f].transform("size").values.astype(float)
            loo = (tot - x) / np.maximum(cnt - 1, 1)
            A, B = x[ia], x[ib]
            pB = B.mean()
            if A.sum() < 5 or pB == 0:
                continue
            p_ba = B[A == 1].mean(); p_bna = B[A == 0].mean()
            lift_raw = p_ba / pB
            lift_spk = B[A == 1].sum() / loo[ib][A == 1].sum()
            # bootstrap over dyads for lift_spk
            convs = np.unique(conv[ia]); bs = []
            byc = {cv: np.where(conv[ia] == cv)[0] for cv in convs}
            for _ in range(300):
                sel = np.concatenate([byc[cv] for cv in rng.choice(convs, len(convs))])
                a_ = A[sel] == 1
                den = loo[ib][sel][a_].sum()
                if den > 0:
                    bs.append(B[sel][a_].sum() / den)
            # lag profile: B's k-th turn after ib
            lag = {}
            for k in (0, 2, 5, 10):
                obs, exp = [], []
                for a_i, b_i in zip(ia[A == 1], ib[A == 1]):
                    ix = spk_turns[spk[b_i]]; p = pos_in_spk[b_i] + k
                    if p < len(ix) and conv[ix[p]] == conv[b_i]:
                        obs.append(x[ix[p]]); exp.append(loo[ix[p]])
                if exp and sum(exp) > 0:
                    lag[k] = round(float(sum(obs) / sum(exp)), 3)
            # self persistence: A's next own turn
            so, se = [], []
            for a_i in ia[A == 1]:
                ix = spk_turns[spk[a_i]]; p = pos_in_spk[a_i] + 1
                if p < len(ix):
                    so.append(x[ix[p]]); se.append(loo[ix[p]])
            out["feat"][f] = {
                "n_A": int(A.sum()), "pB": round(float(pB), 4), "pB_given_A": round(float(p_ba), 4),
                "pB_given_notA": round(float(p_bna), 4), "lift_raw": round(float(lift_raw), 3),
                "lift_spk": round(float(lift_spk), 3),
                "lift_spk_ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)] if bs else None,
                "lag_lift_spk": lag,
                "self_persist_lift_spk": round(float(sum(so) / sum(se)), 3) if se and sum(se) > 0 else None,
            }
        # length mirroring
        la, lb = d.logc.values[ia], d.logc.values[ib]
        dm = d.logc - d.groupby("spk").logc.transform("mean")
        out["length"] = {"r_raw": round(float(stats.pearsonr(la, lb)[0]), 3),
                         "r_within_speaker": round(float(stats.pearsonr(dm.values[ia], dm.values[ib])[0]), 3)}
        # quartile table: B chars (median) vs A's relative length quartile
        q = pd.qcut(dm.values[ia], 4, labels=False, duplicates="drop")
        tb = pd.DataFrame({"q": q, "B_chars": d.total_chars.values[ib], "B_rel": np.exp(dm.values[ib])})
        out["length"]["B_median_chars_by_A_quartile"] = tb.groupby("q").B_chars.median().round(1).tolist()
        out["length"]["B_rel_len_by_A_quartile"] = tb.groupby("q").B_rel.mean().round(3).tolist()
        if c != "empathetic":
            nm = d.n_msgs.values.astype(float); nmd = nm - d.groupby("spk").n_msgs.transform("mean").values
            out["n_msgs"] = {"r_within_speaker": round(float(stats.pearsonr(nmd[ia], nmd[ib])[0]), 3),
                             "pB_multi_given_A_multi": round(float((nm[ib][nm[ia] >= 3] >= 2).mean()), 3),
                             "pB_multi_given_A_single": round(float((nm[ib][nm[ia] == 1] >= 2).mean()), 3)}
        # convergence (skip empathetic: too short)
        if c != "empathetic":
            conv_res = {}
            d["third"] = d.groupby("conv_id").pos.transform(lambda p: pd.qcut(p, 3, labels=False, duplicates="drop"))
            for f in ["laugh", "emo_any", "elong", "abbr", "starts_lower", "end_none", "logc", "q", "n_msgs", "ellipsis"]:
                if f not in d:
                    continue
                r = d.groupby(["conv_id", "third", "speaker"])[f].mean().astype(float).unstack()
                if r.shape[1] != 2:
                    continue
                gap = (r.iloc[:, 0] - r.iloc[:, 1]).abs().unstack()
                gap = gap.dropna(subset=[0, 2]) if 2 in gap else gap
                if 0 not in gap or 2 not in gap or len(gap) < 8:
                    continue
                w = stats.wilcoxon(gap[0], gap[2]) if (gap[0] != gap[2]).any() else None
                conv_res[f] = {"gap_first": round(float(gap[0].mean()), 4), "gap_last": round(float(gap[2].mean()), 4),
                               "n_dyads": int(len(gap)), "share_converging": round(float((gap[2] < gap[0]).mean()), 3),
                               "wilcoxon_p": round(float(w.pvalue), 4) if w else None}
            out["convergence"] = conv_res
            # dyad-level matching vs random pairing
            match = {}
            for f in ["laugh", "emo_any", "elong", "abbr", "starts_lower", "end_none", "logc", "ellipsis", "end_period"]:
                if f not in d: continue
                r = d.groupby(["conv_id", "speaker"])[f].mean().astype(float).unstack().dropna()
                a, b = r.iloc[:, 0].values, r.iloc[:, 1].values
                real = np.mean(np.abs(a - b))
                perm = [np.mean(np.abs(a - rng.permutation(b))) for _ in range(2000)]
                rr = stats.spearmanr(a, b)[0]
                match[f] = {"mean_abs_gap_real": round(float(real), 4), "mean_abs_gap_random": round(float(np.mean(perm)), 4),
                            "p_perm": round(float(np.mean(np.array(perm) <= real)), 4), "spearman_AB": round(float(rr), 3),
                            "n_dyads": int(len(a))}
            out["dyad_matching"] = match
        res[c] = out
        print("==", c, out["n_pairs"], out["n_dyads"])
        for f, v in out["feat"].items():
            print(f"  {f:14s} nA={v['n_A']:6d} pB={v['pB']:.3f} P(B|A)={v['pB_given_A']:.3f} P(B|~A)={v['pB_given_notA']:.3f} lift={v['lift_raw']:.2f} liftSpk={v['lift_spk']:.2f} ci={v['lift_spk_ci95']} lag={v['lag_lift_spk']} self={v['self_persist_lift_spk']}")
        print("  length", out["length"], out.get("n_msgs"))
        for k in ("convergence", "dyad_matching"):
            if k in out:
                print(" ", k)
                for f, v in out[k].items(): print("    ", f, v)
    json.dump(res, open(f"{ADATA}/a4_mirroring.json", "w"), indent=1)


if __name__ == "__main__":
    main()
