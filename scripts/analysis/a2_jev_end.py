"""a2: NEW Jev experiment — better end-of-conversation detectors on whatsapp_nl (vs base P.p_end).
Sample: all jev_base whatsapp turns with a P pass that are the last of their session + 420 random other ones.
State = same as base P pass (8 previous same-session turns, with reply-gap buckets) + clock time of the last message."""
import os, sys, random, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import a2_load as L
from annotate_base import fmt_turn, CTX_TURNS
from jev import ask_many, choice, noul, score, summary
from a2_closings import auc, boot_auc

def qs():
    w = "`next_speaker`"
    return {
        "quiet_after": noul(f"Will {w}'s next turn be the last message of this exchange, with the chat going quiet for hours afterwards?"),
        "open_loop": noul(f"Is there an unanswered question or unresolved request addressed to {w} in the last message?"),
        "resolved": noul("Has the current topic reached a natural conclusion (plan settled, question answered, story finished)?"),
        "closure": score("How close is this conversation to ending for now?", ["in full swing", "slowing down", "wrapping up", "saying goodbye"]),
    }

def main():
    cl = pd.read_pickle(f"{L.SCR}/closing.pkl")
    w = cl[(cl.corpus == "whatsapp_nl") & cl.has_P]
    pos = w[w.to_end == 0]
    neg = w[w.to_end > 0].sample(420, random_state=3)
    samp = pd.concat([pos, neg])
    t = L.turns(); t = t[t.corpus == "whatsapp_nl"]
    tg = {k: g.sort_values("turn_idx").reset_index(drop=True) for k, g in t.groupby("conv_id")}
    items = []
    for _, r in samp.iterrows():
        ct = tg[r.conv_id]
        i = ct.index[ct.turn_idx == r.turn_idx][0]
        ctx = [x for _, x in ct.iloc[max(0, i - CTX_TURNS):i].iterrows() if x.session == r.session]
        last_ts = pd.to_datetime(ctx[-1].ts_end[:19])
        state = {"conversation_so_far": [fmt_turn(x) for x in ctx], "next_speaker": r.speaker,
                 "clock_time_of_last_message": last_ts.strftime("%A %H:%M"),
                 "note": "Dutch WhatsApp chat"}
        items.append((state, qs()))
    print("calls", len(items), flush=True)
    res = ask_many(items, workers=4)
    rows = []
    for (_, r), a in zip(samp.iterrows(), res):
        if a is None: continue
        rows.append({"conv_id": r.conv_id, "turn_idx": r.turn_idx, "to_end": r.to_end, "P_p_end": r.P_p_end,
                     "closing_T": bool(r.bye or r.closing_intent),
                     **{k: v.get("noul", v.get("score")) for k, v in a.items()}})
    df = pd.DataFrame(rows); df.to_pickle(f"{L.SCR}/jev_end.pkl")
    y = df.to_end == 0
    out = {"n": len(df), "pos": int(y.sum())}
    for f, s in {"P_p_end(base)": df.P_p_end, "quiet_after": df.quiet_after, "-open_loop": -df.open_loop,
                 "resolved": df.resolved, "closure": df.closure}.items():
        out[f] = {"auc_last": round(auc(y, s), 3), "ci": [round(v, 3) for v in boot_auc(y.values, s.values)],
                  "auc_closingT": round(auc(df.closing_T, s), 3)}
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X = df[["P_p_end", "quiet_after", "open_loop", "resolved", "closure"]].values
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(X, y, df.conv_id):
        pred[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    out["combined_cv_auc_last"] = round(auc(y, pred), 3)
    print(json.dumps(out, indent=1)); print(summary())
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "../../analysis/data/a2_end_detectors.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
