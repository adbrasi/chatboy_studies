"""a2: export small summaries (openings, reopenings, outcomes) to analysis/data/."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import a2_load as L
S = L.SCR; OUT = os.path.join(os.path.dirname(__file__), "../../analysis/data")
s = pd.read_pickle(f"{S}/sessions.pkl"); x = pd.read_pickle(f"{S}/open_joined.pkl")
out = {}
out["rule_open_type"] = {c: g.open_type.value_counts(normalize=True).round(3).to_dict() for c, g in s.groupby("corpus")}
x["w"] = 1 / x.groupby("conv_id").sidx.transform("size")
out["jev_open_type_whatsapp"] = {"n": len(x), "chats": int(x.conv_id.nunique()),
                                 "raw": x.open_type_y.value_counts(normalize=True).round(3).to_dict(),
                                 "chat_weighted": (x.groupby("open_type_y").w.sum() / x.w.sum()).round(3).to_dict()}
x["gapb"] = pd.cut(x.gap_h, [3, 6, 12, 24, 72, 168, 1e5], labels=["3-6h", "6-12h", "12-24h", "1-3d", "3-7d", ">7d"])
tab = x.groupby("gapb", observed=True).agg(n=("sidx", "size"), greet=("has_greeting", lambda v: (v > .5).mean()),
      ack_gap=("acknowledges_gap", lambda v: (v > .5).mean()), refers_back=("refers_back", lambda v: (v > .5).mean()),
      late_reply=("open_type_y", lambda v: (v == "late_reply").mean()),
      same_as_prev_last=("opener", "size")).round(3)
tab["same_as_prev_last"] = x.assign(sa=x.opener == x.prev_last_speaker).groupby("gapb", observed=True).sa.mean().round(3)
out["reopen_by_gap"] = tab.reset_index().astype({"gapb": str}).to_dict("records")
x["loglen"] = np.log(x.n_turns); x["rel_len"] = x.loglen - x.groupby("conv_id").loglen.transform("mean")
x["replied"] = x.reply_type.notna()
out["outcome_by_type"] = x.groupby("open_type_y").agg(n=("sidx", "size"), replied=("replied", "mean"),
      n_turns_med=("n_turns", "median"), rel_len=("rel_len", "mean"), reply_lat_med_min=("reply_latency_s", lambda v: v.median() / 60)).round(3).reset_index().to_dict("records")
out["outcome_by_hook"] = x.groupby(x.invites_reply > .5).agg(n=("sidx", "size"), replied=("replied", "mean"),
      n_turns_med=("n_turns", "median"), rel_len=("rel_len", "mean")).round(3).reset_index().to_dict("records")
w = s[s.corpus == "whatsapp_nl"]
share = w.groupby("conv_id").opener.agg(lambda v: v.value_counts(normalize=True).iloc[0])
out["initiative_major_share_per_chat"] = share.describe().round(3).to_dict()
out["initiative_chats_over_65pct"] = round((share > .65).mean(), 3)
print(json.dumps(out["initiative_major_share_per_chat"]), out["initiative_chats_over_65pct"])
json.dump(out, open(f"{OUT}/a2_openings_summary.json", "w"), indent=1, default=str)

# per-row Jev labels (texts truncated) for reuse
cols = ["conv_id", "session", "gap_h", "hour", "opener", "prev_last_speaker", "open_text", "reply_text", "reply_latency_s", "n_turns",
        "open_type_x", "open_type_y", "open_type_conf", "acknowledges_gap", "refers_back", "has_greeting", "invites_reply", "warmth", "rel"]
e = x[cols].rename(columns={"open_type_x": "rule_type", "open_type_y": "jev_type"}).copy()
e["open_text"] = e.open_text.str[:120]; e["reply_text"] = e.reply_text.str[:80]
e.round(3).to_csv(f"{OUT}/a2_jev_openings.csv.gz", index=False, compression="gzip")
n = pd.read_pickle(f"{S}/nps_posts.pkl"); jn = pd.read_pickle(f"{S}/jev_nps.pkl").set_index("idx")
f = n.loc[jn.index].join(jn)[["room", "speaker", "text", "act", "addressed", "resp_named", "later_posts_total",
                              "j_open_type", "j_flirting", "j_addressed", "j_easy_reply", "j_pushy_creepy", "j_energy"]]
f["text"] = f.text.str[:120]
f.round(3).to_csv(f"{OUT}/a2_jev_nps_firstposts.csv.gz", index=False, compression="gzip")
pd.read_pickle(f"{S}/jev_end.pkl").round(3).to_csv(f"{OUT}/a2_jev_end.csv.gz", index=False, compression="gzip")
