"""a2: NPS — after a user JOINs, are they greeted by name, and does being greeted relate to them posting?"""
import os, sys, json
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import a2_load as L
m = L.messages(); n = m[m.corpus == "nps_chatroom"].sort_values(["conv_id", "idx"])
rows = []
for room, g in n.groupby("conv_id"):
    g = g.reset_index(drop=True)
    for i, r in g.iterrows():
        if r.dialogue_act == "System" and r.text.startswith("JOIN"):
            u = r.speaker
            nxt = g.iloc[i + 1:i + 26]
            nxt = nxt[nxt.dialogue_act != "System"]
            gr = nxt[(nxt.speaker != u) & (nxt.dialogue_act == "Greet") & nxt.text.str.contains(u, regex=False)]
            posts = nxt[nxt.speaker == u]
            prior = (g.iloc[:i].speaker == u).any()
            first_post_pos = posts.index[0] if len(posts) else None
            greeted_before_post = len(gr) and (first_post_pos is None or gr.index[0] < first_post_pos)
            rows.append({"room": room, "user": u, "returning": bool(prior), "greeted": bool(len(gr)),
                         "greeted_before_posting": bool(greeted_before_post), "posts_next25": len(posts),
                         "wb": bool(len(gr) and gr.text.str.contains(r"\bwb\b|welcome", case=False).any())})
d = pd.DataFrame(rows)
res = {"n_joins": len(d), "greeted": round(d.greeted.mean(), 3),
       "by_returning": d.groupby("returning").agg(n=("greeted", "size"), greeted=("greeted", "mean"), wb=("wb", "mean")).round(3).reset_index().to_dict("records"),
       "posts_if_greeted_first": d.groupby("greeted_before_posting").agg(n=("posts_next25", "size"), any_post=("posts_next25", lambda s: (s > 0).mean()), mean_posts=("posts_next25", "mean")).round(3).reset_index().to_dict("records")}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../../analysis/data/a2_nps_join.json"), "w"), indent=1)
