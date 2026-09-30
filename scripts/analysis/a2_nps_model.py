"""a2: logistic model of public named response to a user's first post (NPS) using Jev labels + room FE."""
import os, sys, json
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
import a2_load as L
d = pd.read_pickle(f"{L.SCR}/nps_posts.pkl"); j = pd.read_pickle(f"{L.SCR}/jev_nps.pkl").set_index("idx")
f = d.loc[j.index].join(j)
f["resp"] = f.resp_named.astype(int)
for c in ["j_flirting", "j_addressed", "j_pushy_creepy", "j_easy_reply"]:
    f[c[2:]] = (f[c] > 0.5).astype(int)
f["greet"] = f.j_open_type.str.startswith("greet").astype(int)
m = smf.logit("resp ~ addressed + flirting + pushy_creepy + greet + j_energy + C(room)", f).fit(disp=0)
res = pd.DataFrame({"OR": np.exp(m.params), "lo": np.exp(m.conf_int()[0]), "hi": np.exp(m.conf_int()[1]), "p": m.pvalues})
res = res.loc[[i for i in res.index if not i.startswith("C(")]].round(3)
print(res)
summ = f.groupby("j_open_type").agg(n=("resp", "size"), resp=("resp", "mean"),
                                     stays=("later_posts_total", lambda s: (s >= 3).mean())).round(3)
out = {"logit_room_FE": res.reset_index().to_dict("records"), "by_type": summ.reset_index().to_dict("records"), "n": len(f)}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../../analysis/data/a2_nps_firstpost.json"), "w"), indent=1)
