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

# all posts (n~7.9k): addressed / greeting / flirt markers with room FE
a = d.copy()
a["resp"] = a.resp_named.astype(int)
a["greet"] = (a.act == "Greet").astype(int)
a["question"] = a.act.isin(["ynQuestion", "whQuestion"]).astype(int)
a["addressed"] = a.addressed.astype(int)
a["open_call"] = a.fl_open_call.astype(int); a["asl"] = a.fl_asl.astype(int)
a["sexual"] = a.fl_sexual.astype(int); a["compliment"] = a.fl_compliment_petname.astype(int)
m2 = smf.logit("resp ~ addressed*greet + question + open_call + asl + sexual + compliment + C(room)", a).fit(disp=0)
r2 = pd.DataFrame({"OR": np.exp(m2.params), "lo": np.exp(m2.conf_int()[0]), "hi": np.exp(m2.conf_int()[1]), "p": m2.pvalues})
r2 = r2.loc[[i for i in r2.index if not i.startswith("C(")]].round(3)
print(r2)
out["logit_allposts_room_FE"] = r2.reset_index().to_dict("records"); out["n_allposts"] = len(a)
print(f.groupby("room").resp.mean().round(2).to_dict())
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../../analysis/data/a2_nps_firstpost.json"), "w"), indent=1)
