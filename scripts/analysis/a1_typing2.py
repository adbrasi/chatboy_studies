"""a1: 'clean' composition formula (messages typed after the partner's message arrived, no overlap) and
think/reading time before typing as a function of the partner's message length. Uses a1_typing output."""
import json, os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "data")
K = pd.read_csv(os.path.join(OUT, "a1_typing_messages.csv.gz")).sort_values(["conv_id", "idx"])
K["prev_len"] = K.groupby("conv_id").final_len.shift(1)
res = {}
clean = K[(K.start_rel_prev_s >= 0) & (K.n_resets == 0) & (K.compose_s > 0.1) & (K.compose_s < 120)]
for name, g in [("clean_all", clean), ("clean_mobile", clean[clean.device == "Mobile"]), ("clean_desktop", clean[clean.device == "Desktop"])]:
    fits = {qq: smf.quantreg("compose_s ~ final_len", g).fit(q=qq).params.round(3).to_dict() for qq in (.25, .5, .75)}
    res[name] = {"n": len(g), "quantreg": fits,
                 "sec_per_char_by_len": g.assign(b=pd.cut(g.final_len, [0, 5, 15, 30, 60, 120, 1000])).groupby("b", observed=True)
                 .apply(lambda x: pd.Series({"n": len(x), "compose_med": x.compose_s.median(), "compose_p25": x.compose_s.quantile(.25), "compose_p75": x.compose_s.quantile(.75)})).round(2).rename(index=str).to_dict(orient="index")}
# think time: partner msg arrived -> first keystroke (only when prev msg is partner's and started after it arrived)
th = K[(K.prev_is_partner == True) & (K.start_rel_prev_s >= 0) & (K.start_rel_prev_s < 120)]
res["think_time"] = {"n": len(th), "p25": round(th.start_rel_prev_s.quantile(.25), 1), "med": round(th.start_rel_prev_s.median(), 1),
                     "p75": round(th.start_rel_prev_s.quantile(.75), 1),
                     "rho_vs_partner_len": round(stats.spearmanr(th.prev_len, th.start_rel_prev_s)[0], 3),
                     "quantreg_med": smf.quantreg("start_rel_prev_s ~ prev_len", th).fit(q=.5).params.round(3).to_dict(),
                     "by_partner_len": th.assign(b=pd.cut(th.prev_len, [0, 10, 30, 60, 1000])).groupby("b", observed=True).start_rel_prev_s.median().round(1).rename(index=str).to_dict(),
                     "by_seriousness": th.assign(b=pd.cut(th.D_seriousness, [-.1, .5, 1.5, 3.1])).groupby("b", observed=True).start_rel_prev_s.agg(["size", "median"]).round(1).rename(index=str).to_dict(orient="index"),
                     "by_vulnerable": th.assign(b=th.D_vulnerable >= .5).groupby("b").start_rel_prev_s.agg(["size", "median"]).round(1).rename(index=str).to_dict(orient="index")}
ov = K[K.prev_is_partner == True]
res["overlap"] = {"share_started_before_partner_msg": round(float((ov.start_rel_prev_s < 0).mean()), 3),
                  "reset_if_overlap": round(float(ov[ov.start_rel_prev_s < 0].n_resets.gt(0).mean()), 3),
                  "reset_if_not": round(float(ov[ov.start_rel_prev_s >= 0].n_resets.gt(0).mean()), 3),
                  "deleted_if_overlap": round(float(ov[ov.start_rel_prev_s < 0].deleted.gt(0).mean()), 3),
                  "deleted_if_not": round(float(ov[ov.start_rel_prev_s >= 0].deleted.gt(0).mean()), 3),
                  "n_overlap": int((ov.start_rel_prev_s < 0).sum())}
print(json.dumps(res, indent=1, default=str))
json.dump(res, open(os.path.join(OUT, "a1_typing2.json"), "w"), indent=1, default=str)
