"""a7: resumo do flerte entre desconhecidos (NPS) + vocabulário EN (NPS flerte × não flerte). Saída: analysis/data/a7_nps_summary.json"""
import json, os, re, sys
from collections import Counter
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_common import OUT
from a7_style import toks, logodds, top_words

p = pd.read_json(os.path.join(OUT, "a7_nps_posts.jsonl"), lines=True)
r = pd.read_json(os.path.join(OUT, "a7_nps_replies.jsonl"), lines=True)
out = {}
p["fl"] = p.is_flirt >= .5
out["n"] = p.grp.value_counts().to_dict()
out["flirt_rate"] = p.groupby("grp").fl.mean().round(3).to_dict()
F = p[p.fl]
out["n_flirt"] = int(len(F))
out["move_dist"] = F.move.value_counts().to_dict()
out["t_int_mean"] = float(F.t_int.mean())
out["t_int_dist"] = F.t_int.round().value_counts().sort_index().to_dict()
out["acts_of_flirts"] = F.act.value_counts().to_dict()
USER = re.compile(r"\d\d-\d\d-[a-z0-9]+User\d+")
F = F.assign(directed=F.text.str.contains(USER))
out["directed_share"] = float(F.directed.mean())
out["broadcast_solicit_share"] = float(F.text.str.contains(r"\bpm\b|\bany (girl|guy|lad|female|male|women|ladies)|\d\d ?/? ?[mf]\b", case=False, regex=True).mean())
# respostas
rr = r[r.R.notna()]
out["directed_flirts"] = int(len(r))
out["got_reply_from_target"] = float(r.R.notna().mean())
ra = rr[rr.addressed >= .5]
out["addressed_replies"] = int(len(ra))
out["resp_dist"] = ra.resp.value_counts().to_dict()
out["resp_alive_mean"] = ra.groupby("resp").alive.mean().round(2).to_dict()
out["r_int_minus_t_int"] = float((ra.r_int - ra.t_int).mean())
out["examples_resp"] = {k: [f"{x.T[:70]} → {x.R[:70]}" for x in g.head(5).itertuples()] for k, g in ra.groupby("resp")}
out["examples_move"] = {k: g.sort_values("move_conf", ascending=False).text.head(6).str[:90].tolist() for k, g in F.groupby("move")}
# vocab EN (NPS): flerte vs não-flerte (entre todos os classificados; prior = todos os posts classificados)
A = Counter(w for t in F.text for w in toks(USER.sub(" ", t)))
N = p[p.is_flirt < .2]
B = Counter(w for t in N.text for w in toks(USER.sub(" ", t)))
prior = A + B
res = logodds(A, B, prior)
out["vocab"] = {"flirt": top_words(res, 40, 3, 1), "neutral": top_words(res, 40, 3, -1), "n_flirt_posts": int(len(F)), "n_neutral_posts": int(len(N))}
json.dump(out, open(os.path.join(OUT, "a7_nps_summary.json"), "w"), ensure_ascii=False, indent=1)
for k, v in out.items():
    if k != "vocab":
        print(k, json.dumps(v, ensure_ascii=False)[:1500])
print("FLIRT:", " ".join(f"{d['w']}({d['n_flirt']})" for d in out["vocab"]["flirt"]))
print("NEUTRAL:", " ".join(f"{d['w']}({d['n_neutral']})" for d in out["vocab"]["neutral"]))
