"""a7: episódios de flerte/afeto — quanto duram, quão contagioso é (P(R é flerte | T flerte)), onde caem na sessão.
Saída: analysis/data/a7_episodes.json"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_common import load, OUT

df = load().sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
a = df[df.annotated].copy()
a["fa"] = (a.D_flirting >= .5) | (a.D_emotion == "affection") | (a.D_intent == "compliment_affection")
out = {}
for c in ["maichat", "whatsapp_nl"]:
    s = a[a.corpus == c].copy()
    g = s.groupby(["conv_id", "session"])
    s["next_fa"] = g.fa.shift(-1); s["next_spk"] = g.speaker.shift(-1); s["next_ti"] = g.turn_idx.shift(-1)
    ok = s.next_fa.notna() & (s.next_spk != s.speaker) & (s.next_ti == s.turn_idx + 1)
    base = s[ok].next_fa.astype(bool).mean()
    cond = s[ok & s.fa].next_fa.astype(bool).mean()
    # comprimento dos episódios (turnos consecutivos fa)
    runs = []
    for _, gg in g:
        r = 0
        for v in gg.fa:
            if v: r += 1
            elif r: runs.append(r); r = 0
        if r: runs.append(r)
    runs = np.array(runs)
    # sessões com algum flerte
    sess = g.fa.any()
    out[c] = {"fa_rate": float(s.fa.mean()), "P_R_fa_given_T_fa": float(cond), "P_R_fa_base": float(base),
              "lift": float(cond / base), "n_pairs_fa": int((ok & s.fa).sum()),
              "episode_len_mean": float(runs.mean()), "episode_len_dist": pd.Series(runs).value_counts().sort_index().to_dict(),
              "sessions_with_fa": float(sess.mean()), "n_sessions": int(len(sess)),
              "fa_in_first_3_turns": float(s[s.fa].turn_in_session.lt(3).mean()),
              "fa_in_last_3_turns": float((s[s.fa].turns_left < 3).mean())}
json.dump(out, open(os.path.join(OUT, "a7_episodes.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
