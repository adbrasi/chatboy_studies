"""a3: contágio emocional, seriedade, volta do humor, antecedentes do riso e p_joke_welcome (só jev_base, sem Jev novo).
Saída: analysis/data/a3_seq.json + prints."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from a3_common import OUT
from a3_load import load, auc, boot_conv

R = {}
D = load()
D["serious"] = D.D_seriousness >= 1.25
D["vuln"] = (D.D_vulnerable >= .5) | (D.D_seeks_support >= .5)
D["playful"] = D.D_playful >= .5
D["joke"] = (D.D_intent == "joke_tease") | (D.D_playful >= .7)
D["prev_serious"] = D.prev_D_seriousness >= 1.25
D["prev_vuln"] = (D.prev_D_vulnerable >= .5) | (D.prev_D_seeks_support >= .5)
D["prev_playful"] = D.prev_D_playful >= .5
D["prev_joke"] = (D.prev_D_intent == "joke_tease") | (D.prev_D_playful >= .7)
D["next_playful"] = D.next_D_playful >= .5

def pr(*a):
    print(*a, flush=True)

# ---------- 1. Contágio: valência / seriedade / playful ----------
pr("\n== 1. contágio ==")
R["contagion"] = {}
rng = np.random.default_rng(1)
for corp, G in list(D.groupby("corpus")) + [("all", D)]:
    out = {}
    for v in ("D_valence", "D_seriousness", "D_playful", "D_arousal"):
        g = G.dropna(subset=["prev_" + v, v])
        r_partner = g[v].corr(g["prev_" + v])
        g2 = G.dropna(subset=["prev2_" + v, v])
        r_own = g2[v].corr(g2["prev2_" + v])
        # linha de base: correlação com um turno aleatório do parceiro na MESMA conversa (nível de conversa)
        perm = []
        for _ in range(50):
            gg = g.copy()
            gg["shuf"] = gg.groupby(["corpus", "conv_id"])["prev_" + v].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
            perm.append(gg[v].corr(gg["shuf"]))
        # within-conv (demeaned)
        dm = g[[v, "prev_" + v]] - g.groupby(["corpus", "conv_id"])[[v, "prev_" + v]].transform("mean")
        out[v] = {"n": len(g), "r_partner_prev": round(r_partner, 3), "r_own_prev": round(r_own, 3),
                  "r_partner_shuffled_same_conv": round(float(np.mean(perm)), 3),
                  "r_within_conv": round(dm[v].corr(dm["prev_" + v]), 3)}
    R["contagion"][corp] = out
    pr(corp, json.dumps(out))
# contágio determinístico (não depende do Jev): riso e emoji
det = {}
for corp, G in list(D.groupby("corpus")) + [("all", D)]:
    g = G[G.prev_speaker.notna()]
    d = {}
    for v in ("laugh", "emoji_any"):
        pp = g[g["prev_" + v] == True][v].mean(); pn = g[g["prev_" + v] == False][v].mean()
        d[v] = {"p_if_partner_did": round(pp, 3), "p_if_partner_didnt": round(pn, 3), "n_did": int((g["prev_" + v] == True).sum())}
    det[corp] = d
R["contagion_deterministic"] = det
# versão pareada dentro da conversa (controla estilo da pessoa/conversa)
pd_ = {}
for corp, G in D[D.prev_speaker.notna()].groupby("corpus"):
    for v in ("laugh", "emoji_any"):
        m = G.groupby(["conv_id", "prev_" + v])[v].mean().unstack("prev_" + v).dropna()
        diff = m[True] - m[False]
        pd_[f"{corp}:{v}"] = {"n_conv": len(m), "mean_p_if_partner_did": round(m[True].mean(), 3),
                              "mean_p_if_partner_didnt": round(m[False].mean(), 3), "share_conv_higher": round((diff > 0).mean(), 3)}
R["contagion_det_paired"] = pd_
pr("det paired", json.dumps(pd_))
pr("determ", json.dumps(det))
# polaridade: parceiro negativo -> eu?
g = D.dropna(subset=["prev_D_valence", "D_valence"])
g = g.assign(pb=pd.cut(g.prev_D_valence, [-.1, 1.5, 2.5, 4.1], labels=["neg", "neu", "pos"]))
R["valence_by_partner_bin"] = g.groupby(["corpus", "pb"], observed=True).D_valence.agg(["mean", "count"]).round(2).reset_index().to_dict("records")
pr(pd.DataFrame(R["valence_by_partner_bin"]))

# ---------- 2. Depois de um turno sério/vulnerável ----------
pr("\n== 2. resposta a turno sério ==")
res = []
for corp, G in D.groupby("corpus"):
    g = G[G.prev_speaker.notna()]
    for cond in ("prev_serious", "prev_vuln"):
        for flag in (True, False):
            s = g[g[cond] == flag]
            res.append({"corpus": corp, "cond": cond, "flag": flag, "n": len(s),
                        "laugh": round(s.laugh.mean(), 3), "emoji": round(s.emoji_any.mean(), 3),
                        "playful": round(s.playful.mean(), 3),
                        "chars_median": float(s.total_chars.median()), "chars_mean": round(s.total_chars.mean(), 1),
                        "n_msgs_mean": round(s.n_msgs.mean(), 2), "has_q": round(s.has_q.mean(), 3),
                        "latency_median_s": float(s.response_latency_s.median()) if s.response_latency_s.notna().any() else None,
                        "serious_self": round(s.serious.mean(), 3),
                        "comfort_intent": round((s.D_intent == "comfort_support").mean(), 3)})
R["after_serious"] = res
pr(pd.DataFrame(res).to_string())
# pareado por conversa (controla pessoa/conversa): diferença média dentro da conversa
def paired(col, cond="prev_serious"):
    g = D[D.prev_speaker.notna()]
    m = g.groupby(["corpus", "conv_id", cond])[col].mean().unstack(cond).dropna()
    diff = (m[True] - m[False])
    return {"n_conv": len(diff), "mean_diff": round(diff.mean(), 3), "share_conv_lower": round((diff < 0).mean(), 3)}
def paired_c(col, cond, corp):
    g = D[D.prev_speaker.notna() & (D.corpus == corp)]
    m = g.groupby(["conv_id", cond])[col].mean().unstack(cond).dropna()
    diff = (m[True] - m[False]).values
    rng2 = np.random.default_rng(0)
    bs = [rng2.choice(diff, len(diff)).mean() for _ in range(2000)]
    return {"n_conv": len(diff), "mean_diff": round(float(diff.mean()), 3), "ci95": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])]}
R["after_serious_paired_by_corpus"] = {f"{corp}:{cond}:{c}": paired_c(c, cond, corp) for corp in ("maichat", "whatsapp_nl") for cond in ("prev_serious", "prev_vuln")
                                       for c in ("laugh", "emoji_any", "total_chars", "playful", "has_q", "n_msgs", "response_latency_s")}
pr(pd.DataFrame(R["after_serious_paired_by_corpus"]).T.to_string())
R["after_serious_paired"] = {c: paired(c) for c in ("laugh", "emoji_any", "total_chars", "playful")}
R["after_vuln_paired"] = {c: paired(c, "prev_vuln") for c in ("laugh", "emoji_any", "total_chars", "playful")}
pr("paired serious", R["after_serious_paired"]); pr("paired vuln", R["after_vuln_paired"])
# latência (whatsapp, resolução de minuto): proporção de resposta em <2 min
w = D[(D.corpus == "whatsapp_nl") & D.prev_speaker.notna() & D.response_latency_s.notna()]
R["wa_latency_after_serious"] = {str(k): {"n": len(x), "p_under_2min": round((x.response_latency_s < 120).mean(), 3),
                                           "median_s": float(x.response_latency_s.median())} for k, x in w.groupby("prev_serious")}
pr("wa latency", R["wa_latency_after_serious"])
m = D[(D.corpus == "maichat") & D.prev_speaker.notna()]
R["mai_latency_after_serious"] = {str(k): {"n": len(x), "median_s": float(x.response_latency_s.median()),
                                            "compose_median": None} for k, x in m.groupby("prev_serious")}
pr("mai latency", R["mai_latency_after_serious"])

# ---------- 3. Volta do humor depois de um momento sério ----------
pr("\n== 3. volta do humor ==")
ep = []
for (corp, cid, sess), G in D.groupby(["corpus", "conv_id", "session"]):
    G = G.sort_values("turn_idx")
    idx = G.turn_idx.tolist(); rows = G.to_dict("records")
    for i, r in enumerate(rows):
        if not r["serious"]: continue
        if i > 0 and rows[i - 1]["serious"] and rows[i - 1]["turn_idx"] == r["turn_idx"] - 1: continue  # só o início do episódio
        opener = r["speaker"]
        # fim do episódio
        j = i
        while j + 1 < len(rows) and rows[j + 1]["serious"] and rows[j + 1]["turn_idx"] == rows[j]["turn_idx"] + 1: j += 1
        ep_len = j - i + 1
        back = None
        for k in range(i + 1, len(rows)):
            if rows[k]["turn_idx"] != rows[k - 1]["turn_idx"] + 1: break
            if rows[k]["laugh"] or rows[k]["D_playful"] >= .5:
                back = (k - i, "opener" if rows[k]["speaker"] == opener else "other", rows[k]["laugh"]); break
        ep.append({"corpus": corp, "len": ep_len, "turns_to_humor": back[0] if back else None,
                   "who": back[1] if back else None, "by_laugh": back[2] if back else None,
                   "censored": back is None, "remaining": len(rows) - i - 1})
E = pd.DataFrame(ep)
R["humor_return"] = {}
for corp, g in list(E.groupby("corpus")) + [("all", E)]:
    gg = g[~g.censored]
    R["humor_return"][corp] = {"n_episodes": len(g), "episode_len_mean": round(g.len.mean(), 2),
                               "share_returned": round(1 - g.censored.mean(), 3),
                               "turns_to_humor_median": float(gg.turns_to_humor.median()) if len(gg) else None,
                               "turns_to_humor_q": gg.turns_to_humor.quantile([.25, .75]).tolist() if len(gg) else None,
                               "who_other": round((gg.who == "other").mean(), 3) if len(gg) else None,
                               "within_2_turns": round((gg.turns_to_humor <= 2).mean(), 3) if len(gg) else None}
pr(json.dumps(R["humor_return"], indent=0))
# a partir do FIM do episódio + linha de base (turno não-sério qualquer)
def hazard(D, start_rows):
    pass
ep2, base2 = [], []
for (corp, cid, sess), G in D.groupby(["corpus", "conv_id", "session"]):
    rows = G.sort_values("turn_idx").to_dict("records")
    hum = [bool(r["laugh"] or r["D_playful"] >= .5) for r in rows]
    for i, r in enumerate(rows):
        ends_ep = r["serious"] and (i + 1 < len(rows)) and not rows[i + 1]["serious"] and rows[i + 1]["turn_idx"] == r["turn_idx"] + 1
        plain = (not r["serious"]) and not hum[i] and (i + 1 < len(rows))
        if not (ends_ep or plain): continue
        opener = r["speaker"]
        k_found = None
        for k in range(i + 1, min(len(rows), i + 11)):
            if rows[k]["turn_idx"] != rows[k - 1]["turn_idx"] + 1: break
            if hum[k]: k_found = k; break
        rec = {"corpus": corp, "k": (k_found - i) if k_found else None, "who": (None if k_found is None else ("same" if rows[k_found]["speaker"] == opener else "other")),
               "by_laugh": None if k_found is None else rows[k_found]["laugh"], "next_hum": hum[i + 1]}
        (ep2 if ends_ep else base2).append(rec)
for name, L_ in (("after_serious_end", ep2), ("baseline_nonserious_nonhumor", base2)):
    X = pd.DataFrame(L_)
    R["humor_return_" + name] = {c: {"n": len(x), "p_humor_next_turn": round(x.next_hum.mean(), 3),
                                     "p_humor_within_10": round(x.k.notna().mean(), 3), "k_median": float(x.k.median()),
                                     "who_other_share": round((x.who == "other").mean() / max(x.k.notna().mean(), 1e-9), 3),
                                     "via_laugh_share": round(x.by_laugh.dropna().astype(bool).mean(), 3)} for c, x in list(X.groupby("corpus")) + [("all", X)]}
    pr(name, json.dumps(R["humor_return_" + name]))
# linha de base: de um turno NÃO sério qualquer, quantos turnos até o próximo turno de humor?
# (o "who" é comparado com 50%)

# ---------- 4. Riso: antecedentes e quem ri ----------
pr("\n== 4. riso ==")
g = D[D.prev_speaker.notna()]
R["laugh_given_prev"] = {}
for corp, G in list(g.groupby("corpus")) + [("all", g)]:
    R["laugh_given_prev"][corp] = {
        "p_laugh_base": round(G.laugh.mean(), 3),
        "p_laugh_after_partner_joke": round(G[G.prev_joke == True].laugh.mean(), 3),
        "p_laugh_after_partner_nonjoke": round(G[G.prev_joke == False].laugh.mean(), 3),
        "p_laugh_after_partner_laugh": round(G[G.prev_laugh == True].laugh.mean(), 3),
        "n_partner_joke": int((G.prev_joke == True).sum()),
        "share_laughs_preceded_by_joke": round((G[G.laugh].prev_joke == True).mean(), 3),
        "share_turns_preceded_by_joke": round((G.prev_joke == True).mean(), 3),
        # quem ri: quem fez a piada (no próprio turno) x quem ouviu (no turno seguinte)
        "p_joker_laughs_in_own_joke_turn": round(G[G.joke].laugh.mean(), 3),
        "p_listener_laughs_next": round(G[G.prev_joke == True].laugh.mean(), 3),
    }
pr(json.dumps(R["laugh_given_prev"], indent=0))
# intent do parceiro antes do riso x antes de não-riso
tab = pd.crosstab(g.prev_D_intent, g.laugh, normalize="columns").round(3)
R["prev_intent_by_laugh"] = tab.to_dict()
pr(tab)
# intent do próprio turno com riso
tab2 = pd.crosstab(D.D_intent, D.laugh, normalize="columns").round(3)
R["own_intent_by_laugh"] = tab2.to_dict(); pr(tab2)
# riso x tensão/reclamação no próprio turno
R["laugh_in_complaint_turns"] = {c: {"p_laugh_if_complain": round(G[G.D_intent == "disagree_complain"].laugh.mean(), 3),
                                     "n_complain": int((G.D_intent == "disagree_complain").sum()),
                                     "p_laugh_if_tension": round(G[G.D_tension >= .5].laugh.mean(), 3),
                                     "n_tension": int((G.D_tension >= .5).sum()),
                                     "p_laugh_base": round(G.laugh.mean(), 3)} for c, G in D.groupby("corpus")}
pr(R["laugh_in_complaint_turns"])
# posição do riso na mensagem
def pos(t):
    import re
    from a3_load import SB
    t2 = t.lower()
    m = re.search(r"(?:a?ha(?:ha)+h?|he(?:he)+|hi(?:hi)+|lo+l+|lmf?ao+|haa+|hah+|[😂🤣😆😅😁😄😹" + SB + "])", t2)
    if not m: return None
    if not re.sub(r"[\W_]+|a?ha(?:ha)+h?|he(?:he)+|hi(?:hi)+|lo+l+|lmf?ao+|haa+|hah+", "", t2): return "only"
    return "start" if m.start() <= 2 else ("end" if m.end() >= len(t2.rstrip(" !.?)(:;xX")) - 2 else "middle")
L = D[D.laugh].copy(); L["pos"] = L.text.map(pos)
R["laugh_position"] = L.groupby("corpus").pos.value_counts(normalize=True).round(3).unstack().to_dict("index")
pr(R["laugh_position"])

# ---------- 5. p_joke_welcome ----------
pr("\n== 5. joke welcome ==")
g = D[D.p_joke_welcome.notna()]
R["joke_welcome"] = {}
for corp, G in list(g.groupby("corpus")) + [("all", g)]:
    j = G[G.joke]
    d = {"n": len(G), "auc_predicts_S_jokes": round(auc(G.joke, G.p_joke_welcome), 3),
         "auc_predicts_S_playful": round(auc(G.playful, G.p_joke_welcome), 3),
         "auc_predicts_S_laugh": round(auc(G.laugh, G.p_joke_welcome), 3)}
    # quando S brincou: o parceiro riu/continuou brincando?
    jj = j[j.next_speaker.notna()]
    jj = jj.assign(recv=(jj.next_laugh == True) | (jj.next_D_playful >= .5))
    d["n_S_joked_with_next"] = len(jj)
    d["auc_welcome_predicts_reception"] = round(auc(jj.recv, jj.p_joke_welcome), 3)
    q = pd.cut(jj.p_joke_welcome, [0, .3, .6, 1.01], labels=["low<.3", "mid", "high>=.6"])
    d["reception_by_bin"] = jj.groupby(q, observed=True).recv.agg(["mean", "count"]).round(3).reset_index().astype(str).to_dict("records")
    d["partner_laughs_by_bin"] = jj.groupby(q, observed=True).next_laugh.mean().round(3).astype(str).to_dict()
    R["joke_welcome"][corp] = d
    pr(corp, json.dumps(d))
# p_joke_welcome vs contexto sério
R["joke_welcome_by_prev_serious"] = {f"{a}|prev_serious={b}": v for (a, b), v in g.groupby(["corpus", "prev_serious"]).p_joke_welcome.mean().round(3).items()}
pr(R["joke_welcome_by_prev_serious"])

# ---------- 6. detectores preditivos já presentes no passe P ----------
pr("\n== 6. P como detectores de 'hora de' ==")
g = D[D.p_tone.notna()].copy()
g["tp_playful"] = g.p_tone_probs.map(lambda d: d.get("playful", 0))
g["tp_support"] = g.p_tone_probs.map(lambda d: d.get("supportive_serious", 0))
g["tp_warm"] = g.p_tone_probs.map(lambda d: d.get("warm_affectionate", 0))
R["P_detectors"] = {}
for corp, G in list(g.groupby("corpus")) + [("all", g)]:
    R["P_detectors"][corp] = {
        "brincar: auc tone_playful->playful": round(auc(G.playful, G.tp_playful), 3),
        "brincar: auc p_laugh->laugh": round(auc(G.laugh, G.p_laugh), 3),
        "serio: auc tone_support->serious": round(auc(G.serious, G.tp_support), 3),
        "acolher: auc tone_support->comfort_intent": round(auc(G.D_intent == "comfort_support", G.tp_support), 3),
        "perguntar: auc p_question->has_q": round(auc(G.has_q, G.p_question), 3),
        "base_rates": {"playful": round(G.playful.mean(), 3), "serious": round(G.serious.mean(), 3),
                       "comfort": round((G.D_intent == "comfort_support").mean(), 3), "has_q": round(G.has_q.mean(), 3)}}
pr(json.dumps(R["P_detectors"], indent=0))
# o que o parceiro faz depois de 'seeks_support' do outro
g = D[D.prev_speaker.notna() & (D.prev_D_seeks_support >= .5)]
R["after_seeks_support_intent"] = g.groupby("corpus").D_intent.value_counts(normalize=True).round(3).unstack(0).fillna(0).to_dict()
R["after_seeks_support_n"] = g.groupby("corpus").size().to_dict()
pr(R["after_seeks_support_n"]); pr(pd.DataFrame(R["after_seeks_support_intent"]))

json.dump(R, open(os.path.join(OUT, "a3_seq.json"), "w"), indent=1, default=str, ensure_ascii=False)
D.to_pickle(os.path.join(os.environ.get("A3_SCRATCH", "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad/a3"), "D.pkl"))
