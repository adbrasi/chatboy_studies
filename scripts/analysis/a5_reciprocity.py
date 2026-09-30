"""a5 — Reciprocidade: devolução de pergunta, quem pergunta, equilíbrio × duração, quem inicia tópicos."""
import json, re, warnings
import numpy as np, pandas as pd
from scipy import stats
from a5_common import load_all, prop_ci, OUT
warnings.filterwarnings("ignore")

RET_EN = re.compile(r"\b(and|n) (you|u|urs|yours)\b|\bw(hat|hbu|bu)\b.*\b(you|u)\b|\bwbu\b|\bhbu\b|\bwby\b|\b(what|how) about (you|u)+\b|"
                    r"\byou\?|\bu\?|\bwhat about urs\b|\band yourself\b|\bhow was yours\b|\bhow bout (you|u)\b|\byouu+\b", re.I)
RET_NL = re.compile(r"\ben (jij|jou|jullie|u|zelf|bij jou|met jou|jouw)\b|\bjij\s*\?|\bjij ook\b|\bmet jou\s*\?|\bzelf\s*\?|"
                    r"\bhoe is het met jou\b|\bhoe gaat het met jou\b|\ben hoe is het\b|\bbij jou\s*\?|\ben die van jou\b", re.I)
df = load_all()
res = {}
for corpus, RE in (("maichat", RET_EN), ("whatsapp_nl", RET_NL)):
    d = df[df.corpus == corpus].copy()
    d["ret"] = d.text.str.contains(RE) & d.has_q
    g = d.groupby(["conv_id"])
    d["prev_q"] = g.has_q.shift(1).fillna(False).astype(bool)
    d["prev_spk"] = g.speaker.shift(1)
    d["prev_text"] = g.text.shift(1)
    d["prev_sess"] = g.session.shift(1)
    after_q = d[d.prev_q & (d.prev_spk != d.speaker) & (d.prev_sess == d.session)]
    k, n = int(after_q.ret.sum()), len(after_q)
    p, lo, hi = prop_ci(k, n)
    r = {"n_resposta_a_pergunta": n, "p_devolve": round(p, 3), "ci": [round(lo, 3), round(hi, 3)],
         "p_devolve_sem_pergunta_antes": round(d[~d.prev_q].ret.mean(), 4)}
    # após pergunta de "como vai" (estilo pessoal genérico)
    how = re.compile(r"how (are|r) (you|u)|how was|how's|hows|what are you (doing|up to)|wyd|hoe (is|gaat|was)|wat (ben|doe) je|alles goed", re.I)
    ah = after_q[after_q.prev_text.str.contains(how)]
    r["n_apos_como_vai"] = len(ah); r["p_devolve_apos_como_vai"] = round(ah.ret.mean(), 3) if len(ah) else None
    # o parceiro responde à devolução? e o engajamento no turno seguinte
    d["next_ret"] = g.ret.shift(-1)
    print(corpus, r)
    print(" exemplos devolução:", after_q[after_q.ret].sample(min(8, int(after_q.ret.sum())), random_state=2)[["prev_text", "text"]].values.tolist())
    # ---- quem pergunta mais: por conversa
    conv = []
    for cid, c in d.groupby("conv_id"):
        spk = c.speaker.value_counts()
        if len(spk) < 2: continue
        a, b = spk.index[:2]
        ca, cb = c[c.speaker == a], c[c.speaker == b]
        qa, qb = ca.has_q.sum(), cb.has_q.sum()
        wa, wb = ca.total_chars.sum(), cb.total_chars.sum()
        sess = c.groupby("session").size()
        o = {"conv_id": cid, "n_turns": len(c), "q_share_max": max(qa, qb) / max(1, qa + qb), "chars_share_max": max(wa, wb) / max(1, wa + wb),
             "q_rate": (qa + qb) / len(c), "q_asym": abs(qa - qb) / max(1, qa + qb), "char_asym": abs(wa - wb) / max(1, wa + wb),
             "sess_len_mean": sess.mean(), "sess_len_median": sess.median(), "ret_rate": c.ret.sum() / max(1, c.has_q.sum()),
             "same_asker_talker": (qa > qb) == (wa > wb), "n_q": qa + qb}
        if "D_engagement" in c and c.D_engagement.notna().sum() > 20:
            o["eng_mean"] = c.D_engagement.mean()
            ts = c[c.D_topic_shift.notna()]
            ts_a, ts_b = (ts[ts.speaker == a].D_topic_shift > .5).sum(), (ts[ts.speaker == b].D_topic_shift > .5).sum()
            o["topic_init_share_max"] = max(ts_a, ts_b) / max(1, ts_a + ts_b)
            o["topic_share_a"] = ts_a / max(1, ts_a + ts_b); o["q_share_a"] = qa / max(1, qa + qb); o["char_share_a"] = wa / max(1, wa + wb)
        conv.append(o)
    cv = pd.DataFrame(conv)
    L = "sess_len_mean" if corpus == "whatsapp_nl" else "n_turns"
    r["n_conv"] = len(cv)
    r["q_share_max_median"] = round(cv.q_share_max.median(), 3)
    r["chars_share_max_median"] = round(cv.chars_share_max.median(), 3)
    r["p_same_person_asks_and_talks_more"] = round(cv.same_asker_talker.mean(), 3)
    for x in ["q_asym", "char_asym", "q_rate", "ret_rate"] + (["eng_mean"] if "eng_mean" in cv else []):
        rho, pv = stats.spearmanr(cv[x], cv[L], nan_policy="omit")
        r[f"spearman_{x}_vs_{L}"] = [round(rho, 3), round(pv, 4)]
    if "topic_init_share_max" in cv:
        r["topic_init_share_max_median"] = round(cv.topic_init_share_max.median(), 3)
        for x in ("q_share_a", "char_share_a"):
            rho, pv = stats.spearmanr(cv[x], cv.topic_share_a, nan_policy="omit")
            r[f"spearman_topic_share_vs_{x}"] = [round(rho, 3), round(pv, 4)]
    # turn-level: balance local — o turno de B após turno de A: A pergunta → B escreve mais?
    d["next_chars"] = g.total_chars.shift(-1); d["next_spk"] = g.speaker.shift(-1)
    x = d[(d.next_spk != d.speaker) & d.next_chars.notna()]
    r["next_chars_median_after_q"] = float(x[x.has_q].next_chars.median()); r["next_chars_median_after_noq"] = float(x[~x.has_q].next_chars.median())
    # quem inicia tópicos (Jev) — relação com quem pergunta, por turno
    if d.D_topic_shift.notna().any():
        j = d[d.D_topic_shift.notna()]
        r["p_topic_shift_turn_has_q"] = round(j[j.D_topic_shift > .5].has_q.mean(), 3)
        r["p_nonshift_turn_has_q"] = round(j[j.D_topic_shift <= .5].has_q.mean(), 3)
    res[corpus] = r
    print(json.dumps(r, indent=1, default=str))
    cv.to_csv(f"{OUT}/a5_conv_balance_{corpus}.csv", index=False)
json.dump(res, open(f"{OUT}/a5_reciprocity.json", "w"), indent=1, default=str)
