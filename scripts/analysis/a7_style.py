"""a7: estilo e vocabulário dos turnos de flerte/afeto × turnos neutros (maichat + whatsapp_nl anotados),
mais digitação (maichat) e uso de marcadores de afeto por tipo de relação no whatsapp_nl inteiro.
Saída: analysis/data/a7_style.json, analysis/data/a7_vocab.json
"""
import json, math, os, re, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_common import load, OUT

EMO = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
EMOT = re.compile(r"(?:<3+|[:;=][-']?[)(DPp*3]+|xx+\b|:\*)", re.I)
TOK = re.compile(r"<3+|[:;=][-']?[)(DPp*3/]+|[\U0001F300-\U0001FAFF\U00002600-\U000027BF]|[a-zà-ÿ']+", re.I)
ELONG = re.compile(r"([a-z])\1{2,}", re.I)
PET_EN = re.compile(r"\b(babe|baby|love|my love|hun|honey|sweetie|darling|dummy|cutie|bb|boo)\b", re.I)
PET_NL = re.compile(r"\b(schat\w*|lief\w*|lieverd|pop|poppetje|knuffel\w*|kusje\w*|dikke kus|scheet\w*|mop|snoes\w*|meisje)\b", re.I)
XX = re.compile(r"\bx{2,}\b|\bxxx*\b", re.I)


def toks(t):
    t = re.sub(r"<[^>]*weggelaten>|\[REMOVED\]|\[numerical_value\]", " ", t)
    out = []
    for w in TOK.findall(t):
        w = w.lower()
        if ELONG.search(w) and w.isalpha():
            out.append(ELONG.sub(r"\1\1", w))  # normaliza alongamento (youuuu -> youu)
            out.append("<alongamento>")
        else:
            out.append(w)
    return out


def logodds(a_counts, b_counts, prior, a0=None):
    """Monroe et al. (2008): log-odds com prior de Dirichlet informativo; retorna z-score por palavra."""
    na, nb = sum(a_counts.values()), sum(b_counts.values())
    npri = sum(prior.values())
    a0 = a0 or 500.0  # força do prior (pseudo-contagem total)
    res = {}
    for w in set(a_counts) | set(b_counts):
        aw = prior.get(w, 0) / npri * a0 + 1e-3
        ya, yb = a_counts.get(w, 0), b_counts.get(w, 0)
        la = math.log((ya + aw) / (na + a0 - ya - aw))
        lb = math.log((yb + aw) / (nb + a0 - yb - aw))
        var = 1 / (ya + aw) + 1 / (yb + aw)
        res[w] = ((la - lb) / math.sqrt(var), ya, yb)
    return res


def top_words(res, n=40, min_count=3, side=1):
    items = [(w, z, ya, yb) for w, (z, ya, yb) in res.items() if (ya if side > 0 else yb) >= min_count]
    items.sort(key=lambda x: -side * x[1])
    return [{"w": w, "z": round(z, 2), "n_flirt": ya, "n_neutral": yb} for w, z, ya, yb in items[:n]]


def feats(texts):
    t = " / ".join(texts)
    return {"chars": len(t), "emoji": len(EMO.findall(t)) > 0, "emoticon_or_xx": bool(EMOT.search(t)),
            "elong": bool(ELONG.search(t)), "q": "?" in t, "excl": "!" in t,
            "lau": bool(re.search(r"\b(a?ha(ha)+h?|he(he)+|hi(hi)+|lo+l|lmf?ao|haha\w*)\b", t, re.I)) or bool(re.search("[\U0001F602\U0001F923\U0001F606]", t)),
            "lower_start": t[:1].islower(), "ellipsis": "..." in t or "…" in t,
            "pet_name": bool(PET_EN.search(t) or PET_NL.search(t)), "xx": bool(XX.search(t))}


def main():
    df = load()
    a = df[df.annotated].copy()
    a["fa"] = (a.D_flirting >= .5) | (a.D_emotion == "affection") | (a.D_intent == "compliment_affection")
    a["neutral"] = (a.D_flirting < .15) & (a.D_emotion != "affection") & (a.D_intent != "compliment_affection")
    out = {"n": {}}
    F = pd.DataFrame([feats(x) for x in a.texts], index=a.index)
    a = a.join(F)
    style = {}
    for c in ["maichat", "whatsapp_nl"]:
        s = a[a.corpus == c]
        g1, g0 = s[s.fa], s[s.neutral]
        out["n"][c] = {"fa": int(len(g1)), "neutral": int(len(g0)), "all": int(len(s))}
        row = {}
        for k in ["chars", "n_msgs"]:
            row[k] = {"fa_med": float(g1[k].median()), "neu_med": float(g0[k].median()),
                      "p": float(mannwhitneyu(g1[k], g0[k]).pvalue)}
        for k in ["emoji", "emoticon_or_xx", "elong", "q", "excl", "lau", "lower_start", "ellipsis", "pet_name", "xx"]:
            x1, x0 = g1[k].mean(), g0[k].mean()
            tab = [[g1[k].sum(), len(g1) - g1[k].sum()], [g0[k].sum(), len(g0) - g0[k].sum()]]
            row[k] = {"fa": round(float(x1), 3), "neu": round(float(x0), 3), "p": float(fisher_exact(tab)[1])}
        # latência (só whatsapp tem resolução de minuto; maichat é ao vivo)
        lat1, lat0 = g1.response_latency_s.dropna(), g0.response_latency_s.dropna()
        row["latency_s"] = {"fa_med": float(lat1.median()), "neu_med": float(lat0.median()), "n1": len(lat1), "n0": len(lat0),
                            "p": float(mannwhitneyu(lat1, lat0).pvalue)}
        # posição na sessão
        row["pos_frac"] = {"fa_mean": float((g1.turn_in_session / g1.sess_len).mean()), "neu_mean": float((g0.turn_in_session / g0.sess_len).mean())}
        row["is_last"] = {"fa": float(g1.is_last.mean()), "neu": float(g0.is_last.mean())}
        style[c] = row
    out["style"] = style
    # digitação (maichat): o flerte é mais hesitante?
    m = a[a.corpus == "maichat"].copy()
    ty = pd.json_normalize(m["typing"]).set_index(m.index)
    m = m.join(ty)
    typing = {}
    for k in ["compose_s_total", "chars_per_s", "deletion_ratio", "max_pause_s", "abandoned_text", "n_deletion_events"]:
        x1, x0 = m[m.fa][k].dropna(), m[m.neutral][k].dropna()
        typing[k] = {"fa_med": float(x1.median()), "neu_med": float(x0.median()), "fa_mean": float(x1.mean()), "neu_mean": float(x0.mean()),
                     "p": float(mannwhitneyu(x1, x0).pvalue)}
    # por caractere: tempo de composição normalizado
    m["s_per_char"] = m.compose_s_total / m.total_chars.clip(lower=1)
    x1, x0 = m[m.fa].s_per_char.dropna(), m[m.neutral].s_per_char.dropna()
    typing["s_per_char"] = {"fa_med": float(x1.median()), "neu_med": float(x0.median()), "p": float(mannwhitneyu(x1, x0).pvalue)}
    m["any_del"] = m.n_deletion_events > 0
    typing["any_deletion"] = {"fa": float(m[m.fa].any_del.mean()), "neu": float(m[m.neutral].any_del.mean())}
    m["abandoned"] = m.abandoned_text > 0
    typing["any_abandoned"] = {"fa": float(m[m.fa].abandoned.mean()), "neu": float(m[m.neutral].abandoned.mean())}
    out["typing_maichat"] = typing
    # vocabulário
    vocab = {}
    for c in ["maichat", "whatsapp_nl"]:
        s = a[a.corpus == c]
        prior = Counter(w for x in df[df.corpus == c].texts for w in toks(" ".join(x)))
        A = Counter(w for x in s[s.fa].texts for w in toks(" ".join(x)))
        B = Counter(w for x in s[s.neutral].texts for w in toks(" ".join(x)))
        r = logodds(A, B, prior)
        vocab[c] = {"flirt": top_words(r, 40, 3, 1), "neutral": top_words(r, 40, 5, -1),
                    "n_tokens": {"flirt": sum(A.values()), "neutral": sum(B.values())}}
    # emojis por grupo (maichat+wa)
    emo = {}
    for c in ["maichat", "whatsapp_nl"]:
        s = a[a.corpus == c]
        E1 = Counter(e for x in s[s.fa].texts for e in EMO.findall(" ".join(x)))
        E0 = Counter(e for x in s[s.neutral].texts for e in EMO.findall(" ".join(x)))
        emo[c] = {"fa": E1.most_common(20), "neutral": E0.most_common(20), "n_fa": int(s.fa.sum()), "n_neu": int(s.neutral.sum())}
    out["emoji"] = emo
    # whatsapp inteiro: marcadores por tipo de relação (relação = moda do D.relationship no chat)
    w = df[df.corpus == "whatsapp_nl"].copy()
    rel = a[a.corpus == "whatsapp_nl"].groupby("conv_id").D_relationship.agg(lambda x: x.value_counts().index[0])
    w["rel"] = w.conv_id.map(rel)
    W = pd.DataFrame([feats(x) for x in w.texts], index=w.index)
    w = w.join(W)
    rel_tab = w.groupby("rel").agg(n_turns=("chars", "size"), n_chats=("conv_id", "nunique"), pet_name=("pet_name", "mean"),
                                   xx=("xx", "mean"), emoji=("emoji", "mean"), emoticon=("emoticon_or_xx", "mean"),
                                   elong=("elong", "mean"), laugh=("lau", "mean"), chars_med=("chars", "median"),
                                   n_msgs_mean=("n_msgs", "mean"))
    out["wa_by_relationship"] = rel_tab.round(3).reset_index().to_dict("records")
    json.dump(out, open(os.path.join(OUT, "a7_style.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(vocab, open(os.path.join(OUT, "a7_vocab.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])
    for c in vocab:
        print(c, "FLIRT:", " ".join(f"{d['w']}({d['n_flirt']})" for d in vocab[c]["flirt"]))
        print(c, "NEUTRAL:", " ".join(f"{d['w']}({d['n_neutral']})" for d in vocab[c]["neutral"]))


if __name__ == "__main__":
    main()
