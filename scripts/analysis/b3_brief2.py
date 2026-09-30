"""b3 — Briefing v2 nos 119 pontos de teste do a9, com 4 atores (os 4 modelos permitidos).
Condições (mesma persona "You are Sam..." + prompt de estilo estático do a9, exceto A):
  A   persona + histórico, sem nada (flash-lite: a mesma geração do a9)
  B1  briefing v1 do a9 (o mesmo texto, agora dado a cada ator)
  N2  v2 SÓ CÓDIGO: mesmo formato do v2, taxas humanas GLOBAIS sorteadas (pergunta, "!", riso, emoji), tamanho-alvo
      global + persona, linha positiva de início, estilo da persona. Nenhum Jev, nenhum retrieval.
  K2  v2 CÓDIGO + CORPUS: N2 + retrieval de 20 casos humanos parecidos (outras conversas): taxas LOCAIS sorteadas,
      tamanho-alvo = mediana dos vizinhos, movimento pelo voto dos vizinhos quando confiante. Sem Jev online.
  B2  v2 COMPLETO: K2 + Jev online (leitura do momento, movimento da arquitetura escolhida no dev com portão de
      confiança, elemento concreto a que reagir, tamanho do Jev, tom quando confiante, seriedade)
  O2  ORÁCULO: B2 com o movimento e o elemento REAIS do humano (teto do "o que escrever")
Os sorteios usam os mesmos números aleatórios em N2/K2/B2/O2 (só a taxa muda).
Uso: python3 b3_brief2.py build | gen <atores> <conds>"""
import hashlib, json, os, random, re, sys
from collections import Counter
import numpy as np
from b3_common import (load_points, kv_load, kv_save, jl_load, A9P, MOVES, FAMILY, FAMILIES, BOT, USER, PERSONA,
                       lchat_many, lstats, chat_msgs, clean_reply, feats, words, ADATA, jdump)
from b3_arch import Retriever
from a9_brief import moment_of, MOMENT_LBL, TH

STATIC = (PERSONA + "\n\nText like a real person on a messaging app: casual and short, lowercase is fine, "
          "no assistant-like phrasing, match the other person's energy and style.")
BAN2 = ["aww", "totally", "absolutely", "amazing", "sounds like", "that sounds", "honestly", "vibe(s)", "super",
        "definitely", "em dashes (—)"]
MOVE2 = {
    "react_only": "Just react in a few words (like 'lol', 'fair', 'true', 'oh no'), nothing more.",
    "answer": f"Answer {USER}'s question plainly.",
    "tease_back": f"Tease {USER} back, dry and short.",
    "joke_riff": "Add one quick line to the joke.",
    "empathize": "Show you get it, in simple words. No advice.",
    "reassure": f"Reassure {USER} briefly, like a friend.",
    "share_own": "Say something of your own about it (your take or what happened to you).",
    "ask_follow_up": f"Ask one specific follow-up about what {USER} said.",
    "flirt_back": "Flirt back lightly.",
    "compliment": f"Compliment {USER} casually.",
    "agree": "Just agree, casually.",
    "disagree": "Push back, casually.",
    "plan": "Confirm or propose the plan concretely.",
    "goodbye": "Say bye, short.",
    "greet_back": "Greet back and add one concrete thing about your moment.",
    "new_topic": "Bring up something new.",
}
FAM2 = {"react": "Keep it to a quick reaction.", "respond": f"Answer {USER} plainly.",
        "contribute": "Say something of your own about it.", "ask": f"Ask one specific follow-up.",
        "play": "Keep the banter going.", "support": "Be supportive, simply.", "ritual": None}
START = "Start plainly (like 'ok', 'oh', 'yeah', 'same', 'i…') or go straight to the content."
DEF = {"q": 0.12, "excl": 0.07, "laugh": 0.07, "emoji": 0.05, "words": 5}


def u(pid, k):
    return int(hashlib.md5(f"{pid}|{k}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def knn_stats(R, p, k=20, m=10):
    nb = R.query(p, k)
    hs = [q["human_f"] for q, _ in nb]
    r = {}
    for key, fk in (("q", "has_q"), ("excl", "excl"), ("laugh", "laugh"), ("emoji", "emoji")):
        loc = np.mean([bool(h[fk]) for h in hs])
        r[key] = (k * loc + m * DEF[key]) / (k + m)
    r["words"] = float(np.median([h["n_words"] for h in hs]))
    mv = Counter(q["gold"]["g_move"] for q, _ in nb)
    r["move"], c = mv.most_common(1)[0]
    r["move_conf"] = c / k
    return r


def len_words_jev(length):
    xs, ys = TH["len_map"]
    return float(np.interp(length, xs, ys))


def build(p, cond, ctx):
    fp, pid = p["fp"], p["id"]
    L = []
    own = max(2.0, fp["median_words_per_bubble"] * max(1.0, fp["bubbles_per_turn"]))
    ks = ctx["knn"][pid]
    rates = dict(DEF) if cond == "N2" else ks
    base = ctx["base"].get(pid, {})
    j9 = p["jev"]
    # ---- momento / movimento / elemento (só com Jev, ou oráculo)
    if cond in ("B2", "O2"):
        mo = moment_of(j9)
        if mo != "casual":
            L.append(f"Moment: {MOMENT_LBL[mo]}.")
    move_line = None
    if cond == "K2" and ks["move_conf"] >= ctx["tau_knn"]:
        move_line = MOVE2[ks["move"]]
    if cond == "B2":
        v = ctx["dist"][pid]
        mi = int(np.argmax(v))
        fam = np.zeros(len(FAMILIES))
        for m_, x in zip(MOVES, v):
            fam[FAMILIES.index(FAMILY[m_])] += x
        if v[mi] >= ctx["tau_move"]:
            move_line = MOVE2[MOVES[mi]]
        elif fam.max() >= ctx["tau_fam"] and FAM2[FAMILIES[int(np.argmax(fam))]]:
            move_line = FAM2[FAMILIES[int(np.argmax(fam))]]
    if cond == "O2":
        move_line = MOVE2[p["gold"]["g_move"]]
    if move_line:
        L.append(move_line)
    el = None
    if cond == "B2":
        e = ctx["elem"].get(pid) or {}
        ep = {k: v for k, v in (e.get("elem_p") or {}).items() if k.startswith("e")}
        if ep and max(ep.values()) >= ctx["tau_elem"]:
            el = p["elements"][int(max(ep, key=ep.get)[1:]) - 1]
    if cond == "O2":
        ge = (p.get("gold_elem") or {}).get("g_elem", "")
        if ge.startswith("e"):
            el = p["elements"][int(ge[1:]) - 1]
    if el:
        L.append(f"React to the concrete thing {USER} said: '{el}'.")
    if cond in ("B2", "O2") and base.get("tone_conf", 0) >= 0.5 and cond == "B2":
        L.append(f"Tone: {base['tone'].replace('_', ' ')}.")
    # ---- tamanho: alvo com folga
    if cond == "N2":
        tgt = DEF["words"]
    elif cond == "K2":
        tgt = ks["words"]
    else:
        tgt = 0.5 * len_words_jev(j9["length"]) + 0.5 * ks["words"]
    mw = int(round(max(1.5, 0.6 * tgt + 0.4 * own)))
    nb = 2 if (fp["bubbles_per_turn"] >= 1.5 and mw >= 7) else 1
    L.append(f"About {mw} words (a bit shorter or longer is fine)" +
             (f", split into {nb} short messages, one per line." if nb > 1 else ", one message."))
    L.append(START)
    # ---- sorteios com a taxa humana do momento (mesmos números aleatórios entre condições)
    serious = cond in ("B2", "O2") and (j9["seriousness"] >= 1.8)
    q = u(pid, "q") < rates["q"]
    L.append("You can end with one short, specific question (not 'what about you?')." if q else "No question.")
    tok = fp.get("laugh_token") or "lol"
    lg = u(pid, "laugh") < rates["laugh"] and not serious
    ex = u(pid, "excl") < rates["excl"]
    em = u(pid, "emoji") < rates["emoji"] and fp["emoji_frac"] >= 0.05
    L.append((f"Laugh the way you usually do ('{tok}')." if lg else "No laughing.") + " " +
             ("One '!' is fine." if ex else "No '!'.") + " " + ("One emoji is fine." if em else "No emoji."))
    if serious:
        L.append("No jokes. Be simple and sincere.")
    st = []
    if fp["lower_frac"] > 0.7:
        st.append("all lowercase")
    if fp["period_frac"] < 0.2:
        st.append("no final period")
    if st:
        L.append("Style: " + "; ".join(st) + ".")
    L.append(f"It must make sense as a direct reply to {USER}'s last message.")
    L.append("Never use: " + ", ".join(BAN2) + ".")
    return "\n".join(L), {"q": q, "laugh": lg, "excl": ex, "emoji": em, "words": mw, "move_line": move_line, "elem": el}


def context():
    """Monta tudo o que o builder precisa, com os limiares escolhidos NO DEV."""
    from b3_eval import build as build_dists
    allp = load_points()
    res = json.load(open(os.path.join(ADATA, "b3_arch_results.json")))
    chosen = res["chosen_on_dev"]
    ev = [p for p in allp if p["eval"] and "gold" in p]
    dists = build_dists(ev, allp)
    if chosen.startswith("mix:"):
        from b3_eval import mix
        _, parts, w = chosen.split(":")
        parts, w = parts.split("+"), [float(x) for x in w.split(",")]
        for p in ev:
            D = dists[p["id"]]
            if all(c in D for c in parts):
                D[chosen] = mix(D, parts, w)
    R = Retriever(allp)
    dev = [p for p in ev if p["split"] == "dev"]

    def pick_tau(scores, target=0.5, min_cov=0.15):
        """menor limiar t com precisão >= target entre os pontos com conf >= t (e cobertura mínima)."""
        best = 1.01
        for t in np.arange(0.3, 0.95, 0.025):
            s = [ok for c, ok in scores if c >= t]
            if len(s) >= min_cov * len(scores) and np.mean(s) >= target:
                best = min(best, float(t))
        return best
    hit = lambda p, m: m == p["gold"]["g_move"] or m in (p.get("gold_luna") or {}).get("all", [])
    mv = [(float(dists[p["id"]][chosen].max()), hit(p, MOVES[int(np.argmax(dists[p["id"]][chosen]))])) for p in dev
          if chosen in dists[p["id"]]]
    fam = []
    for p in dev:
        if chosen not in dists[p["id"]]:
            continue
        v = dists[p["id"]][chosen]
        f = np.zeros(len(FAMILIES))
        for m_, x in zip(MOVES, v):
            f[FAMILIES.index(FAMILY[m_])] += x
        fam.append((float(f.max()), FAMILIES[int(np.argmax(f))] == FAMILY[p["gold"]["g_move"]]))
    kn = []
    for p in dev:
        ks = knn_stats(R, p)
        kn.append((ks["move_conf"], hit(p, ks["move"])))
    E = kv_load("pred_elem")
    el = []
    for p in dev:
        ge = (p.get("gold_elem") or {}).get("g_elem", "")
        e = E.get(p["id"]) or {}
        ep = {k: v for k, v in (e.get("elem_p") or {}).items() if k.startswith("e")}
        if ep and ge:
            el.append((max(ep.values()), max(ep, key=ep.get) == ge))
    ctx = {"chosen": chosen, "dist": {p["id"]: dists[p["id"]].get(chosen) for p in ev},
           "tau_move": pick_tau(mv), "tau_fam": pick_tau(fam, 0.6), "tau_knn": pick_tau(kn),
           "tau_elem": pick_tau(el, 0.5, 0.1), "base": kv_load("pred_base"), "elem": E}
    return ctx, R, allp


def main_build():
    ctx, R, allp = context()
    pts = [p for p in allp if p["a9"]]
    ctx["knn"] = {p["id"]: knn_stats(R, p) for p in pts}
    a9 = {p["id"]: p for p in jl_load(A9P)}
    out = {}
    for p in pts:
        p["jev"] = a9[p["id"]]["jev"]
        out[p["id"]] = {"B1": a9[p["id"]]["brief"]["B"]}
        for c in ("N2", "K2", "B2", "O2"):
            txt, meta = build(p, c, ctx)
            out[p["id"]][c] = txt
            out[p["id"]][c + "_meta"] = meta
    kv_save("briefs2", out)
    th = {k: ctx[k] for k in ("chosen", "tau_move", "tau_fam", "tau_knn", "tau_elem")}
    th["coverage_test"] = {c: {"move_line": float(np.mean([bool(out[i][c + "_meta"]["move_line"]) for i in out])),
                               "elem": float(np.mean([bool(out[i][c + "_meta"]["elem"]) for i in out])),
                               "q": float(np.mean([out[i][c + "_meta"]["q"] for i in out])),
                               "laugh": float(np.mean([out[i][c + "_meta"]["laugh"] for i in out])),
                               "excl": float(np.mean([out[i][c + "_meta"]["excl"] for i in out])),
                               "emoji": float(np.mean([out[i][c + "_meta"]["emoji"] for i in out])),
                               "words_median": float(np.median([out[i][c + "_meta"]["words"] for i in out]))}
                           for c in ("N2", "K2", "B2", "O2")}
    jdump("b3_brief2_thresholds.json", th)
    print(json.dumps(th, indent=1))
    ex = list(out)[:3]
    for i in ex:
        print("-----", i); print(out[i]["B2"])


def main_gen(actors, conds):
    B = kv_load("briefs2")
    a9 = {p["id"]: p for p in jl_load(A9P)}
    G = kv_load("gen2")
    for actor in actors:
        for cond in conds:
            ids = [i for i in B if f"{actor}|{cond}" not in G.get(i, {})]
            items = []
            for i in ids:
                p = a9[i]
                sys_ = PERSONA if cond == "A" else STATIC + "\n\nFor your next message:\n" + B[i][cond]
                items.append(dict(messages=chat_msgs(p, sys_), model=actor, temperature=0.8, max_tokens=300, seed=0))
            res = lchat_many(items, workers=4)
            for attempt in (1, 2):
                bad = [k for k, r in enumerate(res) if not (r or "").strip()]
                if not bad:
                    break
                rr = lchat_many([dict(items[k], seed=100 * attempt) for k in bad], workers=4)
                for k, r in zip(bad, rr):
                    res[k] = r
            for i, r in zip(ids, res):
                G.setdefault(i, {})[f"{actor}|{cond}"] = clean_reply(r) if (r or "").strip() else None
            kv_save("gen2", G)
            print(actor, cond, len(ids), "empty", sum(1 for r in res if not (r or "").strip()), lstats, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        main_build()
    else:
        main_gen(sys.argv[2].split(","), sys.argv[3].split(","))
