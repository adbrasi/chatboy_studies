"""b5: probe pontual. As métricas da literatura (SSI do Meena/LaMDA, falhas de antropomorfismo do CoSER,
human-likeness do CharacterBench), aplicadas pelo Jev com as definições dos papers, premiam o humano real ou a LLM?

Reaproveita os 250 contextos do a8 (resposta humana real + 3 LLMs de baseline no mesmo ponto).
Uma chamada por (contexto, resposta), com 7 perguntas atômicas. ~1.000 chamadas.
Saída: analysis/data/b5_ssi_probe.json
Uso: python3 scripts/analysis/b5_ssi_probe.py [run|analyze]
"""
import json, math, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
from jev import ask_many, noul, score, summary  # noqa: E402

SCR_A8 = "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad/a8"
SCR = "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad/b5"
OUT = os.path.join(ROOT, "analysis", "data", "b5_ssi_probe.json")
os.makedirs(SCR, exist_ok=True)
BOT, USER = "Maya", "Jordan"
CONDS = ["human", "base_gemini", "base_gpt4omini", "base_llama70b"]

# Definições tiradas dos papers (instruções de crowdworker do LaMDA, Apêndice B; rubricas do CoSER, Tab. 30;
# CharacterBench, dimensão Human-likeness). Redação atômica, uma pergunta por critério.
Q = {
    # Meena/LaMDA: sensibleness
    "sensible": noul("Does the CANDIDATE MESSAGE make sense in the context of the conversation? It makes sense if it is "
                     "reasonable in context and does not contradict or ignore anything said earlier; it does not make sense "
                     "if it is confusing, illogical, off-context or factually wrong.",
                     "Makes sense in context", "Does not make sense in context"),
    # Meena/LaMDA: specificity ("Me too" to "I love Eurovision" = not specific)
    "specific": noul("Is the CANDIDATE MESSAGE specific to the conversation? It is NOT specific if it is a generic reply that "
                     "could be used in dozens of different contexts (for example 'ok', 'haha', 'me too', 'nice', 'true').",
                     "Specific to this conversation", "Generic, could fit many contexts"),
    # LaMDA: interestingness
    "interesting": noul("Is the CANDIDATE MESSAGE interesting? It is interesting if it would likely catch someone's attention "
                        "or arouse their curiosity, or if it is insightful, unexpected or witty. If it is monotonous and "
                        "predictable, or if you are unsure, it is not interesting.",
                        "Interesting", "Not interesting"),
    # CoSER, Anthropomorphism / Self-identity
    "assistant_like": noul("Does the sender of the CANDIDATE MESSAGE behave like a 'helpful AI assistant', being overly "
                           "verbose, helpful, didactic, moralistic, submissive or easily persuaded?"),
    # CoSER, Anthropomorphism / Emotional depth
    "no_subtext": noul("Does the CANDIDATE MESSAGE directly speak out thoughts and feelings instead of leaving them implicit "
                       "(subtext), in a way a real person texting would not?"),
    # CoSER, Storyline quality / Flow
    "repeats_other": noul("Does the CANDIDATE MESSAGE repeat or restate the other person's viewpoint or information that "
                          "was already said?"),
    # CharacterBench, Believability / Human-likeness (escala de 5 pontos)
    "humanlike": score("Human-likeness: how natural is the CANDIDATE MESSAGE as a reply in this conversation?",
                       ["very unnatural, clearly machine-like", "somewhat unnatural", "acceptable",
                        "natural", "very natural, exactly like a real person"]),
}


def convo(c, max_turns=8):
    lines = []
    for h in c["history"][-max_turns:]:
        who = BOT if h["who"] == "bot" else USER
        for t in h["texts"]:
            if t.strip():
                lines.append(f"{who}: {t.strip()}")
    return "\n".join(lines)


def state(c, text):
    return (f"Casual one-to-one text chat on a messaging app between {USER} and {BOT}.\n\nCONVERSATION SO FAR:\n"
            f"{convo(c)}\n\nCANDIDATE MESSAGE (next message from {BOT}):\n{text.strip()}")


def load():
    X = json.load(open(f"{SCR_A8}/contexts.json"))
    G = {}
    for line in open(os.path.join(ROOT, "analysis", "data", "a8_generations.jsonl")):
        d = json.loads(line)
        G[d["id"]] = d
    T = {cond: {} for cond in CONDS}
    for c in X:
        g = G.get(c["id"])
        if not g:
            continue
        T["human"][c["id"]] = "\n".join(t.strip() for t in c["human"] if t.strip())
        for cond in CONDS[1:]:
            if g.get(cond):
                T[cond][c["id"]] = g[cond]
    return X, T


def run():
    X, T = load()
    items, meta = [], []
    for c in X:
        for cond in CONDS:
            txt = T[cond].get(c["id"])
            if txt:
                items.append((state(c, txt), Q)); meta.append((c["id"], cond))
    print(len(items), "calls planned")
    res = ask_many(items, workers=4)
    json.dump([{"id": i, "cond": k, "ans": r} for (i, k), r in zip(meta, res)], open(f"{SCR}/ssi_raw.json", "w"))
    print(summary())


def val(a):
    return a["noul"] if a["type"] == "noul" else a["score"]


def auc(pos, neg):
    """P(score humano > score LLM), empates = 0,5."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def analyze():
    X, T = load()
    conv_of = {c["id"]: c["conv"] for c in X}
    raw = json.load(open(f"{SCR}/ssi_raw.json"))
    V = {}  # (id, cond) -> {q: val}
    for r in raw:
        if r["ans"]:
            V[(r["id"], r["cond"])] = {q: val(r["ans"][q]) for q in Q}
    ids = sorted({i for (i, k) in V if all((i, kk) in V for kk in CONDS)})
    convs = sorted({conv_of[i] for i in ids})
    by_conv = {cv: [i for i in ids if conv_of[i] == cv] for cv in convs}
    L = {(i, k): math.log(1 + len(T[k][i])) for i in ids for k in CONDS}
    rng = np.random.default_rng(5)
    out = {"n_contexts": len(ids), "n_convs": len(convs), "questions": {q: Q[q]["instructions"] for q in Q},
           "means": {}, "paired_human_minus_llm": {}, "corr_with_length": {}}
    for q in Q:
        out["means"][q] = {k: float(np.mean([V[(i, k)][q] for i in ids])) for k in CONDS}
        # correlação com log(tamanho), dentro de cada condição e no conjunto
        allv = [V[(i, k)][q] for i in ids for k in CONDS]
        alll = [L[(i, k)] for i in ids for k in CONDS]
        out["corr_with_length"][q] = float(np.corrcoef(allv, alll)[0, 1])
        out["paired_human_minus_llm"][q] = {}
        for k in CONDS[1:]:
            d = np.array([V[(i, "human")][q] - V[(i, k)][q] for i in ids])
            win = float(np.mean(d > 0) + 0.5 * np.mean(d == 0))
            boots = []
            for _ in range(2000):
                sc = rng.choice(convs, len(convs))
                sel = [ids.index(i) for cv in sc for i in by_conv[cv]]
                boots.append(d[sel].mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out["paired_human_minus_llm"][q][k] = {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
                                                   "human_wins_pair": win}
    # Controle de tamanho: AUC humano × LLM só nos pares em que o tamanho é parecido (|log razão| <= 0,4)
    out["length_matched"] = {}
    for k in CONDS[1:]:
        sel = [i for i in ids if abs(L[(i, "human")] - L[(i, k)]) <= 0.4]
        out["length_matched"][k] = {"n": len(sel)}
        for q in Q:
            d = [V[(i, "human")][q] - V[(i, k)][q] for i in sel]
            out["length_matched"][k][q] = float(np.mean(d)) if sel else None
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out["means"], indent=1)); print(json.dumps(out["corr_with_length"], indent=1))
    print(json.dumps(out["paired_human_minus_llm"], indent=1)); print(json.dumps(out["length_matched"], indent=1))


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
