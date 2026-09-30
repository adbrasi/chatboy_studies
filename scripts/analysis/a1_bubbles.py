"""a1: how people split a turn into bubbles. Heuristic bubble-function tagging on all bursts of 2-4 +
Jev classification of bubble function on a sample (maichat: all 2-4 bursts; whatsapp_nl: 350 random)."""
import json, os, re, sys, random
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a1_load import enriched
from jev import ask_many, choice, summary

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "data")
T, M = enriched()
MF = M.set_index(["conv_id", "idx"])

LAUGH = re.compile(r"^\W*((ha|he|hi|ah){2,}h?|lo+l|lmao+|lmfao|rofl|xd+|kk+|hah+a*|jaja+|😂+|🤣+|😆+|:'?\)+|:d+|xd)\W*$", re.I)
REACT = re.compile(r"^\W*(ok(ay|e|ee|ie)?|oke+|k+|yes+|yeah+|yea|yep|ya+|ja+h?|jaa+|jep|nee+|no+|nope|nah|oh+|ohh+|ooh+|aw+h?|aa+h+|ah+|wow+|omg|oo+h|hmm+|mm+|nice|cool|true|same|right|fair|damn|echt|wauw|haha+ ?ja|top|super|leuk|lekker|prima|goed|mooi|jeetje|serieus|wtf|lol ok|sure|thanks|thank you|thx|dank je|dankje|bedankt|sorry|sws|yay|jup|klopt|inderdaad|precies|idd)\W*[!?.]*\W*$", re.I)
EMO_ONLY = re.compile(r"^[\W\s-]+$")


def heur(text, f):
    t = text.strip()
    if f["f_media"] or t.startswith("<") and "weggelaten" in t:
        return "media"
    if f["f_self_correction"] or (len(t) < 30 and (t.startswith("*") or t.endswith("*"))):
        return "correction"
    if f["f_emoji_only"] or (t and EMO_ONLY.match(t) and not any(ch.isalnum() for ch in t)):
        return "emoji"
    if LAUGH.match(t):
        return "laugh"
    if "?" in t:
        return "question"
    if REACT.match(t) or (len(t.split()) <= 2 and len(t) <= 12):
        return "reaction"
    return "content"


rows = []
B = T[T.n_msgs.between(2, 4)]
for _, t in B.iterrows():
    labs = []
    for pos, i in enumerate(t.msg_idxs):
        f = MF.loc[(t.conv_id, i)]
        labs.append(heur(f.text, f))
    rows.append({"corpus": t.corpus, "conv_id": t.conv_id, "turn_idx": t.turn_idx, "n": t.n_msgs, "labs": labs,
                 "chars": t.bubble_chars, "texts": list(t.texts), "gaps": t.gaps})
H = pd.DataFrame(rows)
res = {"heuristic": {}}
for c, g in H.groupby("corpus"):
    allb = Counter(l for ls in g.labs for l in ls)
    pos_first = Counter(ls[0] for ls in g.labs)
    pos_last = Counter(ls[-1] for ls in g.labs)
    pairs = Counter(f"{a}>{b}" for ls in g.labs for a, b in zip(ls, ls[1:]))
    seq = Counter("+".join(ls) for ls in g.labs)
    # question position: among bursts with exactly one question bubble (not in position 0 of 1)
    qb = [ls for ls in g.labs if ls.count("question") == 1]
    q_last = np.mean([ls[-1] == "question" for ls in qb])
    q_chance = np.mean([1 / len(ls) for ls in qb])
    rb = [ls for ls in g.labs if ls.count("reaction") == 1]
    r_first = np.mean([ls[0] == "reaction" for ls in rb])
    r_chance = np.mean([1 / len(ls) for ls in rb])
    lb = [ls for ls in g.labs if ls.count("laugh") + ls.count("emoji") == 1]
    le_last = np.mean([ls[-1] in ("laugh", "emoji") for ls in lb])
    le_first = np.mean([ls[0] in ("laugh", "emoji") for ls in lb])
    # bubble length by position in 2- and 3-bursts
    lenpos = {n: [float(np.median([ch[k] for ch in g[g.n == n].chars])) for k in range(n)] for n in (2, 3, 4)}
    tot = sum(allb.values())
    res["heuristic"][c] = {
        "n_bursts": len(g), "n_bubbles": tot,
        "share": {k: round(v / tot, 3) for k, v in allb.most_common()},
        "first": {k: round(v / len(g), 3) for k, v in pos_first.most_common()},
        "last": {k: round(v / len(g), 3) for k, v in pos_last.most_common()},
        "top_pairs": {k: round(v / sum(pairs.values()), 3) for k, v in pairs.most_common(12)},
        "top_sequences": {k: v for k, v in seq.most_common(12)},
        "question_single: P(last)": round(q_last, 3), "question_single: chance": round(q_chance, 3), "n_q_bursts": len(qb),
        "reaction_single: P(first)": round(r_first, 3), "reaction_single: chance": round(r_chance, 3), "n_r_bursts": len(rb),
        "laugh/emoji_single: P(last)": round(le_last, 3), "P(first)": round(le_first, 3), "n_le": len(lb),
        "median_chars_by_position": lenpos,
    }
print(json.dumps(res["heuristic"], indent=1, ensure_ascii=False))

# examples of patterns
ex = {}
for c, g in H.groupby("corpus"):
    for pat in ["reaction+content", "content+question", "content+emoji", "content+laugh", "content+correction", "laugh+content", "content+content"]:
        s = g[g.labs.apply("+".join) == pat]
        ex[f"{c}:{pat}"] = [x for x in s.sample(min(4, len(s)), random_state=3).texts.tolist()] if len(s) else []
res["examples"] = ex

# ---------- Jev classification of bubble function ----------
FUNCS = {
    "reaction": "a short reaction to what the other person said (ok, yes, no, wow, oh, aww, nice, true)",
    "laugh": "laughter only (haha, lol, lmao, hahaha)",
    "emoji_only": "only emoji(s) or emoticon(s), no words",
    "main_content": "the main statement, answer or piece of information of the turn",
    "question": "asks the other person something",
    "add_on": "continues or elaborates the previous bubble (a second thought, extra detail, the rest of the sentence)",
    "afterthought_new_point": "a new, separate point added after the main content (by the way, also, oh and...)",
    "correction": "fixes a typo or corrects something in a previous bubble (e.g. 'it*')",
    "softener_hedge": "softens, hedges or reframes what was just said (jk, lol just kidding, but anyway, I mean)",
    "greeting_closing": "hello / goodbye / goodnight",
}
rnd = random.Random(11)
mc = H[H.corpus == "maichat"]
wa = H[H.corpus == "whatsapp_nl"].sample(350, random_state=11)
S = pd.concat([mc, wa])
Tix = T.set_index(["conv_id", "turn_idx"])
items = []
for _, r in S.iterrows():
    prev = Tix.loc[(r.conv_id, r.turn_idx - 1)] if (r.conv_id, r.turn_idx - 1) in Tix.index else None
    prev_txt = " / ".join(prev.texts)[:400] if prev is not None else "(none)"
    state = {"previous_message_from_other_person": prev_txt,
             "current_burst": [{"bubble": k + 1, "text": x.strip()[:300]} for k, x in enumerate(r.texts)],
             "note": "The same person sent the bubbles of current_burst one after another, as separate chat messages."}
    qs = {f"b{k + 1}": choice(f"What is the function of bubble {k + 1} within current_burst?", FUNCS) for k in range(len(r.texts))}
    items.append((state, qs))
print("jev calls", len(items), flush=True)
ans = ask_many(items, workers=4)
print(summary())
S = S.assign(jev=[None if a is None else [a[f"b{k + 1}"]["choice"] for k in range(len(a))] for a in ans],
             jev_conf=[None if a is None else [a[f"b{k + 1}"]["confidence"] for k in range(len(a))] for a in ans])
S = S[S.jev.notna()]
S.drop(columns=["texts"]).to_json(os.path.join(OUT, "a1_bubble_functions.jsonl"), orient="records", lines=True, force_ascii=False)
jr = {}
for c, g in S.groupby("corpus"):
    allb = Counter(l for ls in g.jev for l in ls)
    tot = sum(allb.values())
    first = Counter(ls[0] for ls in g.jev)
    last = Counter(ls[-1] for ls in g.jev)
    seq = Counter("+".join(ls) for ls in g.jev)
    pairs = Counter(f"{a}>{b}" for ls in g.jev for a, b in zip(ls, ls[1:]))
    # agreement heuristic vs jev on mapped labels
    mp = {"reaction": "reaction", "laugh": "laugh", "emoji_only": "emoji", "question": "question", "correction": "correction"}
    agree = [(h, j) for hs, js in zip(g.labs, g.jev) for h, j in zip(hs, js)]
    ag = np.mean([mp.get(j, "content") == (h if h != "media" else "content") for h, j in agree])
    # gap before each bubble function (maichat only, seconds)
    gapf = {}
    if c == "maichat":
        acc = {}
        for js, gs in zip(g.jev, g.gaps):
            if len(gs) == len(js) - 1:
                for j, gp in zip(js[1:], gs):
                    acc.setdefault(j, []).append(gp)
        gapf = {k: {"n": len(v), "median_gap_s": round(float(np.median(v)), 1), "p25": round(float(np.percentile(v, 25)), 1),
                    "p75": round(float(np.percentile(v, 75)), 1)} for k, v in acc.items() if len(v) >= 10}
    jr[c] = {"n_bursts": len(g), "n_bubbles": tot, "share": {k: round(v / tot, 3) for k, v in allb.most_common()},
             "first": {k: round(v / len(g), 3) for k, v in first.most_common()},
             "last": {k: round(v / len(g), 3) for k, v in last.most_common()},
             "top_pairs": {k: round(v / sum(pairs.values()), 3) for k, v in pairs.most_common(12)},
             "top_sequences": dict(seq.most_common(15)), "agreement_with_heuristic": round(float(ag), 3),
             "median_conf": round(float(np.median([x for xs in g.jev_conf for x in xs])), 2),
             "gap_before_function": gapf}
res["jev"] = jr
print(json.dumps(jr, indent=1, ensure_ascii=False))
json.dump(res, open(os.path.join(OUT, "a1_bubbles.json"), "w"), indent=1, ensure_ascii=False)
