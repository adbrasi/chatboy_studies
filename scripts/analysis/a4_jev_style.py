"""a4 §5 — Jev as a naturalness / style-identity filter (maichat, English).

For N target turns T of speaker X (with the conversation context before T):
  candidates  real      : what X actually sent (bubbles joined by newline)
              formal    : real, 'cleaned' by code (caps, apostrophes, expand slang, drop laughs/emoji/elongation,
                          final punctuation, one bubble) = same content, LLM-ish surface
              llm_plain : gemini-3.5-flash-lite replying as X from context only
              llm_styled: same LLM, also given X's style samples + instruction to imitate them
              partner   : the partner's reply to the same context (other human, same topic) — identity test
Jev calls
  pairwise Choice (both orders): real vs each other candidate — "which was typed by X?"
  Noul per candidate (with / without X's samples): is_bot, same_person
Code baseline: nearest style profile (deterministic features) for real vs partner.
Output: analysis/data/a4_jev_style.json + a4_jev_style_cases.csv (small)"""
import json, os, random, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from jev import ask_many, choice, noul, summary
import llm
from a4_common import OUT, ADATA
from a4_style_feats import feats

N = int(os.environ.get("A4_N", 60))
rng = random.Random(4)

EXP = {"u": "you", "ur": "your", "im": "I'm", "dont": "don't", "cant": "can't", "idk": "I don't know",
       "tbh": "to be honest", "btw": "by the way", "rn": "right now", "bc": "because", "cuz": "because",
       "cause": "because", "gonna": "going to", "wanna": "want to", "kinda": "kind of", "ya": "you",
       "thats": "that's", "didnt": "didn't", "doesnt": "doesn't", "wasnt": "wasn't", "isnt": "isn't",
       "ive": "I've", "omg": "oh my goodness", "pls": "please", "plz": "please", "tho": "though",
       "whats": "what's", "youre": "you're", "i": "I", "abt": "about", "yh": "yes", "ngl": "honestly",
       "fr": "really", "lowkey": "a little", "defo": "definitely", "wbu": "what about you", "rly": "really",
       "nah": "no", "yep": "yes", "yup": "yes", "ok": "okay", "k": "okay", "convo": "conversation"}
LAUGH = re.compile(r"\b(?:a?ha(?:ha)+h?a*|hah+|he(?:he)+|lo+l+|lmf?ao+|rofl|xd+)\b", re.I)
EMO = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\ufe0f\u200d]|:[a-z_]{3,}:")


def formalize(bubbles):
    out = []
    for b in bubbles:
        t = EMO.sub("", b)
        t = LAUGH.sub("", t)
        t = re.sub(r"([a-zA-Z])\1{2,}", r"\1", t)
        t = re.sub(r"([!?.])\1+", r"\1", t)
        t = re.sub(r"[A-Za-z']+", lambda m: EXP.get(m.group(0).lower(), m.group(0)) if m.group(0).lower() in EXP else m.group(0), t)
        t = re.sub(r"\s+", " ", t).strip(" ,")
        if not t:
            continue
        t = t[0].upper() + t[1:]
        if t[-1] not in ".!?":
            t += "."
        out.append(t)
    if not out:
        out = ["That's so funny!"]
    return " ".join(out)


def build_cases():
    T = pd.read_pickle(f"{OUT}/a4_turns.pkl")
    T = T[T.corpus == "maichat"].sort_values(["conv_id", "turn_idx"]).reset_index(drop=True)
    cases = []
    for cv, g in T.groupby("conv_id"):
        g = g.reset_index(drop=True)
        for i in range(6, len(g) - 1):
            t, prev, nxt = g.iloc[i], g.iloc[i - 1], g.iloc[i + 1]
            if prev.speaker == t.speaker or nxt.speaker == t.speaker:
                continue
            real = "\n".join(t.texts).strip()
            if not (8 <= len(real) <= 220) or not (4 <= len("\n".join(nxt.texts)) <= 220):
                continue
            own = [(j, x) for j, r in g.iterrows() if r.speaker == t.speaker and abs(j - i) > 3 for x in r.texts]
            if len(own) < 10:
                continue
            cases.append((cv, i, t.speaker))
    rng.shuffle(cases)
    # at most 2 cases per conversation, spread
    seen, sel = {}, []
    for c in cases:
        if seen.get(c[0], 0) < 2:
            sel.append(c); seen[c[0]] = seen.get(c[0], 0) + 1
        if len(sel) >= N:
            break
    out = []
    for cv, i, spk in sel:
        g = T[T.conv_id == cv].reset_index(drop=True)
        t, nxt = g.iloc[i], g.iloc[i + 1]
        lab = lambda s: "X" if s == spk else "Y"
        ctx = [{"speaker": lab(r.speaker), "text": "\n".join(r.texts)} for _, r in g.iloc[max(0, i - 8):i].iterrows()]
        own = [x for j, r in g.iterrows() if r.speaker == spk and abs(j - i) > 3 for x in r.texts if len(x.strip()) >= 2]
        r2 = random.Random(hash((cv, i)) % 10**6)
        samples = r2.sample(own, min(10, len(own)))
        # keep chronological-ish order of samples
        samples = [x for x in own if x in samples][:10]
        out.append({"conv_id": cv, "turn_i": i, "speaker": spk, "context": ctx, "samples": samples,
                    "real": "\n".join(t.texts).strip(), "formal": formalize(t.texts),
                    "partner": "\n".join(nxt.texts).strip()})
    return out


SYS_PLAIN = ("You are X, texting with your friend Y in a casual chat app. Write X's next message in reply to the "
             "conversation. Output only the message text.")
SYS_STYLED = ("You are X, texting with your friend Y in a casual chat app. Write X's next message in reply to the "
              "conversation. Imitate X's texting style exactly as in X's earlier messages below: capitalization, "
              "punctuation (or lack of it), slang/abbreviations, laughter, emoji, message length, and splitting a "
              "thought into several short bubbles (one bubble per line). Output only the message text.\n\n"
              "X's earlier messages:\n")


def llm_items(c):
    conv = "\n".join(f"{m['speaker']}: {m['text']}" for m in c["context"])
    user = f"Conversation so far:\n{conv}\n\nX:"
    return [{"messages": [{"role": "system", "content": SYS_PLAIN}, {"role": "user", "content": user}],
             "temperature": 0.8, "max_tokens": 150, "seed": 0},
            {"messages": [{"role": "system", "content": SYS_STYLED + "\n".join(c["samples"])},
                          {"role": "user", "content": user}], "temperature": 0.8, "max_tokens": 150, "seed": 0}]


def clean_llm(s):
    s = (s or "").strip()
    s = re.sub(r"^X:\s*", "", s)
    s = re.sub(r"\nX:\s*", "\n", s)
    return s.strip().strip('"')


Q_PAIR = choice("Which candidate reply was most likely typed by X themself? Compare with X's own earlier messages "
                "(style_samples_of_X): capitalization, punctuation, slang, laughter, emoji, length, splitting into "
                "lines.", {"A": "candidate A was typed by X", "B": "candidate B was typed by X"})
Q_BOT = noul("X_reply was written by an AI chatbot / language model, not typed by a real person texting a friend.")
Q_SAME = noul("X_reply has the same texting style as X's earlier messages in style_samples_of_X (it looks like the same person typed it).")
Q_BOT_NOSAMPLE = noul("The last message (by X) was written by an AI chatbot / language model, not typed by a real person texting a friend.")


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean()))


STYLE_KEYS = ["starts_lower", "all_lower", "laugh", "emo_any", "elong", "abbr", "no_apos", "lower_i", "multi_punct"]


def prof(texts):
    fs = [feats(x, "en") for x in texts]
    v = [np.mean([f[k] for f in fs]) for k in STYLE_KEYS]
    v.append(np.mean([f["end"] in ("period", "excl", "question") for f in fs]))
    v.append(np.mean([np.log1p(f["n_chars"]) for f in fs]) / 5)
    return np.array(v, float)


def main():
    cases = build_cases()
    print("cases", len(cases))
    items = [it for c in cases for it in llm_items(c)]
    outs = llm.chat_many(items, workers=4)
    for k, c in enumerate(cases):
        c["llm_plain"] = clean_llm(outs[2 * k]); c["llm_styled"] = clean_llm(outs[2 * k + 1])
    print("llm", llm.stats)
    CANDS = ["real", "formal", "llm_plain", "llm_styled", "partner"]
    jobs, keys = [], []
    for k, c in enumerate(cases):
        base = {"style_samples_of_X": c["samples"], "conversation_so_far": c["context"]}
        for other in CANDS[1:]:
            for order in (0, 1):
                a, b = (c["real"], c[other]) if order == 0 else (c[other], c["real"])
                st = dict(base, candidate_replies={"A": a, "B": b})
                jobs.append((json.dumps(st, ensure_ascii=False), {"which": Q_PAIR})); keys.append(("pair", k, other, order))
        for cand in CANDS:
            st = dict(base, X_reply=c[cand])
            jobs.append((json.dumps(st, ensure_ascii=False), {"is_bot": Q_BOT, "same_person": Q_SAME})); keys.append(("noul", k, cand, None))
            if cand != "partner":
                st2 = {"conversation": c["context"] + [{"speaker": "X", "text": c[cand]}]}
                jobs.append((json.dumps(st2, ensure_ascii=False), {"is_bot": Q_BOT_NOSAMPLE})); keys.append(("noul_ns", k, cand, None))
    print("jev jobs", len(jobs))
    ans = ask_many(jobs, workers=4)
    print("jev", summary())
    rows = []
    for key, a in zip(keys, ans):
        if a is None:
            continue
        kind, k, cand, order = key
        if kind == "pair":
            pr = a["which"]["probabilities"]
            p_real = pr.get("A", 0) if order == 0 else pr.get("B", 0)
            rows.append({"kind": kind, "case": k, "cand": cand, "order": order, "p_real": p_real,
                         "conf": a["which"].get("confidence")})
        else:
            rows.append({"kind": kind, "case": k, "cand": cand, "is_bot": a["is_bot"]["noul"],
                         "same_person": a.get("same_person", {}).get("noul")})
    R = pd.DataFrame(rows)
    res = {"n_cases": len(cases), "jev": summary(), "llm": dict(llm.stats), "pair": {}, "noul": {}, "noul_nosamples": {}}
    pr = R[R.kind == "pair"]
    for cand, g in pr.groupby("cand"):
        m = g.groupby("case").p_real.mean()
        o0 = g[g.order == 0].p_real; o1 = g[g.order == 1].p_real
        res["pair"][cand] = {"n": int(len(m)), "acc_mean_both_orders": round(float((m > .5).mean()), 3),
                             "mean_p_real": round(float(m.mean()), 3),
                             "acc_orderA": round(float((o0 > .5).mean()), 3), "acc_orderB": round(float((o1 > .5).mean()), 3),
                             "consistent_both_orders": round(float(((g.pivot(index='case', columns='order', values='p_real') > .5).all(1)).mean()), 3)}
    nl = R[R.kind == "noul"]
    for q in ["is_bot", "same_person"]:
        real = nl[nl.cand == "real"].set_index("case")[q]
        res["noul"][q] = {}
        for cand, g in nl.groupby("cand"):
            s = g.set_index("case")[q]
            d = {"mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3),
                 "share_gt_.5": round(float((s > .5).mean()), 3)}
            if cand != "real":
                d["auc_vs_real"] = round(auc(s.values, real.values) if q == "is_bot" else auc(real.values, s.values), 3)
                idx = s.index.intersection(real.index)
                d["paired_win"] = round(float((s[idx] > real[idx]).mean() if q == "is_bot" else (real[idx] > s[idx]).mean()), 3)
            res["noul"][q][cand] = d
    ns = R[R.kind == "noul_ns"]
    real = ns[ns.cand == "real"].set_index("case").is_bot
    for cand, g in ns.groupby("cand"):
        s = g.set_index("case").is_bot
        d = {"mean": round(float(s.mean()), 3), "share_gt_.5": round(float((s > .5).mean()), 3)}
        if cand != "real":
            d["auc_vs_real"] = round(auc(s.values, real.values), 3)
        res["noul_nosamples"][cand] = d
    # code baseline: which of (real, other) is closer to X's style profile
    base = {}
    for other in ["formal", "llm_plain", "llm_styled", "partner"]:
        wins = []
        for c in cases:
            p = prof(c["samples"])
            dr = np.abs(prof(c["real"].split("\n")) - p).sum(); do = np.abs(prof(c[other].split("\n")) - p).sum()
            wins.append(1.0 if dr < do else .5 if dr == do else 0.0)
        base[other] = round(float(np.mean(wins)), 3)
    res["code_baseline_acc"] = base
    # surface stats of candidates
    surf = {}
    for cand in CANDS:
        fs = [feats(x, "en") for c in cases for x in c[cand].split("\n") if x.strip()]
        surf[cand] = {"bubbles_per_reply": round(float(np.mean([len([x for x in c[cand].split("\n") if x.strip()]) for c in cases])), 2),
                      "chars_per_reply": round(float(np.mean([len(c[cand]) for c in cases])), 1),
                      "starts_lower": round(float(np.mean([f["starts_lower"] for f in fs])), 3),
                      "final_punct": round(float(np.mean([f["end"] in ("period", "excl", "question") for f in fs])), 3),
                      "laugh": round(float(np.mean([f["laugh"] for f in fs])), 3),
                      "emoji": round(float(np.mean([f["emo_any"] for f in fs])), 3),
                      "abbr": round(float(np.mean([f["abbr"] for f in fs])), 3)}
    res["surface"] = surf
    json.dump(res, open(f"{ADATA}/a4_jev_style.json", "w"), indent=1)
    pd.DataFrame([{k: c[k] for k in ["conv_id", "turn_i", "real", "formal", "llm_plain", "llm_styled", "partner"]} for c in cases]).to_csv(f"{ADATA}/a4_jev_style_cases.csv", index=False)
    R.to_csv(f"{ADATA}/a4_jev_style_answers.csv", index=False)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
