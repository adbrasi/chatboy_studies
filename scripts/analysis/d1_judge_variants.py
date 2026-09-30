"""d1: judge architectures left unrun in report 10 (b1): V1 rubric-in-state, V2 speaker self-reference,
V3 labeled few-shot in state, V4 sentence-level culprit (for pruning).

Reuses b1 units, dev/test split (by conversation), code features and the existing bank (lean 9 questions).
V1-V3: does adding the variant's Jev answers to [code + lean bank] raise held-out AUC (overall / instructed)?
V4: for LLM replies with >=2 sentences, the Jev picks the sentence a friend would NOT send; we drop it and measure how much
     more human-like the pruned text becomes (code-only LR score from dev) vs rule baselines (drop last / keep first).
Outputs: analysis/data/d1_results.json
"""
import json, math, os, random, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b1_common as B  # noqa: E402
import a8_common as C8  # noqa: E402
from jev import noul, choice, score  # noqa: E402

OUT = os.path.join(B.OUT, "d1_results.json")
BANK = json.load(open(os.path.join(B.SCR, "bank.json")))
LEAN = ["t_answers_several", "t_emoji_decor", "t_blunt", "t_own_thread", "t_direct", "cf_friend_same",
        "p_enthusiasm_1", "s_friend_vs_assistant", "s_energy_vs_other"]
CM = "the candidate_message"

RUBRIC = (
    "Measured in thousands of real chats between friends vs chatbot replies at the same moments: "
    "friends' messages are short (median ~6 words, about 30 characters); ~90% have no final punctuation and most start lowercase; "
    "'!' appears in ~7% of messages; only ~20% end with a question and generic questions like 'what about you?' are rare (~1%); "
    "friends react to a concrete detail rather than naming feelings ('u did scream' not 'that's amazing!'); "
    "about half of replies are plain backchannels ('true', 'haha', 'same'); about 1 in 5 do not answer the last message at all. "
    "Chatbot replies are 2-3x longer, do several things at once (react, comment or restate, then ask a question), "
    "use '!' and emojis far more, validate and praise, and repeat the other person's words.")

Q_COMMON = {
    "v_friend_like": noul(f"Given the reference description of how friends text, {CM} looks like something a friend would send."),
    "v_bot_like": noul(f"Given the reference description, {CM} shows the typical traits of chatbot replies."),
    "v_scale": score(f"Where does {CM} fall between the two styles described in the reference?",
                     ["clearly friend style", "mostly friend style", "mixed", "mostly chatbot style", "clearly chatbot style"]),
}

SELF_Q = {
    "s_longer": noul(f"{CM} is noticeably longer than the messages in `speaker_usual_messages`."),
    "s_more_enthusiastic": noul(f"{CM} is more enthusiastic or exclamatory than the messages in `speaker_usual_messages`."),
    "s_more_formal": noul(f"{CM} is more formal, polished or grammatical than the messages in `speaker_usual_messages`."),
    "s_more_markers": noul(f"{CM} uses more emoji, laughter or slang than the messages in `speaker_usual_messages`."),
    "s_more_questions": noul(f"{CM} asks more questions than the speaker usually does in `speaker_usual_messages`."),
    "s_same_voice": noul(f"{CM} sounds like the same person who wrote `speaker_usual_messages`."),
}

FEW_Q = {
    "f_like_ai": noul(f"{CM} resembles the replies labeled 'chatbot' in `labeled_examples` more than the ones labeled 'person'."),
    "f_scale": score(f"How much does {CM} resemble the 'chatbot' examples rather than the 'person' examples in `labeled_examples`?",
                     ["like the person examples", "a bit like the person examples", "in between", "a bit like the chatbot examples", "like the chatbot examples"]),
}


def cluster_boot_auc(us, s, B_=500, seed=0):
    rng = np.random.default_rng(seed)
    convs = sorted({u["conv"] for u in us})
    idx = {c: [i for i, u in enumerate(us) if u["conv"] == c] for c in convs}
    y = np.array([u["label"] for u in us]); s = np.asarray(s)
    base = B.auc(s[y == 1], s[y == 0])
    vals = []
    for _ in range(B_):
        ii = np.concatenate([idx[c] for c in rng.choice(convs, len(convs))])
        yy, ss = y[ii], s[ii]
        if yy.min() == yy.max():
            continue
        vals.append(B.auc(ss[yy == 1], ss[yy == 0]))
    return round(float(base), 3), [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def speaker_msgs(ctx, k=6):
    return [t for w, t in ctx["history"] if w == ctx["self"]][-k:]


def main():
    random.seed(7)
    ctxs, units = B.load_units()
    for u in units:
        u["conv"] = ctxs[u["ckey"]]["conv"]; u["split"] = ctxs[u["ckey"]]["split"]
        u["instructed"] = u["cond"] in B.INSTRUCTED
    units = [u for u in units if u["uid"] in BANK]
    # sample: all humans + up to 2 LLM per context (1 instructed if present) per split, capped
    by = {}
    for u in units:
        by.setdefault(u["ckey"], []).append(u)
    sample = []
    for ck, us in by.items():
        hum = [u for u in us if u["label"] == 0]
        llm = [u for u in us if u["label"] == 1]
        ins = [u for u in llm if u["instructed"]]; base = [u for u in llm if not u["instructed"]]
        pick = hum + random.sample(base, min(1, len(base))) + random.sample(ins, min(1, len(ins)))
        sample += pick
    dev = [u for u in sample if u["split"] == "dev"]; test = [u for u in sample if u["split"] == "test"]
    print("sample dev", len(dev), "test", len(test))

    # few-shot pool: dev contexts only, never used for test items of the same conversation
    pool_h = [u for u in dev if u["label"] == 0 and 4 <= len(u["text"]) <= 80]
    pool_l = [u for u in dev if u["label"] == 1 and len(u["text"]) <= 220]

    def fewshot(u):
        rnd = random.Random(u["uid"])
        ex = []
        for p in rnd.sample([x for x in pool_h if x["conv"] != u["conv"]], 3):
            ex.append({"label": "person", "reply": p["text"]})
        for p in rnd.sample([x for x in pool_l if x["conv"] != u["conv"]], 3):
            ex.append({"label": "chatbot", "reply": p["text"]})
        rnd.shuffle(ex)
        return ex

    items, keys = [], []
    for u in sample:
        ctx = ctxs[u["ckey"]]
        items.append((B.base_state(ctx, u["text"], {"reference_how_people_text": RUBRIC, "_first": True}), Q_COMMON)); keys.append(("V1", u["uid"]))
        sm = speaker_msgs(ctx)
        if len(sm) >= 3:
            items.append((B.base_state(ctx, u["text"], {"speaker_usual_messages": sm, "_first": True}), SELF_Q)); keys.append(("V2", u["uid"]))
        items.append((B.base_state(ctx, u["text"], {"labeled_examples": fewshot(u), "_first": True}), FEW_Q)); keys.append(("V3", u["uid"]))
    print("V1-3 calls", len(items))
    res = B.run_jev("d1_V123", items)
    feats = {}
    for (v, uid), r in zip(keys, res):
        if r:
            feats.setdefault(uid, {}).update({q: B.val(a) for q, a in r.items()})

    # ---------- learned evaluation
    def X_of(us, cols_jev, extra):
        rows = []
        for u in us:
            other = B.last_other(ctxs[u["ckey"]])
            cf = B.code_feats(u["text"], other)
            r = [cf[c] for c in B.CODE_COLS] + [BANK[u["uid"]].get(q, np.nan) for q in cols_jev]
            r += [feats.get(u["uid"], {}).get(q, np.nan) for q in extra]
            rows.append(r)
        X = np.array(rows, float)
        return X

    results = {"n": {"dev": len(dev), "test": len(test)}}
    configs = {
        "code": ([], []),
        "code+lean": (LEAN, []),
        "code+lean+V1_rubric": (LEAN, list(Q_COMMON)),
        "code+lean+V2_self": (LEAN, list(SELF_Q)),
        "code+lean+V3_fewshot": (LEAN, list(FEW_Q)),
        "code+lean+V1+V2+V3": (LEAN, list(Q_COMMON) + list(SELF_Q) + list(FEW_Q)),
    }
    for name, (cj, ex) in configs.items():
        Xd, Xt = X_of(dev, cj, ex), X_of(test, cj, ex)
        med = np.nanmedian(Xd, axis=0); med = np.where(np.isnan(med), 0, med)
        Xd = np.where(np.isnan(Xd), med, Xd); Xt = np.where(np.isnan(Xt), med, Xt)
        yd = np.array([u["label"] for u in dev])
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(Xd, yd)
        st = m.predict_proba(Xt)[:, 1]
        out = {"all": cluster_boot_auc(test, st)}
        for sub, cond in [("instructed", lambda u: u["label"] == 0 or u["instructed"]), ("base", lambda u: u["label"] == 0 or not u["instructed"])]:
            ii = [i for i, u in enumerate(test) if cond(u)]
            out[sub] = cluster_boot_auc([test[i] for i in ii], st[ii])
        results[name] = out
        print(name, out)
    # standalone single-question AUCs (test)
    single = {}
    for q in list(Q_COMMON) + list(SELF_Q) + list(FEW_Q):
        us = [u for u in test if q in feats.get(u["uid"], {})]
        s = [feats[u["uid"]][q] for u in us]
        sign = -1 if q in ("v_friend_like", "s_same_voice") else 1
        if len(us) > 50:
            single[q] = {"auc": cluster_boot_auc(us, [sign * x for x in s])[0], "n": len(us)}
    results["single_question_auc_test"] = single
    json.dump(results, open(OUT, "w"), indent=1)

    # ---------- V4 sentence culprit
    code_dev = X_of(dev, [], []); yd = np.array([u["label"] for u in dev])
    cm = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(code_dev, yd)

    def code_llm_score(text, other):
        cf = B.code_feats(text, other)
        return float(cm.predict_proba(np.array([[cf[c] for c in B.CODE_COLS]]))[0, 1])

    cands = [u for u in test if u["label"] == 1 and 2 <= len(C8.sentences(u["text"])) <= 6]
    cands = cands[:260]
    items = []
    for u in cands:
        sents = C8.sentences(u["text"])
        opts = {str(i + 1): s for i, s in enumerate(sents)}
        st = B.base_state(ctxs[u["ckey"]], u["text"], {"candidate_sentences": opts})
        qs = {"culprit": choice("If the sender could keep only what a friend texting quickly would really send, which ONE sentence of `candidate_sentences` should be deleted first?",
                                {k: None for k in opts}),
              "keep": choice("Which ONE sentence of `candidate_sentences` carries the real reply and should be kept if only one could be sent?",
                             {k: None for k in opts})}
        for k in opts:
            qs[f"drop_{k}"] = noul(f"Sentence {k} of `candidate_sentences` is filler a friend texting quickly would not write (a generic question, a restatement, validation, or extra enthusiasm).")
        items.append((st, qs))
    res = B.run_jev("d1_V4", items)
    rows = []
    for u, r in zip(cands, res):
        if not r:
            continue
        other = B.last_other(ctxs[u["ckey"]])
        sents = C8.sentences(u["text"])
        cul = int(r["culprit"]["choice"]) - 1
        keep = int(r["keep"]["choice"]) - 1
        drops = [i for i in range(len(sents)) if r.get(f"drop_{i+1}", {}).get("noul", 0) > 0.5]
        variants = {
            "orig": u["text"],
            "jev_drop_culprit": " ".join(s for i, s in enumerate(sents) if i != cul),
            "jev_keep_one": sents[keep],
            "jev_drop_all_filler": " ".join(s for i, s in enumerate(sents) if i not in drops) or sents[keep],
            "rule_drop_last": " ".join(sents[:-1]),
            "rule_keep_first": sents[0],
            "rule_drop_questions": " ".join(s for s in sents if not s.rstrip().endswith("?")) or sents[0],
        }
        rows.append({k: code_llm_score(v, other) for k, v in variants.items()} | {"_texts": variants, "uid": u["uid"]})
    summ = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0] if not k.startswith("_") and k != "uid"}
    # coherence check of the pruned variants (does it still make sense as a reply?)
    coh_items, coh_keys = [], []
    for r in rows:
        u = next(x for x in cands if x["uid"] == r["uid"])
        for k in ("jev_drop_culprit", "jev_keep_one", "rule_keep_first", "rule_drop_last", "rule_drop_questions"):
            coh_items.append((B.base_state(ctxs[u["ckey"]], r["_texts"][k]),
                              {"coh": noul(f"{CM} makes sense as a reply to the other person's last message."),
                               "full": noul(f"{CM} still answers what the other person actually said or asked.")}))
            coh_keys.append(k)
    cres = B.run_jev("d1_V4_coh", coh_items)
    coh = {}
    for k, rr in zip(coh_keys, cres):
        if rr:
            coh.setdefault(k, {"coh": [], "full": []})
            coh[k]["coh"].append(rr["coh"]["noul"]); coh[k]["full"].append(rr["full"]["noul"])
    results["V4"] = {"n": len(rows), "mean_code_llm_score": summ,
                     "coherence": {k: {m: round(float(np.mean(v[m])), 3) for m in v} for k, v in coh.items()},
                     "examples": [{"orig": r["_texts"]["orig"], "jev_drop_culprit": r["_texts"]["jev_drop_culprit"],
                                   "jev_keep_one": r["_texts"]["jev_keep_one"], "rule_keep_first": r["_texts"]["rule_keep_first"]} for r in rows[:12]]}
    human_ref = [code_llm_score(u["text"], B.last_other(ctxs[u["ckey"]])) for u in test if u["label"] == 0]
    results["V4"]["human_mean_code_llm_score"] = round(float(np.mean(human_ref)), 3)
    json.dump(results, open(OUT, "w"), indent=1)
    print(json.dumps(results["V4"]["mean_code_llm_score"]), results["V4"]["coherence"], results["V4"]["human_mean_code_llm_score"])


if __name__ == "__main__":
    main()
