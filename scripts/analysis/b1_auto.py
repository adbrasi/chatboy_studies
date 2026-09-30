"""b1 step 6 (A13): autoresearch feature discovery (TypeSafe cookbook pattern).
An LLM (gpt-6-luna) reads the current question set, its dev coefficients and the WORST dev errors of the code+Jev model
(out-of-fold), and proposes new atomic questions. Jev answers them for every reply (1 call per reply).
Selection is dev-only; test is scored once in b1_eval.
Usage: python3 b1_auto.py propose <round> | answer <round>
Output: scratch b1/auto_q<round>.json (questions), b1/auto_a<round>.json (answers)"""
import json, os, pickle, random, sys
import numpy as np
import b1_common as B
import b1_questions as Q
import b1_llm as L
from jev import noul, score

LEVELS = ["Not present at all", "Barely present, in passing", "Present at a moderate level", "Present strongly", "Dominant: the reply is largely about this"]

TASK = """You are designing yes/no and intensity questions for a small, fast, literal judgment model (it answers each question independently, reading a chat context and one candidate reply).
Goal: the answers become numeric features for a logistic regression that tells whether the candidate reply in a casual text chat between two people who know each other was typed by the real person or written by an AI chatbot pretending to be them, AND that tells a developer WHAT is wrong with an AI reply so it can be fixed.
Important facts:
- Length, sentence count, '?', '!', emoji, lowercase, apostrophes, stock phrases and word overlap are ALREADY measured in code. Do NOT propose questions about them.
- Abstract questions ('does it sound human/natural?') FAIL: the judge believes articulate = human. Ask about concrete, checkable properties of the content, stance, pragmatics or relation to the conversation.
- Real replies are often tangential, blunt, teasing, lazy, reference things outside the visible chat, or answer only a word of the message. AI replies are accommodating, complete, on-topic, balanced, warm, and 'helpful'.
Each question must refer to `the candidate_message` and optionally to `conversation_so_far`, be one literal judgment (no double negatives, no 'and/or' of two ideas), and be answerable from the text alone.
Return JSON: {"questions": [{"name": snake_case, "kind": "presence"|"intensity", "question": "..."}]} with exactly 15 NEW questions (not duplicates of the existing list)."""


def load_state():
    ctxs, units = B.load_units()
    bank = json.load(open(os.path.join(B.SCR, "bank.json")))
    ms = pickle.load(open(os.path.join(B.SCR, "models_scores.pkl"), "rb"))["MODELS"]
    return ctxs, units, bank, ms


def propose(rnd):
    ctxs, units, bank, ms = load_state()
    oof = ms["M4_code+jev_lr"]["scores_dev_oof"]
    by = {u["uid"]: u for u in units}
    ev = [(uid, p, by[uid]["label"]) for uid, p in oof.items()]
    fn = sorted([e for e in ev if e[2] == 1], key=lambda e: e[1])[:20]   # AI replies that looked human
    fp = sorted([e for e in ev if e[2] == 0], key=lambda e: -e[1])[:20]  # human replies that looked AI
    rng = random.Random(rnd)
    ok = rng.sample([e for e in ev if (e[2] == 1 and e[1] > 0.8) or (e[2] == 0 and e[1] < 0.2)], 10)
    def show(e):
        u = by[e[0]]; c = ctxs[u["ckey"]]
        return {"their_last_message": B.last_other(c)[:300], "candidate": u["text"][:400], "truth": "AI" if e[2] else "REAL PERSON",
                "model_p_ai": round(e[1], 2), "model_source": u["cond"]}
    existing = [{"name": k, "q": Q.BANK[k]["instructions"]} for k in Q.ATOMIC]
    prev = []
    for r in range(1, rnd):
        p = os.path.join(B.SCR, f"auto_q{r}.json")
        if os.path.exists(p):
            prev += json.load(open(p))
    msg = (TASK + "\n\nEXISTING QUESTIONS:\n" + json.dumps(existing + [{"name": x["name"], "q": x["question"]} for x in prev])[:9000] +
           "\n\nWORST MISSES (AI replies the model thought were human):\n" + json.dumps([show(e) for e in fn], ensure_ascii=False) +
           "\n\nFALSE ALARMS (real replies the model thought were AI):\n" + json.dumps([show(e) for e in fp], ensure_ascii=False) +
           "\n\nWELL CLASSIFIED (for contrast):\n" + json.dumps([show(e) for e in ok], ensure_ascii=False))
    txt = L.chat([{"role": "user", "content": msg}], model="openai/gpt-6-luna", temperature=0.7, max_tokens=4000, effort="medium", json_mode=True)
    qs = json.loads(txt)["questions"]
    json.dump(qs, open(os.path.join(B.SCR, f"auto_q{rnd}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(qs, indent=1, ensure_ascii=False)); print(L.stats)


def to_jev(qs, rnd):
    out = {}
    for q in qs:
        name = f"a{rnd}_{q['name']}"[:60]
        out[name] = score(q["question"], LEVELS) if q["kind"] == "intensity" else noul(q["question"])
    return out


def answer(rnd):
    ctxs, units, bank, _ = load_state()
    qs = to_jev(json.load(open(os.path.join(B.SCR, f"auto_q{rnd}.json"))), rnd)
    outp = os.path.join(B.SCR, f"auto_a{rnd}.json")
    done = json.load(open(outp)) if os.path.exists(outp) else {}
    todo = [u for u in units if u["uid"] in bank and u["uid"] not in done]
    for i in range(0, len(todo), 400):
        ch = todo[i:i + 400]
        res = B.run_jev(f"A13_auto_r{rnd}", [(B.base_state(ctxs[u["ckey"]], u["text"]), qs) for u in ch])
        for u, r in zip(ch, res):
            if r:
                done[u["uid"]] = {k: B.val(a) for k, a in r.items()}
        json.dump(done, open(outp, "w"))
    print("saved", len(done))


if __name__ == "__main__":
    {"propose": propose, "answer": answer}[sys.argv[1]](int(sys.argv[2]))
