"""b1 step 2 (A1 + A8 + A9 + A10 + A11-stage-2): one Jev call per candidate reply with the whole question bank
(fan-out: 53 atomic traits + 7 counterfactuals + 18 paraphrase variants + 8 anchored/holistic + 1 main-vice Choice).
Output: scratch b1/bank.json {uid: {qid: value, 'v_main_p': {...}}}"""
import json, os, sys
import b1_common as B
import b1_questions as Q

OUTP = os.path.join(B.SCR, "bank.json")


def compact(ans):
    d = {}
    for q, a in ans.items():
        d[q] = B.val(a)
        if a["type"] == "choice":
            d[q + "_p"] = a["probabilities"]; d[q + "_conf"] = a.get("confidence")
        elif a["type"] == "score":
            d[q + "_conf"] = a.get("confidence")
    return d


if __name__ == "__main__":
    ctxs, units = B.load_units()
    done = json.load(open(OUTP)) if os.path.exists(OUTP) else {}
    todo = [u for u in units if u["uid"] not in done]
    print("units", len(units), "todo", len(todo))
    for i in range(0, len(todo), 400):
        chunk = todo[i:i + 400]
        res = B.run_jev("A1_bank", [(B.base_state(ctxs[u["ckey"]], u["text"]), Q.BANK) for u in chunk])
        for u, r in zip(chunk, res):
            if r:
                done[u["uid"]] = compact(r)
        json.dump(done, open(OUTP, "w"))
        print("saved", len(done))
