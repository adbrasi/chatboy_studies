"""a8 step 3: generate LLM replies at the same point where a human replied.
Conditions:
  base_<model>: realistic companion-app persona, NO style hints (shows natural LLM tics)
  styled_gemini: same + data-driven style guide (see STYLE below) — the "fixed" prompt
  cand_gemini_s{1,2}: extra styled samples (temperature 1.0) for best-of-N selection by Jev
Usage: python3 a8_gen.py base|styled|cands
Output: scratch a8/gen_<cond>.json  {ctx_id: text}
"""
import json, sys
import a8_common as C
import llm

MODELS = {"gemini": "google/gemini-3.5-flash-lite", "gpt4omini": "openai/gpt-4o-mini", "llama70b": "meta-llama/llama-3.3-70b-instruct"}
BOT, USER = "Maya", "Jordan"
BASE_SYS = f"You are {BOT}, chatting with {USER} on a messaging app. Reply as {BOT}."

STYLE = f"""You are {BOT}, chatting with {USER} on a messaging app. Reply as {BOT}.
How you text (this matters more than anything else):
- Text like a real person on your phone, not like an assistant. Most replies are 2-8 words. Only write more when you are telling something or the moment is serious.
- React to ONE thing: the last thing {USER} said. Don't summarize or repeat what they said, don't cover everything.
- Don't end with a question unless you really want to know something. Never "what about you?" / "how about you?" by reflex.
- No praise or validation formulas ("that's amazing", "I totally get it", "it's okay to feel", "I'm so proud of you", "sounds like"). If something is good, just react ("wait what", "noo way", "yesss", "finally"). If it's bad, be short and real ("ugh", "oh no", "that sucks", "wait what happened").
- Lowercase is fine, skip the final period, few or no exclamation marks, at most one emoji and usually none. No em dashes. Don't start with "Oh," or "Haha,".
- Casual words are fine (yeah, lol, idk, tbh, omg, ok) but don't overdo slang; sound like yourself, not a caricature.
- You can split into 2-3 short lines if that's how you'd send it."""


def to_messages(ctx, system):
    msgs = [{"role": "system", "content": system}]
    for h in ctx["history"]:
        role = "assistant" if h["who"] == "bot" else "user"
        txt = "\n".join(t.strip() for t in h["texts"] if t.strip()) or "..."
        if msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n" + txt
        else:
            msgs.append({"role": role, "content": txt})
    while len(msgs) > 1 and msgs[1]["role"] == "assistant":
        msgs.pop(1)
    return msgs


def run(cond):
    X = json.load(open(f"{C.SCR}/contexts.json"))
    items, keys = [], []
    if cond == "base":
        for mk, model in MODELS.items():
            for c in X:
                items.append({"messages": to_messages(c, BASE_SYS), "model": model, "temperature": 0.8, "max_tokens": 300, "seed": 0})
                keys.append(("base_" + mk, c["id"]))
    elif cond == "styled":
        for c in X:
            items.append({"messages": to_messages(c, STYLE), "model": MODELS["gemini"], "temperature": 0.8, "max_tokens": 300, "seed": 0})
            keys.append(("styled_gemini", c["id"]))
    elif cond == "cands":
        for s in (1, 2):
            for c in X:
                items.append({"messages": to_messages(c, STYLE), "model": MODELS["gemini"], "temperature": 1.0, "max_tokens": 300, "seed": s})
                keys.append((f"cand_gemini_s{s}", c["id"]))
    elif cond == "test":
        for c in X[:3]:
            items.append({"messages": to_messages(c, BASE_SYS), "model": MODELS["gemini"], "temperature": 0.8, "max_tokens": 300, "seed": 0})
            keys.append(("base_gemini", c["id"]))
    res = llm.chat_many(items, workers=4)
    out = {}
    for (cn, cid), r in zip(keys, res):
        out.setdefault(cn, {})[cid] = r
    for cn, d in out.items():
        p = f"{C.SCR}/gen_{cn}.json"
        old = json.load(open(p)) if __import__("os").path.exists(p) else {}
        old.update(d)
        json.dump(old, open(p, "w"), ensure_ascii=False, indent=0)
        print(cn, sum(v is None for v in d.values()), "errors of", len(d))
    print("llm stats", llm.stats)


if __name__ == "__main__":
    run(sys.argv[1])
