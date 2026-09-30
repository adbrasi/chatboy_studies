"""b1 step 1: replies from the stronger actors (gpt-6-luna, deepseek-flash-latest) at the 250 a8 decision points.
base_* = same minimal persona as a8 (no style hints); styled_* = same a8 data-driven style guide.
Output: scratch b1/gen_<cond>.json  {ctx_id: text}"""
import json, os, sys
from b1_common import SCR, A8CTX
import b1_llm as L
from a8_gen import BASE_SYS, STYLE, to_messages

MODELS = {"luna": "openai/gpt-6-luna", "deepseek": "~deepseek/deepseek-flash-latest"}

if __name__ == "__main__":
    X = json.load(open(A8CTX))
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        X = X[:3]
    items, keys = [], []
    for mk, model in MODELS.items():
        for kind, sysmsg in (("base", BASE_SYS), ("styled", STYLE)):
            for c in X:
                items.append({"messages": to_messages(c, sysmsg), "model": model, "temperature": 0.8, "max_tokens": 900, "seed": 0, "effort": "low"})
                keys.append((f"{kind}_{mk}", c["id"]))
    res = L.chat_many(items, workers=4)
    out = {}
    for (cn, cid), r in zip(keys, res):
        if r:
            out.setdefault(cn, {})[cid] = r
    for cn, d in out.items():
        json.dump(d, open(os.path.join(SCR, f"gen_{cn}.json"), "w"), ensure_ascii=False, indent=0)
        print(cn, len(d))
    lat = sorted(L.stats["latency_s"])
    print({k: v for k, v in L.stats.items() if k != "latency_s"}, "lat_p50", lat[len(lat) // 2] if lat else None)
