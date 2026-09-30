"""b4 — sonda 2: prefill no meio da frase e logit_bias (deepseek) com o tokenizer do DeepSeek-V3."""
import json, os
import b4_llm as L
from tokenizers import Tokenizer
SCR = "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad"
tok = Tokenizer.from_file(os.path.join(SCR, "ds_tok.json"))
voc = tok.get_vocab()
H = [{"role": "system", "content": "You are Sam, chatting with Alex on a messaging app. Reply as Sam."},
     {"role": "user", "content": "omg i passed my driving test!!"}]
out = {}
for m in L.MODELS:
    rs = [L.chat(H, model=m, max_tokens=120, prefill="lol i knew you would", seed=s, tag="probe2") for s in (0, 1)]
    out[m] = {"prefill_mid": [r["text"] for r in rs]}
    print(m, out[m], flush=True)
bang = {str(i): -100 for k, i in voc.items() if "!" in k}
words = [" totally", " amazing", " Amazing", " definitely", " absolutely", " honestly", " Honestly", "😂"]
lb = dict(bang)
for w in words:
    ids = tok.encode(w, add_special_tokens=False).ids
    if len(ids) == 1:
        lb[str(ids[0])] = -100
res = []
for s in range(6):
    a = L.chat(H, model="deepseek", max_tokens=120, seed=s, tag="probe2")
    b = L.chat(H, model="deepseek", max_tokens=120, seed=s, logit_bias=lb, tag="probe2")
    res.append([a["text"], b["text"]])
    print(res[-1], flush=True)
out["deepseek_logit_bias"] = {"n_tokens_biased": len(lb), "pairs": res,
                              "excl_rate_base": sum("!" in a for a, b in res) / 6, "excl_rate_bias": sum("!" in b for a, b in res) / 6}
print(out["deepseek_logit_bias"]["excl_rate_base"], out["deepseek_logit_bias"]["excl_rate_bias"])
json.dump(out, open(os.path.join(L.ROOT, "analysis", "data", "b4_probe2.json"), "w"), ensure_ascii=False, indent=1)
json.dump(lb, open(os.path.join(SCR, "b4_ds_logit_bias.json"), "w"))
print(L.stats)
