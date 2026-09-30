"""b4 — sonda: o que cada provedor realmente aceita (reasoning off, stop, prefill, logit_bias, temperatura)."""
import json, os, sys, requests
import b4_llm as L
H = [{"role": "system", "content": "You are Sam, chatting with Alex on a messaging app. Reply as Sam."},
     {"role": "user", "content": "omg i passed my driving test!!"}]
out = {}
for m in L.MODELS:
    mid = L.MODELS[m]
    r0 = L.chat(H, model=m, reasoning=None, max_tokens=400, tag="probe")          # reasoning padrão do provedor
    r1 = L.chat(H, model=m, max_tokens=400, tag="probe")                           # reasoning desligado
    r2 = L.chat(H, model=m, max_tokens=400, stop=["!"], tag="probe", extra={"_v": "stop"})  # stop em "!"
    r3 = L.chat(H, model=m, max_tokens=400, prefill="wait", tag="probe")         # prefill
    r4 = L.chat(H, model=m, max_tokens=5, tag="probe")                             # corte cego
    out[m] = {"reason_default": [r0["text"], r0["usage"], r0["finish"], round(r0["cost"], 7)],
              "reason_off": [r1["text"], r1["usage"], r1["finish"], round(r1["cost"], 7), r1["latency"]],
              "stop_!": [r2["text"], r2["finish"]], "prefill_wait": [r3["text"], r3["finish"]],
              "max5": [r4["text"], r4["finish"], r4["usage"]]}
    print(m, json.dumps(out[m], ensure_ascii=False, indent=0), flush=True)
json.dump(out, open(os.path.join(L.ROOT, "analysis", "data", "b4_probe.json"), "w"), ensure_ascii=False, indent=1)
print(L.stats)
