"""b2 Parte B, T9: a própria LLM decide a divisão ("divida este texto nas bolhas que uma pessoa mandaria").
Mesmos 800 turnos (dev+teste) do cenário T, 4 modelos autorizados. Mesmo contexto curto (as 3 últimas falas).
Saída: scratchpad/b2/nb_llm_raw.pkl"""
import os, json, re
import numpy as np, pandas as pd
from b2_nb_common import SCR
import b2_llm

raw_R = None
import b2_nb_T as TT  # builds R (no Jev calls at import)
R = TT.R
MODELS = ["google/gemini-3.5-flash-lite", "~deepseek/deepseek-flash-latest", "openai/gpt-6-luna", "inception/mercury-2.5"]


def prompt(r):
    ctx = "\n".join(f"{c['speaker']}: " + " / ".join(c.get("messages", [c.get("text", "")])) for c in r.ctx[-3:])
    return [{"role": "user", "content":
             "You are simulating how real people text in a chat app.\n\nRecent chat (messages separated by ' / '):\n" + ctx +
             f"\n\nNow {r.speaker} replies with this text:\n\"{r.text}\"\n\n"
             "Split the reply into the separate chat messages (bubbles) that a real person would send in a row. "
             "Many people send a whole reply as one message; others split it. Keep exactly the same words in the same order, "
             "do not add or remove anything. Output one message per line and nothing else."}]


items, meta = [], []
for m in MODELS:
    for i, r in enumerate(R.itertuples()):
        items.append({"messages": prompt(r), "model": m, "temperature": 0.3, "max_tokens": 1500,
                      "effort": "low"})
        meta.append((m, i))
print("calls", len(items), flush=True)
out = b2_llm.chat_many(items, workers=4)
res = {m: [None] * len(R) for m in MODELS}
for (m, i), txt in zip(meta, out):
    res[m][i] = txt
pd.to_pickle({"keys": list(R.key), "res": res}, os.path.join(SCR, "nb_llm_raw.pkl"))
print(b2_llm.stats)
