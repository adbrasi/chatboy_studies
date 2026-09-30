"""a2: how sessions end (rules): last-turn type, who closes, closing-ritual length (maichat), end-with-question."""
import os, sys, re, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import a2_load as L
from a2_sessions import BYE, ACK
S = L.SCR
s = pd.read_pickle(f"{S}/sessions.pkl")
EMO = re.compile(r"^[\W_]+$")
out = {}
def last_kind(r):
    t = r.last_text.strip()
    if r.last_farewell: return "farewell/xx"
    if r.last_ack: return "ack(ok/top/goed)"
    if EMO.match(t) or "weggelaten" in t: return "emoji/media only"
    if r.last_has_q: return "question (unanswered)"
    if re.search(r"\b(haha+|hihi|lol|hehe)\b", t, re.I): return "laugh"
    return "statement"
for c in ["whatsapp_nl", "maichat"]:
    x = s[(s.corpus == c) & (s.n_speakers == 2)].copy()
    x["kind"] = x.apply(last_kind, axis=1)
    x["closer_is_opener"] = x.closer == x.opener
    k = x.kind.value_counts(normalize=True).round(3)
    print(c, len(x)); print(k)
    print("closer==opener", x.closer_is_opener.mean().round(3))
    out[c] = {"n_sessions": len(x), "last_kind": k.to_dict(), "closer_is_opener": round(x.closer_is_opener.mean(), 3),
              "tail3_bye_turns_mean": round(x.n_bye_turns_tail.mean(), 2),
              "share_ending_with_bye_in_last3": round((x.n_bye_turns_tail > 0).mean(), 3)}
# who takes the "last word": in whatsapp, does the NEXT session get opened by the one who did NOT close?
w = s[(s.corpus == "whatsapp_nl")].sort_values(["conv_id", "session"])
w["next_opener"] = w.groupby("conv_id").opener.shift(-1)
w["q_end"] = w.last_has_q
y = w[w.next_opener.notna()]
print("P(next opener == closer | ended with q)", y[y.q_end].pipe(lambda z: (z.next_opener != z.closer).mean()).round(3), y.q_end.sum())
print("P(next opener == closer | not q)", y[~y.q_end].pipe(lambda z: (z.next_opener == z.closer).mean()).round(3))
# maichat closing ritual: trailing turns with bye/closing
t = L.turns(); t = t[t.corpus == "maichat"]
ritual = []
for cid, g in t.groupby("conv_id"):
    g = g.sort_values("turn_idx")
    texts = [" / ".join(x) for x in g.texts]
    n = 0
    for tx in reversed(texts):
        if BYE.search(tx) or ACK.match(tx) or EMO.match(tx.strip()) or re.search(r"gtg|gotta go|got to go|have to go|wrap|talk (to you )?(later|soon)|speak soon|ttyl|text (u|you) later|night", tx, re.I):
            n += 1
        else:
            break
    first_pre = None; first_bye = None
    L10 = texts[-10:]
    for i, tx in enumerate(L10):
        if first_pre is None and re.search(r"gtg|gotta go|got to go|have to go|need to go|should go|wrap (this )?up|time over|i'?ll let you go|gonna (sleep|go)|text (u|you) later|talk later|catch up", tx, re.I):
            first_pre = len(L10) - 1 - i
        if first_bye is None and BYE.search(tx):
            first_bye = len(L10) - 1 - i
    ritual.append({"conv": cid, "ritual_turns": n, "pre_close_turns_before_end": first_pre, "turns_after_first_bye": first_bye, "last": texts[-3:]})
r = pd.DataFrame(ritual)
print(r.ritual_turns.describe()); print(r.ritual_turns.value_counts().sort_index())
print("pre-closing found", r.pre_close_turns_before_end.notna().mean(), r.pre_close_turns_before_end.describe())
print("first bye in last10 found", r.turns_after_first_bye.notna().mean(), r.turns_after_first_bye.value_counts().sort_index().to_dict())
out["maichat_ritual"] = {"ritual_turns_mean": round(r.ritual_turns.mean(), 2), "ritual_turns_median": float(r.ritual_turns.median()),
                         "dist": r.ritual_turns.value_counts().sort_index().to_dict(),
                         "share_with_pre_closing": round(r.pre_close_turns_before_end.notna().mean(), 3),
                         "pre_closing_turns_before_end_median": float(r.pre_close_turns_before_end.median()),
                         "share_with_bye_in_last10": round(r.turns_after_first_bye.notna().mean(), 3),
                         "turns_after_first_bye_dist": r.turns_after_first_bye.value_counts().sort_index().to_dict()}
for _, q in r.sample(12, random_state=1).iterrows(): print(q.ritual_turns, q.last)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../../analysis/data/a2_closings_rules.json"), "w"), indent=1, default=str)
