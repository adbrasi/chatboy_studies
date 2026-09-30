"""a5 — experimento Jev NOVO: callbacks, tipo de transição de tópico, pergunta pessoal, devolução de pergunta,
elogio e história. Um call por turno amostrado (maichat + janelas anotadas do whatsapp_nl).
Saída: analysis/data/a5_jev_moves.jsonl (pequeno: ids + respostas escalares)."""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jev import ask_many, choice, noul, summary
from a5_common import load_all, OUT

N_MAI, N_WA = int(os.environ.get("N_MAI", 1600)), int(os.environ.get("N_WA", 1150))
RECENT, EARLIER = 3, 40
MAXC = 220


def gap(sec):
    if sec is None or sec != sec: return None
    if sec < 3 * 3600: return None
    return "hours later" if sec < 24 * 3600 else "next day or later"


def fmt(r, corpus):
    txt = r.text.strip()
    if len(txt) > MAXC: txt = txt[:MAXC] + "…"
    d = {"speaker": r.speaker, "text": txt}
    return d


QS = {
    "callback": noul("Does `current_turn` explicitly refer back to a specific thing (topic, event, joke or detail) that was mentioned in "
                     "`earlier_conversation` but is NOT being discussed in `recent_turns`?",
                     "yes, it brings back something from earlier_conversation", "no, it only relates to recent_turns or is new"),
    "transition": choice("How does `current_turn` relate to the topic of `recent_turns`?", {
        "continues": "stays on the same topic as recent_turns",
        "smooth_shift": "moves to a new topic that is associated with / triggered by something just said",
        "abrupt_shift": "jumps to a new, unrelated topic (e.g. 'btw', 'anyway', 'oh and')",
        "returns_earlier": "returns to a topic from earlier_conversation",
        "reaction_only": "only a short reaction, no topic content"}),
    "personal_q": noul("Does `current_turn` ask the other person a personal question about their life, feelings, opinions, plans "
                       "or experiences (not purely practical logistics)?"),
    "asks_back": noul("Does `current_turn` answer a question from the other person AND ask them the same or a similar question back "
                      "(e.g. 'and you?', 'wbu?', 'en jij?')?"),
    "compliment": noul("Does `current_turn` compliment, praise or flatter the other person?"),
    "story": noul("Does `current_turn` tell a story or anecdote about something that happened to the speaker?"),
}


def main():
    df = load_all()
    df = df.sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
    rnd = random.Random(55)
    mai = df[(df.corpus == "maichat") & (df.turn_idx >= 8)]
    wa = df[(df.corpus == "whatsapp_nl") & df.D_engagement.notna() & (df.turn_idx >= 8)]
    sel = list(mai.sample(min(N_MAI, len(mai)), random_state=5).index) + list(wa.sample(min(N_WA, len(wa)), random_state=5).index)
    items, meta = [], []
    for i in sel:
        r = df.loc[i]
        conv = df[(df.corpus == r.corpus) & (df.conv_id == r.conv_id)]
        prev = conv[conv.turn_idx < r.turn_idx]
        if r.corpus == "maichat":
            prev = prev[prev.session == r.session]
        recent = prev.tail(RECENT)
        earlier = prev.iloc[:-RECENT].tail(EARLIER) if len(prev) > RECENT else prev.iloc[:0]
        if len(earlier) < 3:
            continue
        E = []
        for _, x in earlier.iterrows():
            d = fmt(x, r.corpus)
            g = gap(x.response_latency_s) if r.corpus != "maichat" else None
            if g: d["after_pause"] = g
            E.append(d)
        state = {"earlier_conversation": E, "recent_turns": [fmt(x, r.corpus) for _, x in recent.iterrows()],
                 "current_turn": fmt(r, r.corpus)}
        items.append((state, QS)); meta.append((r.corpus, r.conv_id, int(r.turn_idx)))
    print("calls", len(items), flush=True)
    res = ask_many(items, workers=4)
    with open(f"{OUT}/a5_jev_moves.jsonl", "w", encoding="utf-8") as fo:
        for (c, cid, ti), a in zip(meta, res):
            if a is None: continue
            o = {"corpus": c, "conv_id": cid, "turn_idx": ti}
            for k, v in a.items():
                o[k] = v.get("noul", v.get("choice"))
                if v["type"] == "choice":
                    o[k + "_probs"] = v.get("probabilities")
            fo.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(summary())


if __name__ == "__main__":
    main()
