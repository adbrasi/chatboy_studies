"""c2 — rótulo-ouro independente com openai/gpt-6-luna (2º rotulador) para os turnos reais.
O luna vê MAIS contexto que o Jev (até 24 turnos anteriores, com os intervalos de tempo) e rotula, para cada dimensão,
a direção e o nível {0 nada, 1 leve, 2 moderado, 3 forte, 4 marcante}, e os eventos presentes.
Saída: data/processed/c2_gold_luna.jsonl (incremental)."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from c2_common import PROC, DIMS, EVENTS, llm, parse_json, jl_load, jl_append, gap_text

OUT = f"{PROC}/c2_gold_luna.jsonl"

DIM_DEF = {
    "trust": "how much C trusts U (reliability, honesty, keeping their word)",
    "comfort": "how close to and at ease with U C feels",
    "affection": "C's fondness or romantic attraction toward U",
    "resentment": "C's hurt, irritation or resentment toward U",
    "jealousy": "C's jealousy about U's attention going to someone else",
    "respect": "C's respect and admiration for U",
    "protectiveness": "how worried C is about U / wants to look after U",
    "playfulness": "the playful complicity and banter between them",
}
EVT_DEF = {
    "apology": "U apologizes to C", "hurtful_joke": "U makes a mean-spirited or tasteless joke at C's expense (not friendly banter)",
    "insult_criticism": "U seriously criticizes, insults, blames or belittles C",
    "promise_made": "U promises or commits to do something for/with C",
    "promise_kept": "the message shows U did something they had promised/planned for or with C",
    "cancel_or_broken_promise": "U cancels plans with C, says they can't come, or admits failing to do what they promised",
    "absence_explained": "U explains or apologizes for having been away / not replying",
    "compliment": "U compliments or praises C", "vulnerability": "U reveals something personal, painful or vulnerable",
    "other_person_jealousy": "U mentions another person in a way that could make C jealous",
    "practical_care": "U shows everyday care for C (ate? slept? got home safe? good luck, practical help)",
    "defensive": "U defends/justifies themselves, makes excuses or deflects blame",
    "perceived_lie": "the message would seem dishonest to C (excuse that doesn't add up, contradicts earlier)",
    "dismissive": "U dismisses, ignores or brushes off C's feelings, question or effort",
    "affection_expr": "U expresses affection/love or says they miss C",
    "friendly_tease": "U playfully teases or jokes around with C in a friendly way",
    "gratitude": "U thanks C or shows appreciation", "interest_in_char": "U asks about C's life, day or feelings with genuine interest",
}

SYS = """You are an expert annotator of relationship dynamics in real private chats between two people who know each other.
You will see a chat log and one TARGET message written by U. Judge how the TARGET message changes what C (the other person)
feels toward U, as a real person in C's position would, taking the whole conversation into account (a joke between close
friends is not an offense; an apology only eases hurt if there was something to forgive; ordinary small talk changes nothing).
Most messages in ordinary chat change nothing: use "none" unless there is a real, noticeable effect.

Dimensions (C's feelings toward U):
""" + "\n".join(f"- {d}: {v}" for d, v in DIM_DEF.items()) + """

Levels: 0 = not at all, 1 = slight (small, passing), 2 = moderate (noticeable, lasts a while), 3 = strong (a real effect that
would last days), 4 = profound (a major, lasting change in the relationship).

Events (true only if clearly present in the TARGET message):
""" + "\n".join(f"- {e}: {v}" for e, v in EVT_DEF.items()) + """

Answer ONLY with JSON:
{"dims": {"trust": ["up"|"down"|"none", level], ... all 8 dimensions ...}, "events": [list of event names that are present]}"""


def fmt_log(r):
    lines = []
    for h in r["ctx"]:
        lines.append(f"{'U' if h['from'] == r['U'] else 'C'}: {h['text']}")
    g = r.get("gap_hours")
    gl = f"[{gap_text(g)} later]\n" if g is not None and g >= 3 else ""
    return "\n".join(lines) + f"\n{gl}>>> TARGET U: {r['text']}"


def label(r):
    lang = "Dutch" if r["corpus"] == "whatsapp_nl" else "English"
    m = [{"role": "system", "content": SYS},
         {"role": "user", "content": f"Chat log ({lang}). U and C are the two people.\n\n{fmt_log(r)}"}]
    res = llm(m, model="luna", max_tokens=1500, reasoning={"effort": "low"}, tag="c2_gold")
    j = parse_json(res.get("text"))
    ok = j is not None and isinstance(j.get("dims"), dict)
    return {"id": r["id"], "gold": j if ok else None, "raw": None if ok else (res.get("text") or "")[:500],
            "cost": res.get("cost", 0)}


def main(limit=None):
    turns = [json.loads(l) for l in open(f"{PROC}/c2_real_turns.jsonl")]
    done = {d["id"] for d in jl_load(OUT) if d.get("gold")}
    todo = [r for r in turns if r["id"] not in done][:limit]
    print("todo", len(todo), flush=True)
    B = 40
    cost = 0
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(todo), B):
            recs = list(ex.map(label, todo[i:i + B]))
            jl_append(OUT, recs)
            cost += sum(r["cost"] for r in recs)
            print(i + B, "bad", sum(r["gold"] is None for r in recs), "cost", round(cost, 4), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
