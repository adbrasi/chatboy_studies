"""a7: classifica, com o Jev, o MOVIMENTO de flerte/afeto de um turno T e o MOVIMENTO da resposta R do parceiro.

Conjunto 1: turnos de flerte/afeto já anotados no jev_base (D.flirting>=.5 ou emotion=affection ou
            intent=compliment_affection) que têm resposta do parceiro na mesma sessão.
Conjunto 2 (expansão): janelas contíguas extras de 3 chats de casal do whatsapp_nl (wa_M_003/036/004), fora das
            janelas do jev_base, com uma pergunta de filtro (is_flirt) — sem ver o rótulo base.
Saída: analysis/data/a7_moves.jsonl
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_common import load, OUT
from jev import ask_many, choice, noul, score, summary

MOVES = {
    "tease_banter": "playful teasing, mock insult, banter, playful challenge (e.g. 'brat', 'define finish', 'you wish')",
    "compliment_direct": "explicit compliment about the other person (looks, personality): 'you look great', 'ur sweet', 'you're cute'",
    "compliment_indirect": "indirect appreciation: 'i love how you understand me', 'still my fav', 'that's the softest thing you've said'",
    "miss_longing": "missing the other / wishing to be together: 'I miss you', 'wish you were here', 'I want to cuddle you'",
    "love_declaration": "saying 'I love you' / 'love you too' / 'love you'",
    "pet_name_greeting": "greeting or addressing with a pet name / endearment ('hello my love', 'hey babe', 'schatje')",
    "caring_checkin": "caring: 'did you eat?', 'proud of you', 'get some rest', 'good luck', looking after the other",
    "invitation_plan": "inviting / proposing to do something together (call, date, meet, watch, play)",
    "innuendo_sexual": "double meaning, suggestive or explicitly sexual remark",
    "self_deprecation_charm": "charming self-deprecation or playful modesty",
    "mock_offense_pout": "pretending to be offended, pouting, fake jealousy ('Oh okay bye 😒', 'rude???', 'wow ok')",
    "affectionate_goodbye": "goodbye / goodnight with affection ('night xx', 'sleep well love', 'bye dummy <3')",
    "emoji_or_kiss_only": "only kisses / hearts / affectionate emoji or 'xx' with no words",
    "reassurance_validation": "reassuring or validating the other ('you're not annoying', 'you did great')",
    "vulnerable_share": "sharing a personal feeling or insecurity, seeking closeness",
    "none": "not flirting and not affectionate (neutral, logistics, info)",
}
RESP = {
    "reciprocate": "returns the same affection back ('love you too', 'miss you too', 'you too 😘', compliment back)",
    "tease_back": "teases back / keeps the banter going with a playful jab",
    "flustered_accept": "accepts shyly / flustered / embarrassed ('stop 🥺', 'ugh ur sweet', 'dont expose me', 'shut up')",
    "simple_thanks": "simply accepts or thanks ('thanks!', 'aw thanks')",
    "deflate_humor": "downplays it with humor or dry wit ('it's the same face as yesterday')",
    "escalate": "raises the romantic/sexual intensity beyond the other's turn",
    "engage_enthusiastic": "enthusiastically picks it up, builds on it or asks about it ('I LOVE THIS IDEA', 'what should we do?')",
    "minimal_ack": "minimal acknowledgement only ('ok', 'haha', 'hehe', an emoji)",
    "ignore_change_topic": "ignores the flirt/affection and talks about something else or logistics",
    "mock_offense": "pretends to be offended or pouts",
    "polite_decline": "gently declines or sidesteps the advance without being rude",
    "rude_reject": "rejects rudely / annoyed",
}
INT = ["0: nothing romantic or affectionate", "1: light warmth or friendly fondness",
       "2: clear playful flirting or affection", "3: strong romantic affection (love, longing, pet names)",
       "4: very intense / sexual / passionate"]


def fmt(t, lab=None):
    txt = " / ".join(t["texts"])
    return {"speaker": lab or t["speaker"], "text": txt[:500]}


def questions(S, R, with_filter=False):
    q = {
        "move": choice(f"What flirting/affection move does {S} make in target_turn? Judge only target_turn (not the reply).", MOVES),
        "t_int": score(f"How romantic/affectionate/flirtatious is {S}'s target_turn?", INT),
        "resp": choice(f"How does {R} respond, in reply_turn, to {S}'s target_turn?", RESP),
        "r_int": score(f"How romantic/affectionate/flirtatious is {R}'s reply_turn?", INT),
        "alive": noul(f"Does {R}'s reply_turn keep the flirty/affectionate mood going, inviting more of it?"),
        "shy": noul(f"Does {R}'s reply_turn show shyness, embarrassment or being flustered?"),
    }
    if with_filter:
        q["is_flirt"] = noul(f"Is {S} flirting or being affectionate/romantic in target_turn?")
    return q


def build(df, T, with_filter=False):
    conv = df[(df.corpus == T.corpus) & (df.conv_id == T.conv_id) & (df.session == T.session)]
    prev = conv[conv.turn_idx < T.turn_idx].tail(6)
    nxt = conv[conv.turn_idx == T.turn_idx + 1].iloc[0]
    state = {"previous_turns": [fmt(p) for _, p in prev.iterrows()] or "(start of conversation)",
             "target_turn": fmt(T), "reply_turn": fmt(nxt)}
    return state, questions(T.speaker, nxt.speaker, with_filter)


def main():
    df = load()
    df = df.sort_values(["corpus", "conv_id", "turn_idx"]).reset_index(drop=True)
    a = df[df.annotated]
    has_reply = (a.next_speaker.notna()) & (a.next_speaker != a.speaker) & (a.next_session == a.session)
    fa = a[((a.D_flirting >= .5) | (a.D_emotion == "affection") | (a.D_intent == "compliment_affection")) & has_reply]
    items, meta = [], []
    for _, T in fa.iterrows():
        s, q = build(df, T)
        items.append((s, q)); meta.append({"set": "base", "corpus": T.corpus, "conv_id": T.conv_id, "turn_idx": int(T.turn_idx)})
    # expansão: janelas extras de chats de casal do whatsapp_nl
    random.seed(7)
    for cid, nwin in (("wa_M_003", 3), ("wa_M_036", 1), ("wa_M_004", 1)):
        c = df[(df.corpus == "whatsapp_nl") & (df.conv_id == cid)]
        cand = c[~c.annotated & (c.next_speaker.notna()) & (c.next_speaker != c.speaker) & (c.next_session == c.session)]
        idxs = sorted(cand.turn_idx.tolist())
        starts = random.sample(range(0, max(1, len(idxs) - 60)), nwin)
        chosen = set()
        for s0 in starts:
            chosen.update(idxs[s0:s0 + 60])
        for ti in sorted(chosen):
            T = c[c.turn_idx == ti].iloc[0]
            s, q = build(df, T, with_filter=True)
            items.append((s, q)); meta.append({"set": "expand", "corpus": T.corpus, "conv_id": cid, "turn_idx": int(ti)})
    print("calls:", len(items))
    res = ask_many(items, workers=4)
    with open(os.path.join(OUT, "a7_moves.jsonl"), "w", encoding="utf-8") as f:
        for m, (s, _), r in zip(meta, items, res):
            if r is None:
                continue
            o = dict(m); o["T"] = s["target_turn"]["text"]; o["R"] = s["reply_turn"]["text"]
            for k, v in r.items():
                o[k] = v.get("noul", v.get("score", v.get("choice")))
                if v["type"] == "choice":
                    o[k + "_conf"] = v.get("confidence")
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(summary())


if __name__ == "__main__":
    main()
