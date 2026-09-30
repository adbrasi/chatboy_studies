"""a7: flerte entre desconhecidos no NPS Chat (salas públicas, 2006).

Etapa 1: Jev classifica posts (candidatos por regex + amostra aleatória de controle) quanto a flerte, movimento e intensidade.
Etapa 2: para flertes DIRIGIDOS a alguém (menciona o apelido), acha a próxima fala do destinatário (até 25 posts depois)
         e o Jev classifica a resposta (mesmas classes de a7_moves).
Saídas: analysis/data/a7_nps_posts.jsonl, analysis/data/a7_nps_replies.jsonl
"""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_common import P, OUT
from a7_moves import MOVES, RESP, INT
from jev import ask_many, choice, noul, score, summary

CAND = re.compile(r"\b(sexy|hot|cute|kiss\w*|hug\w*|love|luv|baby|babe|hun|honey|sweet\w*|sweetie|darlin\w*|beautiful|gorgeous|"
                  r"marry|flirt\w*|lick|horny|wanna|pm me|miss(ed)? (you|u|ya)|single|date|girlfriend|boyfriend|wife|husband|"
                  r"handsome|pretty|smile|wink|mwah|muah)\b|:\*|<3|;\)|;-\)|\*hugs?\*|\*kiss", re.I)
USER = re.compile(r"\d\d-\d\d-[a-z0-9]+User\d+")


def short(name):
    m = re.search(r"User(\d+)$", name)
    return f"U{m.group(1)}" if m else name


def clean(t):
    return USER.sub(lambda m: short(m.group(0)), t)


def load_nps():
    rows = [json.loads(l) for l in open(f"{P}/messages.jsonl", encoding="utf-8") if '"nps_chatroom"' in l[:40]]
    by = {}
    for r in rows:
        by.setdefault(r["conv_id"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["idx"])
    return by


def ctx(posts, i, k=6):
    prev = [p for p in posts[max(0, i - 12):i] if p["dialogue_act"] != "System"][-k:]
    return [{"speaker": short(p["speaker"]), "text": clean(p["text"])[:200]} for p in prev]


def stage1(by):
    random.seed(11)
    cands, ctrl = [], []
    for cid, posts in by.items():
        for i, p in enumerate(posts):
            if p["dialogue_act"] == "System" or not p["text"].strip():
                continue
            (cands if CAND.search(p["text"]) else ctrl).append((cid, i))
    ctrl = random.sample(ctrl, 300)
    sel = [(c, i, "cand") for c, i in cands] + [(c, i, "ctrl") for c, i in ctrl]
    items = []
    for cid, i, _ in sel:
        p = by[cid][i]
        S = short(p["speaker"])
        state = {"setting": "public group chat room between strangers (2006)", "previous_posts": ctx(by[cid], i),
                 "target_post": {"speaker": S, "text": clean(p["text"])[:300]}}
        q = {"is_flirt": noul(f"Is {S} flirting, hitting on someone, or being romantic/affectionate in target_post?"),
             "move": choice(f"What flirting/affection move does {S} make in target_post?", MOVES),
             "t_int": score(f"How romantic/affectionate/flirtatious is {S}'s target_post?", INT)}
        items.append((state, q))
    print("stage1 calls", len(items))
    res = ask_many(items, workers=4)
    out = []
    for (cid, i, grp), r in zip(sel, res):
        if r is None:
            continue
        p = by[cid][i]
        out.append({"conv_id": cid, "idx": p["idx"], "i": i, "grp": grp, "speaker": p["speaker"], "act": p["dialogue_act"],
                    "text": p["text"], "is_flirt": r["is_flirt"]["noul"], "move": r["move"]["choice"],
                    "move_conf": r["move"]["confidence"], "t_int": r["t_int"]["score"]})
    with open(os.path.join(OUT, "a7_nps_posts.jsonl"), "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    return out


def stage2(by, posts):
    pairs = []
    for o in posts:
        if o["is_flirt"] < .5:
            continue
        seq = by[o["conv_id"]]
        tgt = [u for u in USER.findall(o["text"]) if u != o["speaker"]]
        if not tgt:
            continue
        X = tgt[0]
        rep = None
        for j in range(o["i"] + 1, min(len(seq), o["i"] + 26)):
            q = seq[j]
            if q["speaker"] == X and q["dialogue_act"] != "System":
                rep = j; break
        pairs.append((o, X, rep))
    items, meta = [], []
    for o, X, rep in pairs:
        if rep is None:
            continue
        seq = by[o["conv_id"]]
        between = [p for p in seq[o["i"] + 1:rep] if p["dialogue_act"] != "System"][-4:]
        S, R = short(o["speaker"]), short(X)
        state = {"setting": "public group chat room between strangers (2006)", "previous_posts": ctx(seq, o["i"], 4),
                 "target_post": {"speaker": S, "text": clean(o["text"])[:300]},
                 "posts_in_between": [{"speaker": short(p["speaker"]), "text": clean(p["text"])[:150]} for p in between],
                 "reply_post": {"speaker": R, "text": clean(seq[rep]["text"])[:300]}}
        q = {"addressed": noul(f"Is {R}'s reply_post a response to {S}'s target_post (rather than to someone else)?"),
             "resp": choice(f"How does {R} respond, in reply_post, to {S}'s target_post?", RESP),
             "r_int": score(f"How romantic/affectionate/flirtatious is {R}'s reply_post?", INT),
             "alive": noul(f"Does {R}'s reply_post keep the flirty mood going, inviting more of it?")}
        items.append((state, q)); meta.append((o, X, seq[rep]))
    print("stage2 pairs", len(pairs), "with reply", len(items))
    res = ask_many(items, workers=4)
    with open(os.path.join(OUT, "a7_nps_replies.jsonl"), "w", encoding="utf-8") as f:
        for o, X, rep in pairs:
            if rep is None:
                f.write(json.dumps({"conv_id": o["conv_id"], "idx": o["idx"], "T": o["text"], "move": o["move"], "t_int": o["t_int"],
                                    "R": None}, ensure_ascii=False) + "\n")
        for (o, X, rp), r in zip(meta, res):
            if r is None:
                continue
            f.write(json.dumps({"conv_id": o["conv_id"], "idx": o["idx"], "T": o["text"], "move": o["move"], "t_int": o["t_int"],
                                "R": rp["text"], "R_act": rp["dialogue_act"], "gap_posts": rp["idx"] - o["idx"],
                                "addressed": r["addressed"]["noul"], "resp": r["resp"]["choice"], "resp_conf": r["resp"]["confidence"],
                                "r_int": r["r_int"]["score"], "alive": r["alive"]["noul"]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    by = load_nps()
    posts = stage1(by)
    stage2(by, posts)
    print(summary())
