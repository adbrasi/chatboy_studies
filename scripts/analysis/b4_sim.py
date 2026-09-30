"""b4 — mecanismo 7: CONTROLADOR DE DISTRIBUIÇÃO por conversa (código), em simulação de 30 turnos seguidos.
Em 8 conversas do maichat (de fora do dev), o bot faz o papel de um dos falantes ("Sam") em 30 turnos seguidos.
As mensagens do outro ("Alex") são as reais; as do próprio bot no histórico são as que ELE gerou antes (é assim que um
vício se auto-reforça num chat real). A cada turno, 1 chamada do Jev (as ~30 perguntas do a9) dá a leitura do momento.
Condições:
  SA   persona pura
  SS   prompt de estilo estático
  ST   briefing de alvos (T) por turno, sem memória da conversa
  STN  ST + normalizador de código
  STC  ST + normalizador + CONTROLADOR: janela das últimas 20 mensagens do bot; taxa de pergunta/riso/emoji/"!" puxada
       para o alvo humano (se a janela passou do alvo, proíbe; se ficou abaixo, libera com prob. maior); cada muleta
       ("wait", "tbh", "lol", "honestly", "literally", "ngl", "fr", "lowkey", "true", "same", "fair", "ok so"...) no máx.
       1 vez a cada 15 msgs (usou -> proibida no briefing); a mesma 1ª palavra no máx. 2 vezes na janela; e pergunta
       não autorizada que ainda sair é podada em código (regra R3).
Alvos humanos: taxas dos turnos humanos do maichat FORA das 8 conversas simuladas.
Uso: python3 b4_sim.py <modelos> <conds> [n_convs] [n_turns]"""
import json, os, re, sys, collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import b4_llm as L
from a9_common import load_mai, turn_text, style_fingerprint, ask_many_timed
from a9_brief import Q as A9Q, state_of, compact
from b4_common import (budget, feats, normalize, units, join_units, words, h01, PERSONA, STATIC, BOT, USER, SCR, ADATA,
                       LAUGH, EMOJI, load_points)
from b4_gen import target_brief, clean

SIMF = os.path.join(SCR, "sim.jsonl")
MARKERS = {"wait": r"\bwait\b", "tbh": r"\btbh\b", "ngl": r"\bngl\b", "honestly": r"\bhonestly\b", "literally": r"\bliterally\b",
           "lol": r"\blo+l\b", "lmao": r"\blmf?ao+\b", "haha": r"\b(?:a?ha){2,}h?\b", "fr": r"\bfr\b", "lowkey": r"\blowkey\b",
           "omg": r"\bomg+\b", "true": r"^\W*true\b", "same": r"^\W*same\b", "fair": r"^\W*fair\b", "ok so": r"^\W*ok(?:ay)? so\b",
           "tho": r"\btho\b", "like": r"\blike\b", "rn": r"\brn\b", "idk": r"\bidk\b", "bro": r"\bbro\b", "😭": "😭", "💀": "💀",
           "vibe": r"\bvibes?\b", "yeah": r"^\W*yeah\b", "oh": r"^\W*oh\b", "ugh": r"\bugh+\b", "ooh": r"\bo+h+\b"}
MARK_RE = {k: re.compile(v, re.I) for k, v in MARKERS.items()}
TRACK = ["q", "laugh", "emoji", "excl"]


def first_word(t):
    w = words((t or "").lower())
    return w[0] if w else ""


def build_convs(n_convs=8, n_turns=30):
    rows = load_mai()
    dev_convs = {p["conv_id"] for p in load_points() if p["split"] == "dev"}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["conv_id"]].append(r)
    cands = []
    for cid, ts in by.items():
        if cid in dev_convs:
            continue
        for spk in sorted({t["speaker"] for t in ts}):
            dps = [i for i in range(1, len(ts)) if ts[i]["speaker"] == spk and ts[i - 1]["speaker"] != spk
                   and ts[i]["session"] == ts[i - 1]["session"] and turn_text(ts[i]["texts"]) and turn_text(ts[i - 1]["texts"])]
            if len(dps) >= n_turns + 2:
                cands.append((cid, spk, dps))
    cands.sort(key=lambda x: h01(x[0], x[1], "sim"))
    chosen, used = [], set()
    for cid, spk, dps in cands:
        if cid in used:
            continue
        chosen.append((cid, spk, dps[1:n_turns + 1])); used.add(cid)
        if len(chosen) == n_convs:
            break
    return by, chosen


def human_targets(by, sim_cids):
    fs = [feats(turn_text(t["texts"])) for cid, ts in by.items() if cid not in sim_cids for t in ts[1:]]
    return {k: float(np.mean([f[k] for f in fs])) for k in TRACK + ["llmish", "perf"]} | {
        "markers_per_msg": {k: float(np.mean([bool(rx.search(turn_text(t["texts"]))) for cid, ts in by.items()
                                               if cid not in sim_cids for t in ts])) for k, rx in MARK_RE.items()}}


def controller_brief(b, win, tgt, pid):
    """Ajusta o orçamento do turno a partir da janela das últimas 20 mensagens do bot (código puro)."""
    b = dict(b)
    notes = []
    if len(win) >= 5:
        for k in TRACK:
            rate = np.mean([f[k] for f in win[-20:]])
            if rate > tgt[k] + 0.03:
                b[k] = False
            elif rate < tgt[k] - 0.03 and not b[k] and h01(pid, "ctl", k) < min(0.5, 2 * tgt[k]):
                b[k] = True
    used = collections.Counter()
    for f in win[-15:]:
        for m in f["_markers"]:
            used[m] += 1
    ban = [m for m, c in used.items() if c >= 1 and m not in ("yeah", "oh", "like")] + [m for m in ("yeah", "oh", "like") if used[m] >= 2]
    fw = collections.Counter(f["_first"] for f in win[-20:])
    ban_first = [w for w, c in fw.items() if c >= 2 and w]
    if ban:
        notes.append("Don't use: " + ", ".join(f"'{m}'" for m in ban[:8]) + ".")
    if ban_first:
        notes.append("Don't start with: " + ", ".join(f"'{w}'" for w in ban_first[:5]) + ".")
    return b, notes, set(ban), set(ban_first)


def sim_conv(model, cond, by, cid, spk, dps, fp_all, jev_read, tgt):
    ts = by[cid]
    own_gen = {}  # turn index -> texto gerado pelo bot
    win, out = [], []
    for step, i in enumerate(dps):
        sess = [x for x in ts[:i] if x["session"] == ts[i]["session"]][-12:]
        while sess and sess[0]["speaker"] == spk:
            sess = sess[1:]
        hist = [{"who": BOT if x["speaker"] == spk else USER,
                 "text": own_gen.get(x["turn_idx"], turn_text(x["texts"])) if x["speaker"] == spk else turn_text(x["texts"])}
                for x in sess]
        pid = f"{cid}_{ts[i]['turn_idx']}"
        p = {"id": pid, "history": hist, "jev": jev_read[pid], "fp": fp_all[(cid, spk)]}
        b = budget(p)
        notes, bans, banf = [], set(), set()
        if cond == "STC":
            b, notes, bans, banf = controller_brief(b, win, tgt, pid)
        if cond == "SA":
            system = PERSONA
        elif cond == "SS":
            system = STATIC
        else:
            system = STATIC + "\n\nFor your next message:\n" + target_brief(p, b) + ("\n" + "\n".join(notes) if notes else "")
        m = [{"role": "system", "content": system}] + [{"role": "assistant" if h["who"] == BOT else "user", "content": h["text"]} for h in hist]
        r = L.chat(m, model=model, max_tokens=200, temperature=0.8, seed=0, tag="sim")
        txt = clean(r.get("text"))
        if not txt:
            r = L.chat(m, model=model, max_tokens=200, temperature=1.0, seed=7, tag="sim")
            txt = clean(r.get("text")) or "ok"
        raw = txt
        if cond in ("STN", "STC"):
            txt = normalize(txt, b)
        if cond == "STC":
            us = units(txt)
            if "?" in txt and not b["q"]:
                us = [u for u in us if "?" not in u] or us
            # muleta proibida que ainda saiu: tira se sobrar texto
            for mk in bans:
                rx = MARK_RE[mk]
                t2 = rx.sub("", join_units(us)).strip(" ,")
                if len(words(t2)) >= 1 and rx.search(join_units(us)):
                    us = [re.sub(r"\s{2,}", " ", rx.sub("", u)).strip(" ,") for u in us]
                    us = [u for u in us if u]
            txt = join_units(us) if us else txt
        own_gen[ts[i]["turn_idx"]] = txt
        f = feats(txt, p)
        f["_markers"] = [k for k, rx in MARK_RE.items() if rx.search(txt)]
        f["_first"] = first_word(txt)
        win.append(f)
        hum = turn_text(ts[i]["texts"])
        out.append({"model": model, "cond": cond, "cid": cid, "step": step, "pid": pid, "text": txt, "raw": raw, "human": hum,
                    "budget": {k: b[k] for k in ("q", "laugh", "emoji", "excl", "words")}, "notes": notes,
                    "cost": r.get("cost", 0), "latency": r.get("latency")})
    return out


def main(models, conds, n_convs=8, n_turns=30):
    by, chosen = build_convs(n_convs, n_turns)
    sim_cids = {c for c, _, _ in chosen}
    tgt = human_targets(by, sim_cids)
    json.dump({"targets": tgt, "convs": [(c, s, len(d)) for c, s, d in chosen]},
              open(os.path.join(ADATA, "b4_sim_setup.json"), "w"), indent=1, ensure_ascii=False)
    # impressão digital da persona (fixa, "configurada"): o falante real na conversa inteira
    fp_all = {}
    for cid, spk, dps in chosen:
        ts = by[cid]
        fp_all[(cid, spk)] = style_fingerprint([turn_text(t["texts"]) for t in ts if t["speaker"] == spk],
                                               [turn_text(t["texts"]) for t in ts if t["speaker"] != spk])
    # leitura do Jev por ponto: state = histórico REAL (o controle compara condições no mesmo estado de leitura)
    items, keys = [], []
    for cid, spk, dps in chosen:
        ts = by[cid]
        for i in dps:
            sess = [x for x in ts[:i] if x["session"] == ts[i]["session"]][-12:]
            while sess and sess[0]["speaker"] == spk:
                sess = sess[1:]
            hist = [{"who": BOT if x["speaker"] == spk else USER, "text": turn_text(x["texts"])} for x in sess]
            items.append((state_of({"history": hist}), A9Q)); keys.append(f"{cid}_{ts[i]['turn_idx']}")
    res = ask_many_timed(items, workers=4)
    jev_read = {k: compact(a) for k, (a, _) in zip(keys, res) if a}
    done = set()
    if os.path.exists(SIMF):
        for l in open(SIMF):
            d = json.loads(l); done.add((d["model"], d["cond"], d["cid"]))
    for model in models:
        for cond in conds:
            todo = [(cid, spk, dps) for cid, spk, dps in chosen if (model, cond, cid) not in done]
            with ThreadPoolExecutor(8) as ex:
                outs = list(ex.map(lambda x: sim_conv(model, cond, by, x[0], x[1], x[2], fp_all, jev_read, tgt), todo))
            with open(SIMF, "a", encoding="utf-8") as f:
                for o in outs:
                    for r in o:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(model, cond, "convs", len(todo), "llm cost total", round(L.stats["cost"], 4), flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2].split(","), *(int(x) for x in sys.argv[3:]))
