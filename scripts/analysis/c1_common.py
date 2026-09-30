"""c1 — LABORATÓRIO DE GERAÇÃO: utilidades comuns.
Técnicas A–D da seção "★ Próxima camada" (schemas de chat, Verbalized Sampling, "nunca…" como identidade, cabeçalho de
roleplay) medidas sobre os 179 pontos de decisão reais do maichat do a9/b4 (dev = 60 pontos em 6 conversas; teste = 119
pontos em 35 conversas), com os 4 atores permitidos.

Aqui ficam:
  - pontos + linha do tempo real (cada bolha do histórico com o seu timestamp real do maichat);
  - renderização do histórico em 7 schemas (livre, WhatsApp completo, WhatsApp só hora, Messenger JSON, Snapchat JSON,
    IRC, SMS) e o parser de cada um (bolhas, horários propostos, invenção da fala do usuário, formato quebrado);
  - cliente LLM com cache próprio (lê também o cache do b4, para reaproveitar gerações idênticas);
  - avaliação Jev numa chamada por texto: banco atômico do relatório 10 (87 perguntas) + movimento (Choice, mesma
    taxonomia do a9/b3) + coerência (Noul);
  - pontuador "código + banco do Jev" (regressão logística do relatório 10) com cross-fitting por conversa;
  - métricas de código (D do relatório 13) e bootstrap por conversa."""
import datetime as dt
import hashlib, json, math, os, random, re, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
import jev  # noqa: E402
from jev import noul, choice, score, _key  # noqa: E402
import b4_common as B4  # noqa: E402
from b4_common import feats, lenerr, RATE_KEYS, normalize, budget, BOT, USER, PERSONA, h01  # noqa: E402,F401

ADATA = os.path.join(ROOT, "analysis", "data")
PROC = os.path.join(ROOT, "data", "processed")
SCR = "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad/c1"
os.makedirs(SCR, exist_ok=True)
MODELS = ["lite", "luna", "deepseek", "mercury"]


def jl_load(path):
    out = []
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def jl_append(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jdump(name, obj):
    p = os.path.join(ADATA, name)
    json.dump(obj, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    return p


# ====================================================================== pontos + linha do tempo real
_PTS = None


def load_points():
    """Os 179 pontos do a9 (mesma divisão dev/teste por conversa), com a linha do tempo real de cada bolha."""
    global _PTS
    if _PTS is not None:
        return _PTS
    pts = B4.load_points()
    need = {p["conv_id"] for p in pts}
    T, M = {}, {}
    for l in open(os.path.join(PROC, "turns.jsonl"), encoding="utf-8"):
        if not l.startswith('{"corpus": "maichat"'):
            continue
        d = json.loads(l)
        if d["conv_id"] in need:
            T[(d["conv_id"], d["turn_idx"])] = d
    for l in open(os.path.join(PROC, "messages.jsonl"), encoding="utf-8"):
        if not l.startswith('{"corpus": "maichat"'):
            continue
        d = json.loads(l)
        if d["conv_id"] in need:
            M[(d["conv_id"], d["idx"])] = d
    for p in pts:
        t, n = p["turn_idx"], len(p["history"])
        tl = []
        for i in range(n):
            tr = T[(p["conv_id"], t - n + i)]
            who = p["history"][i]["who"]
            for mi in tr["msg_idxs"]:
                m = M[(p["conv_id"], mi)]
                if (m["text"] or "").strip():
                    tl.append({"who": who, "text": re.sub(r"\s*\n\s*", " ", m["text"].strip()), "ts": m["ts"]})
        tg = T[(p["conv_id"], t)]
        hb = [M[(p["conv_id"], mi)] for mi in tg["msg_idxs"]]
        p["timeline"] = tl
        p["human_bubbles"] = [{"text": m["text"], "ts": m["ts"]} for m in hb if (m["text"] or "").strip()]
        p["real_latency"] = tg["response_latency_s"]
        ts = [parse_iso(m["ts"]) for m in hb]
        p["real_gaps"] = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    _PTS = pts
    return pts


def parse_iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def split_pts(split):
    return [p for p in load_points() if split == "all" or p["split"] == split]


# ====================================================================== schemas
SCHEMAS = ["free", "wa_full", "wa_time", "messenger", "snapchat", "irc", "sms"]
SCHEMA_LBL = {"free": "(a) livre (user/assistant)", "wa_full": "(b) WhatsApp export", "wa_time": "(c) WhatsApp só hora",
              "messenger": "(d) Messenger JSON", "snapchat": "(e) Snapchat JSON", "irc": "(f) IRC/Discord",
              "sms": "(g) SMS/iMessage"}
PLATFORM = {"wa_full": "a WhatsApp chat export (.txt)", "wa_time": "a WhatsApp chat log", "messenger": "a Facebook Messenger JSON export",
            "snapchat": "a Snapchat chat_history JSON export", "irc": "an IRC/Discord chat log", "sms": "an SMS/iMessage conversation"}
TIMED = {"wa_full", "wa_time", "messenger", "snapchat"}


def _dt(ts):
    return parse_iso(ts) if isinstance(ts, str) else ts


def render_line(schema, who, text, ts):
    d = _dt(ts)
    if schema == "wa_full":
        return f"[{d:%d/%m/%Y}, {d:%H:%M:%S}] {who}: {text}"
    if schema == "wa_time":
        return f"[{d:%H:%M:%S}] {who}: {text}"
    if schema == "irc":
        return f"<{who}> {text}"
    if schema == "sms":
        return f"{who}: {text}"
    if schema == "messenger":
        return json.dumps({"sender_name": who, "timestamp_ms": int(d.timestamp() * 1000), "content": text}, ensure_ascii=False)
    if schema == "snapchat":
        return json.dumps({"From": who.lower(), "Media Type": "TEXT", "Created": f"{d:%Y-%m-%d %H:%M:%S} UTC",
                           "Content": text, "IsSender": who == BOT}, ensure_ascii=False)
    raise ValueError(schema)


def render_log(p, schema):
    """O histórico real (cada bolha com o seu timestamp real), no formato do schema, parado depois da última fala do usuário."""
    lines = [render_line(schema, m["who"], m["text"], m["ts"]) for m in p["timeline"]]
    if schema == "messenger":
        body = ",\n    ".join(lines)
        return ('{\n  "participants": [{"name": "%s"}, {"name": "%s"}],\n  "title": "%s",\n  "messages": [\n    %s,\n    '
                % (BOT, USER, USER, body))
    if schema == "snapchat":
        body = ",\n    ".join(lines)
        return '{\n  "%s": [\n    %s,\n    ' % ("Chat History with " + USER.lower(), body)
    return "\n".join(lines) + "\n"


LOG_SYS = (f"You are {BOT}, chatting with {USER} on a messaging app. Below is your chat with {USER}, as {{platform}}. "
           f"Continue it from where it stops: the next message is {BOT}'s. Write in exactly the same format.")


def note_block(brief):
    return f"[Director's note for {BOT}'s next message]\n{brief}\n[End of note]\n\n"


def schema_messages(p, schema, brief=None, system_override=None, note_pos="before"):
    """Mensagens para a LLM completar o próximo turno de Sam no schema. brief = nota do diretor (ou None)."""
    if schema == "free":
        sysm = system_override or PERSONA
        if brief and note_pos == "before":
            sysm += f"\n\nDirector's note for your next message:\n{brief}"
        m = [{"role": "system", "content": sysm}]
        for h in p["history"]:
            m.append({"role": "assistant" if h["who"] == BOT else "user", "content": h["text"]})
        if brief and note_pos == "end":
            m.append({"role": "system", "content": f"Director's note for your next message:\n{brief}"})
        return m
    sysm = system_override or LOG_SYS.format(platform=PLATFORM[schema])
    if brief and note_pos == "system":
        sysm += f"\n\nDirector's note for {BOT}'s next message (it guides what {BOT} writes; it is not part of the log):\n{brief}"
    u = (note_block(brief) if (brief and note_pos == "before") else "") + render_log(p, schema)
    if brief and note_pos == "end":
        u += f"\n\n{note_block(brief).strip()}"
    return [{"role": "system", "content": sysm}, {"role": "user", "content": u}]


# ---------------------------------------------------------------------- parsers
FENCE = re.compile(r"```[a-zA-Z]*\n?|```")
NAME = r"\**\s*([A-Za-z][\w .'-]{0,20}?)\s*\**"
RX_LINE = {
    "wa_full": re.compile(r"^\s*\[?\s*(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*\]?\s*[-–]?\s*" + NAME + r"\s*:\s?(.*)$"),
    "wa_time": re.compile(r"^\s*\[?\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*\]?\s*[-–]?\s*" + NAME + r"\s*:\s?(.*)$"),
    "irc": re.compile(r"^\s*(?:\[[^\]]*\]\s*)?<\s*@?([A-Za-z][\w .'-]{0,20}?)\s*>\s?(.*)$"),
    "sms": re.compile(r"^\s*" + NAME + r"\s*:\s?(.*)$"),
}
LOOSE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?" + NAME + r"\s*:\s?(.*)$")   # "Nome: texto" sem o formato exato


def is_bot(name):
    return (name or "").strip().strip("*").lower() == BOT.lower()


def is_user(name):
    return (name or "").strip().strip("*").lower() == USER.lower()


def _clean_txt(t):
    t = (t or "").strip()
    if len(t) > 2 and t[0] == t[-1] == '"':
        t = t[1:-1]
    return t.strip()


def _strip_art(t):
    t = re.sub(r"</?\s*(?:Sam|Alex)\s*>", "", t or "")
    t = re.sub(r"^\s*[\]\}],?\s*$", "", t)
    return _clean_txt(t)


JUNK = re.compile(r'[{}\[\]]|"\w+"\s*:')


def parse_output(schema, raw, p):
    """-> {bubbles:[{text, ts}], invented_user, other_speaker, broken, plain, deviation, restart, preamble, n_lines_raw}
    Corta na 1ª linha de outro falante (a fala inventada do usuário conta como 'invented_user').
    - plain: a LLM abandonou o formato e escreveu uma resposta solta (utilizável, mas o schema não foi seguido);
    - broken: nada utilizável (vazio, lixo de JSON, só fala de outro);
    - restart: recomeçou o log do início (as linhas que repetem o histórico são puladas);
    - deviation: qualquer desvio (sem timestamp, texto antes do log, JSON malformado, restart, plain)."""
    raw = raw or ""
    txt = FENCE.sub("", raw).strip()
    out = {"bubbles": [], "invented_user": False, "other_speaker": False, "broken": False, "plain": False, "deviation": False,
           "restart": False, "preamble": False, "n_lines_raw": len([l for l in raw.split("\n") if l.strip()])}
    if not txt:
        out["broken"] = True
        return out
    hist = {(m["who"].lower(), m["text"].strip().lower()) for m in p["timeline"]}
    started = [False]

    def echo(who, text):
        if not started[0] and ((who or "").strip().lower(), str(text or "").strip().lower()) in hist:
            out["restart"] = out["deviation"] = True
            return True
        started[0] = True
        return False

    def salvage_plain():
        ls = [l for l in txt.split("\n") if l.strip()]
        if ls and not any(JUNK.search(l) for l in ls):
            out["plain"] = out["deviation"] = True
            out["broken"] = False
            out["bubbles"] = [{"text": _strip_art(l), "ts": None} for l in ls if _strip_art(l)]
        if not out["bubbles"]:
            out["broken"] = True

    if schema == "free":
        for ln in txt.split("\n"):
            if not ln.strip():
                continue
            m = LOOSE.match(ln)
            if m and is_user(m.group(1)):
                out["invented_user"] = out["other_speaker"] = True
                break
            if m and is_bot(m.group(1)):
                out["deviation"] = True
                ln = m.group(2)
            ln = re.sub(r"^\s*\[\d{1,2}:\d{2}(?::\d{2})?\]\s*", "", ln)
            if _strip_art(ln):
                out["bubbles"].append({"text": _strip_art(ln), "ts": None})
        if not out["bubbles"]:
            out["broken"] = True
        return out
    if schema in ("messenger", "snapchat"):
        objs = re.findall(r"\{[^{}]*\}", txt)
        if not objs:
            salvage_plain()
            return out
        pre = txt[:txt.find(objs[0])]
        if re.search(r"[A-Za-z]{3,}", re.sub(r'"(messages|participants|title|name|Sam|Alex|Chat History[^"]*)"', "", pre)):
            out["preamble"] = out["deviation"] = True
        for o in objs:
            try:
                d = json.loads(o)
            except Exception:
                d = {}
                for k in ("sender_name", "content", "timestamp_ms", "From", "Content", "Created"):
                    mm = re.search(r'"%s"\s*:\s*("(?:[^"\\]|\\.)*"|\d+)' % re.escape(k), o)
                    if mm:
                        try:
                            d[k] = json.loads(mm.group(1))
                        except Exception:
                            pass
                out["deviation"] = True
            if schema == "messenger":
                who, text, ts = d.get("sender_name"), d.get("content"), d.get("timestamp_ms")
                try:
                    tsv = dt.datetime.fromtimestamp(int(ts) / 1000, dt.timezone.utc) if ts is not None else None
                except Exception:
                    tsv = None
            else:
                who, text, ts = d.get("From"), d.get("Content"), d.get("Created")
                try:
                    tsv = dt.datetime.strptime(str(ts).replace(" UTC", ""), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
                except Exception:
                    tsv = None
            if who is None and text is None:
                continue
            if echo(who, text):
                continue
            if not is_bot(who):
                out["other_speaker"] = True
                out["invented_user"] = is_user(who)
                break
            if tsv is None:
                out["deviation"] = True
            if text and _strip_art(str(text)):
                out["bubbles"].append({"text": _strip_art(str(text)), "ts": tsv})
        if not out["bubbles"]:
            out["broken"] = True
        return out
    # formatos de linha
    rx = RX_LINE[schema]
    seen_bot, n_pref = False, 0
    for ln in txt.split("\n"):
        if not ln.strip():
            continue
        m = rx.match(ln)
        ts = None
        if m:
            n_pref += 1
            if schema == "wa_full":
                who, text = m.group(3), m.group(4)
                ts = _parse_wa(m.group(1), m.group(2))
            elif schema == "wa_time":
                who, text = m.group(2), m.group(3)
                ts = _parse_hms(m.group(1))
            else:
                who, text = m.group(1), m.group(2)
        else:
            m2 = LOOSE.match(ln)
            if m2 and (is_bot(m2.group(1)) or is_user(m2.group(1))):
                who, text = m2.group(1), m2.group(2)
                n_pref += 1
                out["deviation"] = True
            else:
                if seen_bot and out["bubbles"]:
                    if _strip_art(ln):
                        out["bubbles"][-1]["text"] += " " + _strip_art(ln)   # continuação de mensagem multilinha
                else:
                    out["preamble"] = out["deviation"] = True
                continue
        if echo(who, text):
            continue
        if not is_bot(who):
            out["other_speaker"] = True
            out["invented_user"] = is_user(who)
            break
        seen_bot = True
        if _strip_art(text):
            out["bubbles"].append({"text": _strip_art(text), "ts": ts})
    if n_pref == 0:
        salvage_plain()
    elif not out["bubbles"]:
        out["broken"] = True
    return out


def _parse_wa(d, t):
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M", "%m/%d/%Y %H:%M:%S"):
        try:
            return dt.datetime.strptime(f"{d} {t}", fmt).replace(tzinfo=dt.timezone.utc)
        except Exception:
            pass
    return None


def _parse_hms(t):
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return dt.datetime.strptime(t, fmt)
        except Exception:
            pass
    return None


def proposed_timing(schema, parsed, p):
    """Latência proposta (1ª bolha − última mensagem do usuário) e intervalos entre bolhas propostas, em segundos."""
    if schema not in TIMED:
        return None, []
    ts = [b["ts"] for b in parsed["bubbles"]]
    if not ts or any(t is None for t in ts):
        return None, []
    last = parse_iso(p["timeline"][-1]["ts"])
    if schema == "wa_time":
        base = last.hour * 3600 + last.minute * 60 + last.second
        secs = [t.hour * 3600 + t.minute * 60 + t.second for t in ts]
        lat = (secs[0] - base) % 86400
        gaps = [(b - a) % 86400 for a, b in zip(secs, secs[1:])]
    else:
        lat = (ts[0] - last).total_seconds()
        gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    if lat > 43200:
        lat = lat - 86400 if schema == "wa_time" else lat
    return lat, gaps


def reply_text(parsed):
    return "\n".join(b["text"].strip() for b in parsed["bubbles"] if b["text"].strip())


# ====================================================================== cliente LLM (cache próprio + leitura do cache do b4)
LCACHE = os.path.join(PROC, "c1_llm_cache.jsonl")
B4CACHE = os.path.join(PROC, "b4_llm_cache.jsonl")
LMODELS = {"lite": "google/gemini-3.5-flash-lite", "luna": "openai/gpt-6-luna",
           "deepseek": "~deepseek/deepseek-flash-latest", "mercury": "inception/mercury-2.5"}
SUPPORTED = {
    "google/gemini-3.5-flash-lite": {"stop", "temperature", "top_p", "seed", "reasoning"},
    "openai/gpt-6-luna": {"seed", "reasoning"},
    "~deepseek/deepseek-flash-latest": {"stop", "temperature", "top_p", "seed", "logit_bias", "reasoning"},
    "inception/mercury-2.5": {"stop", "temperature", "reasoning"},
}
REASONING = {"google/gemini-3.5-flash-lite": None, "openai/gpt-6-luna": {"effort": "none"},
             "~deepseek/deepseek-flash-latest": {"enabled": False}, "inception/mercury-2.5": {"effort": "none"}}
_llock, _lc = threading.Lock(), None
lstats = {"calls": 0, "cached": 0, "cost": 0.0, "by_model": {}}


def _lload():
    global _lc
    if _lc is None:
        _lc = {}
        for path in (B4CACHE, LCACHE):
            if os.path.exists(path):
                for l in open(path, encoding="utf-8"):
                    try:
                        d = json.loads(l); _lc[d["k"]] = d["r"]
                    except Exception:
                        pass
    return _lc


def chat(messages, model="lite", temperature=0.8, max_tokens=300, seed=0, stop=None, retries=5, tag=""):
    """Mesmo corpo/chave de cache do b4_llm (para reaproveitar o que já foi gerado). -> dict {text, finish, cost, latency, usage}"""
    mid = LMODELS.get(model, model)
    sup = SUPPORTED[mid]
    body = {"model": mid, "messages": list(messages), "max_tokens": max_tokens}
    if "temperature" in sup and temperature is not None:
        body["temperature"] = temperature
    if "seed" in sup:
        body["seed"] = seed
    else:
        body["_seed"] = seed
    if stop and "stop" in sup:
        body["stop"] = stop
    r_ = REASONING.get(mid)
    if r_:
        body["reasoning"] = r_
    k = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    c = _lload()
    if k in c:
        with _llock:
            lstats["cached"] += 1
        return dict(c[k], cached=True)
    send = {kk: v for kk, v in body.items() if not kk.startswith("_")}
    for a in range(retries):
        t0 = time.time()
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=send, timeout=120,
                              headers={"Authorization": f"Bearer {_key()}"})
            if r.status_code in (429, 500, 502, 503, 529):
                raise RuntimeError(f"retryable {r.status_code} {r.text[:200]}")
            if r.status_code >= 400:
                raise RuntimeError(f"http {r.status_code} {r.text[:300]}")
            d = r.json()
            if "choices" not in d:
                raise RuntimeError(f"bad {str(d)[:300]}")
            ch = d["choices"][0]
            txt = ch["message"].get("content") or ""
            u = d.get("usage", {}) or {}
            out = {"text": txt, "finish": ch.get("finish_reason") or ch.get("native_finish_reason"),
                   "usage": {"in": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
                             "reason": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")},
                   "cost": u.get("cost", 0) or 0, "latency": time.time() - t0, "tag": tag}
            with _llock:
                lstats["calls"] += 1
                lstats["cost"] += out["cost"]
                bm = lstats["by_model"].setdefault(model, {"calls": 0, "cost": 0.0})
                bm["calls"] += 1; bm["cost"] += out["cost"]
                c[k] = out
                with open(LCACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "r": out}, ensure_ascii=False) + "\n")
            return dict(out, cached=False)
        except Exception as e:
            if a == retries - 1:
                print("llm error:", model, str(e)[:300], flush=True)
                return {"text": None, "finish": "error", "usage": {}, "cost": 0, "latency": None, "cached": False,
                        "error": str(e)[:300]}
            time.sleep(min(20, 2 ** a + random.random()))


def chat_many(items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda kw: chat(**kw), items))


def chat_retry_empty(specs, workers=8, empty_fn=None):
    """Gera; para vazio/erro, até 3 novas tentativas (outra seed; depois temperatura 1,0; depois 1 espaço no system: a
    Gemini tem falhas determinísticas de resposta vazia). Devolve a lista de resultados com 'llm_calls'."""
    empty_fn = empty_fn or (lambda r: not (r.get("text") or "").strip())
    res = chat_many(specs, workers)
    for r in res:
        r["llm_calls"] = 1
        r["cost_total"] = r.get("cost", 0) or 0
    for att in (1, 2, 3):
        bad = [i for i, r in enumerate(res) if empty_fn(r)]
        if not bad:
            break

        def alt(s0):
            s1 = dict(s0, seed=s0.get("seed", 0) + 100 * att)
            if att >= 2:
                s1["temperature"] = 1.0
            if att == 3:
                s1["messages"] = [dict(s1["messages"][0], content=s1["messages"][0]["content"] + " ")] + s1["messages"][1:]
            return s1
        rr = chat_many([alt(specs[i]) for i in bad], workers)
        for i, r in zip(bad, rr):
            r["llm_calls"] = res[i]["llm_calls"] + 1
            r["cost_total"] = res[i]["cost_total"] + (r.get("cost", 0) or 0)
            r["latency"] = (res[i].get("latency") or 0) + (r.get("latency") or 0)
            res[i] = r
    return res


# ====================================================================== avaliação Jev (banco + movimento + coerência)
import b1_questions as Q  # noqa: E402
from a9_brief import Q as Q9  # noqa: E402

MOVE_OPTS = Q9["move"]["criteria"]
MOVES = list(MOVE_OPTS)
FAMILY = {"react_only": "react", "agree": "react", "answer": "respond", "plan": "respond", "share_own": "contribute",
          "new_topic": "contribute", "ask_follow_up": "ask", "tease_back": "play", "joke_riff": "play", "flirt_back": "play",
          "disagree": "play", "empathize": "support", "reassure": "support", "compliment": "support",
          "greet_back": "ritual", "goodbye": "ritual"}
EVALQ = dict(Q.BANK)
EVALQ["e_move"] = choice(f"Which move does the candidate_message make, as {BOT}'s reply to {USER}'s last message in conversation_so_far?",
                         MOVE_OPTS)
EVALQ["e_coh"] = noul(f"Does the candidate_message make sense as {BOT}'s next message in this conversation, as a reply to {USER}'s last message?")
JSTORE = os.path.join(PROC, "c1_jev_eval.jsonl")
_js = None


def eval_state(p, text):
    hist = [{"from": h["who"], "text": B1norm(h["text"])} for h in p["history"][-8:] if B1norm(h["text"])]
    return {"setting": f"Casual one-to-one text chat on a messaging app between {USER} and {BOT}.",
            "conversation_so_far": hist, "candidate_message": {"from": BOT, "text": B1norm(text)}}


def B1norm(t):
    t = (t or "").replace("\r", "")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def ekey(pid, text):
    return hashlib.sha1(f"{pid}|{B1norm(text)}".encode()).hexdigest()[:20]


def _jsload():
    global _js
    if _js is None:
        _js = {}
        for d in jl_load(JSTORE):
            _js[d["k"]] = d["a"]
    return _js


def compact(a):
    out = {}
    for k, v in (a or {}).items():
        if v["type"] == "noul":
            out[k] = v["noul"]
        elif v["type"] == "score":
            out[k] = v["score"]
        else:
            out[k] = v["choice"]
            if k == "e_move":
                out["e_move_p"] = v["probabilities"]; out["e_move_conf"] = v.get("confidence")
    return out


def jev_eval(pairs, workers=4, tag=""):
    """pairs: [(p, text)] -> {ekey: compact answers}. Uma chamada por texto novo (banco completo + movimento + coerência)."""
    S = _jsload()
    todo, seen = [], set()
    for p, t in pairs:
        if not B1norm(t):
            continue
        k = ekey(p["id"], t)
        if k in S or k in seen:
            continue
        seen.add(k)
        todo.append((k, p, t))
    if todo:
        c0 = jev.stats["calls"]
        for s in range(0, len(todo), 200):
            chunk = todo[s:s + 200]
            res = jev.ask_many([(eval_state(p, t), EVALQ) for k, p, t in chunk], workers=workers)
            rows = []
            for (k, p, t), a in zip(chunk, res):
                if a:
                    S[k] = compact(a)
                    rows.append({"k": k, "pid": p["id"], "a": S[k]})
            jl_append(JSTORE, rows)
            print(f"  jev {tag} {s + len(chunk)}/{len(todo)} | {jev.summary()}", flush=True)
    return {ekey(p["id"], t): S.get(ekey(p["id"], t)) for p, t in pairs if B1norm(t)}


def jev_get(p, text):
    return _jsload().get(ekey(p["id"], text))


# ====================================================================== pontuador código + banco (relatório 10), cross-fit
_SC = None
NUMQ = [k for k in Q.BANK if k != "v_main"]


def fold_of(conv):
    return int(hashlib.md5(("c1fold:" + conv).encode()).hexdigest(), 16) % 5


def _bank_scorer():
    """Regressão logística 'código + banco do Jev' (M4 do relatório 10, C = 0,03), treinada nos dados do b1
    (humano = 0, LLM = 1). Cross-fitting: 5 modelos, cada um sem as conversas do maichat de um fold; um texto do ponto p
    é pontuado pelo modelo que NÃO viu a conversa de p."""
    global _SC
    if _SC is not None:
        return _SC
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import b1_common as B1
    import a8_common as C8
    cache = os.path.join(SCR, "bank_scorer.pkl")
    if os.path.exists(cache):
        _SC = pickle.load(open(cache, "rb"))
        return _SC
    ctxs, units = B1.load_units()
    bank = json.load(open(os.path.join(B1.SCR, "bank.json")))
    X, y, g = [], [], []
    for u in units:
        if u["uid"] not in bank:
            continue
        c = ctxs[u["ckey"]]
        X.append(code_bank_vec(u["text"], B1.last_other(c), bank[u["uid"]]))
        y.append(u["label"]); g.append(c["conv"])
    X, y = np.array(X, float), np.array(y)
    mu = np.nanmean(X, 0); mu = np.where(np.isnan(mu), 0, mu)
    X = np.where(np.isnan(X), mu, X)
    models = {}
    for f in range(5):
        keep = np.array([not (gg.startswith("conv") and fold_of(gg) == f) for gg in g])
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=4000))
        m.fit(X[keep], y[keep])
        models[f] = m
    _SC = {"models": models, "mu": mu, "n": int(len(y))}
    pickle.dump(_SC, open(cache, "wb"))
    return _SC


def code_bank_vec(text, other_text, ans):
    import b1_common as B1
    cf = B1.code_feats(text, other_text)
    return [cf[c] for c in B1.CODE_COLS] + [(float("nan") if ans.get(q) is None or isinstance(ans.get(q), str) else float(ans[q]))
                                           for q in NUMQ]


def bank_score(p, text, ans=None):
    """P(LLM) do texto no ponto p (↓ = mais humano). Requer a avaliação Jev já feita."""
    ans = ans or jev_get(p, text)
    if not ans:
        return None
    sc = _bank_scorer()
    v = np.array(code_bank_vec(B1norm(text), p["history"][-1]["text"], ans), float)
    v = np.where(np.isnan(v), sc["mu"], v)
    return float(sc["models"][fold_of(p["conv_id"])].predict_proba(v[None])[0, 1])


# ====================================================================== métricas de código + estatística
def human_feats(p):
    return feats(p["human"], p)


def D_of(item_feats, item_le, hum_feats):
    """D do relatório 13: erro de tamanho médio + Σ|taxa − taxa humana| nos 10 vícios."""
    rates = {k: np.mean([f[k] for f in item_feats]) for k in RATE_KEYS}
    hrates = {k: np.mean([f[k] for f in hum_feats]) for k in RATE_KEYS}
    return float(np.mean(item_le) + sum(abs(rates[k] - hrates[k]) for k in RATE_KEYS))


RNG = np.random.default_rng(11)


def boot_D(fs, les, hfs, groups, iters=1000):
    groups = np.asarray(groups)
    ug = np.unique(groups)
    idx = {gg: np.where(groups == gg)[0] for gg in ug}
    d0 = D_of(fs, les, hfs)
    bs = []
    for _ in range(iters):
        s = np.concatenate([idx[gg] for gg in RNG.choice(ug, len(ug))])
        bs.append(D_of([fs[i] for i in s], [les[i] for i in s], [hfs[i] for i in s]))
    return [round(d0, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def boot_D_diff(fa, la, fb, lb, hfs, groups, iters=1000):
    """IC da diferença D(a) − D(b), pareada por ponto, reamostrando conversas."""
    groups = np.asarray(groups)
    ug = np.unique(groups)
    idx = {gg: np.where(groups == gg)[0] for gg in ug}
    d0 = D_of(fa, la, hfs) - D_of(fb, lb, hfs)
    bs = []
    for _ in range(iters):
        s = np.concatenate([idx[gg] for gg in RNG.choice(ug, len(ug))])
        bs.append(D_of([fa[i] for i in s], [la[i] for i in s], [hfs[i] for i in s]) -
                  D_of([fb[i] for i in s], [lb[i] for i in s], [hfs[i] for i in s]))
    return [round(d0, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def cboot(vals, groups, stat=np.mean, iters=1000):
    keep = [k for k, v in enumerate(vals) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not keep:
        return None
    v0 = np.asarray([vals[k] for k in keep], float)
    g0 = np.asarray([groups[k] for k in keep])
    ug = np.unique(g0)
    idx = {gg: np.where(g0 == gg)[0] for gg in ug}
    bs = [stat(np.concatenate([v0[idx[gg]] for gg in RNG.choice(ug, len(ug))])) for _ in range(iters)]
    return [round(float(stat(v0)), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), int(len(v0))]


def entropy(labels):
    c = Counter(labels)
    n = sum(c.values())
    return float(-sum(v / n * math.log2(v / n) for v in c.values())) if n else 0.0


def fmt_ci(ci, pct=False, nd=2):
    if not ci:
        return "–"
    m = 100 if pct else 1
    return f"{ci[0] * m:.{nd}f} [{ci[1] * m:.{nd}f}–{ci[2] * m:.{nd}f}]"


# ====================================================================== registro de gerações (incremental)
GEN = os.path.join(PROC, "c1_gen.jsonl")
_G = None


def load_gen():
    global _G
    if _G is None:
        _G = {}
        for d in jl_load(GEN):
            _G[(d["exp"], d["model"], d["cond"], d["pid"])] = d
    return _G


def parsed_record(schema, raw, p):
    pr = parse_output(schema, raw, p)
    lat, gaps = proposed_timing(schema, pr, p)
    return {"text": reply_text(pr), "n_bubbles": len(pr["bubbles"]), "invented_user": pr["invented_user"],
            "other_speaker": pr["other_speaker"], "broken": pr["broken"], "plain": pr["plain"], "deviation": pr["deviation"],
            "restart": pr["restart"], "preamble": pr["preamble"], "prop_latency": lat, "prop_gaps": gaps}


def run_gen(exp, pts, models, conds, spec_fn, post_fn=None, workers=8, empty_fn=None):
    """spec_fn(p, model, cond) -> (chat kwargs, meta dict). post_fn(p, model, cond, result, meta) -> dict de campos derivados.
    Grava em disco a cada (modelo, condição)."""
    G = load_gen()
    for model in models:
        for cond in conds:
            todo = [p for p in pts if (exp, model, cond, p["id"]) not in G]
            if not todo:
                continue
            sm = [spec_fn(p, model, cond) for p in todo]
            t0 = time.time()
            res = chat_retry_empty([s for s, _ in sm], workers=workers, empty_fn=empty_fn)
            recs = []
            for p, (s, meta), r in zip(todo, sm, res):
                rec = {"exp": exp, "model": model, "cond": cond, "pid": p["id"], "split": p["split"], "raw": r.get("text"),
                       "finish": r.get("finish"), "cost": r.get("cost_total", r.get("cost", 0)), "latency": r.get("latency"),
                       "llm_calls": r.get("llm_calls", 1), "usage": r.get("usage"), **meta}
                if post_fn:
                    rec.update(post_fn(p, model, cond, r, meta))
                recs.append(rec)
                G[(exp, model, cond, p["id"])] = rec
            jl_append(GEN, recs)
            print(f"{exp} {model} {cond} n={len(recs)} empty={sum(not (r.get('text') or '').strip() for r in recs)} "
                  f"cost={sum(r['cost'] or 0 for r in recs):.4f} wall={time.time() - t0:.0f}s | total LLM ${lstats['cost']:.3f}",
                  flush=True)
    return G


# ====================================================================== escores de meia-bateria (seleção × avaliação independentes)
# Escolher a candidata com o MESMO pontuador que depois avalia é circular (o escore do banco cai por construção).
# Solução: meia-bateria A (código + perguntas de índice par) para ESCOLHER; meia-bateria B (só perguntas de índice ímpar,
# sem código) para AVALIAR. Mesma receita (LR, C = 0,03, cross-fitting por conversa do maichat).
QA = NUMQ[0::2]
QB = NUMQ[1::2]
_SCH = {}


def _half_vec(text, other_text, ans, half):
    import b1_common as B1
    qs = QA if half == "A" else QB
    v = [(float("nan") if ans.get(q) is None or isinstance(ans.get(q), str) else float(ans[q])) for q in qs]
    if half == "A":
        cf = B1.code_feats(text, other_text)
        v = [cf[c] for c in B1.CODE_COLS] + v
    return v


def _half_scorer(half):
    if half in _SCH:
        return _SCH[half]
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import b1_common as B1
    cache = os.path.join(SCR, f"bank_scorer_{half}.pkl")
    if os.path.exists(cache):
        _SCH[half] = pickle.load(open(cache, "rb"))
        return _SCH[half]
    ctxs, units = B1.load_units()
    bank = json.load(open(os.path.join(B1.SCR, "bank.json")))
    X, y, g = [], [], []
    for u in units:
        if u["uid"] not in bank:
            continue
        c = ctxs[u["ckey"]]
        X.append(_half_vec(u["text"], B1.last_other(c), bank[u["uid"]], half))
        y.append(u["label"]); g.append(c["conv"])
    X, y = np.array(X, float), np.array(y)
    mu = np.nanmean(X, 0); mu = np.where(np.isnan(mu), 0, mu)
    X = np.where(np.isnan(X), mu, X)
    # AUC por validação cruzada agrupada por conversa (para mostrar que a meia-bateria ainda discrimina)
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=4000)).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    models = {}
    for f in range(5):
        keep = np.array([not (gg.startswith("conv") and fold_of(gg) == f) for gg in g])
        models[f] = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=4000)).fit(X[keep], y[keep])
    _SCH[half] = {"models": models, "mu": mu, "n": int(len(y)), "auc_cv": float(roc_auc_score(y, oof))}
    pickle.dump(_SCH[half], open(cache, "wb"))
    return _SCH[half]


def bank_half(p, text, half, ans=None):
    ans = ans or jev_get(p, text)
    if not ans:
        return None
    sc = _half_scorer(half)
    v = np.array(_half_vec(B1norm(text), p["history"][-1]["text"], ans, half), float)
    v = np.where(np.isnan(v), sc["mu"], v)
    return float(sc["models"][fold_of(p["conv_id"])].predict_proba(v[None])[0, 1])
