"""c2 — Laboratório do estado da relação vivo (cascata Jev 1 -> Jev 2 -> física em código).

Aqui ficam:
  * a taxonomia (dimensões e eventos) e o banco de perguntas do Jev 1 (texto exato);
  * as perguntas do Jev 2 (magnitude: Score descritivo e Choice de números);
  * a montagem do state (com/sem estado da relação e pendências);
  * a "física" em código (nível -> delta, saturação, decaimento, inércia, histerese, eventos compostos com AND);
  * wrappers de Jev e de LLM com cache próprio (data/processed/c2_*), bootstrap por conversa.
"""
import hashlib, json, math, os, re, sys, time, copy
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
PROC = os.path.join(ROOT, "data", "processed")
ADATA = os.path.join(ROOT, "analysis", "data")

import jev  # noqa: E402
from jev import noul, choice, score  # noqa: E402
jev.CACHE = os.path.join(PROC, "c2_jev_cache.jsonl")   # cache próprio (o global tem 67 MB e é compartilhado)
import b4_llm  # noqa: E402
b4_llm.CACHE = os.path.join(PROC, "c2_llm_cache.jsonl")

MODELS = {"lite": "google/gemini-3.5-flash-lite", "luna": "openai/gpt-6-luna",
          "deepseek": "~deepseek/deepseek-flash-latest", "mercury": "inception/mercury-2.5"}
WORKERS = 4

# ============================================================== taxonomia
DIMS = ["trust", "comfort", "affection", "resentment", "jealousy", "respect", "protectiveness", "playfulness"]
DIM_PT = {"trust": "confiança", "comfort": "conforto/proximidade", "affection": "afeto/interesse romântico",
          "resentment": "ressentimento/irritação", "jealousy": "ciúme", "respect": "respeito/admiração",
          "protectiveness": "proteção/preocupação", "playfulness": "cumplicidade brincalhona"}


def dim_questions(C, U):
    """Jev 1, arquitetura DIM: um Noul por dimensão e direção (16 Nouls)."""
    q = {
        "trust_up": (f"Does `user_message` give {C} a reason to trust {U} more, e.g. {U} keeps their word, is honest, "
                     f"shows they are reliable, or confides in {C}?"),
        "trust_down": (f"Does `user_message` give {C} a reason to trust {U} less, e.g. {U} breaks a promise, cancels, "
                       f"seems to lie or hide something, or is unreliable?"),
        "comfort_up": f"Does `user_message` make {C} feel closer to {U} and more at ease with {U}?",
        "comfort_down": f"Does `user_message` make {C} feel more distant, awkward or uneasy with {U}?",
        "affection_up": f"Does `user_message` make {C} feel more fondness or romantic attraction toward {U}?",
        "affection_down": f"Does `user_message` make {C} feel less fondness or less attraction toward {U}?",
        "resentment_up": f"Does `user_message` hurt, annoy or offend {C}, making {C} more resentful or irritated with {U}?",
        "resentment_down": (f"Does `user_message` ease hurt or irritation that {C} has toward {U}, e.g. through a sincere "
                            f"apology, making amends, or fixing something {U} did wrong?"),
        "jealousy_up": (f"Does `user_message` make {C} feel jealous or threatened that {U}'s attention or interest is "
                        f"going to someone else?"),
        "jealousy_down": (f"Does `user_message` reassure {C} that {U} is not interested in someone else and that {C} "
                          f"has {U}'s attention?"),
        "respect_up": (f"Does `user_message` make {C} respect or admire {U} more, e.g. because of {U}'s competence, "
                       f"integrity, effort or courage?"),
        "respect_down": (f"Does `user_message` make {C} respect {U} less, e.g. because {U} is petty, cruel, cowardly, "
                         f"lazy or dishonest?"),
        "protectiveness_up": f"Does `user_message` make {C} more worried about {U} or want to look after or protect {U}?",
        "protectiveness_down": f"Does `user_message` make {C} less worried about {U}, e.g. {U} shows they are fine or safe now?",
        "playfulness_up": (f"Does `user_message` invite playful banter or an inside joke between {C} and {U}, "
                           f"strengthening their playful complicity?"),
        "playfulness_down": (f"Does `user_message` kill the playful mood between {C} and {U}, e.g. {U} turns cold, "
                             f"rejects a joke or makes things tense?"),
    }
    return {"d_" + k: noul(v) for k, v in q.items()}


BIP_WHAT = {"trust": "how much {C} trusts {U}", "comfort": "how close to and at ease with {U} {C} feels",
            "affection": "how much fondness or romantic attraction {C} feels toward {U}",
            "resentment": "how resentful or irritated {C} is with {U}",
            "jealousy": "how jealous {C} feels about {U}'s attention going to someone else",
            "respect": "how much {C} respects and admires {U}",
            "protectiveness": "how worried {C} is about {U} and how much {C} wants to look after {U}",
            "playfulness": "the playful complicity and banter between {C} and {U}"}
BIP_LEVELS = ["clearly lowers it", "slightly lowers it", "leaves it unchanged", "slightly raises it", "clearly raises it"]


def bip_questions(C, U):
    """Jev 1, arquitetura BIP: um Score bipolar por dimensão (8 Scores, 5 níveis descritivos)."""
    return {"b_" + d: score(f"How does `user_message` change {BIP_WHAT[d].format(C=C, U=U)}?", BIP_LEVELS) for d in DIMS}


EVENTS = ["apology", "hurtful_joke", "insult_criticism", "promise_made", "promise_kept", "cancel_or_broken_promise",
          "absence_explained", "compliment", "vulnerability", "other_person_jealousy", "practical_care", "defensive",
          "perceived_lie", "dismissive", "affection_expr", "friendly_tease", "gratitude", "interest_in_char"]
EVT_PT = {"apology": "pedido de desculpa", "hurtful_joke": "piada de mau gosto", "insult_criticism": "insulto/crítica séria",
          "promise_made": "faz promessa", "promise_kept": "promessa cumprida", "cancel_or_broken_promise": "cancela/quebra promessa",
          "absence_explained": "explica o sumiço", "compliment": "elogio", "vulnerability": "vulnerabilidade",
          "other_person_jealousy": "outra pessoa (ciúme)", "practical_care": "cuidado prático", "defensive": "defensividade",
          "perceived_lie": "mentira percebida", "dismissive": "desdém/ignorar", "affection_expr": "expressa afeto",
          "friendly_tease": "provocação amigável", "gratitude": "gratidão", "interest_in_char": "interesse pelo personagem"}


def event_questions(C, U):
    """Jev 1, arquitetura EVT: um Noul por evento (18 Nouls)."""
    q = {
        "apology": f"{U} apologizes to {C}.",
        "hurtful_joke": (f"{U} makes a joke or teasing remark at {C}'s expense that could genuinely hurt or offend {C} "
                         f"(mean-spirited, in poor taste, or touching a sensitive topic), rather than friendly banter."),
        "insult_criticism": f"{U} seriously criticizes, insults, blames or belittles {C}.",
        "promise_made": f"{U} promises or commits to doing something for or with {C}.",
        "promise_kept": f"{U} reports having done something they had promised or planned to do for or with {C}.",
        "cancel_or_broken_promise": (f"{U} cancels plans with {C}, says they cannot come, or admits failing to do "
                                     f"something they promised {C}."),
        "absence_explained": f"{U} explains or apologizes for having been away or not replying.",
        "compliment": f"{U} compliments, praises or expresses admiration for {C}.",
        "vulnerability": f"{U} reveals something personal, painful or vulnerable about themselves.",
        "other_person_jealousy": (f"{U} mentions spending time with, being interested in, or getting attention from "
                                  f"another person in a way that could make {C} jealous."),
        "practical_care": (f"{U} shows everyday care for {C}: asks if {C} ate, slept or got home safe, wishes {C} "
                           f"luck, or offers practical help."),
        "defensive": f"{U} defends or justifies themselves, makes excuses, or deflects blame.",
        "perceived_lie": (f"{U} gives an excuse that does not add up or says something that contradicts what {U} said "
                          f"earlier, so it could seem dishonest to {C}."),
        "dismissive": f"{U} dismisses, ignores or brushes off {C}'s feelings, question or effort (curt, cold or uninterested).",
        "affection_expr": f"{U} expresses affection or love for {C}, or says they miss {C}.",
        "friendly_tease": f"{U} playfully teases {C} or jokes around with {C} in a friendly way.",
        "gratitude": f"{U} thanks {C} or expresses appreciation for something {C} did.",
        "interest_in_char": f"{U} asks about {C}'s life, day or feelings with genuine interest.",
    }
    q = {k: "Looking only at `user_message` (not at earlier turns), is this true? " + v for k, v in q.items()}
    q["left_waiting"] = (f"In `previous_turns`, did {C}'s last message ask {U} something or need an answer that {U} "
                         f"did not give before `user_message`?")
    return {"e_" + k: noul(v) for k, v in q.items()}


def item_questions(C, U, n_unres, n_prom):
    """Nouls por item (fan-out) para os eventos compostos: a desculpa/reparação tem de se referir a uma pendência."""
    q = {}
    for i in range(n_unres):
        q[f"u_addr_{i}"] = noul(f"Does `user_message` apologize for, make up for, or fix the issue in `unresolved_issues[{i}]`?")
        q[f"u_worse_{i}"] = noul(f"Does `user_message` repeat or make worse the issue in `unresolved_issues[{i}]`?")
    for i in range(n_prom):
        q[f"p_kept_{i}"] = noul(f"Does `user_message` show that {U} did what is described in `open_promises[{i}]`?")
        q[f"p_broke_{i}"] = noul(f"Does `user_message` show that {U} cancels or fails to do what is described in `open_promises[{i}]`?")
    return q


def memory_question(C, U):
    return {"m_remember": noul(f"Is `user_message` something {C} would still remember about {U} a month from now?")}


def jev1_bank(C, U, n_unres=0, n_prom=0, memory=True):
    q = {}
    q.update(dim_questions(C, U)); q.update(bip_questions(C, U)); q.update(event_questions(C, U))
    q.update(item_questions(C, U, n_unres, n_prom))
    if memory:
        q.update(memory_question(C, U))
    return q


# ---------------- mapeamento evento -> dimensões (em código). peso > 0 sobe, < 0 desce.
# "gate": "pending" = só vale se houver pendência (evento composto com AND no código)
EVT_MAP = {
    "apology": [("resentment", -1.0, "pending"), ("trust", +0.3, "pending"), ("comfort", +0.2, None)],
    "hurtful_joke": [("resentment", +1.0, None), ("comfort", -0.5, None), ("playfulness", -0.3, None), ("respect", -0.3, None)],
    "insult_criticism": [("resentment", +1.0, None), ("comfort", -0.7, None), ("affection", -0.4, None),
                         ("respect", -0.3, None), ("trust", -0.3, None)],
    "promise_made": [("trust", +0.1, None)],
    "promise_kept": [("trust", +1.0, None), ("respect", +0.5, None), ("affection", +0.3, None), ("resentment", -0.3, "pending")],
    "cancel_or_broken_promise": [("trust", -1.0, None), ("resentment", +0.8, None), ("comfort", -0.2, None)],
    "absence_explained": [("resentment", -0.3, "pending")],
    "compliment": [("affection", +0.7, None), ("comfort", +0.3, None)],
    "vulnerability": [("comfort", +0.7, None), ("trust", +0.7, None), ("protectiveness", +0.7, None)],
    "other_person_jealousy": [("jealousy", +1.0, None)],
    "practical_care": [("affection", +0.6, None), ("comfort", +0.6, None), ("trust", +0.2, None)],
    "defensive": [("resentment", +0.4, "pending"), ("trust", -0.2, "pending")],
    "perceived_lie": [("trust", -1.0, None), ("resentment", +0.4, None), ("respect", -0.4, None)],
    "dismissive": [("resentment", +0.6, None), ("comfort", -0.5, None), ("affection", -0.3, None)],
    "affection_expr": [("affection", +1.0, None), ("comfort", +0.4, None), ("jealousy", -0.3, None)],
    "friendly_tease": [("playfulness", +1.0, None), ("comfort", +0.3, None)],
    "gratitude": [("affection", +0.3, None), ("respect", +0.2, None), ("comfort", +0.2, None)],
    "interest_in_char": [("comfort", +0.5, None), ("affection", +0.2, None)],
}


def evt_to_dims(ev_p, pending=True):
    """p(dimensão d move na direção dir) pelo noisy-OR ponderado dos eventos mapeados."""
    out = {}
    for d in DIMS:
        for dr, sign in (("up", 1), ("down", -1)):
            prod = 1.0
            for e, lst in EVT_MAP.items():
                for dd, w, gate in lst:
                    if dd == d and np.sign(w) == sign and (gate is None or pending):
                        prod *= 1 - abs(w) * ev_p.get(e, 0.0)
            out[f"{d}_{dr}"] = 1 - prod
    return out


# ============================================================== Jev 2 (magnitude)
MAG_LEVELS = ["not at all", "slightly: a small, passing effect", "moderately: a noticeable effect that lasts a while",
              "strongly: a real effect that would last for days", "profoundly: a major, lasting change in the relationship"]
MAG_PT = ["nada", "leve", "moderado", "forte", "marcante"]
DIR_VERB = {"up": "raise", "down": "lower"}


def jev2_questions(C, U, active):
    """active: lista de (dim, dir). Para cada uma, um Score descritivo e um Choice de números (ideia original)."""
    q = {}
    for d, dr in active:
        what = BIP_WHAT[d].format(C=C, U=U)
        q[f"s_{d}_{dr}"] = score(f"Given the conversation and their relationship, how much does `user_message` "
                                 f"{DIR_VERB[dr]} {what}?", MAG_LEVELS)
        q[f"n_{d}_{dr}"] = choice(f"On a 0-100 scale of {what}, by how many points does `user_message` {DIR_VERB[dr]} it?",
                                  {str(v): f"{v} points" for v in range(0, 101, 10)})
    return q


# ============================================================== state
def bucket(v):
    return ("very low" if v < 0.15 else "low" if v < 0.35 else "moderate" if v < 0.55 else "high" if v < 0.75 else "very high")


def gap_text(hours):
    if hours is None:
        return None
    if hours < 1:
        return "a few minutes"
    if hours < 20:
        return f"about {int(round(hours))} hours"
    d = hours / 24
    return f"about {int(round(d))} day" + ("s" if round(d) != 1 else "")


def make_state(C, U, turns, msg, rel=None, with_state=True, gap_hours=None, ctx=8, extra=None):
    """turns: [{"from","text"}] anteriores; msg: texto do usuário. rel: RelState ou None."""
    st = {"character": C, "user": U}
    if with_state and rel is not None:
        st[f"relationship_of_{C}_toward_{U}"] = {d: bucket(rel.v[d]) for d in DIMS}
        st["unresolved_issues"] = [u["text"] for u in rel.unresolved] or ["none"]
        st["open_promises"] = [p["text"] for p in rel.promises] or ["none"]
        if rel.shared:
            st["shared_history"] = [s for s in rel.shared[-4:]]
        if rel.days_known is not None:
            st["how_long_they_have_known_each_other"] = rel.days_known
    if extra:
        st.update(extra)
    if gap_hours is not None and gap_hours >= 3:
        st["time_since_previous_message"] = gap_text(gap_hours)
    st["previous_turns"] = [{"from": t["from"], "text": t["text"]} for t in turns[-ctx:]]
    st["user_message"] = msg
    return st


def state_as_text(C, U, rel):
    """versão "frases" do estado (para o prompt da LLM, experimento 5)."""
    return rel.sentences(C, U)


# ============================================================== física (código)
SPEC = {
    "delta": [0.0, 0.03, 0.07, 0.15, 0.25],                  # nível (nada..marcante) -> delta
    "rate_up": {"trust": 0.5, "comfort": 0.7, "affection": 0.7, "resentment": 1.0, "jealousy": 1.0,
                "respect": 0.6, "protectiveness": 1.0, "playfulness": 1.2},
    "rate_down": {"trust": 1.5, "comfort": 1.0, "affection": 1.0, "resentment": 0.8, "jealousy": 1.0,
                  "respect": 1.2, "protectiveness": 1.0, "playfulness": 1.2},
    # meia-vida do decaimento em horas, em direção à linha de base (None = não decai)
    "half_life_h": {"trust": None, "comfort": 24 * 14, "affection": 24 * 30, "resentment": 72, "jealousy": 24,
                    "respect": None, "protectiveness": 48, "playfulness": 12},
    "resent_floor_per_sev": 0.08,     # piso do ressentimento enquanto houver pendência: 0.08 x severidade (nível 2-4)
    "apology_repeat_decay": 0.5,      # a n-ésima desculpa pela mesma pendência vale 0.5^(n-1)
    "absence_h": [(24, 1), (60, 2), (24 * 6, 3)],   # sumiço sem explicação: >=24h leve, >=60h moderado, >=6 dias forte
    "detect_thr": 0.5,                # Jev 1: limiar do Noul de direção para aplicar a magnitude
    "evt_thr": 0.5,                   # limiar dos Nouls de evento
    "addr_thr": 0.5,                  # limiar do Noul por pendência (desculpa refere-se à pendência)
    "unres_min_level": 2,             # nível mínimo para virar pendência
    "max_unres": 5,
    "modes": {  # (dimensão, entra, sai, sentido)
        "cold": ("resentment", 0.50, 0.35, ">"),
        "jealous": ("jealousy", 0.45, 0.30, ">"),
        "guarded": ("trust", 0.35, 0.45, "<"),
        "worried": ("protectiveness", 0.70, 0.55, ">"),
        "warm": ("warmth", 0.70, 0.62, ">"),        # warmth = min(trust, comfort)
        "playful": ("playfulness", 0.65, 0.50, ">"),
    },
    "mode_priority": ["cold", "jealous", "guarded", "worried", "warm", "playful"],
    # ---- v1 (anti-deriva; escolhidos no dev, ver relatório): None/False = comportamento v0
    "habituation": False,        # n-ésimo movimento na mesma direção, na mesma sessão (<3 h), vale 1/n
    "mood_decay_msg": 0.0,       # humor (cumplicidade, preocupação, ciúme) volta 10% à base a cada mensagem
    "routine_gate": False,       # subir trust/comfort/affection/playfulness exige nível >= 2 OU um evento concreto
    "routine_thr": 0.5,          # limiar do Noul de direção para as dimensões "de rotina" (sobe)
    # ---- v3 (diagnóstico dos cenários-dev; ver relatório §4c)
    "cell_thr": None,            # {dim_dir: limiar} calibrado no dev real por casamento de taxa (c2_calib.py)
    "delta_interp": None,        # tabela contínua sobre o valor esperado do Score (0..4) -> delta
    "jealousy_rule": False,      # ciúme sobe se evento "outra pessoa" AND afeto >= 0,55 (AND em código)
    "resolve_v3": False,         # regra de resolução de pendência v3d (ver physics_step)
    "resolve_relief": False,     # ao resolver, alívio do ressentimento de nível (severidade - 1)
}
ROUTINE_UP = {"trust": ["promise_kept", "vulnerability", "absence_explained"],
              "comfort": ["vulnerability", "practical_care", "affection_expr", "interest_in_char", "friendly_tease", "gratitude"],
              "affection": ["compliment", "affection_expr", "practical_care"],
              "playfulness": ["friendly_tease"]}
MOOD = ("playfulness", "protectiveness", "jealousy")
DEFAULT_BASE = {"trust": 0.60, "comfort": 0.60, "affection": 0.50, "resentment": 0.10, "jealousy": 0.10,
                "respect": 0.60, "protectiveness": 0.50, "playfulness": 0.50}
EVENT_LABEL = {"hurtful_joke": "made a hurtful joke", "insult_criticism": "insulted or harshly criticized {C}",
               "cancel_or_broken_promise": "cancelled plans / broke a promise", "perceived_lie": "seemed to lie",
               "dismissive": "brushed {C} off", "defensive": "got defensive instead of owning it",
               "absence": "disappeared for {gap} without a word", "resentment_up": "hurt {C}"}


class RelState:
    def __init__(self, base=None, days_known=None, spec=None):
        self.spec = spec or SPEC
        self.base = dict(base or DEFAULT_BASE)
        self.v = dict(self.base)
        self.unresolved = []    # {"text","sev","t","apologies"}
        self.promises = []      # {"text","t"}
        self.shared = []
        self.mode = None
        self.days_known = days_known
        self.t = None           # horas (tempo real do último evento)
        self.session_moves = {}
        self.log = []

    def copy(self):
        return copy.deepcopy(self)

    # ---- decaimento temporal
    def decay(self, t_hours):
        if self.t is None or t_hours is None:
            self.t = t_hours
            return
        dt = max(0.0, t_hours - self.t)
        self.t = t_hours
        if dt >= 3:
            self.session_moves = {}
        if dt <= 0:
            return
        for d, hl in self.spec["half_life_h"].items():
            if not hl:
                continue
            target = self.base[d]
            if d == "resentment" and self.unresolved:
                target = max(target, min(0.6, self.spec["resent_floor_per_sev"] * max(u["sev"] for u in self.unresolved)))
            k = 0.5 ** (dt / hl)
            if d == "resentment" and self.v[d] < target:
                continue  # o piso não faz o ressentimento subir sozinho
            self.v[d] = target + (self.v[d] - target) * k
        # pendências muito antigas esfriam (30 dias)
        self.unresolved = [u for u in self.unresolved if t_hours - u["t"] < 24 * 30]

    def _apply(self, d, sign, level, mult=1.0, cont=None):
        if level <= 0:
            return 0.0
        if cont is not None and self.spec.get("delta_interp"):
            base_delta = float(np.interp(cont, range(5), self.spec["delta_interp"])) * mult
        else:
            base_delta = self.spec["delta"][int(level)] * mult
        v = self.v[d]
        if sign > 0:
            dv = base_delta * self.spec["rate_up"][d] * min(1.0, 2 * (1 - v))
        else:
            dv = -base_delta * self.spec["rate_down"][d] * min(1.0, 2 * v)
        self.v[d] = float(min(1.0, max(0.0, v + dv)))
        return dv

    def warmth(self):
        return min(self.v["trust"], self.v["comfort"])

    def update_mode(self):
        prev = self.mode
        active = []
        for m in self.spec["mode_priority"]:
            dim, enter, leave, sense = self.spec["modes"][m]
            val = self.warmth() if dim == "warmth" else self.v[dim]
            was = m in (self._active_modes if hasattr(self, "_active_modes") else set())
            if sense == ">":
                on = val >= enter or (was and val > leave)
            else:
                on = val <= enter or (was and val < leave)
            if m == "warm" and self.v["resentment"] >= 0.35:
                on = False
            if on:
                active.append(m)
        self._active_modes = set(active)
        self.mode = active[0] if active else "neutral"
        return prev != self.mode

    def sentences(self, C, U):
        """estado como frases (sem números) para o prompt da LLM."""
        v = self.v
        s = []
        s.append(f"{C} {'trusts' if v['trust'] >= 0.55 else 'is not sure she can trust' if v['trust'] >= 0.35 else 'does not trust'} {U}"
                 + (" completely" if v["trust"] >= 0.8 else "") + ".")
        if v["resentment"] >= 0.5:
            s.append(f"{C} is still hurt and irritated with {U}.")
        elif v["resentment"] >= 0.3:
            s.append(f"{C} is a bit sore with {U}.")
        if v["comfort"] >= 0.7:
            s.append(f"{C} feels completely at ease with {U}.")
        elif v["comfort"] < 0.4:
            s.append(f"{C} feels a little distant from {U} right now.")
        if v["affection"] >= 0.7:
            s.append(f"{C} is quite into {U}.")
        if v["jealousy"] >= 0.45:
            s.append(f"{C} feels a sting of jealousy about someone else in {U}'s life.")
        if v["protectiveness"] >= 0.7:
            s.append(f"{C} is worried about {U}.")
        if v["playfulness"] >= 0.65:
            s.append(f"They have an easy, teasing banter going.")
        for u in self.unresolved:
            s.append(f"Unresolved: {u['text']}.")
        return " ".join(s)

    def numbers(self):
        return ", ".join(f"{d} {self.v[d]:.2f}" for d in DIMS) + (
            "; unresolved: " + "; ".join(u["text"] for u in self.unresolved) if self.unresolved else "")


def level_from_score(ans):
    """nível inteiro a partir do Score (limiar sobre o valor esperado, não interpolação)."""
    if not ans:
        return 0
    s = ans.get("score", 0) or 0
    return int(min(4, max(0, math.floor(s + 0.5))))


def level_from_choice(ans):
    if not ans:
        return 0
    v = int(ans["choice"])
    return 0 if v == 0 else 1 if v <= 10 else 2 if v <= 20 else 3 if v <= 40 else 4


def level_from_p(p):
    """magnitude inferida só do Jev 1 (prob. do Noul)."""
    return 0 if p < 0.5 else 1 if p < 0.65 else 2 if p < 0.8 else 3 if p < 0.92 else 4


def excerpt(t, n=70):
    t = re.sub(r"\s+", " ", t or "").strip()
    return t if len(t) <= n else t[:n - 1] + "…"


def physics_step(rel, C, U, msg, a1, a2, t_hours, gap_hours=None, char_waiting=False, mag="score", spec=None):
    """Aplica uma mensagem do usuário ao estado.
    a1: respostas do Jev 1 (com estado); a2: respostas do Jev 2 (magnitudes) ou None.
    mag: 'score' (Jev 2 Score descritivo) | 'choice' (Jev 2 números) | 'p' (só Jev 1).
    Retorna dict de log."""
    spec = spec or rel.spec
    rel.decay(t_hours)
    if spec.get("mood_decay_msg"):
        for d in MOOD:
            rel.v[d] = rel.base[d] + (rel.v[d] - rel.base[d]) * (1 - spec["mood_decay_msg"])
    before = dict(rel.v)
    ev = {e: a1.get("e_" + e, {}).get("noul", 0.0) for e in EVENTS}
    has_pending = bool(rel.unresolved)
    applied = []

    # ---- 1) dimensões: direção pelo Jev 1 (Noul) AND magnitude pelo Jev 2
    for d in DIMS:
        for dr, sign in (("up", 1), ("down", -1)):
            p = a1.get(f"d_{d}_{dr}", {}).get("noul", 0.0)
            thr = spec["routine_thr"] if (dr == "up" and d in ROUTINE_UP) else spec["detect_thr"]
            if spec.get("cell_thr"):
                thr = spec["cell_thr"].get(f"{d}_{dr}", thr)
            forced = False
            if d == "jealousy" and dr == "up" and spec.get("jealousy_rule") and \
                    ev.get("other_person_jealousy", 0) >= 0.7 and rel.v["affection"] >= 0.55:
                forced = True          # AND em código: evento "outra pessoa" + afeto alto
            if p < thr and not forced:
                continue
            cont = None
            if mag == "p" or not a2:
                lvl = level_from_p(max(p, thr if forced else 0))
            elif mag == "choice":
                lvl = level_from_choice(a2.get(f"n_{d}_{dr}"))
            else:
                sa = a2.get(f"s_{d}_{dr}")
                lvl = level_from_score(sa)
                if sa and spec.get("delta_interp"):
                    cont = float(sa.get("score", 0) or 0)
                    lvl = max(lvl, 1) if cont >= 0.5 else lvl
            if forced and lvl == 0:
                lvl, cont = 2, None
            if lvl == 0:
                continue
            if spec.get("routine_gate") and dr == "up" and d in ROUTINE_UP and lvl < 2 and \
                    not any(ev.get(e, 0) >= spec["evt_thr"] for e in ROUTINE_UP[d]):
                continue
            mult = 1.0
            if spec.get("habituation"):
                n = rel.session_moves.get((d, dr), 0) + 1
                rel.session_moves[(d, dr)] = n
                mult /= n
            # "desce" em dimensão de ativação só se houver o que descer (acima da base)
            if dr == "down" and d in ("jealousy", "protectiveness") and rel.v[d] <= rel.base[d] + 0.05:
                continue
            # evento composto: reduzir ressentimento exige pendência (AND em código)
            if d == "resentment" and dr == "down":
                if not has_pending and rel.v["resentment"] <= rel.base["resentment"] + 0.1:
                    continue
                addr = [i for i in range(len(rel.unresolved)) if a1.get(f"u_addr_{i}", {}).get("noul", 0) >= spec["addr_thr"]]
                if has_pending and not addr:
                    mult = 0.5          # alívio genérico, sem tocar na pendência
                elif addr:
                    n_ap = max(rel.unresolved[i]["apologies"] for i in addr)
                    mult = spec["apology_repeat_decay"] ** n_ap
            # ciúme escala com o interesse (não há ciúme sem afeto)
            if d == "jealousy" and dr == "up":
                mult *= 0.5 + rel.v["affection"]
            dv = rel._apply(d, sign, lvl, mult, cont)
            applied.append((d, dr, lvl, round(dv, 4)))

    # ---- 2) sumiço (código: intervalo) AND não explicou (Jev)
    waiting = a1.get("e_left_waiting", {}).get("noul", 0.0) >= spec["evt_thr"]
    if gap_hours is not None and char_waiting and waiting and ev["absence_explained"] < spec["evt_thr"]:
        lvl = 0
        for h, l in spec["absence_h"]:
            if gap_hours >= h:
                lvl = l
        if lvl:
            rel._apply("resentment", +1, lvl)
            rel._apply("comfort", -1, max(1, lvl - 1))
            applied.append(("resentment", "up", lvl, "absence"))
            if lvl >= spec["unres_min_level"]:
                _add_unres(rel, EVENT_LABEL["absence"].format(C=C, gap=gap_text(gap_hours)), lvl, t_hours)

    # ---- 3) pendências: resolver (desculpa/reparação referida à pendência) e criar
    resolved = []
    for i, u in enumerate(list(rel.unresolved)):
        if a1.get(f"u_addr_{i}", {}).get("noul", 0) >= spec["addr_thr"]:
            u["apologies"] += 1
            # resolve se a reparação é forte o bastante: 1a desculpa resolve pendência leve/moderada;
            # pendência forte precisa de reparação concreta (promessa cumprida) ou de duas desculpas
            concrete = ev["promise_kept"] >= spec["evt_thr"]
            if spec.get("resolve_v3"):
                # v3d: reparação concreta inclui explicação crível do sumiço; duas mensagens de reparação bastam
                concrete = concrete or ev["absence_explained"] >= spec["evt_thr"]
                two = u["apologies"] >= 2
            else:
                two = u["apologies"] >= 2 and ev["apology"] >= spec["evt_thr"]
            if u["sev"] <= 2 or concrete or two:
                if not (u.get("repeat_offense", 0) >= 2 and not concrete):   # desculpa repetida sem mudança não resolve
                    resolved.append(u)
        if a1.get(f"u_worse_{i}", {}).get("noul", 0) >= spec["addr_thr"]:
            u["repeat_offense"] = u.get("repeat_offense", 0) + 1
            u["sev"] = min(4, u["sev"] + 1)
    for u in resolved:
        rel.unresolved.remove(u)
        if spec.get("resolve_relief"):
            # fechamento: perdoar alivia o ressentimento de uma vez (nível = severidade - 1)
            rel._apply("resentment", -1, max(1, u["sev"] - 1))
            applied.append(("resentment", "down", max(1, u["sev"] - 1), "resolved"))
        if u["sev"] >= 3:
            rel.shared.append(f"they got past it when {U} made up for: {u['text']}")
    # promessas
    kept_any = False
    for i, pr in enumerate(list(rel.promises)):
        if a1.get(f"p_kept_{i}", {}).get("noul", 0) >= spec["evt_thr"]:
            kept_any = True
            rel.promises.remove(pr)
            if not any(x[0] == "trust" and x[1] == "up" for x in applied):
                rel._apply("trust", +1, 2); applied.append(("trust", "up", 2, "promise_kept_item"))
        elif a1.get(f"p_broke_{i}", {}).get("noul", 0) >= spec["evt_thr"]:
            rel.promises.remove(pr)
            _add_unres(rel, f"{U} broke a promise ({pr['text']})", 3, t_hours)
            if not any(x[0] == "trust" and x[1] == "down" for x in applied):
                rel._apply("trust", -1, 3); applied.append(("trust", "down", 3, "promise_broken_item"))
    if ev["promise_made"] >= spec["evt_thr"] and ev["cancel_or_broken_promise"] < spec["evt_thr"]:
        rel.promises.append({"text": excerpt(msg), "t": t_hours})
        rel.promises = rel.promises[-3:]
    # nova pendência: ressentimento subiu com nível >= 2, rotulada pelo evento mais provável
    res_up = [x for x in applied if x[0] == "resentment" and x[1] == "up" and x[3] != "absence"]
    if res_up and max(x[2] for x in res_up) >= spec["unres_min_level"]:
        cands = ["cancel_or_broken_promise", "hurtful_joke", "insult_criticism", "perceived_lie", "dismissive", "defensive"]
        best = max(cands, key=lambda e: ev[e])
        lab = EVENT_LABEL[best] if ev[best] >= 0.3 else EVENT_LABEL["resentment_up"]
        txt = f"{U} {lab.format(C=C)}: \"{excerpt(msg, 60)}\""
        # não duplica uma pendência que a mensagem só repetiu
        if not any(a1.get(f"u_worse_{i}", {}).get("noul", 0) >= spec["addr_thr"] for i in range(len(rel.unresolved))):
            _add_unres(rel, txt, max(x[2] for x in res_up), t_hours)
    # memória compartilhada (positiva, marcante)
    if a1.get("m_remember", {}).get("noul", 0) >= 0.7 and not res_up:
        pos = [x for x in applied if x[1] == "up" and x[0] in ("affection", "comfort", "trust", "playfulness") and x[2] >= 2]
        if pos:
            rel.shared.append(excerpt(msg, 60)); rel.shared = rel.shared[-8:]
    changed = rel.update_mode()
    rec = {"t": t_hours, "before": before, "after": dict(rel.v), "applied": applied, "mode": rel.mode, "mode_changed": changed,
           "unresolved": [u["text"] for u in rel.unresolved], "promises": [p["text"] for p in rel.promises],
           "resolved": [u["text"] for u in resolved]}
    rel.log.append(rec)
    return rec


def _add_unres(rel, text, sev, t):
    rel.unresolved.append({"text": text, "sev": int(sev), "t": t, "apologies": 0})
    rel.unresolved = rel.unresolved[-rel.spec["max_unres"]:]


def active_dims(a1, thr=0.3):
    return [(d, dr) for d in DIMS for dr in ("up", "down") if a1.get(f"d_{d}_{dr}", {}).get("noul", 0) >= thr]


# ============================================================== utilidades
RNG = np.random.default_rng(7)


def cboot(vals, groups, stat=np.mean, iters=2000, seed=7):
    rng = np.random.default_rng(seed)
    keep = [k for k, v in enumerate(vals) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not keep:
        return None
    v0 = np.asarray([vals[k] for k in keep], float)
    g0 = np.asarray([groups[k] for k in keep])
    ug = np.unique(g0)
    idx = {g: np.where(g0 == g)[0] for g in ug}
    bs = []
    for _ in range(iters):
        s = rng.choice(ug, len(ug))
        bs.append(stat(np.concatenate([v0[idx[g]] for g in s])))
    return [round(float(stat(v0)), 4), round(float(np.nanpercentile(bs, 2.5)), 4),
            round(float(np.nanpercentile(bs, 97.5)), 4), int(len(v0))]


def boot_fn(fn, rows, groups, iters=1000, seed=7):
    """IC por bootstrap por conversa para uma estatística arbitrária fn(rows)."""
    rng = np.random.default_rng(seed)
    g0 = np.asarray(groups)
    ug = np.unique(g0)
    idx = {g: np.where(g0 == g)[0] for g in ug}
    pt = fn(rows)
    bs = []
    for _ in range(iters):
        s = rng.choice(ug, len(ug))
        sub = [rows[i] for g in s for i in idx[g]]
        try:
            v = fn(sub)
            if v is not None and not np.isnan(v):
                bs.append(v)
        except Exception:
            pass
    if pt is None or not bs:
        return [pt, None, None]
    return [round(float(pt), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    # empates: média dos ranks
    from scipy.stats import rankdata
    r = rankdata(allv)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def f1(y, yhat):
    y = np.asarray(y, int); yhat = np.asarray(yhat, int)
    tp = int(((y == 1) & (yhat == 1)).sum()); fp = int(((y == 0) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def jl_append(path, recs):
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jl_load(path):
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path, encoding="utf-8"):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def llm(messages, model="luna", **kw):
    return b4_llm.chat(messages, model=MODELS.get(model, model), **kw)


def parse_json(t):
    if not t:
        return None
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except Exception:
            return None
