"""b6: checagem rápida (sem Jev, sem LLM) de duas perguntas da revisão de literatura.

1) "5 mensagens em menos de 1 minuto é normal?" Frequência de rajadas rápidas (>= 5 bolhas com span <= 60 s)
   no maichat (timestamps em ms) e no whatsapp_nl (resolução de minuto: span <= 60 s = mesma ou minuto seguinte),
   e o que a camada base do Jev (passe D, já anotada) diz desses turnos comparados a turnos de tamanho parecido.
2) A regra de Kalman et al. (2006): >= 70% das respostas dentro da latência média do próprio respondente e
   <= 4% depois de 10x essa média. Latências só dentro da sessão (corte em 3 h), então a cauda está truncada.

Saída: analysis/data/b6_rafagas.json
"""
import json, math, random, collections, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TURNS = ROOT / "data/processed/turns.jsonl"
BASE = ROOT / "data/processed/jev_base.jsonl"
OUT = ROOT / "analysis/data/b6_rafagas.json"
CORPORA = ("maichat", "whatsapp_nl")
random.seed(7)


def load_turns():
    out = collections.defaultdict(list)
    with open(TURNS) as f:
        for line in f:
            t = json.loads(line)
            if t["corpus"] in CORPORA:
                out[t["corpus"]].append(t)
    return out


def load_base():
    d = {}
    with open(BASE) as f:
        for line in f:
            r = json.loads(line)
            if r["corpus"] in CORPORA and r.get("D"):
                d[(r["corpus"], r["conv_id"], r["turn_idx"])] = r["D"]
    return d


def val(D, k):
    x = D.get(k)
    if x is None:
        return None
    if x.get("type") == "noul":
        return x["noul"]
    if "score" in x:
        return x["score"]
    return x.get("choice")


def is_fast_burst(t):
    return t["n_msgs"] >= 5 and t["burst_span_s"] is not None and t["burst_span_s"] <= 60


def boot_ci_diff(a_by_conv, b_by_conv, n=2000):
    """IC por bootstrap por conversa para média(a) - média(b)."""
    convs = sorted(set(a_by_conv) | set(b_by_conv))
    diffs = []
    for _ in range(n):
        smp = [random.choice(convs) for _ in convs]
        a = [x for c in smp for x in a_by_conv.get(c, [])]
        b = [x for c in smp for x in b_by_conv.get(c, [])]
        if a and b:
            diffs.append(st.mean(a) - st.mean(b))
    diffs.sort()
    return [round(diffs[int(0.025 * len(diffs))], 3), round(diffs[int(0.975 * len(diffs)) - 1], 3)]


def main():
    turns = load_turns()
    base = load_base()
    res = {}
    for corpus, ts in turns.items():
        r = {}
        n = len(ts)
        fast = [t for t in ts if is_fast_burst(t)]
        ge5 = [t for t in ts if t["n_msgs"] >= 5]
        r["n_turns"] = n
        r["pct_turns_ge5_bubbles"] = round(100 * len(ge5) / n, 2)
        r["pct_turns_fast_burst_ge5_le60s"] = round(100 * len(fast) / n, 2)
        # quantos falantes fazem isso
        by_spk = collections.defaultdict(lambda: [0, 0])
        for t in ts:
            k = (t["conv_id"], t["speaker"])
            by_spk[k][0] += 1
            by_spk[k][1] += is_fast_burst(t)
        spk_rates = [b / a for a, b in by_spk.values() if a >= 30]
        r["n_speakers_ge30_turns"] = len(spk_rates)
        r["pct_speakers_with_any_fast_burst"] = round(100 * sum(1 for x in spk_rates if x > 0) / max(1, len(spk_rates)), 1)
        r["speaker_rate_p50_p90_max_pct"] = [round(100 * x, 2) for x in (
            st.median(spk_rates), sorted(spk_rates)[int(0.9 * len(spk_rates))], max(spk_rates))] if spk_rates else None
        # top-5 falantes concentram quanto das rajadas?
        cnt = sorted((b for a, b in by_spk.values()), reverse=True)
        r["share_of_fast_bursts_top5_speakers_pct"] = round(100 * sum(cnt[:5]) / max(1, sum(cnt)), 1)
        # comparação com a camada Jev D: rajada rápida x turnos com 1-4 bolhas no MESMO estrato de tamanho
        labeled = [(t, base.get((corpus, t["conv_id"], t["turn_idx"]))) for t in ts]
        labeled = [(t, D) for t, D in labeled if D]
        fastL = [(t, D) for t, D in labeled if is_fast_burst(t)]
        r["n_fast_bursts_with_jev_D"] = len(fastL)
        # estratos de tamanho (total_chars) a partir das rajadas
        def stratum(c):
            return min(4, int(math.log2(max(c, 1)) // 1) - 4) if c >= 16 else -1
        strata = collections.Counter(stratum(t["total_chars"]) for t, _ in fastL)
        ctrl = [(t, D) for t, D in labeled if (not is_fast_burst(t)) and t["n_msgs"] <= 4 and stratum(t["total_chars"]) in strata]
        # reamostra controles com a mesma distribuição de estratos (pesos)
        by_str = collections.defaultdict(list)
        for t, D in ctrl:
            by_str[stratum(t["total_chars"])].append((t, D))
        comp = {}
        for k in ("arousal", "anxious", "tension", "vulnerable", "playful", "seriousness", "valence", "engagement", "flirting", "seeks_support"):
            a_by, b_by = collections.defaultdict(list), collections.defaultdict(list)
            for t, D in fastL:
                v = val(D, k)
                if isinstance(v, (int, float)):
                    a_by[t["conv_id"]].append(v)
            # controle ponderado: média por estrato, ponderada pela freq do estrato nas rajadas
            b_means = []
            for s_, w in strata.items():
                vs = [val(D, k) for t, D in by_str.get(s_, []) if isinstance(val(D, k), (int, float))]
                if vs:
                    b_means.append((w, st.mean(vs)))
                for t, D in by_str.get(s_, []):
                    v = val(D, k)
                    if isinstance(v, (int, float)):
                        b_by[t["conv_id"]].append(v)
            a_all = [x for v in a_by.values() for x in v]
            if not a_all or not b_means:
                continue
            ctrl_w = sum(w * m for w, m in b_means) / sum(w for w, _ in b_means)
            comp[k] = {"fast_burst": round(st.mean(a_all), 3), "ctrl_same_length_weighted": round(ctrl_w, 3),
                       "ci95_diff_unweighted_boot_conv": boot_ci_diff(a_by, b_by, n=1000)}
        r["jev_D_fast_vs_ctrl"] = comp
        # emoção / intenção / fase dominantes
        for k in ("emotion", "intent", "phase"):
            ca = collections.Counter(val(D, k) for t, D in fastL)
            cb = collections.Counter()
            for s_, w in strata.items():
                sub = collections.Counter(val(D, k) for t, D in by_str.get(s_, []))
                tot = sum(sub.values()) or 1
                for lab, c in sub.items():
                    cb[lab] += w * c / tot
            tb = sum(cb.values()) or 1
            ta = sum(ca.values()) or 1
            r[f"{k}_dist_fast_vs_ctrl_pct"] = {lab: [round(100 * ca[lab] / ta, 1), round(100 * cb[lab] / tb, 1)]
                                               for lab in sorted(set(ca) | set(cb), key=lambda x: -ca[x])[:8]}
        # latência: a rajada começa mais rápido/lento? e o parceiro responde como?
        idx = {(t["conv_id"], t["turn_idx"]): t for t in ts}
        lat_fast = [t["response_latency_s"] for t in fast if t["response_latency_s"] is not None]
        lat_all = [t["response_latency_s"] for t in ts if t["response_latency_s"] is not None]
        nxt_fast = [idx[(t["conv_id"], t["turn_idx"] + 1)]["response_latency_s"] for t in fast
                    if (t["conv_id"], t["turn_idx"] + 1) in idx and idx[(t["conv_id"], t["turn_idx"] + 1)]["response_latency_s"] is not None]
        r["latency_before_fast_burst_median_s"] = round(st.median(lat_fast), 1) if lat_fast else None
        r["latency_all_turns_median_s"] = round(st.median(lat_all), 1)
        r["partner_latency_after_fast_burst_median_s"] = round(st.median(nxt_fast), 1) if nxt_fast else None
        # exemplos curtos
        ex = random.sample(fastL, min(6, len(fastL)))
        r["examples"] = [{"texts": [x[:60] for x in t["texts"][:7]], "span_s": t["burst_span_s"],
                          "emotion": val(D, "emotion"), "arousal": val(D, "arousal"), "anxious": val(D, "anxious"),
                          "tension": val(D, "tension")} for t, D in ex]
        # 2) Kalman: por falante, fração <= média própria e > 10x média própria
        per = collections.defaultdict(list)
        for t in ts:
            if t["response_latency_s"] is not None and t["response_latency_s"] >= 0:
                per[(t["conv_id"], t["speaker"])].append(t["response_latency_s"])
        within, beyond = [], []
        agg = []
        for v in per.values():
            if len(v) < 30:
                continue
            m = st.mean(v)
            if m <= 0:
                continue
            within.append(sum(1 for x in v if x <= m) / len(v))
            beyond.append(sum(1 for x in v if x > 10 * m) / len(v))
            agg += v
        mA = st.mean(agg)
        r["kalman"] = {"n_speakers": len(within),
                       "median_pct_within_own_mean": round(100 * st.median(within), 1),
                       "pct_speakers_with_ge70pct_within_mean": round(100 * sum(1 for x in within if x >= 0.7) / len(within), 1),
                       "median_pct_beyond_10x_mean": round(100 * st.median(beyond), 2),
                       "pct_speakers_with_le4pct_beyond_10x": round(100 * sum(1 for x in beyond if x <= 0.04) / len(beyond), 1),
                       "aggregate_mean_s": round(mA, 1),
                       "aggregate_pct_within_mean": round(100 * sum(1 for x in agg if x <= mA) / len(agg), 1),
                       "aggregate_pct_beyond_10x": round(100 * sum(1 for x in agg if x > 10 * mA) / len(agg), 2),
                       "note": "latências só dentro da sessão (sessão nova após 3 h): a cauda longa está truncada"}
        res[corpus] = r
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
