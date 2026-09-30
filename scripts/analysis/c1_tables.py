"""c1 — tabelas em Markdown para o relatório 17, a partir dos JSONs em analysis/data/c1_*.json.
Uso: python3 c1_tables.py schema <split> | header <split> | vs <split> | pipe <split> | constr"""
import json, os, sys

ADATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis", "data")
MODELS = ["lite", "luna", "deepseek", "mercury"]


def J(name):
    p = os.path.join(ADATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


def v(o, k, pct=False, nd=2, ci=False):
    x = o.get(k)
    if x is None:
        return "–"
    m = 100 if pct else 1
    if isinstance(x, list):
        if ci:
            return f"{x[0] * m:.{nd}f} [{x[1] * m:.{nd}f}–{x[2] * m:.{nd}f}]"
        x = x[0]
    return f"{x * m:.{nd if not pct else 0}f}"


def schema(split):
    d = J(f"c1_schema_{split}.json")
    h = d["human"]
    print(f"Humano ({split}, n={h['n']}): palavras med {h['words_med']:.0f}, ? {h['q']:.0%}, bolhas {h['bubbles_mean']:.2f}, "
          f"latência med {h['lat_med']}s, banco P(LLM) {h.get('bank') or 0:.2f}, coerência {h.get('coh') or 0:.2f}\n")
    for note in ("0", "1", "2"):
        conds = [c for c in d["models"]["lite"] if c.endswith("|" + note)]
        if not conds:
            continue
        print(f"\n**Nota = {note}** — D [IC95] / inventa usuário % / quebrado % / solto % / bolhas médias / movimento = humano / banco P(LLM)\n")
        print("| schema | " + " | ".join(MODELS) + " |")
        print("|---|" + "---|" * len(MODELS))
        for c in conds:
            row = []
            for m in MODELS:
                o = d["models"].get(m, {}).get(c)
                if not o:
                    row.append("–"); continue
                row.append(f"{v(o, 'D_ci', ci=True)} · inv {v(o, 'invented_user', True)} · qbr {v(o, 'broken', True)} · "
                           f"solto {v(o, 'plain', True)} · b {v(o, 'bubbles_mean')} · mov {v(o, 'move_match', True)} · "
                           f"banco {v(o, 'bank')}")
            print(f"| {c} | " + " | ".join(row) + " |")
    print("\n**Tempos propostos (sem nota):** latência mediana proposta / real, ρ de Spearman, erro |log2|, intervalo mediano proposto\n")
    print("| schema | " + " | ".join(MODELS) + " |")
    print("|---|" + "---|" * len(MODELS))
    for c in ("wa_full|0", "wa_time|0", "messenger|0", "snapchat|0"):
        row = []
        for m in MODELS:
            o = d["models"].get(m, {}).get(c) or {}
            if o.get("prop_lat_med") is None:
                row.append("–"); continue
            row.append(f"{o['prop_lat_med']:.0f}s/{o['real_lat_med']:.0f}s · ρ {o.get('lat_spearman', 0):.2f} · err {o['lat_logerr']:.2f}"
                       f" · gap {o.get('prop_gap_med') or 0:.0f}s · cob {o['prop_lat_share']:.0%}")
        print(f"| {c} | " + " | ".join(row) + " |")


def generic(fname, conds=None, keys=("D_ci", "move_match", "coh", "bank")):
    d = J(fname)
    for m in MODELS:
        mm = d["models"].get(m)
        if not mm:
            continue
        print(f"\n*{m}*\n")
        print("| condição | D [IC95] | palavras med | ? % | ! % | clichê % | multi % | mov = humano % | coerência | banco P(LLM) | custo US$/resp | lat. p50 s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c, o in mm.items():
            if conds and c not in conds:
                continue
            print(f"| {c} | {v(o, 'D_ci', ci=True)} | {v(o, 'words_med', nd=0)} | {v(o, 'q', True)} | {v(o, 'excl', True)} | "
                  f"{v(o, 'llmish', True)} | {v(o, 'multi', True)} | {v(o, 'move_match', True)} | {v(o, 'coh')} | {v(o, 'bank', ci=True)} | "
                  f"{(o.get('cost_per_resp') or 0) * 1e3:.3f}e-3 | {v(o, 'lat_p50', nd=1)} |")


def vs(split):
    d = J(f"c1_vs_{split}.json")
    print("**Diversidade e formato das candidatas**\n")
    print("| condição | " + " | ".join(MODELS) + " |")
    print("|---|" + "---|" * 4)
    conds = list(d["models"]["lite"])
    for c in conds:
        row = []
        for m in MODELS:
            x = d["models"].get(m, {}).get(c)
            if not x:
                row.append("–"); continue
            q = x["diversity"]
            row.append(f"JSON {q['json_ok'] if q['json_ok'] is not None else float('nan'):.0%} · cand {q['n_cands']:.1f} · distintas "
                       f"{q['distinct_frac'] or 0:.0%} · mov≠ {q['distinct_moves'] or 0:.1f} · p_top {q['p_top_mean'] or 0:.2f}")
        print(f"| {c} | " + " | ".join(row) + " |")
    print("\n**D por estratégia** (IC95 por bootstrap por conversa) · banco P(LLM) · movimento = humano\n")
    strats = ["top", "probw", "rand", "filt_top", "lowpass", "viol", "bank", "bank_pass"]
    for m in MODELS:
        print(f"\n*{m}*\n")
        print("| condição | " + " | ".join(strats) + " |")
        print("|---|" + "---|" * len(strats))
        for c, x in d["models"].get(m, {}).items():
            row = []
            for s in strats:
                o = x["strats"].get(s)
                row.append("–" if not o else f"{v(o, 'D')} · {v(o, 'bank')} · {v(o, 'move_match', True)}%")
            print(f"| {c} | " + " | ".join(row) + " |")


def constr():
    d = J("c1_constraints.json")
    for sp in ("test", "dev"):
        print(f"\n**{sp}**\n")
        print("| ator | formato | n | violação alvo % [IC] | viol. outras % | menção indireta % | esquiva % | recusa % | meta % | frio % | coerência | palavras med | D_ref |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m in MODELS:
            for f in ("NONE", "IMP", "ID"):
                o = d["by_split"][sp].get(m, {}).get(f)
                if not o:
                    continue
                print(f"| {m} | {f} | {o['n']} | {v(o, 'viol_target', True, ci=True)} | {v(o, 'viol_other', True)} | "
                      f"{v(o, 'indirect_topic', True)} | {v(o, 'dodge', True)} | {v(o, 'refuse', True)} | {v(o, 'meta', True)} | "
                      f"{v(o, 'cold', True)} | {v(o, 'coh')} | {o.get('words_med', 0):.0f} | {o.get('D_ref', 0):.2f} |")
    print("\n", d.get("topic_vs_behavior_test"), d.get("manual_check"))


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "schema":
        schema(a[2])
    elif a[1] == "header":
        generic(f"c1_header_{a[2]}.json")
    elif a[1] == "vs":
        vs(a[2])
    elif a[1] == "pipe":
        generic(f"c1_pipeline_{a[2]}.json", a[3].split(",") if len(a) > 3 else None)
    elif a[1] == "constr":
        constr()
