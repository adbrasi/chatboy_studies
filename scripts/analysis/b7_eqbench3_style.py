"""b7: análise dos dados canônicos do EQ-Bench 3 (github.com/EQ-bench/eqbench3, data/canonical_*.json.gz).
Por modelo: médias dos 18 critérios da rubrica, Elo normalizado e tamanho da seção "My response".
Correlações (Spearman) entre modelos e dentro de tarefas (humanlike x tamanho etc.).
Uso: python b7_eqbench3_style.py <dir_eqbench3_data>
"""
import gzip, json, sys, statistics, collections, re
from pathlib import Path

def spearman(x, y):
    import math
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v); i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[s[j+1]] == v[s[i]]: j += 1
            for k in range(i, j+1): r[s[k]] = (i+j)/2
            i = j+1
        return r
    rx, ry = rank(x), rank(y); mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else float('nan')

D = Path(sys.argv[1])
runs = json.load(gzip.open(D/'canonical_leaderboard_results.json.gz'))
elo = json.load(gzip.open(D/'canonical_leaderboard_elo_results.json.gz'))
CRIT = [l.strip() for l in open(D/'rubric_scoring_criteria.txt') if l.strip()]
rows = []; within = collections.defaultdict(list)
for rk, r in runs.items():
    if not isinstance(r, dict) or 'scenario_tasks' not in r: continue
    m = r['model_name']; e = elo.get(m, {})
    agg = collections.defaultdict(list); rlen = []; words = []
    for it, sc in r['scenario_tasks'].items():
        for sid, t in sc.items():
            rs = t.get('rubric_scores')
            if not rs or 'humanlike' not in rs: continue   # só tarefas de roleplay (critérios padrão)
            for c in CRIT:
                if isinstance(rs.get(c), (int, float)): agg[c].append(rs[c])
            pr = [p.get('response', '') for p in (t.get('parsed_responses') or []) if isinstance(p, dict)]
            L = statistics.mean([len(x) for x in pr]) if pr else None
            if L:
                rlen.append(L)
                within[sid].append((rs['humanlike'], rs.get('conversational'), L, rs.get('analytical'), rs.get('demonstrated_empathy')))
    if not agg: continue
    row = {'model': m, 'judge': r.get('judge_model'), 'elo_norm': e.get('elo_norm'),
           'rubric_avg': (r.get('results') or {}).get('average_rubric_score'),
           'n_tasks': len(agg['humanlike']), 'resp_chars': statistics.mean(rlen) if rlen else None}
    for c in CRIT: row[c] = round(statistics.mean(agg[c]), 2) if agg[c] else None
    rows.append(row)

ok = [x for x in rows if x['elo_norm'] is not None and x['resp_chars']]
out = {'n_models': len(ok), 'judges': dict(collections.Counter(x['judge'] for x in ok)), 'between_models_spearman': {}}
for c in CRIT + ['resp_chars']:
    out['between_models_spearman'][f'elo_norm~{c}'] = round(spearman([x['elo_norm'] for x in ok], [x[c] for x in ok]), 3)
for a, b in [('humanlike', 'resp_chars'), ('conversational', 'resp_chars'), ('humanlike', 'analytical'),
             ('humanlike', 'conversational'), ('humanlike', 'demonstrated_empathy'), ('humanlike', 'moralising'),
             ('humanlike', 'sycophantic'), ('humanlike', 'validating'), ('humanlike', 'warmth'), ('humanlike', 'safety_conscious')]:
    out['between_models_spearman'][f'{a}~{b}'] = round(spearman([x[a] for x in ok], [x[b] for x in ok]), 3)
# dentro de cenário: humanlike x tamanho da resposta (centrado por cenário)
w = {'humanlike~len': [], 'conversational~len': [], 'humanlike~analytical': [], 'empathy~len': []}
for sid, v in within.items():
    if len(v) < 10: continue
    w['humanlike~len'].append(spearman([a[0] for a in v], [a[2] for a in v]))
    w['conversational~len'].append(spearman([a[1] for a in v], [a[2] for a in v]))
    w['humanlike~analytical'].append(spearman([a[0] for a in v], [a[3] for a in v]))
    w['empathy~len'].append(spearman([a[4] for a in v], [a[2] for a in v]))
out['within_scenario_mean_spearman'] = {k: round(statistics.mean(v), 3) for k, v in w.items() if v}
out['n_scenarios_within'] = len(w['humanlike~len'])
ok.sort(key=lambda x: -x['elo_norm'])
out['top10_by_elo'] = [{k: x[k] for k in ('model', 'judge', 'elo_norm', 'rubric_avg', 'humanlike', 'conversational', 'analytical', 'moralising', 'sycophantic', 'warmth', 'validating', 'resp_chars')} for x in ok[:10]]
out['bottom5_by_elo'] = [{k: x[k] for k in ('model', 'elo_norm', 'humanlike', 'conversational', 'analytical', 'resp_chars')} for x in ok[-5:]]
out['top10_by_humanlike'] = [{k: x[k] for k in ('model', 'elo_norm', 'humanlike', 'conversational', 'analytical', 'resp_chars')} for x in sorted(ok, key=lambda x: -x['humanlike'])[:10]]
out['all_models'] = ok
json.dump(out, open('/home/user/chatboy_studies/analysis/data/b7_eqbench3_style.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != 'all_models'}, indent=1, ensure_ascii=False))
