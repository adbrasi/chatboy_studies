"""b7: correlações (Spearman, entre modelos) no leaderboard Creative Writing v3 (eqbench.com/creative_writing.js):
Elo × rubrica, tamanho médio, vocab complexity, slop score, repetição. Uso: python b7_cwv3_corr.py <creative_writing.js>"""
import re, csv, io, statistics, json, sys
t = open(sys.argv[1]).read()
m = re.search(r'leaderboardDataCreativeWritingV3 = `(.*?)`', t, re.S)
rows = [r for r in csv.DictReader(io.StringIO(m.group(1).strip())) if r['elo_score'] and r['slop_score']]
def sp(x, y):
    def r(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); o = [0]*len(v)
        for i, j in enumerate(s): o[j] = i
        return o
    a, b = r(x), r(y); ma, mb = statistics.mean(a), statistics.mean(b)
    return sum((p-ma)*(q-mb) for p, q in zip(a, b)) / (sum((p-ma)**2 for p in a)*sum((q-mb)**2 for q in b))**.5
E = [float(r['elo_score']) for r in rows]
out = {'n_models': len(rows)}
for k in ['creative_writing_score', 'avg_length', 'vocab_complexity', 'slop_score', 'repetition_score']:
    out['elo~'+k] = round(sp(E, [float(r[k]) for r in rows]), 3)
out['rubric~length'] = round(sp([float(r['creative_writing_score']) for r in rows], [float(r['avg_length']) for r in rows]), 3)
out['rubric~slop'] = round(sp([float(r['creative_writing_score']) for r in rows], [float(r['slop_score']) for r in rows]), 3)
json.dump(out, open('/home/user/chatboy_studies/analysis/data/b7_cwv3_correlations.json', 'w'), indent=1); print(out)
