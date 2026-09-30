"""b7: superfície das respostas no EQ-Bench 4 (eqbench.com/eqbench4, transcrições públicas).
Baixa N transcrições por modelo (cache em scratchpad) e mede, por turno do modelo avaliado e do persona (usuário simulado):
chars, % com '?', % terminando em '?', % com '!', travessão, parágrafos; correlaciona com Elo e 'naturalness'.
Uso: python b7_eqbench4_surface.py <cache_dir> [n_por_modelo]
"""
import json, sys, os, re, statistics, urllib.request, concurrent.futures as cf
from pathlib import Path
CACHE = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
BASE = "https://eqbench.com/eqbench4/"
UA = {"User-Agent": "curl/8.5.0"}
def get(url, path):
    if path.exists(): return json.load(open(path))
    import time
    for att in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r: data = r.read()
            break
        except Exception:
            if att == 5: raise
            time.sleep(2 + 3*att)
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data); return json.loads(data)
js = urllib.request.urlopen(urllib.request.Request(BASE + "eqbench4_data.js", headers=UA), timeout=90).read().decode()
D = json.loads(js[js.index('{'):js.rindex('}') + 1])
def spearman(x, y):
    def r(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); o = [0]*len(v)
        for i, j in enumerate(s): o[j] = i
        return o
    a, b = r(x), r(y); ma, mb = statistics.mean(a), statistics.mean(b)
    return sum((p-ma)*(q-mb) for p, q in zip(a, b)) / (sum((p-ma)**2 for p in a)*sum((q-mb)**2 for q in b))**.5
def feats(t):
    t = t.strip()
    return dict(chars=len(t), q=('?' in t), endq=t.endswith('?'), excl=('!' in t), dash=('—' in t),
                paras=t.count('\n\n') + 1, words=len(t.split()))
rows = []
def work(m):
    slug = m['transcript_dir'].split('/')[-1]
    idx = get(f"{BASE}eqbench4_docs/{m['transcript_dir']}/index.json", CACHE/slug/'index.json')
    A, U = [], []
    for tr in idx['transcripts'][:N]:
        d = get(f"{BASE}eqbench4_docs/{m['transcript_dir']}/{tr['file']}", CACHE/slug/tr['file'])
        for turn in d['turns']:
            (A if turn['role'] == 'assistant' else U).append(feats(turn['content']))
    agg = lambda L, k: statistics.median([x[k] for x in L]) if k in ('chars', 'words', 'paras') else round(100*statistics.mean([x[k] for x in L]), 1)
    return dict(model=m['model'], elo=m['elo'], naturalness=m['dims']['naturalness_authenticity'],
                validating=m['dims']['validation_propensity'], analytical=m['dims']['analytical'], n_turns=len(A),
                bot={k: agg(A, k) for k in A[0]}, persona_user={k: agg(U, k) for k in U[0]})
with cf.ThreadPoolExecutor(2) as ex:
    rows = list(ex.map(work, D['models']))
rows.sort(key=lambda r: -r['elo'])
out = {'n_per_model': N, 'models': rows, 'spearman': {}}
for k in ('chars', 'endq', 'q', 'excl', 'dash', 'paras'):
    out['spearman'][f'elo~bot_{k}'] = round(spearman([r['elo'] for r in rows], [r['bot'][k] for r in rows]), 3)
    out['spearman'][f'naturalness~bot_{k}'] = round(spearman([r['naturalness'] for r in rows], [r['bot'][k] for r in rows]), 3)
allU = [r['persona_user']['chars'] for r in rows]; allB = [r['bot']['chars'] for r in rows]
out['pooled'] = {'bot_chars_median_of_model_medians': statistics.median(allB), 'bot_chars_range': [min(allB), max(allB)],
                 'persona_chars_median_of_model_medians': statistics.median(allU),
                 'bot_endq_range': [min(r['bot']['endq'] for r in rows), max(r['bot']['endq'] for r in rows)],
                 'bot_excl_range': [min(r['bot']['excl'] for r in rows), max(r['bot']['excl'] for r in rows)],
                 'persona_endq_median': statistics.median(r['persona_user']['endq'] for r in rows),
                 'persona_excl_median': statistics.median(r['persona_user']['excl'] for r in rows)}
json.dump(out, open('/home/user/chatboy_studies/analysis/data/b7_eqbench4_surface.json', 'w'), indent=1)
for r in rows: print(f"{r['model'][:32]:32s} elo={r['elo']:.0f} nat={r['naturalness']:.2f} bot_chars={r['bot']['chars']:.0f} endq={r['bot']['endq']} excl={r['bot']['excl']} dash={r['bot']['dash']} paras={r['bot']['paras']} | user_chars={r['persona_user']['chars']:.0f} u_endq={r['persona_user']['endq']}")
print(json.dumps(out['spearman'], indent=0)); print(out['pooled'])
