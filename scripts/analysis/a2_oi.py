"""a2: what people answer to a bare greeting ('oi'): NPS addressed greets + maichat/whatsapp session openings."""
import pandas as pd, numpy as np, re, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from a2_sessions import GREET
import a2_load as L
S = L.SCR
EL = re.compile(r'([a-zA-Z])\1{2,}')
Q = re.compile(r'\?|how (are|r)|whats|what.?s up|wbu|hoe (gaat|is|staat)|was ist|\bwhat\b|\bsup\b', re.I)
el = lambda t: bool(EL.search(str(t)))
out = {}
d = pd.read_pickle(f"{S}/nps_posts.pkl")
g = d[(d.act == 'Greet') & d.resp_named].copy()
g['el'] = g.text.map(el); g['r_el'] = g.resp_first_text.map(el)
g['r_greet'] = g.resp_first_act == 'Greet'
g['r_q'] = g.resp_first_text.map(lambda t: bool(Q.search(t)))
print('NPS greet->named response n', len(g))
t = g.groupby('el').agg(n=('r_el', 'size'), r_el=('r_el', 'mean'), r_greet=('r_greet', 'mean'), r_q=('r_q', 'mean')).round(3)
print(t); out['nps_greet_reply_by_elong'] = t.reset_index().to_dict('records')
print('lag posts median', g.resp_lag_posts.median())
s = pd.read_pickle(f"{S}/sessions.pkl")
bare = re.compile(r'^\s*(hi+|he+y+|hello+|yo+|sup|hellooo?)\b[\w\s!.?]{0,8}$', re.I)
b = s[(s.open_type == 'greet_only') | ((s.corpus == 'maichat') & s.open_text.map(lambda t: bool(bare.match(t))))]
b = b[b.reply_text.notna()].copy()
b['r_greet'] = b.reply_text.map(lambda t: bool(GREET.match(t)))
b['r_q'] = b.reply_text.map(lambda t: bool(Q.search(t)))
b['r_el'] = b.reply_text.map(el); b['o_el'] = b.open_text.map(el)
b['r_topic'] = b.reply_type.isin(['greet+content', 'content', 'question', 'greet+question'])
t = b.groupby('corpus').agg(n=('r_greet', 'size'), r_greet=('r_greet', 'mean'), r_q=('r_q', 'mean'), r_el=('r_el', 'mean'),
                            o_el=('o_el', 'mean'), r_topic=('r_topic', 'mean'), lat_med_s=('reply_latency_s', 'median'),
                            r_nmsgs=('reply_n_msgs', 'mean')).round(2)
print(t); out['bare_greet_openings'] = t.reset_index().to_dict('records')
print(len(b), b[['r_greet', 'r_q', 'r_el', 'r_topic']].mean().round(2).to_dict())
print(pd.crosstab(b.o_el, b.r_el))
for _, r in b.iterrows(): print(f"  {r.corpus[:3]} {r.open_text[:30]!r:34s} -> {r.reply_text[:60]!r}")
m = s[(s.corpus == 'maichat') & s.reply_text.notna()].copy()
m['o_g'] = m.open_text.map(lambda t: bool(GREET.match(t))); m['r_g'] = m.reply_text.map(lambda t: bool(GREET.match(t)))
print(pd.crosstab(m.o_g, m.r_g))
json.dump(out, open(os.path.join(os.path.dirname(__file__), '../../analysis/data/a2_oi_summary.json'), 'w'), indent=1)

# lexical identity of greeting word (normalize elongation)
def gw(t):
    m_ = GREET.match(str(t))
    if not m_: return None
    w = re.sub(r'[^a-z]', '', m_.group(0).lower().split()[0] if m_.group(0).split() else '')
    return re.sub(r'(.)\1+', r'\1', w)
g['gw'] = g.text.map(gw); g['rgw'] = g.resp_first_text.map(gw)
gg = g[g.gw.notna() & g.rgw.notna()]
print('NPS greet-back pairs', len(gg), 'same word', (gg.gw == gg.rgw).mean().round(3))
print(pd.crosstab(gg.gw, gg.rgw).loc[['hi', 'hey', 'helo', 'hiya'], ['hi', 'hey', 'helo', 'hiya', 'heya']] if len(gg) else '')
b['gw'] = b.open_text.map(gw); b['rgw'] = b.reply_text.map(gw)
bb = b[b.gw.notna() & b.rgw.notna()]
print('pooled greet-back pairs', len(bb), 'same word', (bb.gw == bb.rgw).mean().round(3))
