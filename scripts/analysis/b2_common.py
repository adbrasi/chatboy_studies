"""b2: shared helpers — rajadas (bursts of bubbles in <60 s), dev/test split by conversation, state formatting.

Reuses a1_load.enriched() (maichat + dyadic whatsapp_nl turns with Jev D/P labels).
"""
import json, os, random, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from a1_load import enriched  # noqa

ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(ROOT, "analysis", "data")
SCR = "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad/b2"
os.makedirs(SCR, exist_ok=True)


def split_of(conv_id, seed=2):
    """Deterministic dev/test split by conversation (50/50)."""
    import hashlib
    h = int(hashlib.md5(f"{seed}:{conv_id}".encode()).hexdigest(), 16)
    return "dev" if h % 2 == 0 else "test"


def _ts(M):
    ts = pd.to_datetime(M.ts.astype(str).str.replace(r"[\[\]]", "", regex=True), utc=True, format="mixed", errors="coerce")
    return dict(zip(zip(M.conv_id, M.idx), ts))


def max_in_window(times, win):
    """max number of consecutive bubbles whose first->last span <= win seconds (times sorted, seconds)."""
    best, j = 1 if times else 0, 0
    for i in range(len(times)):
        while times[i] - times[j] > win:
            j += 1
        best = max(best, i - j + 1)
    return best


def load(cache=True):
    fn = os.path.join(SCR, "b2_turns.pkl")
    if cache and os.path.exists(fn):
        return pd.read_pickle(fn), pd.read_pickle(os.path.join(SCR, "b2_msgs.pkl"))
    T, M = enriched()
    tsmap = _ts(M)
    T = T.copy()
    secs = []
    for cid, idxs in zip(T.conv_id, T.msg_idxs):
        t = [tsmap.get((cid, i)) for i in idxs]
        t0 = t[0]
        secs.append([(x - t0).total_seconds() if (x is not None and pd.notna(x) and pd.notna(t0)) else np.nan for x in t])
    T["bub_t"] = secs
    # maichat: strict 60 s window on ms timestamps. WA (minute resolution):
    #   lenient = bubbles whose minute stamps differ by <= 1 min (same or consecutive minute; real span < 120 s)
    #   strict  = bubbles in the SAME minute stamp (real span < 60 s guaranteed)
    def mw(ts, win):
        ts = [x for x in ts if not np.isnan(x)]
        ts = sorted(ts)  # WA has some ordering noise
        return max_in_window(ts, win) if ts else 1
    T["max60"] = [mw(s, 60) if c == "maichat" else mw(s, 60) for s, c in zip(T.bub_t, T.corpus)]
    T["max60_strict"] = [mw(s, 60) if c == "maichat" else mw(s, 0) for s, c in zip(T.bub_t, T.corpus)]
    T["split"] = T.conv_id.map(split_of)
    T.to_pickle(fn)
    M.to_pickle(os.path.join(SCR, "b2_msgs.pkl"))
    return T, M


def gap_bucket(sec):
    if sec is None or (isinstance(sec, float) and np.isnan(sec)):
        return None
    for lim, n in ((60, "within a minute"), (120, "about a minute later"), (900, "a few minutes later"),
                   (3600, "within the hour"), (6 * 3600, "hours later")):
        if sec < lim:
            return n
    return "much later"


def fmt_turn(t, bubbles=True, maxc=300, show_gap=True):
    """t: row (namedtuple or Series) of T. bubbles=True shows the list of separate messages."""
    texts = [str(x).strip()[:maxc] for x in t.texts]
    d = {"speaker": t.speaker}
    if bubbles:
        d["messages"] = texts
    else:
        d["text"] = " ".join(texts)[: maxc * 2]
    if show_gap and t.corpus != "maichat":
        g = gap_bucket(t.response_latency_s)
        if g:
            d["replied"] = g
    return d


def boot_ci(df, fn, n=1000, seed=0, group="conv_id"):
    """Bootstrap by conversation. fn(df)->float. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    groups = df[group].unique()
    idx = {g: np.where(df[group].values == g)[0] for g in groups}
    pt = fn(df)
    vals = []
    for _ in range(n):
        gs = rng.choice(groups, len(groups), replace=True)
        ii = np.concatenate([idx[g] for g in gs])
        try:
            vals.append(fn(df.iloc[ii]))
        except Exception:
            pass
    vals = np.array([v for v in vals if v is not None and not np.isnan(v)])
    return float(pt), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def dump(obj, name):
    fn = os.path.join(OUT, name)
    json.dump(obj, open(fn, "w"), indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return fn
