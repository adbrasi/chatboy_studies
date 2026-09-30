"""a1: loads maichat + whatsapp_nl turns/messages + jev_base into pandas; caches pickles in scratchpad."""
import json, os, pickle
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
P = os.path.join(ROOT, "data", "processed")
CACHE = os.environ.get("A1_CACHE", "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad/a1_cache.pkl")
CORP = ("maichat", "whatsapp_nl")


def _read(fn):
    out = []
    for l in open(os.path.join(P, fn), encoding="utf-8"):
        if '"maichat"' in l[:40] or '"whatsapp_nl"' in l[:40]:
            d = json.loads(l)
            if d["corpus"] in CORP:
                out.append(d)
    return out


def load():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    turns = _read("turns.jsonl")
    msgs = _read("messages_feat.jsonl")
    base = [json.loads(l) for l in open(os.path.join(P, "jev_base.jsonl"), encoding="utf-8")]
    T = pd.DataFrame(turns)
    M = pd.DataFrame(msgs)
    # flatten
    F = pd.json_normalize(M["f"].tolist()).add_prefix("f_")
    M = pd.concat([M.drop(columns=["f"]), F], axis=1)
    if "typing" in M:
        ty = M["typing"].apply(lambda x: x if isinstance(x, dict) else {})
        M = pd.concat([M.drop(columns=["typing"]), pd.json_normalize(ty.tolist()).add_prefix("ty_")], axis=1)
    ty = T["typing"].apply(lambda x: x if isinstance(x, dict) else {})
    T = pd.concat([T.drop(columns=["typing"]), pd.json_normalize(ty.tolist()).add_prefix("ty_")], axis=1)
    # jev base flattened
    rows = []
    for b in base:
        r = {"corpus": b["corpus"], "conv_id": b["conv_id"], "turn_idx": b["turn_idx"]}
        for k, v in (b.get("D") or {}).items():
            r["D_" + k] = v.get("noul", v.get("score", v.get("choice")))
            if v["type"] == "choice":
                r["D_" + k + "_conf"] = v.get("confidence")
        for k, v in (b.get("P") or {}).items():
            r["P_" + k] = v.get("noul", v.get("score", v.get("choice")))
            if k == "p_n_msgs":
                for o, p in v["probabilities"].items():
                    r["P_nm_" + o] = p
                r["P_nm_conf"] = v.get("confidence")
            if k == "p_length":
                r["P_len_conf"] = v.get("confidence")
        r["has_P"] = b.get("P") is not None
        rows.append(r)
    J = pd.DataFrame(rows)
    out = (T, M, J)
    pickle.dump(out, open(CACHE, "wb"))
    return out


if __name__ == "__main__":
    T, M, J = load()
    print(T.shape, M.shape, J.shape)
    print(T.columns.tolist()); print(M.columns.tolist()); print(J.columns.tolist())


def enriched():
    """Turn-level table for maichat + dyadic whatsapp_nl with Jev D/P, partner/previous-turn context and
    intra-burst gaps. Negative/invalid whatsapp latencies -> NaN."""
    import numpy as np
    T, M, J = load()
    T = T.copy()
    nsp = T.groupby("conv_id").speaker.nunique()
    T = T[T.conv_id.map(nsp) == 2]
    T.loc[T.response_latency_s < 0, "response_latency_s"] = np.nan
    T.loc[T.burst_span_s < 0, "burst_span_s"] = np.nan
    T["spk"] = T.conv_id + ":" + T.speaker
    # intra-burst gaps from message timestamps
    Mi = M.set_index(["conv_id", "idx"])
    ts = pd.to_datetime(M.ts.astype(str).str.replace(r"[\[\]]", "", regex=True), utc=True, format="mixed", errors="coerce")
    tsmap = dict(zip(zip(M.conv_id, M.idx), ts))
    chmap = dict(zip(zip(M.conv_id, M.idx), M.f_n_chars))
    gaps = []
    for cid, idxs in zip(T.conv_id, T.msg_idxs):
        if len(idxs) < 2:
            gaps.append([])
            continue
        t = [tsmap[(cid, i)] for i in idxs]
        gaps.append([(b - a).total_seconds() for a, b in zip(t, t[1:]) if pd.notna(a) and pd.notna(b) and b >= a])
    T["gaps"] = gaps
    T["gap_med"] = [float(np.median(g)) if g else np.nan for g in gaps]
    T["bubble_chars"] = [[chmap[(c, i)] for i in ix] for c, ix in zip(T.conv_id, T.msg_idxs)]
    T = T.sort_values(["conv_id", "turn_idx"]).reset_index(drop=True)
    g = T.groupby("conv_id")
    same_sess = g.session.shift(1) == T.session
    T["prev_partner_chars"] = g.total_chars.shift(1).where(same_sess)
    T["prev_partner_nmsgs"] = g.n_msgs.shift(1).where(same_sess)
    T["prev_partner_q"] = g.has_q.shift(1).where(same_sess)
    T["prev_partner_latency"] = g.response_latency_s.shift(1).where(same_sess)
    gs = T.groupby("spk")
    T["prev_own_nmsgs"] = gs.n_msgs.shift(1)
    T["prev_own_chars"] = gs.total_chars.shift(1)
    T["next_partner_latency"] = g.response_latency_s.shift(-1).where(g.session.shift(-1) == T.session)
    T["next_partner_chars"] = g.total_chars.shift(-1).where(g.session.shift(-1) == T.session)
    T["is_last_in_session"] = (g.session.shift(-1) != T.session)
    T = T.merge(J, on=["corpus", "conv_id", "turn_idx"], how="left")
    return T, M
