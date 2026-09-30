"""a8 step 4: human verbal tics per 1,000 messages, per corpus (+ Dutch equivalents for whatsapp_nl).
Output: analysis/data/a8_human_tics.json"""
import re, statistics as st
from collections import Counter
import a8_common as C

NL = {"haha-família": r"\b(?:a?ha(?:ha)+h?|hah+|hihi\w*)\b", "echt": r"\becht\b", "gewoon": r"\bgewoon\b", "wel": r"\bwel\b",
      "nou": r"\bnou\b", "hoor": r"\bhoor\b", "zeg": r"\bzeg\b", "toch": r"\btoch\b", "oke/ok (início)": r"^\s*(?:oke+|ok+|okee)\b",
      "ja/jaa (início)": r"^\s*ja+\b", "nee (início)": r"^\s*nee+\b", "xx/x (beijo)": r"(?:^|\s)x+\s*$", "pff/ugh": r"\bp+f+\b",
      "ehm/uhm": r"\b(?:ehm+|uhm+|hmm+)\b", "maar/en (início)": r"^\s*(?:maar|en)\b", "dus (início)": r"^\s*dus\b",
      "idd/wtf/omg": r"\b(?:idd|wtf|omg)\b", "haha no fim (pontuação)": r"\b(?:haha\w*|hihi\w*)\s*$", "lekker/leuk": r"\b(?:lekker|leuk)\b"}
NL = {k: re.compile(v, re.I) for k, v in NL.items()}


def rates(texts):
    n = len(texts)
    agg = Counter()
    chars = []
    for t in texts:
        f = C.style_feats(t)
        chars.append(f["chars"])
        for k, v in f.items():
            if isinstance(v, bool) and v:
                agg[k] += 1
    r = {k: round(1000 * v / n, 1) for k, v in sorted(agg.items())}
    r["_n"] = n
    r["_median_chars"] = st.median(chars)
    return r


def main():
    M = C.messages()
    out = {}
    for corp in ("maichat", "nps_chatroom", "nus_sms", "empathetic"):
        texts = [m["text"].replace("_comma_", ",") for m in M[corp] if m.get("dialogue_act") != "System" and m["text"].strip()]
        out[corp] = rates(texts)
    wa = [m["text"] for m in M["whatsapp_nl"] if m["text"].strip() and not m["f"]["media"]]
    r = rates(wa)
    for k, rx in NL.items():
        r["nl:" + k] = round(1000 * sum(bool(rx.search(t)) for t in wa) / len(wa), 1)
    out["whatsapp_nl"] = r
    # typing self-corrections (maichat typing logs) — the invisible edits
    ty = [m["typing"] for m in M["maichat"] if m.get("typing")]
    out["maichat_typing"] = {
        "n": len(ty),
        "pct_msgs_with_deletion": round(100 * sum(x["n_deletion_events"] > 0 for x in ty) / len(ty), 1),
        "pct_msgs_abandoned_text": round(100 * sum(x["peak_len"] > len(m["text"]) for x, m in zip(ty, [m for m in M["maichat"] if m.get("typing")])) / len(ty), 1),
        "pct_visible_star_correction": round(100 * sum(m["f"]["self_correction"] for m in M["maichat"]) / len(M["maichat"]), 2),
    }
    # typos that stay visible: crude — words not in a vocabulary built from empathetic (edited, clean) text
    vocab = Counter(w for m in M["empathetic"] for w in re.findall(r"[a-z]+", m["text"].lower()))
    vocab = {w for w, c in vocab.items() if c >= 3}
    for corp in ("maichat", "nus_sms", "nps_chatroom"):
        toks = [w for m in M[corp] for w in re.findall(r"[a-z]{4,}", m["text"].lower())]
        oov = [w for w in toks if w not in vocab]
        out.setdefault("oov", {})[corp] = {"pct_tokens_oov_4plus": round(100 * len(oov) / max(1, len(toks)), 1),
                                            "top_oov": Counter(oov).most_common(40)}
    C.save("a8_human_tics.json", out)
    keys = [k for k in out["maichat"] if k.startswith("tic:")] + ["final_punct", "final_period", "starts_lower", "all_lower", "elongation", "no_apostrophe", "self_correction", "le3_words", "any_q", "any_excl", "multi_excl", "emoji", "laugh", "_median_chars"]
    print(f"{'feature':40s}" + "".join(f"{c[:10]:>11s}" for c in ("maichat", "nps_chatroom", "nus_sms", "empathetic", "whatsapp_nl")))
    for k in keys:
        print(f"{k:40s}" + "".join(f"{out[c].get(k, 0):>11}" for c in ("maichat", "nps_chatroom", "nus_sms", "empathetic", "whatsapp_nl")))
    print({k: v for k, v in out["whatsapp_nl"].items() if k.startswith("nl:")})
    print(out["maichat_typing"])
    for c, v in out["oov"].items():
        print(c, v["pct_tokens_oov_4plus"], v["top_oov"][:25])


if __name__ == "__main__":
    main()
