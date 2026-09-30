"""Normalize all corpora into one message-level JSONL schema.

Output: data/processed/messages.jsonl, one line per message:
  corpus, conv_id, idx, speaker, text, ts (ISO or null), lang,
  plus corpus-specific fields (typing features, dialogue-act labels, gold emotion...).
"""
import csv, glob, html, json, os, re, xml.etree.ElementTree as ET
from datetime import datetime

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RAW, OUT = os.path.join(ROOT, "raw"), os.path.join(ROOT, "processed")


def iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def maichat():
    for f in sorted(glob.glob(f"{RAW}/maichat/maichat/*.json")):
        d = json.load(open(f))
        cid = os.path.basename(f)[:-5]
        first = d["firstId"]["$oid"]
        prev_submit = None
        for i, m in enumerate(d["messages"]):
            submit = iso(m["time"]["$date"])
            logs = [(iso(l["time"]["$date"]), l["message"]) for l in m["logs"]]
            logs.sort(key=lambda x: x[0])
            typing = {}
            if logs:
                start = logs[0][0]
                pauses = [(b[0] - a[0]).total_seconds() for a, b in zip(logs, logs[1:])]
                shrinks = [(a[1], b[1]) for a, b in zip(logs, logs[1:]) if len(b[1]) < len(a[1])]
                deleted = sum(len(a) - len(b) for a, b in shrinks)
                # peak length reached vs final length: text written then removed
                peak = max(len(t) for _, t in logs)
                typing = {
                    "compose_s": round((submit - start).total_seconds(), 3),
                    "n_states": len(logs),
                    "n_deletion_events": len(shrinks),
                    "chars_deleted": deleted,
                    "peak_len": peak,
                    "max_pause_s": round(max(pauses), 3) if pauses else 0.0,
                    "n_pauses_over_2s": sum(p > 2 for p in pauses),
                    # gap between the previous message in the chat (either speaker) and starting to type
                    "idle_before_typing_s": round((start - prev_submit).total_seconds(), 3) if prev_submit else None,
                    # did the author start typing before the previous message arrived (overlap)?
                }
            yield {
                "corpus": "maichat", "conv_id": cid, "idx": i,
                "speaker": "A" if m["ofUser"]["$oid"] == first else "B",
                "text": m["content"], "ts": submit.isoformat(), "lang": "en",
                "typing": typing,
            }
            prev_submit = submit


def berntzen():
    for f in sorted(glob.glob(f"{RAW}/berntzen/*.events.tsv")):
        cid = os.path.basename(f).split(".")[0].replace("ACAD_WhatsApp_", "wa_")
        rows = [l.rstrip("\n").split("\t") for l in open(f, encoding="utf-8") if l.strip()]
        speakers = {}
        for i, r in enumerate(rows):
            actor, ts, _cls, text = (r + [""] * 4)[:4]
            spk = speakers.setdefault(actor, chr(ord("A") + len(speakers)) if len(speakers) < 26 else actor)
            media = None
            if re.search(r"<‎?(afbeelding|video|audio|geluid|contact|locatie|sticker|GIF)[^>]*weggelaten>", text):
                media = re.search(r"(afbeelding|video|audio|geluid|contact|locatie|sticker|GIF)", text).group(1)
            yield {
                "corpus": "whatsapp_nl", "conv_id": cid, "idx": i, "speaker": spk,
                "text": text, "ts": ts + ":00", "lang": "nl", "media": media,
                "n_speakers": None,
            }


def nps():
    for f in sorted(glob.glob(f"{RAW}/nps/nps_chat/*.xml")):
        cid = "nps_" + os.path.basename(f).split("_")[0]
        for i, p in enumerate(ET.parse(f).getroot().iter("Post")):
            yield {
                "corpus": "nps_chatroom", "conv_id": cid, "idx": i, "speaker": p.get("user"),
                "text": (p.text or "").strip(), "ts": None, "lang": "en",
                "dialogue_act": p.get("class"),
            }


def nus():
    # not conversations: individual SMS grouped by sender
    for i, m in enumerate(ET.parse(glob.glob(f"{RAW}/nus/*.xml")[0]).getroot().iter("message")):
        prof = m.find("source/userProfile")
        yield {
            "corpus": "nus_sms", "conv_id": "sms_user_" + (prof.findtext("userID") or "?"), "idx": i,
            "speaker": prof.findtext("userID"), "text": (m.findtext("text") or "").strip(),
            "ts": None, "lang": "en", "country": prof.findtext("country"),
            "age": prof.findtext("age"), "gender": prof.findtext("gender"),
        }


def empathetic():
    fix = lambda s: s.replace("_comma_", ",")
    for split in ("train", "valid", "test"):
        with open(f"{RAW}/ed/empatheticdialogues/{split}.csv", encoding="utf-8") as fh:
            for line in fh.readlines()[1:]:
                p = line.rstrip("\n").split(",")
                if len(p) < 8:
                    continue
                yield {
                    "corpus": "empathetic", "conv_id": p[0], "idx": int(p[1]) - 1,
                    # speaker_idx is a worker id; turns strictly alternate, A = the one describing the situation
                    "speaker": "A" if int(p[1]) % 2 == 1 else "B",
                    "text": fix(p[5]), "ts": None, "lang": "en",
                    "gold_emotion": p[2], "situation": fix(p[3]), "split": split,
                }


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    counts = {}
    with open(f"{OUT}/messages.jsonl", "w", encoding="utf-8") as out:
        for gen in (maichat, berntzen, nps, nus, empathetic):
            for rec in gen():
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[rec["corpus"]] = counts.get(rec["corpus"], 0) + 1
    print(counts)
