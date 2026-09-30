"""a3 (emoção/empatia/humor): definições compartilhadas — taxonomia de 32 emoções do EmpatheticDialogues,
famílias (cascata), estratégias do ouvinte e funções do riso."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
SCR = os.environ.get("A3_SCRATCH", "/tmp/claude-0/-home-user-chatboy-studies/e75482e2-714d-5677-93c0-fc3fdb95e99c/scratchpad/a3")
OUT = os.path.join(ROOT, "analysis", "data")

EMO32 = {
    "surprised": "caught off guard by something unexpected",
    "excited": "eager, thrilled about something happening or coming",
    "joyful": "very happy, delighted",
    "content": "calmly satisfied, at peace",
    "proud": "proud of oneself or of someone close",
    "impressed": "admiring someone else's ability or achievement",
    "grateful": "thankful for something someone did",
    "hopeful": "hoping something good will happen",
    "confident": "sure of oneself or of a good outcome",
    "anticipating": "looking forward to / waiting for an upcoming event",
    "prepared": "feeling ready for something because they got ready",
    "trusting": "relying on / trusting someone",
    "faithful": "loyal, keeping faith or commitment",
    "sentimental": "tender, moved by memories or meaning",
    "nostalgic": "longing for the past",
    "caring": "caring for / looking after someone",
    "sad": "sad, unhappy",
    "lonely": "alone, missing company",
    "devastated": "crushed by a serious loss or blow",
    "disappointed": "let down, expectations not met",
    "afraid": "scared of something",
    "terrified": "extremely scared",
    "anxious": "worried, nervous about something",
    "apprehensive": "uneasy, wary about something ahead",
    "angry": "angry, mad",
    "furious": "extremely angry, enraged",
    "annoyed": "irritated, bothered by something minor",
    "disgusted": "repulsed, grossed out",
    "jealous": "envious or jealous of someone",
    "guilty": "feeling guilty for something they did",
    "ashamed": "ashamed of oneself",
    "embarrassed": "embarrassed, awkward in front of others",
}
FAMILIES = {
    "joy_pride": ["surprised", "excited", "joyful", "content", "proud", "impressed", "grateful"],
    "hope_confidence": ["hopeful", "confident", "anticipating", "prepared", "trusting", "faithful"],
    "warm_nostalgic": ["sentimental", "nostalgic", "caring"],
    "sadness": ["sad", "lonely", "devastated", "disappointed"],
    "fear_anxiety": ["afraid", "terrified", "anxious", "apprehensive"],
    "anger_disgust": ["angry", "furious", "annoyed", "disgusted", "jealous"],
    "shame_guilt": ["guilty", "ashamed", "embarrassed"],
}
FAMILY_DESC = {
    "joy_pride": "happiness, excitement, pride, gratitude, admiration, pleasant surprise",
    "hope_confidence": "hope, confidence, anticipation, readiness, trust, loyalty",
    "warm_nostalgic": "tender, sentimental, nostalgic or caring feelings",
    "sadness": "sadness, loneliness, disappointment, devastation",
    "fear_anxiety": "fear, terror, anxiety, apprehension",
    "anger_disgust": "anger, fury, annoyance, disgust, jealousy",
    "shame_guilt": "guilt, shame, embarrassment",
}
FAM_OF = {e: f for f, es in FAMILIES.items() for e in es}
POS_FAM = {"joy_pride", "hope_confidence", "warm_nostalgic"}
def polarity(e):
    if e == "surprised":
        return "ambiguous"
    return "positive" if FAM_OF[e] in POS_FAM else "negative"
