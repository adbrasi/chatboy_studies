"""a8 step 10: the LLM 'three-beat' template (reaction/validation -> comment/paraphrase -> follow-up question).
Output: analysis/data/a8_template.json"""
import json, re
import a8_common as C
from a8_compare import load_all

REACT = re.compile(r"^\W*(?:oh|aw+|wow|omg|haha\w*|lol|yay|ugh|no way|wait|hey|hi|yes+|yeah|totally|absolutely|definitely|i totally|i know|i'm (?:so )?sorry|that's|that is|that sounds|sounds|i get|i understand|same|right|nice|great|amazing|awesome|congrat)", re.I)
X, R = load_all()
ids = [c["id"] for c in X]
out = {}
for cn, d in R.items():
    n3 = n_any = 0
    for i in ids:
        s = C.sentences(d.get(i) or "")
        if not s:
            continue
        three = len(s) >= 3 and bool(REACT.search(s[0])) and s[-1].rstrip().endswith("?")
        two = len(s) >= 2 and bool(REACT.search(s[0])) and s[-1].rstrip().endswith("?")
        n3 += three; n_any += two
    out[cn] = {"pct_3beat": round(100 * n3 / len(ids), 1), "pct_react_then_question": round(100 * n_any / len(ids), 1)}
C.save("a8_template.json", out)
print(out)
