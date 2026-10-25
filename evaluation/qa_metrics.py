"""Official-style Hotpot answer normalization; extraction never sees the gold answer."""
import re
import string
from collections import Counter


def normalize(text):
    text=text.lower()
    text="".join(c for c in text if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b"," ",text).split())


def extract_answer(body):
    match=re.search(r"^(?:#{1,6}\s*)?(?:\*\*)?(?:答案|Answer)(?:\*\*)?\s*[:：]\s*(.+)$",body,re.M|re.I)
    return match.group(1).replace("**","").strip() if match else None


def answer_metrics(prediction,gold):
    a,b=normalize(prediction or ""),normalize(gold)
    exact=float(a==b)
    if a in ("yes","no","noanswer") or b in ("yes","no","noanswer"):
        return {"exact_match":exact,"f1":exact}
    overlap=sum((Counter(a.split()) & Counter(b.split())).values())
    precision=overlap/max(1,len(a.split()));recall=overlap/max(1,len(b.split()))
    return {"exact_match":exact,"f1":2*precision*recall/(precision+recall) if overlap else 0.0}
