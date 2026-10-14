"""Entity checker: does the English keep the Hindi amounts, numbers, names, dates and negations?

    python checker/checker.py "पाँच लाख… नहीं नहीं, पाँच करोड़।" "five lakh... no no, five crore."

Hindi entities come from the gold file (checker/gold/clipNN.json) or, for a new clip,
from hindi_entities(), which uses the number parser plus the negation and repair rules
below. preserved(entity, english, version) is the checker's verdict for one entity.
"""
import os
import re
import sys
import unicodedata

from rapidfuzz import fuzz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "prototypes"))
import hindi_number_parser_prototype as hn  # noqa: E402


def N(s):
    return unicodedata.normalize("NFC", s)


# ---------------------------------------------------------------- English numbers
ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                   "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
ONES["oh"] = 0
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
SCALE = {"thousand": 10**3, "k": 10**3, "lakh": 10**5, "lakhs": 10**5, "lac": 10**5, "lacs": 10**5,
         "million": 10**6, "crore": 10**7, "crores": 10**7, "billion": 10**9}
MONTHS = {m: i + 1 for i, m in enumerate("january february march april may june july august september "
                                         "october november december".split())}
ORD = {"first": 1, "second": 2, "third": 3, "fifth": 5, "eighth": 8, "ninth": 9, "twelfth": 12}


def en_tokens(text):
    t = text.lower().replace("₹", " ").replace("rs.", " ").replace("-", " ")
    t = re.sub(r"(\d),(?=\d)", r"\1", t)                 # 5,00,000 / 500,000 -> 500000
    t = re.sub(r"(\d)(k|lakh|crore|million)\b", r"\1 \2", t)
    return re.findall(r"\d+(?:\.\d+)?|[a-z']+|[….;!?]", t)   # punctuation (not commas) ends a number


ORDINAL = {w: i + 1 for i, w in enumerate("first _ third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
                                         "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth "
                                         "twentieth".split()) if w != "_"}   # "second" left out: also a time unit


def word_val(w):
    if w in ("twice", "thrice"):                          # "do baar" -> "twice"
        return 2 if w == "twice" else 3
    if w in ORDINAL:                                      # "on the third and fifth" (dates, counts)
        return ORDINAL[w]
    if re.fullmatch(r"\d+(\.\d+)?", w):
        return float(w) if "." in w else int(w)
    return ONES.get(w, TENS.get(w))


def en_amounts(text):
    """All numeric values in an English string: "five crore" -> 50000000, "2.5 lakh" -> 250000,
    "three and a half lakh" -> 350000, "50 million" -> 50000000."""
    toks, out = en_tokens(text), []
    total = grp = 0
    last_scale, prev = None, None                          # prev: kind of the previous number token

    def flush():
        nonlocal total, grp, last_scale, prev
        if prev is not None:
            v = total + grp
            out.append(int(v) if float(v).is_integer() else v)
        total = grp = 0
        last_scale = prev = None

    i = 0
    while i < len(toks):
        w, nxt = toks[i], toks[i + 1] if i + 1 < len(toks) else ""
        v = word_val(w)
        if v is not None:
            if prev is None or prev == "scale":
                grp, prev = v, ("tens" if w in TENS else "num")
            elif prev == "tens" and isinstance(v, int) and 0 < v < 10 and w in ONES:
                grp, prev = grp + v, "num"                 # "twenty five"
            elif prev == "hundred" and v < 100:
                grp, prev = grp + v, ("tens" if w in TENS else "num")   # "two hundred (and) five"
            else:
                flush()
                continue                                   # re-read w as the start of a new number
        elif w == "hundred" and prev in ("num", "tens", None):
            grp, prev = (grp or 1) * 100, "hundred"
        elif w in SCALE and prev is not None:
            s = SCALE[w]
            if last_scale is not None and s >= last_scale and prev != "scale":
                g = grp                                    # "five lakh five crore": two amounts
                grp = 0
                flush()
                grp, prev = g, "num"
            if prev == "scale":                            # "five hundred crore"-style stacking
                total, grp = total * s, 0
            else:
                total, grp = total + grp * s, 0
            last_scale, prev = s, "scale"
        elif w == "and" and prev is not None and nxt == "a" and i + 2 < len(toks) and toks[i + 2] in ("half", "quarter"):
            grp += 0.5 if toks[i + 2] == "half" else 0.25  # "three and a half lakh"
            i += 2
        elif w == "and" and prev == "hundred" and word_val(nxt) is not None:
            pass
        elif w == "half" and prev is None:                 # "half a lakh"
            grp, prev = 0.5, "num"
            i += 1 if nxt == "a" else 0
        elif w == "a" and prev is None and (nxt in SCALE or nxt == "hundred"):
            grp, prev = 1, "num"                           # "a lakh", "a hundred"
        elif w == "point" and prev in ("num", "tens") and word_val(nxt) is not None:
            k, ds = i + 1, ""
            while k < len(toks) and toks[k] in ONES and ONES[toks[k]] < 10:
                ds, k = ds + str(ONES[toks[k]]), k + 1
            if ds:
                grp += int(ds) / 10 ** len(ds)
            i = k
            continue
        else:
            flush()
        i += 1
    flush()
    return out


def en_digits(text):
    """Digits in reading order: "nine eight double seven 5" -> "98775"."""
    toks, s, i = en_tokens(text), "", 0
    while i < len(toks):
        w = toks[i]
        rep = {"double": 2, "triple": 3}.get(w)
        if rep and i + 1 < len(toks) and (toks[i + 1] in ONES and ONES[toks[i + 1]] < 10 or toks[i + 1].isdigit()):
            d = str(ONES.get(toks[i + 1], toks[i + 1]))
            s, i = s + d * rep, i + 2
            continue
        if w.isdigit():
            s += w
        elif w in ONES and ONES[w] < 10:
            s += str(ONES[w])
        i += 1
    return s


def en_times(text):
    """Clock times: "4:30", "half past four", "quarter to five", "four thirty pm" -> "H:MM"."""
    t = text.lower()
    out = [f"{int(h)}:{m}" for h, m in re.findall(r"\b(\d{1,2})\s?[:.]\s?(\d{2})\b", t)]
    words = {**{k: v for k, v in ONES.items() if 1 <= v <= 12}, **{str(v): v for v in range(1, 13)}}
    for pre, h in re.findall(r"\b(half past|quarter past|quarter to)\s+(\w+)", t):
        if h in words:
            hh = words[h]
            out.append({"half past": f"{hh}:30", "quarter past": f"{hh}:15",
                        "quarter to": f"{(hh - 2) % 12 + 1}:45"}[pre])
    for h, m in re.findall(r"\b(\w+)\s+(fifteen|thirty|forty five|forty-five)\b", t):
        if h in words:
            out.append(f"{words[h]}:{ {'fifteen': '15', 'thirty': '30'}.get(m, '45')}")
    for h in re.findall(r"\b(\w+)\s+o'?clock", t):
        if h in words:
            out.append(f"{words[h]}:00")
    for h in re.findall(r"\b(\d{1,2}|\w+)\s*(?:am|pm|a\.m\.|p\.m\.)", t):
        if h in words:
            out.append(f"{words[h]}:00")
    return out


def en_dates(text):
    """(day, month) pairs: "15 August", "August 15th", "the fifteenth of August"."""
    t, out = text.lower(), []
    day = r"(\d{1,2}|[a-z]+)(?:st|nd|rd|th)?"
    for d, m in re.findall(day + r"\s+(?:of\s+)?(" + "|".join(MONTHS) + r")\b", t) + \
            [(d, m) for m, d in re.findall(r"\b(" + "|".join(MONTHS) + r")\s+" + day, t)]:
        v = int(d) if d.isdigit() else ONES.get(d, ORD.get(d))
        if v is None and d.endswith("th"):
            v = ONES.get(d[:-2]) or ONES.get(d[:-1] if d.endswith("eth") else d[:-2])
        if v:
            out.append((v, MONTHS[m]))
    return out


# ---------------------------------------------------------------- negation
HI_NEG = {N(w) for w in ("नहीं", "नही", "मत", "न", "ना", "नहि",
                        # English negations written in Devanagari (Level-1 code-mix)
                        "नॉट", "नेवर", "डोन्ट", "डोंट", "डिडन्ट", "डिडंट", "वोन्ट", "वोंट", "कान्ट", "कांट",
                        "इज़न्ट", "इजन्ट", "डज़न्ट", "डजन्ट", "वाज़न्ट", "शुडन्ट", "कुडन्ट")}
HI_REPAIR = {N(w) for w in ("नहीं", "नही", "ना", "सॉरी", "मतलब", "आई मीन", "आय मीन", "नो", "सॉरी नहीं")}
REPAIR_PHRASES = [N(w) for w in ("नहीं नहीं", "नही नही", "ना ना", "सॉरी नहीं", "सॉरी नही", "सॉरी ना", "नहीं सॉरी", "नही सॉरी")]
EN_NEG = re.compile(r"\b(not|no|never|neither|nor|without|nothing|none|nobody|cannot)\b|n't\b")


def hi_negations(sent):
    """Negation cues in a Hindi sentence, minus the tag question "…ना?" and repairs ("नहीं नहीं", "सॉरी नहीं")."""
    s = N(sent)
    for r in REPAIR_PHRASES:
        s = s.replace(r, " ")
    s = re.sub(N(r"\bना\s*\?"), " ? ", s)                 # tag question: "आप भेज देंगे ना?"
    s = re.sub(N(r"(\S)\s+ना\s*([,।?]|$)"), r"\1 \2", s)  # sentence-final ना is a tag too
    return sum(w in HI_NEG for w in re.findall(r"[^\s।॥,.?!…;:]+", s))


def en_negations(text):
    t = re.sub(r"\bno[,.…\s]+no\b", " ", text.lower())   # "no, no, five crore" is the repair
    return len(EN_NEG.findall(t))


# ---------------------------------------------------------------- names
def name_key(s):
    s = s.lower()
    for a, b in (("w", "v"), ("sh", "s"), ("ph", "f"), ("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[^a-z ]", "", s))


_U = None


LABIAL = "\u092a\u092b\u092c\u092d\u092e"            # pa pha ba bha ma: anusvara before them is "m"


def romanize(hi):
    """Readable Hinglish for display and name matching (the aligner keeps raw uroman):
    nasal dot -> n ("haan", "main", "bank"), retroflex tt/dd -> t/d, long ii/uu -> i/u, final aa -> a."""
    global _U
    if _U is None:
        import uroman
        _U = uroman.Uroman()
    s = N(hi)
    s = re.sub("[\u0902\u0901](?=[" + LABIAL + "])", "\u092e\u094d", s)   # nasal before labial: ma + virama
    s = re.sub("[\u0902\u0901]", "\u0928\u094d", s)                      # otherwise: na + virama
    r = _U.romanize_string(s, lcode="hin")
    for a, b in (("tth", "th"), ("ddh", "dh"), ("tt", "t"), ("dd", "d"), ("ii", "i"), ("uu", "u")):
        r = r.replace(a, b)
    r = re.sub(r"([bcdfghjklmnpqrstvwxyz]{2})a\b", r"\1", r)   # silent final schwa: "bainka" -> "baink"
    return re.sub(r"aa\b", "a", r)


def name_score(ent, en):
    cands = [ent.get("value") or "", romanize(ent["hi"])]
    return max(fuzz.partial_ratio(name_key(c), name_key(en)) for c in cands if c)


NAME_THRESHOLD = 85


# ---------------------------------------------------------------- Hindi side (no gold)
ARTICLE = N("\u090f\u0915")    # "ek" alone is usually the article ("ek request" = a request)
# ponytail: drops a genuine lone "ek" too ("ek hi baar"); context rule if that shows up in the clips


def hindi_entities(sent):
    """Auto-extracted entities for one Hindi sentence (used when no gold file exists).
    Two values of the same type (amount, digits, time) with a repair cue between them
    ("पाँच लाख… नहीं नहीं, पाँच करोड़", "नौ आठ सात सॉरी नहीं नौ आठ छह", "चार बजे… नहीं, साढ़े चार बजे"):
    the first is abandoned, the second corrected, and a नहीं between them is not a negation."""
    toks = hn.tokenize(sent)
    ents = [{"type": e.kind, "value": e.value, "span": (e.start, e.end), "status": "plain"} for e in hn.parse(sent)
            if not (e.kind == "AMT" and e.end - e.start == 1 and toks[e.start] == ARTICLE)]
    repair_toks = set()
    for a, b in zip(ents, ents[1:]):
        gap = range(a["span"][1], b["span"][0])
        words = [toks[i] for i in gap if re.match(r"\w", toks[i])]
        if a["type"] == b["type"] and a["type"] in ("AMT", "DIG", "TIME") and words and all(w in HI_REPAIR for w in words):
            a["status"], b["status"] = "abandoned", "corrected"
            repair_toks.update(gap)
    for e in ents:
        e["hi"] = " ".join(toks[e["span"][0]:e["span"][1]])
    n_neg = hi_negations(" ".join(t for i, t in enumerate(toks) if i not in repair_toks))
    ents += [{"type": "NEG", "hi": "", "value": 1, "status": "plain"}] * n_neg
    return ents


# ---------------------------------------------------------------- the verdict
def present(ent, en):
    t, v = ent["type"], ent["value"]
    if t == "AMT":
        return any(abs(x - v) < 1e-6 for x in en_amounts(en))
    if t == "RANGE":
        xs = en_amounts(en)
        return v[0] in xs and v[1] in xs
    if t == "DIG":
        return str(v) in en_digits(en)
    if t == "TIME":
        return v in en_times(en)
    if t == "DATE":
        return tuple(v) in en_dates(en)
    if t == "NAME":
        return name_score(ent, en) >= NAME_THRESHOLD
    if t == "AMBIG":
        return all(x in en_amounts(en) for x in v) or "".join(map(str, v)) in en_digits(en)
    raise ValueError(t)


def preserved(ent, en, version="fluent", n_neg=None):
    """Checker verdict for one entity in one English sentence.
    NEG: the English has at least as many negation cues as the Hindi (n_neg).
    abandoned values must be absent from the fluent version; the literal one keeps the disfluency."""
    if ent["type"] == "NEG":
        return en_negations(en) >= (n_neg if n_neg is not None else 1)
    if ent.get("status") == "abandoned":
        return present(ent, en) if version == "literal" else not present(ent, en)
    return present(ent, en)


if __name__ == "__main__":
    hi, en = sys.argv[1], sys.argv[2]
    ents = hindi_entities(hi)
    n_neg = sum(e["type"] == "NEG" for e in ents)
    for e in ents:
        print(e["type"], e.get("status"), e["value"], e["hi"], "->", preserved(e, en, n_neg=n_neg))
