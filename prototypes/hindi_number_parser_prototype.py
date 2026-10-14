"""Hindi spoken-number parser (prototype for the Part D entity checker).

Turns Hindi number words in a Level-1 transcript into typed values:

    AMT    an amount            "ढाई लाख"                -> 250000
    DIG    a digit string       "नौ आठ डबल सात"          -> "9877"
    RANGE  an approximate range "दो तीन लाख"             -> (200000, 300000)
    TIME   a clock time         "साढ़े चार बजे"           -> "4:30"
    AMBIG  two bare numbers that could be a range, a digit pair or a count
                                "एक एक"                  -> (1, 1)

The 0-99 word tables are vendored from NeMo text-processing 1.2.0
(data/numbers/*.tsv, Apache-2.0, see data/NEMO_LICENSE).

Usage:
    python hindi_number_parser_prototype.py "पाँच लाख नहीं नहीं पाँच करोड़"
    python hindi_number_parser_prototype.py            # runs the demo phrases
    python hindi_number_parser_prototype.py --data DIR "..."   # other tables
"""
import argparse
import os
import re
import unicodedata
from collections import namedtuple

Entity = namedtuple("Entity", "kind value start end")  # start/end: token indices, end exclusive

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def N(s):
    # NFC makes the two Unicode spellings of nukta letters (ड़ / ड़) identical
    return unicodedata.normalize("NFC", s)


DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
EN = {N(k): v for k, v in {"ज़ीरो": 0, "जीरो": 0, "वन": 1, "टू": 2, "थ्री": 3, "फोर": 4, "फ़ोर": 4,
                           "फाइव": 5, "सिक्स": 6, "सेवन": 7, "एट": 8, "नाइन": 9}.items()}
HOMOGRAPH = {N(w) for w in ("वन", "टू", "एट", "फोर", "फ़ोर")}  # also "forest", "to", "at", "for"
REPEAT = {N("डबल"): 2, N("ट्रिपल"): 3}
MULT = {N(k): v for k, v in {"सौ": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 10**5,
                             "करोड़": 10**7, "करोड": 10**7, "अरब": 10**9}.items()}
CAN_START = {N("सौ"), N("हज़ार"), N("हजार")}            # "हज़ार रुपये" = 1000
FRAC = {N("साढ़े"): 0.5, N("सवा"): 0.25, N("पौने"): -0.25}  # prefixes: साढ़े X = X + 0.5
SPEC = {N("डेढ़"): 1.5, N("ढाई"): 2.5}
DECIMAL = N("दशमलव")                                 # "दो दशमलव पाँच" = 2.5
OCLOCK = {N("बजे"), N("बज")}
OCLOCK_AND = N("बजकर")
MINUTE = {N("मिनट")}

W = {}  # Hindi word -> 0..99, filled by load_tables()


def load_tables(data_dir=DATA_DIR):
    W.clear()
    for name in ("digit.tsv", "teens_and_ties.tsv"):
        with open(os.path.join(data_dir, "numbers", name), encoding="utf8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 2:
                    W[N(p[1])] = int(p[0].translate(DEV_DIGITS))
    W[N("शून्य")] = 0


def tokenize(text):
    text = N(text).translate(DEV_DIGITS)
    text = re.sub(r"[-\"'“”‘’()\[\]{}]", " ", text)   # "आठ-दस" -> "आठ दस"; quotes/brackets are transparent
    return re.findall(r"[।॥,.?!…;:]|[^\s।॥,.?!…;:]+", text)


def num(tok):
    """Value of a bare number token (word, English digit word or digits), else None."""
    if tok in W:
        return W[tok]
    if tok in EN:
        return EN[tok]
    if tok.isdigit():
        return int(tok)
    return None


def clean(v):
    return int(v) if float(v).is_integer() else v


def read_run(toks, i):
    """Collect consecutive bare numbers; डबल/ट्रिपल repeat the next one.
    Returns (values, end, forced_digits)."""
    vals, forced = [], False
    while i < len(toks):
        t = toks[i]
        if t in REPEAT and i + 1 < len(toks) and num(toks[i + 1]) is not None:
            vals += [num(toks[i + 1])] * REPEAT[t]
            forced, i = True, i + 2
        elif num(t) is not None:
            forced = forced or t in EN or (t.isdigit() and len(t) >= 3)
            vals.append(num(t))
            i += 1
        else:
            break
    return vals, i, forced


def read_amount(toks, i):
    """Read "[frac] number [सौ] [number] multiplier ..." left to right.
    After a number, a multiplier that is not smaller than the previous one starts a
    new amount, so "पाँच लाख पाँच करोड़" gives two amounts. Directly after another
    multiplier, a larger one scales the whole amount ("पाँच सौ करोड़" = 500 crore,
    "एक हज़ार करोड़" = 1000 crore), and a repeated one is distributive ("सौ सौ रुपये"
    = 100 each). Returns (amounts, end, big_mult_used)."""
    amounts, total, grp, frac = [], 0, 0, None
    last_mult, prev, big = None, None, False      # prev: kind of the previous token
    while i < len(toks):
        t = toks[i]
        if t in FRAC:
            if prev == "num":
                break
            frac, prev = FRAC[t], "frac"
        elif t in SPEC:
            if prev == "num":
                break
            grp, prev = grp + SPEC[t], "num"
        elif t == DECIMAL and prev == "num":
            k = i + 1
            while k < len(toks) and num(toks[k]) is not None and num(toks[k]) < 10:
                k += 1
            digits = "".join(str(num(w)) for w in toks[i + 1:k])
            if not digits:
                break
            grp += int(digits) / 10 ** len(digits)
            i = k
            continue
        elif num(t) is not None:
            if prev == "num":
                break                               # two numbers in a row: not one amount
            grp += num(t) + (frac or 0)
            frac, prev = None, "num"
        elif t in MULT and (prev in ("num", "frac", "mult") or (prev is None and t in CAN_START)):
            m = MULT[t]
            if prev == "mult":
                before = MULT[toks[i - 1]]
                if m == before:                     # "सौ सौ", "हज़ार हज़ार": distributive
                    i += 1
                    continue
                if m < before:
                    break
                total, grp, last_mult, big = (total + grp) * m, 0, m, True
                i += 1
                continue
            if prev is None:
                grp = 1
            elif prev == "frac":                    # "सवा लाख" = 1.25 lakh, "सवा सौ" = 125
                grp, frac = 1 + frac, None
            if m == 100:
                grp *= 100
            else:
                if last_mult is not None and m >= last_mult:
                    amounts.append(clean(total))
                    total = 0
                total, grp, last_mult, big = total + grp * m, 0, m, True
            prev = "mult"
        else:
            break
        i += 1
    if prev == "frac":                              # dangling "साढ़े" with no number: leave it
        i -= 1
    if prev is not None and (prev != "frac" or last_mult is not None):
        amounts.append(clean(total + grp))
    return amounts, i, big


def as_time(v, minutes=None):
    h = int(v)
    m = minutes if minutes is not None else round((v - h) * 60)
    return f"{h or 12}:{m:02d}"                     # "पौने एक बजे" = 12:45


def parse(text):
    if not W:
        load_tables()
    toks = tokenize(text)
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if num(t) is None and t not in REPEAT and t not in FRAC and t not in SPEC and t not in CAN_START:
            i += 1
            continue
        vals, j, forced = read_run(toks, i)
        nxt = toks[j] if j < len(toks) else None
        run = toks[i:j]
        if run and len(vals) < 4 and all(w in EN for w in run) and any(w in HOMOGRAPH for w in run):
            i = j                                       # "वन विभाग", "पेमेंट टू", "वन टू वन मीटिंग"
            continue
        if len(vals) >= 3 or (len(vals) >= 2 and forced):
            out.append(Entity("DIG", "".join(str(v) for v in vals), i, j))
            i = j
            continue
        if len(vals) == 2:
            a, b = vals
            if a == b and nxt in MULT:                  # "दो दो सौ रुपये" = 200 each
                amounts, j, _ = read_amount(toks, i + 1)
                out += [Entity("AMT", v, i, j) for v in amounts]
                i = j
                continue
            if b > a:                                   # "दो तीन लाख", "आठ नौ लोग"
                m = 1
                if nxt in MULT:
                    m, j = MULT[nxt], j + 1
                out.append(Entity("RANGE", (a * m, b * m), i, j))
            else:                                       # "एक एक", "नौ आठ"
                out.append(Entity("AMBIG", (a, b), i, j))
            i = j
            continue
        if not vals and t in REPEAT:                    # "डबल बेड": not a number
            i += 1
            continue
        amounts, j, big = read_amount(toks, i)
        if not amounts:
            i += 1
            continue
        nxt = toks[j] if j < len(toks) else None
        if len(amounts) == 1 and not big and nxt in OCLOCK:
            out.append(Entity("TIME", as_time(amounts[0]), i, j + 1))
            i = j + 1
            continue
        if len(amounts) == 1 and not big and nxt == OCLOCK_AND:
            mins, k, _ = read_amount(toks, j + 1)
            if len(mins) == 1 and k < len(toks) and toks[k] in MINUTE:
                out.append(Entity("TIME", as_time(amounts[0], int(mins[0])), i, k + 1))
                i = k + 1
                continue
            out.append(Entity("TIME", as_time(amounts[0]), i, j + 1))
            i = j + 1
            continue
        for a in amounts:
            out.append(Entity("AMT", a, i, j))
        i = j
    return out


DEMO = ["पच्चीस हज़ार रुपये", "ढाई लाख", "साढ़े तीन करोड़", "पौने दो लाख", "सवा पाँच सौ",
        "एक करोड़ बीस लाख पचास हज़ार", "नंबर है नौ आठ सात छह पाँच चार तीन दो एक शून्य",
        "नाइन एट डबल सात फाइव", "पाँच लाख… नहीं नहीं, पाँच करोड़।", "दो तीन लाख",
        "साढ़े चार बजे", "एक एक", "आठ नौ लोग आए", "पौने दो", "ट्रिपल सात", "डबल बेड चाहिए"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("text", nargs="*", help="Hindi text (default: demo phrases)")
    ap.add_argument("--data", default=DATA_DIR, help="folder containing numbers/*.tsv")
    args = ap.parse_args()
    load_tables(args.data)
    for s in [" ".join(args.text)] if args.text else DEMO:
        print(s, "->", [(e.kind, e.value) for e in parse(s)])
