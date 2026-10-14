"""Checker tests.   python tests/test_checker.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checker"))
from checker import en_amounts, en_dates, en_digits, en_negations, en_times, hi_negations, hindi_entities, preserved  # noqa: E402

AMOUNTS = [
    ("five crore", [50000000]), ("₹5 crore", [50000000]), ("50 million", [50000000]),
    ("Rs. 5,00,000", [500000]), ("500,000 rupees", [500000]), ("2.5 crore", [25000000]),
    ("two point five crore", [25000000]), ("three and a half lakh", [350000]),
    ("one and a half crore", [15000000]), ("a lakh", [100000]), ("half a lakh", [50000]),
    ("twenty five thousand", [25000]), ("one crore twenty lakh fifty thousand", [12050000]),
    ("five lakh... no no, five crore", [500000, 50000000]), ("two hundred and five", [205]),
    ("five hundred crore", [5000000000]), ("it was cut twice", [2]), ("on the third and fifth", [3, 5]), ("5k", [5000]), ("1.25 lakh", [125000]),
]
for text, want in AMOUNTS:
    assert en_amounts(text) == want, (text, en_amounts(text), want)

assert en_digits("my number is nine eight double seven 5") == "98775"
assert en_digits("98765 43210") == "9876543210"
assert en_digits("triple 3") == "333"
assert "4:30" in en_times("at 4:30 pm") and "4:30" in en_times("half past four")
assert "4:30" in en_times("at 4. 30 pm")
assert "4:45" in en_times("quarter to five") and "12:45" in en_times("quarter to one")
assert (15, 8) in en_dates("on 15th August") and (15, 8) in en_dates("August fifteen")
assert en_negations("no no, five crore") == 0 and en_negations("it did not close") == 1
assert en_negations("don't send it") == 1

assert hi_negations("अकाउंट बंद नहीं हुआ।") == 1
assert hi_negations("आप भेज देंगे ना?") == 0                     # tag question
assert hi_negations("पाँच लाख… नहीं नहीं, पाँच करोड़।") == 0      # repair, not negation
assert hi_negations("मैसेज मत भेजना।") == 1
assert hi_negations("अंकित ने बोला था कि इट वोन्ट बी अ प्रॉब्लम, पर अभी भी कुछ नहीं हुआ।") == 2   # code-mixed

ents = hindi_entities("लोन पाँच लाख… नहीं नहीं, पाँच करोड़ का है।")
amts = [(e["value"], e["status"]) for e in ents if e["type"] == "AMT"]
assert amts == [(500000, "abandoned"), (50000000, "corrected")], amts
ab, co = [e for e in ents if e["type"] == "AMT"]
assert preserved(co, "The loan is five crore.") and preserved(ab, "The loan is five crore.")
assert not preserved(ab, "The loan is five lakh, I mean five crore.")              # fluent leaks it
assert preserved(ab, "five lakh... no no, five crore", version="literal")

ents = hindi_entities("नंबर है नौ आठ सात सॉरी नहीं नौ आठ छह डबल चार तीन दो नाइन एक ज़ीरो।")
assert [(e["type"], e["value"], e["status"]) for e in ents] == [("DIG", "987", "abandoned"), ("DIG", "9864432910", "corrected")], ents
ents = hindi_entities("साइट विज़िट चार बजे… नहीं, साढ़े चार बजे रख लीजिए।")
assert [(e["type"], e["value"], e["status"]) for e in ents] == [("TIME", "4:00", "abandoned"), ("TIME", "4:30", "corrected")], ents
ents = hindi_entities("पाँच लाख नहीं मिला, दस लाख मिला।")         # a real negation between two amounts
assert sum(e["type"] == "NEG" for e in ents) == 1, ents

assert not [e for e in hindi_entities("\u092e\u0948\u0902\u0928\u0947 \u090f\u0915 \u0930\u093f\u0915\u094d\u0935\u0947\u0938\u094d\u091f \u0921\u093e\u0932\u0940\u0964") if e["type"] == "AMT"]   # "maine ek request daali": article

name = {"type": "NAME", "hi": "राहुल शर्मा", "value": "Rahul Sharma"}
assert preserved(name, "Book it for Rahul Sharma.") and preserved(name, "for Raahul Sharmaa")
assert not preserved(name, "Book it for my brother.")
print("OK: checker tests passed")
