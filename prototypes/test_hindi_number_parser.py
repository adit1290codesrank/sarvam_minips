"""Tests for hindi_number_parser_prototype.py.

Run:  python prototypes/test_hindi_number_parser.py     (or: pytest prototypes/)
"""
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hindi_number_parser_prototype as hn  # noqa: E402  (importing must not need CLI args)

CASES = [
    # the original 12 prototype phrases
    ("पच्चीस हज़ार रुपये", [("AMT", 25000)]),
    ("ढाई लाख", [("AMT", 250000)]),
    ("साढ़े तीन करोड़", [("AMT", 35000000)]),
    ("पौने दो लाख", [("AMT", 175000)]),
    ("सवा पाँच सौ", [("AMT", 525)]),
    ("एक करोड़ बीस लाख पचास हज़ार", [("AMT", 12050000)]),
    ("नंबर है नौ आठ सात छह पाँच चार तीन दो एक शून्य", [("DIG", "9876543210")]),
    ("नाइन एट डबल सात फाइव", [("DIG", "98775")]),
    ("पाँच लाख… नहीं नहीं, पाँच करोड़।", [("AMT", 500000), ("AMT", 50000000)]),
    ("सत्तासी हज़ार उनहत्तर", [("AMT", 87069)]),
    ("निन्यानवे", [("AMT", 99)]),
    ("अट्ठानवे लाख", [("AMT", 9800000)]),
    # bugs found in the first prototype
    ("दो तीन लाख", [("RANGE", (200000, 300000))]),      # was 500000 (summed)
    ("आठ नौ लोग आए", [("RANGE", (8, 9))]),
    ("आठ-दस लोग", [("RANGE", (8, 10))]),
    ("नौ आठ लोग आए", [("AMBIG", (9, 8))]),              # was 17
    ("एक एक", [("AMBIG", (1, 1))]),                     # was digit string "11"
    ("साढ़े चार बजे", [("TIME", "4:30")]),               # was 4
    ("पौने पाँच बजे", [("TIME", "4:45")]),
    ("ढाई बजे", [("TIME", "2:30")]),
    ("चार बजकर बीस मिनट", [("TIME", "4:20")]),
    ("पौने दो", [("AMT", 1.75)]),                        # was DIG "2"
    ("डेढ़ सौ", [("AMT", 150)]),
    ("ट्रिपल सात", [("DIG", "777")]),                    # was DIG "7"
    ("डबल बेड चाहिए", []),                              # used to crash (ValueError)
    # other checks
    ("पाँच लाख पाँच करोड़", [("AMT", 500000), ("AMT", 50000000)]),  # rising unit: new amount
    ("पाँच करोड़ पाँच लाख", [("AMT", 50500000)]),
    ("दो सौ पचास करोड़", [("AMT", 2500000000)]),
    ("एक लाख साढ़े तीन हज़ार", [("AMT", 103500)]),
    ("तिहत्तर", [("AMT", 73)]),
    ("इक्यावन", [("AMT", 51)]),
    ("पांच लाख", [("AMT", 500000)]),                    # spelling without chandrabindu
    ("5 लाख", [("AMT", 500000)]),
    ("हज़ार रुपये", [("AMT", 1000)]),
    ("शून्य रुपये", [("AMT", 0)]),
    ("नौ आठ सात… सॉरी, नौ आठ छह", [("DIG", "987"), ("DIG", "986")]),
    ("मैंने खाता बंद नहीं करवाया", []),
    # bugs found by the adversarial audit of the rewrite
    ("पाँच सौ करोड़", [("AMT", 5000000000)]),            # was 500
    ("सौ करोड़", [("AMT", 1000000000)]),
    ("डेढ़ सौ करोड़", [("AMT", 1500000000)]),
    ("एक हज़ार करोड़", [("AMT", 10000000000)]),
    ("दो हज़ार पाँच सौ करोड़", [("AMT", 25000000000)]),
    ("सवा सौ", [("AMT", 125)]),                          # was 100
    ("सवा लाख", [("AMT", 125000)]),                      # was nothing
    ("पौने सौ", [("AMT", 75)]),
    ("पाँच करोड", [("AMT", 50000000)]),                  # करोड without nukta, was 5
    ("“पाँच लाख”", [("AMT", 500000)]),                   # quotes, was nothing
    ("(पाँच लाख)", [("AMT", 500000)]),
    ("'पाँच' लाख", [("AMT", 500000)]),
    ("वन विभाग में", []),                                # वन = forest, was 1
    ("पेमेंट टू मिस्टर शर्मा", []),                         # टू = to, was 2
    ("वन टू वन मीटिंग", []),                             # one-to-one meeting, was "121"
    ("नाइन एट सेवन सिक्स", [("DIG", "9876")]),            # English digit words still work
    ("पौने एक बजे", [("TIME", "12:45")]),                # was 0:45
    ("दो दो सौ रुपये", [("AMT", 200)]),                  # distributive, 200 each
    ("सौ सौ रुपये", [("AMT", 100)]),
    ("ओटीपी है ट्रिपल ज़ीरो फोर", [("DIG", "0004")]),
    ("दो दशमलव पाँच करोड़", [("AMT", 25000000)]),        # decimals, was 2 and 50000000
    ("साढ़े तीन दशमलव", [("AMT", 3.5)]),                # dangling दशमलव is ignored
]


def got(text):
    return [(e.kind, e.value) for e in hn.parse(text)]


def test_cases():
    for text, want in CASES:
        assert got(text) == want, f"{text}: got {got(text)}, want {want}"


def test_amounts_are_int_when_whole():
    assert all(type(e.value) is int for e in hn.parse("ढाई लाख, साढ़े तीन करोड़, पौने दो लाख"))


def test_nukta_unicode_forms_match():
    composed = "\u0915\u0930\u094b\u095c"            # करोड़ with ड़ as one code point (U+095C)
    decomposed = "\u0915\u0930\u094b\u0921\u093c"    # ड (U+0921) + nukta (U+093C)
    assert composed != decomposed
    assert unicodedata.normalize("NFC", composed) == unicodedata.normalize("NFC", decomposed)
    assert got("पाँच " + composed) == got("पाँच " + decomposed) == [("AMT", 50000000)]


def test_token_spans():
    e = hn.parse("मेरा लोन पाँच करोड़ का है")[0]
    assert (e.start, e.end) == (2, 4)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
    print(f"OK: {len(CASES)} phrases and {len(tests)} tests passed")
