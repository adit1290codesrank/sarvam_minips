"""Label checker/verdicts.csv one row at a time in the terminal (romanised, ASCII only).

    python3 tools/label_verdicts.py

Keys: 1 = kept, 0 = lost, s = skip (unsure), b = back one row, q = save and quit.
Every answer is saved immediately, so you can quit and resume any time.
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHEET = os.path.join(ROOT, "checker/verdicts.csv")


def question(r):
    t, st, ver = r["type"], r["status"], r["version"]
    if t == "NEG":
        return f"Hindi has {r['value']} negation(s). Does the English keep the NEGATIVE meaning?"
    if st == "abandoned":
        if ver == "literal":
            return f"{r['value']} was taken back by the speaker. LITERAL version: does it still say it (the false start)?"
        return f"{r['value']} was taken back by the speaker. Does the English correctly LEAVE IT OUT?  (1 = left out)"
    what = {"AMT": "amount", "DIG": "digits", "TIME": "time", "DATE": "date [day, month]", "NAME": "name"}.get(t, t)
    return f"Does the English keep the {what} {r['value']}  (any format is fine)?"


def main():
    rows = list(csv.DictReader(open(SHEET, encoding="utf-8")))
    cols = list(rows[0].keys())

    def save():
        with open(SHEET, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(rows)

    i = next((k for k, r in enumerate(rows) if r["human"].strip() == ""), len(rows))
    while i < len(rows):
        r = rows[i]
        done = sum(x["human"].strip() in ("0", "1") for x in rows)
        print("\n" + "-" * 78)
        print(f"[{i + 1}/{len(rows)}]  labelled {done}   {r['clip']} sentence {r['sent']}   system: {r['system']} {r['version']}")
        print(f"  Hindi said : {r['sent_roman']}")
        print(f"  Entity     : {r['type']}  '{r['hi_roman']}'  = {r['value']}")
        print(f"  English    : {r['en']}")
        print(f"\n  {question(r)}")
        a = input("  1 / 0 / s(kip) / b(ack) / q(uit) > ").strip().lower()
        if a in ("1", "0"):
            r["human"] = a
            save()
            i += 1
        elif a == "s":
            i += 1
        elif a == "b":
            i = max(0, i - 1)
        elif a == "q":
            break
    save()
    print(f"\nSaved. {sum(x['human'].strip() in ('0', '1') for x in rows)}/{len(rows)} labelled.")


if __name__ == "__main__":
    main()
