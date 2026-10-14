"""Score the checker against human verdicts.

    python checker/eval_checker.py draft-gold   # checker/gold/clipNN.json from the parser (EDIT BY HAND)
    python checker/eval_checker.py sheet        # checker/verdicts.csv: one row per (entity, English output)
    python checker/eval_checker.py eval         # accuracy / precision / recall per entity type

Gold entity: {"sent": 2, "type": "AMT|DIG|TIME|DATE|NAME|RANGE|NEG", "hi": <Hindi span>,
              "value": 50000000, "status": "plain|abandoned|corrected"}
  NAME value = English spelling ("Rahul"); DATE value = [day, month]; NEG value = 1.
verdicts.csv is labelled blind: it has no checker column. Fill "human" with
  1 = the English keeps this entity (for an abandoned value in a non-literal version:
      the English correctly leaves it out), 0 = it does not.
All negations of a sentence share one row (value = how many). Regenerating the sheet
keeps the labels already filled in.
"""
import csv
import glob
import json
import os
import sys

from checker import hindi_entities, preserved, romanize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "checker/gold")
SHEET = os.path.join(ROOT, "checker/verdicts.csv")
COLS = ["clip", "sent", "system", "version", "type", "value", "status", "hi_roman", "sent_roman", "en", "human"]


def clips():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "transcripts/L1/clip*.txt")))


def draft_gold():
    os.makedirs(GOLD, exist_ok=True)
    for c in clips():
        path = os.path.join(GOLD, c + ".json")
        if os.path.exists(path):
            print(f"{path} exists, not overwriting")
            continue
        sents = [l.strip() for l in open(os.path.join(ROOT, "transcripts/L1", c + ".txt"), encoding="utf-8") if l.strip()]
        ents = [{"sent": s, "type": e["type"], "hi": e["hi"], "value": e["value"], "status": e["status"]}
                for s, h in enumerate(sents) for e in hindi_entities(h)]
        with open(path, "w", encoding="utf-8") as f:
            f.write("[\n" + ",\n".join(json.dumps(e, ensure_ascii=False) for e in ents) + "\n]\n")
        print(f"{path}: {len(ents)} entities drafted -- add NAME/DATE, fix types and statuses by hand")


def rows():
    """One row per (gold entity, English output), negations collapsed per sentence, with the checker verdict."""
    trans = [json.loads(l) for l in open(os.path.join(ROOT, "translations.jsonl"), encoding="utf-8")]
    out = []
    for c in clips():
        gold = json.load(open(os.path.join(GOLD, c + ".json"), encoding="utf-8"))
        sents = [l.strip() for l in open(os.path.join(ROOT, "transcripts/L1", c + ".txt"), encoding="utf-8") if l.strip()]
        for t in (t for t in trans if t["clip"] == c):
            ents = [e for e in gold if e["sent"] == t["sent"]]
            n_neg = sum(e["type"] == "NEG" for e in ents)
            ents = [e for e in ents if e["type"] != "NEG"] + ([{"type": "NEG", "value": n_neg, "hi": sents[t["sent"]]}] if n_neg else [])
            ver = "literal" if t["version"] == "literal" else "fluent"
            for e in ents:
                out.append({"clip": c, "sent": t["sent"], "system": t["system"], "version": t["version"],
                            "type": e["type"], "value": json.dumps(e["value"], ensure_ascii=False),
                            "status": e.get("status", "plain"), "hi_roman": romanize(e["hi"]),
                            "sent_roman": romanize(t["hi"]), "en": t["en"],
                            "checker": int(preserved(e, t["en"], ver, n_neg=n_neg)), "human": ""})
    return out


def key(r):
    return (r["clip"], str(r["sent"]), r["system"], r["version"], r["type"], r["value"], r["status"])


def sheet():
    old = {}
    if os.path.exists(SHEET):
        old = {key(r): r["human"] for r in csv.DictReader(open(SHEET, encoding="utf-8"))}
    rs = sorted(rows(), key=lambda r: (r["clip"], r["sent"], r["type"], r["value"], r["system"], r["version"]))
    for r in rs:
        r["human"] = old.get(key(r), "")
    with open(SHEET, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, COLS, extrasaction="ignore")       # no checker column: label blind
        w.writeheader()
        w.writerows(rs)
    print(f"{SHEET}: {len(rs)} rows, {sum(r['human'] != '' for r in rs)} labelled")


def evaluate():
    human = {key(r): r["human"].strip() for r in csv.DictReader(open(SHEET, encoding="utf-8"))}
    rs = rows()
    for r in rs:
        r["human"] = human.get(key(r), "")
    lab = [r for r in rs if r["human"] in ("0", "1")]
    f = lambda a, b: f"{a / b:.0%}" if b else "–"  # noqa: E731
    lines = ["| entity type | n | checker accuracy | precision | recall | TP | FP | FN | TN |", "|---|---|---|---|---|---|---|---|---|"]
    for t in sorted({r["type"] for r in lab}) + ["all"]:
        sel = [r for r in lab if t == "all" or r["type"] == t]
        tp = sum(r["checker"] == 1 and r["human"] == "1" for r in sel)
        fp = sum(r["checker"] == 1 and r["human"] == "0" for r in sel)
        fn = sum(r["checker"] == 0 and r["human"] == "1" for r in sel)
        tn = len(sel) - tp - fp - fn
        lines.append(f"| {t} | {len(sel)} | {f(tp + tn, len(sel))} | {f(tp, tp + fp)} | {f(tp, tp + fn)} | {tp} | {fp} | {fn} | {tn} |")
    lines += ["", "| system × version | entities | preserved (checker) | preserved (human) |", "|---|---|---|---|"]
    for sv in sorted({(r["system"], r["version"]) for r in rs}):
        sel = [r for r in rs if (r["system"], r["version"]) == sv]
        hl = [r for r in sel if r["human"] in ("0", "1")]
        hum = f(sum(r["human"] == "1" for r in hl), len(hl)) if hl else "–"
        lines.append(f"| {sv[0]} {sv[1]} | {len(sel)} | {f(sum(r['checker'] for r in sel), len(sel))} | {hum} |")
    dis = [r for r in lab if str(r["checker"]) != r["human"]]
    lines += ["", f"Checker disagrees with human on {len(dis)} of {len(lab)} labelled rows:", ""]
    lines += [f"- {r['clip']}:{r['sent']} {r['system']} {r['version']} {r['type']} {r['value']} ({r['hi_roman'][:40]}) -> "
              f"checker {r['checker']}, human {r['human']}: “{r['en']}”" for r in dis]
    print("\n".join(lines))


if __name__ == "__main__":
    {"draft-gold": draft_gold, "sheet": sheet, "eval": evaluate}[sys.argv[1]]()
