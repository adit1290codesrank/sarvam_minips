"""Compare hand-marked word boundaries with the aligner's.

    python align/eval_boundaries.py            # all clips with align/hand_labels/clipNN.txt

Hand labels are an Audacity label export, one point label per boundary:
    1.234<TAB>1.234<TAB>17:s      (word 17's start;  "17:e" = its end)
Word indices are the "i" in align/out/clipNN.json (shown in the aligner's label track).
Error e = t_aligner - t_hand.
"""
import glob
import json
import os
import random
import re
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def roman_names():
    """(clip, word index) -> romanised word, from TODO.md, so output stays ASCII."""
    out, clip = {}, None
    path = os.path.join(ROOT, "align/hand_labels/TODO.md")
    for line in open(path, encoding="utf-8") if os.path.exists(path) else []:
        m = re.match(r"## (clip\d+)", line)
        clip = m.group(1) if m else clip
        m = re.search(r"\*\*(\d+):s\*\*\s+start of '([^']+)'", line)
        if m:
            out[(clip, int(m.group(1)))] = m.group(2)
    return out


ROMAN = roman_names()


def load_pairs():
    rows = []
    for hp in sorted(glob.glob(os.path.join(ROOT, "align/hand_labels/clip*.txt"))):
        clip = os.path.basename(hp)[:-4]
        words = json.load(open(os.path.join(ROOT, "align/out", clip + ".json"), encoding="utf-8"))
        for line in open(hp, encoding="utf-8"):
            p = line.split("\t")
            if len(p) < 3 or not p[0].strip():
                continue
            idx, edge = p[2].strip().split(":")
            w = words[int(idx)]
            t_al = w["start"] if edge == "s" else w["end"]
            rows.append((clip, ROMAN.get((clip, int(idx)), idx), edge, (t_al - float(p[0])) * 1000))
    return rows


def summary(errs):
    a = [abs(e) for e in errs]
    rng = random.Random(0)
    boots = sorted(st.mean(rng.choices(a, k=len(a))) for _ in range(2000))
    return {"n": len(a), "mean_abs_ms": st.mean(a), "median_abs_ms": st.median(a), "bias_ms": st.mean(errs),
            "le20": sum(x <= 20 for x in a) / len(a), "le50": sum(x <= 50 for x in a) / len(a),
            "le100": sum(x <= 100 for x in a) / len(a), "gt200": sum(x > 200 for x in a),
            "ci95": (boots[50], boots[1949])}


def table(rows):
    lines = ["| boundaries | n | mean abs ms | median abs ms | bias ms | ≤20 ms | ≤50 ms | ≤100 ms | >200 ms | 95% CI of mean (bootstrap) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, sel in (("starts", "s"), ("ends", "e"), ("all", None)):
        errs = [r[3] for r in rows if sel is None or r[2] == sel]
        if not errs:
            continue
        s = summary(errs)
        lines.append(f"| {name} | {s['n']} | {s['mean_abs_ms']:.0f} | {s['median_abs_ms']:.0f} | {s['bias_ms']:+.0f} | "
                     f"{s['le20']:.0%} | {s['le50']:.0%} | {s['le100']:.0%} | {s['gt200']} | "
                     f"{s['ci95'][0]:.0f}–{s['ci95'][1]:.0f} |")
    worst = sorted(rows, key=lambda r: -abs(r[3]))[:5]
    lines += ["", "Largest errors: " + "; ".join(f"{c} '{w}' {x:+.0f} ms" for c, w, e, x in worst)]
    return "\n".join(lines)


if __name__ == "__main__":
    rows = load_pairs()
    print(table(rows) if rows else "no hand labels in align/hand_labels/")
