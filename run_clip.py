"""One command for a new clip: convert -> align -> translate -> check -> wait-k timeline.

    python run_clip.py new.m4a --transcript new.txt       # Level-1 transcript, one sentence per line
    python run_clip.py new.m4a --asr-draft                # no transcript yet: print a Whisper draft and stop

Writes everything to out/<name>/: wavs, words.json, Audacity labels, translations, checks,
timeline.png. Entities come from the parser (amounts, digits, times, negations, repairs);
names need a gold file, so they are not checked here.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for d in ("align", "checker", "translate", "partE_ml", "tools"):
    sys.path.insert(0, os.path.join(ROOT, d))

from align_mms import align  # noqa: E402
from checker import hindi_entities, preserved, romanize  # noqa: E402
from convert import convert  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--transcript")
ap.add_argument("--asr-draft", action="store_true")
ap.add_argument("--k", type=int, default=3)
args = ap.parse_args()

name = os.path.splitext(os.path.basename(args.audio))[0]
out = os.path.join(ROOT, "out", name)
wav = convert(args.audio, out)
print(f"1. converted -> {wav[16000]}")

if args.asr_draft or not args.transcript:
    print("2. Whisper draft (fix to Level-1 style, save one sentence per line, rerun with --transcript):\n")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools/asr_draft.py"), wav[16000]], check=True)
    sys.exit()

words = align(wav[16000], args.transcript)
json.dump(words, open(os.path.join(out, "words.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(os.path.join(out, "labels.txt"), "w", encoding="utf-8") as f:
    f.writelines(f"{w['start']:.3f}\t{w['end']:.3f}\t{w['i']}:{w['w']}\n" for w in words)
print(f"2. aligned {len(words)} words -> {out}/words.json, labels.txt (Audacity: File > Import > Labels)")

from translate import mt, sentences  # noqa: E402

hi = sentences(args.transcript)
en = mt(hi)
print("3. translated (NLLB-200 1.3B) and 4. checked:")
checks = []
for s, (h, e) in enumerate(zip(hi, en)):
    ents = hindi_entities(h)
    n_neg = sum(x["type"] == "NEG" for x in ents)
    print(f"\n  [{s}] {h}\n      {e}")
    for x in ents:
        ok = preserved(x, e, "fluent", n_neg=n_neg)
        checks.append({"sent": s, "type": x["type"], "value": x["value"], "status": x["status"], "ok": ok})
        print(f"      {'OK  ' if ok else 'MISS'} {x['type']:5s} {x['status']:9s} {x['value']}")
json.dump({"hi": hi, "en": en, "checks": checks}, open(os.path.join(out, "checks.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\n   entities preserved: {sum(c['ok'] for c in checks)}/{len(checks)}")

from wait_k import DUR, holds_for, playback, run_sentence, save_cache  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
for ax, regime in zip(axes, ("prefix", "adaptive")):
    free = 0.0
    for w in words:
        ax.barh(1, w["end"] - w["start"], left=w["start"], color="#9bb7d4", edgecolor="white")
        ax.text(w["start"], 1.45, romanize(w["w"]), fontsize=6, rotation=45)   # matplotlib cannot shape Devanagari
    for s, h in enumerate(hi):
        sw = [w for w in words if w["sent"] == s]
        hw = [w["w"] for w in sw]
        ents = [dict(x, sent=s) for x in hindi_entities(h)]
        en_w, ready = run_sentence(hw, [w["end"] for w in sw], args.k, regime, holds_for(hw, ents), en[s].split())
        starts, free = playback(ready, free)
        for t, w in zip(starts, en_w):
            ax.barh(0, DUR, left=t, color="#f2b880", edgecolor="white")
            ax.text(t, -0.45, w, fontsize=6, rotation=-45, va="top")
    ax.set_yticks([0, 1], ["English", "Hindi"])
    ax.set_title(f"wait-{args.k}, {regime}", fontsize=9)
axes[-1].set_xlabel("seconds")
fig.tight_layout()
fig.savefig(os.path.join(out, "timeline.png"), dpi=150)
save_cache()
print(f"5. wait-{args.k} timeline -> {out}/timeline.png")
