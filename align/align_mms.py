"""Word timings for a Hindi Level-1 transcript with torchaudio's MMS_FA aligner.

    python align/align_mms.py clips/16k/clip01.wav transcripts/L1/clip01.txt align/out/clip01

Writes <out>.json  [{"i", "w", "sent", "start", "end"}, ...]  (seconds)
       <out>.labels.txt   Audacity label track (start<TAB>end<TAB>i:word)

MMS_FA only knows lowercase Latin, so each word is romanised with uroman. A word that
romanises to nothing (digits, stray symbols) gets the gap between its timed neighbours.
"""
import json
import re
import sys
import unicodedata

import soundfile as sf
import torch
import uroman
from torchaudio.pipelines import MMS_FA as B

PUNCT = r"[।॥?,.!…;:\"“”'‘’()\-]"


def read_words(path):
    """[(sentence_index, word)] from a one-sentence-per-line transcript."""
    out = []
    for s, line in enumerate(l for l in open(path, encoding="utf-8") if l.strip()):
        out += [(s, w) for w in re.sub(PUNCT, " ", unicodedata.normalize("NFC", line)).split()]
    return out


def align(wav_path, txt_path):
    wav, sr = sf.read(wav_path, dtype="float32")
    assert sr == B.sample_rate == 16000, f"{wav_path}: need 16 kHz mono, got {sr}"
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    words = read_words(txt_path)
    U = uroman.Uroman()
    rom = [re.sub(r"[^a-z']", "", U.romanize_string(w, lcode="hin").lower()) for _, w in words]
    keep = [i for i, r in enumerate(rom) if r]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = B.get_model().to(dev)
    x = torch.from_numpy(wav).unsqueeze(0).to(dev)
    with torch.inference_mode():
        em, _ = model(x)
    spans = B.get_aligner()(em[0].cpu(), B.get_tokenizer()([rom[i] for i in keep]))
    sec = x.size(1) / em.size(1) / sr                    # seconds per emission frame (~20 ms)
    dur = x.size(1) / sr
    t = {i: (s[0].start * sec, s[-1].end * sec) for i, s in zip(keep, spans)}
    for i in range(len(words)):                          # untimed word: gap between neighbours
        if i not in t:
            a = max((t[j][1] for j in t if j < i), default=0.0)
            b = min((t[j][0] for j in t if j > i), default=dur)
            t[i] = (a, max(a, b))
    out = [{"i": i, "w": w, "sent": s, "start": round(t[i][0], 3), "end": round(t[i][1], 3)}
           for i, (s, w) in enumerate(words)]
    # legality checks the dataset audit would apply
    assert all(0 <= o["start"] <= o["end"] <= dur + 1e-3 for o in out)
    assert all(a["start"] <= b["start"] for a, b in zip(out, out[1:]))
    return out


if __name__ == "__main__":
    wav_path, txt_path, out = sys.argv[1:4]
    words = align(wav_path, txt_path)
    json.dump(words, open(out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(out + ".labels.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{o['start']:.3f}\t{o['end']:.3f}\t{o['i']}:{o['w']}\n" for o in words)
    print(f"{out}.json: {len(words)} words, last ends {words[-1]['end']:.2f}s")
