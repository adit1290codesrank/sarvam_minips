"""Part E (ML track): simulate wait-k on the clips' real Hindi timeline, score entities, plot.

    python partE_ml/wait_k.py                 # all clips with align/out + gold -> partE_ml/out/
    python partE_ml/wait_k.py --test          # unit test against the Part A worked example

Timeline: Hindi word j ends at its MMS_FA end time. English word i is *ready* when the
policy has read g(i) Hindi words (plus compute delay c), and playback is serialised as in
PS §5.1: start_i = max(ready_i, end_{i-1}); every English word lasts DUR seconds.

Regimes (offline analyses on gold transcripts, not a submittable system):
  oracle    NLLB's full-sentence translation, re-timed with g(i) = min(k+i-1, J): entity
            accuracy is fixed, only lag moves. Same MT model as the other two regimes, so
            the regimes differ only in policy
  prefix    test-time wait-k: after reading g words, NLLB continues the committed English
            from the Hindi prefix and commits one more word. No take-backs.
  adaptive  prefix, but never write while inside a gold hold span (an entity, a
            self-repair from the abandoned value to the corrected one, or the three words
            before a negation cue): read to the end of the span first
"""
import argparse
import glob
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "checker"))
sys.path.insert(0, os.path.join(ROOT, "translate"))
from checker import hindi_entities, hi_negations, preserved  # noqa: E402

DUR, C = 0.35, 0.0          # seconds per English word, compute delay
KS = [1, 2, 3, 4, 5, 7, 10]
CACHE = os.path.join(ROOT, "partE_ml/cache.json")
OUT = os.path.join(ROOT, "partE_ml/out")


# ---------------------------------------------------------------- latency (SimulEval logic)
def al(delays, X, ref_len, laal=False):
    """Average Lagging. delays: when each target word was emitted; X: source length."""
    if delays[0] > X:
        return delays[0]
    gamma = (max(len(delays), ref_len) if laal else ref_len) / X
    tot = 0.0
    for t, d in enumerate(delays):
        tot += d - t / gamma
        if d >= X:
            return tot / (t + 1)
    return tot / len(delays)


def playback(ready, free=0.0):
    starts = []
    for r in ready:
        s = max(r, free)
        starts.append(s)
        free = s + DUR
    return starts, free


def test():
    src_end = list(range(1, 10))                       # 9 source words, word j ends at j
    for ps, want in ((False, 2.625), (True, 3.6875)):
        ready = [src_end[min(3 + i + (1 if ps else 0), 9) - 1] for i in range(8)]
        assert abs(al(ready, 9, 8) - want) < 1e-9, (ps, al(ready, 9, 8))
    print("OK: wait-3 AL = 2.625, and k+i indexing gives 3.6875")


# ---------------------------------------------------------------- NLLB with a forced English prefix
_cache = None


def continue_en(hi, committed, one_word):
    """English continuation of `committed` given Hindi `hi`. one_word: return the next word
    (None if the model ends the sentence), else the full remainder as a word list."""
    global _cache
    if _cache is None:
        _cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    key = json.dumps([hi, committed])
    if key not in _cache:
        import torch
        from translate import load
        tok, model, dev = load("nllb")
        start = [model.config.decoder_start_token_id, tok.convert_tokens_to_ids("eng_Latn")]
        pre = tok(text_target=" ".join(committed), add_special_tokens=False).input_ids if committed else []
        x = tok(hi, return_tensors="pt").to(dev)
        dec = torch.tensor([start + pre], device=dev)
        with torch.inference_mode():
            y = model.generate(**x, decoder_input_ids=dec, num_beams=1, do_sample=False, max_new_tokens=120)
        _cache[key] = tok.decode(y[0, dec.size(1):], skip_special_tokens=True).strip()
    words = _cache[key].split()
    return (words[0] if words else None) if one_word else words


def save_cache():
    if _cache is not None:
        json.dump(_cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=0, sort_keys=True)


# ---------------------------------------------------------------- policies
def run_sentence(hi_words, ends, k, regime, holds, full_en):
    """Returns (english words, ready times). hi_words/ends: this sentence's Hindi words and end
    times; holds: list of (a, b) word-index spans (0-based, inclusive) to read through."""
    J = len(hi_words)
    if regime == "oracle":
        return full_en, [ends[min(k + i, J) - 1] + C for i in range(len(full_en))]

    def extend(g):                                    # never stop reading inside a hold span
        if regime == "adaptive":
            for a, b in holds:
                if a < g <= b:                        # read words 0..g-1; span not finished
                    g = b + 1
        return min(g, J)

    out, ready, g = [], [], extend(min(k, J))
    while len(out) < 3 * J + 5:
        if g >= J:
            rest = continue_en(" ".join(hi_words), out, one_word=False)
            out += rest
            ready += [ends[-1] + C] * len(rest)
            break
        w = continue_en(" ".join(hi_words[:g]), out, one_word=True)
        if w is not None:
            out.append(w)
            ready.append(ends[g - 1] + C)
        g = extend(g + 1)
    return out, ready


def token_span(sent_words, hi):
    """Word indices of phrase `hi` inside the sentence (first match), or None."""
    target = [unicodedata.normalize("NFC", w) for w in re.sub(r"[।॥?,.!…;:]", " ", hi).split()]
    ws = [unicodedata.normalize("NFC", w) for w in sent_words]
    for s in range(len(ws) - len(target) + 1):
        if target and ws[s:s + len(target)] == target:
            return s, s + len(target) - 1
    return None


def load_clip(clip):
    words = json.load(open(os.path.join(ROOT, "align/out", clip + ".json"), encoding="utf-8"))
    sents = [l.strip() for l in open(os.path.join(ROOT, "transcripts/L1", clip + ".txt"), encoding="utf-8") if l.strip()]
    gp = os.path.join(ROOT, "checker/gold", clip + ".json")
    if os.path.exists(gp):
        gold = json.load(open(gp, encoding="utf-8"))
    else:
        gold = [dict(e, sent=s) for s, h in enumerate(sents) for e in hindi_entities(h)]
    full = {}
    for line in open(os.path.join(ROOT, "translations.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r["clip"] == clip and r["system"] == "nllb":
            full[r["sent"]] = r["en"]
    return words, sents, gold, full


def holds_for(sent_words, ents):
    spans = []
    for e in ents:
        if e["type"] == "NEG":
            continue
        sp = token_span(sent_words, e["hi"])
        if sp:
            spans.append(sp)
    amts = [(e, token_span(sent_words, e["hi"])) for e in ents if e.get("status") in ("abandoned", "corrected")]
    for (a, sa), (b, sb) in zip(amts, amts[1:]):      # repair: hold from abandoned to corrected
        if a["status"] == "abandoned" and b["status"] == "corrected" and sa and sb:
            spans.append((sa[0], sb[1]))
    base = hi_negations(" ".join(sent_words))          # verb group before each negation cue:
    for j in range(len(sent_words)):                   # word j is a cue if dropping it lowers the count
        if hi_negations(" ".join(sent_words[:j] + sent_words[j + 1:])) < base:
            spans.append((max(0, j - 3), j))
    return spans


def simulate(clips, ks=KS):
    rows = []
    for regime in ("oracle", "prefix", "adaptive"):
        for k in ks:
            tot = ok = leak = ab = 0
            ent_by_type = {}
            als, laals, endoffs, startoffs, pauses = [], [], [], [], []
            for clip in clips:
                words, sents, gold, full = clip_data[clip]
                free, first_start, last_end, clip_starts = 0.0, None, words[-1]["end"], []
                for s, hi in enumerate(sents):
                    sw = [w for w in words if w["sent"] == s]
                    hw, ends = [w["w"] for w in sw], [w["end"] for w in sw]
                    ents = [e for e in gold if e["sent"] == s]
                    en, ready = run_sentence(hw, ends, k, regime, holds_for(hw, ents), full[s].split())
                    if not en:
                        continue
                    starts, free = playback(ready, free)
                    clip_starts += starts
                    first_start = starts[0] if first_start is None else first_start
                    t0, X = sw[0]["start"], sw[-1]["end"] - sw[0]["start"]
                    d = [x - t0 for x in starts]
                    ref = len(full[s].split())
                    als.append(al(d, X, ref))
                    laals.append(al(d, X, ref, laal=True))
                    text = " ".join(en)
                    n_neg = sum(e["type"] == "NEG" for e in ents)
                    for e in ents:
                        if e.get("status") == "abandoned":
                            ab += 1
                            leak += not preserved(e, text, "fluent")
                            continue
                        good = preserved(e, text, "fluent", n_neg=n_neg)
                        tot, ok = tot + 1, ok + good
                        t = ent_by_type.setdefault(e["type"], [0, 0])
                        t[0], t[1] = t[0] + 1, t[1] + good
                pauses.append(sum(b - a - DUR > 0.25 for a, b in zip(clip_starts, clip_starts[1:])))
                startoffs.append(first_start - words[0]["start"])
                endoffs.append(free - last_end)
            mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
            rows.append({"regime": regime, "k": k, "AL_s": mean(als), "LAAL_s": mean(laals),
                         "StartOffset_s": mean(startoffs), "EndOffset_s": mean(endoffs),
                         "pauses_gt_250ms": sum(pauses), "entity_acc": ok / tot if tot else float("nan"),
                         "n_entities": tot, "abandoned_leaked": f"{leak}/{ab}",
                         "by_type": {t: v[1] / v[0] for t, v in ent_by_type.items()}})
            print(f"{regime:8s} k={k:2d} LAAL={rows[-1]['LAAL_s']:.2f}s acc={rows[-1]['entity_acc']:.2f}", flush=True)
    save_cache()
    return rows


def table(rows):
    lines = ["| regime | k | AL s | LAAL s | StartOffset s | EndOffset s | pauses >250 ms | entity acc | abandoned value spoken |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['regime']} | {r['k']} | {r['AL_s']:.2f} | {r['LAAL_s']:.2f} | {r['StartOffset_s']:.2f} | "
                     f"{r['EndOffset_s']:.2f} | {r['pauses_gt_250ms']} | {r['entity_acc']:.0%} ({r['n_entities']}) | {r['abandoned_leaked']} |")
    return "\n".join(lines)


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for regime, mk in (("oracle", "o"), ("prefix", "s"), ("adaptive", "^")):
        rs = [r for r in rows if r["regime"] == regime]
        a.plot([r["LAAL_s"] for r in rs], [r["entity_acc"] for r in rs], marker=mk, label=regime)
        for r in rs:
            a.annotate(f"k={r['k']}", (r["LAAL_s"], r["entity_acc"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        b.plot([r["k"] for r in rs], [r["EndOffset_s"] for r in rs], marker=mk, label=regime)
    a.set(xlabel="Length-adaptive Average Lagging, LAAL (s)", ylabel="entity accuracy", title="Entity accuracy vs lag")
    b.set(xlabel="k", ylabel="EndOffset (s)", title="How long after the speaker the English ends")
    a.legend()
    a.grid(alpha=.3)
    b.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--clips", nargs="*")
    args = ap.parse_args()
    test()
    if args.test:
        sys.exit()
    clips = args.clips or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ROOT, "align/out/clip*.json")))
    clip_data = {c: load_clip(c) for c in clips}
    rows = simulate(clips)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, "results.md"), "w").write(table(rows) + "\n")
    plot(rows, os.path.join(OUT, "acc_vs_lag.png"))
    print(table(rows))
