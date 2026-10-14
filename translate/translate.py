"""Translations for every clip -> translations.jsonl

    python translate/translate.py            # all transcripts/L1/clip*.txt

Three systems:
  llm   literal + fluent, written by an LLM (Claude Opus 5.5) from translate/PROMPT.md and
        stored in translate/llm/clipNN.json -- the cache *is* the output, so reruns are identical
  nllb  open MT baseline, facebook/nllb-200-distilled-1.3B, greedy, run here; every output is
        cached in translate/cache/nllb.json keyed by the Hindi text
  it2   ai4bharat/indictrans2-indic-en-1B, beam 5, cached in translate/cache/it2.json. Inputs get
        the "hin_Deva eng_Latn" tags by hand: IndicTransToolkit does not import on transformers 5
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "facebook/nllb-200-distilled-1.3B"
IT2 = "ai4bharat/indictrans2-indic-en-1B"
_m = {}


def sentences(txt_path):
    return [l.strip() for l in open(txt_path, encoding="utf-8") if l.strip()]


def load(system):
    if system not in _m:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        name, kw = (MODEL, {"src_lang": "hin_Deva", "tgt_lang": "eng_Latn"}) if system == "nllb" else (IT2, {"trust_remote_code": True})
        tok = AutoTokenizer.from_pretrained(name, **kw)
        model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=system == "it2",
                                                      dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
        model.generation_config.max_length = None
        _m[system] = (tok, model, dev)
    return _m[system]


def mt(texts, system="nllb"):
    """Hindi -> English, cached per system. NLLB is also used for Part E prefixes."""
    import torch
    path = os.path.join(ROOT, f"translate/cache/{system}.json")
    cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    todo = sorted({t for t in texts if t not in cache})
    if todo:
        tok, model, dev = load(system)
        for b in range(0, len(todo), 16):
            batch = todo[b:b + 16]
            if system == "nllb":
                x = tok(batch, return_tensors="pt", padding=True).to(dev)
                kw = {"forced_bos_token_id": tok.convert_tokens_to_ids("eng_Latn"), "num_beams": 1}
            else:
                x = tok([f"hin_Deva eng_Latn {t}" for t in batch], return_tensors="pt", padding=True).to(dev)
                kw = {"num_beams": 5, "use_cache": False}   # its remote code breaks on the new KV-cache API
            with torch.inference_mode():
                y = model.generate(**x, **kw, do_sample=False, max_new_tokens=200)
            out = tok.batch_decode(y, skip_special_tokens=True)
            cache.update(zip(batch, (re.sub(r"\s+'(\w)", r"'\1", e).strip() for e in out)))   # IT2: "hasn 't" -> "hasn't"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(cache, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=0, sort_keys=True)
    return [cache[t] for t in texts]


def rows_for(clip, txt_path):
    hi = sentences(txt_path)
    rows = [{"clip": clip, "sent": i, "hi": h, "system": sys_, "version": "mt", "en": e}
            for sys_ in ("nllb", "it2") for i, (h, e) in enumerate(zip(hi, mt(hi, sys_)))]
    llm_path = os.path.join(ROOT, "translate/llm", clip + ".json")
    if os.path.exists(llm_path):
        llm = json.load(open(llm_path, encoding="utf-8"))
        assert [r["hi"] for r in llm] == hi, f"{llm_path} is stale: transcript changed, re-translate"
        rows += [{"clip": clip, "sent": i, "hi": r["hi"], "system": "llm", "version": v, "en": r[v]}
                 for i, r in enumerate(llm) for v in ("literal", "fluent")]
    return rows


if __name__ == "__main__":
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "transcripts/L1/clip*.txt"))):
        clip = re.sub(r"\.txt$", "", os.path.basename(p))
        rows += rows_for(clip, p)
    with open(os.path.join(ROOT, "translations.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"translations.jsonl: {len(rows)} rows")
