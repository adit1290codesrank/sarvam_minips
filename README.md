# Sarvam.ai Mini PS: Parts D and E (ML track)

Six code-mixed Hindi clips → Level-1 transcripts → word timings → literal and fluent translations → an entity checker → a wait-k simulation of where a streaming Hindi→English model should wait.

All six clips were recorded by me, of my own voice, for this task.

Every number in the report is in **[RESULTS.md](RESULTS.md)**, with the command that reproduces it.

## Run it

```bash
make env            # Python 3.12 venv, torch 2.11 (CUDA 12.8 wheels; use .../whl/cpu without a GPU), MMS_FA smoke test
make results        # tests → convert → align → translate → checker sheet → wait-k → RESULTS.md
python run_clip.py new.m4a --transcript new.txt    # everything on a new clip → out/new/
python run_clip.py new.m4a --asr-draft             # no transcript yet: print a Whisper draft to correct
```

`make results` reruns from cached model outputs (`translate/cache/`, `partE_ml/cache.json`), so it gives identical numbers. Delete the caches to recompute them on a GPU.

## What is where

| Step | Code | Data / outputs |
|---|---|---|
| D1 Clips | `tools/convert.py` (→ 24 kHz and 16 kHz mono WAV) | `clips/raw/clip01–06` |
| D2 Level-1 transcripts | `tools/asr_draft.py` (Whisper large-v3-turbo draft, then corrected by ear) | `transcripts/L1/clipNN.txt` |
| D3 Word timings | `align/align_mms.py` (torchaudio MMS_FA + uroman) | `align/out/clipNN.json`, `clipNN.labels.txt` (Audacity) |
| D4 30 hand-checked boundaries | `align/eval_boundaries.py` | `align/hand_labels/` (marked blind in Audacity; `TODO.md` lists them) |
| D5 Translation | `translate/translate.py`: LLM literal + fluent (`translate/llm/`, prompt in `PROMPT.md`), IndicTrans2 1B, NLLB-200 1.3B | `translations.jsonl` |
| D6 Checker | `checker/checker.py` (amounts, digits, times, dates, names, negation, self-repairs), built on `prototypes/hindi_number_parser_prototype.py` | `checker/gold/`, `checker/verdicts.csv` (164 blind human verdicts) |
| D6 Checker accuracy | `checker/eval_checker.py eval` | RESULTS.md §D5–D6 |
| E wait-k (ML track) | `partE_ml/wait_k.py`: oracle / prefix / adaptive regimes, AL, LAAL, EndOffset, pauses | `partE_ml/out/` (table, `acc_vs_lag.png`) |
| Tests | `tests/test_checker.py`, `prototypes/test_hindi_number_parser.py`, `partE_ml/wait_k.py --test` (wait-3 AL = 2.625) | `make test` |
| Interview demo | `run_clip.py` | `out/<name>/` (timings, translations, checks, `timeline.png`) |

## Headline results (details in RESULTS.md)

- **Alignment:** 44 ms mean, 29 ms median error over 30 hard boundaries; 90% within 100 ms; the aligner is 33 ms late on average.
- **Checker:** 98% accuracy against blind human verdicts (161/164). The 3 misses are listed with reasons.
- **Entities preserved (human verdicts):** LLM 100%, IndicTrans2 88%, NLLB 83%. Both open MT systems kept every self-corrected ("abandoned") value.
- **wait-k:**
  - Prefix wait-1 keeps 65% of entities, against 88% for the full sentence.
  - Holding through entity, repair and negation spans reaches 85% at 2.2 s LAAL; fixed wait-k needs k = 10 (3.4 s) to beat that.
  - Times and negations are hurt most by early commitment; names, dates and digit-by-digit numbers are not.

## Notes

- Part E is an offline simulation on gold transcripts with an external MT model. That would be a banned cascade in the real system; here it only measures where prefix translation breaks.
- The NeMo number tables in `prototypes/data/` are Apache-2.0 (see `NEMO_LICENSE`).
- Known limitations: see the end of RESULTS.md.
