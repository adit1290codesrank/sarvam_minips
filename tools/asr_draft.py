"""First-draft Hindi transcript with Whisper large-v3-turbo, to be corrected by ear.

    python tools/asr_draft.py clips/16k/clip01.wav > transcripts/draft/clip01.txt

Whisper normalises: it drops fillers, may write numbers as digits and English words in
Latin script. Level-1 rules need all of those fixed by hand (see README).
"""
import sys

import soundfile as sf
import torch
from transformers import pipeline

wav, sr = sf.read(sys.argv[1], dtype="float32")
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
               dtype=torch.float16, device=0 if torch.cuda.is_available() else -1)
out = asr({"raw": wav, "sampling_rate": sr}, return_timestamps=True,
          generate_kwargs={"language": "hindi", "task": "transcribe"})
for c in out["chunks"]:
    print(c["text"].strip())
