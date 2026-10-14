"""Phone recording -> 24 kHz (PS format) and 16 kHz (aligner) mono 16-bit WAV.

    python tools/convert.py clips/raw/clip01.m4a [...]   # -> clips/24k/clip01.wav, clips/16k/clip01.wav
"""
import os
import shutil
import subprocess
import sys

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ffmpeg():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg                                # bundled static binary
    return imageio_ffmpeg.get_ffmpeg_exe()


def convert(src, out_dir=os.path.join(ROOT, "clips")):
    name = os.path.splitext(os.path.basename(src))[0]
    outs = {}
    for sr in (24000, 16000):
        d = os.path.join(out_dir, f"{sr // 1000}k")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, name + ".wav")
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", str(sr),
                        "-c:a", "pcm_s16le", dst], check=True)
        outs[sr] = dst
    return outs


if __name__ == "__main__":
    for f in sys.argv[1:]:
        o = convert(f)
        print(f"{o[16000]}: {sf.info(o[16000]).duration:.1f} s")
