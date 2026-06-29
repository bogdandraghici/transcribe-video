#!/usr/bin/env python3
"""diarize.py — acoustic speaker diarization via pyannote.audio.

Prints a JSON array of {"start","end","speaker"} segments to stdout. Heavy deps
(torch + pyannote.audio) live in a dedicated .venv-diarize that this script
bootstraps on first use, so the light transcription venv stays untouched.

Importable without torch: the pyannote/torch imports happen inside run_diarization,
and ensure_deps() only runs under __main__.
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path


def _resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    f = Path.home() / ".hf_token.env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
            return line
    return None


def _diar_kwargs(num: int | None, lo: int | None, hi: int | None) -> dict:
    if num:
        return {"num_speakers": num}
    kw: dict = {}
    if lo:
        kw["min_speakers"] = lo
    if hi:
        kw["max_speakers"] = hi
    return kw


def ensure_deps():
    """Ensure pyannote.audio is importable, else build .venv-diarize and re-exec."""
    try:
        import pyannote.audio  # noqa: F401
        return
    except ImportError:
        pass

    here = Path(__file__).resolve().parent
    venv_dir = here / ".venv-diarize"
    venv_py = venv_dir / "bin" / "python"
    in_venv = os.path.realpath(sys.executable) == os.path.realpath(str(venv_py))

    if not in_venv:
        if not venv_py.exists():
            print("· Creating local .venv-diarize (first run)…", file=sys.stderr)
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        os.execv(str(venv_py), [str(venv_py), *sys.argv])
    else:
        print("· Installing torch + pyannote.audio into .venv-diarize "
              "(large, first run only)…", file=sys.stderr)
        subprocess.check_call(
            [str(venv_py), "-m", "pip", "install", "-q", "--upgrade", "pip",
             "pyannote.audio"]
        )
        os.execv(str(venv_py), [str(venv_py), *sys.argv])


def die(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def extract_audio(src: Path, dst: Path):
    """Extract mono 16 kHz WAV from any audio/video input."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        die(f"ffmpeg audio extraction failed: {r.stderr.strip()}")


def run_diarization(wav_path: str, token: str | None, device: str, kwargs: dict) -> list[dict]:
    from pyannote.audio import Pipeline  # heavy — imported lazily
    import torch

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token
    )
    if pipeline is None:
        die("could not load pyannote pipeline — is the HF token valid and the "
            "model's terms accepted? (pyannote/speaker-diarization-3.1 + "
            "pyannote/segmentation-3.0)")
    pipeline.to(torch.device(device))
    annotation = pipeline(wav_path, **kwargs)
    return [
        {"start": float(seg.start), "end": float(seg.end), "speaker": label}
        for seg, _track, label in annotation.itertracks(yield_label=True)
    ]


def main():
    ap = argparse.ArgumentParser(description="Acoustic diarization via pyannote.audio.")
    ap.add_argument("audio", type=Path, help="audio or video file")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        die("ffmpeg not found on PATH")
    if not args.audio.exists():
        die(f"file not found: {args.audio}")
    token = _resolve_token(args.hf_token)
    if not token:
        die("no HuggingFace token — set $HF_TOKEN, pass --hf-token, or create "
            "~/.hf_token.env")

    workdir = Path(tempfile.mkdtemp(prefix="diarize_"))
    try:
        wav = workdir / "audio.wav"
        print("· extracting audio…", file=sys.stderr)
        extract_audio(args.audio, wav)
        print("· running pyannote diarization (this can take a while on CPU)…",
              file=sys.stderr)
        kwargs = _diar_kwargs(args.num_speakers, args.min_speakers, args.max_speakers)
        segments = run_diarization(str(wav), token, args.device, kwargs)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    n_spk = len({s["speaker"] for s in segments})
    print(f"· {len(segments)} segments, {n_spk} speaker(s)", file=sys.stderr)
    json.dump(segments, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    ensure_deps()
    main()
