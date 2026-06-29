#!/usr/bin/env python3
"""
transcribe.py — Video → diarized, timestamped, verbatim transcript via Gemini.

Romanian-primary with English words kept exactly as spoken. Output drops straight
into the meetinginsights skill's "Session transcript" intake.

Usage:
    export GEMINI_API_KEY=...        # or pass --api-key
    python3 transcribe.py path/to/video.mp4

The script extracts audio with ffmpeg, uploads it via the Gemini Files API, and
transcribes it. Long recordings are split into ~20 min chunks and stitched back
together with continuous absolute timestamps. On first run it creates a local
.venv and installs google-genai automatically.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import shutil
import argparse
import subprocess
import tempfile
from datetime import date
from pathlib import Path


# --------------------------------------------------------------------------- #
# Dependency bootstrap: ensure google-genai is importable, else create a local
# .venv, install it there, and re-exec this script under that interpreter.
# --------------------------------------------------------------------------- #
def ensure_deps():
    try:
        import google.genai  # noqa: F401
        return
    except ImportError:
        pass

    here = Path(__file__).resolve().parent
    venv_dir = here / ".venv"
    venv_py = venv_dir / "bin" / "python"
    in_venv = os.path.realpath(sys.executable) == os.path.realpath(str(venv_py))

    if not in_venv:
        if not venv_py.exists():
            print("· Creating local .venv (first run)…", file=sys.stderr)
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        os.execv(str(venv_py), [str(venv_py), *sys.argv])  # re-enter under venv
    else:
        print("· Installing google-genai into .venv…", file=sys.stderr)
        subprocess.check_call(
            [str(venv_py), "-m", "pip", "install", "-q", "--upgrade", "pip", "google-genai"]
        )
        os.execv(str(venv_py), [str(venv_py), *sys.argv])  # re-exec to import it


ensure_deps()

from google import genai           # noqa: E402
from google.genai import types     # noqa: E402

import diarize_reconcile  # noqa: E402  (same dir; on sys.path when run as a script)


def run_diarizer(audio: Path, num_speakers: int | None, device: str,
                 hf_token: str | None = None):
    """Shell out to diarize.py; return segment list, or None on any failure."""
    here = Path(__file__).resolve().parent
    cmd = [sys.executable, str(here / "diarize.py"), str(audio), "--device", device]
    if num_speakers:
        cmd += ["--num-speakers", str(num_speakers)]
    if hf_token:
        cmd += ["--hf-token", hf_token]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"⚠ acoustic diarization failed; keeping Gemini's speaker labels.\n"
              f"{r.stderr.strip()}", file=sys.stderr)
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        print("⚠ diarization output was not valid JSON; keeping Gemini's labels.",
              file=sys.stderr)
        return None


CHUNK_DEFAULT_MIN = 20
DEFAULT_MODEL = "gemini-2.5-pro"

PROMPT = """\
You are a professional transcriptionist. Transcribe this audio recording EXACTLY as \
spoken. This is a user-testing / meeting session, mostly in ROMANIAN with some ENGLISH \
technical words mixed in.

Rules:
- Transcribe VERBATIM. Do NOT translate. Do NOT summarize, paraphrase, or correct grammar.
- Keep Romanian as Romanian and English words as English — write each word in the \
language it was actually spoken in. Preserve Romanian diacritics (ă, â, î, ș, ț).
- Diarize: identify distinct speakers and label them "Speaker 1", "Speaker 2", etc. Keep \
the same label for the same voice throughout.
- Start every speaker turn on its own line, prefixed with a timestamp in [HH:MM:SS] \
format marking when that turn begins, like:
[00:00:04] Speaker 1: ...
[00:00:11] Speaker 2: ...
- Use timestamps RELATIVE to the start of THIS audio clip (the clip starts at 00:00:00). \
Use exactly [HH:MM:SS] with whole seconds only — NO milliseconds, frames, or extra fields.
- If a stretch is unintelligible, write [neinteligibil] rather than guessing.
- Output ONLY the transcript lines. No preamble, no commentary, no markdown fences.
"""


def die(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def probe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    r = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    if r.returncode != 0:
        die(f"ffprobe failed: {r.stderr.strip()}")
    try:
        return float(r.stdout.strip())
    except ValueError:
        die(f"could not parse duration from ffprobe: {r.stdout!r}")


def extract_audio(video: Path, dst: Path, start: float | None = None, dur: float | None = None):
    """Extract (a slice of) the audio track to mono 16 kHz FLAC."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(dst)]
    r = run(cmd)
    if r.returncode != 0:
        die(f"ffmpeg audio extraction failed: {r.stderr.strip()}")
    if not dst.exists() or dst.stat().st_size == 0:
        die("ffmpeg produced no audio — does the video have an audio track?")


def upload_and_wait(client: genai.Client, path: Path):
    """Upload a file to the Gemini Files API and poll until ACTIVE."""
    f = client.files.upload(file=str(path))
    waited = 0
    while f.state.name == "PROCESSING":
        time.sleep(2)
        waited += 2
        f = client.files.get(name=f.name)
        if waited > 600:
            die("file stuck in PROCESSING for >10 min")
    if f.state.name == "FAILED":
        die(f"Gemini failed to process the uploaded file: {f.name}")
    return f


def speaker_hint(roster: str | None) -> str:
    """Build a prompt suffix that steers diarization with a known speaker roster."""
    if not roster or not roster.strip():
        return ""
    return (
        "\nKNOWN SPEAKER ROSTER (use to keep diarization consistent and correct):\n"
        f"{roster.strip()}\n"
        "- There are exactly this many distinct people — do NOT invent extra speakers.\n"
        "- Keep one stable \"Speaker N\" label per real person for the whole clip.\n"
        "- When two voices sound similar, decide who spoke by WHO WOULD SAY IT — their "
        "role/expertise and what they refer to (their own work, domain, plans) — not by "
        "acoustic guess alone. A person never refers to themselves in the third person.\n"
    )


VOICE_ATTRS_HINT = (
    "\nVOICE-GENDER TAGGING (do this in addition to the rules above):\n"
    "- Immediately after the speaker label, append the PERCEIVED voice gender of that "
    "turn in parentheses: (m) male, (f) female, (?) genuinely unclear or too short. Like:\n"
    "[00:00:04] Speaker 1 (m): ...\n"
    "[00:00:11] Speaker 2 (f): ...\n"
    "- Judge the gender PER TURN from the voice itself, INDEPENDENTLY of the Speaker label. "
    "If a turn sounds female but you labeled it the same as a male speaker, still tag it "
    "(f) — the mismatch is a useful signal of a likely diarization slip. Do not force the "
    "tag to agree with the label.\n"
)


def voice_attrs_hint(on: bool) -> str:
    return VOICE_ATTRS_HINT if on else ""


def transcribe_clip(client: genai.Client, model: str, path: Path, hint: str = "") -> str:
    f = upload_and_wait(client, path)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[f, PROMPT + hint],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=32768,
            ),
        )
    finally:
        try:
            client.files.delete(name=f.name)
        except Exception:
            pass

    text = (resp.text or "").strip()
    if not text:
        reason = ""
        try:
            reason = f" (finish_reason={resp.candidates[0].finish_reason})"
        except Exception:
            pass
        die(f"Gemini returned an empty transcript{reason} — content may have been blocked")
    return text


# Leading bracketed timestamp token: 2–4 numeric fields separated by : . or ,
# Captures the whole token so no fragment is left behind in the output line.
TS_RE = re.compile(r"^\s*[\[(]\s*(\d{1,3}(?:[:.,]\d{1,3}){1,3})\s*[\])]\s*")


def fmt_ts(total: float) -> str:
    total = int(round(max(0.0, total)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def parse_rel_seconds(token: str, chunk_len: float) -> float:
    """Parse a chunk-relative timestamp token into seconds.

    Gemini is inconsistent about format across calls: MM:SS, HH:MM:SS, or even
    MM:SS:mmm (milliseconds). We know the chunk's length, so we disambiguate: a
    three-field reading interpreted as HH:MM:SS that overshoots the chunk (or whose
    'seconds' field is > 59) must really be MM:SS:mmm.
    """
    parts = [int(p) for p in re.split(r"[:.,]", token)]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    # 3+ fields: decide HH:MM:SS vs MM:SS:mmm
    a, b, c = parts[0], parts[1], parts[2]
    hms = a * 3600 + b * 60 + c
    if c > 59 or hms > chunk_len + 120:
        return a * 60 + b  # MM:SS(:mmm) — drop the sub-second field
    return hms             # HH:MM:SS


def reoffset(text: str, offset: float, chunk_len: float) -> str:
    """Rewrite each line's leading [timestamp] to an absolute one (+offset)."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = TS_RE.match(line)
        if m:
            abs_t = offset + parse_rel_seconds(m.group(1), chunk_len)
            out.append(f"{fmt_ts(abs_t)} {line[m.end():]}")
        else:
            # continuation / non-timestamped line: keep as-is
            out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(
        description="Transcribe a video (RO+EN) to a diarized, timestamped transcript via Gemini."
    )
    ap.add_argument("video", type=Path, help="path to the video (or audio) file")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default {DEFAULT_MODEL})")
    ap.add_argument("--chunk-minutes", type=float, default=CHUNK_DEFAULT_MIN,
                    help=f"max minutes per chunk (default {CHUNK_DEFAULT_MIN})")
    ap.add_argument("--api-key", default=None, help="Gemini API key (else $GEMINI_API_KEY)")
    ap.add_argument("--keep-audio", action="store_true", help="keep the extracted audio files")
    ap.add_argument("--speakers", default=None,
                    help="known speaker roster/roles to steer diarization, e.g. "
                         "'Speaker 1 = Bogdan (designer, presents prototype); "
                         "Speaker 2 = Ana (PM)'")
    ap.add_argument("--voice-attrs", action="store_true",
                    help="tag each line with perceived voice gender (m)/(f)/(?) — a strong "
                         "prior for separating speakers in cross-gender casts (judged per "
                         "turn, independent of the speaker label, so it flags mislabels)")
    ap.add_argument("--diarize", action="store_true",
                    help="overlay acoustic (pyannote) diarization on the full audio and "
                         "relabel speakers by voice, flagging contested/merged turns — "
                         "the robust fix for same-gender speaker confusion")
    ap.add_argument("--diarize-device", default="cpu", choices=["cpu", "mps"],
                    help="device for pyannote (default cpu; mps = Apple-Silicon GPU)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output path (default <video>.transcript.txt)")
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        die("ffmpeg/ffprobe not found on PATH")
    if not args.video.exists():
        die(f"file not found: {args.video}")
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        die("no API key — set GEMINI_API_KEY or pass --api-key")

    client = genai.Client(api_key=api_key)

    out_path = args.out or args.video.with_suffix(args.video.suffix + ".transcript.txt")
    chunk_secs = max(60.0, args.chunk_minutes * 60.0)

    duration = probe_duration(args.video)
    n_chunks = max(1, -(-int(duration) // int(chunk_secs)))  # ceil
    print(f"· {args.video.name}: {duration/60:.1f} min → {n_chunks} chunk(s)", file=sys.stderr)

    hint = speaker_hint(args.speakers) + voice_attrs_hint(args.voice_attrs)
    if args.speakers:
        print("· steering diarization with the provided speaker roster", file=sys.stderr)
    if args.voice_attrs:
        print("· tagging perceived voice gender per line (m/f/?)", file=sys.stderr)

    workdir = Path(tempfile.mkdtemp(prefix="transcribe_"))
    pieces: list[str] = []
    diarized_ok = False
    try:
        for i in range(n_chunks):
            start = i * chunk_secs
            dur = min(chunk_secs, duration - start)
            audio = workdir / f"chunk_{i:03d}.flac"
            label = f"chunk {i+1}/{n_chunks} ({start/60:.0f}–{(start+dur)/60:.0f} min)"
            print(f"· extracting {label}…", file=sys.stderr)
            extract_audio(args.video, audio, start=start if n_chunks > 1 else None,
                          dur=dur if n_chunks > 1 else None)
            print(f"· transcribing {label}…", file=sys.stderr)
            raw = transcribe_clip(client, args.model, audio, hint)
            pieces.append(reoffset(raw, start, dur))
            if args.keep_audio:
                shutil.copy(audio, args.video.parent / audio.name)

        body = "\n\n".join(pieces) if n_chunks > 1 else pieces[0]

        if args.diarize:
            diar_audio = workdir / "full_for_diarization.flac"
            print("· extracting full audio for acoustic diarization…", file=sys.stderr)
            extract_audio(args.video, diar_audio)  # whole file, not a chunk slice
            n_spk = diarize_reconcile.roster_speaker_count(args.speakers)
            if n_spk:
                print(f"· diarizing with num_speakers={n_spk} (from roster)…",
                      file=sys.stderr)
            segments = run_diarizer(diar_audio, n_spk, args.diarize_device)
            if segments:
                body = diarize_reconcile.reconcile(body, segments, duration)
                diarized_ok = True
                print("· acoustic diarization applied — speaker labels relabeled by "
                      "voice; contested turns flagged.", file=sys.stderr)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    header = (
        f"# Transcript of {args.video.name} — generated {date.today().isoformat()} "
        f"via {args.model}\n"
        f"# Verbatim, diarized, timestamped. Romanian as spoken; English words kept "
        f"as English.\n"
    )
    if args.speakers:
        header += f"# Speaker roster (diarization steered): {args.speakers.strip()}\n"
    if args.voice_attrs:
        header += ("# Per-line voice gender tag after the label: (m) male, (f) female, "
                   "(?) unclear — perceived per turn, independent of the Speaker label; "
                   "a tag that disagrees with its label flags a likely diarization slip.\n")
    if diarized_ok:
        header += (
            "# Acoustic diarization: pyannote/speaker-diarization-3.1. Speaker labels are "
            "acoustic clusters (stable across the whole recording), ordered by first "
            "appearance.\n"
            "# Flag legend: ‹reattr gemini=Sx conf=c› = acoustic reassigned this line off "
            "Gemini's label x (confidence c); ‹mixed Sx/Sy› = the turn's audio spans >1 "
            "speaker (likely a merged turn — review by listening).\n"
        )
    elif n_chunks > 1:
        header += (
            "# NOTE: speaker labels may reset at chunk boundaries "
            f"(~{int(chunk_secs/60)} min); reconcile downstream.\n"
        )

    out_path.write_text(header + "\n" + body + "\n", encoding="utf-8")
    print(f"✓ wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
