"""
Step 1.1 + 1.4 — Audio standardization and quality check.

Converts every recording in Audios/ to 16 kHz mono 16-bit WAV, assigns a stable
session_id, and computes the QC metrics the framework asks for in section 1.4
(loudness, clipping ratio, SNR estimate) plus the L/R correlation of the stereo
files, which tells us whether the stereo->mono downmix loses anything.

Outputs:
  data/sessions.csv    one row per recording (session manifest)
  data/audio_qc.csv    one row per recording (quality metrics)
  <WORK>/audio_16k/{session_id}.wav
"""
import json
import re
import subprocess
import sys
import wave
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
AUDIO_SRC = PROJECT / "Audios"
# Scratch space for models, converted audio and intermediate JSON. Kept
# outside the project folder so cloud-sync clients do not churn on it.
# Override with the HCP_WORK environment variable.
WORK = Path(os.environ.get("HCP_WORK") or (Path.home() / "hcp_work"))
AUDIO_16K = WORK / "audio_16k"
FFMPEG = WORK / "bin" / "ffmpeg.exe"
FFPROBE = WORK / "bin" / "ffprobe.exe"

AUDIO_16K.mkdir(parents=True, exist_ok=True)
(PROJECT / "data").mkdir(exist_ok=True)

# Filenames look like 2025-06-12_1100_HCPFocusGroup_Recording.m4a
NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{4})_")


def probe(path: Path) -> dict:
    out = subprocess.run(
        [str(FFPROBE), "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def frame_energy(x: np.ndarray, sr: int, win_ms: int = 20) -> np.ndarray:
    n = int(sr * win_ms / 1000)
    if n <= 0 or len(x) < n:
        return np.array([np.mean(x ** 2) + 1e-12])
    trimmed = x[: len(x) // n * n].reshape(-1, n)
    return np.mean(trimmed ** 2, axis=1) + 1e-12


def db(v: float) -> float:
    return float(20 * np.log10(max(v, 1e-12)))


def lr_correlation(src: Path, channels: int) -> float | None:
    """Pearson r between L and R. ~1.0 means the stereo file is effectively mono."""
    if channels != 2:
        return None
    tmp = AUDIO_16K / "_lrprobe.wav"
    subprocess.run(
        [str(FFMPEG), "-y", "-v", "error", "-i", str(src),
         "-t", "300", "-ar", "16000", "-ac", "2", "-c:a", "pcm_s16le", str(tmp)],
        check=True,
    )
    with wave.open(str(tmp), "rb") as w:
        raw = w.readframes(w.getnframes())
    st = np.frombuffer(raw, dtype=np.int16).astype(np.float32).reshape(-1, 2)
    tmp.unlink(missing_ok=True)
    if st[:, 0].std() < 1e-6 or st[:, 1].std() < 1e-6:
        return None
    return float(np.corrcoef(st[:, 0], st[:, 1])[0, 1])


def main() -> None:
    files = sorted(p for p in AUDIO_SRC.iterdir() if p.suffix.lower() in {".m4a", ".mp3", ".wav"})
    if not files:
        sys.exit(f"No audio found in {AUDIO_SRC}")

    sessions, qc = [], []
    for i, src in enumerate(files, start=1):
        session_id = f"S{i:02d}"
        info = probe(src)
        stream = info["streams"][0]
        fmt = info["format"]
        m = NAME_RE.match(src.name)
        channels = int(stream.get("channels", 1))

        dst = AUDIO_16K / f"{session_id}.wav"
        subprocess.run(
            [str(FFMPEG), "-y", "-v", "error", "-i", str(src),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
            check=True,
        )

        x, sr = read_wav(dst)
        e = frame_energy(x, sr)
        # Speech level vs noise floor, estimated from the energy distribution.
        p95, p10 = np.percentile(e, 95), np.percentile(e, 10)
        snr_est = float(10 * np.log10(p95 / p10))
        peak = float(np.max(np.abs(x)))
        clip_ratio = float(np.mean(np.abs(x) >= 0.999))
        # Fraction of 20 ms frames plausibly containing speech (>= noise floor + 12 dB)
        speech_frac = float(np.mean(e >= p10 * 10 ** 1.2))

        sessions.append({
            "session_id": session_id,
            "source_file": src.name,
            "interview_date": m.group(1) if m else "",
            "interview_time": f"{m.group(2)[:2]}:{m.group(2)[2:]}" if m else "",
            "src_codec": stream.get("codec_name"),
            "src_sample_rate": int(stream.get("sample_rate", 0)),
            "src_channels": channels,
            "src_bitrate_kbps": round(int(fmt.get("bit_rate", 0)) / 1000),
            # The framework (1.1) wants recording condition logged; codec+sr+ch is
            # our only observable proxy for the device, so we treat it as one.
            "recording_device": f"{stream.get('codec_name')}_{stream.get('sample_rate')}_{channels}ch",
            "duration_min": round(float(fmt["duration"]) / 60, 2),
            "wav_16k_path": str(dst),
        })

        qc.append({
            "session_id": session_id,
            "duration_min": round(float(fmt["duration"]) / 60, 2),
            "peak_dbfs": round(db(peak), 2),
            "rms_dbfs": round(db(float(np.sqrt(np.mean(x ** 2)))), 2),
            "clip_ratio": round(clip_ratio, 6),
            "snr_est_db": round(snr_est, 2),
            "speech_frac": round(speech_frac, 3),
            "lr_corr": lr_correlation(src, channels),
        })
        print(f"  {session_id}  {src.name[:44]:<44} "
              f"snr={snr_est:5.1f}dB  rms={db(float(np.sqrt(np.mean(x**2)))):6.1f}dBFS", flush=True)

    sdf = pd.DataFrame(sessions)
    qdf = pd.DataFrame(qc)
    sdf.to_csv(PROJECT / "data" / "sessions.csv", index=False, encoding="utf-8-sig")
    qdf.to_csv(PROJECT / "data" / "audio_qc.csv", index=False, encoding="utf-8-sig")

    print(f"\nWrote {len(sdf)} sessions -> data/sessions.csv, data/audio_qc.csv")
    print(f"Total audio: {sdf['duration_min'].sum() / 60:.2f} h")
    print("\nRecording conditions:")
    print(sdf.groupby("recording_device").agg(
        n=("session_id", "size"), total_min=("duration_min", "sum")).to_string())
    print("\nWorst 5 by estimated SNR (framework 1.4 asks us to eyeball these):")
    print(qdf.nsmallest(5, "snr_est_db")[
        ["session_id", "snr_est_db", "rms_dbfs", "clip_ratio", "speech_frac"]].to_string(index=False))


if __name__ == "__main__":
    main()
