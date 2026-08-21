"""
Step 1.2 (part 1) — ASR with word-level timestamps.

Two departures from the framework, both forced by what the data turned out to be:

1. LANGUAGE is pinned rather than detected per file. Language ID over all 30
   recordings returns English with p>=0.92 everywhere, so `language="en"` is
   set explicitly -- letting Whisper re-detect per file risks a single noisy
   opening being mislabelled and decoded in the wrong language.

2. TOOLING. WhisperX's diarization half needs pyannote checkpoints that are
   gated behind an HF licence + token, unavailable here. We therefore use
   faster-whisper for ASR (this script) and build diarization separately from
   ungated cam++ speaker embeddings (script 03).

Usage:
  python scripts/02_transcribe.py                 # all sessions
  python scripts/02_transcribe.py S01 S02         # named sessions
  python scripts/02_transcribe.py --model medium  # override model size
"""
import argparse
import json
import os
import time
from pathlib import Path

# Scratch space for models, converted audio and intermediate JSON. Kept
# outside the project folder so cloud-sync clients do not churn on it.
# Override with the HCP_WORK environment variable.
WORK = Path(os.environ.get("HCP_WORK") or (Path.home() / "hcp_work"))
os.environ.setdefault("HF_HOME", str(WORK / "models" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ctranslate2 needs the cuBLAS/cuDNN DLLs that torch already ships
_TORCH_LIB = WORK / "venv" / "Lib" / "site-packages" / "torch" / "lib"
if _TORCH_LIB.is_dir():
    os.add_dll_directory(str(_TORCH_LIB))

import pandas as pd  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = WORK / "asr_json"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(size: str, device: str) -> WhisperModel:
    # int8_float16 keeps large-v3 near 2.5 GB so it survives on a contended
    # 6 GB desktop card; accuracy loss vs float16 is negligible.
    ct = "int8_float16" if device == "cuda" else "int8"
    print(f"[init] faster-whisper {size} on {device}/{ct}", flush=True)
    return WhisperModel(size, device=device, compute_type=ct,
                        download_root=str(WORK / "models" / "whisper"))


def transcribe(model: WhisperModel, wav: Path) -> dict:
    t0 = time.time()
    segments, info = model.transcribe(
        str(wav),
        language="en",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,  # avoids runaway repetition on long audio
    )
    segs = []
    for s in segments:
        segs.append({
            "id": s.id,
            "start_s": round(s.start, 3),
            "end_s": round(s.end, 3),
            "text": s.text.strip(),
            "avg_logprob": round(s.avg_logprob, 4),
            "no_speech_prob": round(s.no_speech_prob, 4),
            "words": [
                {"w": w.word, "s": round(w.start, 3), "e": round(w.end, 3),
                 "p": round(w.probability, 3)}
                for w in (s.words or [])
            ],
        })
    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_s": round(info.duration, 2),
        "elapsed_s": round(time.time() - t0, 1),
        "segments": segs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(PROJECT / "data" / "sessions.csv")
    if args.sessions:
        df = df[df["session_id"].isin(args.sessions)]
    todo = [r for _, r in df.iterrows()
            if args.overwrite or not (OUT_DIR / f"{r['session_id']}.json").exists()]
    if not todo:
        print("all requested sessions already transcribed")
        return

    model = load_model(args.model, args.device)
    for i, row in enumerate(todo, start=1):
        sid, wav = row["session_id"], Path(row["wav_16k_path"])
        print(f"[{i}/{len(todo)}] {sid} ({row['duration_min']:.1f} min) ...",
              end=" ", flush=True)
        try:
            out = transcribe(model, wav)
        except Exception as e:
            print(f"FAILED {type(e).__name__}: {e}", flush=True)
            (OUT_DIR / f"{sid}.error.txt").write_text(f"{type(e).__name__}: {e}",
                                                      encoding="utf-8")
            continue
        out["session_id"] = sid
        out["asr_model"] = args.model
        (OUT_DIR / f"{sid}.json").write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8")
        nw = sum(len(s["words"]) for s in out["segments"])
        print(f"{len(out['segments'])} segs / {nw} words / {out['elapsed_s']:.0f}s "
              f"(RTF {out['elapsed_s'] / max(out['duration_s'], 1):.3f})", flush=True)


if __name__ == "__main__":
    main()
