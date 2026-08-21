"""
Run the whole pipeline in order. Every stage is resumable, so re-running is safe
and only does outstanding work.

  python scripts/run_all.py                 # everything
  python scripts/run_all.py --from 05       # resume from a stage
  python scripts/run_all.py --skip 08 09b   # skip the LLM stages
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# Interpreter used for each stage. Defaults to the venv inside HCP_WORK if one
# is there, otherwise to whatever interpreter is running this script.
_venv = Path(os.environ.get("HCP_WORK") or (Path.home() / "hcp_work")) \
    / "venv" / "Scripts" / "python.exe"
PY = os.environ.get("HCP_PYTHON") or (str(_venv) if _venv.exists() else sys.executable)

STAGES = [
    ("01", "01_prepare_audio.py", [], "audio -> 16 kHz mono + QC"),
    ("02", "02_transcribe.py", [], "Whisper large-v3 ASR (GPU)"),
    ("03", "03_diarize.py", [], "cam++ diarization"),
    ("04", "04_build_utterances.py", [], "L1 table + de-identification"),
    ("05", "05_acoustic_features.py", [], "eGeMAPS + MSP-dim emotion"),
    ("06", "06_topic_segments.py", [], "L2 topic segmentation"),
    ("07", "07_text_sentiment.py", [], "text sentiment"),
    ("08", "08_llm_coding.py", [], "LLM thematic coding (slow)"),
    ("09", "09_fusion.py", [], "scheme A fusion + discord + 4.4 sample"),
    ("09b", "09b_fusion_llm.py", ["--sample", "250"], "scheme B fusion (slow)"),
    ("10", "10_aggregate_models.py", [], "L2/L3 aggregation + mixed models"),
    ("11", "11_figures.py", [], "figures"),
    ("12", "12_qc_checks.py", [], "QC worksheets (2.3 listening, 4.3 reading)"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    stages = STAGES
    if args.start:
        ids = [s[0] for s in STAGES]
        if args.start not in ids:
            sys.exit("unknown stage %s; valid: %s" % (args.start, ", ".join(ids)))
        stages = STAGES[ids.index(args.start):]
    if args.only:
        stages = [s for s in stages if s[0] in args.only]
    stages = [s for s in stages if s[0] not in args.skip]

    failed = []
    for sid, script, extra, desc in stages:
        print("\n" + "=" * 72)
        print("STAGE %-4s %s" % (sid, desc))
        print("=" * 72, flush=True)
        t0 = time.time()
        r = subprocess.run([PY, str(PROJECT / "scripts" / script), *extra],
                           cwd=str(PROJECT))
        dt = time.time() - t0
        if r.returncode != 0:
            print("[FAIL] stage %s exited %d after %.0fs" % (sid, r.returncode, dt))
            failed.append(sid)
            if sid in ("01", "02", "03", "04"):
                sys.exit("stage %s is a hard dependency; stopping" % sid)
        else:
            print("[ok] stage %s in %.0fs" % (sid, dt), flush=True)

    print("\n" + "=" * 72)
    print("FAILED STAGES: %s" % (", ".join(failed) if failed else "none"))
    print("=" * 72)


if __name__ == "__main__":
    main()
