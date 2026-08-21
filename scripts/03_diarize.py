"""
Step 1.2 (part 2) — Speaker diarization from ungated cam++ embeddings.

Replaces WhisperX's pyannote stage, which needs a gated HF checkpoint. Method is
the standard embedding + spectral clustering recipe (Wang et al. 2018):

  1. speech regions come from Whisper word timestamps (already VAD-filtered)
  2. 1.5 s windows, 0.75 s hop, one 192-d cam++ embedding each
  3. cosine affinity -> row-wise pruning -> symmetrise -> normalised Laplacian
  4. eigengap picks the speaker count; k-means on the leading eigenvectors
  5. temporal median smoothing, then each word takes the nearest window's label
  6. consecutive same-speaker words become one turn (= the framework's utterance)

Speaker count is estimated, not assumed, because the filenames say "FocusGroup"
while the pilot session is plainly a one-on-one interview.

Usage: python scripts/03_diarize.py [S01 S02 ...] [--max-speakers 6]
"""
import argparse
import json
import os
from pathlib import Path

# Scratch space for models, converted audio and intermediate JSON. Kept
# outside the project folder so cloud-sync clients do not churn on it.
# Override with the HCP_WORK environment variable.
WORK = Path(os.environ.get("HCP_WORK") or (Path.home() / "hcp_work"))
os.environ.setdefault("MODELSCOPE_CACHE", str(WORK / "models"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from funasr import AutoModel  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
ASR_DIR = WORK / "asr_json"
OUT_DIR = WORK / "diar_json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
WIN_S, HOP_S, MIN_WIN_S = 1.5, 0.75, 0.6


def speech_intervals(words, max_gap=0.35):
    """Merge word timings into contiguous speech regions."""
    iv = []
    for w in words:
        if not iv or w["s"] - iv[-1][1] > max_gap:
            iv.append([w["s"], w["e"]])
        else:
            iv[-1][1] = max(iv[-1][1], w["e"])
    return [(a, b) for a, b in iv if b - a >= 0.2]


def make_windows(intervals):
    out = []
    for a, b in intervals:
        if b - a <= WIN_S:
            if b - a >= MIN_WIN_S:
                out.append((a, b))
            continue
        t = a
        while t + MIN_WIN_S <= b:
            out.append((t, min(t + WIN_S, b)))
            t += HOP_S
    return out


def embed(model, audio, windows, batch=64):
    vecs = []
    for i in range(0, len(windows), batch):
        chunk = [audio[int(s * SR): int(e * SR)] for s, e in windows[i: i + batch]]
        res = model.generate(input=chunk, extract_embedding=True)
        for r in res:
            vecs.append(np.asarray(r["spk_embedding"]).reshape(-1))
    v = np.stack(vecs)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def spectral_labels(emb, min_k=1, max_k=6, prune=0.90):
    """Refined spectral clustering; eigengap chooses the speaker count."""
    n = len(emb)
    if n < 4:
        return np.zeros(n, dtype=int), 1
    sim = emb @ emb.T
    # row-wise pruning: keep the strongest links, which denoises the affinity
    thr = np.quantile(sim, prune, axis=1, keepdims=True)
    a = np.where(sim >= thr, sim, sim * 0.01)
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0)
    d = a.sum(1)
    d[d <= 0] = 1e-9
    dinv = 1.0 / np.sqrt(d)
    lap = np.eye(n) - (a * dinv[:, None]) * dinv[None, :]
    vals, vecs = np.linalg.eigh(lap)
    vals = np.clip(vals, 0, None)
    top = min(max_k + 1, n)
    gaps = np.diff(vals[:top])
    k = int(np.argmax(gaps)) + 1
    k = max(min_k, min(k, max_k))
    if k == 1:
        return np.zeros(n, dtype=int), 1
    feat = vecs[:, :k]
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(feat)
    return lab, k


def smooth(labels, width=5):
    out = labels.copy()
    half = width // 2
    for i in range(len(labels)):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        vals, cnt = np.unique(labels[lo:hi], return_counts=True)
        out[i] = vals[np.argmax(cnt)]
    return out


def build_turns(words, centers, labels, max_gap=1.5):
    """Assign each word the nearest window's speaker, then group into turns."""
    turns = []
    for w in words:
        mid = (w["s"] + w["e"]) / 2
        spk = int(labels[int(np.argmin(np.abs(centers - mid)))])
        if turns and turns[-1]["spk"] == spk and w["s"] - turns[-1]["end_s"] <= max_gap:
            turns[-1]["end_s"] = w["e"]
            turns[-1]["words"].append(w["w"])
        else:
            turns.append({"spk": spk, "start_s": w["s"], "end_s": w["e"],
                          "words": [w["w"]]})
    for t in turns:
        t["text"] = "".join(t.pop("words")).strip()
        t["start_s"], t["end_s"] = round(t["start_s"], 3), round(t["end_s"], 3)
    return [t for t in turns if t["text"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--max-speakers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    files = sorted(ASR_DIR.glob("S*.json"))
    if args.sessions:
        files = [f for f in files if f.stem in args.sessions]
    files = [f for f in files if args.overwrite or not (OUT_DIR / f.name).exists()]
    if not files:
        print("nothing to diarize")
        return

    print("[init] loading cam++ (cpu: faster than gpu here, per-call overhead dominates)",
          flush=True)
    model = AutoModel(model="cam++", device="cpu", disable_update=True)

    for i, f in enumerate(files, start=1):
        sid = f.stem
        asr = json.loads(f.read_text(encoding="utf-8"))
        words = [dict(w, seg=s["id"]) for s in asr["segments"]
                 for w in s["words"] if w["e"] > w["s"]]
        if not words:
            print(f"[{i}/{len(files)}] {sid}: no words, skipped", flush=True)
            continue
        wav = WORK / "audio_16k" / f"{sid}.wav"
        audio, _ = sf.read(str(wav), dtype="float32")
        wins = make_windows(speech_intervals(words))
        print(f"[{i}/{len(files)}] {sid}: {len(words)} words, {len(wins)} windows ...",
              end=" ", flush=True)
        emb = embed(model, audio, wins)
        labels, k = spectral_labels(emb, max_k=args.max_speakers)
        labels = smooth(labels)
        centers = np.array([(a + b) / 2 for a, b in wins])
        turns = build_turns(words, centers, labels)
        word_spk = []
        for w in words:
            mid = (w["s"] + w["e"]) / 2
            word_spk.append({
                "s": w["s"], "e": w["e"], "w": w["w"], "seg": w["seg"],
                "spk": int(labels[int(np.argmin(np.abs(centers - mid)))]),
            })
        dur = {}
        for t in turns:
            dur[t["spk"]] = dur.get(t["spk"], 0) + t["end_s"] - t["start_s"]
        (OUT_DIR / f"{sid}.json").write_text(json.dumps({
            "session_id": sid, "n_speakers_est": k, "n_turns": len(turns),
            "speaker_seconds": {str(a): round(b, 1) for a, b in sorted(dur.items())},
            "turns": turns,
            # word-level labels let script 04 rebuild units at any granularity
            # (sentence-like segment vs full turn) without re-embedding audio
            "words": word_spk,
        }, ensure_ascii=False), encoding="utf-8")
        share = ", ".join(f"spk{a}:{b/sum(dur.values())*100:.0f}%"
                          for a, b in sorted(dur.items()))
        print(f"k={k}, {len(turns)} turns, {share}", flush=True)


if __name__ == "__main__":
    main()
