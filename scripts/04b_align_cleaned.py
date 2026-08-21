"""
Join the human-reviewed transcripts to the audio timeline.

The cleaned transcripts have the right words and the right speaker labels but no
timestamps; the ASR pass has timestamps but ~10% word error. This step gives
each cleaned sentence a start and end time by aligning the two word streams, so
the acoustic measures (pitch, loudness, pace) can be attached to text a human
has verified.

Method: both transcripts describe the same conversation in the same order, so a
plain word-level diff aligns them well. Matched blocks give each cleaned word an
ASR word index and therefore a time; unmatched runs (ASR errors, removed
fillers) are interpolated between their nearest matched neighbours.

Unit choice. Coding runs on whole turns (cheap: 1,197 of them), but acoustic
emotion needs finer units or a 60-word answer collapses to one number. So turns
are split into sentences here, and `turn_index` is carried through so turn-level
codes can be joined back onto the sentences that make them up.

Writes the canonical L1 table, replacing the ASR-derived one (kept as
data/utterances_asr.csv for reference).

Usage: python scripts/04b_align_cleaned.py
"""
import difflib
import json
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
# Scratch space for models, converted audio and intermediate JSON. Kept
# outside the project folder so cloud-sync clients do not churn on it.
# Override with the HCP_WORK environment variable.
WORK = Path(os.environ.get("HCP_WORK") or (Path.home() / "hcp_work"))
DIAR = WORK / "diar_json"

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
NOISE_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
WORD_RE = re.compile(r"[a-z0-9']+")


def norm(text):
    return WORD_RE.findall(NOISE_RE.sub(" ", str(text).lower()))


def split_sentences(text, max_words=45):
    """Sentence-ish units; long run-ons are chopped so one unit stays local."""
    out = []
    for s in SENT_SPLIT.split(str(text).strip()):
        s = s.strip()
        if not s:
            continue
        w = s.split()
        if len(w) <= max_words:
            out.append(s)
        else:
            for i in range(0, len(w), max_words):
                out.append(" ".join(w[i:i + max_words]))
    return out or [str(text).strip()]


def align_times(clean_words, asr_words):
    """Map each cleaned word index -> (start, end) seconds."""
    # ASR tokens arrive as ' Yeah,' -- with a leading space and punctuation --
    # so they must be normalised the same way as the cleaned words or nothing
    # ever matches. Keep a 1:1 index with asr_words so times stay addressable.
    hyp = []
    for w in asr_words:
        tok = WORD_RE.findall(str(w["w"]).lower())
        hyp.append(tok[0] if tok else "\x00")
    sm = difflib.SequenceMatcher(a=clean_words, b=hyp, autojunk=False)
    idx = np.full(len(clean_words), -1, dtype=int)
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            idx[i + k] = j + k
    # fill gaps by interpolating between the nearest matched anchors
    known = np.flatnonzero(idx >= 0)
    if len(known) == 0:
        return None, 0.0
    filled = idx.copy().astype(float)
    filled[idx < 0] = np.nan
    xs = np.arange(len(clean_words))
    filled = np.interp(xs, xs[known], idx[known].astype(float))
    times = []
    for v in filled:
        j = int(round(min(max(v, 0), len(asr_words) - 1)))
        times.append((asr_words[j]["s"], asr_words[j]["e"]))
    return times, len(known) / len(clean_words)


def main() -> None:
    src = DATA / "cleaned_turns.csv"
    if not src.exists():
        raise SystemExit("run scripts/02b_parse_cleaned.py first")
    turns = pd.read_csv(src)
    sessions = pd.read_csv(DATA / "sessions.csv").set_index("session_id")

    rows, report = [], []
    for sid, g in turns.groupby("session_id", sort=True):
        dj = DIAR / ("%s.json" % sid)
        if not dj.exists():
            print("  [skip] %s: no ASR word timings" % sid)
            continue
        asr_words = json.loads(dj.read_text(encoding="utf-8"))["words"]
        g = g.sort_values("turn_index")

        # one flat word stream for the whole session, remembering which
        # sentence each word came from
        clean_words, owner, units = [], [], []
        for _, r in g.iterrows():
            for sent in split_sentences(r.text):
                ws = norm(sent)
                if not ws:
                    continue
                units.append({"turn_index": int(r.turn_index),
                              "speaker_role": r.speaker_role,
                              "participant_id": r.participant_id,
                              "speaker_label": r.speaker_label,
                              "text": sent})
                for w in ws:
                    clean_words.append(w)
                    owner.append(len(units) - 1)

        times, cov = align_times(clean_words, asr_words)
        if times is None:
            print("  [skip] %s: alignment failed" % sid)
            continue

        owner = np.asarray(owner)
        for ui, u in enumerate(units):
            sel = np.flatnonzero(owner == ui)
            if len(sel) == 0:
                continue
            st = min(times[k][0] for k in sel)
            en = max(times[k][1] for k in sel)
            if en <= st:
                en = st + 0.25
            nw = len(u["text"].split())
            dur = round(en - st, 3)
            rows.append({
                "utt_id": "%s_%04d" % (sid, ui + 1),
                "session_id": sid,
                "participant_id": u["participant_id"],
                "speaker_raw": u["speaker_label"],
                "speaker_role": u["speaker_role"],
                "turn_index": u["turn_index"],
                "seg_index": ui + 1,
                "start_s": round(st, 3),
                "end_s": round(en, 3),
                "duration_s": dur,
                "text": u["text"],
                "n_words": nw,
                "n_chars": len(u["text"]),
                "words_per_s": round(nw / dur, 3) if dur > 0 else 0.0,
                "is_backchannel": int(dur < 1.0 or nw < 3),
            })
        report.append({"session_id": sid, "units": len(units),
                       "align_coverage": round(cov, 3)})
        print("  %s: %d units, %.0f%% of words matched to audio"
              % (sid, len(units), 100 * cov), flush=True)

    out = pd.DataFrame(rows)
    out = out.merge(sessions[["interview_date", "recording_device", "duration_min"]],
                    left_on="session_id", right_index=True, how="left")

    asr_path = DATA / "utterances.csv"
    if asr_path.exists() and not (DATA / "utterances_asr.csv").exists():
        asr_path.rename(DATA / "utterances_asr.csv")
        print("\nkept the ASR-derived table as data/utterances_asr.csv")
    out.to_csv(asr_path, index=False, encoding="utf-8-sig")

    rep = pd.DataFrame(report)
    rep.to_csv(PROJECT / "outputs" / "tables" / "alignment_quality.csv",
               index=False, encoding="utf-8-sig")

    p = out[out.speaker_role == "participant"]
    print("\nwrote data/utterances.csv from the cleaned transcripts")
    print("  %d units (%d participant, %d after backchannels), %d participants"
          % (len(out), len(p), int((p.is_backchannel == 0).sum()),
             p.participant_id.nunique()))
    print("  alignment coverage: median %.0f%%, worst %.0f%% (%s)"
          % (100 * rep.align_coverage.median(), 100 * rep.align_coverage.min(),
             rep.loc[rep.align_coverage.idxmin(), "session_id"]))
    bad = rep[rep.align_coverage < 0.6]
    if len(bad):
        print("  [warn] low-coverage sessions, timings there are approximate: %s"
              % bad.session_id.tolist())


if __name__ == "__main__":
    main()
