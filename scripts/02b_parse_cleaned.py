"""
Parse the human-reviewed transcripts and score the ASR against them.

The study team supplied `Cleaned Transcripts/` -- 30 .docx files, every one
reviewed by a person. They are better than the ASR output in four ways that
matter, and they settle an item the framework flags as mandatory:

  * the words are human-verified, so text analysis stops inheriting ASR errors;
  * speakers are labelled explicitly ("PN-028", "Facilitator 1"), which replaces
    both the diarization and the study-ID recovery with ground truth;
  * they are already de-identified ("[first name] [last name]");
  * framework 1.2 requires a transcription accuracy figure, which could not be
    produced without a human reference. Now it can -- see the WER table.

What they do NOT have is timestamps, so the acoustic side still needs the ASR
pass: pitch/loudness/pace can only be measured against the audio timeline.
The two are joined in a later step.

Outputs:
  data/cleaned_turns.csv              authoritative text, one row per speaker turn
  outputs/tables/asr_accuracy.csv     per-session WER against the human reference

Usage: python scripts/02b_parse_cleaned.py
"""
import difflib
import re
from pathlib import Path

import pandas as pd
from docx import Document

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "Cleaned Transcripts"
DATA = PROJECT / "data"
TABLES = PROJECT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

SPEAKER_RE = re.compile(r"^(facilitator\s*\d*|pn[-\s]?\d+|participant\s*\d*|"
                        r"interviewer\s*\d*)\s*$", re.I)
PN_RE = re.compile(r"pn[-\s]?(\d+)", re.I)
STAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{4})")
# bracketed redactions and transcriber annotations are not spoken words
NOISE_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def norm_words(text):
    t = NOISE_RE.sub(" ", str(text).lower())
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return [w for w in t.split() if w]


def wer(ref, hyp):
    """Word error rate via difflib opcodes (fast enough for whole sessions)."""
    if not ref:
        return float("nan"), 0
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    err = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            err += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            err += i2 - i1
        elif tag == "insert":
            err += j2 - j1
    return err / len(ref), len(ref)


def parse_docx(path):
    doc = Document(str(path))
    turns, speaker = [], None
    for para in doc.paragraphs:
        t = para.text.strip()
        if not t:
            continue
        if SPEAKER_RE.match(t):
            speaker = t
            continue
        if speaker is None:
            continue
        m = PN_RE.search(speaker)
        turns.append({
            "speaker_label": speaker,
            "speaker_role": "participant" if m else "interviewer",
            "participant_id": "PN%03d" % int(m.group(1)) if m else None,
            "text": t,
        })
    return turns


def main() -> None:
    if not SRC.exists():
        raise SystemExit("missing folder: %s" % SRC)
    sessions = pd.read_csv(DATA / "sessions.csv")
    stamp_to_sid = {}
    for _, r in sessions.iterrows():
        m = STAMP_RE.match(str(r.source_file))
        if m:
            stamp_to_sid[m.group(1)] = r.session_id

    rows, unmatched = [], []
    for f in sorted(SRC.glob("*.docx")):
        m = STAMP_RE.match(f.name)
        sid = stamp_to_sid.get(m.group(1)) if m else None
        if sid is None:
            unmatched.append(f.name)
            continue
        for i, t in enumerate(parse_docx(f), start=1):
            rows.append({"session_id": sid, "turn_index": i,
                         "source_docx": f.name, **t})

    df = pd.DataFrame(rows)
    # a facilitator turn carries no study id; participants keep theirs
    df.to_csv(DATA / "cleaned_turns.csv", index=False, encoding="utf-8-sig")
    print("parsed %d turns from %d files -> data/cleaned_turns.csv"
          % (len(df), df.session_id.nunique()))
    if unmatched:
        print("  [warn] could not map to a session: %s" % unmatched)

    part = df[df.speaker_role == "participant"]
    print("  participant turns: %d | distinct study IDs: %d"
          % (len(part), part.participant_id.nunique()))
    multi = (part.groupby("session_id").participant_id.nunique()
                 .loc[lambda s: s > 1])
    if len(multi):
        print("  multi-participant sessions: %s" % multi.to_dict())

    # ---- framework 1.2: ASR accuracy against the human reference -----------
    asr = pd.read_csv(DATA / "utterances.csv", low_memory=False)
    out = []
    for sid, g in df.groupby("session_id"):
        for role in ["participant", "interviewer"]:
            ref = norm_words(" ".join(g[g.speaker_role == role].text))
            hyp = norm_words(" ".join(
                asr[(asr.session_id == sid) & (asr.speaker_role == role)].text.fillna("")))
            if not ref:
                continue
            e, n = wer(ref, hyp)
            out.append({"session_id": sid, "role": role, "ref_words": n,
                        "hyp_words": len(hyp), "wer": round(e, 4)})
    acc = pd.DataFrame(out)
    acc.to_csv(TABLES / "asr_accuracy.csv", index=False, encoding="utf-8-sig")

    print("\n=== ASR accuracy vs human-reviewed transcripts (framework 1.2) ===")
    for role, g in acc.groupby("role"):
        tot_ref = g.ref_words.sum()
        weighted = (g.wer * g.ref_words).sum() / tot_ref
        print("  %-12s word error rate %.1f%%  (median %.1f%%, worst %.1f%% in %s)"
              % (role, 100 * weighted, 100 * g.wer.median(),
                 100 * g.wer.max(), g.loc[g.wer.idxmax(), "session_id"]))
    print("\nwrote outputs/tables/asr_accuracy.csv")
    print("Note: WER here mixes real ASR errors with legitimate editorial "
          "cleanup (fillers removed, false starts tidied), so it is an upper "
          "bound on true transcription error.")


if __name__ == "__main__":
    main()
