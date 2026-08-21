"""
Step 1.3 — Topic segmentation (the framework's L2 unit).

The framework offers manual marking or a semi-automatic LLM pass. A third route
is available here and is both cheaper and more auditable: interviewers announce
every section transition out loud ("the next section asks about challenges and
areas of improvement"). Boundaries are detected by matching transition cues
against *interviewer* speech only, then forced to run in guide order.

Two things make this harder than it sounds, and both are handled here:

  1. L1 units are sentence-sized, so a transition phrase is routinely split
     across two of them ("...the next set of questions will be regarding
     challenges and a" + "reas for improvement"). Matching per utterance misses
     these. Instead the interviewer's speech is concatenated into one stream with
     a char-offset -> timestamp index, and cues are matched against the stream.
  2. The interviewer reads a scripted description of the PSW role before the
     first question, and it contains the words "services" and "challenges".
     Ordered matching protects against this: each section is only searched for
     *after* the previous section's boundary, and the boilerplate precedes them.

Still semi-automatic in the framework's sense: data/topic_boundaries.csv lists
every boundary with the sentence that triggered it, which is the worksheet for
the 20% manual verification framework 1.3 asks for.

Usage: python scripts/06_topic_segments.py
"""
import re
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"

# Ordered: a section can only be entered after the ones before it.
#
# There is deliberately no `closing` section. The wrap-up after the last
# scripted question ran to only 3% of participant speech from 19 of 34
# people, and mixed genuine forward-looking answers with meta-comments about
# the study itself. Dropping the boundary lets it fold into `sustainability`,
# whose content it shares, and brings that section to full coverage.
# Interviewers vary the wording a lot, so each section carries a transition
# frame ("next few questions", "next section", ...) plus its own first question.
_NEXT = r"(?:next|following)\s+(?:few\s+|set\s+of\s+|group\s+of\s+|couple\s+of\s+)?" \
        r"(?:questions?|sections?)"
_SECT = r"(?:next\s+section|this\s+section|section\s+(?:here\s+)?is)"

SECTIONS = [
    ("experiences", [
        r"\bfirst question\b[^.?]{0,90}\bdescribe your experience",
        r"\bdescribe your experiences?\b[^.?]{0,50}peer support worker",
        r"questions?\s+(?:related to|about)\s+your experiences",
        r"\bget into the first question\b",
        r"\bjump into (?:our|the) first question\b",
        r"\bstart (?:off )?with (?:the |our )?(?:first )?questions?\b",
    ]),
    ("services", [
        _NEXT + r"[^.?]{0,80}\bservices\b",
        _SECT + r"[^.?]{0,60}\bservices\b",
        r"\bservices provided\b[^.?]{0,80}\b(?:met|meet)\s+the\s+needs\b",
        r"\bhave the services provided\b",
        r"questions?[^.?]{0,60}\bED workflow\b",
    ]),
    ("challenges", [
        _NEXT + r"[^.?]{0,80}\bchallenges\b",
        _SECT + r"[^.?]{0,60}\bchallenges\b",
        r"\bchallenges and areas (?:of|for) improvement\b",
        r"\bhave you experienced (?:any )?challenges\b",
    ]),
    ("sustainability", [
        _NEXT + r"[^.?]{0,80}(?:sustainab|future development)",
        _SECT + r"[^.?]{0,60}(?:sustainab|future development|the future)",
        r"\bsustainability and future development\b",
        r"\bwhat role do you see peer support workers\b",
        r"\bbenefit in sustaining\b",
    ]),
]

# Whisper renders "ED" as EU/AD/EV/E.D. depending on the speaker.
ED_FIX = re.compile(r"\b(the )?(EU|E\.?D\.?|AD|EV|ADE)\b")


def normalise(text: str) -> str:
    t = ED_FIX.sub(lambda m: (m.group(1) or "") + "ED", str(text))
    return re.sub(r"\s+", " ", t).strip()


def build_stream(iv: pd.DataFrame):
    """Concatenate interviewer speech, keeping a char-offset -> row index map."""
    parts, index, pos = [], [], 0
    for i, r in enumerate(iv.itertuples()):
        t = normalise(r.text)
        if not t:
            continue
        parts.append(t)
        index.append((pos, pos + len(t), i))
        pos += len(t) + 1
    return " ".join(parts), index


def locate(index, offset):
    """Row index of the utterance containing this char offset."""
    for a, b, i in index:
        if a <= offset < b:
            return i
    return index[-1][2] if index else 0


def find_boundaries(iv: pd.DataFrame):
    stream, index = build_stream(iv)
    if not stream:
        return []
    rows = list(iv.itertuples())
    found, search_from = [], 0
    for name, pats in SECTIONS:
        best = None
        for p in pats:
            m = re.compile(p, re.I).search(stream, search_from)
            if m and (best is None or m.start() < best.start()):
                best = m
        if best is None:
            continue
        r = rows[locate(index, best.start())]
        found.append({"topic_segment": name, "start_s": r.start_s,
                      "utt_id": r.utt_id,
                      "cue": re.sub(r"\s+", " ", best.group(0))[:160]})
        search_from = best.end()
    return found


def main() -> None:
    src = DATA / "utterances_acoustic.csv"
    if not src.exists():
        src = DATA / "utterances.csv"
    utt = pd.read_csv(src)

    all_bounds, tagged = [], []
    for sid, grp in utt.groupby("session_id", sort=True):
        grp = grp.sort_values("start_s").copy()
        iv = grp[grp.speaker_role == "interviewer"]
        bounds = find_boundaries(iv)
        for b in bounds:
            all_bounds.append({"session_id": sid, **b})

        seg = pd.Series("intro_consent", index=grp.index)
        for b in bounds:
            seg[grp.start_s >= b["start_s"]] = b["topic_segment"]
        grp["topic_segment"] = seg.values
        tagged.append(grp)

        names = [b["topic_segment"] for b in bounds]
        missing = [n for n, _ in SECTIONS if n not in names]
        print("%s: %d/%d boundaries%s"
              % (sid, len(bounds), len(SECTIONS),
                 ("  MISSING: " + ",".join(missing)) if missing else ""), flush=True)

    out = pd.concat(tagged).sort_values(["session_id", "start_s"])
    out.to_csv(DATA / "utterances_topics.csv", index=False, encoding="utf-8-sig")
    bdf = pd.DataFrame(all_bounds)
    bdf.to_csv(DATA / "topic_boundaries.csv", index=False, encoding="utf-8-sig")

    order = ["intro_consent"] + [n for n, _ in SECTIONS]
    p = out[(out.speaker_role == "participant") & (out.is_backchannel == 0)]
    tab = (p.groupby("topic_segment")
             .agg(n_utt=("utt_id", "size"),
                  n_sessions=("session_id", "nunique"),
                  minutes=("duration_s", lambda s: round(s.sum() / 60, 1)))
             .reindex(order).dropna(how="all"))
    print("\nParticipant speech by topic segment (backchannels excluded):")
    print(tab.to_string())

    if len(bdf):
        cov = bdf.groupby("session_id").size()
        full = int((cov == len(SECTIONS)).sum())
        print("\nBoundary coverage: %d/%d sessions complete, median %d/%d found"
              % (full, out.session_id.nunique(), cov.median(), len(SECTIONS)))
    print("\nwrote data/utterances_topics.csv and data/topic_boundaries.csv")
    print("Spot-check data/topic_boundaries.csv (framework 1.3 asks for 20%).")


if __name__ == "__main__":
    main()
