"""
Step 1.2 (part 3) — Build the L1 utterance table, with de-identification.

Does four things:

  A. Speaker role mapping. The framework insists this is confirmed per session,
     never defaulted. Two independent signals are used, and any disagreement is
     flagged in data/role_assignment_check.csv for manual review:
       - slate signal:   whoever reads the recording slate
                         ("for the audio recording today is ...") = interviewer
       - consent signal: whoever says "my study ID is PN0xx" = participant
       - question rate:  fallback only, interviewers ask and participants answer

  B. Participant IDs. Study IDs (PN0xx) are spoken aloud in the verbal consent,
     so participant_id is recovered from the audio itself.

  C. De-identification. Real names are spoken during consent. Per the chosen
     policy, outputs carry only study IDs: PERSON spans (regex on the consent
     formula plus BERT NER over every turn) become [NAME], and the name-to-ID
     mapping is written to data/private/ so it stays out of any report.

  D. Utterance fields per the framework's 1.2 spec, with backchannels flagged
     (<1 s or <3 words) rather than deleted, so the timeline stays intact.

Usage: python scripts/04_build_utterances.py [--no-ner]
"""
import argparse
import json
import os
import re
from pathlib import Path

# Scratch space for models, converted audio and intermediate JSON. Kept
# outside the project folder so cloud-sync clients do not churn on it.
# Override with the HCP_WORK environment variable.
WORK = Path(os.environ.get("HCP_WORK") or (Path.home() / "hcp_work"))
os.environ.setdefault("HF_HOME", str(WORK / "models" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
DIAR_DIR = WORK / "diar_json"
DATA = PROJECT / "data"
PRIVATE = DATA / "private"
PRIVATE.mkdir(parents=True, exist_ok=True)

# Study IDs are spoken aloud and Whisper renders them inconsistently:
#   "PN-0nn", "PN0nn", "pn0nn"     plain
#   "P p.m. 004"                   the "PN" heard as "p.m."
#   "P-N-O-21", "PNO25"            a spoken zero written as the letter O
#   "PN00nn8"                      spurious trailing digits
#   "PIN 032"                      "PN" heard as the word "pin"
# So the prefix is matched loosely and the body is normalised in norm_id().
_PREFIX = r"P[\s.\-]*I?[\s.\-]*(?:[NM]|p\.?\s*m\.?)[\s.\-]*"
_BODY = r"([0-9Oo][0-9Oo\s.\-]{0,8})"
STUDY_ID_RE = re.compile(r"\b" + _PREFIX + _BODY, re.I)
# Fallback: the number sometimes lands in a later turn than the "PN" -- a
# participant reads "My study ID is PN" then, seconds later, "study id 0nn".
LOOSE_ID_RE = re.compile(
    r"study\s*(?:id|iv)\s*(?:is\s*)?(?:" + _PREFIX + r")?" + _BODY, re.I)
# diarization sometimes splits "My" off the front of the consent turn
NAME_RE = re.compile(r"\b(?:[Mm]y\s+)?name is\s+([A-Z][\w\u2019-]+(?:\s+[A-Z][\w\u2019-]+){0,3})")
SLATE_RE = re.compile(r"for the (purposes of the )?audio recording", re.I)
# "study ID" also comes back as "study iv" / "study I.D." from the ASR
CONSENT_RE = re.compile(r"study\s*(?:id|iv|i\.\s*d\.?)|i consent to participat", re.I)


def get_ner():
    from transformers import pipeline
    return pipeline("token-classification", model="dslim/bert-base-NER",
                    aggregation_strategy="simple", device=-1)


# Tokens NER sometimes mislabels as people in disfluent speech.
GAZ_STOP = {"okay", "yeah", "yep", "uh", "um", "er", "mm", "hmm", "like", "sorry",
            "thanks", "thank", "hey", "hi", "oh", "well", "you", "and", "the",
            "ed", "psw", "psws", "narcan", "triage", "nurse", "doctor"}
# Add your site's own abbreviations (hospital, unit, team acronyms) here: NER
# regularly mistakes short all-caps tokens for surnames.


WORD_CH = re.compile(r"[\w’-]")


def _widen(text, a, b):
    """Grow a span to whole-word edges.

    The NER head splits names across word-pieces, so a short first name can come
    back as a 2-character span (e.g. 'Anneke' tagged only as 'An'). Left as-is that both under-redacts and poisons the gazetteer with
    a 2-character stub, so every span is widened to its full word first.
    """
    while a > 0 and WORD_CH.match(text[a - 1]):
        a -= 1
    while b < len(text) and WORD_CH.match(text[b]):
        b += 1
    return a, b


def ner_persons(text, ner):
    """PERSON spans and surface forms from the cased NER model."""
    spans, found = [], set()
    if ner is None or not text.strip():
        return spans, found
    try:
        for e in ner(text):
            if e["entity_group"] == "PER" and e["score"] >= 0.80:
                a, b = _widen(text, int(e["start"]), int(e["end"]))
                spans.append((a, b))
                found.add(text[a:b].strip())
    except Exception:
        pass
    return spans, found


def build_gazetteer(entities, corpus_caps):
    """Case-insensitive name list, needed because Whisper emits some passages
    entirely in lower case, where a cased NER model silently fails -- a NER-only
    pass left several lower-cased first and last names in the text.

    A token is only admitted if NER tagged it as a person somewhere AND it was
    seen capitalised somewhere in the corpus -- that keeps ordinary words that
    happen to look like names out of the list.
    """
    gaz = set()
    for ent in entities:
        for tok in re.split(r"[^\w’-]+", ent):
            tok = tok.strip("’-")
            if (len(tok) >= 3 and tok.isalpha()
                    and tok.lower() not in GAZ_STOP
                    and tok.lower() in corpus_caps):
                gaz.add(tok.lower())
    return gaz


def _loose(token):
    """Allow doubled-letter variation: 'anneke' also matches 'aneke'.

    Whisper is inconsistent about gemination in names, so the same person can
    show up spelled with both a single and a doubled consonant. Collapsing each repeated-letter run to {1,2}
    catches both without the blast radius of general fuzzy matching.
    """
    out, i = [], 0
    while i < len(token):
        j = i
        while j + 1 < len(token) and token[j + 1] == token[i]:
            j += 1
        out.append(re.escape(token[i]) + "{1,2}")
        i = j + 1
    return "".join(out)


def gaz_spans(text, gaz):
    if not gaz:
        return []
    pat = r"\b(?:%s)\b" % "|".join(_loose(t) for t in
                                   sorted(gaz, key=len, reverse=True))
    return [(m.start(), m.end()) for m in re.finditer(pat, text, re.I)]


def redact(text, spans_in, extra_names, gaz):
    """Replace PERSON spans with [NAME]; returns (clean_text, names_found)."""
    found = set()
    spans = list(spans_in)
    for m in NAME_RE.finditer(text):
        spans.append((m.start(1), m.end(1)))
        found.add(m.group(1).strip())
    for nm in extra_names:
        for m in re.finditer(re.escape(nm), text, re.I):
            spans.append((m.start(), m.end()))
    spans.extend(gaz_spans(text, gaz))
    if not spans:
        return text, found
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, prev = [], 0
    for s, e in merged:
        out.append(text[prev:s])
        out.append("[NAME]")
        prev = e
    out.append(text[prev:])
    return "".join(out), found


def build_units(words):
    """L1 unit = a speaker-homogeneous, sentence-like span.

    Whisper already segments into sentence-ish spans; we split those further
    wherever the diarizer changes speaker. Using the full conversational turn
    instead would leave only ~2.7k units for 12.75 h, far too coarse for the
    trajectory analysis in framework 5.2 (which slides a 5-7 unit window).
    turn_index is retained so turn structure is still available for coding.
    """
    units, turn_index, prev_spk = [], 0, None
    for w in words:
        if w["spk"] != prev_spk:
            turn_index += 1
            prev_spk = w["spk"]
        key = (w["spk"], w["seg"], turn_index)
        if units and units[-1]["key"] == key:
            units[-1]["end_s"] = w["e"]
            units[-1]["words"].append(w["w"])
        else:
            units.append({"key": key, "spk": w["spk"], "seg": w["seg"],
                          "turn_index": turn_index, "start_s": w["s"],
                          "end_s": w["e"], "words": [w["w"]]})
    out = []
    for u in units:
        text = "".join(u.pop("words")).strip()
        u.pop("key")
        if text:
            u["text"] = text
            u["start_s"], u["end_s"] = round(u["start_s"], 3), round(u["end_s"], 3)
            out.append(u)
    return out


def norm_id(body):
    """Normalise a spoken ID body to PN###.

    '0nn'->PN0nn, 'O-nn'->PN0nn, 'Onn'->PN0nn, '00nn8'->PN0nn.

    A spoken zero is often transcribed as the letter O, so O is folded to 0.
    Whisper also runs spurious digits onto the end ("PN00nn8" for PN0nn), so
    only the first three digits are trusted; anything longer is flagged in
    role_assignment_check.csv for human confirmation.
    """
    digits = re.sub(r"[^0-9]", "", body.replace("O", "0").replace("o", "0"))
    if not digits:
        return None, True
    return "PN%03d" % int(digits[:3]), len(digits) > 3


def speaker_ids(turns, interviewer_spk):
    """Map each non-interviewer speaker to their own study ID.

    Per-speaker rather than per-session because several recordings hold two
    consenting participants; assigning a single id per session would silently
    merge two people into one L3 unit.
    """
    ids, suspect = {}, {}
    for tier, rx, need_consent in ((1, STUDY_ID_RE, True), (2, LOOSE_ID_RE, False)):
        for t in turns:
            spk = t["spk"]
            if spk == interviewer_spk or spk in ids:
                continue
            if need_consent and not CONSENT_RE.search(t["text"]):
                continue
            m = rx.search(t["text"])
            if m:
                pid, odd = norm_id(m.group(1))
                if pid is None:
                    continue
                ids[spk] = pid
                suspect[spk] = odd or tier == 2
    return ids, suspect


def assign_roles(turns):
    """Return (mapping spk->role, diagnostics dict)."""
    spk_ids = sorted({t["spk"] for t in turns})
    consent_spk = slate_spk = None
    for t in turns:
        if consent_spk is None and CONSENT_RE.search(t["text"]):
            consent_spk = t["spk"]
        if slate_spk is None and SLATE_RE.search(t["text"]):
            slate_spk = t["spk"]
    qrate, dur = {}, {}
    for s in spk_ids:
        ts = [t for t in turns if t["spk"] == s]
        qrate[s] = sum("?" in t["text"] for t in ts) / max(len(ts), 1)
        dur[s] = sum(t["end_s"] - t["start_s"] for t in ts)
    interviewer = slate_spk if slate_spk is not None else max(qrate, key=qrate.get)
    mapping = {s: ("interviewer" if s == interviewer else "participant")
               for s in spk_ids}
    return mapping, {
        "slate_spk": slate_spk,
        "consent_spk": consent_spk,
        "interviewer_spk": interviewer,
        # the participant should be the one giving consent, not the interviewer
        "consent_agrees": (consent_spk is None or consent_spk != interviewer),
        "question_rate_interviewer": round(qrate[interviewer], 3),
        "speaker_seconds": {s: round(dur[s], 1) for s in spk_ids},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ner", action="store_true")
    args = ap.parse_args()

    sessions = pd.read_csv(DATA / "sessions.csv").set_index("session_id")
    files = sorted(DIAR_DIR.glob("S*.json"))
    if not files:
        raise SystemExit("no diarization output yet; run scripts/03_diarize.py first")

    ner = None if args.no_ner else get_ner()
    rows, checks, namemap = [], [], []

    # ---- pass 1: parse sessions, run NER once, collect every person name -----
    parsed, entities, caps = [], set(), set()
    for f in files:
        sid = f.stem
        d = json.loads(f.read_text(encoding="utf-8"))
        turns = d["turns"]
        units = build_units(d["words"]) if d.get("words") else turns
        mapping, diag = assign_roles(turns)

        interviewer_spk = diag["interviewer_spk"]
        ids, suspect = speaker_ids(turns, interviewer_spk)

        # Speakers with a trivial share of the non-interviewer speech are
        # clustering artefacts (one recording yields a 0.5 s third cluster).
        # They are
        # marked 'unassigned' so they never enter the participant analysis.
        spk_dur = {}
        for t in turns:
            spk_dur[t["spk"]] = spk_dur.get(t["spk"], 0) + t["end_s"] - t["start_s"]
        non_iv = {s: d for s, d in spk_dur.items() if s != interviewer_spk}
        total_non_iv = sum(non_iv.values()) or 1.0
        minor = {s for s, d in non_iv.items()
                 if d / total_non_iv < 0.02 and s not in ids}

        spk_pid = {}
        for s in spk_dur:
            if s == interviewer_spk:
                spk_pid[s] = "INTERVIEWER_" + sid
            elif s in ids:
                spk_pid[s] = ids[s]
            elif len(non_iv) - len(minor) == 1 and s not in minor:
                spk_pid[s] = "UNK_" + sid
            else:
                spk_pid[s] = "UNK_%s_spk%d" % (sid, s)
        for s in minor:
            mapping[s] = "unassigned"

        # spoken names, per speaker, for the private mapping
        spoken_by_spk = {}
        for t in turns:
            if t["spk"] in spoken_by_spk or t["spk"] == interviewer_spk:
                continue
            m = NAME_RE.search(t["text"])
            if m:
                spoken_by_spk[t["spk"]] = m.group(1).strip()
        for s, nm in spoken_by_spk.items():
            namemap.append({"session_id": sid, "participant_id": spk_pid.get(s, ""),
                            "spoken_name": nm})
            entities.add(nm)
        spoken = next(iter(spoken_by_spk.values()), None)

        spans_per_unit = []
        for t in units:
            sp, fnd = ner_persons(t["text"], ner)
            spans_per_unit.append(sp)
            entities |= fnd
            for w in re.findall(r"\b[A-Z][a-z’-]{2,}\b", t["text"]):
                caps.add(w.lower())
        parsed.append((sid, d, turns, units, mapping, diag, spk_pid,
                       spoken, spans_per_unit, suspect, minor))
        print("%s: scanned %d units" % (sid, len(units)), flush=True)

    gaz = build_gazetteer(entities, caps)
    print("\n[deid] %d person surface forms -> %d gazetteer tokens applied "
          "case-insensitively" % (len(entities), len(gaz)), flush=True)

    # ---- pass 2: redact and emit ------------------------------------------
    for (sid, d, turns, units, mapping, diag, spk_pid, spoken,
         spans_per_unit, suspect, minor) in parsed:
        extra = [spoken] if spoken else []

        for j, t in enumerate(units, start=1):
            clean, _ = redact(t["text"], spans_per_unit[j - 1], extra, gaz)
            nw = len(clean.split())
            dur = round(t["end_s"] - t["start_s"], 3)
            rows.append({
                "utt_id": "%s_%04d" % (sid, j),
                "session_id": sid,
                "participant_id": spk_pid[t["spk"]],
                "speaker_raw": "spk%d" % t["spk"],
                "speaker_role": mapping[t["spk"]],
                "turn_index": t.get("turn_index", j),
                "seg_index": t.get("seg"),
                "start_s": t["start_s"],
                "end_s": t["end_s"],
                "duration_s": dur,
                "text": clean,
                "n_words": nw,
                "n_chars": len(clean),
                "words_per_s": round(nw / dur, 3) if dur > 0 else 0.0,
                # framework 1.2: keep short acknowledgements on the timeline but
                # exclude them from emotion analysis
                "is_backchannel": int(dur < 1.0 or nw < 3),
            })

        pids = sorted({v for k, v in spk_pid.items()
                       if mapping.get(k) == "participant"})
        checks.append({
            "session_id": sid,
            "participant_ids": "|".join(pids),
            "n_participants": len(pids),
            "id_needs_review": any(suspect.values()),
            "minor_speakers_dropped": len(minor),
            "n_speakers_est": d["n_speakers_est"],
            "n_turns": len(turns),
            "n_units": len(units),
            "name_recovered": bool(spoken),
            **{k: (json.dumps(v) if isinstance(v, dict) else v)
               for k, v in diag.items()},
        })
        print("%s: pid=%s k=%d turns=%d units=%d roles_ok=%s%s"
              % (sid, ",".join(pids) or "NONE", d["n_speakers_est"], len(turns),
                 len(units), diag["consent_agrees"],
                 "  REVIEW-ID" if any(suspect.values()) else ""), flush=True)

    utt = pd.DataFrame(rows)
    utt = utt.merge(sessions[["interview_date", "recording_device", "duration_min"]],
                    left_on="session_id", right_index=True, how="left")
    utt.to_csv(DATA / "utterances.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(checks).to_csv(DATA / "role_assignment_check.csv", index=False,
                                encoding="utf-8-sig")
    if namemap:
        pd.DataFrame(namemap).to_csv(PRIVATE / "name_map.csv", index=False,
                                     encoding="utf-8-sig")

    p = utt[utt.speaker_role == "participant"]
    known = p[~p.participant_id.str.startswith("UNK")].session_id.nunique()
    print("\nutterances.csv: %d turns across %d sessions"
          % (len(utt), utt.session_id.nunique()))
    print("  participant turns: %d (%d after dropping backchannels)"
          % (len(p), len(p[p.is_backchannel == 0])))
    print("  participant IDs recovered: %d/%d" % (known, utt.session_id.nunique()))
    print("  private name map: %d rows -> data/private/name_map.csv" % len(namemap))


if __name__ == "__main__":
    main()
