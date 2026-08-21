"""
Step 3.1 (step B) — LLM thematic coding of participant utterances.

Implements the framework's prompt design, with two deliberate changes:

  * Consecutive utterances are sent together with the interviewer's turns left in
    as context (framework: "keep the context"), but only participant lines are
    coded.
  * Demographics are NOT placed in the prompt. The framework's template includes
    them "for context only", but its own section 8 lists feeding demographics to
    the coder as a classic trap that makes later group comparison circular.
    Since RQ3 is precisely a demographic comparison, they are withheld.

Calls the local `claude` CLI in headless mode, so no API key is needed. Work is
chunked and cached per chunk under <WORK>/coding/, so the run is resumable and a
failed chunk never costs the whole session.

Usage:
  python scripts/08_llm_coding.py --limit-sessions 1     # pilot
  python scripts/08_llm_coding.py                        # everything
"""
import argparse
import json
import re
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
# Scratch space for models, converted audio and intermediate JSON. Kept
# outside the project folder so cloud-sync clients do not churn on it.
# Override with the HCP_WORK environment variable.
WORK = Path(os.environ.get("HCP_WORK") or (Path.home() / "hcp_work"))
CACHE = WORK / "coding"
CACHE.mkdir(parents=True, exist_ok=True)

VALID_STANCE = {"positive", "negative", "mixed", "neutral", "na"}
# v2 format: "### A1 `builds_rapport` — Builds rapport and trust with patients"
CODE_RE = re.compile(r"^###\s+(\w\d{1,2})\s+`([a-z_]+)`\s*[—-]\s*(.+)$", re.M)


def load_codebook():
    """Return (condensed_text, id->slug, id->label).

    The full codebook.md is ~12 KB and is re-sent on every one of ~113 calls,
    which dominated runtime. Only the machine-relevant part is sent: the domain
    headings, and one line per code (id, label, definition). The Table 4
    citations, examples and preamble are for human readers and are dropped.
    """
    text = (PROJECT / "docs" / "codebook.md").read_text(encoding="utf-8")
    codes, labels, lines = {}, {}, []
    for block in re.split(r"\n## ", text):
        head = block.split("\n", 1)[0].strip()
        if head.startswith("Domain"):
            lines.append("\n" + head)
        for m in CODE_RE.finditer(block):
            cid, slug, label = m.group(1), m.group(2), m.group(3).strip()
            codes[cid], labels[cid] = slug, label
            body = []
            for ln in block[m.end():].lstrip("\n").split("\n"):
                ln = ln.strip()
                if not ln or ln.startswith(("###", "##", "- *")):
                    break
                body.append(ln)
            lines.append("%s = %s. %s" % (cid, label, " ".join(body)[:230]))
    if not codes:
        sys.exit("could not parse any codes out of docs/codebook.md")
    return "\n".join(lines).strip(), codes, labels


def build_prompt(codebook, codes, rows):
    lines = []
    for r in rows:
        tag = "P" if r["speaker_role"] == "participant" else "I"
        mark = r["utt_id"] if tag == "P" else "-"
        lines.append("[%s] %s | %s" % (tag, mark, r["text"]))
    listing = "\n".join(lines)
    ids = [r["utt_id"] for r in rows if r["speaker_role"] == "participant"]
    valid = ", ".join(sorted(codes))
    return f"""You are a qualitative research coder working on a healthcare study.
Below is a continuous excerpt from an interview with a healthcare provider (HCP)
about an emergency department peer support worker program. The study team calls
these workers "peers" and the initiative "the peer program"; use that framing.

Lines marked [I] are the interviewer and are CONTEXT ONLY - do not code them.
Lines marked [P] are the participant and each has an utterance id.

CODEBOOK
{codebook}

EXCERPT
{listing}

TASK
Code every [P] line. Output a JSON array with exactly {len(ids)} objects, one per
[P] line, in the same order, each shaped:
{{"utt_id": "<the id>", "topics": ["A1","C3"], "stance": "positive|negative|mixed|neutral|na",
  "key_phrase": "<verbatim substring, <=20 words, or empty>",
  "confidence": "high|medium|low"}}

Rules:
- topics must be drawn only from these code ids: {valid}
- use [] when no code clearly applies; do not force a code
- key_phrase must appear verbatim in that utterance
- stance is toward the peer program specifically
- output ONLY the JSON array, no prose, no markdown fences

The utterance ids to code, in order: {json.dumps(ids)}"""


_THROTTLE = threading.Semaphore(1)   # serialise the actual spawn, not the wait
_LAST_CALL = [0.0]
MIN_GAP_S = 1.5

# Hitting the account's usage cap is not a transient error: retrying just burns
# wall-clock (5 retries x 100 chunks is ~100 min of pure thrash) and the cap will
# still be there. Detect it and stop the whole run at once, leaving every
# completed chunk cached so a later re-run resumes from exactly where it stopped.
# The real message is "You've hit your session limit - resets 1:30am
# (<local timezone>)". An earlier pattern list missed it because it matched
# "limit reached" but not "hit your ... limit", so the run spent the night
# retrying a cap that could not clear until 1:30am. Match the word "limit" in
# any of its phrasings, plus the usual quota wording.
LIMIT_RE = re.compile(
    r"\blimit\b|quota|too many requests|429|"
    r"insufficient credit|out of credit|upgrade your plan", re.I)


class UsageLimit(RuntimeError):
    """Raised when the CLI reports an account limit rather than a hiccup."""


_STOP = threading.Event()


def call_claude(prompt, timeout=900, tries=5):
    """Run one headless claude call, backing off when the CLI refuses.

    Sustained parallel invocation gets throttled: the CLI starts returning
    exit 1 with an empty stderr after roughly 15-20 rapid calls. A previous run
    lost 105 of 119 chunks that way. Spawns are spaced by MIN_GAP_S and each
    chunk retries with exponential backoff plus jitter.
    """
    last = ""
    for attempt in range(tries):
        if _STOP.is_set():
            raise UsageLimit("stopped: account limit reached earlier in this run")
        with _THROTTLE:
            gap = time.monotonic() - _LAST_CALL[0]
            if gap < MIN_GAP_S:
                time.sleep(MIN_GAP_S - gap)
            _LAST_CALL[0] = time.monotonic()
        p = subprocess.run(
            ["claude", "-p", "--output-format", "text", "--allowed-tools", ""],
            input=prompt, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout
        # the CLI puts some failures on stdout, so inspect both streams
        blob = ((p.stderr or "") + " " + (p.stdout or "")).strip()
        if LIMIT_RE.search(blob):
            _STOP.set()
            raise UsageLimit(blob[:300] or "account limit reported by CLI")
        last = "exit %d: %s" % (p.returncode, blob[:200])
        if attempt < tries - 1:
            time.sleep(min(60, 4 * (2 ** attempt)) + random.uniform(0, 3))
    raise RuntimeError("claude CLI failed after %d tries (%s)" % (tries, last))


def parse_json_array(raw):
    t = raw.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    a, b = t.find("["), t.rfind("]")
    if a == -1 or b == -1:
        raise ValueError("no JSON array in model output")
    return json.loads(t[a:b + 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=45,
                    help="participant utterances per LLM call")
    ap.add_argument("--limit-sessions", type=int, default=0)
    ap.add_argument("--sessions", nargs="*", default=[])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel claude CLI calls; each chunk is independent")
    args = ap.parse_args()

    src = DATA / "utterances_text.csv"
    if not src.exists():
        sys.exit("run scripts/07_text_sentiment.py first")
    sent = pd.read_csv(src, low_memory=False)
    codebook, codes, _labels = load_codebook()
    print("[init] %d codes loaded from codebook" % len(codes), flush=True)

    # Code whole turns, not sentences.
    #
    # The L1 table is sentence-sized because acoustic emotion needs short units,
    # but a theme belongs to what someone said, not to each clause of it.
    # Coding turns cuts the work from ~94 calls to ~48 and gives the coder a
    # complete answer to judge. Codes are expanded back onto the sentences of
    # each turn afterwards, so the L1 table still carries them.
    utt = (sent.groupby(["session_id", "turn_index"], as_index=False)
               .agg(utt_id=("utt_id", "first"),
                    speaker_role=("speaker_role", "first"),
                    start_s=("start_s", "min"),
                    is_backchannel=("is_backchannel", "min"),
                    text=("text", lambda s: " ".join(str(x) for x in s))))
    print("[init] %d sentences -> %d turns to code"
          % (len(sent), int(((utt.speaker_role == "participant")
                             & (utt.is_backchannel == 0)).sum())), flush=True)

    sids = sorted(utt.session_id.unique())
    if args.sessions:
        sids = [s for s in sids if s in args.sessions]
    if args.limit_sessions:
        sids = sids[:args.limit_sessions]

    # ---- plan every chunk up front, then run them concurrently -------------
    jobs, all_rows = [], []
    for sid in sids:
        g = utt[utt.session_id == sid].sort_values("start_s").reset_index(drop=True)
        codeable = g[(g.speaker_role == "participant") & (g.is_backchannel == 0)]
        targets = codeable.utt_id.tolist()
        if not targets:
            continue
        chunks = [targets[i:i + args.batch] for i in range(0, len(targets), args.batch)]
        for ci, chunk in enumerate(chunks):
            out_f = CACHE / ("%s_%02d.json" % (sid, ci))
            if out_f.exists() and not args.overwrite:
                all_rows.extend(json.loads(out_f.read_text(encoding="utf-8")))
                continue
            # include surrounding context, not just the codeable lines
            lo = g.index[g.utt_id == chunk[0]][0]
            hi = g.index[g.utt_id == chunk[-1]][0]
            window = g.loc[max(0, lo - 2):min(len(g) - 1, hi + 1)]
            rows = window[["utt_id", "speaker_role", "text"]].to_dict("records")
            for r in rows:
                if r["utt_id"] not in chunk:
                    r["speaker_role"] = "interviewer"  # context only
            jobs.append((sid, ci, out_f, set(chunk), build_prompt(codebook, codes, rows)))

    print("[plan] %d chunks cached, %d to run on %d workers"
          % (len(all_rows) and len(list(CACHE.glob('*.json'))) or 0,
             len(jobs), args.workers), flush=True)

    def run_one(job):
        sid, ci, out_f, valid_ids, prompt = job
        parsed = None
        for attempt in (1, 2):
            try:
                parsed = parse_json_array(call_claude(prompt))
                break
            except UsageLimit as e:
                return sid, ci, None, "LIMIT: %s" % str(e)[:120]
            except Exception as e:
                if attempt == 2:
                    return sid, ci, None, str(e)[:120]
                prompt += ("\n\nIMPORTANT: your previous reply was not a bare "
                           "JSON array. Output ONLY the array.")
        keep = []
        for o in parsed or []:
            if not isinstance(o, dict) or o.get("utt_id") not in valid_ids:
                continue
            topics = [t for t in (o.get("topics") or []) if t in codes]
            st = o.get("stance")
            keep.append({
                "utt_id": o["utt_id"],
                "code_ids": "|".join(topics),
                "code_names": "|".join(codes[t] for t in topics),
                "n_codes": len(topics),
                "llm_stance": st if st in VALID_STANCE else "neutral",
                "key_phrase": str(o.get("key_phrase") or "")[:200],
                "code_confidence": o.get("confidence", "medium"),
            })
        out_f.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
        return sid, ci, keep, None

    failures = 0
    if jobs:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, j): j for j in jobs}
            for fut in as_completed(futs):
                sid, ci, keep, err = fut.result()
                done += 1
                if err:
                    failures += 1
                    print("  [%d/%d] %s chunk %d FAILED: %s"
                          % (done, len(jobs), sid, ci, err), flush=True)
                else:
                    all_rows.extend(keep)
                    print("  [%d/%d] %s chunk %d: %d coded"
                          % (done, len(jobs), sid, ci, len(keep)), flush=True)

    # A previous run logged 1381 chunk failures ("claude CLI exit 1"), coded 3
    # chunks, and still exited 0 because `all_rows` was non-empty from cache --
    # so the downstream figures were silently built on two sessions. Partial
    # coding must fail loudly instead of looking like success.
    if _STOP.is_set():
        done_n = len(list(CACHE.glob("*.json")))
        sys.exit(
            "\n" + "=" * 70 +
            "\nSTOPPED: the account usage limit was reached."
            "\n  %d of the planned chunks are complete and cached - none of that work"
            "\n  is lost. Re-run the same command when the limit resets and it"
            "\n  resumes from where it stopped.\n" % done_n + "=" * 70)

    if jobs and failures:
        rate = failures / len(jobs)
        print("\n!! %d of %d chunks failed (%.0f%%)" % (failures, len(jobs), 100 * rate))
        if rate > 0.10:
            sys.exit("aborting: coding is too incomplete to build results on. "
                     "Cached chunks are kept, so re-running resumes.")

    if not all_rows:
        sys.exit("nothing coded")
    coded = pd.DataFrame(all_rows).drop_duplicates(subset="utt_id")
    # expand turn-level codes back onto every sentence of that turn
    key = utt[["session_id", "turn_index", "utt_id"]].rename(
        columns={"utt_id": "turn_utt_id"})
    coded = coded.merge(key, left_on="utt_id", right_on="turn_utt_id", how="left")
    coded = coded.drop(columns=["utt_id", "turn_utt_id"])
    merged = sent.merge(coded, on=["session_id", "turn_index"], how="left")
    merged.to_csv(DATA / "utterances_coded.csv", index=False, encoding="utf-8-sig")

    print("\nwrote data/utterances_coded.csv")
    print("  coded utterances: %d" % len(coded))

    # Coverage guard: every session must actually carry codes. Reporting theme
    # percentages over 34 participants when only 2 have codes produces the
    # tell-tale "every theme sits at 6%" (= 2/34) artefact.
    with_codes = merged[merged.code_ids.notna() & (merged.code_ids != "")]
    n_sess, n_tot = with_codes.session_id.nunique(), merged.session_id.nunique()
    n_ppl = with_codes.participant_id.nunique()
    print("  sessions with codes: %d/%d | participants with codes: %d"
          % (n_sess, n_tot, n_ppl))
    if n_sess < n_tot:
        print("  !! %d session(s) have no codes -- do NOT report theme "
              "percentages until this is resolved" % (n_tot - n_sess))
    exploded = (coded.assign(c=coded.code_ids.str.split("|"))
                     .explode("c").query("c != '' and c == c"))
    print("\nCode frequency:")
    freq = exploded.c.value_counts()
    for cid, n in freq.items():
        print("  %-4s %-26s %4d" % (cid, codes.get(cid, "?"), n))
    print("\nLLM stance distribution:")
    print(coded.llm_stance.value_counts().to_string())


if __name__ == "__main__":
    main()
