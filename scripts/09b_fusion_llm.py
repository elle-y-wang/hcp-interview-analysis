"""
Step 4.2 — Scheme B fusion: acoustic cues supplied to an LLM as context.

The framework calls this the recommended primary scheme because it can handle
"positive words, flat voice" cases that a linear weighting cannot, and because it
emits a reason, which suits a qualitative-primary write-up.

Scope. Running it over every participant utterance would mean thousands of LLM
calls for a study whose quantitative layer is explicitly auxiliary. It is
therefore run on the utterances where scheme A cannot speak:
  * every discordant utterance (framework 4.3 - the qualitative payoff), and
  * a stratified comparison sample, so scheme A and scheme B can be compared on
    common ground.
Use --all to run the whole corpus instead.

Deviation from the framework's prompt: it names a discrete acoustic label
(ac_label) from emotion2vec. Our English model emits continuous dimensions, so
the prompt reports those as within-speaker SD deviations, which is strictly more
informative and matches how the rest of the pipeline is normalised.

Reproducibility: framework 4.2 asks for temperature=0 and a repeat-consistency
check. The `claude` CLI does not expose temperature, so instead --repeat re-runs
a subsample and reports the agreement rate, which is what the framework actually
wants to know.

Usage:
  python scripts/09b_fusion_llm.py --sample 250
  python scripts/09b_fusion_llm.py --all
  python scripts/09b_fusion_llm.py --repeat 60      # consistency check only
"""
import argparse
import json
import re
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
TABLES = PROJECT / "outputs" / "tables"
# Scratch space for models, converted audio and intermediate JSON. Kept
# outside the project folder so cloud-sync clients do not churn on it.
# Override with the HCP_WORK environment variable.
WORK = Path(os.environ.get("HCP_WORK") or (Path.home() / "hcp_work"))
CACHE = WORK / "fusion_llm"
CACHE.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def txt(v):
    """Empty string for NaN; pandas NaN is truthy so `or ""` does not catch it."""
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def fmt(v, digits=1):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else ("%+.*f" % (digits, v))


def build_prompt(items):
    blocks = []
    for it in items:
        blocks.append(
            "UTTERANCE {uid}\n"
            "  text: \"{text}\"\n"
            "  acoustic model (relative to this speaker's own average, in SD):\n"
            "    valence {v} SD, arousal {a} SD, dominance {d} SD\n"
            "  voice measurements (same speaker-relative scale):\n"
            "    pitch {f0} SD, loudness {ld} SD, speech rate {sr} SD, "
            "pause length {pz} SD\n"
            "  preceding context: \"{prev}\"\n"
            "  following context: \"{nxt}\"".format(
                uid=it["utt_id"], text=txt(it["text"])[:600],
                v=fmt(it.get("ac_valence_z")), a=fmt(it.get("ac_arousal_z")),
                d=fmt(it.get("ac_dominance_z")), f0=fmt(it.get("f0_mean_z")),
                ld=fmt(it.get("loudness_mean_z")), sr=fmt(it.get("words_per_s_z")),
                pz=fmt(it.get("unvoiced_seg_len_z")),
                prev=txt(it.get("prev"))[:220], nxt=txt(it.get("next"))[:220])
        )
    ids = [it["utt_id"] for it in items]
    return """You are analysing how a healthcare provider sounds and what they say during a
research interview about an emergency department peer support worker program.

For each utterance below you are given the words AND objective measurements of
how the speaker's voice differed from their OWN average at that moment. Judge
the speaker's emotional state by combining both channels. A calm voice saying
positive words is not the same as a flat, quiet voice saying positive words.

{blocks}

Output a JSON array of exactly {n} objects, one per utterance, in the same order:
{{"utt_id": "<id>",
  "emotion": "<one word: neutral|engaged|enthusiastic|thoughtful|frustrated|concerned|sad|uncomfortable|amused>",
  "valence": <integer -2..2>,
  "arousal": <integer 0..2>,
  "text_acoustic_agree": <true|false>,
  "reasoning": "<one short sentence>"}}

Set text_acoustic_agree to false when the wording and the voice point in
different directions. Output ONLY the JSON array, no prose, no markdown fences.

ids in order: {ids}""".format(blocks="\n\n".join(blocks), n=len(items),
                              ids=json.dumps(ids))


def call_claude(prompt, timeout=900):
    p = subprocess.run(
        ["claude", "-p", "--output-format", "text", "--allowed-tools", ""],
        input=prompt, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("claude CLI exit %d: %s" % (p.returncode, p.stderr[:300]))
    return p.stdout


def parse_array(raw):
    t = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    t = re.sub(r"\s*```$", "", t).strip()
    a, b = t.find("["), t.rfind("]")
    if a == -1 or b == -1:
        raise ValueError("no JSON array")
    return json.loads(t[a:b + 1])


def run_batches(items, batch, tag, overwrite=False):
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        f = CACHE / ("%s_%03d.json" % (tag, i // batch))
        if f.exists() and not overwrite:
            out.extend(json.loads(f.read_text(encoding="utf-8")))
            print("  batch %d cached" % (i // batch), flush=True)
            continue
        prompt = build_prompt(chunk)
        parsed, valid = None, {c["utt_id"] for c in chunk}
        for attempt in (1, 2):
            try:
                parsed = parse_array(call_claude(prompt))
                break
            except Exception as e:
                print("  batch %d attempt %d failed: %s"
                      % (i // batch, attempt, str(e)[:110]), flush=True)
                prompt += "\n\nIMPORTANT: reply with ONLY the bare JSON array."
        if parsed is None:
            continue
        keep = []
        for o in parsed:
            if not isinstance(o, dict) or o.get("utt_id") not in valid:
                continue
            try:
                keep.append({
                    "utt_id": o["utt_id"],
                    "llm_emotion": str(o.get("emotion", ""))[:32],
                    "llm_valence": int(np.clip(int(o.get("valence", 0)), -2, 2)),
                    "llm_arousal": int(np.clip(int(o.get("arousal", 0)), 0, 2)),
                    "llm_agree": bool(o.get("text_acoustic_agree", True)),
                    "llm_reasoning": str(o.get("reasoning", ""))[:300],
                })
            except (TypeError, ValueError):
                continue
        f.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
        out.extend(keep)
        print("  batch %d: %d/%d" % (i // batch, len(keep), len(chunk)), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = DATA / "utterances_fused.csv"
    if not src.exists():
        sys.exit("run scripts/09_fusion.py first")
    utt = pd.read_csv(src)
    p = utt[(utt.speaker_role == "participant") & (utt.is_backchannel == 0)
            & utt.fused_valence.notna()].copy()
    p = p.sort_values(["session_id", "start_s"])
    p["prev"] = p.groupby("session_id").text.shift(1)
    p["next"] = p.groupby("session_id").text.shift(-1)

    if args.all:
        target = p
    else:
        disc = p[p.discord_flag == 1]
        rest = p[p.discord_flag == 0]
        n = max(args.sample - len(disc), 0)
        if n and len(rest):
            rest["vbin"] = pd.qcut(rest.fused_valence, 3, labels=False, duplicates="drop")
            per = max(1, n // max(rest.groupby(["topic_segment", "vbin"],
                                               observed=True).ngroups, 1))
            rest = (rest.groupby(["topic_segment", "vbin"], observed=True,
                                 group_keys=False)
                        .apply(lambda g: g.sample(min(len(g), per), random_state=11)))
        target = pd.concat([disc, rest]).drop_duplicates(subset="utt_id")
    print("[run] scheme B on %d utterances (%d discordant)"
          % (len(target), int(target.discord_flag.sum())), flush=True)

    cols = ["utt_id", "text", "prev", "next", "ac_valence_z", "ac_arousal_z",
            "ac_dominance_z", "f0_mean_z", "loudness_mean_z", "words_per_s_z",
            "unvoiced_seg_len_z"]
    items = target[[c for c in cols if c in target.columns]].to_dict("records")
    rows = run_batches(items, args.batch, "main", args.overwrite)
    if not rows:
        sys.exit("scheme B produced nothing")

    res = pd.DataFrame(rows).drop_duplicates(subset="utt_id")
    merged = utt.merge(res, on="utt_id", how="left")
    merged.to_csv(DATA / "utterances_fused_llm.csv", index=False, encoding="utf-8-sig")
    print("\nwrote data/utterances_fused_llm.csv (%d judged)" % len(res))

    j = merged[merged.llm_valence.notna()]
    print("\nScheme A vs scheme B, over the %d judged utterances:" % len(j))
    print("  corr(fused_valence, llm_valence) = %+.3f"
          % j.fused_valence.corr(j.llm_valence))
    print("  corr(fused_arousal, llm_arousal) = %+.3f"
          % j.fused_arousal.corr(j.llm_arousal))
    print("\nLLM emotion labels:")
    print(j.llm_emotion.value_counts().head(10).to_string())
    agree_flag = j.llm_agree.astype(str).str.lower().isin(["true", "1"])
    print("\ntext/acoustic agreement per the LLM: %d of %d disagree"
          % (int((~agree_flag).sum()), len(j)))
    ct = pd.crosstab(j.discord_flag, ~agree_flag)
    print("\nscheme A discord_flag (rows) vs scheme B disagreement (cols):")
    print(ct.to_string())

    if args.repeat:
        sub = j.sample(min(args.repeat, len(j)), random_state=3)
        print("\n[consistency] re-running %d utterances" % len(sub), flush=True)
        again = pd.DataFrame(run_batches(
            target[target.utt_id.isin(sub.utt_id)][
                [c for c in cols if c in target.columns]].to_dict("records"),
            args.batch, "repeat", overwrite=True))
        m = sub.merge(again, on="utt_id", suffixes=("_1", "_2"))
        if len(m):
            print("  valence exact match: %.1f%%"
                  % (100 * (m.llm_valence_1 == m.llm_valence_2).mean()))
            print("  valence within 1:    %.1f%%"
                  % (100 * ((m.llm_valence_1 - m.llm_valence_2).abs() <= 1).mean()))
            print("  agree-flag match:    %.1f%%"
                  % (100 * (m.llm_agree_1 == m.llm_agree_2).mean()))
            m[["utt_id", "llm_valence_1", "llm_valence_2", "llm_arousal_1",
               "llm_arousal_2"]].to_csv(TABLES / "scheme_b_consistency.csv",
                                        index=False, encoding="utf-8-sig")
            print("  wrote outputs/tables/scheme_b_consistency.csv")


if __name__ == "__main__":
    main()
