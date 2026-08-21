"""
Step 4 — Fusing the acoustic and text channels at utterance level.

Implements framework 4.1 (scheme A, weighted late fusion) plus 4.3 (discordant
utterance detection) and prepares the 4.4 validation sample. Scheme B (feeding
acoustic cues to an LLM) lives in scripts/09b_fusion_llm.py.

Sample size does not support training a fusion model, so weights are fixed and a
sensitivity analysis over w_a is reported, exactly as 4.1 requires.

Note on scales: ac_valence comes from the MSP-dim model on 0..1, tx_valence from
a 3-class sentiment head on -1..1. They are only ever combined after
within-speaker z-scoring, so the raw scales never interact.

Outputs:
  data/utterances_fused.csv     L1 table with fused_valence / fused_arousal / discord
  data/annotation_sample.csv    stratified sample for the human validation in 4.4
  outputs/tables/fusion_sensitivity.csv

Usage: python scripts/09_fusion.py [--wa 0.5] [--sample 400]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
TABLES = PROJECT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wa", type=float, default=0.5, help="acoustic weight")
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--discord-thresh", type=float, default=0.5)
    args = ap.parse_args()

    src = DATA / "utterances_coded.csv"
    if not src.exists():
        src = DATA / "utterances_text.csv"
    utt = pd.read_csv(src)
    print("[in] %s (%d rows)" % (src.name, len(utt)))

    wa, wt = args.wa, 1.0 - args.wa

    # Fusion is only meaningful for real participant speech.
    mask = (utt.speaker_role == "participant") & (utt.is_backchannel == 0) \
        & utt.ac_valence_z.notna() & utt.tx_valence_z.notna()

    utt["fused_valence"] = np.where(
        mask, wa * utt.ac_valence_z + wt * utt.tx_valence_z, np.nan)
    # framework 4.1: arousal has essentially no text channel, so it stays acoustic
    utt["fused_arousal"] = np.where(mask, utt.ac_arousal_z, np.nan)

    # framework 4.3: opposite signs, both clearly non-trivial
    utt["discord_flag"] = (
        mask
        & (np.sign(utt.tx_valence_z) != np.sign(utt.ac_valence_z))
        & (utt.tx_valence_z.abs() > args.discord_thresh)
        & (utt.ac_valence_z.abs() > args.discord_thresh)
    ).astype(int)
    # direction is what makes a discord interesting to read
    utt["discord_type"] = np.where(
        utt.discord_flag == 0, "",
        np.where(utt.tx_valence_z > 0, "positive_words_flat_voice",
                 "negative_words_bright_voice"))

    utt.to_csv(DATA / "utterances_fused.csv", index=False, encoding="utf-8-sig")

    sub = utt[mask]
    print("\nfusion applied to %d participant utterances" % len(sub))
    print("  corr(ac_valence_z, tx_valence_z) = %+.3f"
          % sub.ac_valence_z.corr(sub.tx_valence_z))
    n_d = int(utt.discord_flag.sum())
    print("  discordant utterances: %d (%.1f%% of fused)"
          % (n_d, 100 * n_d / max(len(sub), 1)))
    if n_d:
        print(utt.loc[utt.discord_flag == 1, "discord_type"]
              .value_counts().to_string())

    # --- 4.1 sensitivity analysis -------------------------------------------
    # intro_consent is left out of the ranking for the same reason it is left
    # out of the models and figures: it is scripted consent recitation, not an
    # answer, and reading aloud scores far higher than spontaneous speech. Left
    # in, it won "most positive topic" at every weight and said nothing.
    rank = sub[sub.topic_segment != "intro_consent"]
    rows = []
    for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        f = w * sub.ac_valence_z + (1 - w) * sub.tx_valence_z
        fr = w * rank.ac_valence_z + (1 - w) * rank.tx_valence_z
        by_topic = fr.groupby(rank.topic_segment).mean()
        rows.append({
            "w_acoustic": w,
            "corr_with_default": f.corr(0.5 * sub.ac_valence_z + 0.5 * sub.tx_valence_z),
            "most_negative_topic": by_topic.idxmin() if len(by_topic) else None,
            "most_positive_topic": by_topic.idxmax() if len(by_topic) else None,
            **{("mean_%s" % k): round(v, 3) for k, v in by_topic.items()},
        })
    sens = pd.DataFrame(rows).round(3)
    sens.to_csv(TABLES / "fusion_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\nSensitivity of topic ranking to the acoustic weight:")
    keep = ["w_acoustic", "most_negative_topic", "most_positive_topic"]
    print(sens[keep].to_string(index=False))

    # --- 4.4 stratified validation sample -----------------------------------
    # Framework asks for 300-500 covering participants, topics and emotion range,
    # plus deliberate over-sampling of discordant cases.
    pool = sub.copy()
    pool["val_bin"] = pd.qcut(pool.fused_valence, 3,
                              labels=["low", "mid", "high"], duplicates="drop")
    strata = ["session_id", "topic_segment", "val_bin"]
    n_strata = pool.groupby(strata, observed=True).ngroups
    per = max(1, args.sample // max(n_strata, 1))
    samp = (pool.groupby(strata, observed=True, group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per), random_state=7)))
    disc = utt[utt.discord_flag == 1]
    samp = pd.concat([samp, disc]).drop_duplicates(subset="utt_id")
    if len(samp) > args.sample:
        forced = samp[samp.discord_flag == 1]
        rest = samp[samp.discord_flag == 0].sample(
            max(args.sample - len(forced), 0), random_state=7)
        samp = pd.concat([forced, rest])

    cols = ["utt_id", "session_id", "participant_id", "topic_segment",
            "start_s", "end_s", "duration_s", "text", "discord_flag", "discord_type"]
    out = samp[cols].sort_values(["session_id", "start_s"]).copy()
    # blank columns for the two human annotators required by 4.4
    for c in ["rater1_valence", "rater1_arousal", "rater1_emotion",
              "rater2_valence", "rater2_arousal", "rater2_emotion", "notes"]:
        out[c] = ""
    out.to_csv(DATA / "annotation_sample.csv", index=False, encoding="utf-8-sig")

    print("\nwrote data/utterances_fused.csv")
    print("wrote data/annotation_sample.csv (%d utterances, %d discordant)"
          % (len(out), int(out.discord_flag.sum())))
    print("wrote outputs/tables/fusion_sensitivity.csv")
    print("\nNOTE: 4.4 needs two humans to fill the rater columns while listening "
          "to the audio. Until then, no model-vs-human agreement can be reported.")


if __name__ == "__main__":
    main()
