"""
Step 5 — Aggregate to L2 / L3 and fit the mixed-effects models.

Builds topic_level.csv and participant_level.csv per framework 5.1, then fits the
5.3 model with a participant random intercept.

Demographics: the framework's model is
    fused_valence ~ topic_segment * gender + occupation_category + C(interviewer_id)
If data/participants.csv exists it is merged in and the full model is fitted.
It does not exist yet for this dataset, so the reduced model
    fused_valence ~ topic_segment + recording_device
is fitted instead and RQ3 is left explicitly unanswered rather than guessed at.
Re-run this script once demographics are supplied; nothing else changes.

Per framework 5.3: random intercept only (30 groups cannot support random
slopes), ICC reported, and any interaction term is flagged exploratory.

Usage: python scripts/10_aggregate_models.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
TABLES = PROJECT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

TOPIC_ORDER = ["experiences", "services", "challenges", "sustainability", "closing"]

# `intro_consent` is excluded from every topic comparison. It is not a response
# to an interview question: it holds the scripted verbal consent recitation and
# setup chatter ("let me see if I can find it"), and reading aloud is slower and
# more articulated than spontaneous speech (2.59 vs 3.09 words/s). Left in, it
# scored +0.59 SD on tone -- far above any real section -- and, being first
# alphabetically and in guide order, it silently became the model's reference
# level, so every topic coefficient was really "quieter than reading a script".
EXCLUDE_TOPICS = ["intro_consent"]


def code_columns(df):
    """Long form of the multi-label code column."""
    if "code_ids" not in df.columns:
        return pd.DataFrame(columns=["utt_id", "participant_id", "topic_segment", "code"])
    d = df[df.code_ids.notna() & (df.code_ids != "")].copy()
    d["code"] = d.code_ids.str.split("|")
    return d.explode("code")[["utt_id", "participant_id", "topic_segment", "code"]]


def main() -> None:
    utt = pd.read_csv(DATA / "utterances_fused.csv")
    p = utt[(utt.speaker_role == "participant") & (utt.is_backchannel == 0)].copy()
    p = p[p.fused_valence.notna()]
    n_all = len(p)
    p = p[~p.topic_segment.isin(EXCLUDE_TOPICS)]
    print("[in] %d analysable participant utterances (%d dropped as %s), "
          "%d participants"
          % (len(p), n_all - len(p), "/".join(EXCLUDE_TOPICS),
             p.participant_id.nunique()))

    demo_f = DATA / "participants.csv"
    demo = pd.read_csv(demo_f) if demo_f.exists() else None
    if demo is not None:
        print("[in] participants.csv found: %d rows, columns %s"
              % (len(demo), list(demo.columns)))
        p = p.merge(demo, on="participant_id", how="left")
    else:
        print("[in] no data/participants.csv -> RQ3 (demographics) not answered")

    # ---------- L2: topic level ------------------------------------------
    codes_long = code_columns(p)
    top_codes = (codes_long.groupby(["participant_id", "topic_segment"])["code"]
                 .agg(lambda s: "|".join(pd.Series(s).value_counts().head(3).index))
                 .rename("dominant_codes"))

    topic = (p.groupby(["participant_id", "session_id", "topic_segment"])
               .agg(n_utt=("utt_id", "size"),
                    talk_time_s=("duration_s", "sum"),
                    mean_valence=("fused_valence", "mean"),
                    sd_valence=("fused_valence", "std"),
                    mean_arousal=("fused_arousal", "mean"),
                    sd_arousal=("fused_arousal", "std"),
                    mean_tx_valence=("tx_valence", "mean"),
                    prop_negative=("llm_stance", lambda s: float((s == "negative").mean())
                                   if s.notna().any() else np.nan),
                    prop_discord=("discord_flag", "mean"))
               .round(4).reset_index()
               .join(top_codes, on=["participant_id", "topic_segment"]))
    topic.to_csv(DATA / "topic_level.csv", index=False, encoding="utf-8-sig")

    # ---------- L3: participant level ------------------------------------
    def most_emotional(g):
        s = g.groupby("topic_segment").fused_arousal.mean()
        return s.idxmax() if len(s) else None

    part = (p.groupby(["participant_id", "session_id"])
              .agg(n_utt=("utt_id", "size"),
                   talk_time_min=("duration_s", lambda s: round(s.sum() / 60, 2)),
                   overall_mean_valence=("fused_valence", "mean"),
                   overall_mean_arousal=("fused_arousal", "mean"),
                   arousal_range=("fused_arousal", lambda s: s.quantile(.9) - s.quantile(.1)),
                   valence_sd=("fused_valence", "std"),
                   prop_discord=("discord_flag", "mean"),
                   n_topics=("topic_segment", "nunique"))
              .round(4).reset_index())
    part["most_emotional_topic"] = [most_emotional(g) for _, g in
                                    p.groupby(["participant_id", "session_id"])]

    # ------------------------------------------------------------------
    # Between-person measures that survive within-speaker z-scoring.
    #
    # framework 2.1 centres every feature on the speaker's own mean, so
    # `overall_mean_valence` is ~0 for everybody by construction (SD across the
    # 34 participants is 0.02 against a within-person SD of 0.84, and the mixed
    # model returns ICC = 0). Comparing groups on it -- which is what 5.3's
    # model literally asks for -- is comparing noise.
    #
    # What does carry between-person signal, and is used for RQ3 instead:
    #   * within-person topic contrasts (how far a person moves off their own
    #     baseline when the subject changes) -- SD 0.30 across participants
    #   * expressiveness (valence_sd, arousal_range)
    #   * prop_discord, talk time, and theme prevalence, none of which are
    #     centred per speaker
    wide = (p.groupby(["participant_id", "topic_segment"]).fused_valence.mean()
             .unstack("topic_segment"))
    base = "experiences"
    if base in wide.columns:
        for t in ["services", "challenges", "sustainability"]:
            if t in wide.columns:
                part["shift_%s_vs_%s" % (t, base)] = (
                    part.participant_id.map(wide[t] - wide[base]).round(4))
    aro = (p.groupby(["participant_id", "topic_segment"]).fused_arousal.mean()
            .unstack("topic_segment"))
    if base in aro.columns and "challenges" in aro.columns:
        part["arousal_shift_challenges"] = part.participant_id.map(
            aro["challenges"] - aro[base]).round(4)
    n_codes = (codes_long.groupby("participant_id")["code"].nunique()
               .rename("n_distinct_codes"))
    part = part.join(n_codes, on="participant_id")
    sess = pd.read_csv(DATA / "sessions.csv")[["session_id", "recording_device",
                                               "interview_date", "duration_min"]]
    part = part.merge(sess, on="session_id", how="left")
    if demo is not None:
        part = part.merge(demo, on="participant_id", how="left")
    part.to_csv(DATA / "participant_level.csv", index=False, encoding="utf-8-sig")

    # One test per contrast, on the 34 person-level differences rather than on
    # 3,966 sentences. This is the honest unit for "did the topic move people":
    # each participant contributes exactly one number, so the framework's
    # warning about treating utterances as independent (framework 8) does not
    # apply, and it is the comparison the within-person centring was built for.
    shifts = [c for c in part.columns if c.startswith("shift_")]
    if shifts:
        rows = []
        for c in shifts:
            v = part[c].dropna()
            if len(v) < 5:
                continue
            t_stat, pv = stats.ttest_1samp(v, 0.0)
            rows.append({"contrast": c.replace("shift_", "").replace("_vs_", " vs "),
                         "n_people": len(v), "mean_shift": round(v.mean(), 3),
                         "sd": round(v.std(), 3), "t": round(float(t_stat), 2),
                         "p": round(float(pv), 4),
                         "n_moved_down": int((v < 0).sum())})
        if rows:
            cd = pd.DataFrame(rows)
            cd.to_csv(TABLES / "topic_shift_tests.csv", index=False,
                      encoding="utf-8-sig")
            print("\nWithin-person topic shifts (one value per participant, "
                  "vs their own 'experiences' baseline):")
            print(cd.to_string(index=False))
            print("wrote outputs/tables/topic_shift_tests.csv")

    print("\nwrote data/topic_level.csv (%d rows), data/participant_level.csv (%d rows)"
          % (len(topic), len(part)))

    # ---------- code frequency (RQ1) --------------------------------------
    if len(codes_long):
        freq = (codes_long.groupby("code")
                .agg(n_utterances=("utt_id", "size"),
                     n_participants=("participant_id", "nunique"))
                .sort_values("n_utterances", ascending=False))
        freq["pct_participants"] = (100 * freq.n_participants
                                    / p.participant_id.nunique()).round(1)
        freq.to_csv(TABLES / "code_frequency.csv", encoding="utf-8-sig")
        print("\nTop codes (RQ1):")
        print(freq.head(12).to_string())

    # ---------- topic x emotion (RQ2) -------------------------------------
    order = [t for t in TOPIC_ORDER if t in set(p.topic_segment)]
    heat = (p.groupby("topic_segment")
              .agg(n_utt=("utt_id", "size"),
                   n_participants=("participant_id", "nunique"),
                   mean_valence=("fused_valence", "mean"),
                   mean_arousal=("fused_arousal", "mean"),
                   mean_tx_valence=("tx_valence", "mean"),
                   prop_discord=("discord_flag", "mean"))
              .reindex(order).round(3))
    heat.to_csv(TABLES / "topic_emotion_summary.csv", encoding="utf-8-sig")
    print("\nTopic x emotion (RQ2):")
    print(heat.to_string())

    # ---------- 5.3 mixed effects -----------------------------------------
    if p.participant_id.nunique() < 5:
        print("\n[skip] mixed model needs more participants (have %d)"
              % p.participant_id.nunique())
        return

    md = p.copy()
    md["topic_segment"] = pd.Categorical(md.topic_segment, categories=order)
    have_demo = demo is not None and "gender" in md.columns

    # Two models, because one formula cannot answer both questions here.
    #
    # Framework 5.3 asks for topic, gender, profession and device in a single
    # mixed model on the fused (within-speaker z-scored) score. But z-scoring
    # sets every person's own mean to 0, so any predictor that is *constant
    # within a person* -- profession, device, gender's main effect -- has
    # almost no variance left to explain (between-person SD of the person
    # means is 0.07 for valence, 0.13 for arousal, against a within-person SD
    # of 0.87). Those coefficients are not identified: fitting them anyway
    # gave the arousal model exact-zero estimates with a 95% CI of
    # +/-1e7, i.e. a degenerate fit reported as if it were a result.
    #
    # So:
    #   WITHIN  - on the fused z-scores, where the framework's centring makes
    #             topic contrasts clean. Carries topic and topic x gender (a
    #             within x between interaction, which *is* identified: it asks
    #             whether men and women shift by different amounts).
    #   BETWEEN - on the raw, uncentred acoustic/text scores, which is the only
    #             place person-level differences still exist. Carries gender,
    #             profession and device.
    # Every coefficient is tagged with the model it came from so nothing gets
    # read against the wrong denominator.
    RAW = {"fused_valence": "ac_valence", "fused_arousal": "ac_arousal"}

    specs = [("within", "%s ~ C(topic_segment)" +
              (" * C(gender)" if have_demo else ""), None)]
    if have_demo:
        specs.append(("between",
                      "%s ~ C(gender) + C(occupation_category) "
                      "+ C(recording_device)", RAW))

    results = []
    for dv in ["fused_valence", "fused_arousal"]:
        for scope, tmpl, remap in specs:
            y = dv if remap is None else remap.get(dv)
            if y is None or y not in md.columns:
                continue
            f = tmpl % y
            sub = md[md[y].notna()]
            try:
                m = smf.mixedlm(f, data=sub, groups=sub["participant_id"]).fit()
            except Exception as e:
                print("\n[warn] %s %s model failed: %s" % (dv, scope, e))
                continue
            # ICC = between-participant variance / total
            vg = float(m.cov_re.iloc[0, 0])
            icc = vg / (vg + float(m.scale))
            ci = m.conf_int()
            rows, bad = [], 0
            for name in m.params.index:
                if name in ("Group Var", "Intercept"):
                    continue
                lo, hi = float(ci.loc[name, 0]), float(ci.loc[name, 1])
                # a degenerate term: no estimate, or an interval so wide it
                # carries no information. Dropped rather than plotted.
                if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo > 10:
                    bad += 1
                    continue
                rows.append({"dv": y, "scope": scope, "term": name,
                             "estimate": round(float(m.params[name]), 4),
                             "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                             "p": round(float(m.pvalues[name]), 4),
                             "icc": round(icc, 3), "formula": f})
            results.extend(rows)
            print("\n=== mixed model [%s]: %s ===" % (scope, y))
            print("  ICC (participant) = %.3f  | n=%d, groups=%d%s"
                  % (icc, m.nobs, sub.participant_id.nunique(),
                     "  [%d degenerate term(s) dropped]" % bad if bad else ""))
            print(m.summary().tables[1].to_string())

    if results:
        pd.DataFrame(results).to_csv(TABLES / "mixed_model_coefficients.csv",
                                     index=False, encoding="utf-8-sig")
        print("\nwrote outputs/tables/mixed_model_coefficients.csv")
    if not have_demo:
        print("\nRQ3 REMAINS OPEN: supply data/participants.csv keyed on "
              "participant_id (PN###) and re-run this script.")


if __name__ == "__main__":
    main()
