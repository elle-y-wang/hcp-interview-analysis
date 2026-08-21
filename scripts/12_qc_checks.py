"""
Framework 2.3 and 4.3 worksheets — the checks that need human ears.

The framework asks for three acoustic sanity checks and a qualitative reading of
discordant utterances. Two of the three are computable and are printed here; the
one that needs a person ("listen to 30 high-arousal utterances and confirm they
sound plausible") is exported as a worksheet with timestamps, so the listener can
jump straight to each clip.

Outputs:
  outputs/tables/qc_high_arousal_listen.csv   30 clips to listen to (2.3 item 2)
  outputs/tables/qc_discord_reading.csv       discordant utterances + context (4.3)
  outputs/tables/qc_acoustic_checks.csv       computed correlations (2.3 item 3)

Usage: python scripts/12_qc_checks.py
"""
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
TABLES = PROJECT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def hms(sec):
    sec = float(sec)
    return "%02d:%02d:%05.2f" % (int(sec // 3600), int((sec % 3600) // 60), sec % 60)


def main() -> None:
    src = DATA / "utterances_fused.csv"
    if not src.exists():
        raise SystemExit("run scripts/09_fusion.py first")
    utt = pd.read_csv(src)
    sess = pd.read_csv(DATA / "sessions.csv")[["session_id", "source_file"]]
    p = utt[(utt.speaker_role == "participant") & (utt.is_backchannel == 0)
            & utt.ac_valence.notna()].merge(sess, on="session_id", how="left")

    # --- 2.3 item 1: is the model just saying "neutral" to everything? -------
    print("=== 2.3(1) spread of the acoustic model ===")
    desc = p[["ac_valence", "ac_arousal", "ac_dominance"]].describe().round(3)
    print(desc.to_string())
    iqr = desc.loc["75%"] - desc.loc["25%"]
    print("\nIQR: %s" % iqr.round(3).to_dict())
    print("Interview speech is low-arousal by nature, so a narrow arousal range is\n"
          "expected; the framework's advice is to lean on continuous dimensions\n"
          "and trajectories rather than discrete labels, which is what we do.")

    # --- 2.3 item 3: arousal should track F0 and loudness -------------------
    rows = []
    for a, b in [("ac_arousal_z", "f0_mean_z"), ("ac_arousal_z", "loudness_mean_z"),
                 ("ac_arousal_z", "words_per_s_z"), ("ac_valence_z", "tx_valence_z"),
                 ("ac_arousal_z", "f0_p80_z"), ("ac_valence_z", "hnr_z")]:
        if a in p.columns and b in p.columns:
            sub = p.dropna(subset=[a, b])
            r = sub[a].corr(sub[b]) if len(sub) > 3 else float("nan")
            expect = "positive" if b != "hnr_z" else "either"
            ok = "OK" if (expect != "positive" or (r == r and r > 0)) else "CHECK ALIGNMENT"
            rows.append({"x": a, "y": b, "pearson_r": round(r, 3), "n": len(sub),
                         "expected": expect, "verdict": ok})
    checks = pd.DataFrame(rows)
    checks.to_csv(TABLES / "qc_acoustic_checks.csv", index=False, encoding="utf-8-sig")
    print("\n=== 2.3(3) alignment checks ===")
    print(checks.to_string(index=False))

    # --- 2.3 item 2: worksheet for the listening check -----------------------
    top = p.nlargest(30, "ac_arousal_z")[
        ["utt_id", "session_id", "source_file", "participant_id", "topic_segment",
         "start_s", "end_s", "ac_arousal_z", "ac_valence_z", "text"]].copy()
    top["timestamp"] = top.start_s.map(hms)
    top["sounds_plausible_y_n"] = ""
    top["listener_note"] = ""
    top.to_csv(TABLES / "qc_high_arousal_listen.csv", index=False, encoding="utf-8-sig")
    print("\n=== 2.3(2) listening worksheet ===")
    print("wrote outputs/tables/qc_high_arousal_listen.csv (30 clips, with timestamps)")

    # --- 4.3: discordant utterances with surrounding context -----------------
    utt_sorted = utt.sort_values(["session_id", "start_s"]).reset_index(drop=True)
    disc_idx = utt_sorted.index[utt_sorted.discord_flag == 1]
    recs = []
    for i in disc_idx:
        row = utt_sorted.loc[i]
        lo, hi = max(0, i - 2), min(len(utt_sorted) - 1, i + 2)
        ctx = utt_sorted.loc[lo:hi]
        recs.append({
            "utt_id": row.utt_id, "session_id": row.session_id,
            "participant_id": row.participant_id, "topic_segment": row.topic_segment,
            "timestamp": hms(row.start_s), "discord_type": row.discord_type,
            "tx_valence_z": round(row.tx_valence_z, 2),
            "ac_valence_z": round(row.ac_valence_z, 2),
            "text": row.text,
            "context": " || ".join("[%s] %s" % (c.speaker_role[:1].upper(), c.text)
                                   for _, c in ctx.iterrows()),
            "interpretation": "",
        })
    if recs:
        d = pd.DataFrame(recs).merge(sess, on="session_id", how="left")
        d.to_csv(TABLES / "qc_discord_reading.csv", index=False, encoding="utf-8-sig")
        print("\n=== 4.3 discordant utterances ===")
        print("wrote outputs/tables/qc_discord_reading.csv (%d utterances)" % len(d))
        print(d.discord_type.value_counts().to_string())
        print("\nFirst few, for a sense of what they look like:")
        for _, r in d.head(3).iterrows():
            print("  [%s %s] tx%+.1f ac%+.1f  %s"
                  % (r.session_id, r.timestamp, r.tx_valence_z, r.ac_valence_z,
                     r.text[:95]))
    else:
        print("\n=== 4.3 === no discordant utterances at the current threshold")


if __name__ == "__main__":
    main()
