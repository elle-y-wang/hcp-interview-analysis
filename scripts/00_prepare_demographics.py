"""
Step 1.1 — Build data/participants.csv from the study's demographic export.

Source: "Phase II HCP Demographic Data.xlsx" (34 rows, one per participant).

The export keys on "PN-001"; the pipeline keys on "PN001" (recovered from the
spoken consent statements), so the ID is normalised on the way in. The 34 rows
match the 34 participants recovered from audio exactly, which is a useful
independent check that no study ID was mis-transcribed.

Columns are renamed to the short snake_case names the framework's section 5.3
model expects (gender, occupation_category, ...), and the free-text fields the
models cannot use are dropped from the analysis table but kept in
data/private/participants_freetext.csv, since they contain participants' own
words and can be re-identifying in a small sample.

Usage: python scripts/00_prepare_demographics.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
PRIVATE = DATA / "private"
PRIVATE.mkdir(parents=True, exist_ok=True)
SRC = PROJECT / "Phase II HCP Demographic Data.xlsx"

RENAME = {
    "Study ID": "participant_id",
    "Age": "age",
    "CatAge": "age_group",
    "Sex": "sex",
    "Gender": "gender",
    "Province of Practice": "province",
    "HCP Profession - HCP Answers": "occupation",
    "Specialty - HCP Answers": "specialty",
    "Years Worked in HC - Categorical": "years_hc_group",
    "Years Worked in HC - HCP Answers": "years_hc_raw",
    "Years Worked in ED - Categorical": "years_ed_group",
    "Years Worked in ED - HCP Answers": "years_ed_raw",
    "Areas Most Commonly Worked In - HCP Answers": "areas_worked",
    "Has Worked with a PSW - HCP Answers": "worked_with_peer",
    "Number of Times HCP Worked with PSW Directly - Categorical": "times_with_peer_group",
    "Would Work with a PSW More Frequently - Categorical": "would_work_more",
}

FREETEXT = [
    "Gender (Text Box)", "City of Practice (Text Box)",
    "HCP Profession - Other - HCP Answers", "Other Area Worked (Text Box)",
    "If yes, approximately how many times has a peer support worker directly "
    "supported you in caring for a patient? - HCP Answers",
    "If you have not worked with a peer support worker, what were the reasons? "
    "- HCP Answers",
    "Would you consider working with a peer support worker more frequently? "
    "Why or why not? (1-2 sentences) - HCP Answers",
]

# Framework 5.3: a profession with fewer than 5 participants cannot support a
# group test. Raw counts are Nurse 12, Security 6, Social Worker 5, Physician 5,
# EMED RA 4, Pharmacist 2 -- so the last two are pooled into one "Pharmacy &
# research" group, which is the only merge needed to get every cell to n>=5.
# The ungrouped `occupation` column is kept for descriptive reporting.
OCC_GROUP = {
    "Nurse": "Nursing",
    "Physician": "Physician",
    "Social Worker": "Social work",
    "Security": "Security",
    "Pharmacist": "Pharmacy & research",
    "EMED RA": "Pharmacy & research",
}


def num_years(v):
    """'5', '12', 'No answer', '15+' -> float or NaN."""
    m = re.search(r"\d+", str(v))
    return float(m.group(0)) if m else np.nan


def main() -> None:
    if not SRC.exists():
        raise SystemExit("missing %s" % SRC.name)
    raw = pd.read_excel(SRC)
    print("[in] %s: %d rows, %d columns" % (SRC.name, *raw.shape))

    df = raw.rename(columns=RENAME)
    df["participant_id"] = (df.participant_id.astype(str).str.upper()
                            .str.replace(r"[^0-9]", "", regex=True)
                            .apply(lambda s: "PN%03d" % int(s) if s else None))

    df["years_hc"] = df.years_hc_raw.map(num_years)
    df["years_ed"] = df.years_ed_raw.map(num_years)
    df["occupation"] = df.occupation.astype(str).str.strip()
    df["occupation_category"] = df.occupation.map(
        lambda o: OCC_GROUP.get(o, "Pharmacy & research" if o and o != "nan" else np.nan))

    for c in ["Triage (Y/N)", "A-Side (Y/N)", "B-Side (Y/N)", "C-Side (Y/N)",
              "PEAT (Y/N)"]:
        if c in df.columns:
            df["area_" + c.split()[0].lower().replace("-", "_")] = (
                df[c].astype(str).str.upper().eq("Y").astype(int))

    # The team's preliminary analysis restricted to participants who had worked
    # directly with a peer (N=20). Flagged so results can be compared like for
    # like without rebuilding the table.
    df["direct_peer_contact"] = df.worked_with_peer.astype(str).str.strip().str.lower().eq("yes")

    keep = ["participant_id", "age", "age_group", "sex", "gender", "province",
            "occupation", "occupation_category", "specialty",
            "years_hc", "years_hc_group", "years_ed", "years_ed_group",
            "areas_worked", "worked_with_peer", "times_with_peer_group",
            "would_work_more", "direct_peer_contact"]
    keep += [c for c in df.columns if c.startswith("area_")]
    out = df[[c for c in keep if c in df.columns]].copy()
    out.to_csv(DATA / "participants.csv", index=False, encoding="utf-8-sig")

    ft = ["participant_id"] + [c for c in FREETEXT if c in raw.columns]
    df[ft].to_csv(PRIVATE / "participants_freetext.csv", index=False,
                  encoding="utf-8-sig")

    print("\nwrote data/participants.csv (%d rows)" % len(out))
    print("wrote data/private/participants_freetext.csv (free-text, not for sharing)")

    print("\n--- checks against the team's Table 1 ---")
    print("age median %.1f (Q1-Q3 %.0f-%.0f), range %d-%d"
          % (out.age.median(), out.age.quantile(.25), out.age.quantile(.75),
             out.age.min(), out.age.max()))
    print("sex: %s" % out.sex.value_counts().to_dict())
    print("direct peer contact: %d of %d" % (out.direct_peer_contact.sum(), len(out)))
    print("\noccupation:")
    print(out.occupation.value_counts().to_string())
    print("\noccupation_category (used in models):")
    print(out.occupation_category.value_counts().to_string())

    # does the demographic file line up with the IDs recovered from audio?
    upath = DATA / "utterances.csv"
    if upath.exists():
        u = pd.read_csv(upath)
        audio_ids = set(u[u.speaker_role == "participant"].participant_id.unique())
        demo_ids = set(out.participant_id)
        print("\n--- ID reconciliation with audio ---")
        print("in audio only: %s" % (sorted(audio_ids - demo_ids) or "none"))
        print("in demographics only: %s" % (sorted(demo_ids - audio_ids) or "none"))
        print("matched: %d" % len(audio_ids & demo_ids))


if __name__ == "__main__":
    main()
