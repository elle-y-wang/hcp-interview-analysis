"""
Compare this pipeline's theme frequencies against the team's manual Table 4.

`docs/table4_reference.csv` maps every Table 4 row onto the codebook v2 code that
subsumes it. Several Table 4 rows collapse into one code here (Table 4 has ~100
codes with a long single-mention tail), so the fair comparison for a code is
against the **highest** of the Table 4 rows feeding it: if 85% of the team's
sample mentioned "builds rapport", then a code covering rapport plus "more
willing to speak to staff" should reach at least that.

The two are not the same denominator and are not expected to match exactly:
  * Table 4 covers 20 of 34 participants (17 transcripts, limited by time);
  * this covers all 34 across all 30 recordings.
So the useful signals are rank agreement and gross discrepancies -- a code far
below its Table 4 floor suggests the automated coder is missing something a
human saw.

Outputs: outputs/tables/table4_comparison.csv

Usage: python scripts/13_compare_table4.py
"""
import re
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
TABLES = PROJECT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def codebook_labels():
    text = (PROJECT / "docs" / "codebook.md").read_text(encoding="utf-8")
    return {m.group(1): m.group(3).strip() for m in re.finditer(
        r"^###\s+(\w\d{1,2})\s+`([a-z_]+)`\s*[—-]\s*(.+)$", text, re.M)}


def main() -> None:
    src = DATA / "utterances_coded.csv"
    if not src.exists():
        raise SystemExit("run scripts/08_llm_coding.py first")
    u = pd.read_csv(src, low_memory=False)
    p = u[(u.speaker_role == "participant") & (u.is_backchannel == 0)]
    n_people = p.participant_id.nunique()

    d = p[p.code_ids.notna() & (p.code_ids != "")].copy()
    if d.empty:
        raise SystemExit("no codes present yet")
    d["code"] = d.code_ids.str.split("|")
    d = d.explode("code")
    mine = (d.groupby("code").participant_id.nunique()
             .rename("people").to_frame())
    mine["pct_of_%d" % n_people] = (100 * mine.people / n_people).round(1)

    ref = pd.read_csv(PROJECT / "docs" / "table4_reference.csv")
    floor = (ref.groupby("code_id").table4_pct_of_20.max()
                .rename("table4_max_pct"))
    top = (ref.sort_values("table4_pct_of_20", ascending=False)
              .groupby("code_id").table4_label.first().rename("table4_top_row"))

    labels = codebook_labels()
    out = mine.join(floor).join(top)
    out["label"] = [labels.get(c, c) for c in out.index]
    out["gap_vs_table4"] = (out["pct_of_%d" % n_people] - out.table4_max_pct).round(1)
    out = out[["label", "people", "pct_of_%d" % n_people,
               "table4_max_pct", "table4_top_row", "gap_vs_table4"]]
    out = out.sort_values("pct_of_%d" % n_people, ascending=False)
    out.to_csv(TABLES / "table4_comparison.csv", encoding="utf-8-sig")

    print("Theme prevalence: this pipeline (n=%d) vs manual Table 4 (n=20)\n" % n_people)
    print(out.drop(columns="table4_top_row").to_string())

    miss = out[out.gap_vs_table4 < -25].dropna(subset=["gap_vs_table4"])
    if len(miss):
        print("\nCodes well below their Table 4 level -- worth checking whether "
              "the automated coder is under-applying them:")
        for c, r in miss.iterrows():
            print("  %-4s %-42s %.0f%% here vs %.0f%% in Table 4"
                  % (c, r.label[:42], r["pct_of_%d" % n_people], r.table4_max_pct))
    absent = sorted(set(ref.code_id) - set(mine.index))
    if absent:
        print("\nIn Table 4 but never applied here: %s" % ", ".join(absent))
    print("\nwrote outputs/tables/table4_comparison.csv")


if __name__ == "__main__":
    main()
