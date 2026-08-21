"""
Step 6.2 — Figures, written for a reader with no background in the project.

Design rules applied throughout:
  * No jargon on the page. "valence" -> "tone", "arousal" -> "energy",
    "within-speaker z-score" -> "compared with how that person usually sounds",
    "discordant utterance" -> "words and voice disagreed". Code ids like B4
    never appear; the codebook's plain-English label is used instead.
  * The title states the finding, not the method. The subtitle says how to read
    the chart. A footnote states what the numbers are counted over.
  * Percentages are of *people*, not utterances, wherever a person-level claim
    is made -- that is what a reader assumes a percentage means, and it matches
    how the team's own Table 4 reports.

Figures:
  fig1  what providers talked about              (RQ1)
  fig2  overall stance toward the programme      (RQ1, headline)
  fig3  how each section of the interview felt   (RQ2)
  fig4  emotional arc through the interview      (RQ2)
  fig5  individual examples
  fig6  where words and voice disagreed          (4.3)
  fig7  differences by profession                (RQ3)
  fig8  what predicts tone (model)               (5.3)

Usage: python scripts/11_figures.py
"""
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
FIGS = PROJECT / "outputs" / "figures"
TABLES = PROJECT / "outputs" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)

# intro_consent is excluded everywhere: it holds the scripted verbal-consent
# recitation and setup chatter rather than an answer to any question, and
# reading aloud is slower and more articulated than spontaneous speech, which
# made it score +0.59 SD on tone -- far above any real section. See the note in
# scripts/10_aggregate_models.py.
EXCLUDE_TOPICS = ["intro_consent"]

TOPIC_LABEL = {
    "experiences": "Their experience\nworking with peers",
    "services": "The services\npeers provide",
    "challenges": "Problems\n& gaps",
    "sustainability": "The future of\nthe programme",
    "closing": "Closing\nthoughts",
}
TOPIC_ORDER = list(TOPIC_LABEL)
CONTENT_TOPICS = ["experiences", "services", "challenges", "sustainability"]

DOMAIN_LABEL = {
    "A": "What peers do",
    "B": "What difference it makes",
    "C": "Problems and gaps",
    "D": "Suggestions",
    "E": "Training and sustainability",
}
DOMAIN_COLOR = {"A": "#2b6cb0", "B": "#2f855a", "C": "#c53030",
                "D": "#b7791f", "E": "#6b46c1"}
STANCE_COLOR = {"positive": "#2f855a", "mixed": "#b7791f",
                "neutral": "#a0aec0", "negative": "#c53030"}
INK, MUTED, GRID = "#1a202c", "#4a5568", "#cbd5e0"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 170,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.35,
})


def _wrap(text, fig, chars_per_inch):
    """Hard-wrap to the figure width.

    savefig.bbox is "tight", so a text line wider than the axes does not get
    clipped -- it drags the saved canvas out with it (one long footnote turned
    fig8 into a 4,484px-wide strip). Wrapping keeps every caption inside the
    figure it belongs to.
    """
    width = max(40, int(fig.get_figwidth() * chars_per_inch))
    return "\n".join(textwrap.wrap(str(text), width=width) or [""])


def titleize(fig, headline, subtitle, footnote=None, y=0.98):
    """Draw the headline/subtitle/footnote block and return the (top, bottom)
    fractions still free for the axes.

    tight_layout knows nothing about fig.text, so a caller passing a guessed
    `rect` had headlines sitting on top of axis titles whenever the text
    happened to wrap to an extra line. Measuring the block here and handing the
    numbers back makes the reservation exact instead of guessed.
    """
    def _lines(txt, pts, cpi):
        w = _wrap(txt, fig, cpi)
        return w, (w.count("\n") + 1) * pts * 1.34 / (fig.get_figheight() * 72.0)

    head, h_head = _lines(headline, 14.5, 8.0)
    sub, h_sub = _lines(subtitle, 9.8, 12.4)
    fig.text(0.012, y, head, ha="left", va="top",
             fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.012, y - h_head, sub, ha="left", va="top",
             fontsize=9.8, color=MUTED)
    top = y - h_head - h_sub - 0.035
    bottom = 0.02
    if footnote:
        note, h_note = _lines(footnote, 8.2, 15.0)
        fig.text(0.012, 0.004, note, ha="left", va="bottom",
                 fontsize=8.2, color=MUTED, style="italic")
        bottom = h_note + 0.03
    return top, bottom


def codebook_labels():
    """code id -> plain-English label (the text after the em dash)."""
    text = (PROJECT / "docs" / "codebook.md").read_text(encoding="utf-8")
    return {m.group(1): m.group(3).strip() for m in re.finditer(
        r"^###\s+(\w\d{1,2})\s+`([a-z_]+)`\s*[—-]\s*(.+)$", text, re.M)}


CODE_COLS = ["code_ids", "code_names", "n_codes", "llm_stance", "key_phrase",
             "code_confidence"]


def load():
    """Emotion columns come from the fused table; theme codes come from the
    coded table.

    These are two different files and they go stale independently: stage 09
    (fusion) copies whatever codes existed when it ran, so re-running the coder
    without re-running fusion leaves `utterances_fused.csv` holding an old,
    partial set. That is what produced a theme chart where every bar sat at
    2/34 = 6%. Codes are therefore always re-joined from `utterances_coded.csv`,
    which is the file the coder actually writes.
    """
    utt = pd.read_csv(DATA / "utterances_fused.csv", low_memory=False)
    coded_f = DATA / "utterances_coded.csv"
    if coded_f.exists():
        coded = pd.read_csv(coded_f, low_memory=False)
        cols = [c for c in CODE_COLS if c in coded.columns]
        if cols:
            fresh = coded[["utt_id"] + cols]
            n_old = utt.code_ids.notna().sum() if "code_ids" in utt.columns else 0
            n_new = fresh.code_ids.notna().sum()
            utt = utt.drop(columns=[c for c in cols if c in utt.columns])
            utt = utt.merge(fresh, on="utt_id", how="left")
            if n_new != n_old:
                print("[load] codes re-joined from utterances_coded.csv "
                      "(%d coded rows; the fused table held %d)" % (n_new, n_old))
    p = utt[(utt.speaker_role == "participant") & (utt.is_backchannel == 0)].copy()
    demo = None
    if (DATA / "participants.csv").exists():
        demo = pd.read_csv(DATA / "participants.csv")
        p = p.merge(demo, on="participant_id", how="left")
    return utt, p, demo


def code_long(p):
    if "code_ids" not in p.columns:
        return pd.DataFrame(columns=list(p.columns) + ["code"])
    d = p[p.code_ids.notna() & (p.code_ids != "")].copy()
    d["code"] = d.code_ids.str.split("|")
    return d.explode("code")


def fig1_themes(p, n_people):
    labels = codebook_labels()
    d = code_long(p)
    if d.empty:
        return
    # Refuse to draw a theme chart from partial coding. When only 2 of 34
    # participants carry codes, every bar lands on 2/34 = 6% and the chart looks
    # plausible while meaning nothing.
    coded_people = d.participant_id.nunique()
    if coded_people < 0.9 * n_people:
        print("  [skip] fig1: only %d of %d participants have codes -- coding is "
              "incomplete, theme percentages would be meaningless"
              % (coded_people, n_people))
        return
    freq = (d.groupby("code")
             .agg(people=("participant_id", "nunique"), mentions=("utt_id", "size"))
             .sort_values("people"))
    freq = freq[freq.people >= 2]
    if freq.empty:
        return
    freq["pct"] = 100 * freq.people / n_people

    fig, ax = plt.subplots(figsize=(11.4, max(5.0, 0.36 * len(freq) + 2.4)))
    y = np.arange(len(freq))
    colors = [DOMAIN_COLOR.get(c[0], MUTED) for c in freq.index]
    ax.barh(y, freq.pct, color=colors, alpha=.92, height=.72)
    ax.set_yticks(y, [labels.get(c, c) for c in freq.index], fontsize=9.4)
    for i, (pct, ppl) in enumerate(zip(freq.pct, freq.people)):
        ax.text(pct + 1.2, i, "%.0f%%  (%d of %d)" % (pct, ppl, n_people),
                va="center", fontsize=8.4, color=MUTED)
    ax.set_xlim(0, min(112, max(freq.pct) * 1.34))
    ax.set_xlabel("Percentage of providers who raised this")
    ax.xaxis.grid(True, color=GRID, lw=.7, alpha=.7)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=DOMAIN_COLOR[k], label=v)
                       for k, v in DOMAIN_LABEL.items()],
              loc="lower right", fontsize=9, framealpha=.96)
    _top, _bot = titleize(fig, "What emergency department staff said about the peer programme",
             "Each bar is the share of interviewed providers who raised that "
             "topic at least once. Colour groups related topics.",
             "%d healthcare providers interviewed. Topics raised by only one "
             "person are not shown." % n_people)
    fig.tight_layout(rect=[0, _bot, 1, _top])
    fig.savefig(FIGS / "fig1_what_they_talked_about.png")
    plt.close(fig)


def fig2_stance(p, n_people):
    if "llm_stance" not in p.columns or p.llm_stance.isna().all():
        return
    d = p[p.llm_stance.notna() & (p.llm_stance != "na")]
    if d.empty:
        return
    per = (d.assign(pos=d.llm_stance.eq("positive"))
             .groupby("participant_id").pos.mean())
    overall = d.llm_stance.value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    order = [s for s in ["positive", "neutral", "mixed", "negative"] if s in overall]
    ax = axes[0]
    left = 0
    for s in order:
        ax.barh([0], [overall[s]], left=left, color=STANCE_COLOR[s], height=.5)
        if overall[s] > 5:
            ax.text(left + overall[s] / 2, 0, "%s\n%.0f%%" % (s.capitalize(), overall[s]),
                    ha="center", va="center", fontsize=10.5,
                    color="white" if s != "neutral" else INK, fontweight="bold")
        left += overall[s]
    ax.set_xlim(0, 100); ax.set_ylim(-.45, .45)
    ax.set_yticks([])
    ax.set_xlabel("Percentage of everything providers said about the programme")
    ax.set_title("All statements pooled", fontsize=11, loc="left", color=INK, pad=10)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=.7, alpha=.6); ax.set_axisbelow(True)

    ax = axes[1]
    vals = (per * 100).sort_values()
    ax.barh(np.arange(len(vals)), vals, color="#2f855a", alpha=.85, height=.82)
    ax.axvline(50, color=MUTED, ls="--", lw=1)
    ax.set_yticks([]); ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of that person's statements that were positive")
    ax.set_ylabel("Each bar is one provider", fontsize=9.4)
    ax.set_title("Person by person", fontsize=11, loc="left", color=INK, pad=10)
    ax.xaxis.grid(True, color=GRID, lw=.7, alpha=.6); ax.set_axisbelow(True)
    n_pos = int((vals > 50).sum())
    ax.text(52, len(vals) * .05, "%d of %d providers\nwere mostly positive"
            % (n_pos, len(vals)), fontsize=9.2, color=MUTED)

    _top, _bot = titleize(fig, "Providers spoke about the peer programme far more positively "
                  "than negatively",
             "Left: every evaluative statement pooled together. Right: the same "
             "measure calculated separately for each provider.",
             "Statements classified by an automated coder reading the "
             "transcript; a second human coder has not yet checked them. "
             "%d providers." % n_people)
    fig.tight_layout(rect=[0, _bot, 1, _top])
    fig.savefig(FIGS / "fig2_overall_stance.png")
    plt.close(fig)


def fig3_sections(p):
    order = [t for t in TOPIC_ORDER if t in set(p.topic_segment)]
    if not order:
        return
    m = p.groupby("topic_segment")[["fused_valence", "fused_arousal"]].mean().reindex(order)
    n = p.groupby("topic_segment").participant_id.nunique().reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    specs = [("fused_valence", "Tone", "more positive", "more negative"),
             ("fused_arousal", "Energy", "more animated", "flatter")]
    ypos = np.arange(len(order))
    for ax, (col, name, hi, lo) in zip(axes, specs):
        v = m[col].values
        cols = ["#2f855a" if x >= 0 else "#c53030" for x in v]
        ax.barh(ypos, v, color=cols, alpha=.9, height=.6)
        ax.axvline(0, color=MUTED, lw=1)
        lim = max(0.3, np.nanmax(np.abs(v)) * 1.6)
        ax.set_xlim(-lim, lim)
        for i, x in enumerate(v):
            ax.text(x + (0.04 * lim if x >= 0 else -0.04 * lim), i, "%+.2f" % x,
                    va="center", ha="left" if x >= 0 else "right",
                    fontsize=8.6, color=MUTED)
        ax.set_title(name, fontsize=11.5, loc="left", color=INK, pad=10)
        ax.set_xlabel("<-  %s          %s  ->" % (lo, hi), fontsize=9.2)
        ax.xaxis.grid(True, color=GRID, lw=.7, alpha=.6); ax.set_axisbelow(True)
    axes[0].set_yticks(ypos, ["%s\n(%d people)" % (TOPIC_LABEL[t], n[t]) for t in order],
                       fontsize=9.2)
    axes[0].invert_yaxis()
    shift = ""
    f = TABLES / "topic_shift_tests.csv"
    if f.exists():
        ts = pd.read_csv(f)
        row = ts[ts.contrast.str.startswith("challenges")]
        if len(row):
            r = row.iloc[0]
            shift = (" Measured against each person's own answers about their "
                     "experience, %d of %d providers sounded more negative on "
                     "problems (average drop %.2f, p=%.4f)."
                     % (r.n_moved_down, r.n_people, abs(r.mean_shift), r.p))
    _top, _bot = titleize(fig, "Tone dipped when the conversation reached problems "
                  "- a small shift, but a consistent one",
             "Each section compared with how that same person sounded across "
             "their whole interview. Zero means \"exactly their own normal\"." + shift,
             "Tone combines the words used and the voice (pitch, loudness, pace, "
             "voice quality); energy comes from the voice alone. Units are "
             "standard deviations of each person's own range. The p-value is a "
             "one-sample t-test over 34 people, one value each.")
    fig.tight_layout(rect=[0, _bot, 1, _top])
    fig.savefig(FIGS / "fig3_how_each_section_felt.png")
    plt.close(fig)


def fig4_arc(p):
    topics = [t for t in CONTENT_TOPICS if t in set(p.topic_segment)]
    if not topics or p.participant_id.nunique() < 5:
        return
    nbin = 5
    xs, mean, lo, hi = [], [], [], []
    for ti, t in enumerate(topics):
        sub = p[p.topic_segment == t]
        for b in range(nbin):
            vals = []
            for _, g in sub.groupby("participant_id"):
                g = g.sort_values("start_s")
                if not len(g):
                    continue
                pos = np.linspace(0, 1, len(g), endpoint=False)
                sel = g[(pos >= b / nbin) & (pos < (b + 1) / nbin)]["fused_valence"]
                if len(sel):
                    vals.append(sel.mean())
            if len(vals) >= 5:
                mu, sd = np.mean(vals), np.std(vals, ddof=1)
                ci = 1.96 * sd / np.sqrt(len(vals))
                xs.append(ti + (b + .5) / nbin)
                mean.append(mu); lo.append(mu - ci); hi.append(mu + ci)
    if len(xs) < 3:
        return
    fig, ax = plt.subplots(figsize=(11.6, 4.9))
    ax.axhline(0, color=MUTED, ls="--", lw=1)
    ax.fill_between(xs, lo, hi, color="#2b6cb0", alpha=.18)
    ax.plot(xs, mean, color="#2b6cb0", lw=2.4)
    for ti in range(1, len(topics)):
        ax.axvline(ti, color=GRID, lw=1)
    top = ax.get_ylim()[1]
    for ti, t in enumerate(topics):
        ax.text(ti + .5, top, TOPIC_LABEL[t].replace("\n", " "),
                ha="center", va="bottom", fontsize=9.4, color=INK)
    ax.set_xticks([]); ax.set_xlim(0, len(topics))
    ax.set_ylabel("Tone\n(<- more negative   |   more positive ->)", fontsize=9.4)
    ax.yaxis.grid(True, color=GRID, lw=.7, alpha=.6); ax.set_axisbelow(True)
    _top, _bot = titleize(fig, "Tone dipped as the conversation moved to problems, then "
                  "recovered when it turned to the future",
             "The average across all providers as each interview moved through "
             "its four main sections. Shaded band is the 95% confidence interval.",
             "Each interview was stretched to the same length within each "
             "section so they could be averaged. Zero = each person's own normal.",
             y=0.99)
    fig.tight_layout(rect=[0, _bot, 1, _top])
    fig.savefig(FIGS / "fig4_emotional_arc.png")
    plt.close(fig)


def fig5_cases(p):
    stat = p.groupby("participant_id").agg(n=("utt_id", "size"),
                                           spread=("fused_valence", "std"))
    stat = stat[stat.n >= 25].dropna()
    if len(stat) < 2:
        return
    pick = list(stat.spread.nlargest(2).index) + list(stat.spread.nsmallest(1).index)
    pick = list(dict.fromkeys(pick))[:3]
    band = {"intro_consent": "#f7fafc", "experiences": "#ebf8ff",
            "services": "#f0fff4", "challenges": "#fff5f5",
            "sustainability": "#faf5ff", "closing": "#f7fafc"}
    fig, axes = plt.subplots(len(pick), 1, figsize=(11.6, 2.5 * len(pick)))
    axes = np.atleast_1d(axes)
    for ax, pid in zip(axes, pick):
        g = p[p.participant_id == pid].sort_values("start_s")
        w = max(3, min(9, len(g) // 8))
        for t, gg in g.groupby("topic_segment"):
            ax.axvspan(gg.start_s.min() / 60, gg.end_s.max() / 60,
                       color=band.get(t, "#f7fafc"), lw=0)
        ax.axhline(0, color=MUTED, ls="--", lw=.9)
        ax.plot(g.start_s / 60,
                g.fused_valence.rolling(w, center=True, min_periods=1).mean(),
                color="#2b6cb0", lw=2.0)
        d = g[g.discord_flag == 1]
        if len(d):
            ax.scatter(d.start_s / 60, d.fused_valence, s=34, color="#c53030",
                       zorder=5, edgecolor="white", linewidth=.8)
        occ = ""
        if "occupation" in g.columns and g.occupation.notna().any():
            occ = str(g.occupation.dropna().iloc[0])
        ax.set_ylabel("%s\n%s" % (pid, occ), fontsize=8.6)
        ax.set_xlim(0, g.end_s.max() / 60)
        ax.yaxis.grid(True, color=GRID, lw=.6, alpha=.5); ax.set_axisbelow(True)
    axes[-1].set_xlabel("Minutes into the interview")
    axes[0].legend(handles=[
        Line2D([], [], color="#2b6cb0", lw=2, label="tone (smoothed)"),
        Line2D([], [], marker="o", color="#c53030", lw=0, markersize=7,
               label="words and voice disagreed")],
        fontsize=8.6, loc="upper right", ncol=2, framealpha=.95)
    _top, _bot = titleize(fig, "Individual interviews: some providers varied a lot, others "
                  "stayed level throughout",
             "Three example providers. Background shading marks the section of "
             "the interview; red dots mark moments where the words and the voice "
             "pointed in different directions.",
             "Tone smoothed over neighbouring sentences. Zero = that person's "
             "own normal.")
    fig.tight_layout(rect=[0, _bot, 1, _top])
    fig.savefig(FIGS / "fig5_individual_examples.png")
    plt.close(fig)


def fig6_discord(p):
    sub = p.dropna(subset=["tx_valence_z", "ac_valence_z"])
    if len(sub) < 20:
        return
    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    ok = sub[sub.discord_flag == 0]
    bad = sub[sub.discord_flag == 1]
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    ax.scatter(ok.tx_valence_z, ok.ac_valence_z, s=11, color="#a0aec0", alpha=.45,
               label="words and voice agree")
    ax.scatter(bad.tx_valence_z, bad.ac_valence_z, s=32, color="#c53030", alpha=.9,
               edgecolor="white", linewidth=.5, label="mismatch - worth listening to")
    lim = float(np.nanpercentile(np.abs(np.r_[sub.tx_valence_z.values,
                                              sub.ac_valence_z.values]), 99))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("What they said\n(<- negative words        positive words ->)",
                  fontsize=9.4)
    ax.set_ylabel("How they sounded\n(<- flat, subdued        warm, bright ->)",
                  fontsize=9.4)
    for x, y, t in [(.72, .94, "positive words,\nwarm voice"),
                    (.72, .04, "positive words,\nflat voice"),
                    (.03, .94, "negative words,\nwarm voice"),
                    (.03, .04, "negative words,\nflat voice")]:
        ax.text(x, y, t, transform=ax.transAxes, fontsize=8.4, color=MUTED)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, frameon=False)
    _top, _bot = titleize(fig, "Words and voice usually matched - the exceptions are the "
                  "interesting moments",
             "Each dot is one sentence. Red dots are sentences where a provider "
             "used positive words but sounded flat, or the reverse.",
             "%d of %d sentences flagged. These are listed with timestamps in "
             "outputs/tables/qc_discord_reading.csv so they can be listened to."
             % (len(bad), len(sub)))
    fig.tight_layout(rect=[0, _bot, 1, _top])
    fig.savefig(FIGS / "fig6_words_vs_voice.png")
    plt.close(fig)


def fig7_groups(p, demo):
    if demo is None or "occupation_category" not in p.columns:
        return
    labels = codebook_labels()
    d = code_long(p)
    # NOT overall mean tone: within-speaker z-scoring forces every person's mean
    # to ~0, so a between-person comparison of level is noise (see script 10).
    # The within-person shift when the subject turns to problems does vary.
    wide = (p.groupby(["participant_id", "topic_segment"]).fused_valence.mean()
             .unstack("topic_segment"))
    if not {"challenges", "experiences"} <= set(wide.columns):
        return
    per = (wide["challenges"] - wide["experiences"]).rename("tone_shift").reset_index()
    per = per.merge(p[["participant_id", "occupation_category"]].drop_duplicates(),
                    on="participant_id").dropna(subset=["tone_shift"])

    fig = plt.figure(figsize=(13.6, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=.55)

    ax = fig.add_subplot(gs[0, 0])
    groups = sorted(per.occupation_category.dropna().unique(),
                    key=lambda g: per[per.occupation_category == g].tone_shift.median())
    data = [per[per.occupation_category == g].tone_shift.dropna() for g in groups]
    bp = ax.boxplot(data, vert=False, widths=.55, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.6),
                    flierprops=dict(marker=""))
    for patch in bp["boxes"]:
        patch.set(facecolor="#ebf4ff", edgecolor="#2b6cb0", lw=1.1)
    rng = np.random.default_rng(3)
    for i, v in enumerate(data, start=1):
        ax.scatter(v, rng.normal(i, .07, len(v)), s=26, color="#2b6cb0", alpha=.75,
                   zorder=3, edgecolor="white", linewidth=.5)
    ax.axvline(0, color=MUTED, ls="--", lw=1)
    ax.set_yticks(range(1, len(groups) + 1),
                  ["%s\n(%d people)" % (g, len(v)) for g, v in zip(groups, data)],
                  fontsize=9.2)
    ax.set_xlabel("<- tone dropped further        tone held up ->", fontsize=9.2)
    ax.set_title("How far each person's tone dropped\n"
                 "when talking about problems",
                 fontsize=10.5, loc="left", color=INK, pad=10)
    ax.xaxis.grid(True, color=GRID, lw=.7, alpha=.6); ax.set_axisbelow(True)

    ax2 = fig.add_subplot(gs[0, 1])
    if not d.empty:
        top = (d.groupby("code").participant_id.nunique()
                .sort_values(ascending=False).head(10).index[::-1])
        occs = sorted(p.occupation_category.dropna().unique())
        mat, ann = [], []
        for c in top:
            row, arow = [], []
            for o in occs:
                pool = p[p.occupation_category == o].participant_id.nunique()
                said = d[(d.code == c) & (d.occupation_category == o)].participant_id.nunique()
                row.append(100 * said / pool if pool else np.nan)
                arow.append("%d/%d" % (said, pool))
            mat.append(row); ann.append(arow)
        mat = np.array(mat, dtype=float)
        im = ax2.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax2.set_xticks(range(len(occs)), occs, fontsize=8.6, rotation=18, ha="right")
        ax2.set_yticks(range(len(top)),
                       ["\n".join(textwrap.wrap(labels.get(c, c), 26)) for c in top],
                       fontsize=8.4)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax2.text(j, i, ann[i][j], ha="center", va="center", fontsize=7.6,
                             color="white" if mat[i, j] > 55 else INK)
        ax2.set_title("How many people in each profession\n"
                      "raised the ten most common topics",
                      fontsize=10.5, loc="left", color=INK, pad=10)
        ax2.grid(False)
        cb = fig.colorbar(im, ax=ax2, shrink=.82, pad=.02)
        cb.set_label("% of that profession", fontsize=8.6)

    _top, _bot = titleize(fig, "Differences between professions are small, and this sample "
                  "can only show large ones",
             "Left: each dot is one provider. Right: each cell reads \"people "
             "who raised it / people in that profession\".",
             "34 providers, 5-12 per profession -- descriptive, not a test. "
             "Overall tone is deliberately not compared between people: every "
             "measure is centred on the speaker's own average, so everyone's "
             "overall level is zero by construction.")
    # tight_layout silently does nothing here (gridspec + an attached
    # colorbar), which is why the panel titles kept landing on the subtitle.
    # -0.07 leaves room for the two-line per-panel titles, which sit above the
    # axes and are not part of what titleize measures.
    fig.subplots_adjust(top=_top - 0.07, bottom=_bot + 0.09, left=0.10, right=0.97)
    fig.savefig(FIGS / "fig7_by_profession.png")
    plt.close(fig)


def fig8_model():
    """Two panels, because the analysis fits two models (see script 10).

    Left  - how one person's tone moved between sections of the interview.
    Right - how providers differed from each other overall.
    They use different scales and different baselines, so plotting them on one
    axis (as an earlier version did) invited exactly the wrong comparison.
    """
    f = TABLES / "mixed_model_coefficients.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    if "scope" not in d.columns:
        return
    d = d[d.term.notna()].copy()

    def pretty(term):
        t = str(term).replace("C(", "").replace(")", "").replace("[T.", " = ").rstrip("]")
        t = (t.replace("topic_segment", "Section").replace("occupation_category", "")
              .replace("recording_device", "Recording setup").replace("gender", "")
              .replace("sex", "Sex"))
        for k, v in TOPIC_LABEL.items():
            t = t.replace(k, v.replace("\n", " "))
        t = (t.replace("aac_32000_1ch", "handheld recorder")
              .replace("mp3_44100_2ch", "room / video-call recording"))
        t = re.sub(r"^\s*=\s*", "", t).replace("]", "").replace(":", " x ")
        t = re.sub(r"\bx\s*=\s*", "x ", t).strip()
        return re.sub(r"\s{2,}", " ", t)

    panels = [
        ("within", "fused_valence",
         "Within one provider:\nhow their tone moved between sections",
         "vs. the same person on their own experience"),
        ("between", "ac_valence",
         "Between providers:\nhow they differed from each other",
         "vs. a woman in nursing, handheld recorder"),
    ]
    have = [(s, dv, h, b) for s, dv, h, b in panels
            if len(d[(d.scope == s) & (d.dv == dv)])]
    if not have:
        return

    fig, axes = plt.subplots(1, len(have), figsize=(13.4, 5.0))
    axes = np.atleast_1d(axes)
    for ax, (scope, dv, head, base) in zip(axes, have):
        s = d[(d.scope == scope) & (d.dv == dv)].iloc[::-1].copy()
        s["label"] = s.term.map(pretty)
        y = np.arange(len(s))
        ax.axvline(0, color=MUTED, ls="--", lw=1)
        ax.errorbar(s.estimate, y,
                    xerr=[s.estimate - s.ci_low, s.ci_high - s.estimate],
                    fmt="o", ms=5, lw=1.3, color="#a0aec0", ecolor="#a0aec0",
                    capsize=3)
        sig = (s.p < .05).values
        ax.scatter(np.asarray(s.estimate)[sig], y[sig], s=62, color="#2b6cb0",
                   zorder=5, edgecolor="white", linewidth=.8)
        ax.set_yticks(y, s.label, fontsize=9.0)
        ax.set_title(head, fontsize=10.6, fontweight="bold", color=INK,
                     loc="left", pad=9)
        ax.set_xlabel("more negative  <-      ->  more positive\n%s" % base,
                      fontsize=8.4, color=MUTED)
        ax.xaxis.grid(True, color=GRID, lw=.7, alpha=.6)
        ax.set_axisbelow(True)
        ax.margins(y=.12)
        ax.tick_params(axis="x", labelsize=8.2)
        ax.locator_params(axis="x", nbins=5)

    icc = d[d.scope == "between"].icc
    icc = float(icc.iloc[0]) if len(icc) else np.nan
    _top, _bot = titleize(fig, "What moved the tone of these interviews",
             "Filled blue dots are differences big enough to be distinguished "
             "from zero; bars are 95% confidence intervals. The two panels "
             "answer different questions and are not on the same scale.",
             "Mixed-effects models over 34 providers (3,966 statements). Left "
             "panel uses each person's own average as their zero, so it cannot "
             "speak to differences between people; the right panel uses "
             "uncentred scores, where %.0f%% of the variation sits between "
             "providers. Profession and recording-setup comparisons are "
             "exploratory: the recording setup itself shifts the measurement, "
             "so differences between groups recorded differently may be "
             "equipment, not feeling." % (100 * icc if np.isfinite(icc) else 0))
    fig.tight_layout(rect=[0, _bot, 1, _top])
    fig.savefig(FIGS / "fig8_what_predicts_tone.png")
    plt.close(fig)


def main() -> None:
    for old in FIGS.glob("fig*.png"):
        old.unlink()
    _utt, p, demo = load()
    p = p[p.fused_valence.notna()]
    p = p[~p.topic_segment.isin(EXCLUDE_TOPICS)]
    n_people = p.participant_id.nunique()
    print("[in] %d analysable sentences from %d providers" % (len(p), n_people))

    for fn, args in [(fig1_themes, (p, n_people)), (fig2_stance, (p, n_people)),
                     (fig3_sections, (p,)), (fig4_arc, (p,)), (fig5_cases, (p,)),
                     (fig6_discord, (p,)), (fig7_groups, (p, demo)),
                     (fig8_model, ())]:
        try:
            fn(*args)
        except Exception as e:
            print("  [warn] %s failed: %s: %s" % (fn.__name__, type(e).__name__, e))

    made = sorted(FIGS.glob("*.png"))
    print("\nwrote %d figures to outputs/figures/" % len(made))
    for f in made:
        print("  %s" % f.name)


if __name__ == "__main__":
    main()
