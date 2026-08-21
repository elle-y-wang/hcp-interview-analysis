# Multimodal analysis of healthcare-provider interviews

Analysis pipeline for a set of interviews with emergency department staff about a
**peer support worker programme** — people with lived experience of substance use
working alongside clinical staff in the ED.

The pipeline treats each interview as three nested levels — utterance, topic
segment, participant — and measures each utterance **two independent ways**: what
the words say, and what the voice sounds like. Where the two disagree is treated
as signal, not noise.

> **No study data is in this repository.** Recordings, transcripts, demographics
> and all derived tables are excluded by `.gitignore` and stay on the analysis
> machine. What is here is code, the codebook, and the method write-up.

---

## What it does

| # | script | does |
|---|---|---|
| 01 | `01_prepare_audio.py` | 16 kHz mono conversion; loudness, clipping and SNR checks |
| 02 | `02_transcribe.py` | Whisper `large-v3` ASR with word-level timestamps |
| 03 | `03_diarize.py` | Speaker diarization: cam++ embeddings + spectral clustering |
| 04 | `04_build_utterances.py` | Utterance table, speaker roles, study IDs, de-identification |
| 04b | `04b_align_cleaned.py` | Aligns human-corrected transcripts onto the audio timeline |
| 05 | `05_acoustic_features.py` | eGeMAPS features + continuous arousal/valence; within-speaker z-scoring |
| 06 | `06_topic_segments.py` | Topic segmentation from spoken section cues |
| 07 | `07_text_sentiment.py` | Text-side sentiment |
| 08 | `08_llm_coding.py` | Thematic coding against `docs/codebook.md` |
| 09 | `09_fusion.py` | Weighted fusion, discord flags, validation sample |
| 09b | `09b_fusion_llm.py` | Alternative fusion: acoustic cues passed to an LLM |
| 10 | `10_aggregate_models.py` | Topic- and participant-level tables, mixed-effects models |
| 11 | `11_figures.py` | Figures, written for a reader with no project background |
| 12 | `12_qc_checks.py` | QC worksheets for manual listening and reading |
| 13 | `13_compare_table4.py` | Compares automated theme frequencies against manual coding |

Stages are resumable — re-running only does outstanding work.

```bash
python scripts/run_all.py              # everything
python scripts/run_all.py --from 05    # resume from a stage
python scripts/run_all.py --only 10 11 # just these
python scripts/run_all.py --skip 08 09b
```

## Models used

| Model | Role |
|---|---|
| `Systran/faster-whisper-large-v3` | Speech to text, with per-word timestamps |
| `iic/speech_campplus_sv_zh-cn_16k-common` | 192-d speaker embeddings for diarization (identity, not language) |
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | Continuous arousal / dominance / valence from the voice |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment of the text alone |
| `dslim/bert-base-NER` | Person-name detection for de-identification |
| Claude (local CLI, headless) | Applies the fixed codebook; does not invent themes |
| openSMILE **eGeMAPSv02** | Not a model — 88 interpretable acoustic measures (pitch, loudness, rate, voice quality) |

## Two design decisions worth knowing

**Every measure is z-scored within speaker.** Some people are simply more
animated than others; pooling speakers would mostly measure personality. Zero
means "exactly how this person normally sounds". The grouping key is
`(session, speaker)`, not session — pooling an interviewer with their
participant folds the wrong baseline in.

**That choice forces two models, not one.** Within-speaker centring sets each
person's mean to zero, so predictors that are constant within a person
(profession, gender, recording device) have almost no variance left to explain.
Those are fitted separately on uncentred scores. Every coefficient row carries a
`scope` column saying which model produced it. See `docs/method_report.md` §10.

## Setup

Python 3.11. Model weights, converted audio and cached intermediates live outside
the project folder, in `$HCP_WORK` (default `~/hcp_work`) so cloud-sync clients
do not churn on them.

```bash
export HCP_WORK=/path/to/scratch      # optional; defaults to ~/hcp_work
pip install -r requirements.txt
```

`ffmpeg` and `ffprobe` are expected at `$HCP_WORK/bin`.

> `transformers` is pinned to **4.46.3**: 5.x changes a loading path the
> emotion model depends on.

## Repository contents

```
scripts/   the pipeline
docs/      codebook.md, interview_guide.md, method_report.md
```

`docs/method_report.md` is the substantive document — it covers what the data
turned out to be, every substitution made and why, the validation checks, and
the limitations. Read it before reusing any of this.

## Data handling

Participants are identified by study number only. Person names spoken during the
interviews are removed by a combined NER, gazetteer and fuzzy-match pass, and the
only name mapping is written to a directory that is never committed.

Excluded from this repository and from any share or supplement:

- audio recordings and transcripts, original or corrected
- demographics
- every derived table under `data/` and `outputs/`
- the session-to-participant mapping

## Status

The automated coding reproduces the research team's manual theme ranking closely,
but **formal inter-rater reliability has not yet been established** — that needs
a second human coder working through a held-out sample. Treat the theme
frequencies as provisional until that is done. Remaining manual checks are listed
at the end of `docs/method_report.md`.

The quantitative measures are positioning tools. This is a qualitative study
first: the numbers describe patterns and point at passages worth close reading,
and are not offered as validated affect measurement.
