"""
Step 2 — Acoustic emotion features at the utterance level (L1).

Two parallel tracks, exactly as framework section 2 specifies:

  2.1 openSMILE eGeMAPSv02 functionals (88 interpretable dims)
  2.2 a deep speech-emotion model

For 2.2 we use audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim, trained
on English conversational speech (MSP-Podcast) and, more importantly, emitting
continuous arousal / dominance / valence directly rather than discrete labels. That suits framework section 8's warning
that discrete labels are near-useless in low-arousal interview speech and that
continuous dimensions should be preferred.

Within-speaker z-scoring (framework 2.1, "must do") is applied to every feature
using each participant's own mean and SD, which simultaneously removes the
speaker's baseline, their gender-linked F0, and the two recording conditions.

Usage: python scripts/05_acoustic_features.py [--device cuda] [--limit N]
"""
import argparse
import os
from pathlib import Path

# Scratch space for models, converted audio and intermediate JSON. Kept
# outside the project folder so cloud-sync clients do not churn on it.
# Override with the HCP_WORK environment variable.
WORK = Path(os.environ.get("HCP_WORK") or (Path.home() / "hcp_work"))
os.environ.setdefault("HF_HOME", str(WORK / "models" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from transformers import Wav2Vec2Processor  # noqa: E402
from transformers.models.wav2vec2.modeling_wav2vec2 import (  # noqa: E402
    Wav2Vec2Model, Wav2Vec2PreTrainedModel,
)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
SR = 16000
DIM_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

# The subset framework 2.1 asks to report explicitly, mapped to eGeMAPS names.
HEADLINE = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": "f0_mean",
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0": "f0_p20",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0": "f0_p80",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "f0_cv",
    "loudness_sma3_amean": "loudness_mean",
    "loudness_sma3_percentile80.0": "loudness_p80",
    "jitterLocal_sma3nz_amean": "jitter",
    "shimmerLocaldB_sma3nz_amean": "shimmer",
    "HNRdBACF_sma3nz_amean": "hnr",
    "VoicedSegmentsPerSec": "voiced_per_s",
    "MeanVoicedSegmentLengthSec": "voiced_seg_len",
    "MeanUnvoicedSegmentLength": "unvoiced_seg_len",
}


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Arousal / dominance / valence regressor, per the model card."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0]
        pooled = torch.mean(hidden, dim=1)
        return self.classifier(pooled)


def within_speaker_z(df, cols, group=("session_id", "speaker_raw")):
    """Framework 2.1: z-score each feature inside each speaker.

    The grouping key is the *speaker*, not the session: each recording holds an
    interviewer as well as a participant, and pooling them would fold the
    interviewer's baseline pitch and level into the participant's mean/SD.
    """
    g = df.groupby(list(group))[cols]
    mu, sd = g.transform("mean"), g.transform("std")
    return ((df[cols] - mu) / sd.replace(0, np.nan)).fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-dur", type=float, default=0.30)
    args = ap.parse_args()

    import opensmile
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)

    print("[init] loading %s on %s" % (DIM_MODEL, args.device), flush=True)
    proc = Wav2Vec2Processor.from_pretrained(DIM_MODEL)
    net = EmotionModel.from_pretrained(DIM_MODEL).to(args.device).eval()

    utt = pd.read_csv(DATA / "utterances.csv")
    if args.limit:
        utt = utt.head(args.limit)
    print("[run] %d utterances across %d sessions"
          % (len(utt), utt.session_id.nunique()), flush=True)

    rows = []
    for sid, grp in utt.groupby("session_id", sort=True):
        wav = WORK / "audio_16k" / ("%s.wav" % sid)
        audio, sr = sf.read(str(wav), dtype="float32")
        assert sr == SR, "expected 16 kHz, got %d" % sr
        done = 0
        for _, r in grp.iterrows():
            a, b = int(r.start_s * SR), int(r.end_s * SR)
            x = audio[max(a, 0):min(b, len(audio))]
            rec = {"utt_id": r.utt_id}
            if len(x) < args.min_dur * SR:
                rows.append(rec)  # too short to characterise; stays NaN
                continue
            try:
                f = smile.process_signal(x, SR).iloc[0]
                for src, dst in HEADLINE.items():
                    rec[dst] = float(f[src])
                rec["egemaps_ok"] = 1
            except Exception:
                rec["egemaps_ok"] = 0
            with torch.no_grad():
                inp = proc(x, sampling_rate=SR, return_tensors="pt").input_values
                out = net(inp.to(args.device)).cpu().numpy().reshape(-1)
            # model card order: arousal, dominance, valence (each in 0..1)
            rec["ac_arousal"], rec["ac_dominance"], rec["ac_valence"] = map(float, out)
            rows.append(rec)
            done += 1
        print("  %s: %d/%d utterances featurised" % (sid, done, len(grp)), flush=True)

    feats = pd.DataFrame(rows)
    merged = utt.merge(feats, on="utt_id", how="left")

    # Within-speaker standardisation on everything that is a real measurement.
    zcols = [c for c in list(HEADLINE.values()) + ["ac_arousal", "ac_dominance",
                                                   "ac_valence", "words_per_s"]
             if c in merged.columns]
    zs = within_speaker_z(merged, zcols)
    zs.columns = ["%s_z" % c for c in zs.columns]
    merged = pd.concat([merged, zs], axis=1)

    merged.to_csv(DATA / "utterances_acoustic.csv", index=False, encoding="utf-8-sig")
    print("\nwrote data/utterances_acoustic.csv (%d rows, %d cols)"
          % (len(merged), merged.shape[1]))

    # framework 2.3 checks belong on real participant speech only
    ok = merged[(merged.speaker_role == "participant")
                & (merged.is_backchannel == 0)].dropna(subset=["ac_valence"])
    n_feat = merged.ac_valence.notna().sum()
    print("  utterances with acoustic features: %d (%.0f%%)"
          % (n_feat, 100 * n_feat / max(len(merged), 1)))
    print("  participant, non-backchannel utterances used for checks: %d" % len(ok))
    if len(ok):
        print(ok[["ac_arousal", "ac_dominance", "ac_valence"]].describe().round(3).to_string())
        # framework 2.3 sanity check: arousal should track F0 and loudness
        for c in ["f0_mean_z", "loudness_mean_z"]:
            if c in ok.columns:
                print("  corr(ac_arousal_z, %s) = %+.3f"
                      % (c, ok["ac_arousal_z"].corr(ok[c])))


if __name__ == "__main__":
    main()
