"""
Step 3.2 — Text-side sentiment, for contrast against the acoustic side.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest (3-class, ungated),
chosen for being trained on short conversational English rather than on product
reviews or newswire.

Framework 3.2 is explicit that this measures something different from the
acoustic side -- "was the content positive or negative", versus "what state was
the speaker in". Divergence between them is the signal of interest (see 4.3),
not noise, so both are kept on their own scales and z-scored within speaker.

Outputs tx_valence in [-1, 1] as p(pos) - p(neg), plus the raw class
probabilities and a discrete label.

Usage: python scripts/07_text_sentiment.py [--device cuda] [--batch 32]
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
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def within_speaker_z(df, cols, group=("session_id", "speaker_raw")):
    g = df.groupby(list(group))[cols]
    mu, sd = g.transform("mean"), g.transform("std")
    return ((df[cols] - mu) / sd.replace(0, np.nan)).fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    src = DATA / "utterances_topics.csv"
    if not src.exists():
        raise SystemExit("run scripts/06_topic_segments.py first")
    utt = pd.read_csv(src)

    print("[init] loading %s on %s" % (MODEL, args.device), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    net = AutoModelForSequenceClassification.from_pretrained(MODEL).to(args.device).eval()
    # id2label is {0: negative, 1: neutral, 2: positive}
    labels = [net.config.id2label[i].lower() for i in range(net.config.num_labels)]
    ineg, ipos = labels.index("negative"), labels.index("positive")

    texts = utt.text.fillna("").astype(str).tolist()
    probs = np.full((len(texts), len(labels)), np.nan, dtype=float)
    idx = [i for i, t in enumerate(texts) if t.strip()]
    print("[run] scoring %d non-empty utterances" % len(idx), flush=True)

    for s in range(0, len(idx), args.batch):
        chunk = idx[s:s + args.batch]
        enc = tok([texts[i] for i in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=256).to(args.device)
        with torch.no_grad():
            logits = net(**enc).logits
        probs[chunk] = torch.softmax(logits, dim=-1).cpu().numpy()
        if s % (args.batch * 40) == 0:
            print("  %d/%d" % (s, len(idx)), flush=True)

    for j, name in enumerate(labels):
        utt["tx_p_%s" % name] = probs[:, j]
    utt["tx_valence"] = probs[:, ipos] - probs[:, ineg]
    utt["tx_emotion_label"] = [labels[int(np.argmax(r))] if not np.isnan(r[0]) else None
                               for r in probs]
    # framework 3.2 also wants a stance field; the 3-class output is the natural
    # source for it, with 'mixed' when no class dominates
    def stance(row):
        if np.isnan(row[0]):
            return None
        top = int(np.argmax(row))
        if row[top] < 0.50:
            return "mixed"
        return {ineg: "negative", ipos: "positive"}.get(top, "neutral")
    utt["tx_stance"] = [stance(r) for r in probs]

    z = within_speaker_z(utt, ["tx_valence"])
    utt["tx_valence_z"] = z["tx_valence"]

    utt.to_csv(DATA / "utterances_text.csv", index=False, encoding="utf-8-sig")
    print("\nwrote data/utterances_text.csv (%d rows)" % len(utt))

    p = utt[(utt.speaker_role == "participant") & (utt.is_backchannel == 0)]
    print("\nParticipant utterance stance distribution:")
    print(p.tx_stance.value_counts(dropna=False).to_string())
    print("\ntx_valence by topic segment:")
    print(p.groupby("topic_segment").tx_valence.agg(["size", "mean", "std"]).round(3).to_string())
    if "ac_valence_z" in p.columns:
        sub = p.dropna(subset=["ac_valence_z", "tx_valence_z"])
        if len(sub) > 2:
            print("\ncorr(tx_valence_z, ac_valence_z) = %+.3f over %d utterances"
                  % (sub.tx_valence_z.corr(sub.ac_valence_z), len(sub)))
            print("(modest positive is expected; near-zero or negative means the "
                  "two channels disagree, which is what framework 4.3 hunts for)")


if __name__ == "__main__":
    main()
