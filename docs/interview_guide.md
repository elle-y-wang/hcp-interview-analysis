# Interview guide (reconstructed)

**Status: reconstructed from the recordings, not supplied.** The framework's
input list (§1.1) expects `interview_guide.md` as a given. No guide was provided
with this dataset, so the structure below was recovered from what the
interviewers actually say on tape — they announce section transitions out loud
("the next section asks about challenges and areas of improvement"), which makes
the guide recoverable and makes automatic L2 segmentation feasible.

**Please check this against the real protocol before it is used in a write-up.**
Section names here are descriptive labels chosen to match the spoken cues; the
study's own names may differ.

**Study**: Emergency Department Peer Support Worker Visibility Study. Peer
support workers (PSWs) with lived experience of substance use were introduced
into a hospital emergency department. These are healthcare provider (HCP)
interviews about that program.

---

## Session structure

| # | `topic_segment` | What it covers | Typical spoken cue |
|---|---|---|---|
| 0 | `intro_consent` | Recording slate, verbal consent (name + study ID), demographics survey check, interviewer reads the PSW role description | "for the audio recording today is…" |
| 1 | `experiences` | Direct experience working with PSWs; effect on interactions with patients who use substances; patient engagement; willingness to stay for care; patient experience in the ED; impact on the ED team; specific cases | "with that, I will get into the first question… describe your experiences" |
| 2 | `services` | The services PSWs actually provide; whether they met patient needs; which services are most valuable | "the next group of questions asks about their services" |
| 3 | `challenges` | Challenges encountered; role boundaries; areas for improvement | "the next section asks about challenges and areas of improvement" |
| 4 | `sustainability` | Ongoing role for PSWs; integration into ED workflow; training/support needed by PSWs and by HCPs; whether the program should be sustained | "the next section is on sustainability and future development" |
| 5 | `closing` | Any additional insights; logistics (email, gift card) | "that was our last question for today. Do you have any additional insights?" |

---

## Questions observed on tape

Wording varies between interviewers; these are representative.

### 1. Experiences
- Can you describe your experiences working with the peer support worker in the ED?
- How has the presence of a peer support worker influenced your interactions with patients who use substances?
- Have you noticed any changes in patient engagement since implementing the peer support worker?
- Have you noticed any changes in willingness to stay for care?
- Have you noticed any changes in patient experiences in the ED, both positive or negative?
- How has the peer support worker impacted care provided by the ED team?
- Can you describe any specific cases or examples where the peer support worker influenced a patient's trajectory in the ED, and the impact of that?
- In what way do you feel the peer support worker is perceived by nurses or doctors?

### 2. Services
- Have the services provided by the peer support worker met the needs of patients?

### 3. Challenges
- Have you experienced any challenges working with the peer support worker?
- What areas of improvement would you suggest?

### 4. Sustainability and future development
- What role do you see peer support workers fulfilling in the ED setting on an ongoing basis?
- How could we better incorporate peer support workers into existing ED workflows and team dynamics?
- What additional training or supports do peer support workers need to collaborate more effectively with healthcare providers?
- On the flip side, what additional training or supports might healthcare providers need to collaborate more effectively with peer support workers?
- Do you see a benefit in sustaining the peer support worker program into the future?
- What suggestions do you have for ensuring the long-term sustainability of the program?

### 5. Closing
- Do you have any additional insights?

---

## Notes affecting analysis

- **Almost all are one-on-one interviews, not focus groups.** Every filename says
  `HCPFocusGroup` and the interviewers say "this is a healthcare provider focus
  group", but diarization finds two speakers in nearly every recording, with the
  consent statement read by a single named individual.
  **Four sessions are the exception**, each with three speakers and two
  consenting participants (the interviewer addresses them in the plural). Study
  IDs are therefore assigned **per speaker, not per session**, so paired
  participants stay separate L3 units. Per-session speaker and participant counts are in
  `data/role_assignment_check.csv` (`n_participants`).
- **One recording produces a spurious third cluster** of ~0.5 s. Speakers
  holding under 2% of the non-interviewer speech and carrying no study ID are
  labelled `unassigned` and excluded from analysis.
- **The questions overlap heavily.** Participants frequently say "I think I
  already answered that", and interviewers acknowledge the guide is "kind of
  repetitive". Adjacent segments in the `experiences` block are therefore not
  cleanly separable, which matters when interpreting topic-level emotion
  contrasts.
- **ASR transcribes "ED" as "EU"/"AD"/"EV" fairly often.** Normalised in
  `scripts/06_topic_segments.py` before cue matching.
