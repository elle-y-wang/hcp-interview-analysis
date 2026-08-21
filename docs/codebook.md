# Codebook v2 — HCP experiences of the ED Peer Program

**Status: v2, aligned to the team's preliminary analysis.**
Revised from v1 after reading *ED PSW Study_Phase II_HCP FG Preliminary
Analysis_11SEPT2025.docx*. Three changes:

1. **Terminology now matches the team's.** The report says "**peer**" and "the
**peer program**", not "PSW" / "peer support worker". Codes are named the same
   way so results drop straight into the existing write-up.
2. **Category structure now follows the team's Table 4**, which is itself
   organised around the interview guide (experiences → impacts → services →
   feedback → sustainability).
3. **Code wording is taken from Table 4 wherever a code exists there**, so this
   analysis and the manual analysis are directly comparable rather than two
   parallel vocabularies.

**Scope difference worth stating in any write-up.** The team's preliminary
frequency analysis covers **20 of 34 participants** (17 transcripts, limited by
time). This pipeline codes **all 34 participants across all 30 recordings**, so
percentages here will not match Table 4 exactly and are not expected to. Use
`direct_peer_contact` in `data/participants.csv` to reproduce a comparable
subset.

**Granularity.** Table 4 has ~100 codes, many appearing once. That tail is too
sparse for reliable automated coding, so codes appearing in ≥10% of the team's
sample are kept close to verbatim and the long tail is folded into its parent.
33 codes in five domains.

Framework §3.1 step C (two human coders, Krippendorff's α) is still outstanding.

Format: `### ID `slug` — Plain-English label used in figures`

---

## Domain A — What peers actually do

### A1 `builds_rapport` — Builds rapport and trust with patients
Peer connects with patients, gains their trust, gets them to open up; patients
speak more freely to the peer than to clinical staff.
- *Maps Table 4 rows: "Peer builds rapport with patients", "Peer makes patients more willing to speak to staff"*

### A2 `emotional_support` — Offers emotional support and company
Sitting with, listening to, checking in on patients; being present through long
waits; letting patients offload. Includes spending time clinical staff cannot.
- *Maps Table 4 rows: "Peer provides emotional support for patients", "Peer spends more time with patients"*

### A3 `basic_needs` — Provides food, blankets and other basics
Practical comfort needs: food, drink, blankets, clothing, phone charging,
accompanying a patient outside for air or a cigarette, wheelchair help.
- *Maps Table 4 rows: "Peer provides basic needs (e.g. food, blanket)", "Peer can accompany patients outside"*

### A4 `lived_experience` — Draws on their own lived experience
The peer's own history of substance use is what makes the connection credible;
"they've been there".
- *Maps Table 4 rows: "Peer has lived experience"*

### A5 `liaison_advocacy` — Acts as a go-between and advocates for patients
Relays patient concerns to staff and staff plans to patients; speaks up for
patient needs during the ED visit.
- *Maps Table 4 rows: "Peer liaises between patient and provider", "Peer advocates for pt's"*

### A6 `finds_monitors_patients` — Keeps an eye on and finds patients
Checking the waiting room and triage, noticing who is struggling, locating
patients who have wandered off, proactive rounding.
- *Maps Table 4 rows: "Peer can keep an eye on and find patients", "Peer supports in triage", "Peers are proactive"*

### A7 `prevents_escalation` — Calms situations before they escalate
De-escalation, defusing conflict, an alternative to security, keeping the
patient and the department safe.
- *Maps Table 4 rows: "Peer prevents escalation", "Peer promotes pt safety"*

### A8 `harm_reduction` — Supports harm reduction
Harm reduction supplies and education, naloxone, safer use conversations.
- *Maps Table 4 rows: "Peer supports with providing harm reduction resources"*

### A9 `connects_to_resources` — Connects patients to services beyond the ED
Detox, treatment, social work, housing, community supports; discharge planning
and follow-up.
- *Maps Table 4 rows: "Peer connects patients to other resources", "Peer supports with discharge"*

### A10 `alerts_responds` — Alerts staff and responds to calls
Flags changes in a patient's condition to clinical staff; comes when paged;
debriefs with staff afterwards.
- *Maps Table 4 rows: "Peer alerts HCP staff to changes in patient condition", "Peer responds to calls from staff", "Peer debriefs with staff"*

---

## Domain B — What difference the peer program makes

### B1 `patient_engagement` — Patients engage more with their care
- *Maps Table 4 rows: "Peer program promotes patient engagement in care"*

### B2 `willingness_to_stay` — Patients are more willing to stay for care
Reduced leaving against medical advice; patience through long waits.
- *Maps Table 4 rows: "Peer program increases patient's willingness to stay"*

### B3 `patient_experience` — Patients have a better experience in the ED
- *Maps Table 4 rows: "Peer program provides a positive experience", "Peer fosters positive experience in the ED"*

### B4 `benefits_staff` — Frees up staff and helps the team
The peer absorbs work clinical staff have no time for; improves patient flow;
makes the team's job easier.
- *Maps Table 4 rows: "Peer role benefits HCP staff", "Peers improve patient flow in the ED"*

### B5 `changes_staff_practice` — Changes how staff themselves treat patients
Role-modelling; humanising patients to staff; prompting reflection on bias and
stigma; promoting dignity and respect; educating staff.
- *Maps Table 4 rows: "Peer helps humanize patients to staff", "Peer educates staff", "Peer promotes patient dignity and respect"*

### B6 `meets_patient_needs` — Peer services meet patients' needs
- *Maps Table 4 rows: "Peer services meet the needs of patients with substance use disorder"*

### B7 `no_change_observed` — No change noticed, or cannot say
Explicit statements of no observed effect or inability to judge. Coded
deliberately: framework §8 warns against reading absence of a theme as absence
of evidence.
- *Maps Table 4 rows: "Peer program has not influenced HCP's interactions...", "Cannot say whether..."*

---

## Domain C — Problems, gaps and confusion

### C1 `role_scope_unclear` — Unclear what the peer's role and limits are
What the peer is and is not allowed to do, scope of practice, what may fairly be
asked of them.
- *Maps Table 4 rows: "Peer's role and scope of practice are unclear"*

### C2 `availability_gaps` — Peers are not always available
Limited hours, no overnight or weekend cover, too few peers, single point of
failure.
- *Maps Table 4 rows: "Peer services are not always available"*

### C3 `referral_confusion` — Staff are unsure how to reach or refer to a peer
- *Maps Table 4 rows: "HCP confusion around how they can refer patients to peers"*

### C4 `overlap_other_teams` — Overlap with harm reduction nurses, social work, CPAS
Unclear boundaries between the peer program and other teams doing adjacent work;
unclear reporting lines.
- *Maps Table 4 rows: "HCP confusion around role of peer versus HRN", "Role delineation between peer and social work..."*

### C5 `documentation_access` — Charting, Cerner access and documentation gaps
Peers cannot chart, lack system access, or it is unclear how encounters are
recorded; miscommunication about the patient's plan.
- *Maps Table 4 rows: "HCP confusion around how peer documents encounters", "Peers need appropriate amount of access to Cerner"*

### C6 `training_gaps_uncertainty` — Uncertainty about what training peers have
- *Maps Table 4 rows: "HCP confusion around training peer receives"*

### C7 `burden_or_challenge` — A challenge, or added burden on staff
Explicit reports of difficulty working with a peer, or of the peer adding to
staff workload. Rare but important.
- *Maps Table 4 rows: "HCP has experienced challenges working with peer", "Peer increases burden on HCP staff"*

---

## Domain D — Suggestions for integration and workflow

### D1 `join_handover_huddle` — Include peers in handover and huddles
- *Maps Table 4 rows: "Peers should join shift handover/huddle"*

### D2 `integrate_into_team` — Treat peers as part of the team
Invite to education days and debriefs, shadow shifts, mass emails, recognition
as equally important, getting staff buy-in.
- *Maps Table 4 rows: "Invite peers to education days and debriefs", "Peers should be better integrated with HCPs", "Make peers feel part of the team"*

### D3 `expand_program` — Expand the program beyond the ED
Other departments, other hospitals, community outreach.
- *Maps Table 4 rows: "Peer program should be expanded beyond the ED", "...beyond the hospital"*

### D4 `extend_hours` — More coverage: 24/7, overnight, more peers per shift
- *Maps Table 4 rows: "Peer program should be offered 24/7", "Peers should work overnight"*

### D5 `improve_referral_pathway` — Build a clear referral and contact route
A referral system in Cerner, a way to flag patients, a known way to page a peer.
- *Maps Table 4 rows: "Improve referral system for peers in Cerner", "Optimize workflows between peers and harm reduction nurses"*

### D6 `clarify_role_formally` — Formally define and communicate the peer role
Written scope, orientation for staff, making all staff aware the program exists.
- *Maps Table 4 rows: "Peer role should be better clarified", "Make all HCPs aware of peer program"*

---

## Domain E — Training and long-term sustainability

### E1 `training_for_peers` — Training peers would benefit from
Naloxone, verbal de-escalation, hospital workflow, communication, BLS, medical
jargon, cultural safety.
- *Maps Table 4 rows: "Naloxone training", "Communication classes", "General hospital workflow"*

### E2 `training_for_staff` — Training staff need to work well with peers
What the peer role is and how to engage it; trauma-informed practice;
de-stigmatising and addressing bias.
- *Maps Table 4 rows: "Peer roles and methods to engage with them", "De-stigmatizing, addressing bias"*

### E3 `hire_more_peers` — Hire more peers
- *Maps Table 4 rows: "Hire more peers"*

### E4 `peer_wellbeing` — Protect peers' wellbeing
Burnout, emotional labour, supervision, support structures, boundaries,
re-traumatisation, hiring people who are a good fit.
- *Maps Table 4 rows: "Promote peer wellbeing", "Manage peer burnout", "Peers don't have supervision in the ED"*

### E5 `sustain_program` — The program should continue
Sees benefit in sustaining it; continue in current capacity; funding; measuring
impact; feedback processes; proper reporting structure.
- *Maps Table 4 rows: "HCP sees benefit in sustaining the peer program", "Continue in current capacity", "Measure indicators of peer impact"*

---

## Stance field

Each coded utterance also carries one `stance` toward the peer program:

| Value | Meaning |
|---|---|
| `positive` | endorses, praises, reports benefit |
| `negative` | criticises, reports harm, problem or barrier |
| `mixed` | both at once, or endorses with a reservation |
| `neutral` | descriptive, no evaluation |
| `na` | not about the program (logistics, consent, backchannel) |

This is stance **toward the program**, deliberately a different construct from
`tx_valence` (sentiment of the wording) and `ac_valence` (vocal affect).
Framework §4.3 exploits mismatches between the three.

---

## Coding rules

1. Code **participant** utterances only; interviewer turns are context.
2. Multi-label is expected — assign every code that clearly applies.
3. Assign no code rather than a doubtful one; mark `confidence: low` when torn.
4. `key_phrase` must be a verbatim substring of the utterance, ≤20 words.
5. Never infer from demographics. Framework §8 lists this as the trap that makes
   later group comparison circular, and RQ3 is exactly such a comparison, so
   demographics are withheld from the coding prompt entirely.
