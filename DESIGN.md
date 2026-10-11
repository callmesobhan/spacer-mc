# spacer-mc: design

Monte Carlo uncertainty propagation for hydrogel rectal-spacer dose sparing in prostate radiotherapy:
photon external beam, LDR brachytherapy and HDR brachytherapy.

> Research and teaching code, built only from published aggregate data. Not validated for clinical decisions.
> No patient data belongs in this repo.

## 0. Decisions and status

| Question | Decision | Consequence |
|---|---|---|
| Source of dose-gradient data | **Published papers only: the 57 articles in the author's local `DATA_Articles` collection (not included in this repo), plus any added later** | No dose grids. Calibrate to reported aggregates and test on held-out studies (section 6). Cited by DOI in [refs/references.bib](refs/references.bib); PDFs are never committed |
| Audience | **Portfolio, but robust** | Fewer features, higher bar. Traceability and honest limits come before coverage (section 9) |
| Modalities | **Photon EBRT, LDR and HDR brachytherapy. Protons deferred** | Two kinds of dose engine, three regimes. Protons out of scope; the radiobiology here must not be used for them |
| Order | **EBRT, then LDR, then HDR** | HDR is the riskiest and the first cut if scope shrinks |

Evidence status ([data/priors.yaml](data/priors.yaml), [docs/SOURCES.md](docs/SOURCES.md), [docs/PAPERS_NEEDED.md](docs/PAPERS_NEEDED.md)):

- 73 articles catalogued with DOIs: the author's 57, 15 free papers retrieved separately (read only; no copies are kept here), and Hamstra 2018, supplied by the author.
  **32 have had their results read, 12 partly, 29 are catalogued only.**
- 92 evidence rows: **68 verified** (26 from tables or figures, 42 from the text), 6 matched by string only, 16 abstract-only, 2 derived here and labelled as such. Rows taken from a
  review's compilation are marked `secondary`.
- Integrity is tested: every row cites a bibliography key, `verified: true` is only allowed when a table or the text was read, and the
  pooling code reproduces a published meta-analysis from its forest-plot values.
- **16 papers this project relies on are still second-hand or abstract-only**, listed with full citations in [docs/PAPERS_NEEDED.md](docs/PAPERS_NEEDED.md). Four are Tier 1 (useful, none blocking); twelve
  are optional, each with a substitute already in hand. None is free to read.

## 1. Problem

A rectal spacer (PEG or hyaluronic-acid hydrogel, or a balloon, between prostate and rectum) is judged on one
post-injection scan. Delivered rectal dose also depends on what that scan cannot show: how much separation was
achieved and how evenly, where the rectum sits each day, whether the spacer changes during treatment, and, for
brachytherapy, how source decay interacts with spacer lifetime.

**Central question.** For a given regime and spacer material, what is the *distribution* of rectal dose with and
without a spacer, and what achieved separation gives 95% confidence of meeting a rectal constraint?

**The hypothesis that makes this more than a re-run of the trials.** The pivotal trial shows a PEG spacer is stable for
about 10-12 weeks and then loses separation steeply: 10.9 +/- 5.8 mm imaged the week before month 3, 6.8 +/- 5.4 mm the
week after (Mariados 2015). Meanwhile isotopes deliver dose on very different clocks. From decay alone (derived, checked
numerically):

| Isotope (half-life from King 2021, Table 1) | Half-life | Share of total dose by day 90 |
|---|---|---|
| Cs-131 | 9.7 d | 99.8% |
| Pd-103 | 17.0 d | 97.4% |
| I-125 | 59.4 d | **65%** (80% not until ~day 138) |

So a PEG spacer plausibly covers nearly all of a Pd-103 or Cs-131 implant and only about two thirds of an I-125 implant. The
three isotopes make a built-in contrast, and prescriptions differ too (monotherapy I-125 144-145 Gy, Pd-103 125 Gy, Cs-131 115 Gy;
combination boosts lower). Published LDR
studies measure dose on implant day or day 30 and cannot see this. Choi 2026 states that over 80% of I-125 dose is delivered
within 3 months, which does not match decay physics; this is logged as a discrepancy, not adopted. A null result (the
loss is too gradual or too small to matter) is a valid finding.

A second time effect cuts across the spacer's: in LDR *without* a spacer, rectal D2cc rose from 77% (day 1) to 88% (day 30)
(Haddad 2023), which the authors attribute to changing prostate oedema displacing the sources. That is a sizeable shift in
rectal dose with no spacer involved, so it has to be modelled.

## 2. What "Monte Carlo" means here

| Layer | Method | Role |
|---|---|---|
| Uncertainty propagation | Sample anatomy, placement, motion, kinetics and model parameters; push each sample through a dose model | **The core.** Answers the central question |
| Particle transport | Geant4-based engines (OpenTOPAS, GATE, egs_brachy) | Optional cross-check of the analytic engines. Not required for v1 |

A hydrogel spacer is nearly water-equivalent. It lowers rectal dose by **moving the rectum into a lower-dose region**, not
by attenuating radiation, so geometry and gradient drive the result, not transport.

## 3. Three regimes, three engines

| | EBRT (IMRT, VMAT, SBRT) | LDR (I-125, Pd-103 permanent seeds) | HDR (Ir-192, 1-2 fractions) |
|---|---|---|---|
| Dose engine | Gradient model calibrated to paired DVHs | **TG-43** on a template seed layout; seeds fixed to the prostate, rectum moves | **TG-43** on a template dwell plan plus a simplified re-optimisation |
| What the spacer does | Moves the rectum down the gradient | Same. After-implant injection is a pure shift; before-implant also deforms the prostate (Morita 2020) | Same, and the plan is re-optimised, which may also help dorsal coverage. Evidence conflicts (Haddad vs Honda) |
| Time | 5-44 fx over 1-9 wk; PEG flat to ~12 wk | ~1 year of decay vs a spacer lasting ~3 months; oedema resolves over the first month | Irrelevant |
| Radiobiology | EQD2 then LKB NTCP | **Dosimetric endpoints only** (D1cc, D2cc, RV100). LKB does not transfer | EQD2 at high dose per fraction (LQ caveat); dosimetric endpoints first |
| Calibrate on | Whalley 2016, Hamstra 2017, Miller 2020 | Teyateeti 2022, Choi 2026 (plus abstract-only cohorts) | Held out: Haddad 2023, Honda 2025 (plus abstract-only) |

## 4. Architecture

```
Scenario (YAML) -> sampler -> geometry -> dose engine -> DVH / D_xcc -> endpoints
                     |                        ^
                kinetics(t) ----- thickness(t) /  + source decay + oedema (LDR)
```

```python
class DoseEngine(Protocol):
    def dose_at(self, points_mm: np.ndarray, t_days: float = 0.0) -> np.ndarray: ...
```

Backends in build order: `AnalyticFalloff` (done); `TG43` (point/line source, consensus source data to be pinned and cited);
`DoseGrid` (any regular grid); `TransportGrid` (an external transport run loaded as a grid).

## 5. Uncertainty model

Inputs live in [data/priors.yaml](data/priors.yaml). Four levels:

- **Study level.** Miller 2020 reports I^2 = 97% for V70, but **almost all of that comes from one study**. Chao 2019's entry (-1.1) is an HDR
  value in percent of prescription, not an EBRT Gy value. Re-pooling the other five EBRT studies (this repo's analysis; the code reproduces Miller's
  published numbers first) gives -7.90 (95% CI -9.58 to -6.23) with I^2 = 50% and tau about 1.3 percentage points: moderate heterogeneity,
  not near-total. Even those five are at different thresholds, because Miller took 'the reported value closest to V70': te Velde's is V75
  (7.4% to 2.1%) and Whalley's is V65 against a control cohort (13.5% to 5.3%). Kwon 2026's larger meta-analysis (35 studies) pools mL and %,
  EBRT and brachytherapy, and shows I^2 of 95-99% for that reason; it also appears to contain an extraction error (a 'te Velde 2017' row that is
  Taggar 2018's V100 values). Pooled values from reviews are inputs to check, not facts. The metric-free summary is the **relative V70
  reduction by study, which runs from 25% to 80%**. In LDR the spread is large: median D2cc about 20% in Pd-103-dominated cohorts (Taggar 20.5%, Teyateeti 19.9%) and 45.7% in
  an I-125 cohort (Lin) at similar separation. Model a study-level random effect, with isotope and prescription type as covariates.
- **Patient level.** Separation and its cranio-caudal profile (spacer base/mid/apex 1.6/1.2/1.0 cm vs 0.9/0.3/0.2 cm without);
  asymmetry; baseline gap; systematic prostate shift (posterior > 6.5 mm in 27% of controls, 0% with spacer). Placement failures
  give a real tail: 3 of 149 patients had a *worse* rV70 with the spacer, and in another series 8.8% of PEG placements left no
  spacer at the apex (Ben-Dor 2025).
- **Placement quality is a centre effect.** Perfect symmetry was 49.0% in Fischer-Valuck (MRI-scored, trial centres) and 14.7% for PEG
  in Ben-Dor (CT-scored, single centre), while severe asymmetry was similar (20.1% vs 23.5%). Model the symmetry distribution as a
  study-level random effect, not a constant. **Represent the spacer as a per-location thickness profile, not a symmetry category:** in 822 patients
  (Prusty 2025) a thickness-based quality score predicted rectal dose at every level, while the Fischer-Valuck symmetry score predicted none, and
  the thickness score was also scored more reproducibly (kappa 0.83-0.91 vs 0.61).
- **Fraction / time level.** Daily motion, heavy-tailed (Polizzi 2024: 77% of fractions within +/-5 mm of the patient's average
  prostate-rectum distance and 3.6% beyond 10 mm; a Student-t with 6 degrees of freedom and 3.7 mm scale matches both numbers, a Gaussian
  underestimates the tail by half). Kinetics are **material-specific**: PEG is flat with apparent swelling (+10-20% volume) to ~12 wk,
  then loses separation steeply around week 12-14, is back near baseline by ~5 months (Uhl 2014: 10.5 mm at end of an 8-week course, 2.9 mm three months later) and is gone on MRI by 6-12 months; **HA falls 28-32% in thickness within the first month and
  then plateaus** (Björeland 2023), which halves its dose benefit (rectal V70 reduction 56% at planning, 31% from mid-treatment). Oedema
  decay matters for LDR. Onset of loss is a distribution, not a date: in four patients with LDR plus EBRT (Kubo 2021) two had lost most separation by day 93-99 while two had not (20.7 mm at day 76, 11.8 mm at day 74). Track rectal *separation*, not gel volume: with DuraSeal in LDR the gel volume half-life was 47 d but the separation half-life 110 d, because oedema persists (Heikkila 2014, quoted in Vaggers 2021; primary paper not yet read).
- **Model level.** LKB parameters. Published fits disagree (TD50 68.5-81.9 Gy, n 0.07-0.26) with modest predictive power.

Daily-motion priors are now better supported but none of the daily data include spacers. Polizzi 2024 gives the separation-noise
distribution; Hedrick 2017 and Cuccia 2020 bound intrafraction motion with a spacer (3D shift > 5 mm in 2.5% of fields; AP SD under 1 mm
on MR-Linac). Two papers give **validation targets**: delivered rectal dose runs above planned even with daily image guidance, by
7.1 +/- 7.6% (IMRT) and 5.0 +/- 7.4% (SBRT) of DVH (Patrick 2024) and by a median daily/planned rectal Dmean ratio of 1.16 (IQR 1.02-1.31)
(Petrikowski 2025). A motion model that cannot reproduce that upward drift in the no-spacer arm is wrong. Picardi 2016 (the only paper
reporting systematic and random errors with and without a spacer) is paywalled and still abstract-only.

## 6. Calibration and validation from published data only

1. **Calibrate** free dose-model parameters so simulated aggregates match published ones, with a study-level random effect.
   - *EBRT, separation curve.* **Susil 2010** gives the first target: nine real IMRT plans (78 Gy) with the rectum shifted away from a fixed dose cloud, mean rectal V70 23.0% at
     0 mm, 15.1% at 5 mm, 3.7% at 10 mm and 0.0% at 15 mm, saturating beyond 15 mm. This is the model's own shift approach, so it is a like-for-like check.
   - *EBRT, paired DVHs.* Fit the gradient (penumbra, 50% position, and a low-dose tail term) to the **9-point paired DVH of Whalley 2016**
     (V30-V82; n = 30; mean separation gain 10.5 mm), the 3-point DVH of Hamstra 2017 (V50/V70/V80, 21/10/4% vs 10/2/0.1%) and **Song 2013's paired DVH from V10 to V75 (n = 45, with
     patient-level SDs)**. Song also shows baselines differ by centre (p = 0.002) while the reduction does not (p = 0.31), so include a centre term on the baseline, not on the effect.
     The low-dose end is a deliberate test of the model: V30 falls only from 49.4% to 42.7%. A pure Gaussian falloff has a thin
     tail (the M0 toy gives ~0.5% of prescription at the wall behind a 10 mm spacer), so it is expected to miss this and the
     model needs a tail term. Whether it does is untested until M2.
   - *LDR.* Rectal D1cc/D2cc against separation using Teyateeti 2022 (n = 363, medians with ranges) and Choi 2026, with isotope
     (Pd-103 vs I-125) and prescription (monotherapy vs boost) as covariates, since King 2021 shows they differ by isotope and use.
   - *SBRT ceiling.* Hwang 2018 (36.25 Gy in 5 fx, n = 20): a single separation measurement explains only 44-60% of rectal dose variance (27% for V80). A geometry-only model
     that explains much more than that is overfitting.
   - *SBRT.* Grossman 2023 (45 Gy in 5 fx, n = 42) gives a geometry-to-dose slope: a quality score of 1 instead of 2 raised rectal
     D1cc by 798 cGy (95% CI 347-1249).
   - *Motion.* The delivered-versus-planned rectal dose is a **range, not one number**: about zero when dose is accumulated over fractions (Bostel 2019:
     Dmean +0.14 +/- 3.73 Gy, not significant) up to +5 to +7% of DVH per fraction (Patrick 2024) and a median daily/planned Dmean ratio of 1.16
     (Petrikowski 2025). The model should fall inside that range and say which definition it matches.
2. **Hold out whole studies, never patients.**
   - *EBRT:* leave-one-study-out over the **five EBRT** per-study V70 differences in Miller 2020's forest plot (Chao 2019 excluded: HDR metric);
     the 2.7% failure tail (Fischer-Valuck 2017); Mariados 2015 as a separate check on the pivotal-trial protocol; Björeland 2023's
     HA time course (V70 reduction 56% -> 31%) as a check on the kinetics model.
   - *LDR:* leave-one-study-out across cohorts. *HDR:* Haddad 2023 and Honda 2025 held out entirely.
   Report agreement as predictive-interval coverage: does the held-out value fall inside the 90% interval, and how often?
3. **Report failures** with the likely reason.
4. **Toxicity is a usable but limited check.** Wong 2025 (17 studies, 3,200 men) gives late grade >=2 rectal toxicity RR 0.25 (95% CI 0.15-0.42),
   1.62% vs 9.35%: the NTCP layer's predicted late risk ratio should fall inside that interval. The older Miller 2020 estimate (RR 0.23, 0.06-0.99)
   is too wide to constrain anything. LKB parameter sets disagree, so this check mainly rules out parameter sets, not the geometry.
5. **Spread, not just means.** The pivotal trial gives patient-level SDs for the spacer group (Mariados 2015, Table 2, n = 148): rectal V70 12.4 +/- 5.4% before and 3.3 +/- 3.2% after,
   V50 25.7 +/- 11.1 to 12.2 +/- 8.7, V80 4.6 +/- 3.1 to 0.6 +/- 0.9. Song 2013 gives more (V70 13.0 +/- 6.2 to 5.1 +/- 4.2, n = 45). The relative reduction rises with dose
   (52% at V50, 86% at V80). The Monte Carlo should reproduce these spreads and this profile, since a model can fit every mean and still get the tail wrong. (A second-hand
   'Hamstra 2018' SD in Kwon 2026 turned out to be a standard error read off a chart and converted; it is not used.)

Limits, stated before any result: paired before/after plans are each re-optimised, so they mix geometry with planner behaviour. The authors of the
pivotal-trial secondary analysis (Hamstra 2018) note the spacer also lowered penile bulb dose (21 vs 11 Gy) and suggest the optimiser, freed from the rectal goal, served secondary
goals. Susil's shift-only simulation (no re-optimisation) gives 83% V70 reduction at 10 mm against 61-79% in the trials' re-optimised plans, which is consistent with part of the geometric
gain being spent elsewhere (a hypothesis of this project, not a finding). The model keeps a geometry term, calibrated on Susil, and a separate planner term; aggregates lose patient-level correlation; pivotal-trial papers share patients; cohorts mix
techniques; LDR plans may anticipate the spacer (Choi: 79% met a rectal constraint before the spacer vs 100% after), so a
"pre-spacer" dose is not a clean counterfactual; and morphology explains little of LDR dose variation (R^2 = 0.17).

## 7. Estimands

- EBRT: mean rectal-wall dose; V30-V82; P(meeting the Pederson tiers V70 <=10%, V65 <=20%, V40 <=40%); LKB NTCP and the paired reduction.
  The tiers are informative: Whalley's paired plans cross V70 <=10% (12.3% without a spacer, 3.7% with).
- LDR: D1cc, D2cc, RV100 against time; the share of total dose delivered while the spacer is present.
- HDR: D2cc, V75, and prostate V100 / D90.
- **Brachytherapy: lead with dose headroom; use pass/fail only where it discriminates.** The guideline objectives (Henry 2022: rectum
  D2cc <=145 Gy for I-125 monotherapy, i.e. 100% of prescription; <=10 Gy for a 15 Gy HDR boost) are far above observed doses (LDR D2cc
  19.9-47.0% in Teyateeti; HDR 7.4-9.2 Gy in Beaudry's abstract), so they are met with or without a spacer. The tighter volume constraints
  do discriminate: RTOG 0321 rectal V75 < 1 cc was missed by 6.2% (4/65) without a spacer and by none with one in HDR plus EBRT (Chao 2019),
  but is trivially met in salvage HDR (median 0.06 cm3, Lee 2023). Choose the constraint per setting.
- All: minimum separation for 95% confidence of meeting each constraint; Sobol indices; Monte Carlo standard error and a
  convergence plot for every reported number.

## 8. Numerics

- **Seeding.** One `SeedSequence` spawned per study / patient / fraction; reproducible regardless of worker count.
- **Paired arms** share sampled offsets and motion, which guarantees spacer dose <= no-spacer dose per patient (tested). It is
  *not* a big variance win: in the M0 toy the paired standard error was 0.00042 vs 0.00044 unpaired.
- **Quasi-Monte Carlo** (`scipy.stats.qmc.Sobol`) compared against plain MC. **Stopping rule:** CI half-width below tolerance.

## 9. Engineering bar ("robust")

- **Traceability, enforced.** [tests/test_priors.py](tests/test_priors.py) checks every row cites a real bibliography key and that
  `verified: true` is only used for rows read from a table or the text. Mutation-checked: a fake key and a promoted
  abstract-only row both fail the suite.
- **Reading policy.** A number is not "verified" because it appears in an abstract or matches a string in the full text. Context
  must be read. Tables and figures are rendered and read, because text extraction scrambles them (it did for the forest plot).
- **Reproducibility.** Pinned environment; `make repro` rebuilds every figure from a clean clone; output hashes in CI.
- **Tests.** Closed-form benchmarks, property tests (`hypothesis`), golden regression outputs, a test per unit conversion.
- **Static checks and CI.** `ruff`, `pyright`; GitHub Actions on Python 3.11-3.13, macOS and Linux.
- **Documents.** README leading with limitations; `MODEL_CARD.md`; `METHODS.md`; `CITATION.cff`; `SECURITY.md`; a licence (MIT).
- **No patient data.** No PDFs, figures or copied text; extracted numbers with citations only.

## 10. Layout

```
spacer-mc/
  DESIGN.md  MODEL_CARD.md  METHODS.md  README.md  CITATION.cff
  refs/references.bib        57 articles, DOIs from Crossref
  docs/SOURCES.md            every source, what it is used for, how deeply it was read
  data/priors.yaml           48 evidence rows
  configs/                   one YAML per scenario (regime, material, schedule)
  src/spacer_mc/
    dose.py        AnalyticFalloff [done]  TG43 [M3]  DoseGrid [later]
    sampling.py    1-D sampler [done]
    geometry.py    3-D prostate / rectum / spacer profile [M2]
    kinetics.py    three-phase thickness(t); oedema; decay [M4]
    radbio.py      DVH, EQD2, LKB (EBRT only) [M2]
    calibrate.py   random-effects fit, leave-one-study-out [M2, M3]
    analysis.py    estimands, Sobol, convergence [M5]
  tests/  notebooks/  validation_report/
```

## 11. Milestones and releases

| | Deliverable | Done when |
|---|---|---|
| M0 | 1-D sampler and analytic benchmark | 4 tests pass (**done**) |
| M1 | Evidence base | 73 DOIs and bibliography (**done**); evidence table with integrity tests (**done**); everything in the local collection worth reading, the 15 free papers and Hamstra 2018 read (**done**); **still to do:** the 16 papers in `docs/PAPERS_NEEDED.md`; Tier 1 are Heikkila 2014, Beaudry 2025, Kahn 2020 and Agrawal 2026, none blocking |
| M2 | **v0.1 EBRT**: 3-D geometry with profile and asymmetry, DVH with a tail, LKB; calibrate to Whalley and Hamstra | Calibrated model, leave-one-study-out over the six V70 studies, honest hold-out report |
| M3 | **v0.2 LDR**: TG-43 engine; calibrate on LDR cohorts | Leave-one-study-out report with isotope as covariate |
| M4 | Time dimension: three-phase kinetics, oedema, decay x spacer lifetime; EBRT schedules by material | Share-of-dose-protected curve for Pd-103 vs I-125, including a null if that is the answer |
| M5 | Sensitivity and required-separation curves | Ranked inputs; 95%-confidence separation per regime |
| M6 | **v0.3 HDR**: template plan and re-optimisation, held out against Haddad / Honda | Report with failures shown |
| M7 | Hardening: model card, CI matrix, `make repro`, write-up | A clean clone reproduces every figure |

If time runs short, cut M6.

## 12. Risks

1. **Second-hand and abstract-only rows.** Rows from reviews are marked `secondary`, and several LDR and HDR cohorts (Kahn, Agrawal, Morita, Nehlsen, Beaudry, Wu) are still abstract-level.
   They can inform priors but should not drive a headline claim until the full texts are read.
2. **Calibrating on aggregates** can fit the average and still be wrong for an individual. State it.
3. **HDR re-optimisation** cannot be recovered from published aggregates; the model may look plausible and be wrong.
4. **LKB transferability.** Fitted mostly to 3D-CRT-era data; the QUANTEC authors warn IMRT may differ.
5. **A published meta-analysis can mix metrics.** Miller 2020's pooled V70 includes an HDR percent-of-prescription value; the code now
   reproduces its numbers so the effect of excluding it is checkable. Treat pooled values from reviews as inputs to check, not facts.
   Other disagreements are real, not data errors: Haddad vs Honda on HDR coverage; Choi's ">80% in 3 months" vs physics; the
   abstract (47.7%) vs body (49.0%) symmetric-placement count in Fischer-Valuck. Report these, do not average them away.
6. **Funding and conflicts.** The pivotal-trial papers read here (Mariados 2015, Hamstra 2017, Fischer-Valuck 2017) report research
   funding from the device maker (Augmenix), and some authors hold shares or receive honoraria. Several other cohorts are
   single-institution. Weight them accordingly and say so in the write-up.

## 13. Open items

1. **The 16 papers in [docs/PAPERS_NEEDED.md](docs/PAPERS_NEEDED.md)** need institutional access (requested). Four are Tier 1: Heikkila 2014 (separation half-life), Beaudry 2025 (HDR
   separation sweep), Kahn 2020 and Agrawal 2026 (LDR). None blocks anything. The other twelve are optional, each with a substitute already in hand. The repo will be updated
   when they arrive.
2. **EBRT is ready to calibrate.** Susil 2010 (separation curve), Mariados 2015 Table 2, Song 2013 and Whalley 2016 (paired DVHs with patient-level SDs), and Hamstra 2017 give targets for the gradient, the
   tail and the spread. Next step: milestone M2. LDR and HDR are the parts still waiting on the Tier 1 papers.
3. **The 29 catalogued-only papers** in the local collection are safety reports, case reports, quality-of-life studies and guidelines. No dose data is expected in them.
