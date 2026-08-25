# Elicit Prompt Set: Tile Grid Phase as an Uncontrolled Variable

Built 2026-08-25 for the frame A paper. Supersedes
`claude/elicit-prompt-set-lit-review.md`, which was written for the abandoned
three arm comparison and targets a different site and a different question.

**Target: 50 to 70 papers starred across twelve searches, 20 to 30 cited.**
Roughly 5 per search. Related Work currently cites three.

## How to use each block

1. Paste the QUERY into Elicit's Find Papers search.
2. Add each line under COLUMNS as a custom extraction column.
3. Screen on the columns, not the abstracts. Star anything that yields a number,
   a protocol, or a stated absence you can cite.
4. Export starred results to BibTeX at the end of each search.

## Run order, and why it is not the numbered order

**Run 1, 2 and 3 first.** Prompt 1 tests whether the paper's novelty claim
survives contact with the literature. Prompts 2 and 3 establish that the
procedure being criticised is widely used and that its configuration goes
unreported. If prompt 1 turns up a paper that already varies tile offset and
measures identity churn, stop and tell me before drafting Related Work.

Prompts 4 through 8 are the theoretical backbone. Prompts 9 through 12 are
domain grounding and can run last.

---

## 1. Has anyone already varied tile position?

Backs the novelty claim. This is the prompt that could kill the paper, so run
it first and screen it hardest.

QUERY

```
Does the position or offset of the tiling grid used for patch based inference on large images affect which objects a detector finds, and has any study varied tile origin while holding tile size and overlap fixed?
```

COLUMNS

```
Was tile position or grid offset varied as an experimental variable
Were tile size and overlap held fixed while position varied
What was measured (aggregate count, per object identity, both)
Number of grid positions tested
Reported variation in detection count
Reported variation in detection identity or set membership
Domain and imagery type
Whether the effect was framed as a problem or as an augmentation
```

**Screen:** star anything that varies tile origin at all, even as test time
augmentation rather than as a sensitivity study. Those are the closest prior
work and must be cited whether or not they framed it as we do.

---

## 2. What tiling configuration do detection papers actually report?

Backs the claim that grid offset is undisclosed. A stated absence is citable
evidence if you can show it across a body of papers.

QUERY

```
What preprocessing and inference configuration details do object detection papers on large aerial, satellite, or remote sensing images report when they use patch based or sliding window inference?
```

COLUMNS

```
Tile or patch size reported
Overlap or stride reported
Tile origin or grid offset reported
Merging rule across tiles reported (NMS, WBF, other)
Merging threshold value reported
Whether padding or edge handling at image borders is described
Whether code is released
Venue and year
```

**Screen:** star papers that report tile size and overlap but not origin. That
combination is the evidence. Keep a tally: how many of N report origin. That
tally is a sentence in the Introduction.

---

## 3. DeepForest: usage base, released weights, documented behaviour

Establishes that the procedure under criticism is widely used, and gets the
citations for the model itself.

QUERY

```
How is the DeepForest pretrained individual tree crown detection model used, configured, and evaluated across published studies, and what inference settings do those studies report?
```

COLUMNS

```
DeepForest version or checkpoint used
Whether predict_tile or a custom tiler was used
Patch size and overlap reported
Score threshold reported
Imagery source, sensor, and ground sample distance
Forest or landscape type
Reported precision, recall, F1, and the IoU threshold used
Number of annotated reference crowns
Stated failure modes
```

**Screen:** any paper reporting a metric without stating its IoU threshold is
not citable as a comparison point. Note it as a usage instance and move on.

---

## 4. Detection stability, consistency, and repeatability metrics

Backs claim 2 and extends the three papers already cited. Miller 2019, Zhang and
Wang 2016 and Tung 2022 should reappear here; the point is to find what came
after them.

QUERY

```
How is the stability, consistency, or repeatability of object detector outputs measured across repeated runs, perturbed inputs, or successive video frames, and how well do those measures correlate with accuracy?
```

COLUMNS

```
Perturbation or repetition source (video frames, dropout, augmentation, seeds, tiling)
Stability or consistency metric defined
How correspondence between runs is established (IoU, tracking, clustering)
Correlation between the stability metric and an accuracy metric
Sign and magnitude of that correlation
Whether instability was attributed to a specific mechanism
Dataset and domain
Venue and year
```

**Screen:** the correlation column is the one that matters. Our claim 4 is a
second instance of a stability and accuracy decoupling, so any paper reporting
that correlation is directly citable.

---

## 5. Merging and consensus across multiple detection passes

Backs claim 4 from the other side. Test time augmentation and ensembling assume
consensus improves quality. Our null says it does not, at least for this
perturbation.

QUERY

```
How are detections merged or fused across multiple inference passes, augmentations, or ensemble members, and does requiring agreement across passes improve detection precision?
```

COLUMNS

```
Merging method (NMS, soft NMS, weighted box fusion, clustering, voting)
Number of passes or ensemble members
Source of variation between passes
Whether a consensus or vote threshold was applied
Change in precision from requiring consensus
Change in recall from requiring consensus
Whether agreement was validated against ground truth or assumed
Compute cost of the multi pass approach
```

**Screen:** star anything that treats agreement across passes as a confidence
proxy. That assumption is what our null contradicts.

---

## 6. Edge, seam, and boundary artefacts in patch based inference

Backs claim 3. Seam pinning is a boundary artefact, and the segmentation
literature has a longer history with these than detection does.

QUERY

```
What artefacts arise at patch boundaries when deep learning models are applied to large images in tiles, and what mitigations such as overlap, cropping, padding, or blending are used to remove them?
```

COLUMNS

```
Task (detection, semantic segmentation, instance segmentation, regression)
Artefact described (discontinuity, seam line, truncated object, duplicate, other)
Whether the artefact was quantified or only described
Magnitude of the artefact if quantified
Mitigation applied
Residual artefact after mitigation
Whether overlap alone was shown to be sufficient
Imagery type and tile size
```

**Screen:** distinguish papers that mitigate blindly from papers that measure
the artefact first. The measuring ones are the citable ones.

---

## 7. Reproducibility and nondeterminism in deep learning pipelines

Positions the contribution as a reproducibility finding rather than an ecology
finding, which is what frame A requires.

QUERY

```
What sources of run to run variation affect deep learning model outputs at inference time, and how much of reported performance variation is attributable to undisclosed preprocessing or configuration choices rather than to the model?
```

COLUMNS

```
Source of variation studied (seed, hardware, library version, preprocessing, hyperparameter)
Whether the source is disclosed in typical papers
Magnitude of the resulting performance variation
Metric affected
Whether aggregate metrics concealed the variation
Proposed reporting standard or checklist
Domain
Venue and year
```

**Screen:** star anything proposing a reporting standard. Our conclusion is a
reporting recommendation, and it is stronger if it joins an existing line of
argument rather than inventing one.

---

## 8. Researcher degrees of freedom and specification sensitivity

The methodological frame. This is the literature that gives "an undisclosed
practitioner controlled parameter changes the result" its name.

QUERY

```
How do undisclosed analytical choices, researcher degrees of freedom, or specification sensitivity affect reported empirical results, and what multiverse or specification curve methods have been proposed to expose them?
```

COLUMNS

```
Field or domain studied
Method for exposing sensitivity (multiverse analysis, specification curve, robustness sweep)
Number of specifications enumerated
Range of results across specifications
Whether the headline result reversed under some specification
Proposed disclosure or preregistration remedy
Whether the method has been applied in computer vision or remote sensing
Venue and year
```

**Screen:** this literature is mostly outside computer vision. Cite two or three
as framing, not more. Do not let it take over Related Work.

---

## 9. Over segmentation and fragmentation in tree crown delineation

Backs section 4.5. Our F1 of 0.13 against a 90.6 percent tree level rate needs
company in the literature or a reviewer will read it as a broken pipeline.

QUERY

```
How common is over segmentation or crown fragmentation in individual tree crown detection and delineation from aerial imagery, how is it quantified, and how is it distinguished from missed detections?
```

COLUMNS

```
Method used for crown delineation
Over segmentation rate or detections per reference tree
How over segmentation was defined and measured
Whether one to one assignment was enforced
Reported precision, recall, and F1 with IoU threshold
Whether a tree level detection rate was reported separately
Canopy type and closure
Effect of crown size on fragmentation
```

**Screen:** the "detections per reference tree" column is the key one. Any
paper reporting a median or mean there is a direct comparison point for our 2.0.

---

## 10. Evaluation and matching protocols for crown detection

Backs the Method section. Our matching rules need precedent: containment
primary, one to one Hungarian at IoU 0.5 secondary, nested scoring sets.

QUERY

```
What matching rules and evaluation protocols are used to compare detected tree crowns against reference annotations in aerial imagery, and how are one to many and many to one matches handled?
```

COLUMNS

```
Matching rule (IoU, centroid containment, Hungarian assignment, distance threshold)
Threshold value used
How one to many matches were handled
How many to one matches were handled
Metrics reported
Whether multiple scoring sets or annotation confidence tiers were used
Whether unannotated regions were excluded from precision
Stated limitation of the protocol
```

**Screen:** the "unannotated regions" column matters. Our 67.5 percent coverage
limit forbids reading unmatched detections as false positives, and precedent for
that exclusion is worth having.

---

## 11. Annotation protocol and inter annotator agreement for crown delineation

Backs the Limitations section. One annotator with no field verification is a
known weakness and needs a protocol to cite.

QUERY

```
What protocols and inter annotator agreement statistics are used when manually delineating individual tree crowns in high resolution aerial or drone imagery for use as evaluation ground truth?
```

COLUMNS

```
Annotation unit (bounding box, polygon, point)
Number of annotators
Agreement statistic reported and its value
Written annotation rules or decision criteria
How overlapping or occluded crowns were handled
How small or understory trees were handled
Whether field verification was performed
Estimated annotation time per crown
```

**Screen:** star anything reporting an agreement statistic with a value. A
single number from the literature lets us say what a second annotator would
plausibly have changed, which is stronger than saying we did not have one.

---

## 12. Resolution and ground sample distance effects on crown detection

Backs the Method justification for running at 7.78 cm rather than native
3.89 cm.

QUERY

```
How does ground sample distance or image resolution affect individual tree crown detection accuracy in aerial and drone RGB imagery, and does higher resolution always improve detection?
```

COLUMNS

```
Ground sample distances compared
Detection method used
Accuracy at each resolution
Whether accuracy was non monotone in resolution
Whether crown fragmentation increased at higher resolution
Median detected object size at each resolution
Sensor and flight altitude
Canopy type
```

**Screen:** the non monotone column is what justifies our downsampling. If no
paper reports it, that becomes a stated methodological choice rather than a
cited one, and the gate table carries the argument alone.

---

## Screening rules across all twelve

- **Date limits.** 2019 and later for prompts 3, 9, 10, 11, 12. 2016 and later
  for prompts 4, 5, 6, 7. No limit on prompt 8, which has older foundational
  work. Prompt 1 and 2 unrestricted, since an old paper that already did this
  matters more than a recent one that did not.
- **A metric without a stated IoU threshold is not a comparison point.** Record
  the paper as context and do not quote its number.
- **Star for stated absences, not just findings.** Prompt 2 is looking for what
  papers do not report. A paper that omits tile origin is a data point.
- **Stop at roughly 6 stars per prompt.** Twelve prompts at 5 to 6 gives 60 to
  72, which is the target with margin for duplicates.
- **Export BibTeX at the end of every search**, not at the end of the session.

## Gaps these searches will not fill

Ours to supply, not Elicit's.

- The site description for Mission 000103: species mix, canopy closure, stem
  density, acquisition date.
- The DeepForest package version and checkpoint identifier actually used.
- Compute environment and runtime for the sixteen position sweep.
- Whether `Methods_tile_grid_phase.docx` exists, and if not, the Method section
  written from scratch.

## Where each prompt lands in the paper

| prompt | section | claim backed |
| --- | --- | --- |
| 1 | Introduction, Related Work | novelty |
| 2 | Introduction | claim 1, the reporting gap |
| 3 | Introduction, Method | claim 1, usage base |
| 4 | Related Work, Discussion | claim 2, claim 4 |
| 5 | Related Work, Discussion | claim 4 |
| 6 | Related Work, Discussion | claim 3 |
| 7 | Introduction, Discussion, Conclusion | framing |
| 8 | Introduction, Discussion | framing |
| 9 | Results 4.5, Discussion | over segmentation |
| 10 | Method | matching protocol |
| 11 | Limitations | annotation quality |
| 12 | Method | resolution choice |
