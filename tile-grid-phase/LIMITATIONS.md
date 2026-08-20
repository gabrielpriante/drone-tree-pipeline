# Limitations

Every limitation this project carries, in one place, so the paper does not have
to reassemble them from three documents.

Each entry has three lines: **what it is**, **the evidence**, **what it
forbids**. The third line is the operative one. None of these are softened, and
none should be softened in the paper. A reviewer who finds one of these before
we name it will discount everything near it.

---

## 1. Scope

**What.** One 950 by 950 px window, 73.9 m square, from one orthomosaic, one
site, one forest type: Northern California closed conifer canopy, OFO Mission
000103.

**Evidence.** `phase_sweep.py` constants. Working window col_off 4820, row_off
5260, size 2000 native.

**Forbids.** Any claim that generalises past this window. Not to other sites,
other forest types, other missions, other seasons, or other parts of this same
orthomosaic. The grid position effect is demonstrated here; its magnitude
elsewhere is unmeasured.

---

## 2. One annotator, no agreement statistic

**What.** All 110 annotations were drawn by one person. No inter annotator
agreement exists. The intra annotator repeat that `ANNOTATION_PROTOCOL.md`
section 7.1 specifies as the substitute has not been performed.

**Evidence.** `ANNOTATION_PROTOCOL.md` section 8 item 1. No second pass file
exists in `ground_truth/`.

**Forbids.** Any claim about annotation reliability. The protocol names the
weaker substitute and even that has not been run, so the ground truth has no
reliability statistic of any kind attached to it.

---

## 3. No field verification

**What.** Nothing is validated against a ground survey, a stem map, or lidar.
This is photo interpretation of nadir RGB.

**Evidence.** `ANNOTATION_PROTOCOL.md` section 8 item 3.

**Forbids.** Calling the annotations truth without qualification. They are one
person's reading of one image. Precision, recall and F1 are computed against
that reading, not against the forest.

---

## 4. Metrics are conditional on visibility

**What.** Crowns not resolvable in nadir RGB, fully suppressed understory in
particular, are absent from the ground truth. Annotation was performed on the
2x downsampled 7.78 cm image, not at native 3.89 cm.

**Evidence.** `ANNOTATION_PROTOCOL.md` section 8 items 4 and 5, and rule 2.4,
which says absence of an annotation means not resolvable rather than not
present.

**Forbids.** Reading recall as a share of the true stem population. It is a
share of the visible, annotated population. The true stem count for this window
is unknown and no number in this project estimates it.

---

## 5. The annotation tool forced mutually exclusive classes

**What.** The export carried one `label_name` per box from tree, understory,
uncertain, snag. The protocol treats class, layer and confidence as three
independent fields and the export cannot represent that.

**Evidence.** `ANNOTATION_PROTOCOL.md` section 8 item 7, and the docstring of
`load_ground_truth.py`. An `uncertain` box carries no record of layer, an
`understory` box carries no record of confidence, a snag that was also
uncertain could not be represented at all.

**Forbids.** Treating the 11 uncertain, 18 understory and 8 snag counts as
measurements. They are lower bounds. The 99 `certain` boxes include 18
understory and 8 snags that were never asked about confidence, so the certain
subset is contaminated and "metrics on certain only" is weaker than it sounds.

---

## 6. Annotation covers part of the chip

**What.** Coverage runs from 47.9 percent of core area at scoring set 1 to 67.5
percent at set 5.

**Evidence.** `ground_truth/match_metrics.csv`, `annotation_coverage_share`.
Rasterised onto the 950 grid so overlap counts once.

**Forbids.** Reading detections outside every annotation as false positives.
At scoring set 5, 81 of 274 detections fall outside all 110 annotations, and 34
of those are support 16 detections agreed on by every survey. Either the
annotation missed them or they are consistent false positives, and this
experiment cannot distinguish the two. **No false positive rate is available
anywhere in this project.** Precision as reported is a floor.

---

## 7. The 164 clusters at supports 5 to 15 have no account

**What.** The seam mechanism explains the bottom of the support distribution
and nothing else. 164 clusters sit between supports 5 and 15 with 10 pinned,
6.1 percent.

**Evidence.** `seam_pinning_all_by_support.csv`. 292 of the 302 pinned clusters,
96.7 percent, sit at supports 1 to 4.

**Forbids.** Saying the seam mechanism explains the instability. It explains
the low support end. The middle of the distribution is unexplained and goes in
as a stated open question, not as an implication.

---

## 8. Per level pinning verdicts are unreportable at supports 5 to 15

**What.** Every support level from 5 to 15 has n below 70. The shuffled null
share collapses to exactly 0.0000 at that n, so a single pinned cluster returns
an empirical p of 0 and the script prints PINNED.

**Evidence.** `seam_pinning_all_by_support.csv`, `null_share_hi95` column.
Support 8 at 1 of 16 and support 11 at 2 of 11 are the specific artefacts.

**Forbids.** Quoting any per level verdict outside supports 1, 2, 3, 4 and 16.
The shares and the counts at sparse levels stand. The labels do not.
`fig_pinning_by_support.png` hatches those bars for this reason.

---

## 9. Three tiling regimes, not one

**What.** Grid positions do not all run the same number of tiles. 25 tiles at
dx000_dy000, 20 at the six positions with exactly one offset at zero, 16 at the
remaining nine.

**Evidence.** `phase_summary.csv`, `n_tiles`. Mean core counts 288, 277.50,
271.33 by regime.

**Forbids.** Describing the 15 non zero positions as comparable or homogeneous.
They span two regimes. The cv of 0.0245 over those 15 is correct as computed
and is the figure to quote, but the word comparable does not apply to it.

---

## 10. Over segmentation is measured and uncorrected

**What.** Median 2.0 detections per annotated tree, 33 of 64 trees carrying two
or more, tail to 12.

**Evidence.** `ground_truth/match_per_annotation.csv`,
`n_detections_contained`. `fig_detections_per_tree.png`.

**Forbids.** Reading any count in this project as a count of trees. The 710,
the 274, the 115, the per position totals: every one is a count of
**detections**. None has been adjusted. Any tree level claim needs the matching
output, not these counts.

---

## 11. The spatial variance test is untested by the synthetic control

**What.** The control that validated the elongation and seam proximity tests
did not exercise the Miller spatial variance axis prediction.

**Evidence.** `analyse_mechanism.py` docstring. The control returned 45.83
percent concordant, p 1, because the planted observations were jittered
isotropically at random rather than shifted by the tiling, so there was no axis
structure for the variance test to find.

**Forbids.** Treating the spatial variance result on real data as validated.
It is exploratory. The other two mechanism tests recovered a planted mechanism
at p 7.4e-07 and p 1.2e-07; this one recovered nothing because there was
nothing planted for it.

---

## 12. The five scoring sets are nested

**What.** Set 1's 64 annotations are a subset of set 2's 73, of set 3's 84, of
set 4's 102, of set 5's 110.

**Evidence.** `SCORING_SETS` in `match_ground_truth.py`, and the `n_ann` column
of `ground_truth/match_metrics.csv`.

**Forbids.** Calling agreement across the five sets a replication. They share
most of their data, so the Spearman being negative on all five is close to
guaranteed by the overlap. The correct claim is that the null is robust to the
choice of scoring set. **State the nesting in the same breath as the strength
claim**, or a reviewer who spots it will discount the passage.

---

## 13. The support null is close to a boundary, and should be quoted as such

**What.** The smallest Fisher p anywhere in the five sets is 0.0753, at scoring
set 3.

**Evidence.** `ground_truth/match_metrics.csv`, `fisher_p_all16_vs_some`.

**Forbids.** Writing "nothing approached significance". 0.0753 is close enough
that a reader recomputing will see it. Name the value first. The null is still
the right reading: five nested sets, no p below 0.05, Spearman near zero and
negative, and the effect sizes are 47.0 against 52.7 percent.

---

## 14. Naming inconsistency, crowns against detections

**What.** The Support structure results section of `README.md` says "crowns"
throughout, meaning clustered detections.

**Evidence.** `README.md`. Renaming would touch the terminology table and the
Miller definitions of cluster and observation.

**Forbids.** Nothing, on its own. It is a consistency defect rather than a
correctness one, and the "Everything here counts detections, not trees" section
does the protective work. Recorded so it is not mistaken for a claim that the
detections are crowns.

---

# Limitations not on the list you gave me

Eight more the record carries. Same three line form.

## 15. One resolution, chosen from a gate of two

**What.** The experiment runs at 7.78 cm. Only native 3.89 cm and this 2x
downsample were ever compared.

**Evidence.** `run_gate.py`, the two row gate table. 1201 detections at native
against 311 at 7.78 cm.

**Forbids.** Any claim that 7.78 cm is optimal, or that the grid position
effect has the same magnitude at other resolutions. Two points do not
characterise a curve.

---

## 16. One overlap ratio

**What.** Patch 400, overlap 0.25, stride 300. Overlap was held fixed by design
so that phase was the only variable, and it was never varied.

**Evidence.** `phase_sweep.py` constants.

**Forbids.** Extending the finding to other overlap settings. Higher overlap
would plausibly reduce the effect and lower overlap increase it, and neither is
measured. This is arguably the most obvious next experiment.

---

## 17. Coarse sampling of phase space

**What.** 16 positions on a 4 by 4 sub stride grid, offsets 0, 75, 150, 225 on
a 300 px stride.

**Evidence.** `PHASE_OFFSETS` in `phase_matching.py`.

**Forbids.** Reading the support distribution as continuous. A crown found at
4 of 16 sampled positions might be found at very different rates under finer
sampling. The 16 are a sample of the stride, not the stride.

---

## 18. One clustering configuration

**What.** BSAS with intra sample exclusivity at IoU 0.5. Miller's taxonomy
contains other merging strategies and none were tried.

**Evidence.** `phase_matching.py` docstring. Thresholds 0.3 and 0.4 were swept,
the strategy was not.

**Forbids.** Claiming the support distribution is strategy independent. Only
threshold sensitivity was tested, and it moved the singleton count from 129 to
196.

---

## 19. The seam pinning null assumes uniform placement

**What.** The shuffled null places each box uniformly at random over the core,
holding size and offset.

**Evidence.** `null_draw` in `check_seam_pinning.py`, `rng.random()` over
`[CORE_INSET, WIN_SIZE - CORE_INSET]`.

**Forbids.** Reading the null as a model of where trees are. Real crowns are
not uniformly distributed and cluster with canopy structure. The null tests
against uniform placement, which is the right control for a geometric artefact
and the wrong one for anything ecological.

---

## 20. The over segmentation tail rests on two annotations

**What.** The two trees carrying 12 detections each are the two largest
annotations, 283 and 185 m2.

**Evidence.** `ground_truth/match_per_annotation.csv` joined on `area_m2`.

**Forbids.** Quoting the tail as a general property. The median of 2.0 and the
33 of 64 are the robust statements. Everything past about 5 detections per tree
is a handful of boxes on a handful of very large crowns.

---

## 21. The 8 unpinned single axis clusters are a separate phenomenon

**What.** Of the 71 single axis sensitive clusters, 63 are pinned slivers at
median aspect 2.57 and 8 are ordinary sized roughly square boxes at aspect
1.06.

**Evidence.** `check_seam_pinning.py`, the geometry split table.

**Forbids.** Describing the 8 alongside the 63. They are a different thing with
n of 8, and nothing explains them.

---

## 22. Detector and score threshold were not varied

**What.** One pretrained DeepForest model, `weecology/deepforest-tree`, no
retraining or fine tuning for this site, and no score threshold sweep. The
effective score floor is DeepForest's config `score_thresh` of 0.1.

**Evidence.** `phase_sweep.py` model load and the `SCORE_THRESHOLD = None`
constant with its comment.

**Forbids.** Any claim that the effect is a property of tiled inference in
general rather than of this model at this threshold. A model retrained on this
canopy, or a different score floor, might fragment differently.

---

# The one thing none of this touches

The instability finding does not depend on the annotation being right, on the
annotator being skilled, on the coverage being complete, or on any of items 2
through 6. **Sixteen surveys of one photograph disagree with each other about
which trees are present, and that is measurable without any ground truth at
all.** Everything ground truth adds is the explanation of what the disagreeing
detections are. If a reviewer rejects the annotation entirely, the sweep result
stands and only the matching section falls.
