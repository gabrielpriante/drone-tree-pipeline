# 3. Method

Draft v1, 2026-08-25. Target 1200 words. Current: approximately 1230.
Condensed from `Methods_tile_grid_phase.docx` (working draft, 20 August 2026),
which runs roughly 4500 words and remains the full record. Bracketed notes are
drafting flags, not paper text.

---

## 3.1 Data

All imagery comes from a single orthomosaic, Open Forest Observatory drone
Mission 000103, released under CC BY 4.0 and covering closed conifer canopy in
northern California. The mission was selected because Open Forest Observatory
withheld it from machine learning training and reserved it for model evaluation,
so it is not part of any published training corpus for the detector under test.
The raster is 11632 by 12458 px in EPSG:32610 at a native ground sample distance
of 3.89 cm, with four bands of which the fourth is alpha and no declared nodata
value.

The experiment operates on one 2000 px square window at native resolution, at
column offset 4820 and row offset 5260, so that sixteen full inference passes
remain tractable and every pass covers identical ground.

Inference runs at 7.78 cm, a factor of two downsample, decided by a gate run
before any sweep was performed. At native resolution the detector returns 1201
objects at a median score of 0.273 and a median box width of 0.97 m; at 7.78 cm
it returns 311 at a median score of 0.348 and a median box width of 2.10 m.
Conifer crowns in this stand are not 0.97 m across, and inspection confirmed the
native pass was returning branch scale fragments. Every subsequent number is
produced at 7.78 cm.

## 3.2 Detector and tiling configuration

Detections come from DeepForest with the released pretrained weights and no fine
tuning. No local annotation was used to adapt the model, because the question
under test concerns an inference time parameter and adapting the model would
confound that parameter with training effects.

Inference is tiled, with patch size 400 px, patch overlap 0.25 and a derived
stride of 300 px, held fixed at every grid position. Boxes from overlapping
tiles are reconciled with `torchvision.ops.nms` at IoU 0.15, which is the same
function and threshold DeepForest passes into its own mosaic step, so any
difference between our tiler and the library cannot be attributed to the merge.

## 3.3 The manipulated variable

Given a fixed patch size and overlap ratio, the origin of the tiling grid
remains free, and the DeepForest interface exposes no control over it. The
experiment manipulates that origin and holds everything else constant.

Sixteen grid positions were run as a full factorial over sub stride offsets of
0, 75, 150 and 225 px on each axis. Because the stride is 300 px, offsets of 0
and 300 describe the same grid, so these four values sample the offset space at
quarter stride intervals without duplication.

`predict_tile` provides no control over grid origin, so the sweep uses a hand
rolled tiler over `predict_image` that cuts the grid at a specified offset, runs
inference per tile, translates boxes into window coordinates and merges under
the configuration above. Because a reimplementation of a library function is a
source of doubt, it was validated against the library before use: at offset zero
both paths return 311 detections, all 311 boxes match at IoU 0.5, and the median
matched IoU is 0.9344. The residual difference from unity is floating point
ordering in the merge rather than a difference in grid construction.

## 3.4 Boundary treatment and tiling regimes

Shifting the grid origin also changes what happens at the window edge, which
would introduce an artefact covarying with the manipulated variable. We
therefore run the grid over a canvas one full stride larger than the scored
window on every side and fill that margin with real imagery read from the
orthomosaic rather than with synthetic pixels. The expanded read is 3200 px
square at native resolution, downsampled to a 1600 px canvas at 7.78 cm, giving
300 px of real margin on each side with alpha coverage verified at 100 percent
on all four. An earlier version reflect padded the window instead; under that
design tiles containing synthetic pixels reached up to 325 px into the scored
region at some positions, with intrusion depth varying by offset. Nothing here
rests on it.

Tile origins on an axis are generated at fixed stride across the canvas, so an
offset of zero admits five origins on that axis while any nonzero offset admits
four. The sixteen positions therefore span three tiling regimes rather than one:
twenty five tiles at the single position with both offsets zero, twenty at the
six with exactly one offset zero, and sixteen at the remaining nine, with mean
core counts of 288, 277.50 and 271.33 respectively. We report the zero offset
position separately throughout and quote the fifteen position pooled figures,
mean 273.80 and coefficient of variation 0.0245, as the experiment's spread.

## 3.5 Cross position correspondence

Matching is restricted to a core region inset 25 px from the window edge, giving
a scored region of 950 by 950 px or 73.9 m square. The inset is a nominal
boundary guard rather than a contamination guard, since the real pixel margin
removes the need for the latter.

Vocabulary follows Miller et al. A *sample* is one grid position, an
*observation* is one detection box from one sample, a *cluster* is a set of
observations judged to be the same object, and *support* is the number of
distinct samples contributing to a cluster, from 1 to 16. Observations are
grouped using basic sequential algorithmic scheme clustering with intra sample
exclusivity, following the same work; exclusivity is what makes support a count
out of sixteen rather than a count of boxes.

The association threshold is IoU 0.5, far below the 0.95 used in the source
work, which associates stochastic forward passes of one model on one image where
agreement is expected to be tight. Here the observations come from different
tilings and positional disagreement is the phenomenon under study. Because the
threshold could drive the headline result, the full clustering was rerun at IoU
0.3 and 0.4 and every threshold sensitive quantity is reported at its
conservative end. The cluster total of 710 was confirmed two independent ways
off disk.

## 3.6 Seam pinning test

To test whether unstable detections are locked to the processing grid, we
measure the distance from each box edge to the nearest grid boundary, keeping
the smaller of the two edges, with the cluster value taken as the median over
its observations. The null is empirical: each box keeps its size and its offset
while its position is shuffled uniformly within the core, over 1000 iterations
at a fixed seed. An empirical null was chosen over a theoretical one because
small boxes fall near arbitrary features by chance and a theoretical null would
not control for the size distribution of the boxes under test. Pin tolerance is
1.0 px, critical p is 0.001, and the three possible verdicts were stated before
any number was seen.

The test runs first over the 71 single axis sensitive clusters, where each has
one sensitive axis carrying one offset. Extending it to all 710 requires a
general rule, since an arbitrary cluster has neither: the gap on each axis is
measured against that observation's own grid, the observation gap is the minimum
of the two, and the cluster gap is the median over observations. Taking a
minimum over two axes inflates the pinned share, so the null is constructed
identically and the inflation cancels. Per support level nulls are used rather
than one pooled null, because box size varies with support and box size drives
the null.

## 3.7 Conditional geometry test

Because seam pinning and box shape are mechanically coupled, testing whether
shape independently distinguishes stable from unstable detections requires the
pinned clusters removed. We control by exclusion rather than regression, since a
box cut at a boundary is small and elongated by the same process, and a model
asked to separate the two would be unstable. The comparison is unpinned support
1 against all of support 16, with aspect ratio designated the primary endpoint
and log box area the secondary, both fixed in advance. Tests are two sided Mann
Whitney U with rank biserial effect sizes, bootstrap confidence intervals at
10000 resamples, and Holm correction across the two endpoints. Thresholds for
ruling geometry in, ruling it out, and returning inconclusive were written down
before any number was computed, and the verdict is printed mechanically against
them.

## 3.8 Reference annotation and matching

A written annotation protocol was committed to version control before the first
box was drawn, so its timestamp precedes the data it governs. Annotation was
blind, with no detection output visible, on a 950 by 950 px chip identical to
the scored core. The operative rules were one apex to one tree, with a lit
branch cluster on the flank of a larger crown not annotated; two apices joined
by continuous foliage annotated as two overlapping boxes; a minimum crown
diameter of 1.5 m; and annotation by apex position rather than by box
containment at the chip edge. Cast shadow was never included. The export
contains 110 boxes at a median of 4.47 by 5.52 m, an annotated stem density of
approximately 200 per hectare.

Matching uses containment as the primary rule and one to one assignment as the
secondary, and the order is deliberate. The median annotated box is 4.47 by
5.52 m while the median detection is 2.10 m wide, so a detection at half the
linear dimension of the tree containing it cannot exceed roughly 0.25 IoU even
when perfectly centred. One to one IoU matching at 0.5 would therefore report
near total failure as a property of the metric rather than of the detector.
Containment counts detections with at least 50 percent of the detection's own
area inside an annotation; the direction matters, since a fragment inside a
large crown counts while a box swallowing a crown does not. Hungarian assignment
at IoU 0.5 runs second for comparability with the cross position matching and
with the wider literature, and is labelled as penalising fragmentation by
construction.

Five nested scoring sets are reported rather than one. The primary set is live,
canopy, certain and not edge clipped, and the analysis is repeated four times
adding back one excluded group at a time. Detections come from the single grid
position `dx225_dy075`, selected by a rule applied before any matching was run:
it sits at the median of the sixteen on core detection count, 274 against a
fifteen position mean of 273.80, and on one off detections, 13 against a median
of 13 with a minimum of 6 and a maximum of 17.

---

## Drafting flags

- **The docx carries a superseded claim.** Section 5.3, the correction note,
  ends "The shape hypothesis holds." That is the exact sentence removed from
  `README.md` and `analyse_support.py` over the last two sessions. It must be
  qualified in the docx the same way before anyone reads that file, or the
  refuted claim survives in the most authoritative looking document in the
  project.
- **Two status notes in the docx are stale.** Section 8 opens with "Outputs are
  pending at the time of writing," and section 8.6 states that precision, recall
  and F1 are not computed. All three exist now. The deferral logic in 8.6 is
  still correct as a description of the order in which decisions were made, and
  should be rewritten in past tense rather than deleted, because the fact that
  the true positive rule was fixed after the distribution was seen and not
  chosen to suit it is worth preserving.
- **Section 5.4 of the docx conflates two counts,** describing the single axis
  pattern as occurring in 74 of 86 when 74 is the structured count and 71 is the
  single axis count. Now resolvable: the 3 are two by two blocks. Same fix as
  applied to Results 4.3.
- **Still missing from Method:** the DeepForest package version and checkpoint
  identifier, and the compute environment and runtime for the sixteen position
  sweep. Neither is in the docx. Both are one line each.
- **Cut from the docx and not replaced:** the annotation export label breakdown,
  the coordinate conversion asserts, the export discovery logic, the internal
  consistency asserts, the synthetic control, and the terminology inconsistency
  note. All belong in the repository rather than in an eight page paper. If a
  reviewer asks how the reference data was validated, the protocol and the docx
  answer it.
- **Word budget.** Section 3.8 is the longest subsection and the most
  compressible. The containment versus IoU justification should survive any cut,
  since it is what stops a reviewer reading F1 0.13 as a broken pipeline.
