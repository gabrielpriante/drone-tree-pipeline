# Tile Grid Phase Experiment

Does the *position* of the DeepForest tiling grid, at a fixed overlap ratio,
change which tree crowns get detected? These scripts establish the raster,
pick a working window, and gate the question at a chosen resolution.

## Scripts

| Script | What it does |
| --- | --- |
| `check_raster.py` | Prints size, band count, dtype, CRS, nodata, pixel size, GSD, footprint and bounds for the orthomosaic. |
| `find_window.py` | Locates a 2000 px square window centred on the valid (alpha > 0) region of the mosaic and reports percent valid coverage. |
| `run_gate.py` | Reads the working window, writes it at native and half resolution, runs DeepForest on both, and reports detection count, score distribution and median box width. |
| `draw_boxes.py` | Overlays the boxes from `run_gate.py` onto the window PNGs for visual inspection. |
| `phase_sweep.py` | The experiment proper. Lays the tiling grid down 16 times at a 4x4 grid of sub-stride phase offsets, holding tile size and overlap ratio fixed, and reports how much the detection set moves. Run. |
| `check_expanded_window.py` | Confirms the expanded read region used by `phase_sweep.py` is inside the raster and that its margin carries valid alpha on all four sides. Read only. |
| `check_tiler_vs_predict_tile.py` | Zero offset sanity check. Runs the hand rolled tiler and `predict_tile()` on the same bare window with identical settings and reports how far they agree. Passed: 311 of 311 boxes matched, median matched IoU 0.9344. |
| `phase_matching.py` | Shared read only module. Reloads the per phase box CSVs and redoes the clustering at any threshold. Imports no model. |
| `check_match_sensitivity.py` | Redoes cross phase matching at IoU 0.3, 0.4 and 0.5 and reports distinct crowns, support histogram, and singleton count for each. Tests whether the singleton pile is a clustering artefact. |
| `draw_support.py` | Draws the core region at one phase with crowns coloured by support band, plus a second figure pooling singletons from all 16 phases. |
| `analyse_support.py` | Box geometry by support band, pattern of support at 4, 8 and 12 against an exact null, axis marginals, and per phase count spread with phase 0 separated. |
| `analyse_mechanism.py` | Mechanism tests: elongation axis against sensitivity axis, seam proximity at the samples found versus missed, and Miller spatial variance per cluster. |
| `check_seam_pinning.py` | Tests whether the 71 single axis sensitive detections have a box edge on a grid boundary, against a shuffled null. |
| `check_seam_pinning_all.py` | The same test over all 710 clusters, reported by support level, with support 1 kept separate from the 2 to 15 band. |
| `build_figure.py` | Builds the non technical figure from `core_clean.png`. Derives its counts at run time. |
| `load_ground_truth.py` | Converts the annotation export to the protocol schema. Asserts the chip to window offset rather than trusting it. |
| `match_ground_truth.py` | Matches detections against annotations. Containment primary, one to one IoU secondary, five nested scoring sets. |
| `build_results_figures.py` | The three paper figures: support histogram, pinning by support, detections per annotated tree. |

Generated outputs (`*.png`, `*_boxes.csv`) are gitignored. Rerun the scripts to
regenerate them.

Ground truth annotation follows `tile-grid-phase/ANNOTATION_PROTOCOL.md`,
written before annotation began.

## Settled decisions

### Dataset

OFO Mission 000103, CC BY 4.0. Withheld from ML training and reserved by OFO
for model evaluation.

### Orthomosaic

- 11632 x 12458 px
- EPSG:32610
- 3.89 cm GSD
- 4 bands, band 4 is alpha
- no nodata value set

### Working window

`col_off 4820`, `row_off 5260`, `size 2000`, at native resolution.

### Experiment resolution

The experiment runs at **7.78 cm**, downsampled by a factor of 2 from native.
Justified by crown fragmentation at native resolution.

### Gate result at the working window

| Resolution | Detections | Median score | Median box width |
| --- | --- | --- | --- |
| native 3.89 cm | 1201 | 0.273 | 0.97 m |
| downsampled 7.78 cm | 311 | 0.348 | 2.10 m |

At 3.89 cm the median box is under a metre wide, narrower than a real crown at
this site, and confidence is low. Halving the resolution yields fewer,
higher confidence, crown sized boxes. 7.78 cm is therefore the operating point
for the phase sweep.

## Phase sweep design

`phase_sweep.py` isolates grid position as the only variable.

**Phases.** `PATCH_SIZE` 400, `PATCH_OVERLAP` 0.25, so stride is 300 px.
Offsets are swept at 0, 75, 150 and 225 px in both x and y, 16 positions
covering one stride in quarter steps. Overlap ratio is identical throughout.

**Tiling.** `predict_tile()` gives no control over grid offset, so tiles are
cut by hand and `predict_image()` is called per tile, with boxes translated
back to scored window coordinates and merged by one `torchvision.ops.nms` pass
at IoU 0.15. That is the same function and the same threshold DeepForest's own
`mosaic()` uses, since `predict_tile()` passes `iou_threshold=0.15` into it.

**Constant tile size.** Every tile is exactly 400 px under every phase.
DeepForest is sensitive to input size, and if tile size co varied with phase
the two effects could not be separated. The grid therefore runs over a canvas
larger than the scored region.

**Real pixel margin.** The margin is real imagery read from the orthomosaic,
not padded or reflected pixels.

- Expanded read, native: `col_off 4220`, `row_off 4660`, `size 3200`
- Downsampled by 2 to a 1600 px canvas at 7.78 cm
- 300 px of real margin on every side, at experiment resolution
- Alpha coverage verified at 100 percent on all four margin bands

An earlier version reflect padded the window instead. That was wrong. Tiles at
some phases reached up to 325 px into the scored region while still containing
synthetic pixels, far past the 100 px inset that was supposed to exclude them.
Depth by phase was 100 px at phase 0, 175, 250 and 325 px at phases 75, 150 and
225, asymmetric between the low and high edge because nonzero phases emit four
tiles per axis rather than five.

**Scored region.** The scored region is the original 2000 px native working
window, the inner 1000 px of the canvas at experiment resolution. The margin is
context only and is never scored. A box is kept when its centre falls inside
the scored region.

**Nominal inset.** `CORE_INSET` is 25 px, about 1.9 m. It is not a
contamination guard, the real pixel margin removed that need. It exists because
a crown sitting on the scored boundary can have its centre fall inside at one
phase and outside at the next, which would register as instability that is
really a boundary artefact. 25 px is roughly two median box radii at the gate's
2.10 m median width and costs under 10 percent of the scored area.

**Matching.** Boxes from different phases are the same crown at IoU >=
`MATCH_IOU`, default 0.5, clustered greedily from the highest scoring box
outward, at most one box per phase per cluster. Cluster support is then a clean
1 to 16 count, and the headline number is what fraction of crowns survive all
16 grid positions.

**Tunable constants.** `NMS_IOU`, `MATCH_IOU` and `CORE_INSET` are top level
constants in `phase_sweep.py`. `MATCH_IOU` can be varied to 0.3 or 0.4 without
touching any logic.

**Score floor.** `SCORE_THRESHOLD` is `None`, no extra filter. DeepForest's own
config `score_thresh` is 0.1, so 0.1 is the effective floor regardless.

## Before the sweep runs

`check_tiler_vs_predict_tile.py` must pass. It runs the hand rolled tiler at
offset 0 and `predict_tile()` on the same bare 1000 px window, with patch size,
overlap and NMS threshold held identical, and reports detection counts from
each plus how many boxes match at IoU 0.5. If the hand rolled path does not
reproduce `predict_tile()` at zero offset, every phase result afterwards is
untrustworthy.

The check deliberately uses no margin, because `predict_tile()` has no concept
of one. `phase_sweep.py` prints its own gate drift comparison at phase (0, 0)
instead.

## Sweep result, first run

Core region only. `phase_sweep.py` ran at `MATCH_IOU` 0.5. The matching was
then redone at 0.4 and 0.3 by `check_match_sensitivity.py`, which reads the
saved per phase CSVs and loads no model.

| match IoU | distinct crowns | found in all 16 | pct | found in exactly 1 | pct | found in 2 to 15 | pct | median support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 591 | 139 | 23.52 | 129 | 21.83 | 323 | 54.65 | 4.0 |
| 0.4 | 651 | 127 | 19.51 | 167 | 25.65 | 357 | 54.84 | 4.0 |
| 0.5 | 710 | 115 | 16.20 | 196 | 27.61 | 399 | 56.20 | 4.0 |

The support histogram is U shaped at every threshold: a pile at 1, a trough
from 5 to 14, a spike at 16, with a local peak at 4.

### Two facts that do not depend on the threshold

1. **The 2 to 15 band sits near 55 percent at every threshold**, 54.65, 54.84
   and 56.20 percent. Most crowns are neither fully stable nor one offs. They
   appear at some grid positions and not others.
   That band is not one thing. See Seam mechanism below: supports 2 to 4
   behave like the singletons, supports 5 to 15 do not and are unexplained.
2. **Median support is exactly 4.0 at every threshold.** The typical crown is
   found at a quarter of the grid positions.

Neither moves when the matching threshold moves, so neither is an artefact of
where the matching line was drawn.

### What is threshold sensitive

The singleton count is not stable. It runs 129, 167, 196 as the threshold
tightens from 0.3 to 0.5, and singletons account for 56.3 percent of the drop
in distinct crowns from 0.5 to 0.3. Some of the singleton pile at 0.5 is one
crown split across phases by a strict threshold.

**Quote IoU 0.3 as the conservative case.** At 0.3, 129 crowns, 21.83 percent,
are found at exactly one of 16 grid positions. That is the floor. The looser
the threshold, the harder it is to call a singleton an artefact.

The percentage found in all 16 is threshold sensitive in the other direction,
23.52 at 0.3 down to 16.20 at 0.5. The conservative claim there is 16.20
percent.

### Visual check

Read from `support_dx000_dy000_iou05.png`: the singletons sit on real crowns,
not on bare ground or shadow. They are flickering detections, not junk.

A second impression from that figure, that the unstable boxes are larger than
the stable ones, **was wrong and the measurements refuted it.** See the
correction under Support structure results. Unstable boxes are smaller and more
elongated. They look larger by eye because they are more elongated, and a few
long boxes spanning several crowns draw attention away from the many small
ones.

That observation is the reason for `analyse_support.py`. The working hypothesis
was that ambiguous multi crown groupings resolve differently depending on where
the seam falls, which predicted large malformed unstable boxes. The measured
geometry points the other way: unstable detections are small and elongated, and
the mechanism is seam pinning rather than severing: 63 of 71 have a box edge
sitting exactly on a grid boundary, and 136 of the 196 singletons are pinned.
`analyse_mechanism.py` tests that directly.

Caveat on that figure: it drew phase (0, 0) only, so it showed 17 of 196
singletons, and from the 25 tile regime. `draw_support.py` now defaults to
`dx075_dy075` and also writes a pooled figure covering all 16 phases.

### Per phase count spread

The spread quoted from the first run, min 262, max 288, cv 0.0269, was taken
over all 16 phases. It therefore mixes two tiling regimes, see the next
section. `analyse_support.py` recomputes it over the 15 four tile phases with
phase (0, 0) quoted separately. **Quote the 15 phase spread as the
experiment's spread.**

## Phase 0 is a different tiling regime

**Do not pool phase (0, 0) with the other 15 phases without saying so.**

Phase 0 fits 5 tile origins per axis. Every other phase fits 4. The canvas is
1600 px, the patch is 400 px and the stride is 300 px, so origins run
`range(phase, 1201, 300)`. At phase 0 that yields 0, 300, 600, 900, 1200. At
phase 75 the fifth origin would be 1275, which overruns, so only four are
emitted. Same for 150 and 225.

That is 25 tiles at phase 0 against 16 elsewhere, and the extra tiles raise how
many times each scored pixel is seen.

| phase | tiles | mean tile coverage, scored | mean tile coverage, core |
| --- | --- | --- | --- |
| 0 | 25 | 1.9600 | 1.8726 |
| 75 | 16 | 1.6900 | 1.7313 |
| 150 | 16 | 1.6900 | 1.7313 |
| 225 | 16 | 1.6900 | 1.7313 |

Predicted excess coverage at phase 0 over the others is 16.0 percent in the
scored region and 8.2 percent in the core. Observed excess in detections is
14.7 percent scored, 344 against roughly 300, and 5.5 percent core, 288 against
roughly 273.

Both the direction and the rough magnitude track, and the excess shrinks from
scored to core in the prediction exactly as it does in the observation. That is
consistent with tile count being the cause. It is not a controlled test: the
tile count was not varied independently of the phase, so this is corroboration
rather than proof.

Consequences:

- Phase 0 sees more of the scored region more often, so it has more chances to
  detect a given crown and more chances to emit a false positive.
- Pooling it with the other 15 mixes two tiling regimes. Any per phase spread
  quoted across all 16, including the cv of 0.0269, carries that mixture.
- A clean version of the experiment would either hold tile count fixed across
  phases, for example by choosing a canvas size that admits the same count at
  every offset, or report phase 0 separately.

## Terminology

Following Miller, Dayoub, Milford and Sunderhauf, *Evaluating Merging
Strategies for Sampling-based Uncertainty Techniques in Object Detection*,
ICRA 2019, arXiv 1809.06006.

| term | meaning here |
| --- | --- |
| sample | one run of the detector over the scene, that is one grid phase. 16 samples. |
| observation | one detection box produced by one sample. |
| cluster | a set of observations from different samples judged to be the same underlying object. |
| support | the number of samples contributing an observation to a cluster, 1 to 16. |

Our clustering is **BSAS** in their taxonomy, Basic Sequential Algorithmic
Scheme, with **intra sample exclusivity**: at most one observation per sample
per cluster. That is their best performing configuration. Their semantic
affinity component does not apply, this problem has one class.

Their spatial affinity threshold is IoU 0.95, appropriate for MC Dropout
samples over an identical image where boxes are near coincident. Our samples
come from different tilings, so boxes legitimately shift, which is why our
threshold is far lower. The 0.3 to 0.5 sweep is the response to their finding
that results are sensitive to that threshold.

Variable names in the code are unchanged. This is documentation only.

## Related work

- **Miller, Dayoub, Milford and Sunderhauf, ICRA 2019, arXiv 1809.06006.**
  Merging strategies for sampling based uncertainty in object detection.
  Source of the clustering taxonomy, the intra sample exclusivity rule, and
  the spatial variance measure used here.
- **Zhang and Wang, 2016, arXiv 1611.06467,** *On The Stability of Video
  Detection and Tracking*. Showed that the stability metric has low
  correlation with the accuracy metric.
- **Tung et al., 2022, IEEE Multimedia, arXiv 2207.13890,** *Why Accuracy Is
  Not Enough: The Need for Consistency in Object Detection*. Reported
  consistency between 83.2 and 97.1 percent on video.

**The Miller hook.** Their spatial variance separates spatially accurate from
inaccurate observations. The result below says the inaccurate ones here are not
diffusely uncertain: they are locked to the processing grid, with a box edge
sitting exactly on a tile boundary. A variance measure cannot see that
difference. Two detections with identical coordinate variance can be, in one
case, a real crown the detector localises loosely, and in the other, a fragment
whose boundary is an artefact of where the grid fell.

Zhang and Wang and Tung et al. establish that aggregate metrics hide
instability. **Our contribution is sharper and narrower at once: the grid
position does not merely shift which real trees are found, it determines
whether a fragment is reported as a tree at all. It does this to two thirds of
the detections that appear only once, and to none of the detections that appear
every time.** The parameter that controls it is undisclosed, practitioner
controlled, and never reported.

**Scope, and it is not optional.** The seam mechanism explains the bottom of
the distribution. It does not explain the middle. Supports 5 to 15, 164
clusters, remain unaccounted for.

**And a null that belongs beside it.** Stability does not predict whether a
detection lands on a real tree. See Ground truth and matching. That is a second
instance of Zhang and Wang's result in a different domain, and it means the
grid position finding is about reproducibility, not about quality.

## Support structure results

At `MATCH_IOU` 0.5, core region only.

### Geometry by support band

| band | median width | median area | median aspect | above aspect 2 |
| --- | --- | --- | --- | --- |
| support 1 | 1.59 m | 2.16 m2 | 1.72 | 43.88 percent |
| support 16 | 2.49 m | 6.36 m2 | 1.05 | 0.00 percent |

Rank correlation with support:

| quantity | rho | reading |
| --- | --- | --- |
| width | +0.3043 | stable crowns are LARGER |
| area | +0.3605 | stable crowns are LARGER |
| aspect | -0.4597 | UNCONDITIONAL, confounded with seam pinning, see below |

**Correction on the record.** The hypothesis going in, taken from a visual
read of the first figure, was that unstable detections are larger and more
malformed. The size half is refuted: stable detections are larger, not smaller.
The shape half is UNCONDITIONAL and is confounded with seam pinning. Severing
is refuted: at the surveys where a crown was missed it was better contained
inside a tile, not worse. The slivers are manufactured by the tiling, not
severed by it. Once pinned clusters are excluded, median aspect at support 1
falls from 1.7185 to 1.0645 against 1.0516 at support 16, and the conditional
effect is r_rb +0.2223, verdict INCONCLUSIVE.

### Support 4

| quantity | value |
| --- | --- |
| crowns | 86 |
| structured, not scattered | 74, 86.05 percent |
| null expectation for structured | 2.42 percent |
| all_dy_one_dx, sensitive to dx | 40 |
| all_dx_one_dy, sensitive to dy | 31 |
| median aspect | 2.23, the highest of any support level |
| median area | 1.67 m2, the smallest of any support level |

The null is exact, every 4 subset of the 4 by 4 grid enumerated. 86.05 percent
against 2.42 percent is not a chance pattern.

### Axis marginals

| quantity | value |
| --- | --- |
| mean distinct dx per crown | 2.4141 |
| mean distinct dy per crown | 2.4704 |
| sign test on discordant pairs, two sided p | 0.313 |

Joint table mass sits in the corners: 196 at (1, 1), 201 at (4, 4), 40 and 31
at the off corners, middle nearly empty.

### How to state this finding

**Not as anisotropy.** There is no evidence that one axis matters more overall.
The two single axis classes are roughly balanced, 40 against 31, and the axis
marginals show no global bias at p 0.313.

**State it as: each crown is sensitive to exactly one axis, and which axis
varies by crown.** The corner heavy joint table says the same thing. A crown
is either robust to both axes or hostage to one, with very little in between.

### Per phase count spread

| set | n_core |
| --- | --- |
| 15 four tile phases | mean 273.80, cv 0.0245 |
| phase (0, 0), 25 tiles | 288, 5.19 percent above the 15 phase mean |

**Quote cv 0.0245 over the 15 four tile phases as the experiment's spread.**
The earlier figure, cv 0.0269 over all 16, mixed two tiling regimes.

The observed 5.19 percent excess sits below the 8.2 percent predicted from
mean tile coverage in the core. Direction and rough magnitude still agree.

## Seam mechanism

Three claims, each with its own status.

**Severing is refuted, and this does not rest on the centroid measure.** At the
grid positions where a detection was missed it was better contained inside a
tile, not worse: median containment margin 130.3292 px missed against 90.6744
px found. A crown cut by a seam would show the opposite.

**The sign test stands as a statement about the data.** Median distance from
box centroid to nearest boundary was 4.7736 px where found and 48.5241 px where
missed, two sided p 8.47e-22. The measurement is correct. The mechanism first
attached to it was not.

**The positive claim inverts.** These are not detections a seam spared. They
are fragments the tiling manufactured. `check_seam_pinning.py` compared box
edges against grid boundaries under a null that holds box size and shuffles
position: 63 of 71 single axis sensitive clusters have an edge within 1 px of a
boundary, median edge gap 0.0 px against a null median of 31.4 px, empirical p
below 1 in 1000.

### Across all 710 clusters

`check_seam_pinning_all.py` extends the test. Measurement moves to the
observation, since a cluster spanning many grid positions has no single grid:
gap on x against that observation's own dx grid, gap on y against its dy grid,
observation gap is the minimum of the two, cluster gap is the median over
observations. The null is built the same way so the two axis inflation cancels.

The 71 reproduce exactly under the looser rule, 0.8873 pinned, median gap 0.0
px, so the two statistics reconcile rather than one replacing the other.

| population | clusters | pinned | share | median gap |
| --- | --- | --- | --- | --- |
| support 1 | 196 | 136 | 0.694 | 0.13 px |
| support 2 to 15 | 399 | 166 | 0.416 | 3.60 px |
| support 16 | 115 | 0 | 0.000 | 12.46 px |
| all | 710 | 302 | 0.425 | 3.18 px |

| support | clusters | pinned | share | verdict |
| --- | --- | --- | --- | --- |
| 1 | 196 | 136 | 0.694 | PINNED |
| 2 | 79 | 43 | 0.544 | PINNED |
| 3 | 70 | 48 | 0.686 | PINNED |
| 4 | 86 | 65 | 0.756 | PINNED |
| 5 to 15, per level (dagger) | 164 total | 10 total | 0.000 to 0.200 | not reportable |
| 16 | 115 | 0 | 0.000 | not pinned |

**(dagger) Every level from 5 to 15 has n below 70 and its verdict is not
reportable.** The null share there collapses to exactly 0.0000, so one pinned
cluster returns an empirical p of 0 and the script prints PINNED. Per level
shares and counts are in `seam_pinning_all_by_support.csv`. The verdict column
is trustworthy only at supports 1 to 4 and at 16.

### The three statements this licenses

1. **96.7 percent of all pinning sits at supports 1 to 4**, 292 of 302 pinned
   clusters.
2. **Support 16 is zero of 115.** Not low, zero.
3. **Supports 5 to 15, 164 clusters with 10 pinned, have no account.** The seam
   mechanism does not explain them and nothing else does yet.

The 41.6 percent for the 2 to 15 band is correct and misleading if quoted
alone, because 156 of its 166 pinned clusters sit at supports 2, 3 and 4.

### The 2 to 15 band splits in two

- **Supports 2 to 4**, 235 clusters, 156 pinned, 66.4 percent. These behave
  like the singletons and share their mechanism.
- **Supports 5 to 15**, 164 clusters, 10 pinned, 6.1 percent. These do not.
  Nothing in this repository explains them.

Do not describe the band as one population.

### Singletons

136 of 196 are pinned. At support 1 the instability finding and the seam
finding are largely one mechanism, not two. The remaining 60 unpinned
singletons are separate and should not be described alongside the 136.

## Ground truth and matching

110 annotations on `core_clean.png` under `ANNOTATION_PROTOCOL.md`, converted
by `load_ground_truth.py`, matched by `match_ground_truth.py`. Detections are
dx225_dy075, 274 in the core. Scoring set 1 is live, canopy, certain, not edge
clipped, 64 annotations. Scoring set 5 is all 110.

### Over segmentation, measured

**Median 2.0 detections per annotated tree, and 33 of 64 trees carry two or
more.** Full distribution in `fig_detections_per_tree.png`: 6 with none, 25
with one, 17 with two, 5 with three, 4 with four, 7 with five or more, tail to
12. The two trees at 12 are the two largest annotations, 283 and 185 m2.

Containment rule: at least 50 percent of the DETECTION's own area inside the
annotation. A fragment inside a large crown counts, a box swallowing the crown
does not.

### The pair the results section turns on

True positive rule: one to one Hungarian assignment at IoU 0.5.

| scoring set | n_ann | TP | precision | recall | F1 | tree rate | det containment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 22 | 8.0 | 34.4 | 0.1302 | 90.6 | 50.0 |
| 5 | 110 | 28 | 10.2 | 25.4 | 0.1458 | 85.5 | 70.4 |

**Recall 34.4 percent against tree level detection rate 90.6 percent.** A
detector that finds nine of ten annotated trees scores an F1 of 0.13 because it
reports each of them two or three times.

**These rates have different denominators and different units and must never be
arithmetically combined.** Precision is per detection under a one to one rule,
recall is per annotation under the same rule, tree rate is per annotation under
a containment rule, detection containment is per detection under a containment
rule.

**Median matched IoU 0.664.** Localisation is not the failure mode. When one
detection stands for one tree it is a good box. The 42 unmatched annotations
are trees that were split, not trees found sloppily.

### Stability does not predict correspondence

| band | n | inside an annotation | share |
| --- | --- | --- | --- |
| found by all sixteen | 115 | 54 | 47.0 percent |
| found by some, not all | 146 | 77 | 52.7 percent |
| found once | 13 | 6 | 46.2 percent |

Scoring set 1. Fisher exact p 0.384 for all sixteen against some but not all,
p 1.000 against found once. Spearman -0.083. On scoring set 5: 70.4, 72.6 and
46.2 percent, Fisher p 0.782 and 0.114, Spearman -0.008.

**The null is robust to the choice of scoring set. That is not the same as
replicating five times.** The five sets are NESTED by construction: the 64 in
set 1 are a subset of the 73 in set 2, of the 84 in set 3, of the 102 in set 4,
of all 110 in set 5. They share most of their data, so the sign agreeing across
all five is close to guaranteed by the overlap and is not five independent
confirmations.

The smallest Fisher p anywhere in the five is **0.0753, at scoring set 3**. The
Spearman is negative on all five: -0.0832, -0.0956, -0.0933, -0.0507, -0.0076.
All five rows are in `ground_truth/match_metrics.csv`.

The null is reported as a sentence rather than a figure, because three bars at
roughly equal height with a 47.7 point interval on the third invites a reader
to see a gap that is not there.

### Annotation coverage, and what it forbids

The 110 annotations cover **67.5 percent** of the core by area, the 64 in
scoring set 1 cover **47.9 percent**.

**Detections outside every annotation cannot be read as false positives.** At
scoring set 5, 81 of 274 detections fall outside all 110 annotations, and 34 of
those 81 are support 16 detections agreed on by every survey. Either the
annotation missed them or they are consistent false positives. This experiment
cannot distinguish the two.

## Figures

| file | what it shows |
| --- | --- |
| `fig_support_histogram.png` | The U shaped support distribution, 710 clusters, median marked at 4.0. |
| `fig_pinning_by_support.png` | Pinned share by support. A high plateau at 1 to 4 and a floor after, NOT a monotone decline. Sparse levels hatched with n printed. |
| `fig_detections_per_tree.png` | Detections per annotated tree, scoring set 1, median marked at 2.0. |
| `figure_same_count_different_trees.png` | The non technical figure. See The figure below. |

## Everything here counts detections, not trees

**Over segmentation is now measured, and it is uncorrected.** Median 2.0
detections per annotated tree, 33 of 64 trees carrying two or more. See Ground
truth and matching. Every number in this file, the 710 included, remains a
count of DETECTIONS and not of trees, and none of them has been adjusted for
it. Any tree level claim needs the matching output, not these counts.

Ground truth now exists: 110 annotations, one annotator, no field
verification, photo interpretation of nadir RGB at 7.78 cm. Precision, recall
and F1 are therefore available and are reported above, conditional on
visibility rather than on the true stem population. See `ANNOTATION_PROTOCOL.md`
section 8 for the full limitation list.

The instability finding does not rest on any of that. Repeat surveys of one
photograph disagree with each other whether or not the annotation is right.

## The figure

`build_figure.py` writes `figure_same_count_different_trees.png` for a non
technical reader. Frame is same count, different trees. One population only:
the 274 detections on the list from a single survey. The 710 union, the 196
singletons and the 16.20 percent do not appear on it.

**Selection rule for the survey shown.** dx225_dy075, at the median of the
sixteen on both count and one off detections: 274 against a 15 position mean of
273.80, and 13 one off detections against a median of 13, min 6, max 17. The
earlier default dx075_dy075 has 6 one off detections, the fewest of any
position, and should not be used for anything published.
