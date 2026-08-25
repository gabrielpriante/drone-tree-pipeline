# 4. Results

Draft v1, 2026-08-25. Target 1600 words. Current: approximately 1650.
Every number traced to `RESULTS_NUMBERS.md`. Bracketed notes are drafting
flags, not paper text.

---

## 4.1 Aggregate detection counts are stable across grid positions

Moving the tiling grid changes how many detections the detector reports by very
little. Across the fifteen positions at which neither offset is zero, the mean
count inside the scored core is 273.80 with a coefficient of variation of
0.0245. A practitioner who ran the pipeline twice at different offsets and
compared only totals would conclude the procedure was reproducible.

Those fifteen positions are not a homogeneous set, and we report them as three
tiling regimes rather than two. Tile origins are emitted at a fixed stride
across a canvas of fixed size, so an offset of zero on an axis admits five
origins on that axis while any nonzero offset admits four. The sixteen
positions therefore comprise one position with twenty five tiles, six with
twenty, and nine with sixteen, with mean core counts of 288, 277.50 and 271.33
respectively. The zero offset position sits 5.19 percent above the fifteen
position mean, against an excess of 8.2 percent predicted from mean tile
coverage in the core. Direction and rough magnitude agree, which is consistent
with tile count driving the difference, though tile count was not varied
independently of phase and this is corroboration rather than a controlled test.

We report the pooled fifteen position figures as the experiment's spread and
treat the zero offset position separately throughout.

## 4.2 Detection identities are not stable

Clustering detections across the sixteen grid positions at an intersection over
union threshold of 0.5 yields 710 distinct clusters. Of these, 115, or 16.20
percent, are recovered at every one of the sixteen positions. 196 clusters, or
27.61 percent, are recovered at exactly one. The median support is 4.0: the
typical cluster is found at a quarter of the grid positions. The distribution
is U shaped, with a pile at support 1, a local peak at support 4, a trough
through the middle, and a spike at support 16 [Figure 2].

Two features of this distribution do not depend on where the matching line is
drawn. Repeating the clustering at thresholds of 0.3 and 0.4 leaves the median
support at exactly 4.0 and holds the share of clusters at supports 2 through 15
between 54.65 and 56.20 percent. Most detections are neither fully stable nor
one offs.

Two other features are threshold sensitive, and we quote the conservative end
of each. The singleton count runs 129, 167 and 196 as the threshold tightens
from 0.3 to 0.5, so some of the singleton pile at 0.5 is one detection split
across positions by a strict rule. The share recovered at all sixteen positions
moves in the other direction, from 23.52 percent at 0.3 to 16.20 percent at
0.5. We therefore claim 129 singletons and 16.20 percent full agreement, each
being the least favourable value available to us.

The contrast with Section 4.1 is the paper's central observation. The count is
reproducible to within roughly two percent. The membership of the detection set
is not.

## 4.3 The unstable tail is pinned to the processing grid

Detections found at exactly four of the sixteen positions are overwhelmingly
structured rather than scattered: 74 of 86, against an exact null expectation of
2.42 percent. Of the 74, 40 are recovered at every offset on one axis and a
single offset on the other, and 31 show the reverse. The axis marginals show no
global bias, with a mean of 2.4141 distinct x offsets against 2.4704 distinct y
offsets and a two sided sign test at p 0.313. The finding is that each such
detection is sensitive to exactly one axis and that which axis varies by
detection, not that one axis matters more overall.

The initial hypothesis for these 71 was severing: a crown cut by a tile seam
leaves a fragment that survives only when the seam falls elsewhere. **That
hypothesis is refuted.** At the positions where a detection was missed it was
better contained inside a tile, not worse, with a median containment margin of
130.3292 px at the missed samples against 90.6744 px at the found samples.
Severing predicts the opposite sign.

The mechanism is instead seam pinning. Comparing each box edge against the grid
boundaries on its sensitive axis, under a null that holds box size and offset
fixed and shuffles position, 63 of the 71 have an edge within one pixel of a
boundary. The observed median edge gap is 0.0 px against a null median of
31.4169 px with a 95 percent null band of 22.29 to 40.68 px, and the observed
pinned share of 0.8873 sits against a null share of 0.0282 with a band of
0.0000 to 0.0704. Empirical p is below one in a thousand on both quantities.
These are not crowns a seam spared. They are fragments the tiling manufactured.

Extending the same test to all 710 clusters, 302, or 42.5 percent, are pinned.
The distribution by support is not a monotone decline but a plateau followed by
a floor [Figure 3]: 136 of 196 singletons are pinned (69.4 percent, median gap
0.13 px), 156 of the 235 clusters at supports 2 through 4 are pinned (66.4
percent), and **zero of the 115 clusters at support 16 are pinned** (median gap
12.46 px). 292 of the 302 pinned clusters, or 96.7 percent, sit at supports 1
through 4. A detection recovered at every grid position never has an edge on a
grid boundary.

The band from support 2 to 15 must not be quoted as one population. Its 41.6
percent pinned share is correct as computed and misleading in isolation,
because 156 of its 166 pinned clusters sit at supports 2, 3 and 4. The
remaining 164 clusters at supports 5 through 15 carry 10 pinned clusters
between them, and the seam mechanism does not account for them. We return to
this in Section 6.

## 4.4 Footprint geometry does not survive the seam control

An obvious competing explanation is that unstable detections are simply
different in shape, and that the tiling is incidental. Unconditionally, the
association is strong: rank correlations with support are +0.3043 for box
width, +0.3605 for box area and -0.4597 for aspect ratio. Median aspect falls
from 1.7185 at support 1 to 1.0516 at support 16, and while 43.88 percent of
singletons exceed an aspect of 2, none of the 115 fully supported clusters do.

The size half of that reading refutes the intuition it was formed against:
stable detections are larger, not smaller. The shape half is confounded with
seam pinning, since 136 of the 196 singletons are pinned and a fragment cut at a
boundary is elongated by construction.

Excluding pinned clusters resolves the confound. Against 60 unpinned singletons
and all 115 fully supported clusters, median aspect at support 1 falls from
1.7185 to 1.0645, against 1.0516 at support 16 [Figure 4]. Under a two sided
Mann Whitney U test with Holm correction across two prespecified endpoints, the
aspect effect is r_rb +0.2223 with a bootstrap 95 percent confidence interval of
+0.0295 to +0.4133, AUC 0.6112, adjusted p 0.0320. The secondary endpoint, log
box area, gives r_rb -0.1467 with an interval of -0.3270 to +0.0409 and adjusted
p 0.1121. A size matched subsample at 58 against 58 returns r_rb +0.2420,
confirming the residual aspect effect is not a size artefact, and edge gap is
not a predictor once shape and size are in the model (p 0.5611).

Against thresholds fixed before any of these numbers were seen, **the verdict
is inconclusive.** The effect misses the 0.30 bar required to rule geometry in,
and the interval excludes zero, so geometry cannot be ruled out either. At 60
against 115 the design has roughly 80 percent power at r_rb 0.30, and effects
smaller than that cannot be resolved here. What the test establishes is that
the dominant component of the unconditional association is seam artefact, and
that what remains is too small for this design to characterise.

## 4.5 The detector finds most annotated trees and reports each one repeatedly

110 crowns in the scored core were annotated under a protocol written and
committed before annotation began. Matching the 274 detections from a single
representative grid position against the 64 annotations in the strictest
scoring set gives a median of 2.0 detections per annotated tree, with 33 of the
64 carrying two or more. The distribution runs 6 trees with no detection, 25
with one, 17 with two, 5 with three, 4 with four and 7 with five or more,
reaching 12 at the tail; the two trees at 12 are the two largest annotations,
at 283 and 185 m2.

Under one to one Hungarian assignment at IoU 0.5, that scoring set yields
precision 8.0 percent, recall 34.4 percent and F1 0.1302, against a tree level
detection rate of 90.6 percent under a containment rule [Table 1]. The pair is
the argument: a detector that finds nine of every ten annotated trees scores an
F1 of 0.13 because it reports each of them two or three times. Localisation is
not the failure mode. Median matched IoU is 0.664, comfortably above the 0.5
bar, and the 42 unmatched annotations are trees that were split rather than
trees that were found sloppily.

These rates carry different denominators and different units and must not be
arithmetically combined. Precision is per detection under the one to one rule,
recall is per annotation under the same rule, and the tree level rate is per
annotation under a containment rule.

## 4.6 Consensus across grid positions does not predict correspondence

If instability were a quality signal, detections agreed on by every grid
position would land on annotated trees more often than those found once. They
do not. Of the 115 detections at full support, 47.0 percent fall inside an
annotation; of the 146 at intermediate support, 52.7 percent do; of the 13
found once, 46.2 percent do. Fisher exact tests return p 0.384 for full support
against intermediate and p 1.000 against found once, and the Spearman
correlation between support and correspondence is -0.083.

Repeating this on the widest scoring set gives 70.4, 72.6 and 46.2 percent,
with Fisher p 0.782 and 0.114 and a Spearman of -0.008. The sign is negative on
all five scoring sets, at -0.0832, -0.0956, -0.0933, -0.0507 and -0.0076, and
the smallest Fisher p anywhere among them is 0.0753. We report this as
robustness to the choice of scoring set rather than as five replications, since
the five sets are nested by construction and share most of their data.

Consensus across grid positions is therefore not a filter for detection
quality. The instability documented in Section 4.2 is a reproducibility
problem, not a hidden accuracy signal, and cannot be resolved by requiring
agreement across offsets.

---

## Drafting flags

- **Figure references** are placeholders. Numbering matches the skeleton: Fig 2
  support histogram, Fig 3 pinning by support, Fig 4 geometry by tier, Table 1
  metrics by scoring set.
- **Table 1 is not drafted here.** It is a direct transcription of the two
  scoring set rows and belongs in the LaTeX, not in prose.
- **Section 4.5 opening** says "a single representative grid position."
  The selection rule (median on count and on one off detections) belongs in
  Method, not Results. Confirm it is stated there before this sentence ships.
- **Section 4.3** cites 71 single axis sensitive detections derived from the 86
  at support 4. The step from 74 structured to 71 single axis is not spelled
  out in the source docs in a form I could state without inferring. Verify
  before this ships.
- **The 8 unpinned members of the 71** are omitted here deliberately. They are
  ordinary sized and roughly square and are a different phenomenon. If a
  reviewer asks, they go in a footnote, not the body.
- **No claim about DeepForest's behaviour in general** appears anywhere above.
  Every statement is about this orthomosaic, this window and this resolution.
  Keep it that way.

---

## What I deliberately did not do

* No claim about DeepForest generally. Every sentence is scoped to this orthomosaic, this window, this resolution. That scoping is what makes the paper defensible on one site.
* The 8 unpinned members of the 71 are omitted. They are a different phenomenon and the docs say not to describe them alongside the 63.
* The consensus null stays prose, per the decision not to draw three near equal bars.
* No hedging language added beyond what the numbers support. Where the verdict is inconclusive I say inconclusive.

## Two places where the draft is doing rhetorical work you should be aware of

Section 4.2's closing line ("The count is reproducible to within roughly two percent. The membership of the detection set is not.") is the paper's thesis compressed into two sentences. If you want a different emphasis, this is the sentence to change.

Section 4.4 frames geometry as a competing explanation being eliminated, which is what makes an inconclusive result useful rather than a loose end. The subsection title says "does not survive the seam control," which is accurate: the unconditional effect does not survive. It does not claim geometry is null.
