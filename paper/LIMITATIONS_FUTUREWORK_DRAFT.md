# 6. Limitations

Draft v1, 2026-08-25. Target 500 words. Current: approximately 560.
Every number traced to `RESULTS_NUMBERS.md`. Bracketed notes are drafting
flags, not paper text.

---

**One site, one window, one detector.** Every result here comes from a single
scored region of a single orthomosaic, a closed conifer canopy in northern
California, processed by one pretrained detector at one checkpoint. We make no
claim about how large the effect is at other sites, in other canopy types, or
under other detectors. The mechanism we identify, boxes pinned to grid
boundaries, is a property of tiled inference rather than of this model, and we
expect it to be testable elsewhere, but we have not tested it elsewhere.

**Tile count is not independent of offset.** A zero offset on an axis admits
five tile origins where any nonzero offset admits four, so the sixteen
positions span three tiling regimes at twenty five, twenty and sixteen tiles.
The zero offset position sees more of the scored region more often and detects
correspondingly more. We report it separately and quote the fifteen position
spread throughout, but tile count was never varied independently of position, so
the agreement between the 5.19 percent observed excess and the 8.2 percent
predicted from tile coverage is corroboration rather than a controlled test. A
cleaner design would choose a canvas size admitting equal tile counts at every
offset.

**Cluster membership depends on the matching threshold.** The singleton count
runs 129, 167 and 196 as the clustering threshold tightens from 0.3 to 0.5, so
some singletons at 0.5 are one detection split across positions by a strict
rule. We quote the conservative end of every threshold sensitive quantity, but
the exact composition of the tails is not threshold invariant. The median
support of 4.0 and the roughly 55 percent share at intermediate support are.

**The middle of the support distribution has no account.** 164 clusters sit at
supports 5 through 15 with 10 pinned between them. Seam pinning explains the
tail and not the middle, and nothing in this work explains it. We state this as
a scope boundary rather than resolving it.

**The geometry test is inconclusive, not null.** At 60 unpinned singletons
against 115 fully supported clusters the design has roughly 80 percent power at
r_rb 0.30, and the observed effect sits below that with a confidence interval
excluding zero. We can say the dominant component of the unconditional shape
association is seam artefact. We cannot say a residual shape effect is absent.

**Ground truth rests on one annotator and photo interpretation.** 110 crowns
were delineated by a single annotator on nadir RGB at 7.78 cm, under a protocol
written and committed before annotation began, with no field verification. All
accuracy figures are therefore conditional on visibility rather than on the true
stem population, and no inter annotator agreement statistic is available. The
five scoring sets are nested by construction, at 64, 73, 84, 102 and 110
annotations, so the sign of the consensus null agreeing across all five reflects
shared data rather than five independent confirmations. The smallest Fisher p
anywhere among them is 0.0753.

**Annotation coverage forbids reading unmatched detections as errors.** The 110
annotations cover 67.5 percent of the scored core by area and the strictest
scoring set covers 47.9 percent. At the widest set, 81 of 274 detections fall
outside every annotation, and 34 of those are agreed on by all sixteen grid
positions. Either the annotation missed them or they are consistent false
positives, and this experiment cannot distinguish the two.

**All counts are detections, not trees.** Over segmentation is measured at a
median of 2.0 detections per annotated tree, with 33 of 64 trees carrying two or
more, and it is uncorrected. The 710 clusters and every per position total
remain counts of detections. No tree level claim in this paper rests on them.

Finally, the instability finding does not depend on the annotation being
correct. Repeated surveys of one photograph disagree with each other whether or
not the reference is right.

---

# 7. Future Work

Draft v1. Target 300 words. Current: approximately 310.

---

**Replication at a second site.** The cheapest useful extension is to repeat the
sweep on an orthomosaic with a different canopy structure and report the support
distribution and pinned share without annotating it. The instability finding
requires no ground truth, so a second site costs inference and clustering alone.
That would establish whether the seam pinning mechanism and the shape of the
support distribution survive a change of canopy, which is the single largest
open question about the generality of this result.

**Crown shape from segmentation rather than boxes.** Our geometry test is
limited to axis aligned bounding boxes, which admit aspect ratio and area and
nothing else. Prompting a segmentation model with the existing boxes would yield
crown polygons and therefore circularity, solidity and fitted ellipse
eccentricity, letting the shape question be asked of planform outline rather
than of a box. In closed canopy with touching crowns, mask bleed into
neighbouring crowns would require a manual quality audit before any such measure
could be trusted.

**Three dimensional crown form.** Whether a crown is conical or rounded is a
property of its vertical profile, and no nadir planform measure recovers it. A
canopy height model or dense photogrammetric cloud sampled inside crown masks
would give apex sharpness, profile skew and crown ratio. That is the version of
the shape question worth asking, and it is out of reach of imagery alone.

**Mitigation, and why it is not obvious.** Aggregating detections across
multiple grid positions is the natural fix, but our consensus null argues
against the simplest form of it: agreement across positions does not predict
correspondence with an annotated tree, so a vote threshold would filter
reproducibly rather than accurately. What aggregation would buy is
reproducibility, not precision. Establishing which aggregation rule, if any,
recovers both is left open.

**Reporting.** The immediately actionable outcome is smaller than any of the
above: report the tile origin alongside tile size and overlap, and release the
tiling code. A result obtained at an undisclosed grid offset is not reproducible
by anyone else.

---

## Drafting flags

- **Section 6, paragraph 1** claims the mechanism is a property of tiled
  inference rather than of DeepForest. That is reasoning, not a measurement, and
  it is the one inferential step in this section. It is phrased as an
  expectation rather than a finding. Keep it that way, or cut it.
- **Section 7, mitigation paragraph** is the strongest content here and is the
  paragraph most likely to draw a reviewer question, because it argues against
  the obvious fix. Verify that the consensus null as written in 4.6 actually
  supports it before this ships. It does as I read it, but the inference runs
  through two sections.
- **No inter annotator agreement statistic is quoted** because none exists for
  this dataset. Elicit prompt 11 is looking for a literature value that would
  let us say what a second annotator would plausibly have changed. If it lands,
  add one sentence here. If it does not, this stays as written.
- **The second site paragraph** does not name a candidate mission. Naming one we
  have not checked for GSD, canopy type, and withheld from training status would
  be an unsupported claim. Leave it unnamed.
- **Word budget.** 6 and 7 together run approximately 870 against a skeleton
  budget of 800. Trim from 6 if the page count binds; the coverage and threshold
  paragraphs are the most compressible.
