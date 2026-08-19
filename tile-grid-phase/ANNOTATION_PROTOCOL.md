# Ground truth annotation protocol, tile grid phase experiment

Version 1.0. Written before annotation began.

Target image: `core_clean.png`, 950 by 950 px, 7.78 cm per pixel, 73.9 m
square. Northern California closed conifer canopy, Open Forest Observatory
Mission 000103, CC BY 4.0.

This file is the protocol of record. If a rule changes mid annotation, stop,
revise this file, note the version, and re annotate anything drawn under the
old rule. Do not carry two rule sets in one dataset.

---

## 0. Why each rule exists

Every rule below answers a question a reviewer can ask. The rules are not
chosen for convenience. They are chosen so that when someone asks "how did
you decide that", the answer is written down and was written down first.

The single largest threat to this dataset is not imprecision. It is
circularity: annotating in a way that is shaped, consciously or not, by what
the detector already found. Section 1 exists entirely to prevent that.

---

## 1. Before you draw anything

**1.1 Annotate blind.** Do not have `support_dx225_dy075_iou05.png`,
`support_pooled_singletons_iou05.png`, `figure_same_count_different_trees.png`,
or any per phase CSV open, visible, or loaded in the annotation tool. Close
them. If you have looked at them recently, that is unavoidable at this point,
but do not consult them while drawing.

This is the rule that makes the resulting metrics mean anything. Ground truth
drawn while looking at predictions measures your agreement with the detector,
not the detector's agreement with the forest.

**1.2 Record the coordinate frame before the first box.** `core_clean.png`
was cropped by `CORE_INSET` = 25 px. Chip pixel (0, 0) is window pixel
(25, 25).

Decide now, write it in the output file header, and do not change it:

> Annotations are stored in CHIP pixel coordinates. Add 25 to both x and y to
> obtain window coordinates comparable to `phase_boxes_*.csv` and
> `phase_stability.csv`.

A silent 25 px offset will look like a systematic spatial bias in every match
and will be very hard to diagnose later.

**1.3 Set the tool up for boxes, not polygons.** DeepForest emits axis aligned
boxes and every match in this experiment is box IoU. A polygon would have to
be reduced to its bounding box before it could be compared, so the extra
precision is discarded at scoring time. Draw boxes.

**1.4 Pilot before committing.** Annotate one 475 by 475 px quadrant first,
under these rules, and stop. Count how long it took and how many decisions
felt genuinely ambiguous. If more than roughly one in ten boxes required a
coin flip, the rules need tightening before you spend hours applying them to
the full chip. Revise this file, bump the version, and restart the quadrant.

---

## 2. The unit rule: what counts as one tree

**2.1 One apex, one tree.** In nadir imagery of closed conifer canopy, an
individual tree presents as a roughly radial crown with an identifiable apex,
the sunlit leader at the top of the tree, with branch structure radiating
outward and a shadow cast to one side. Each distinct apex is one tree.

This is the rule that decides the over segmentation question. A bright branch
cluster on the flank of a larger crown is not a tree, because it has no apex
of its own. A separate leader rising through the canopy is a tree, even if its
crown is partly overlapped by a neighbour.

**2.2 When you cannot find an apex, do not annotate.** If a patch of foliage
has no identifiable leader, it belongs to a neighbouring crown or it is
understory too obscured to resolve. Either way it is not an annotation.

**2.3 Interlocking crowns.** Closed canopy means crowns touch and overlap.
Two apices with continuous foliage between them are still two trees. Draw two
boxes. They will overlap. That is correct and expected, and box overlap in
ground truth is normal in this forest type.

**2.4 Do not annotate what you cannot see.** If a crown is mostly hidden
beneath a neighbour and only a fragment is visible, and that fragment has no
apex, it is not annotated. Absence of an annotation means "not resolvable in
this image", not "no tree present". Section 8 records this as a known
limitation.

---

## 3. Inclusion and exclusion

**3.1 Live conifers with a visible apex.** Annotate. This is the primary
class.

**3.2 Snags and standing dead trees.** Annotate, and flag them. There are
several bright white dead crowns clearly visible on this chip and the detector
is placing boxes on some of them.

Flagging rather than excluding matters. If you exclude snags from ground truth
while the detector reports them, every snag detection becomes a false positive
by construction, and you will have manufactured an error rate out of a
labelling choice. Flag them, then report metrics both ways: all trees, and
live trees only.

**3.3 Understory trees visible through canopy gaps.** Annotate if and only if
an apex is visible and the crown meets the minimum size in 3.5. Flag them as
understory. These are the hardest calls and separating them lets you report
metrics with and without.

**3.4 Broadleaf or non conifer individuals, if any are present.** Annotate and
flag by class. Do not silently fold them into the conifer class.

**3.5 Minimum crown size.** Set a floor and apply it without exception.

Suggested starting value: **1.5 m crown diameter**, which is about 19 px at
7.78 cm. Below roughly this size at this resolution you cannot reliably tell a
small tree from a lit branch of a larger one, which is the exact failure that
made native resolution unusable for this experiment.

Fix the value during the pilot, record it here, and never change it mid
dataset. Record the value you chose:

> Minimum annotated crown diameter: ______ m ( ______ px )

**3.6 Non tree objects.** Bare ground, rock, shadow, downed logs, and the
road or skid trail visible at the lower left are not annotated. If the
detector reports boxes there, those are false positives and that is a real
result.

---

## 4. Box geometry

**4.1 Box the visible foliage, not the inferred crown.** Draw the tightest
axis aligned box containing the foliage you can actually see belonging to that
apex. Do not extend the box to where you believe the crown continues beneath a
neighbour. Inferred extent is not observable and cannot be checked by a second
annotator.

**4.2 Do not include the shadow.** Cast shadow is not crown. The sun angle in
this image is consistent, so including shadow would introduce a systematic
directional bias into every box.

**4.3 Include the full apex.** The box must contain the leader. If it does
not, the box is not describing the tree it claims to.

**4.4 Consistency beats precision.** A box drawn 5 px too large on every tree
is far less damaging than boxes drawn inconsistently. Pick a convention for
where foliage ends against dark background, apply it uniformly, and do not
agonise over individual pixels.

---

## 5. Chip edge handling

**5.1 Apex inside, annotate. Apex outside, do not.** A crown whose apex falls
inside the chip is annotated even if its box is clipped by the edge. A crown
whose apex falls outside is not annotated even if foliage intrudes.

**5.2 Flag every edge clipped box.** Any box touching the chip boundary gets
`edge_clipped = 1`. Edge clipped boxes have artificially reduced extent and
will match poorly on IoU through no fault of the detector. You will want to be
able to exclude them from scoring and report both numbers.

**5.3 Do not extend a box beyond the chip.** Clip it at the boundary.

---

## 6. What to record per box

Minimum columns. Add more if useful, never fewer.

| column | values | notes |
| --- | --- | --- |
| `tree_id` | integer, unique | sequential, never reused |
| `xmin`, `ymin`, `xmax`, `ymax` | float, chip px | see 1.2 on the frame |
| `class` | live / snag / other | 3.1, 3.2, 3.4 |
| `layer` | canopy / understory | 3.3 |
| `edge_clipped` | 0 / 1 | 5.2 |
| `confidence` | certain / uncertain | 6.1 |
| `note` | free text, optional | why a hard call went the way it did |

**6.1 Flag your own uncertainty, and use the flag freely.** Mark `uncertain`
whenever you would not defend the box to a colleague looking over your
shoulder. Do not treat it as an admission of weakness. A dataset that reports
metrics on certain only, and again on all boxes, is more informative and more
honest than one that pretends every call was clean.

Expect a meaningful share of `uncertain` in closed canopy. That is the forest,
not the annotator.

**6.2 Write the note while the call is fresh.** A one line note on hard cases
is what lets you re apply the same reasoning three hours later on the far side
of the chip.

---

## 7. Quality control

**7.1 Intra annotator repeat.** You are one annotator, so inter annotator
agreement is not available. The accepted substitute is to re annotate a
subset, blind to the first pass, and report agreement between your two passes.

Re annotate **10 to 15 percent** of the chip. Ideally leave at least several
days between passes. If the schedule does not allow that, do it same session
with the first pass hidden, and say so plainly when reporting. A weak
agreement statistic that is described accurately is citable. One that is
described as something it is not, is not.

**7.2 Report agreement as both a count and an overlap statistic.** How many
trees appeared in both passes, and the median IoU of the matched boxes. Those
two numbers together tell a reader whether your disagreement is about
existence or about extent, and those have different consequences.

**7.3 Work in passes, not in one sweep.** First pass, obvious canopy trees.
Second pass, hard calls and understory. Third pass, review the whole chip for
misses. Fatigue in a single long sweep produces a spatial gradient in
annotation density, which will look exactly like a real spatial pattern in the
results.

**7.4 Record the time spent.** Annotation time per crown is a number reviewers
ask for and it is trivially easy to capture at the time and impossible to
reconstruct later.

---

## 8. Known limitations to state in the paper

Write these down now, so they are recorded as anticipated rather than
discovered under questioning.

1. Single annotator. No inter annotator agreement is available. Intra
   annotator repeat is a weaker substitute and is reported as such.
2. The annotator is not a trained field forester. Species level judgments are
   not made and none are claimed.
3. No field verification. Nothing here is validated against a ground survey,
   stem map, or lidar. This is photo interpretation.
4. Crowns not resolvable in nadir RGB, fully suppressed understory in
   particular, are absent from the ground truth. Detection metrics are
   therefore conditional on visibility, not on the true stem population.
5. Annotation was performed on a 2x downsampled image at 7.78 cm, not at
   native 3.89 cm resolution.
6. The chip is one 73.9 m square window from one orthomosaic at one site in
   one forest type. Nothing generalises beyond it without further work.

---

## 9. What this unlocks

Once this exists, and not before, the following become answerable:

- Whether a detection is a whole tree or a fragment of one, which is the over
  segmentation question that has been open since the start and is currently
  uncorrected in every count in this experiment.
- Whether the 136 pinned singletons are fragments of real trees, which the
  seam result implies but has not shown against truth.
- Whether the 115 detections found by all sixteen surveys correspond to real
  trees, and whether stability is a usable proxy for correctness.
- Whether the 164 unexplained clusters at supports 5 to 15 are real trees
  detected intermittently, or something else.

Until it exists, no claim about accuracy, correctness, precision, or recall
is available anywhere in this project.
