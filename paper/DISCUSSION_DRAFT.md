# 5. Discussion

Draft v1, 2026-08-25. Target 700 words. Current: approximately 740.
Bracketed notes are drafting flags, not paper text.

---

## 5.1 A parameter that is free, consequential, and mostly invisible

Once patch size and overlap ratio are chosen, the origin of the tiling grid
remains free, and the DeepForest interface exposes no control over it. That
combination is what makes this parameter awkward. It is not a setting a
practitioner tunes badly; it is a setting most practitioners do not know they
are setting. The grid origin falls where the boundaries of the supplied image
put it, so the phase is determined by how the area of interest was cropped.

The practical consequence follows directly. Two analysts working from the same
orthomosaic, running the same detector at the same patch size and overlap, who
crop their area of interest a few dozen pixels apart, will receive different
lists of trees. On our data the two lists would agree on roughly a sixth of the
detections at full agreement and would differ in the identity of a substantial
share of the rest, while reporting totals within a few percent of each other.
Neither analyst would have made a mistake, and neither would have any indication
from their outputs that anything had happened.

## 5.2 Why aggregate metrics conceal this

Detection count across the fifteen positions we treat as comparable has a
coefficient of variation of 0.0245. Any check based on totals, and any accuracy
metric computed independently per run against a fixed reference, returns
substantially the same answer at every grid position. The instability is
invisible to them because it is not a change in how many objects are reported or
in how well they score; it is a change in which objects those are.

Seeing it requires matching runs against each other rather than each against
ground truth. That is a cheap operation and it is not part of any standard
evaluation protocol we are aware of. The broader implication is that a detector
can post stable, reproducible looking aggregate numbers while its per object
output is substantially irreproducible, and nothing in the usual reporting would
reveal the gap. [Sharpen once prompt 2 gives the reporting tally.]

## 5.3 The uncertainty here is structured, not diffuse

Miller et al. extract spatial uncertainty as the coordinate variance of boxes
within a cluster, and show it separates spatially accurate from inaccurate
observations. Our result says something more specific about what the inaccurate
ones are. They are not diffusely uncertain. They are locked to the processing
grid, with a box edge sitting exactly on a tile boundary in 136 of 196 cases at
support 1 and in none of 115 at support 16.

A variance measure cannot see that distinction. Two clusters with identical
coordinate variance may be, in one case, a real crown the detector localises
loosely across positions, and in the other, a fragment whose boundary is an
artefact of where the grid fell. Distance from a box edge to the nearest grid
boundary distinguishes them and costs nothing to compute. We suggest it as a
diagnostic alongside variance rather than in place of it.

This also revises the intuitive account of why tiling hurts. Severing, the idea
that a boundary cuts a crown and the fragment is lost, predicts that missed
detections were worse contained inside a tile. Our data show the opposite sign.
The mechanism is not destruction of real objects at boundaries but manufacture
of spurious ones along them.

## 5.4 Consensus across positions is not a remedy

The natural mitigation is to run several grid positions and keep what they agree
on. Our results argue against the simplest form of that. Support does not
predict whether a detection lands inside an annotated tree, with the correlation
slightly negative on all five scoring sets and the smallest Fisher p anywhere
being 0.0753. Of the 81 detections falling outside every annotation at the
widest scoring set, 34 are agreed on by all sixteen positions. Unanimity across
grid positions is not evidence that a tree is there.

A vote threshold would therefore select for reproducibility rather than for
accuracy. That is not worthless: reproducibility is exactly what is currently
missing, and an aggregation rule that makes the output independent of the crop
would be a real improvement even if it left precision unchanged. But it should
be adopted for that reason and described in those terms, not as a quality
filter. This is a second instance, in a different domain and under a different
perturbation, of the decoupling between stability and accuracy that Zhang and
Wang report for video.

## 5.5 Scope

Everything above rests on one orthomosaic, one window, one canopy type and one
detector. The mechanism we identify is geometric rather than model specific,
which is why we expect it to appear wherever inference is tiled, but expectation
is not evidence and we have not tested it elsewhere. What the result licenses is
narrower and, we think, still worth saying: the grid origin belongs in the
reported configuration alongside patch size and overlap, and a result obtained
at an undisclosed origin cannot be reproduced by anyone else.

---

## Drafting flags

- **5.1, second paragraph, contains the paper's strongest practitioner
  sentence** and also its loosest number. "Roughly a sixth" is 16.20 percent and
  is fine; "a substantial share of the rest" is deliberately vague because the
  exact disagreement rate between two arbitrary positions is not a quantity we
  computed. Either compute it or leave the vagueness. Do not invent a figure.
- **5.2 has a placeholder.** The claim that no standard protocol matches runs
  against each other is currently asserted from absence of knowledge. Elicit
  prompts 2 and 4 should either support it or force it to soften. Do not ship
  the sentence until one of them reports.
- **Tung et al. is not cited here, deliberately.** Their reported consistency of
  83.2 to 97.1 percent on video sits arrestingly against our 16.20 percent, and
  the comparison is not valid: different perturbation, different correspondence
  rule, different domain. It belongs in Related Work as context, not in
  Discussion as a contrast.
- **5.3 is the section Elizabeth is most likely to react to**, since the
  diagnostic suggestion is the closest thing here to a methods contribution
  someone else could adopt. If she wants the computing contribution foregrounded,
  this is the paragraph to expand.
- **5.4 does not overclaim the mitigation.** It says aggregation would buy
  reproducibility and not precision, which follows from 4.6, and it stops there
  rather than proposing a specific rule. Proposing one without testing it would
  be an unsupported claim.
- **Word budget.** 740 against 700. Section 5.5 is the compressible one; it
  partly restates Limitations and can drop to two sentences if the page count
  binds.
