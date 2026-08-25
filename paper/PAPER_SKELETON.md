# Paper Skeleton

State as of 2026-08-25. Target: arXiv preprint, October 1 2026.

## 1. Target and budget

- **Format.** 8 pages, two column, roughly 6000 body words. References not counted.
- **Venue class.** Written to CV workshop length so it can be submitted later without cutting.
- **Frame A.** Tile grid phase is an uncontrolled variable in a widely used inference procedure. Methods critique, not an ecology result.
- **Working title.** *Same Count, Different Trees: Tile Grid Phase as an Uncontrolled Variable in Individual Tree Crown Detection*

| section | words | figures |
| --- | --- | --- |
| Abstract | 200 | |
| 1 Introduction | 700 | Fig 1 |
| 2 Related Work | 700 | |
| 3 Method | 1200 | |
| 4 Results | 1600 | Fig 2, 3, 4 + Table 1 |
| 5 Discussion | 700 | |
| 6 Limitations | 500 | |
| 7 Future Work | 300 | |
| 8 Conclusion | 150 | |
| **total** | **6050** | **4 figures, 1 table** |

## 2. Section list, one line each

1. **Introduction.** A free parameter nobody reports changes which trees a detector finds, and this paper measures how much.
2. **Related Work.** Detection stability and consensus merging have been studied on video and on MC Dropout samples; grid phase has not.
3. **Method.** Dataset, working window, resolution choice, the 16 phase sweep, cross phase clustering, the seam pinning null, ground truth protocol, matching rules.
4. **Results.** Counts are stable, identities are not; seam pinning explains the tail; geometry does not survive the control; consensus does not predict correspondence.
5. **Discussion.** What this means for anyone running tiled inference, and why aggregate metrics conceal it.
6. **Limitations.** One site, one annotator, over segmentation uncorrected, supports 5 to 15 unexplained, phase 0 confound.
7. **Future Work.** Second site, planform segmentation, 3D crown form, phase ensembling as a mitigation.
8. **Conclusion.** Report the grid offset, or the result is not reproducible.

## 3. Claim ladder

Everything in the draft supports one of these or gets cut.

1. **Tile grid phase is a free parameter of tiled inference.** It is practitioner controlled, undisclosed by default, and absent from every reported configuration we found.
2. **Varying it alone leaves counts stable and identities unstable.** Detection count cv is 0.0245 across the 15 four tile positions. Only 16.20 percent of clusters appear at all 16 positions. Median support is 4 of 16.
3. **The unstable tail has a mechanism, and it is not the obvious one.** Severing is refuted. The mechanism is seam pinning: 136 of 196 singletons have a box edge sitting exactly on a grid boundary, against zero of 115 at support 16. These are fragments the tiling manufactured.
4. **Consensus across grid positions is not a quality filter.** Support does not predict whether a detection lands inside an annotated tree. Spearman is negative on all five scoring sets; the smallest Fisher p anywhere is 0.0753.
5. **The middle of the distribution is unexplained.** 164 clusters at supports 5 to 15, 10 of them pinned. Stated as a scope boundary, not solved.

Claim 3 carries the paper. Claim 4 is the result a reader will remember. Claim 5 is what makes the other four credible.

## 4. Number to section map

Every headline number gets exactly one home. Numbers not listed here are supporting detail or get cut.

| section | numbers |
| --- | --- |
| **3 Method** | 11632 x 12458 px, EPSG:32610, 3.89 cm GSD; window col_off 4820 row_off 5260 size 2000; 7.78 cm experiment resolution; gate 1201 vs 311 detections, median box 0.97 m vs 2.10 m; patch 400, overlap 0.25, stride 300; offsets 0/75/150/225; NMS IoU 0.15; MATCH_IOU 0.5; CORE_INSET 25 px; tiler validation 311 of 311, median IoU 0.9344; 110 annotations, 5 nested scoring sets, set 1 = 64 |
| **4.1 Counts stable** | mean 273.80, cv 0.0245 over 15 positions; phase 0 = 288, 5.19 percent above; three tiling regimes 25/20/16 tiles at 1/6/9 positions |
| **4.2 Identities unstable** | 710 clusters; 115 at all 16 (16.20 percent); 196 singletons at IoU 0.5, 129 at 0.3 as the conservative end; median support 4.0; 2 to 15 band near 55 percent at every threshold |
| **4.3 Seam pinning** | severing refuted, containment margin 130.3292 px missed vs 90.6744 px found; 63 of 71 pinned, median edge gap 0.0 px vs null median 31.4169 px, empirical p < 0.001; 302 of 710 pinned overall; 136 of 196 singletons; 0 of 115 at support 16; 96.7 percent of pinning at supports 1 to 4 (292 of 302) |
| **4.4 Geometry ruled out as a competing explanation** | unconditional aspect rho -0.4597; median aspect at support 1 falls 1.7185 to 1.0645 against 1.0516; r_rb +0.2223, 95 pct CI +0.0295 to +0.4133, AUC 0.6112, Holm p 0.0320; log area r_rb -0.1467; size matched 58 v 58, r_rb +0.2420; verdict INCONCLUSIVE; power ~80 pct at r_rb 0.30 |
| **4.5 Over segmentation** | median 2.0 detections per annotated tree; 33 of 64 carry two or more; distribution 6/25/17/5/4/7, tail to 12; precision 8.0, recall 34.4, F1 0.1302, tree rate 90.6, detection containment 50.0; median matched IoU 0.664; 42 unmatched annotations |
| **4.6 Consensus null** | 47.0 / 52.7 / 46.2 percent by band at set 1; Fisher p 0.384 and 1.000; Spearman -0.083; set 5 gives 70.4 / 72.6 / 46.2, p 0.782 and 0.114; smallest Fisher anywhere 0.0753 at set 3; Spearman negative on all five |
| **6 Limitations** | annotation coverage 67.5 percent (110) and 47.9 percent (64); 81 of 274 detections outside all annotations, 34 of them support 16; 164 clusters at supports 5 to 15 with 10 pinned; one annotator, no field verification |

**Numbers with no home, and therefore cut:** the full 16 row support histogram (Fig 2 carries it), the per support pinning table (Fig 3 carries it), the elongation axis and spatial variance tasks from `analyse_mechanism.py`, and the synthetic control results. All stay in the repo.

## 5. Figure inventory

| # | file | section | role |
| --- | --- | --- | --- |
| 1 | `figure_same_count_different_trees.png` | Introduction | Qualitative hook. Same count, different trees, on real imagery. |
| 2 | `fig_support_histogram.png` | 4.2 | Load bearing. The U shape is claim 2. |
| 3 | `fig_pinning_by_support.png` | 4.3 | Load bearing. The plateau at 1 to 4 and the floor after is claim 3. |
| 4 | `figure_geometry_support.png` | 4.4 | The collapse from left panel to right panel is the competing explanation being eliminated. |
| T1 | metrics table | 4.5 | Scoring sets 1 and 5: precision, recall, F1, tree rate, containment. |

**Cut: `fig_detections_per_tree.png`.** Its content compresses to one sentence without loss, since the distribution numbers already appear in 4.5 and the histogram shape carries no argument the sentence does not. Goes to supplementary.

**If a fifth figure is needed later,** restore that one rather than adding anything new.

**Not a figure, deliberately.** The consensus null stays prose. Three bars at roughly equal height invite a reader to see a gap that is not there.

## 6. Gap list

Missing content, by section, with owner.

| gap | section | owner |
| --- | --- | --- |
| Related Work is three papers (Miller 2019, Zhang and Wang 2016, Tung 2022). Needs 20 to 30 citations from a 50 to 70 paper review. | 2 | Gabe, Elicit set to follow |
| DeepForest citation and version. Weinstein et al. papers, plus the released checkpoint identifier and package version actually used. | 3 | Gabe, five minutes |
| Site description: species composition, canopy closure, stem density, acquisition date. | 3 | Methods docx, verify against OFO mission page |
| Compute environment and runtime for the 16 phase sweep. | 3 | Gabe |
| Data and code availability statement. Repo URL, OFO mission link, CC BY 4.0 attribution. | end matter | Gabe |
| Author list and affiliations. | title block | Gabe and Elizabeth |
| Abstract. | | written last |
| arXiv category and endorsement. Elizabeth has one arXiv paper, in cs.AI, so cs.CV endorsement is not automatic. | logistics | Friday meeting |

**One inconsistency to resolve before Methods is drafted.** The README's phase 0 section describes two tiling regimes, 25 tiles at phase 0 against 16 elsewhere. The project doc describes three: 25 tiles at 1 position, 20 at 6 positions, 16 at 9 positions. The three regime version is correct, because a position with dx at zero and dy nonzero gets five origins on one axis and four on the other. The README also quotes 5.5 percent core excess where the project doc quotes 5.19 percent. Use the project doc's three regime table and the 5.19 percent figure. Flagging because Methods is the section most likely to be drafted straight from the README.

## 7. Order of writing

1. **Results.** Every number exists. Writing it first fixes what the rest of the paper has to introduce and defend.
2. **Method.** Consolidate from `Methods_tile_grid_phase.docx`, add the geometry test and the resolution inconsistency fix above.
3. **Limitations.** Material exists across three docs and `LIMITATIONS.md`; this is consolidation, not new writing.
4. **Future Work.** Three scoped paragraphs: second site, planform segmentation via box prompted masks, 3D crown form via CHM.
5. **Discussion.** Written once Results and Limitations fix what can be claimed.
6. **Related Work.** Written after the Elicit review lands, so it cites rather than gestures.
7. **Introduction.** Written second to last. Easier once the paper it introduces exists.
8. **Conclusion, then Abstract.** Both compressions of finished text.

Literature review runs in parallel from week one and only blocks step 6.

## Timeline

| week | work |
| --- | --- |
| Aug 25 | Skeleton locked. Results drafted. Elicit set generated. |
| Sep 1 | Method, Limitations, Future Work. Lit review running. |
| Sep 8 | Discussion. Related Work as citations land. Figures finalised. |
| Sep 15 | Full draft assembled. Every number traced to `RESULTS_NUMBERS.md`. |
| Sep 22 | Elizabeth reads. Revisions. |
| Sep 29 to Oct 1 | Format, references, arXiv upload. |
