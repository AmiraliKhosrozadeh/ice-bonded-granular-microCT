# Review loop, 2026-09-18

Ten expert lenses per round, findings checked against the manuscript, the
data files and the scanner records, corrections applied, rebuilt, next round.
"Checked, no change" means the point was examined and the manuscript was
found to be right or already covered.

## Round 1

Experts: ice mechanics / frozen ground, granular mechanics and DEM/BPM,
micro-CT physics, DVC methods, statistics, fracture mechanics, porous-media
transport, journal editor (Powder Technology), technical editor / consistency,
reproducibility and software.

### Changed
1. **Micro-CT physics — acquisition parameters were partly wrong.** Read the
   scanner's PRM record of 23 of the 30 scans: projections are 900–1100 per
   scan (1000 for most, 1100 for G3 and A4, 900 for S2 and A2's first scan),
   not "1000 for beads, 900 for sand"; measurement time 22–31 min, not 30/31.
   The stated magnification (3.97) and voxel (24.77 µm) did not agree
   (100 µm / 3.97 = 25.2 µm); the 24.77 µm is the instrument's
   scale-calibrated voxel (ORIVOXELSIZE), now said so. §2.3 rewritten.
2. **Notation clash.** $z$ was both the axial coordinate and the coordination
   number. Coordination number is now $Z$ (nomenclature, §2.5.3, Table 3
   header $\bar Z$). Nomenclature also said $\mathcal N(i)$ has 16 neighbours;
   it is 12 for beads and 16 for sand.
3. **Statistics.** Table 4's Benjamini–Hochberg value for G3 was "< 0.02";
   with 400 resamplings the smallest attainable p is 1/401 = 0.0025, times
   nine specimens = 0.022, so it is "< 0.03". Corrected.
4. **Fracture mechanics.** "Mixed-mode" was used for a mixed interfacial–
   cohesive *path*; in fracture mechanics mixed mode means the loading mode
   (I/II/III). Reworded in §3.3, the abstract and the highlight.
5. **Editor.** `\journal{}` was empty → Powder Technology. Cross-reference
   style unified: "Section~\ref" → "§", mid-sentence "Figure~\ref" → "Fig."
   (sentence-initial "Figure" kept, 3 places). §2.5.1 referred the reader to
   "§2.5.1" for the sand tracking (itself) → "described below".
6. **Roughness claim (discussion).** "The two abraded specimens bracket the
   as-received one" was not what Table 4 shows: G1 (HR) is the most damaged
   glass specimen and G4 (HR) the least, so the two abraded specimens sit at
   the two ends of the glass spread. Rewritten to that, which is the honest
   basis for "no roughness effect separable from scatter".
7. **Ice physics.** Added to Limitations that at −5 °C a premelted film
   persists at the ice–grain interface (Dash, Rempel & Wettlaufer 2006 added
   to ref.bib), so the adhesive share is that of a wetted interface and may
   differ at lower temperature.
8. **Layout.** Fig. 13 (bond sections + crack-at-face) had overflowed the page
   by 16 pt since an earlier session; second image reduced to 0.58 linewidth.
   Build now has no float warnings.

### Checked, no change
- Nominal strain-rate range 4e-4 to 3e-2 s^-1 is consistent with 0.010 and
  0.60 mm/s on 20–27 mm columns.
- Kozeny–Carman in Carman form with k_K = 5 for surface per unit bulk volume:
  correct; the ratio reported cancels the constant.
- Voxel-face surface overestimate factor up to sqrt(3): correct.
- p < 0.003 for G3's cluster (1/401 = 0.0025): correct.
- Abstract's "about half lost at least half its ice" agrees with the 48 %
  adhesive share at C_HI = 0.50.
- Highlights are all within 85 characters.
- False-positive null (2963 intact throats) is stated with its rate.

### Open for the author (not fixable from the data here)
- Number of companion compression tests behind the mean ± SD strengths in
  §3.2 is not stated.
- Rates for G2, G4, G5, A2, A3, A4 are still "--" in Table 1.

## Round 2

Experts: permafrost / frozen-ground geotechnics, machine-learning
segmentation, DEM/BPM modeller, metrology and uncertainty, tomography
artefacts, materials scientist (gamma-alumina), second statistician,
DVC rigid-body handling, figure/caption editor, language editor.

### Changed
1. **Fig. 5 (force records) contradicted §2.2.** §2.2 says the in-situ
   traces precede the load-cell recalibration and are not used
   quantitatively, yet §3.2 read as if the strengths came from the figure,
   and the caption called the shaded bands "load levels at which scans were
   acquired". Caption and text now say: one in-situ record per material,
   indicative force axis, bands are the loading segments before the first
   and second loaded scans (load steps 2 and 3 in the paper's numbering);
   strengths are from the companion tests without scanning.
2. **Gamma-alumina internal porosity** was never mentioned. Catalyst-grade
   gamma-alumina is mesoporous; imbibed water freezes inside the grain and is
   classed as grain. Added to Limitations, phrased conditionally — the author
   should confirm the beads' pore volume (open item).
3. **Thermal history during a scan** was not stated (22–31 min per scan under
   sustained load at −5 °C, creep during the hold not separated from the
   increment). Added to §2.2.
4. **U-Net training details** (synthetic data, configuration, weights) are
   not in the text; pointed to the repository, where they are.
5. **Package inventory** in §2 said only SPAM and PuMA are published
   packages; Dragonfly, scikit-learn and scikit-image are used too. Corrected.

### Checked, no change
- Bond neck radius 0.72–0.83 R and "matrix bond, not pendular neck" reading:
  consistent with S = 0.71–0.91 in Table 1.
- Rigid-body removal (Kabsch on the support quarter, 0.73° tilt = 208 µm
  gradient) is fully specified.
- Shortening = 1 − h/h0 with h from the reconstruction: metrology stated.
- Beam hardening is named as the cause of the sand's axial drift; no filter at
  150 kV is stated. Ring artefacts are not an issue in the phase maps shown.

## Round 3

Experts: data-consistency auditor (every number in the text against the
tables and the data files), tortuosity/transport referee, sand geotechnics,
bond-census statistician, crack-class analyst, figure editor, and four
re-reads by the Round 1 experts on the changed sections.

### Changed
1. **Body share of G3's crack** was "a fifth to a third to a half"; the
   volumes (1.4/7.4, 23/72, 49/142 mm3) give 0.19, 0.32, 0.35. Now "a fifth
   to a third and holding there".
2. **Majority-cohesive specimens at C_HI = 0.50** were listed as three (G1,
   A1, A4); Table 2 has A2 at 27/23 too, so four, with G2 and G4 split evenly.
3. **Coordination fall** "between a half and two bonds per grain": Table 3
   gives 0.7 (A4) to 2.1 (G3). Corrected to 0.7–2.1.
4. **Tortuosity "less than 0.2 in the alumina and the sand"**: S3 rises by
   0.56 (2.67 → 3.24, Table S4). S3 is omitted from Fig. 17 without a stated
   reason; the reason (its second scan was classified with a specimen mask
   0.8 mm wider than its first, measured today: 9.7 vs 10.5 mm equivalent
   diameter) is now given in §3.6 and the Fig. 17 caption, and the claim is
   restricted to S1 and S2.
5. **S1's second scan is at zero shortening** (column length 18.15 mm in both
   scans), which the text never said while using S1 as a "higher-rate"
   specimen. Stated in §3.5; the higher-rate observation now rests on S3.

### Checked, no change
- Table 1: S = phi/(1 − rho) and eps = 1 − phi − rho hold for every row to
  rounding.
- 85 % / 77 % without A3 / 56 % A1 / 100 % G3 / 48–75–85–93 %: all reproduce
  from Table 2.
- G1 body crack overtaking surface crack between scans 2 and 3: 45 < 61 then
  56 > 49 mm3, correct.
- Delta crack in Table 4 for G1, A2, A4 and the A4 floor (+43/+19): reproduce
  from crack_volumes_tight.csv.
- Glass tortuosity rise 0.2–0.4 (G2–G5) and 2 (G1), unloaded ranges
  1.5–1.6 / 1.8–2.0 / 2.2–2.7: reproduce from Table S4.
- Median local rotation 9–14°: reproduces from Table 4.
- S2 "shortens by a fifth through its second increment": 0.048 → 0.248.

## Round 4

Experts: abstract-versus-body auditor, literature reviewer (frozen-soil CT),
shape-gate bias referee, rate-effects referee, terminology/units editor,
nomenclature auditor, equation checker, table-arithmetic checker,
cryo-experimentalist, conclusions referee.

### Changed
1. **Rate confound for the beads.** Table 1 now carries platen rates and they
   differ by a factor of seven between glass specimens; G1 (most damaged) was
   the fastest, G3 the slowest. Limitations now says the rate is confounded
   with the specimen for the beads as for the sand, and that no material
   contrast in the paper is a rate contrast.
2. **Temperature notation** unified to `\SI{-5}{\celsius}` (eight places
   used `$-5\,^\circ$C`).
3. **Nomenclature** pruned of five entries that appear nowhere in the text
   (u_r/r, lambda_2/lambda_3, BPM, PFS, REV).

### Checked, no change
- Every abstract claim maps to a results paragraph (85 % of 431; half at
  C_HI = 0.50; one clean interfacial break; separation vs rupture; surface
  first, no plane; sand compaction then flow; one specimen per condition).
- Table 2 column sums (8680 followed, 431 answerable) are correct.
- Shape-gate bias is acknowledged (alpha_b a lower bound; failure-mode census
  a statement about the widest tenth of the bridges).
- Conclusions' "twice as fast" and "orders of magnitude" restate §3.6.

### Open for the author
- Introduction says frozen-soil CT is "overwhelmingly of specimens at rest";
  recent in-situ triaxial frozen-sand CT work exists and a referee may cite
  it — worth a literature check before submission.

## Round 5

Experts: figure/caption referee, supplement-versus-paper auditor, methods
structure referee ("Results report, Methods explain"), flowchart reviewer,
cross-reference/numbering checker, equation-number checker, table-symbol
checker, journal production editor, second data auditor, reproducibility.

### Changed
1. **Ice-path detour map was defined inside Results** (§3.6) and absent from
   the analysis flowchart (Fig. 1). Its definition (envelope, seeds, healing
   rule, 0.3 mm threshold, sand margin, what it is not) now sits in §2.5.4
   after the tortuosity; §3.6 only reports; Fig. 1 branch C lists "ice-path
   detour map" and the outputs box names it. Control-box anchoring in the
   flowchart moved to the tallest branch so nothing overlaps.
2. **Supplement Table S1** still used $\bar z$ for the coordination number;
   now $\bar Z$ as in the paper.

### Checked, no change
- Hard-coded figure numbers in the supplement (Fig. 11 crack classes, 12
  orientation, 15 S3 fields, 18 map) and "Eq. (7)", "Table 2 of the paper"
  match the current numbering of the paper.
- Supplement §S4 shares (G1 10.5, G3 6, G2 3.3, G5 1.2, G4 0.1, A4 1.2/3.3,
  A2 1.0, A3 0.8 %) match Table S3.
- Fig. 11 has four panels for G3's four scans; Fig. 12 nine panels; Fig. 10
  panels (a)–(g) as captioned.

## Round 6

Experts: hostile methods-reproducibility referee, envelope/mask consistency
checker, DBSCAN/statistics referee, ice-film metrologist, sand-segmentation
referee, scan-count auditor, language editor, table-footnote checker,
conclusions-versus-discussion referee, production editor.

### Changed
1. **Envelope definitions conflated.** §2.5.4 said the detour map runs
   "inside the grain envelope of §2.5.2", but the crack classes use the bead
   *labels* with a four-voxel skin and the map uses the grain *phase* with a
   two-voxel skin. Now stated precisely.
2. **Conclusions contradicted the Discussion**: "more magnification rather
   than more specimens" versus "a deliberate ice-content series with
   replication is the experiment that would settle it". Conclusions now
   separate the two: replication for ice content and roughness,
   magnification for the failure mode and the sand.

### Checked, no change
- Crack-class parameters in §2.5.2, Table S1 and crack_split_tight.py agree
  (closing 1.25 R, 4-voxel skin, 2 mm3 void ceiling, 0.125 mm half-width,
  20-voxel surface skin, 100-voxel specks, one bead diameter platen
  clearance).
- Scan count: 13 glass + 10 alumina + 7 sand = 30, as stated; "seven sand
  scans" for the U-Net inference is right.
- 114 µm = 4.6 voxels at 24.77 µm.
- DBSCAN minimum samples and cluster floor are in Table S1; the text gives
  the pre-specified radius and the sweep.
- "namely" appears five times, "which is …" chains 21 times: a stylistic
  trait, left as the authors' voice.

## Round 7

Experts: transport referee (Kozeny–Carman numbers), kinematics referee,
bibliography auditor, LaTeX build auditor, discussion-versus-results
consistency, conclusions checker, and four re-reads of §3.6 and §4.

### Changed
1. **Permeability "two to three orders of magnitude" in the glass** was true
   for G1 (1800×), G3 (200×) and G5 (180×) but not G2 and G4 (18× and 28×,
   section36_core.csv). §3.6, §4.1 and the Conclusions now say "one to three
   orders", with the per-specimen factors given in §3.6, and the alumina's
   0.5–1.6× is stated.

### Checked, no change
- Sand permeability "less than one order" (S2 5.3×); alumina "not at all".
- Local rotation 95th percentile 17–37° (16.6–36.5), A4 floor 0.4°, E_eq
  medians 0.27–0.40: reproduce from kinematics_summary.csv.
- All 31 cited keys exist in ref.bib; BibTeX reports no warnings; no
  undefined references or citations in the build; no oversized floats.
- Glass void-fraction gain per unit shortening about twice the alumina's:
  consistent with the per-specimen slopes (glass 0.23–0.77, alumina
  0.11–0.57, fits through the origin).

## Round 8

Experts: copy editor (spelling, British/American consistency), abstract
editor against the journal limit, highlights checker, keywords checker,
front-matter/affiliation checker, CRediT checker, AI-declaration checker,
funding/acknowledgement checker, appendix checker, and a reader new to the
paper checking the abstract stands alone.

### Changed
1. **Abstract was 282 words**; Powder Technology asks for at most 250. Cut
   to 250 without dropping a claim (tightened the opening two sentences and
   the list of analyses).
2. **"grayscale"** appeared once against seven "greyscale"; unified.

### Checked, no change
- Crude spell-check (pyspellchecker over the de-LaTeXed text): the only
  unknown words are proper nouns, LaTeX and British spellings; no typo.
- Highlights: four items, 68–83 characters each (limit five items, 85
  characters).
- Keywords: seven, within the journal's range.
- Affiliations, corresponding author, CRediT, AI declaration, competing
  interests, funding (DFG HE-4526/40-1), appendix (calibration only): present
  and consistent with the title page.

## Round 9

Experts: typesetting/production referee, page-layout reviewer, font/PDF
compliance checker, literature-claim verifier, second copy editor, and five
re-reads of the sections changed in rounds 1–8 for coherence.

### Changed
1. **Typesetting.** Six marginal overfull lines (2–8 pt) removed with
   microtype character protrusion (expansion left off: the cm-super Type 1
   fonts in this MiKTeX do not support it and the build failed with it on).
   Fonts embedded are all Type 1, as production requires.
2. **Unverifiable literature number.** "imaged at five times this
   resolution" for Ní Bhreasail et al. was not checkable here; now "at a
   finer voxel than this".

### Checked, no change
- Every page rendered and scanned: no orphaned captions, no figure over a
  margin, no overlap; the half-empty pages are the preprint float placement
  and disappear in the journal layout.
- Supplement rebuilt with the $\bar Z$ symbol; no errors or undefined
  references.

## Round 10

Experts: final read-through as the handling editor, as Referee 1 (ice/frozen
ground), Referee 2 (imaging and DVC), Referee 3 (granular mechanics), plus
the data auditor, the statistician, the copy editor, the production editor,
the supplement auditor and the repository auditor, each re-reading the
passages changed in rounds 1–9 as typeset.

### Changed
1. Range typesetting in §2.3 ("22 min to 31 min" → "22 to 31 min").

### Checked, no change
- The rewritten §2.3, §2.5.4 (detour map), §3.2 and §4.2 read coherently
  as typeset; cross-references resolve (§2.2, §2.5.2, §4.2, Eq. (11)).
- Build: 61 pages, no errors, no undefined references or citations, no
  oversized floats, five overfull lines all under 8 pt.
- Supplement: 17 pages, no errors; symbols and figure numbers consistent
  with the paper.
- Repository copies of the supplement and settings table updated.

## Still open for the author (cannot be settled from the files here)
1. Number of companion compression tests behind each mean ± SD strength
   (§3.2).
2. Platen rates for G2, G4, G5, A2, A3, A4 (Table 1 shows "--").
3. Whether the gamma-alumina beads are porous (pore volume from the supplier
   data sheet); the Limitations sentence is conditional until then.
4. A literature check for recent in-situ CT of frozen soils under load
   before the Introduction's "overwhelmingly at rest" stands.
5. Fig. 9 was made outside the repository; its band labels use "scan 1/2"
   for the loaded scans, which the caption now translates, but relabelling
   the figure itself would be cleaner.

## External pre-submission review (powder_technology_review_2026-09-18.md), validity check

Each finding was checked against the manuscript, the data files and the
analysis scripts. "Valid, fixed" means the manuscript was changed;
"valid, author" means it needs information not in the files; "partly" and
"not valid" are explained.

| # | Finding | Verdict | Action |
|---|---|---|---|
| P2 | Alumina CV is 0.44/2.87 = 15 %, not 36 % | valid, fixed | §3.2 rewritten: glass 48 %, alumina and sand 15 %; only the glass is less repeatable, and the bridge-count explanation is now qualified |
| P1 | Calibration chronology conflicts (pre-test deadweights in §2.2, post-test recalibration in §2.2, pre-test texture analyser in Appendix A) | valid, author | No calibration record in the repository; the author must state which calibration applied to which records |
| P3 | Adhesive majority depends on threshold: 48.5 % at C_HI = 0.50 is not a majority, so "sets the size of the majority and not its direction" was false | valid, fixed | §3.3 now says the share is 48/75/85/93 %, a majority at every setting but the loosest, where the classes are even |
| P4 | Separation is the observed final state, not a demonstrated sequence | valid, fixed | Abstract, highlight, §3.3 opening and Conclusions now say broken bonds are *found* separated at the last scan |
| P5 | Permeability is a Kozeny–Carman estimate | partly (already stated in §3.6) | "Kozeny–Carman permeability estimate" now also in §4.1 and Conclusions; "cancelled" replaced by "offset … within a factor of two" |
| P6 | Shape causality confounded with grain material | valid, fixed | §4.1 and Conclusions: one bead material per shape, shape and grain strength confounded, contrast exploratory |
| 1 | Literature claim "overwhelmingly at rest" | valid, softened to "mostly" | author still to check recent in-situ frozen-soil CT |
| 1 | Morphology similarity does not prove independence from the failure process | valid, fixed | now "the same mixed morphology has been seen under both freezing and loading" |
| 2 | Temperature: controller −15 °C vs chamber −5 °C, sensor location, uniformity | valid, author | sensor position and uncertainty are not in the files |
| 2 | Preparation details depend on an unpublished companion manuscript | valid, author | packing, water dosing, freezing history must come from the author |
| 2 | Hold/relaxation during 22–31 min scans | partly (hold stated in §2.2 since round 2) | no relaxation record exists; the A4 near-zero increment (0.04 mm motion) is the only motion check and is already cited |
| 3 | Sand model selected and scored on the same scans; PSD agreement does not validate boundaries or identity | valid, fixed | Limitations now say so explicitly and that no held-out deformed subvolume exists |
| 3 | Bead thresholds: valleys (text) vs Otsu (caption) | valid, fixed | stage0 script `scan_histogram_analysis.py` takes the midpoints between the three histogram peaks; text and Fig. 3 caption now say midpoints / "the two thresholds" |
| 3 | Sand phase assignment: Otsu three-class vs "quartz and ice share one mode" | valid, fixed | §2.4.1 now says the quartz fraction of Table 1 is taken from the labels, the histogram separating air from solid only |
| 3 | Median matching criteria cannot establish every correspondence | valid, author | per-pair tables exist in the repository (paper/data/bonds); reporting per-match distributions is an addition, not a correction |
| 4 | Number of companion tests, meaning of ± | valid, author | open since round 1 |
| 4 | Strain at peak does not prove initial stiffness | valid, fixed | §3.2: "reaches its peak at the smallest shortening"; modulus stated as not evaluated |
| 4 | "Twice as fast": define the fit | partly | the caption already says one line per material through the origin; weighting is equal per scan (fig_microevo.py) — left as is |
| 5 | "Interface-proximal"/upper-bound wording; derive the 4.6-voxel limit | partly | the upper-bound statement exists in §3.3; the origin of 4.6 voxels is not in the files — author |
| 5 | Broken fraction lower bound not proven | valid, fixed | §2.5.3 now gives retention (21–74 % of first-scan bonds followed, Tables 3 and 4) and the all-survive/all-fail bounds (9–87 % in G1), and calls the lower-estimate reading an assumption |
| 5 | Lens footprint 0.40 R vs neck radius 0.72–0.83 R | valid clarification, fixed | the neck is the connected ice patch over the whole plane, not within the lens (Table 3 caption); §2.5.3 now says so |
| 6 | Conduction conclusions need qualification | fixed with P5 | |
| 6 | Voxel-face bias does not necessarily cancel | valid, fixed | §2.5.4: "largely cancels … which is an assumption" |
| 6 | Acceptance criteria do not demonstrate convergence | valid, fixed | §2.5.4 now says the criteria are necessary not sufficient and residuals were not archived (checked: the ice_tortuosity JSON records D_eff and tau only) |
| 6 | Tortuosity domain is cubes, not the core | valid, fixed | §3.6 opening corrected |
| 6 | KC "cancel" requires S/S0 = (eps/eps0)^1.5 | valid, fixed | wording "offset … within a factor of two" (A1: 1.62^3/1.63^2 = 1.6, matches k/k0) |
| 7 | Non-significant clustering does not prove absence of a front | valid, fixed | §3.3, abstract and highlight now "no planar organisation resolved"; synthetic-plane power test not done — author |
| 7 | Null population should be the eligible (followed) bonds | valid clarification, fixed | planar_test.py already resamples from the followed bonds (`s[s.connected1]`); the text wrongly said "all bonds present"; corrected |
| 7 | Reciprocity ≠ correctness; known-motion tests | partly (already acknowledged in §2.5.1) | known-motion test not done — author |
| 7 | Stress-relief mechanism is a hypothesis | valid, fixed | "the kinematics suggest why"; Conclusions "consistent with" |
| 8 | ChatGPT-assisted graphical images must be identified | valid, author | which figures were designed with ChatGPT is not in the files; Elsevier policy needs the tool named in the captions concerned |
| 8 | Pin the repository to a release | valid, fixed | tag v1.0 pushed to GitHub and cited in Data availability |
| 9 | Abstract colon and parallel list | partly | house style avoids colons; "namely" added instead |
| 10 | Tube 15 mm ID vs bore radius 6.40 mm (12.8 mm) | valid, author | §2.2 lists a 15 mm PMMA cylinder and a 13 mm aluminium insert called the piston; the reconstructions show a 12.8 mm bore; the apparatus description must say which part the specimen sits in |
| 10 | Sand grain 6.5 voxels vs 218 µm / 24.77 µm = 8.8 | valid, fixed | now "8–9 voxels (D50 194–225 µm)" |
| 10 | Scan counts sum to 30; bond-table sums | correct, no change | |
