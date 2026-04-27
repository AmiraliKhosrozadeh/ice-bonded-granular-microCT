# Postprocessing pipeline (Glass + Alumina specimens)

Same pipeline, same script names, same inputs, for every specimen
under `Glass\` and `Alumina\`. Replace `<spam>` with the specimen's
spam folder name and `<pre>` with the specimen prefix used in the
script names.

| specimen folder                | spam folder                  | prefix         | scans   | transitions  |
|--------------------------------|------------------------------|----------------|---------|--------------|
| Glass\Glass_75_1700_T5_HR      | Glass_75_spam                | `75`           | 1, 2, 3 | (1,2), (2,3) |
| Glass\Glass_100_1700_T5_HR     | Glass_T5_HR_spam             | `t5hr`         | 1, 2    | (1,2)        |
| Glass\Glass_100_1700_T7        | Glass_1700_spam              | `t7`           | 1..4    | (1,2)..(3,4) |
| Alumina\Alumina_100_1800_T5    | Alumina_100_1800_T5_spam     | `100_1800_T5`  | 1..3    | (1,2), (2,3) |
| Alumina\Alumina_175_1800_T5    | Alumina_175_1800_T5_spam     | `175_1800_T5`  | 1, 2    | (1,2)        |
| Alumina\Alumina_75_1000_T5     | Alumina_75_1000_T5_spam      | `75_1000_T5`   | 1, 2    | (1,2)        |
| Alumina\Alumina-75-1800-T7     | Alumina_75_1800_T7_spam      | `75_1800_T7`   | 1..3    | (1,2), (2,3) |

---

## Stage 0 — what must already exist before postprocessing

Per scan `N`, produced by the per-scan Dragonfly segmentation pipeline:

```
<spam>\data\
  ct_scan{N}.tif                    raw CT (uint16)
  specimen_mask_scan{N}.tif         specimen mask (uint8 0/1)
  bead_labels_scan{N}.tif           watershed bead labels (uint16)
```

Plus, after running stage 1 below, one further file per scan:

```
<spam>\data\
  ice_mask_scan{N}.tif              binary ice phase from Dragonfly's ROI_Ice
```

Also produced earlier in the pipeline (kept for reference):

```
<spam>\data\
  *_aligned.tif                     each of the above padded into the
                                    common world-fixed voxel frame
  aligned_meta.py                   common frame origin / shape / offsets
```

---

## Stage 1 — Dragonfly side (one paste per scan)

Open each scan's session in Dragonfly (with the `ROI_Ice` ROI already
created by `scan{N}_segmentation.py`), then in the Python console:

```python
exec(open(r"E:\RPTU-images\CT_images\<specimen_dir>\scan{N}_dragonfly\dragonfly_export_ice_roi.py",
          encoding='utf-8').read())
```

Writes `<spam>\data\ice_mask_scan{N}.tif`. Repeat for every scan
in the specimen.

If you want to run PuMA-style metrics (continuum tortuosity, thermal
conductivity, orientation, surface area) inside Dragonfly:

```python
exec(open(r"E:\RPTU-images\CT_images\<specimen_dir>\<spam>\dragonfly_run_puma.py",
          encoding='utf-8').read())
```

---

## Stage 2 — WSL postprocessing scripts

Run from WSL (`source ~/spam-venv/bin/activate`). These use the
aligned data already exported by Stage 0.

Each script needs the things in its **Inputs** column already present.
Outputs all land under `<spam>\results_<prefix>\` unless noted.

| order | script                                      | inputs                                                   | what it produces                                                                                |
|-------|---------------------------------------------|----------------------------------------------------------|-------------------------------------------------------------------------------------------------|
|   1   | `spam_<pre>_permeability.py`                | `tortuosity_evolution\summary.csv`, `crack_analysis\crack_summary.txt` | Kozeny-Carman air-permeability per direction per scan + crack-vs-Δk plot                 |
|   2   | `spam_<pre>_crack_sparse_graph.py`          | `crack_analysis\transition_AtoB\damage_3d_*.tif`         | per-transition ball-and-bar of CRACK / ICE_FRACTURE / CAVITY clusters                            |
|   3   | `spam_<pre>_bond_sparse_graph.py`           | `bead_labels_scan{N}_aligned.tif`, `ice_bonds_scan{N}.csv` | per-scan ball-and-bar bead-bond network (initial → final story)                               |
|   4   | `spam_<pre>_crack_validate_with_ct.py`      | `ct_scan{B}_aligned.tif`, `damage_3d_*.tif`              | CT-slice panels with damage overlay, two slices per transition, side-by-side validation        |
|   5   | `spam_<pre>_ice_sparse_graph.py`            | `ice_mask_scan{N}.tif` (preferred) or `ct_scan{N}_aligned.tif` + thresholds | per-scan skeleton graph: `nodes_scan{N}.csv`, `edges_scan{N}.csv` |
|   6   | `spam_<pre>_sparse_graph_compare.py`        | output of step 5                                         | per-transition `edge_change_AtoB.csv`, `crack_candidates_AtoB.csv`, 3D render of broken edges    |
|   7   | `spam_<pre>_crack_paths.py`                 | `crack_candidates_AtoB.csv`                              | DBSCAN-clustered broken edges → ordered crack-path polylines + 3D figure                          |
|   8   | `spam_<pre>_crack_voids.py`                 | `nodes_scan{N}.csv`                                      | density-based vacancy/crack detection (faded-skeleton background + coloured void clusters)       |

Steps 5–8 build on each other. Steps 1–4 are independent.

---

## Stage 3 — what each new postprocessing answers

| step | question it answers |
|------|----------------------|
| 1    | How does the gas / liquid permeability of the specimen change with compression? Is the change driven by porosity alone or by crack-induced flow paths? |
| 2    | Where are the damage clusters in the specimen, and what type of damage (planar crack vs ice fracture vs compaction void)? |
| 3    | How does the bead-to-bead bond network thin out / rearrange between scans? Strong vs thin bonds. |
| 4    | Are the sparse-graph clusters the same features that show up as visible damage in the real CT slice? |
| 5    | Topology of the ice phase: junctions and skeleton segments. Quantitative basis for the rest. |
| 6    | Which skeleton edges survived between scans, which weakened, which broke? Each broken edge is a crack candidate. |
| 7    | The actual crack paths through the specimen (oriented, coloured by cluster). |
| 8    | Spatial distribution of crack regions: places where the local skeleton density is anomalously low — the cleanest visual signature for a paper. |

---

## Per-specimen tunables (open each script, edit at the top)

```python
SCANS        = [1, 2, 3]            # match the specimen's scan list
TRANSITIONS  = [(1, 2), (2, 3)]     # consecutive pairs from SCANS
NX_E, NY_E, NZ_E = 803, 707, 1241   # common-frame shape from <spam>\data\aligned_meta.py
TH = {
    1: dict(air=<T_AIR_ICE>, ig=<T_ICE_GLASS>, ga=<T_GLASS_AL>),  # from scan{N}_histogram_analysis.py
    ...
}
```

For glass / alumina specimens with no aluminum peak, set
`T_GLASS_AL = 99999` (otherwise the Al-first subtraction will eat
into glass / alumina).

---

## Ready-to-paste full WSL pipeline (replace `<>` placeholders)

```bash
source ~/spam-venv/bin/activate
SPAM=/mnt/e/RPTU-images/CT_images/<specimen_dir>/<spam>

python $SPAM/spam_<pre>_permeability.py
python $SPAM/spam_<pre>_crack_sparse_graph.py
python $SPAM/spam_<pre>_bond_sparse_graph.py
python $SPAM/spam_<pre>_crack_validate_with_ct.py
python $SPAM/spam_<pre>_ice_sparse_graph.py
python $SPAM/spam_<pre>_sparse_graph_compare.py
python $SPAM/spam_<pre>_crack_paths.py
python $SPAM/spam_<pre>_crack_voids.py
```

Run these **sequentially** (parallel runs OOM-kill the WSL VM,
each step loads ~3 GB of aligned TIFFs).

---

## Outputs summary (per specimen, after the full pipeline)

```
<spam>\results_<pre>\
  permeability\                               (Kozeny-Carman + Δk plots)
  crack_analysis\transition_AtoB\
       crack_sparse_graph_3D.png
       crack_validation_CT_slices.png
       failure_modes_*.png / .csv
  bond_sparse_graph\bond_graph_scan{N}.png
  sparse_graph\
       nodes_scan{N}.csv, edges_scan{N}.csv
       edge_change_AtoB.csv, crack_candidates_AtoB.csv
       sparse_graph_change_3D_AtoB.png
       crack_paths\crack_paths_AtoB.csv,
                   crack_paths_summary_AtoB.txt,
                   crack_paths_3D_AtoB.png
       voids\voids_scan{N}.csv,
             voids_3D_scan{N}.png
```
