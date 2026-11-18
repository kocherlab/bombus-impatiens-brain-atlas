# Region annotations

Hand-drawn anatomical regions for every Xenium section, one GeoJSON file per
capture region, exported from QuPath (pixel coordinates; 0.2125 µm per pixel).
Regions were hand-drawn by S.W. Wolf, E. Bwonya and S.Z.G. Liu.

Layout: `data/annotations/<brain>/<set>/<capture region>.geojson`, where the
file name is the Xenium output name of the capture region it covers,
`output-XETG00151__<slide>__Region_<n>__<date>__<time>.geojson` (the same name
the raw data and the per-crop h5ad files carry). `queen/` and `worker/` hold one
directory per annotation set:

| set | contents | read by |
|---|---|---|
| `v1/` | the original six regions (lamina, medulla, lobula and the three Kenyon-cell calyx compartments) | `pipelines/wholebrain` (`label`), and as the fallback for sections absent from the newest set |
| `v2/` | a later export adding the antennal lobe and the central complex; kept as the record it was built from | `scripts/regions/build_annotations_v3.py` |
| `v3/` | `v1/` plus the three added regions | `scripts/regions/build_annotations_v4.py` |
| `v4/` | `v3/` deduplicated, with standardised polygon names: the set the region flags use | `bimp_st.nine_region_common.annotation_sources` (newest set present) |

Which file covers which crop of a capture region is `ANNOTATION_MAPPING` in
`pipelines/wholebrain/<brain>/labeling.py`, keyed by these file names.

## Earlier file names

The queen files were exported under names with an export-time prefix
(`YYYYMMDD_HHMM_`) or an annotator tag; they were renamed to the scheme above
with their content unchanged. The worker files already used it.

| earlier name (queen) | name now (`output-XETG00151__` + ...) | sets |
|---|---|---|
| `20251216_1246_output-XETG00151__0040240__Region_1__20251204__195015.geojson` | `…0040240__Region_1__20251204__195015.geojson` | v1, v2, v3, v4 |
| `20251216_1246_output-XETG00151__0040240__Region_2__20251204__195015.geojson` | `…0040240__Region_2__20251204__195015.geojson` | v1, v2, v3, v4 |
| `20251216_1246_output-XETG00151__0040240__Region_3__20251204__195015.geojson` | `…0040240__Region_3__20251204__195015.geojson` | v1, v2, v3, v4 |
| `20251216_1246_output-XETG00151__0040240__Region_4__20251204__195015.geojson` | `…0040240__Region_4__20251204__195015.geojson` | v1, v2, v3, v4 |
| `20251216_1246_output-XETG00151__0040240__Region_5__20251204__195015.geojson` | `…0040240__Region_5__20251204__195015.geojson` | v1, v2, v3, v4 |
| `20251216_1246_output-XETG00151__0040240__Region_6__20251204__195015.geojson` | `…0040240__Region_6__20251204__195015.geojson` | v1, v2, v3, v4 |
| `20251216_1246_output-XETG00151__0040240__Region_7__20251204__195015.geojson` | `…0040240__Region_7__20251204__195015.geojson` | v2 |
| `output-XETG00151__0040240__Region_7__20251204__195015(Bwonya).geojson` | `…0040240__Region_7__20251204__195015.geojson` | v1, v3, v4 |
| `20251216_1246_output-XETG00151__0040240__Region_8__20251204__195015.geojson` | `…0040240__Region_8__20251204__195015.geojson` | v1, v2, v3, v4 |
| `20251216_1324_output-XETG00151__0040243__Region_1__20251204__195015.geojson` | `…0040243__Region_1__20251204__195015.geojson` | v1, v2, v3, v4 |
| `20251216_1324_output-XETG00151__0040243__Region_2__20251204__195015.geojson` | `…0040243__Region_2__20251204__195015.geojson` | v2 |
| `output-XETG00151__0040243__Region_2__20251204__195015(Bwonya).geojson` | `…0040243__Region_2__20251204__195015.geojson` | v1, v3, v4 |
| `20251216_1324_output-XETG00151__0040243__Region_3__20251204__195015.geojson` | `…0040243__Region_3__20251204__195015.geojson` | v2 |
| `output-XETG00151__0040243__Region_3__20251204__195015(Bwonya).geojson` | `…0040243__Region_3__20251204__195015.geojson` | v1, v3, v4 |
| `20251216_1324_output-XETG00151__0040243__Region_4__20251204__195015.geojson` | `…0040243__Region_4__20251204__195015.geojson` | v1, v2, v3, v4 |
| `20251216_1324_output-XETG00151__0040243__Region_5__20251204__195015.geojson` | `…0040243__Region_5__20251204__195015.geojson` | v1, v2, v3, v4 |
| `20251216_1324_output-XETG00151__0040243__Region_6__20251204__195015.geojson` | `…0040243__Region_6__20251204__195015.geojson` | v1, v2, v3, v4 |
| `20251216_1324_output-XETG00151__0040243__Region_7__20251204__195015.geojson` | `…0040243__Region_7__20251204__195015.geojson` | v1, v2, v3, v4 |
| `20251216_1324_output-XETG00151__0040243__Region_8__20251204__195015.geojson` | `…0040243__Region_8__20251204__195015.geojson` | v1, v2, v3, v4 |
