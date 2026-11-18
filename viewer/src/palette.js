/* Palette — anatomy → hue family. Hue encodes real brain anatomy, not decoration.
   Kenyon cells = golds · optic lobe = cyans/blues · glia = violets ·
   neurons = greens · immune = coral. Caste: queen gold / worker teal. */
window.BIMP = window.BIMP || {};
BIMP.palette = (function () {
  const HEX = {
    0:"#6FD36A", 1:"#B06CE0", 2:"#B8641C", 3:"#8E5AD8", 4:"#FF7A6B", 5:"#F5C542",
    6:"#C77D1A", 7:"#4FBF7A", 8:"#FAD87A", 9:"#22D3EE", 10:"#2BB3E0", 11:"#3B95D6",
    12:"#4F7BE0", 13:"#37C6B0", 14:"#1FB2A6", 15:"#5AA9E6", 16:"#2FA8E8", 17:"#46C7D8",
    18:"#6D8FE8", 19:"#2CD1B4", 20:"#3FA0DC", 21:"#58D6C4", 22:"#9C7A3A", 23:"#9A79C4",
    24:"#98D98A", 25:"#D06AC8", 26:"#7BA9FF", 27:"#E8A22E", 28:"#6E4FD0",
    29:"#3E4552",   // "Mixed" — clusters below the subtype purity floor (dim slate)
  };
  // Grouping into anatomical families (ids are tangram_cell_type category order).
  const FAMILIES = [
    { name: "Mushroom body · Kenyon cells", ids: [5, 27, 6, 8, 2, 22] },
    { name: "Optic lobe",                   ids: [16, 17, 12, 11, 9, 10, 15, 20, 19, 13, 21, 14, 18, 26] },
    { name: "Glia",                         ids: [1, 3, 23, 25, 28] },
    { name: "Projection & other neurons",   ids: [7, 0, 24] },
    { name: "Immune",                       ids: [4] },
    { name: "Unresolved",                   ids: [29] },   // Mixed (subtype purity floor)
  ];
  function hexToRgb(h) {
    return [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
  }
  // The three Kenyon-cell calyx compartments, colored exactly as in the
  // exploded-regions SI figure (scripts/regions/si_exploded_regions.py REGION_PALETTE), so the
  // viewer and the figures agree. Applied in cell-type view to cells inside each mask.
  const KC_REGION = {
    in_kc_ic: "#B07AA1", in_kc_nc: "#F28E2B", in_kc_oc: "#EDC948",
    in_lamina: "#59A14F", in_medulla: "#4E79A7", in_lobula: "#E15759",
    // Same swatches the figures use for the antennal lobe and central complex
    // (nine_region_common.REGION_COLORS), so the viewer and the figures agree.
    in_al: "#76B7B2", in_central_complex: "#3D2C8D",
  };
  const KC_ORDER = ["in_kc_ic", "in_kc_nc", "in_kc_oc"];
  const UNREGIONED = "#39404F";   // cells outside every hand-drawn mask, in region view
  const QUEEN = hexToRgb("#F5C542");
  const WORKER = hexToRgb("#2FD3C6");

  // Display names — the figures' wording, applied where a name is SHOWN, never to
  // the data values (manifest names stay the export's raw strings, so comparisons and a
  // regenerated manifest both keep working). Mirrors src/bimp_st/
  // cluster_labels get_pooled_display / get_canonical_display / MIXED_DISPLAY /
  // DISPLAY_NAME_OVERRIDES and nine_region_common.DISPLAY;
  // identity for any name already correct.
  //   "groups" — the seven pooled groups (or a hierarchical-vote export, where some
  //              clusters keep a fine name): "(all subtypes)" marks a pool.
  //   "types"  — a canonical-17 type (the Kenyon split, and the raw per-cell call): the
  //              generic "Kenyon cells" entry is the reference's unclassified residual.
  // "Mixed" (no decided label at either level) is shown as "Other", as in the figures.
  const GROUP_DISPLAY = {
    "Kenyon cells": "Kenyon cells (all subtypes)",
    "Glia": "Glia (all subtypes)",
    "Projection neurons": "Projection neurons (all subtypes)",
    "Optic lobe": "Optic lobe cells",
    "Mixed": "Other",
  };
  const TYPE_DISPLAY = {
    "Kenyon cells": "Kenyon cells (unclassified)",
    "Glia": "Glia (all subtypes)",
    "Projection neurons": "Projection neurons (all subtypes)",
    "Optic lobe": "Optic lobe cells",
    "Mixed": "Other",
  };
  // Once a glial subtype is named beside it (hierarchical_vote export), the pooled
  // "Glia" is the residual, not the whole lineage: DISPLAY_NAME_OVERRIDES says
  // "Glia (unresolved)". Decided from the manifest at load, so either export reads right.
  const FINE_GLIA = ["Astrocyte-like glia", "Cortex glia", "Perineurial glia",
                     "Subperineurial glia", "Other glia"];
  function configureNames(M) {
    const shown = [].concat(M.canonTypes || [], M.cellTypes || []);
    const residual = shown.some((n) => FINE_GLIA.indexOf(n) >= 0);
    GROUP_DISPLAY.Glia = TYPE_DISPLAY.Glia = residual ? "Glia (unresolved)" : "Glia (all subtypes)";
  }
  // The three Kenyon cell compartments are named for the population that dominates
  // each (fig:atlas_overview panel B / Supp. Methods; nine_region_common.DISPLAY): Large-type =
  // non-compact, Small-type = inner compact, Class II = outer compact. Keyed on the
  // mask column, not the label, so it holds for any manifest vintage.
  const REGION_DISPLAY = {
    in_kc_nc: "Large-type compartment", in_kc_ic: "Small-type compartment",
    in_kc_oc: "Class II compartment",
    in_lamina: "Lamina", in_medulla: "Medulla", in_lobula: "Lobula",
    in_al: "Antennal lobe", in_central_complex: "Central complex",
  };
  const REGION_GROUP_DISPLAY = { "Mushroom body · Kenyon cells": "Mushroom body · Kenyon cell compartments" };
  function displayName(name, vocab) {
    if (name == null) return name;
    const map = vocab === "groups" ? GROUP_DISPLAY : TYPE_DISPLAY;
    return map[name] || name;
  }
  // ---- the cell-type hierarchy (coarse groups and fine-scale types) ----
  // POOL_MAP from src/bimp_st/cluster_labels.py: every fine label to its group. OL_1..OL_13
  // pool to "Optic lobe"; a name that is already a group ("Glia", "Kenyon cells", "Mixed")
  // maps to itself, which makes it that group's residual child.
  const POOL_MAP = {
    "Large-type Kenyon cells": "Kenyon cells", "Middle-type Kenyon cells": "Kenyon cells",
    "Small-type Kenyon cells": "Kenyon cells", "Class II Kenyon cells": "Kenyon cells",
    "Other Kenyon cells": "Kenyon cells", "Kenyon cells": "Kenyon cells",
    "Anterodorsal projection neurons": "Projection neurons",
    "Lateral projection neurons": "Projection neurons",
    "Other neurons": "Other neurons", "Photoreceptors": "Photoreceptors",
    "Astrocyte-like glia": "Glia", "Cortex glia": "Glia", "Perineurial glia": "Glia",
    "Subperineurial glia": "Glia", "Other glia": "Glia", "Hemocytes": "Hemocytes",
  };
  function poolOf(name) { return POOL_MAP[name] || (/^OL_\d+$/.test(name) ? "Optic lobe" : name); }
  // In the tree the group row IS "all subtypes", so it carries the bare name; the residual
  // child is the one that needs qualifying (TYPE_DISPLAY: "Glia (unresolved)").
  const TREE_GROUP = { "Optic lobe": "Optic lobe cells", "Mixed": "Other" };
  function groupName(name) { return TREE_GROUP[name] || name; }
  // cluster_labels.DISPLAY_COLOR_OVERRIDES: the export's residual "Glia" swatch is exactly
  // Perineurial glia's, so the two drew identically once both are on screen.
  const FINE_COLOR = { "Glia": "#9BC4C4" };
  function fineColor(name, hex, residual) { return residual && FINE_COLOR[name] ? FINE_COLOR[name] : hex; }
  // The restricted Kenyon types on the DARK stage. The fig:kc_subtypes five are all ambers
  // and browns; Large-type, Small-type and Class II (the three it leads with) keep its colours
  // exactly, and only these two DEVIATE from the figure, apart by lightness, so that the five
  // can be told apart over black. The export's manifest is untouched.
  const KC_COLOR = { "Middle-type Kenyon cells": "#E8D9A8", "Other Kenyon cells": "#7D6E62" };
  function kcColor(name, hex) { return KC_COLOR[name] || hex; }
  // Kenyon-voted cells outside the compartments were never reclassified: a cool slate, outside
  // the Kenyon warm family, instead of the export's pale "Kenyon cells" sand (now Middle-type's).
  const KC_OUTSIDE = "#5A6170";
  function regionLabel(d) { return REGION_DISPLAY[d.key] || d.label; }
  function regionGroupLabel(g) { return REGION_GROUP_DISPLAY[g] || g; }

  return { HEX, FAMILIES, hexToRgb, QUEEN, WORKER, KC_REGION, KC_ORDER, UNREGIONED,
           displayName, regionLabel, regionGroupLabel, configureNames, poolOf, groupName, fineColor, kcColor, KC_OUTSIDE };
})();
