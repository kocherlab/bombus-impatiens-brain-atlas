"""Which annotation GeoJSON covers which crop, for the queen (brain1) brain.

Data only, no imports: the shared code in ../ (labeling_core.py) and
bimp_st.nine_region_common._load_labeling both execute this file standalone and
read ANNOTATION_MAPPING. It is the single source of the annotation-to-crop
table.
"""

# Maps annotation files to their target h5ad crop files.
# Each annotation covers specific brain(s) which correspond to A or B crops.
#
# Format: "annotation_filename": ["crop_suffix1", "crop_suffix2", ...]
# The crop suffix is the part after the region name (e.g., "2_7A", "2_7B")
#
# To determine which crop your annotation belongs to:
# 1. Check annotation Y range in pixels from GeoJSON
# 2. Compare to crop Y ranges from convert_single_region.py:
#    - 2_7A: Y 0-18555 pixels (0-3943 µm)
#    - 2_7B: Y 18555-37204 pixels (3943-7906 µm)
# 3. Annotation with Y > 18555 pixels belongs to B crop
#
ANNOTATION_MAPPING = {
    # Slide 1 (0040240) - no prefix
    "output-XETG00151__0040240__Region_1__20251204__195015.geojson": [
        "1A",
        "1B",
    ],
    "output-XETG00151__0040240__Region_2__20251204__195015.geojson": [
        "2A",
        "2B",
    ],
    "output-XETG00151__0040240__Region_3__20251204__195015.geojson": [
        "3A",
        "3B",
    ],
    "output-XETG00151__0040240__Region_4__20251204__195015.geojson": [
        "4A",
        "4B",
    ],
    "output-XETG00151__0040240__Region_5__20251204__195015.geojson": [
        "5A",
        "5B",
    ],
    "output-XETG00151__0040240__Region_6__20251204__195015.geojson": [
        "6A",
        "6B",
    ],
    "output-XETG00151__0040240__Region_7__20251204__195015.geojson": [
        "7A",
        "7B",
    ],
    "output-XETG00151__0040240__Region_8__20251204__195015.geojson": [
        "8A"
    ],
    # Slide 2 (0040243) - prefix 2_
    "output-XETG00151__0040243__Region_1__20251204__195015.geojson": [
        "2_1A",
        "2_1B",
    ],
    "output-XETG00151__0040243__Region_2__20251204__195015.geojson": [
        "2_2A",
        "2_2B",
    ],
    "output-XETG00151__0040243__Region_3__20251204__195015.geojson": [
        "2_3A",
        "2_3B",
    ],
    "output-XETG00151__0040243__Region_4__20251204__195015.geojson": [
        "2_4A",
        "2_4B",
    ],
    "output-XETG00151__0040243__Region_5__20251204__195015.geojson": [
        "2_5A",
        "2_5B",
    ],
    "output-XETG00151__0040243__Region_6__20251204__195015.geojson": [
        "2_6A",
        "2_6B",
    ],
    "output-XETG00151__0040243__Region_7__20251204__195015.geojson": [
        "2_7A",
        "2_7B",
    ],
    "output-XETG00151__0040243__Region_8__20251204__195015.geojson": [
        "2_8A"
    ],
}
