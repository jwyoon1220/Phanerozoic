# Data

| File | Source | License |
|---|---|---|
| `raw/paleodem/Map01_PALEOMAP_1deg_Holocene_0Ma.nc` | Scotese & Wright (2018), PALEOMAP PaleoDEMs — https://www.earthbyte.org/paleodem-resource-scotese-and-wright-2018/ | CC BY 4.0 (`raw/paleodem/LICENSE.txt`) |
| `raw/dem30/Copernicus_DSM_COG_10_*.tif` | Copernicus DEM GLO-30 (AWS open data, `copernicus-dem-30m`) | Copernicus DEM licence — free use with attribution: "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved" |
| `raw/polyhaven/*` (models: fir_tree_01, island_tree_01, tree_small_02, fern_02, rock_moss_set_01, boulder_01, grass_medium_01, shrub_02, searsia_lucida; textures: forest_leaves_02, forest_leaves_04, forest_ground_04, lichen_rock, rock_surface, sand_01, dry_ground_rocks, sparse_grass, snow_02) | Poly Haven — https://polyhaven.com (fetched by `tools/assets/fetch_polyhaven.py` from `data/assets/polyhaven.txt`, MD5-verified) | CC0 1.0 (public domain, no attribution required; credited anyway) |
| `raw/hygdata_v41.csv` | HYG Database v4.1 — https://github.com/astronexus/HYG-Database (Hipparcos, Yale BSC, Gliese) | CC BY-SA 4.0 (see `raw/HYG_LICENSE`) |

`raw/` holds source data; it is downloaded automatically by the build if missing
(`cmake/Download.cmake`). Baked runtime data (`stars.bin`) is written to `<build>/data/`
by `phx_bake_stars`. The baked star file is a derivative of HYG and therefore also
CC BY-SA 4.0 — credit "HYG Database by David Nash (astronexus)" in the game credits.

The global DEM uses the **0 Ma (modern geography)** PaleoDEM — the 50 Ma map is in
palaeo-coordinates and was rejected so that Huashan and the star sky keep their real
coordinates. Eocene corrections (sea level, Tibet, polar ice) are applied at load time
in `engine/world/HeightField.cpp`. Cite: Scotese, C.R., and Wright, N., 2018, PALEOMAP
Paleodigital Elevation Models (PaleoDEMS) for the Phanerozoic.

Regional 30 m cells (`PHX_DEM30_CELLS` in CMake, default 33–34°N × 109–110°E around
Huashan) are baked by `phx_bake_dem30` into mip-mapped `.phxdem` files and memory-mapped
at runtime. GLO-30 is a *surface* model (DSM): it includes modern forest canopy and
buildings (e.g. Huayin town at the foot of the massif), and its 30 m posts undercut sharp
summits (Huashan South Peak: 2016 m in the DEM vs. 2154.9 m surveyed).

Tree species are **stand-ins** chosen by silhouette and foliage, because no CC0 models
of the Eocene genera exist: fir_tree_01 → Metasequoia (rust autumn tint applied in the
shader), island_tree_01 → Lauraceae evergreen, tree_small_02 → Ginkgo (gold autumn
tint). `phx_bake_tree` reduces the film-resolution sources (1.6–4.2 M triangles) to three
near-field LODs; distant trees are impostors baked at start-up from the full meshes.
