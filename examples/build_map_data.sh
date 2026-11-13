#!/usr/bin/env bash
# Rebuild the Network map's data from Natural Earth (public domain) into ONE
# file, `frontend/data/countries.json`: TopoJSON already projected (Equal
# Earth) and fitted to a 1000-pixel-wide frame, origin at the top left, so the
# dashboard draws it as it is and does no map arithmetic of its own. Objects:
#
#   default, IN  — country outlines in two border views: Natural Earth's de
#                  facto borders, and its India point-of-view edition (India's
#                  official boundaries), shown to a computer in India as Google
#                  Maps and Mapbox's `worldview` do. Each outline carries `c`,
#                  its two-letter code (null for an area no country code names).
#   points       — each code's label point (Natural Earth's LABEL_X / LABEL_Y,
#                  placed by hand inside a country's main body), where a dot
#                  and a route arc go (`network-map.js::regionCenter`). A
#                  country too small to keep an outline is drawn from it alone.
#   frame, grid, equator — the map's edge and a 30° graticule.
#
# Prior art: Mike Bostock's "Command-Line Cartography" and us-atlas v3 ship
# PRE-PROJECTED TopoJSON for a static map, drawn with no projection in the
# browser; mapshaper's `-o width=` writes exactly that ("pixel coordinates,
# origin in the top left ... for direct conversion to SVG"). Its `-proj`
# cuts polygons at the antimeridian, `densify` bends straight edges as the
# projection does and `clip-bbox` cuts the frame — the three things the
# dashboard's own projection code had done by hand (and the antimeridian one
# wrongly for a filled ring). The browser-side alternative, d3-geo, would add
# ~36 KB + d3-array to the payload for a map that never moves.
#
# Both views are simplified together, so a border they share is the same line
# in both (mapshaper keeps the topology of combined files).
#
# Usage: examples/build_map_data.sh [work dir]   (needs curl, node/npx, python3)
set -euo pipefail

NE_TAG=v5.1.2
MAPSHAPER=mapshaper@0.6.121
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/frontend/data/countries.json"
WORK="${1:-$(mktemp -d)}"
mkdir -p "$WORK"
cd "$WORK"

base="https://raw.githubusercontent.com/nvkelso/natural-earth-vector/$NE_TAG/geojson"
for f in ne_10m_admin_0_countries ne_10m_admin_0_countries_ind \
         ne_50m_admin_0_countries ne_50m_admin_0_map_units; do
  [ -s "$f.geojson" ] || curl -sSfL -m 300 -o "$f.geojson" "$base/$f.geojson"
done
cp ne_10m_admin_0_countries.geojson default.geojson
cp ne_10m_admin_0_countries_ind.geojson IN.geojson

# Label points, the graticule and the frame, in longitude and latitude; the
# projection bends them with everything else.
python3 -I - "$WORK" <<'EOF'
import json, sys
work = sys.argv[1]
def code(p):
    for k in ("ISO_A2_EH", "ISO_A2", "WB_A2"):
        v = p.get(k)
        if isinstance(v, str) and len(v) == 2 and v.isalpha():
            return v.upper()
    return None
at = {}
# Countries first; map units add the places an address lookup names on their
# own though Natural Earth counts them in a country (French Guiana, Réunion);
# the 1:10m countries the outlines come from add what 1:50m leaves out
# (Gibraltar), so every outline drawn has a point.
for name in ("ne_50m_admin_0_countries", "ne_50m_admin_0_map_units", "ne_10m_admin_0_countries"):
    for f in json.load(open(f"{work}/{name}.geojson"))["features"]:
        p = f["properties"]
        c = code(p)
        if c and c != "AQ" and c not in at:
            at[c] = [p["LABEL_X"], p["LABEL_Y"]]
def layer(name, features):
    json.dump({"type": "FeatureCollection", "features": features}, open(f"{work}/{name}.geojson", "w"))
def line(coords):
    return {"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": coords}}
layer("points", [{"type": "Feature", "properties": {"c": c},
                  "geometry": {"type": "Point", "coordinates": xy}} for c, xy in sorted(at.items())])
S, N = -58, 84   # the frame: Antarctica is not drawn, nor the empty band below Cape Horn
layer("grid", [line([[lon, S], [lon, N]]) for lon in range(-150, 151, 30)]
              + [line([[-180, lat], [180, lat]]) for lat in (-30, 30, 60)])
layer("equator", [line([[-180, 0], [180, 0]])])
layer("frame", [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon",
    "coordinates": [[[-180, S], [-180, N], [180, N], [180, S], [-180, S]]]}}])
print(f"{len(at)} label points")
EOF

# A code: Natural Earth's ISO_A2_EH where it is two letters; Somaliland and
# Northern Cyprus carry none and are drawn with Somalia and Cyprus — what an
# address lookup answers for a computer there. Simplified AFTER projecting, so
# what `densify` added stays only where the projection bends a line; 25 km
# between kept points (a pixel of the 1000-wide map is ~40 km), and Equal Earth
# keeps areas true, so the island cut is in real square kilometres.
proj=(-proj +proj=eqearth densify clip-bbox=-180,-58,180,84 target='*')
fit=(-o format=topojson width=1000 margin=4 quantization=10000 target='*')
npx --yes "$MAPSHAPER" -i default.geojson IN.geojson points.geojson frame.geojson combine-files \
  -each "c = (ISO_A2_EH && /^[A-Z]{2}$/.test(ISO_A2_EH)) ? ISO_A2_EH : (NAME == 'Somaliland' ? 'SO' : (NAME == 'N. Cyprus' ? 'CY' : null))" target=default,IN \
  -filter "c != 'AQ'" target=default,IN \
  -filter-fields c target=default,IN \
  "${proj[@]}" \
  -simplify visvalingam weighted interval=25000 keep-shapes \
  -filter-islands min-area=300km2 remove-empty target=default,IN \
  "${fit[@]}" outlines.json
# The graticule in a run of its own: in the outlines' topology every crossing
# of a grid line and a border splits both into more arcs (the file grew by
# half). The same frame in both runs gives both the same fit and the same
# quantization, so its arcs join the outlines' file as they are.
npx --yes "$MAPSHAPER" -i grid.geojson equator.geojson frame.geojson combine-files \
  "${proj[@]}" "${fit[@]}" lines.json

# An outline too small to survive the quantization (Vatican City) comes out
# with no geometry; topojson-client hands it back as `geometry: null`. Drop it
# here — the country is still drawn, from its label point — so no reader of
# the file meets one.
python3 -I - outlines.json lines.json "$OUT" <<'EOF'
import json, sys
outlines, lines, out = sys.argv[1:]
topo = json.load(open(outlines))
grid = json.load(open(lines))
if grid["transform"] != topo["transform"]:
    sys.exit(f"the two runs were fitted differently: {grid['transform']} vs {topo['transform']}")
offset = len(topo["arcs"])
def shift(a):
    if isinstance(a, list):
        return [shift(x) for x in a]
    return a + offset if a >= 0 else ~(~a + offset)
topo["arcs"] += grid["arcs"]
for name in ("grid", "equator"):
    obj = grid["objects"][name]
    for g in obj["geometries"]:
        g["arcs"] = shift(g["arcs"])
    topo["objects"][name] = obj
points = {g["properties"]["c"] for g in topo["objects"]["points"]["geometries"]}
for view in ("default", "IN"):
    obj = topo["objects"][view]
    empty = [g["properties"].get("c") for g in obj["geometries"] if not g.get("type")]
    unmarked = [c for c in empty if c and c not in points]
    if unmarked:
        sys.exit(f"{view}: {unmarked} lost their outline and have no label point")
    obj["geometries"] = [g for g in obj["geometries"] if g.get("type")]
    print(f"{view}: {len(obj['geometries'])} outlines; drawn from their point alone: {empty}")
json.dump(topo, open(out, "w"), separators=(",", ":"))
EOF
echo "built $OUT in $WORK"
