# Third-party licenses (frontend)

The SwarmLLM frontend bundles the following third-party assets. Their
upstream license terms apply to the bundled copies.

## topojson-client

- File: `js/topojson-client.min.js` (v3.1.0)
- Upstream: https://github.com/topojson/topojson-client
- Copyright 2019 Mike Bostock
- License: BSD 3-Clause
  https://github.com/topojson/topojson-client/blob/master/LICENSE

## World map outlines and country points

- File: `data/countries.json` — country outlines (Natural Earth v5.1.2's
  1:10m admin-0 countries and its India point-of-view edition, simplified
  together) and each country's label point (LABEL_X / LABEL_Y, from the 1:50m
  admin-0 countries and map units), projected into one TopoJSON by
  `examples/build_map_data.sh` with mapshaper (MPL-2.0; a build tool, not
  bundled)
- Upstream: https://www.naturalearthdata.com/ ·
  https://github.com/nvkelso/natural-earth-vector
- Natural Earth is in the public domain
  https://www.naturalearthdata.com/about/terms-of-use/

## IBM Plex Sans / IBM Plex Mono

- Files: `fonts/ibm-plex-sans-latin-wght-normal.woff2` (variable, wght 100–700),
  `fonts/ibm-plex-mono-latin-{400,600,700}-normal.woff2`
- Upstream: https://github.com/IBM/plex
- Copyright 2017 IBM Corp.
- License: SIL Open Font License 1.1
  https://github.com/IBM/plex/blob/master/LICENSE.txt

Latin subsets, taken from the Fontsource builds. Bundled rather than fetched
from a CDN so the dashboard renders identically offline and on an air-gapped
node, and so loading it tells no third party that this node exists.
