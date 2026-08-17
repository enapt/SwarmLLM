# Third-party licenses (frontend)

The SwarmLLM frontend bundles the following third-party assets. Their
upstream license terms apply to the bundled copies.

## topojson-client

- File: `js/topojson-client.min.js` (v3.1.0)
- Upstream: https://github.com/topojson/topojson-client
- Copyright 2019 Mike Bostock
- License: BSD 3-Clause
  https://github.com/topojson/topojson-client/blob/master/LICENSE

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
