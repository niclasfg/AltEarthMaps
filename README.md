# Tectonic Globe 4.0

A local, seeded, Earth-sized procedural globe. Python evolves a coarse spherical
crust model through 1,000 million years, builds climate and major drainage guides,
and serves a WebGL2 viewer. The browser adds the existing Runevision-based fine
terrain. No accounts, API keys, map services or imagery downloads are used.

## Start

Use Python 3.11–3.13. Open a terminal in this folder and run:

```sh
python -m pip install -r requirements.txt
python terrain.py
```

The viewer opens at http://127.0.0.1:8770. Drag to rotate the globe, scroll to zoom,
and use the existing +/−/Globe buttons. Keep the terminal open. Ctrl+C stops it.
The browser must support WebGL2, floating-point render targets and at least 11
texture units. Enable hardware acceleration. No 3-D editing tools are required.

The ZIP includes the completed default world's cache. Its first launch therefore
loads that world rather than simulating it again. A changed seed, tectonic setup,
planet, climate or coarse hydrology configuration rebuilds the guide automatically.
On the development CPU, a fresh default build took about 84 seconds. This is not
a prediction for every machine. Increasing `mesh_subdivisions` to 6 costs much
more CPU time and memory. Later panning/zooming does not rerun geological time.

## Configure

All exposed settings are in `config.toml`, with units and explanatory comments.
Save the file, stop Python, and start it again. Browser refresh alone cannot change
server-generated shader constants. The cache folder name depends on generation
settings and the Python simulation source, so unrelated previous worlds are not
silently reused. Old cache folders can be deleted while the program is stopped.

`[tectonics]` controls the duration, spherical simulation mesh, plate count,
initial thickness and velocity vectors, rift ages, density contrasts, oceanic
cooling, strain-memory lifetimes and root spreading. A one-element plate vector
is broadcast; a longer vector must have one value per initial plate. Rift ages
are in millions of years BEFORE the generated present and must lie between zero
and the configured duration. They are prescribed model events, not observations.

`[relief]` and `[erosion]` retain per-wavelength vectors. Their `collision_gain`,
`rift_gain` and `transform_gain` independently set each layer's response to the
tectonic output. A single value broadcasts over that section's wavelengths.
These controls affect actual generated elevation, not just display colours.
The erosion core remains Runevision-based; the coordinate and scale adapters
are retained from the preceding globe build.

Ocean volume is in cubic kilometres. Sea level is solved from that inventory and
the evolving crust's buoyancy; there is no after-the-fact percentage-land slider.
Initial continental crust fraction includes submerged continental margins and
is not the same as the final emerged-land fraction.

## What this model is

A reduced-order **kinematic** lithosphere model, not a mantle-convection solver
or a reconstruction of Earth's actual last billion years. Euler rotations are
prescribed and slowly varied. Crust is transported conservatively; collisions,
spreading, rifting, subduction and buoyancy update its state. Plate motion is not
computed from slab-pull/ridge-push force balance. The numerical checks establish
specific implementation properties, not geological predictive accuracy.

The default tectonic mesh is about 223 km across a cell. The shader's metre-scale
relief is conditional procedural refinement, not metre-resolution tectonics.
Major drainage is routed on a separate coarser global grid. Local erosion gullies
are not all guaranteed to connect to that coarse river skeleton. Climate, snow,
vegetation and colour remain approximate and are not photographic satellite data.

Read `METHOD.md` for the equations, approximations, output channels and sources;
read `VERIFICATION.md` for the checks actually run on this release.

## Files

- `tectonics.py`: moving crust, collisions, spreading, cooling and isostasy.
- `planet.py`: initial crust field, coarse map sampling, climate and drainage.
- `terrain.py`: configuration validation and local HTTP server.
- `viewer.js`, `index.html`, `shaders/`: the existing globe viewer and refinement.
- `cache/<key>/tectonics_state.npz`: present-day numerical fields on sphere nodes.
- `cache/<key>/tectonics_history.npz`: saved coarse snapshots, not resumable checkpoints.
- `cache/<key>/tectonics_report.json`: plate averages, events and numerical budgets.
- `cache/<key>/*.bin`: the maps actually consumed by the browser.

No test framework, external asset pack or extra editor UI is bundled.
