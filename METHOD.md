# How this planet is generated

This document describes the implemented model, including its compromises. It is
not a claim to reproduce Cortial et al.'s full implementation or a calibrated
geodynamic model. The procedural plate-evolution / subsequent terrain-amplification
architecture follows the problem formulation in their 2019 paper [1]. The actual
finite-volume transport and collision rules below are this implementation.

## 1. Initial crust and plates

Time runs FORWARD from a synthetic 1,000 Ma initial condition to a synthetic
present. Continuous, seeded 3-D scalar fields on the sphere specify the initial
continental crust. Independently, spherical Voronoi cells initialize plate IDs.
Each plate can contain both continental and oceanic crust; it is not one island.

After initialization the Voronoi diagram is NOT recomputed from moving seeds.
Doing that would relabel material instead of transporting its history. Plates
are represented as moving area fractions on the finite-volume sphere. The
integration mesh is the spherical Voronoi dual of a refined icosahedron, not a
latitude-longitude raster. Longitude seams and poles therefore require no special
advection boundary conditions. Texture maps are only made after evolution.

Per plate and cell, the state stores continental area fraction C, oceanic fraction
O, continental crust volume per cell area V, oceanic age moment A, compression,
extension and shear histories, and an accreted basalt volume. Initially C+O,
summed over plates, is one. Continental and oceanic material have different
prescribed densities; continental thickness can differ between initial plates.

## 2. Plate motion and numerical transport

The rigid velocity on plate p is

    v_p(n) = R (omega_p cross n).

Here n is a unit surface position, R is radius in km and omega is radians/Myr.
Euler-pole rotations are the standard spherical plate-motion representation [2].
One cm/year equals 10 km/Myr. A common initial whole-globe rotation is removed
because it changes reference frame, not relative plate motion.

Each dual cell edge has an analytically integrated normal solid-rotation flux:

    Q_ij = R^2 omega dot (dual_endpoint_2 - dual_endpoint_1).

Endpoint orientation is chosen consistently. Fluxes telescope around each cell,
so a constant field stays constant under rigid rotation. Equal and opposite
fluxes transfer extensive quantities between adjacent cells.

Area carriers use limited MUSCL reconstruction (monotonized-central limiter) and
two-stage SSP Runge–Kutta integration. An approximately collinear upstream mesh
neighbour supplies the backward difference. This is an unstructured adaptation,
not a claim of a full formal convergence proof for this particular discretization.
The transport method uses the conservative reconstruction approach described in
[3]; the chosen time step is also limited by the sum of outgoing area fluxes.

V and A use the SAME reconstructed C and O fluxes, respectively. Other extensive
tracers use the combined coverage flux. Tracer concentrations use donor-cell
values: tracer transport is first order in space, while carrier reconstruction
is second order. This deliberately avoids retaining crust volume where C has
become zero, or age where O is zero. Transport is conservative; it is not free of
numerical diffusion over long runs.

Angular velocities are prescribed, with seeded targets changing on configured
epochs and a finite response time. There is no solved mantle flow, traction,
slab pull, ridge push, viscosity field or feedback from forces onto those targets.

## 3. Collision, subduction and stretching

Different plate velocities produce locally superposed or incomplete coverages.
Interaction rules resolve this after each transport step.

For excess coverage, oceanic material is consumed first, oldest first. Removed
age and deformation memories leave with their carrier. A configured fraction of
consumed ocean-crust volume supplies an approximate magmatic addition to an
area-weighted buoyant upper plate. The nominal consumed oceanic crust is 7 km
thick. The basalt volume created and subsequently removed is accounted separately;
no mantle melting or magma chemistry is calculated.

Where continental material remains in excess, the locally dominant continental
carrier is the overriding plate. The thinner of the other continental patches
is consumed first. Horizontal area is removed but its crust volume is transferred
to the overriding continent. Thus shortening increases thickness V/C instead of
merely changing a pixel's colour. Selecting the upper plate by dominance is an
explicit mixed-cell rule, not a solved fault-polarity or slab-dip model.

Where coverage is deficient, a still-thick continental margin stretches into its
share of the opening without receiving new crust volume. Its thickness decreases.
Once below the breakup threshold, further openings create age-zero oceanic crust.
Oceanic area and age are then advected, can be subducted later, and cool with time.

Configured rift events split a large plate along a great circle through its
weighted centre. The two parts inherit material unchanged, then receive opposing
velocity increments. Rifts are prescribed events. Suture events combine plates
when their integrated continental contact exceeds a configured consumed-area
threshold. The material survives the label merge. Both types of event are logged.

## 4. Root spreading and history

Continental and basalt thickness differences drive conservative, same-plate
lateral spreading. Its coefficient has a bounded thickness-dependent mobility.
Each connected edge pair is integrated as an exact two-reservoir diffusion step:

    d(Hi-Hj)/dt = -K (1/mi + 1/mj) (Hi-Hj),

with carrier masses mi=Ci*area_i (or total plate area for basalt). Equal and
opposite crust volume is transferred using the exponential solution. A forward
and reverse half-sweep reduces edge-order bias. Every pair is a convex mixture,
so even a tiny receiving carrier cannot create a new thickness maximum. This is
an operator-split reduced relaxation, not a full thin-viscous-sheet or flexure
solve; splitting error and frozen-coefficient error remain.

Continental volume above the reference-thickness column relaxes into an explicit
unresolved sediment reservoir. The rate grows as (H/H_reference)^power, so
sustained prescribed convergence cannot build arbitrarily thick roots against a
fixed, very slow denudation rate. The power and timescale are exposed model
parameters, not fitted rheology or a substitute for solved force balance. Arc
addition also relaxes. Those volumes are recorded as sinks in the surface-crust
budget; sediment is NOT spatially deposited in basin geometry. Total
mantle/crust/sediment chemistry and energy are not solved.

Compression, extension and shear are material histories transported with crust
and exponentially decayed using separate configured lifetimes. These channels
can therefore remain away from an active present-day boundary. Current boundary
rates are independently calculated from relative Euler velocities, decomposed
into convergence, divergence and tangential sliding. A transform is not classified
as a collision just because its total relative speed is large.

History-to-mask mappings are saturating functions, for example

    compression_mask = 1 - exp(-compression_history / strain_scale).

They are artistic control fields derived from simulated state, not measurements
of a particular mountain range's elevation or proof that every old belt survives.

## 5. Buoyancy, ocean cooling and sea level

Continental freeboard uses a common reference column and a simplified Airy law:

    delta_h = (1 - rho_continent/rho_mantle) delta_crust_thickness.

Different plates consequently develop different average heights through material
composition and deformation; no random per-polygon vertical offsets are added
at the end. This is deliberately a crustal approximation. Real continental
buoyancy also depends on lithospheric mantle structure [4]; crust thickness alone
is not sufficient to predict real elevations.

Ocean depth follows a finite-thickness conductive plate-cooling Fourier series:

    tau = L^2 / (pi^2 kappa)
    cooling(t) = 1 - sum_{odd k=1..63}[exp(-k^2 t/tau)/k^2]
                       / sum_{odd k=1..63}[1/k^2]
    d(t) = d_ridge + d_subsidence * cooling(t).

The finite sum is normalized so newborn seafloor is exactly at the configured
ridge depth. The model has the young-floor subsidence and old-floor saturation
behaviour of plate-cooling models [5], but its coefficients remain exposed rather
than fitted to observations. Continental crust thickness and thermal plate
thickness are different quantities. Current trenches are added on the oceanic
side as a parameterized freeboard correction, not by evolving slab geometry.

The planet has a fixed water volume. Sea datum L is found by solving

    V_water = sum_i area_i max(L - b_i, 0) / (1 - rho_water/rho_mantle),

where b is unloaded freeboard. The resulting submerged elevations include water
loading. Sea level is not adjusted to enforce Earth's emerged-land percentage.
The budget refers to the coarse simulated surface; procedural fine relief and
coarse interpolation can slightly change the rendered basin volume.

## 6. The outputs really feed the terrain

The returned simulation fields produce these equirectangular float32 maps:

| File | R | G | B | A |
|---|---|---|---|---|
| `macro.bin` | isostatic height, km | orogenic ruggedness | coast distance, km | surface-material proxy |
| `tectonics.bin` | transported compression memory | transported extension memory | transported shear memory | ocean age, Myr |
| `crust.bin` | continental fraction | conditional continental thickness, km | igneous height contribution, km | ruggedness |
| `climate.bin` | sea-level temperature | log rainfall | current convergence strength | current divergence strength |

Coast distance is a raster approximation, not a geodesic distance solver. Crust
thickness and ocean age are zero where their carrier is effectively absent.
The map dimensions are recorded in `meta.json`; rows run south to north.

The next terrain stage starts at the simulated height, not an unrelated height
field. For each relief or erosion layer k, its local amplitude includes

    plain_fraction + (1-plain_fraction) * clamp(
        ruggedness * collision_gain[k]
        + extension_memory * rift_gain[k]
        + shear_memory * transform_gain[k], 0, 1).

Ruggedness includes surviving crustal thickening and compression history.
Compression history also changes erosion ridge rounding: the old-belt multiplier
is strongest where recent compression has decayed. A mask's value does not
independently determine elevation; it modulates refinement of the already
isostatic terrain. Mantle-derived basalt and continental fraction contribute to
surface material selection. Climate and coarse drainage are rebuilt from the new
terrain guides. Runevision's fading-gully filter [6] supplies the existing local
mountain detail; it does not solve tectonics or guarantee connected local streams.

## 7. Resolution and limits

The default mesh has 10,242 cells, about 223 km nominal spacing. This limits the
width and location accuracy of resolved tectonic belts. Interpolation and
procedural refinement do not increase the physical simulation's resolution.
The model is seeded and repeatable in the tested environment, not guaranteed
bit-identical across all numerical libraries/platforms.

The model evolves overlapping material fractions rather than a fully conforming
moving polygon mesh. Mixed interfaces are diffuse and numerical diffusion affects
small continents. Continental polarity, magma production, rifts, sutures and root
relaxation are simplified rules. There is no physical calibration establishing
realistic plate-area statistics, continental hypsometry or billion-year trajectories.
The default world's geography is an example output, not Earth reconstructed.

## Sources

[1] Cortial, Peytavie, Galin & Guérin (2019), *Procedural Tectonic Planets*,
Computer Graphics Forum 38(2), 1–11. Architecture/reference problem, not copied
simulation code. https://doi.org/10.1111/cgf.13614

[2] GPlates, *FiniteRotation* documentation. Spherical Euler-pole representation;
this app does not depend on pyGPlates.
https://www.gplates.org/docs/pygplates/generated/pygplates.finiterotation

[3] van Leer (1979), *Towards the ultimate conservative difference scheme. V. A
second-order sequel to Godunov's method*, Journal of Computational Physics 32,
101–136. https://doi.org/10.1016/0021-9991(79)90145-1

[4] USGS, *Lithospheric buoyancy and continental intraplate stresses* (2003).
https://www.usgs.gov/publications/lithospheric-buoyancy-and-continental-intraplate-stresses

[5] Parsons & Sclater (1977), *An analysis of the variation of ocean floor
bathymetry and heat flow with age*, Journal of Geophysical Research 82(5),
803–827. https://doi.org/10.1029/JB082i005p00803

[6] Rune Skovbo Johansen (2026), *Fast and Gorgeous Erosion Filter*.
https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html

The retained global drainage guide uses Priority-Flood depression filling;
see Barnes, Lehman & Mulla, *Priority-Flood: An optimal depression-filling and
watershed-labeling algorithm for digital elevation models* (2014),
https://doi.org/10.1016/j.cageo.2013.04.024.
