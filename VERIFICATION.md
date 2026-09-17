# Release verification — Tectonic Globe 4.0

These are checks of this implementation, not geological validation against Earth.
Run in the development container with Python 3.13.5, NumPy
2.3.5, SciPy 1.17.0 and Numba 0.65.1.

## Numerical checks

- A spherical mesh's area agreed with 4*pi*R^2 to relative error 1.17e-16.
- The integrated solid-rotation flux had maximum discrete divergence
  9.68e-18 per Myr.
- A constant carrier field remained constant to 4.44e-16.
- A broad material cap transported for 100 Myr had relative volume change
  0 and centroid error
  0.00585 degrees against analytic Euler
  rotation. Its initially uniform 35 km column differed by at most
  1.14e-13 km afterwards.
- An isolated continental collision transferred volume rather than deleting it:
  56.5 volume-units before and after; overriding thickness rose from 40 to
  61.875 km. This is a controlled kernel example, not a geographical prediction.
- An isolated ocean/continent collision consumed 0.4 area-units of ocean and
  retained the continent. A spreading gap created 0.3 area-units of age-zero
  seafloor without creating age moment. A continental rift reduced a 35 km
  column to 26.25 km at unchanged volume.
- A strong lateral-exchange test with a tiny receiving carrier created no new
  thickness extrema; final columns were 35.000
  to 38.950 km from an initial 35–70 km range.
  Relative volume error was -1.1e-18.
- The ocean plate-cooling curve increased monotonically at ages 0, 10, 50, 100,
  200 and 500 Myr, from 2.6 km towards the configured 6 km asymptote.

## Included default world

The full 1000 Myr evolution completed in 668 steps on
10,242 spherical cells (223.2 km nominal
spacing). 18 initial plates became 8 plates,
with 6 prescribed rift events and 16 triggered
continental sutures. Current states and snapshots are in the included cache.

The continental volume ledger (initial minus explicitly logged sediment loss)
closed to relative error -9.17e-16.
Maximum summed-coverage error was 1e-13.
The minimum transported density seen was -1.95e-17,
consistent with roundoff, not a macroscopic negative area.
The solved water volume was 1,350,000,000 km^3 for a target
of 1,350,000,000 km^3. This is on the coarse numerical mesh.

The present guide ranged from -4.944 to
4.891 km relative to sea level. Interpolated emerged
land covered approximately 16.3% of the sphere.
Fine shader relief can extend beyond that coarse range. Land percentage is an
outcome, not a forced Earth match. 42,760 coarse river
reaches were generated; 0 uphill coarse
receiver links were detected. That does not guarantee metre-scale hydrology.

The fresh guide build took 84.0 s, including
73.8 s for tectonic evolution/mesh setup. This is one CPU
measurement, not a cross-machine speed promise. The archive includes this cache;
an unchanged configuration reuses it.

## Browser and terrain coupling

The actual GLSL programs compiled and rendered in WebGL2 using:

    ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)

A 1,000 x 760 viewer rendered spans of 18,500 km, 2,600 km, 100 km and 1 km.
No JavaScript exceptions, WebGL errors or non-finite visible heights were reported.
The 1 km test is about 1 m/pixel. Completion times with blocking readback were
1.66, 1.88, 3.10, 5.73 s respectively on software
rendering. These are not hardware-GPU frame rates or cache-miss performance tests.

An ablation preserved the same coarse elevation and set transported deformation
history and derived ruggedness inputs to zero. On 743,410
land pixels in the regional test, bare-ground height changed by a mean of
625.5 m and a maximum of
3417.5 m. The test verifies
that the controls change geometry, not merely colour. It does not validate the
chosen artistic amplitude calibration. The original maps were then restored.

The existing + button and wheel changed the view width as expected:
2600.0 -> 1300.0 ->
1300.0 km. Python syntax and configuration validation
also passed, and a fresh process found the expected cached world.

The container's browser policy blocked ordinary URL navigation, including
localhost. Browser tests therefore used the real local HTTP server through an
in-memory fetch bridge. Server responses and browser execution were tested, but
normal localhost navigation was not directly verified in this environment.
The shipped viewer itself contains no such bridge or testing instrumentation.

## Not established by these checks

There is no independent geological calibration, resolution-convergence study of
the billion-year dynamics, mantle-force validation or photographic comparison.
The prescribed kinematics and collision/relaxation rules remain reduced-order.
No claim is made that all climate boundaries, coastlines or local rivers are
physically correct. The code does not reproduce the complete Cortial application.
