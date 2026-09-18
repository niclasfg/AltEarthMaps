# Source and modification notices

The Runevision Phacelle Noise / Advanced Terrain Erosion Filter sections are
Copyright (c) 2025 Rune Skovbo Johansen and distributed under MPL-2.0. Their
notices remain in the GLSL. See `licenses/MPL-2.0.txt` and the retained reference
file `shaders/runevision_reference.glsl`.

The spherical coordinate adapter, integer hash, per-layer configuration, display
sampling and tectonic modulation are modifications to the preceding globe build.
They are not the unmodified Shadertoy demonstration, nor a claim to reproduce its
perspective rendering pixel-for-pixel. The modified GLSL source is included.

The new `tectonics.py` is an original reduced-order implementation, not copied
from the Cortial research application or from GPlates. Its mathematical and
geophysical references and the limits of the analogy are in `METHOD.md`.

NumPy, SciPy and Numba are external dependencies installed through pip. They are
not bundled into this ZIP; their respective licenses apply to those packages.
The retained Python/JavaScript viewer and newly written Python tectonic code are
provided under `licenses/MIT.txt`. The MPL-covered GLSL sections retain MPL-2.0.

The continent seeding/growth recipe (`continents.py`, written independently for
this codebase) follows standard tectonic-planet practice; Red Blob Games'
planet-generation notes and the World Orogen generator's ocean-land assignment
(GPL-3.0) were used as method references only, with no code reused. The
settings panel's tabs/overlays/regenerate layout is likewise inspired by World
Orogen's interface; its implementation here is original.
