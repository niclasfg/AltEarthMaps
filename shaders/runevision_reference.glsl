/*
Advanced Terrain Erosion Filter and Phacelle Noise
Copyright (c) 2025 Rune Skovbo Johansen.
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. See ../licenses/MPL-2.0.txt.

Source: https://www.shadertoy.com/view/wXcfWn (Common and Buffer A).
Source snapshot: Ethanout/folded-mountains-datapack, commit
bf68a28eaf27a9793dd259af7a019c492def0b33,
references/shadertoy-wXcfWn/{common,buffer-a-terrain}.glsl.

The executable expressions below preserve the reference algorithms.
Only comments/whitespace and unused demonstration code have been removed.
The reference hash and noised functions originate with Inigo Quilez:
https://www.shadertoy.com/view/XdXBRH . See THIRD_PARTY.md.

Do not replace this with a generic noise or a 3-by-3 Gaussian convolution.
The 4-by-4 support, truncated weight, separate slope accumulators, mask
recurrence, normalization and amplitude schedule are all significant.
*/
#define PI 3.14159265358979
#define TAU 6.28318530717959
#define clamp01(x) clamp(x, 0.0, 1.0)

vec2 hash(in vec2 x) {
    const vec2 k = vec2(0.3183099, 0.3678794);
    x = x * k + k.yx;
    return -1.0 + 2.0 * fract(16.0 * k * fract(x.x * x.y * (x.x + x.y)));
}

vec3 noised(in vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    vec2 ga = hash(i + vec2(0.0, 0.0));
    vec2 gb = hash(i + vec2(1.0, 0.0));
    vec2 gc = hash(i + vec2(0.0, 1.0));
    vec2 gd = hash(i + vec2(1.0, 1.0));
    float va = dot(ga, f - vec2(0.0, 0.0));
    float vb = dot(gb, f - vec2(1.0, 0.0));
    float vc = dot(gc, f - vec2(0.0, 1.0));
    float vd = dot(gd, f - vec2(1.0, 1.0));
    return vec3(va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd),
        ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) +
        du * (u.yx * (va - vb - vc + vd) + vec2(vb, vc) - va));
}

vec4 PhacelleNoise(in vec2 p, vec2 normDir, float freq, float offset, float normalization) {
    vec2 sideDir = normDir.yx * vec2(-1.0, 1.0) * freq * TAU;
    offset *= TAU;
    vec2 pInt = floor(p);
    vec2 pFrac = fract(p);
    vec2 phaseDir = vec2(0.0);
    float weightSum = 0.0;
    for (int i = -1; i <= 2; i++) {
        for (int j = -1; j <= 2; j++) {
            vec2 gridOffset = vec2(i, j);
            vec2 gridPoint = pInt + gridOffset;
            vec2 randomOffset = hash(gridPoint) * 0.5;
            vec2 vectorFromCellPoint = pFrac - gridOffset - randomOffset;
            float sqrDist = dot(vectorFromCellPoint, vectorFromCellPoint);
            float weight = exp(-sqrDist * 2.0);
            weight = max(0.0, weight - 0.01111);
            weightSum += weight;
            float waveInput = dot(vectorFromCellPoint, sideDir) + offset;
            phaseDir += vec2(cos(waveInput), sin(waveInput)) * weight;
        }
    }
    vec2 interpolated = phaseDir / weightSum;
    float magnitude = sqrt(dot(interpolated, interpolated));
    magnitude = max(1.0 - normalization, magnitude);
    return vec4(interpolated / magnitude, sideDir);
}

float pow_inv(float t, float power) {
    return 1.0 - pow(1.0 - clamp01(t), power);
}
float ease_out(float t) {
    float v = 1.0 - clamp01(t);
    return 1.0 - v * v;
}
float smooth_start(float t, float smoothing) {
    if (t >= smoothing) return t - 0.5 * smoothing;
    return 0.5 * t * t / smoothing;
}
vec2 safe_normalize(vec2 n) {
    float l = length(n);
    return (abs(l) > 1e-10) ? (n / l) : n;
}

vec4 ErosionFilter(
    in vec2 p, vec3 heightAndSlope, float fadeTarget,
    float strength, float gullyWeight, float detail, vec4 rounding, vec4 onset, vec2 assumedSlope,
    float scale, int octaves, float lacunarity,
    float gain, float cellScale, float normalization,
    out float ridgeMap, out float debug
) {
    strength *= scale;
    fadeTarget = clamp(fadeTarget, -1.0, 1.0);
    vec3 inputHeightAndSlope = heightAndSlope;
    float freq = 1.0 / (scale * cellScale);
    float slopeLength = max(length(heightAndSlope.yz), 1e-10);
    float magnitude = 0.0;
    float roundingMult = 1.0;
    float roundingForInput = mix(rounding.y, rounding.x, clamp01(fadeTarget + 0.5)) * rounding.z;
    float combiMask = ease_out(smooth_start(slopeLength * onset.x, roundingForInput * onset.x));
    float ridgeMapCombiMask = ease_out(slopeLength * onset.z);
    float ridgeMapFadeTarget = fadeTarget;
    vec2 gullySlope = mix(heightAndSlope.yz, heightAndSlope.yz / slopeLength * assumedSlope.x, assumedSlope.y);
    for (int i = 0; i < octaves; i++) {
        vec4 phacelle = PhacelleNoise(p * freq, safe_normalize(gullySlope), cellScale, 0.25, normalization);
        phacelle.zw *= -freq;
        float sloping = abs(phacelle.y);
        gullySlope += sign(phacelle.y) * phacelle.zw * strength * gullyWeight;
        vec3 gullies = vec3(phacelle.x, phacelle.y * phacelle.zw);
        vec3 fadedGullies = mix(vec3(fadeTarget, 0.0, 0.0), gullies * gullyWeight, combiMask);
        heightAndSlope += fadedGullies * strength;
        magnitude += strength;
        fadeTarget = fadedGullies.x;
        float roundingForOctave = mix(rounding.y, rounding.x, clamp01(phacelle.x + 0.5)) * roundingMult;
        float newMask = ease_out(smooth_start(sloping * onset.y, roundingForOctave * onset.y));
        combiMask = pow_inv(combiMask, detail) * newMask;
        ridgeMapFadeTarget = mix(ridgeMapFadeTarget, gullies.x, ridgeMapCombiMask);
        float newRidgeMapMask = ease_out(sloping * onset.w);
        ridgeMapCombiMask = ridgeMapCombiMask * newRidgeMapMask;
        strength *= gain;
        freq *= lacunarity;
        roundingMult *= rounding.w;
    }
    ridgeMap = ridgeMapFadeTarget * (1.0 - ridgeMapCombiMask);
    debug = fadeTarget;
    vec3 heightAndSlopeDelta = heightAndSlope - inputHeightAndSlope;
    return vec4(heightAndSlopeDelta, magnitude);
}

vec3 FractalNoise(vec2 p, float freq, int octaves, float lacunarity, float gain) {
    vec3 n = vec3(0.0);
    float nf = freq;
    float na = 1.0;
    for (int i = 0; i < octaves; i++) {
        n += noised(p * nf) * na * vec3(1.0, nf, nf);
        na *= gain;
        nf *= lacunarity;
    }
    return n;
}
