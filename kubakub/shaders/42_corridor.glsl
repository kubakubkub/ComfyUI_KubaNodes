// about: an endless walk down a corridor of lit panels. Loops without a jump; the low end of a sound brightens the lights.
// type: space
uniform float panels;     // 2 1 12      pairs of panels you pass in one loop
uniform float width;      // 1.3 0.5 3   how wide the corridor is against its height
uniform float lights;     // 0.6 0 1     strips of light on the ceiling and floor
uniform float fog;        // 0.7 0 1     how dark the far end is
uniform float sound;      // 0 0 2
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec2 a = vec2(abs(p.x) / width, abs(p.y));
    float d = max(a.x, a.y), z = min(0.5 / max(d, 1e-4), 40.0);      // the far end is one flat wall: no shimmer in the last pixels
    bool side = a.x > a.y;
    float u = side ? p.y * z : p.x * z / width;                 // across the wall, -0.5 .. 0.5
    float v = z * 0.5 + kLoop * 2.0 * floor(panels + 0.5);        // an even count: the two panel tones meet again
    float seam = max(smoothstep(0.04, 0.0, abs(fract(v) - 0.5) - 0.46), side ? smoothstep(0.03, 0.0, abs(abs(u) - 0.25) - 0.0) : 0.0);
    float tone = smoothstep(0.47, 0.53, abs(fract(v * 0.5 + 0.25) - 0.5) * 2.0);
    vec3 col = mix(colour_a * 0.22, colour_b * 0.2, side ? tone : 1.0 - tone) + vec3(0.9, 0.8, 0.75) * seam * 0.35;
    if (!side) {
        float strip = smoothstep(0.1, 0.06, abs(u)) * smoothstep(0.38, 0.3, abs(fract(v) - 0.5));
        col += mix(colour_b, vec3(1.0, 0.95, 0.85), 0.5) * strip * lights * 2.0 * (1.0 + sound * kLow);
    }
    col *= mix(1.0, smoothstep(0.0, 0.5, d), fog);
    fragColor = vec4(col * kMask(kUV), 1.0);
}
