// about: fine lines flowing like wood grain or a topographic map. Loops without a jump.
// type: pattern
uniform float lines;      // 18 2 120    how many lines
uniform float scale;      // 1.6 0.2 8   how large the bends are (more = smaller)
uniform float thin;       // 0.25 0.03 0.9 how thin the lines are
uniform float drift;      // 0.8 0 6     how far it moves in one loop
uniform float warm;       // 0.4 0 1
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
float hash(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}
float noise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x), mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x), mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * noise(p); p = p * 2.02 + 17.1; a *= 0.5; }
    return v;
}

float height(vec2 p, float t) { return fbm(vec3(p + fbm(vec3(p * 0.7, t)) * 1.5, t * 0.6)); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y * scale;
    float t = kLoop * drift;
    float h = mix(height(p, t), height(p, t - drift), kLoop);
    float x = h * lines;
    float w = fwidth(x) * 1.2;
    float d = abs(fract(x) - 0.5) * 2.0;
    float v = smoothstep(thin + w, thin - w, d);
    vec3 col = mix(colour_a, colour_b, clamp(warm + (h - 0.5) * 1.5, 0.0, 1.0)) * v;
    fragColor = vec4(col * kMask(kUV), 1.0);
}
