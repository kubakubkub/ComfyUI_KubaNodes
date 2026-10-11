// about: flames rising from the bottom edge. Loops without a jump; the low end of a sound makes them taller.
// type: pattern
uniform float height;     // 0.7 0.1 1.5  how high the flames reach
uniform float scale;      // 4 1 20       how many tongues across
uniform float rise;       // 3 0 12       how far the flames travel up in one loop
uniform float sound;      // 0 0 2        the low end of the sound adds height
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

float flame(vec2 p, float t) {
    return fbm(vec3(p.x, p.y - t, t * 0.4)) * 0.6 + fbm(vec3(p.x * 2.1, p.y * 2.1 - t * 1.7, 3.0)) * 0.4;
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    vec2 p = vec2(uv.x * iResolution.x / iResolution.y, uv.y) * scale;
    float t = kLoop * rise;
    float n = mix(flame(p, t), flame(p, t - rise), kLoop);
    float h = height * (1.0 + sound * kLow);
    float v = clamp((n * 1.6 - uv.y / h) * 1.8 + (1.0 - uv.y / h) * 0.6, 0.0, 1.0);
    v *= smoothstep(h * 1.25, 0.0, uv.y);
    vec3 col = vec3(1.5 * v, 1.5 * v * v * v, v * v * v * v * v * v);
    fragColor = vec4(col * kMask(uv), 1.0);
}
