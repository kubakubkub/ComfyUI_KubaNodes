// about: slow folding smoke: noise warped by noise. Loops without a jump.
uniform float scale;      // 3 0.5 20   how many folds across the picture
uniform float warp;       // 1.2 0 4    how strongly it folds
uniform float drift;      // 1.5 0 8    how far it moves in one loop
uniform float contrast;   // 1.4 0.5 4
uniform float sound;      // 0 0 2      the low end of the sound pushes the folds

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

float smoke(vec2 p, float t) {
    float w = warp * (1.0 + sound * kLow);
    vec2 q = vec2(fbm(vec3(p, t)), fbm(vec3(p + 5.2, t + 1.7)));
    vec2 r = vec2(fbm(vec3(p + w * q + 1.7, t * 1.3)), fbm(vec3(p + w * q + 8.3, t * 1.3 + 2.8)));
    return fbm(vec3(p + w * r, t));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y * scale;
    float t = kLoop * drift;
    float v = mix(smoke(p, t), smoke(p, t - drift), kLoop);      // the end fades into the start: a loop
    v = pow(clamp(v * 1.3, 0.0, 1.0), contrast);
    vec3 col = mix(vec3(0.05, 0.03, 0.09), vec3(0.63, 0.53, 0.72), smoothstep(0.1, 0.6, v));
    col = mix(col, vec3(0.95, 0.54, 0.35), smoothstep(0.5, 0.9, v));
    fragColor = vec4(col * kMask(kUV), 1.0);
}
