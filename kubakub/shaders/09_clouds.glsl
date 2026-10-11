// about: soft clouds drifting across a sky. Loops without a jump.
// type: pattern
uniform float cover;      // 0.5 0 1     how much of the sky is cloud
uniform float scale;      // 2.5 0.5 12  how big the clouds are (more = smaller)
uniform float drift;      // 1.5 0 8     how far they travel in one loop
uniform float dusk;       // 0.3 0 1     0 = a day sky, 1 = sunset
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

float sky(vec2 p, float t) {
    return fbm(vec3(p.x + t, p.y * 1.6, t * 0.35)) * 0.65 + fbm(vec3(p * 2.3 + vec2(t * 1.4, 0.0), 4.0 + t * 0.5)) * 0.35;
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    vec2 p = vec2(uv.x * iResolution.x / iResolution.y, uv.y) * scale;
    float t = kLoop * drift;
    float v = mix(sky(p, t), sky(p, t - drift), kLoop);
    float c = smoothstep(0.62 - cover * 0.4, 0.82 - cover * 0.4, v);
    vec3 day = mix(vec3(0.62, 0.78, 0.95), vec3(0.22, 0.45, 0.82), uv.y);
    vec3 eve = mix(vec3(0.98, 0.58, 0.32), vec3(0.30, 0.22, 0.48), uv.y);
    vec3 col = mix(day, eve, dusk);
    vec3 cloud = mix(vec3(1.0), mix(vec3(0.75, 0.78, 0.85), vec3(0.95, 0.6, 0.5), dusk), smoothstep(0.5, 1.0, v));
    col = mix(col, cloud, c);
    fragColor = vec4(col * kMask(uv), 1.0);
}
