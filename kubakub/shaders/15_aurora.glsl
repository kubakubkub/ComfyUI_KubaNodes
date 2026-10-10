// about: curtains of northern light swaying across the top. Loops without a jump.
uniform float height;     // 0.6 0.1 1    how far down the curtains hang
uniform float folds;      // 3 0.5 12     how many folds across
uniform float sway;       // 1.2 0 6      how far they move in one loop
uniform float green;      // 0.6 0 1      0 = violet, 1 = green
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

float curtain(vec2 uv, float t) {
    float x = uv.x * folds + fbm(vec3(uv.x * folds * 0.6, t, 2.0)) * 2.0;
    float band = fbm(vec3(x, t * 0.7, 7.0));
    float rays = 0.6 + 0.4 * noise(vec3(x * 9.0, uv.y * 1.5, t));
    float top = 1.0 - uv.y;
    float body = smoothstep(height, 0.0, top + (band - 0.5) * 0.5) * smoothstep(0.0, 0.08, top);
    return body * rays * smoothstep(0.15, 0.6, band) * 1.8;
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float t = kLoop * sway;
    float v = mix(curtain(uv, t), curtain(uv, t - sway), kLoop);
    vec3 a = mix(vec3(0.63, 0.35, 0.95), vec3(0.2, 1.0, 0.55), green);
    vec3 b = mix(vec3(0.95, 0.54, 0.35), vec3(0.5, 0.4, 1.0), green);
    vec3 col = vec3(0.01, 0.015, 0.04) + mix(b, a, smoothstep(0.3, 1.0, uv.y)) * v * 1.6;
    fragColor = vec4(col * kMask(uv), 1.0);
}
