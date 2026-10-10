// about: rays of light fanning out from a point above, slowly shifting. Loops without a jump.
uniform float rays;       // 14 2 80     how many rays
uniform float cx;         // 0.5 -0.5 1.5 where the light is, 0 = left edge, 1 = right edge
uniform float cy;         // 1.15 0 2    its height, 1 = the top edge
uniform float reach;      // 1.2 0.2 3   how far the rays reach
uniform float warm;       // 0.7 0 1
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);
const float TAU = 6.2831853;
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

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 asp = vec2(iResolution.x / iResolution.y, 1.0);
    vec2 p = (kUV - vec2(cx, cy)) * asp;
    float r = length(p), th = atan(p.y, p.x);
    float a = TAU * kLoop;
    float v = 0.0;
    for (int i = 1; i <= 3; i++) {
        float fi = float(i);
        v += (0.5 + 0.5 * sin(th * floor(rays) * fi * 0.5 + a * (mod(fi, 2.0) * 2.0 - 1.0) + fi * 1.9)) / fi;
    }
    v = pow(v / 1.83, 2.5) * smoothstep(reach, 0.0, r);
    vec3 col = mix(vec3(0.75, 0.85, 1.0), vec3(1.0, 0.82, 0.55), warm) * v * 1.6 + vec3(1.0) * smoothstep(0.12, 0.0, r) * 0.6;
    fragColor = vec4(col * kMask(kUV), 1.0);
}
