// about: small lights wandering in the dark, each on its own path. Loops without a jump.
uniform float count;      // 40 1 120    how many lights
uniform float size;       // 0.02 0.002 0.1
uniform float wander;     // 0.12 0 0.5  how far each one roams
uniform float twinkle;    // 0.6 0 1     how much they blink
uniform float sound;      // 0 0 2       the highs of a sound make them flare
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);
const float TAU = 6.2831853;
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 asp = vec2(iResolution.x / iResolution.y, 1.0);
    vec2 p = kUV * asp;
    float a = TAU * kLoop;
    vec3 col = vec3(0.0);
    int n = int(clamp(floor(count), 1.0, 120.0));
    for (int i = 0; i < 120; i++) {
        if (i >= n) break;
        float fi = float(i);
        vec2 c = vec2(hash1(vec2(fi, 1.3)), hash1(vec2(fi, 2.9)));
        float t1 = floor(hash1(vec2(fi, 4.1)) * 3.0) + 1.0, t2 = floor(hash1(vec2(fi, 5.7)) * 3.0) + 1.0;
        c += wander * vec2(cos(a * t1 + fi), sin(a * t2 + fi * 1.7));
        float d = length(p - c * asp);
        float blink = 1.0 - twinkle * (0.5 + 0.5 * sin(a * (floor(hash1(vec2(fi, 8.8)) * 5.0) + 2.0) + fi * 3.0));
        float s = size * (1.0 + sound * kHigh);
        col += mix(LAV, ORG, hash1(vec2(fi, 6.2))) * blink * (s * s) / (d * d + s * s * 0.15) * 0.4;
    }
    fragColor = vec4(col * kMask(kUV), 1.0);
}
