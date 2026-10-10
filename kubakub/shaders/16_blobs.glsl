// about: soft blobs of light that melt into each other, like a lava lamp. Loops without a jump.
uniform float blobs;      // 6 1 12      how many blobs
uniform float size;       // 0.16 0.03 0.5
uniform float edge;       // 0.15 0.01 1  0 = hard blobs, 1 = a soft glow
uniform float sound;      // 0 0 2        a kick makes them swell
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
    float a = TAU * kLoop, sum = 0.0, mixc = 0.0;
    int n = int(clamp(floor(blobs), 1.0, 12.0));
    float s = size * (1.0 + sound * kLow * 0.6);
    for (int i = 0; i < 12; i++) {
        if (i >= n) break;
        float fi = float(i);
        vec2 c = vec2(hash1(vec2(fi, 2.3)), hash1(vec2(fi, 5.1)));
        vec2 r = vec2(hash1(vec2(fi, 8.7)), hash1(vec2(fi, 4.4))) * 0.3 + 0.05;
        float turns = floor(hash1(vec2(fi, 6.6)) * 2.0) + 1.0;
        c += r * vec2(cos(a * turns + fi * 2.1), sin(a * turns + fi * 1.3));
        float d = length(p - c * asp);
        float w = s * s / (d * d + 1e-4);
        sum += w;
        mixc += w * hash1(vec2(fi, 3.9));
    }
    float v = smoothstep(1.0 - edge, 1.0 + edge, sum);
    vec3 col = mix(LAV, ORG, clamp(mixc / max(sum, 1e-4), 0.0, 1.0)) * v;
    col += vec3(1.0, 0.95, 0.9) * smoothstep(2.5, 6.0, sum) * 0.4;
    fragColor = vec4(col * kMask(kUV), 1.0);
}
