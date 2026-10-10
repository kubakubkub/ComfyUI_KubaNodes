// about: bars that jump with the music: lows on the left, highs on the right. Needs a sound (a built-in beat without one).
uniform float bars;       // 24 3 120    how many bars
uniform float gap;        // 0.2 0 0.8   space between bars
uniform float mirror;     // 1 0 1       1 = bars grow from the middle line, 0 = from the bottom
uniform float glow;       // 0.4 0 1
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float n = floor(bars), i = floor(uv.x * n), x = (i + 0.5) / n;
    float low = smoothstep(0.55, 0.0, x), high = smoothstep(0.45, 1.0, x), mid = 1.0 - low - high;
    float level = kLow * low + kMid * max(mid, 0.0) + kHigh * high;
    level *= 0.55 + 0.45 * hash1(vec2(i, floor(kBeats)));
    level = max(level, 0.02 + 0.05 * kBeat);
    float y = mirror > 0.5 ? abs(uv.y - 0.5) * 2.0 : uv.y;
    float inbar = step(gap * 0.5, abs(fract(uv.x * n) - 0.5) * -1.0 + 0.5);
    float v = smoothstep(level, level - 0.02, y) * inbar;
    float halo = glow * smoothstep(level + 0.25, level, y) * 0.35 * inbar;
    vec3 col = mix(LAV, ORG, x) * (v + halo) + vec3(1.0) * v * smoothstep(level - 0.06, level, y) * 0.5;
    fragColor = vec4(col * kMask(uv), 1.0);
}
