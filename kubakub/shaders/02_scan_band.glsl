// about: a band of light that runs along the field (connect kubakub mask field; without one: left to right).
uniform float width;      // 0.15 0.01 1  how wide the band is
uniform float soft;       // 0.5 0 1      0 = hard edges, 1 = a soft glow
uniform float bands;      // 1 1 12       how many bands run at once
uniform float beat;       // 0 0 1        how much brighter it gets on a beat
uniform float warm;       // 0.5 0 1      0 = violet, 1 = orange

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float f = kField(uv);
    float n = max(1.0, floor(bands));
    float x = fract(f * n - kLoop);
    float d = min(x, 1.0 - x) * 2.0 / n;                  // distance to the middle of the band, in field units
    float hw = width * 0.5 / n;
    float v = 1.0 - smoothstep(hw * (1.0 - soft), hw + 1e-4, d);
    v *= 1.0 + beat * kBeat;
    vec3 col = mix(vec3(0.63, 0.53, 0.72), vec3(0.95, 0.54, 0.35), warm);
    col = mix(col, vec3(1.0), v * v * 0.6) * v * kMask(uv);
    fragColor = vec4(col, 1.0);
}
