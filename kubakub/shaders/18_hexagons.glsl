// about: a honeycomb whose cells light up one after another. Loops without a jump; a beat can flash them.
// type: pattern
// arrow: right
uniform float cells;      // 10 2 60     cells across the picture height
uniform float gap;        // 0.08 0 0.5  the dark line between cells
uniform float lit;        // 0.35 0.05 1 how many cells are lit at once
uniform float beat;       // 0 0 1
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

vec4 hexCell(vec2 p) {                             // xy = position in the cell, zw = the cell's id
    const vec2 s = vec2(1.0, 1.7320508);
    vec4 c = floor(vec4(p, p - vec2(0.5, 1.0)) / s.xyxy) + 0.5;
    vec4 h = vec4(p - c.xy * s, p - (c.zw + 0.5) * s);
    return dot(h.xy, h.xy) < dot(h.zw, h.zw) ? vec4(h.xy, c.xy) : vec4(h.zw, c.zw + 0.5);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = fragCoord / iResolution.y * cells;
    vec4 h = hexCell(p);
    vec2 q = abs(h.xy);
    float d = max(dot(q, vec2(0.5, 0.8660254)), q.x);          // 0 in the middle, 0.5 on the border
    float inside = smoothstep(0.5 - gap * 0.5, 0.5 - gap * 0.5 - 0.03, d);
    float phase = fract(hash1(h.zw) - kLoop);
    float on = smoothstep(lit, 0.0, phase) + smoothstep(1.0 - lit * 0.15, 1.0, phase);
    vec3 col = mix(colour_a, colour_b, hash1(h.zw + 4.2)) * (0.08 + on * (1.0 + beat * kBeat));
    fragColor = vec4(col * inside * kMask(kUV), 1.0);
}
