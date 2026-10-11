// about: rings that run out from a point (or in); a drawn arrow's start is the point. Loops without a jump; a beat can flash them.
// type: pattern
uniform float rings;      // 6 1 60     rings between the point and the edge
uniform float width;      // 0.35 0.02 1 how much of each ring is lit
uniform float soft;       // 0.3 0 1
uniform float cx;         // 0.5 0 1    the point, 0 = left edge, 1 = right edge
uniform float cy;         // 0.5 0 1    0 = bottom, 1 = top
uniform float inward;     // 0 0 1      1 = the rings run towards the point
uniform float beat;       // 0 0 1      how much brighter on a beat

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - (kVector > 0.5 ? kFrom : vec2(cx, cy)) * iResolution.xy) / iResolution.y;
    float n = max(1.0, floor(rings));
    float dir = inward > 0.5 ? 1.0 : -1.0;
    float x = fract(length(p) * n + dir * kLoop);
    float d = abs(x - 0.5) * 2.0;                         // 0 in the middle of a ring, 1 between two
    float v = 1.0 - smoothstep(width * (1.0 - soft), width + 1e-4, d);
    v *= 1.0 + beat * kBeat;
    vec3 col = mix(vec3(0.63, 0.53, 0.72), vec3(0.95, 0.54, 0.35), clamp(length(p), 0.0, 1.0)) * v;
    fragColor = vec4(col * kMask(kUV), 1.0);
}
