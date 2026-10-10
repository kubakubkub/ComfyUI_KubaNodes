// about: lines of light that follow the outline of every region inwards (connect kubakub mask field, edge distance). Loops without a jump.
uniform float rings;      // 5 1 30      lines between the edge of a region and its middle
uniform float thin;       // 0.3 0.03 0.95
uniform float speed;      // 2 -8 8      lines that pass in one loop (negative = outwards)
uniform float outline;    // 1 0 1       a steady line on the outline itself
uniform float beat;       // 0 0 1       how much brighter on a beat
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float f = kField(uv), m = kMask(uv);
    float x = f * max(1.0, floor(rings)) - kLoop * floor(speed + 0.5 * sign(speed));
    float w = fwidth(x) * 1.2;
    float v = smoothstep(thin + w, thin - w, abs(fract(x) - 0.5) * 2.0);
    float edge = clamp(fwidth(m) * 3.0, 0.0, 1.0) + smoothstep(0.02 + fwidth(f) * 2.0, 0.0, f) * step(0.5, m);
    vec3 col = mix(LAV, ORG, f) * v * (1.0 + beat * kBeat) * m + vec3(1.0, 0.95, 0.9) * clamp(edge, 0.0, 1.0) * outline;
    fragColor = vec4(col, 1.0);
}
