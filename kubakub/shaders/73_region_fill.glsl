// about: every region fills up with light from its own bottom edge, one after another, and empties again (connect your regions). Loops without a jump.
// type: regions
uniform float stagger;    // 0.6 0 1     how far apart the regions start (0 = all at once)
uniform float soft;       // 0.08 0 0.5  how soft the rising edge is
uniform float wave;       // 0.03 0 0.2  how much the rising edge waves
uniform float sideways;   // 0 0 1       1 = from the left edge instead of from the bottom
uniform float beat;       // 0 0 1       how much brighter on a beat
uniform vec3 colour_a;    // #a187b7   the colour at the bottom
uniform vec3 colour_b;    // #f18a58   the colour at the top

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float id = kRegion(uv), r = kRegionRand(uv);
    vec2 q = kRegionUV(uv);
    float along = mix(q.y, q.x, step(0.5, sideways)), across = mix(q.x, q.y, step(0.5, sideways));
    float t = fract(kLoop - r * stagger);
    float level = 1.0 - abs(2.0 * t - 1.0);           // up, then down again
    level = level * (1.0 + 2.0 * soft + wave) - soft - wave;
    float surface = level + wave * sin(6.2831853 * (across * 2.0 + kLoop * 2.0 + r));
    float v = smoothstep(surface + soft, surface - soft, along);
    vec3 col = mix(colour_a, colour_b, along) * v * (1.0 + beat * kBeat);
    fragColor = vec4(col * step(0.0, id), 1.0);
}
