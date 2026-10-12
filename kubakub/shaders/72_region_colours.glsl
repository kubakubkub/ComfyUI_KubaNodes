// about: every region in a colour of its own between two colours, breathing at its own pace (connect your regions). Loops without a jump.
// type: regions
uniform float spread;     // 1 0 1       how far the regions' colours lie apart (0 = all the same)
uniform float breathe;    // 0.6 0 1     how much each region dims and comes back
uniform float pace;       // 2 0 8       breaths in one loop
uniform float edge;       // 0.25 0 1    darker towards the edge of a region
uniform float beat;       // 0 0 1       how much brighter on a beat
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float id = kRegion(uv), r = kRegionRand(uv);
    float inside = step(0.0, id);
    float wave = 0.5 + 0.5 * cos(6.2831853 * (kLoop * floor(pace + 0.5) + r));
    float light = 1.0 - breathe * wave;
    float rim = mix(1.0, smoothstep(0.0, 0.6, kRegionEdge(uv)), edge);
    vec3 col = mix(colour_a, colour_b, 0.5 + (r - 0.5) * spread) * light * rim * (1.0 + beat * kBeat);
    fragColor = vec4(col * inside, 1.0);
}
