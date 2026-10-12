// about: stripes that fit every region's own box, each region at its own angle and pace (connect your regions). Loops without a jump.
// type: regions
uniform float lines;      // 6 1 40      stripes across a region
uniform float thin;       // 0.5 0.05 0.95
uniform float turn;       // 1 0 1       how different the regions' angles are (0 = all upright)
uniform float speed;      // 2 -8 8      stripes that pass in one loop
uniform float edge;       // 0.3 0 1     a line of light on the outline of every region
uniform float beat;       // 0 0 1       how much brighter on a beat
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float id = kRegion(uv), r = kRegionRand(uv);
    vec2 q = kRegionUV(uv) - 0.5;
    float a = 3.14159265 * floor(r * 4.0) / 4.0 * turn;        // one of four angles
    float x = (q.x * cos(a) + q.y * sin(a)) * max(1.0, floor(lines));
    float way = fract(r * 7.0) < 0.5 ? 1.0 : -1.0;
    float w = fwidth(x) * 1.2;                        // measured before the stripes move: the same at every moment of the loop
    x -= kLoop * floor(speed + 0.5 * sign(speed)) * way;
    float v = smoothstep(thin + w, thin - w, abs(fract(x) - 0.5) * 2.0);
    float e = kRegionEdge(uv);
    float rim = smoothstep(0.06 + fwidth(e) * 2.0, 0.0, e) * edge;
    vec3 col = mix(colour_a, colour_b, r) * v * (1.0 + beat * kBeat) + vec3(1.0, 0.95, 0.9) * rim;
    fragColor = vec4(col * step(0.0, id), 1.0);
}
