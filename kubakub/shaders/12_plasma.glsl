// about: the classic plasma: soft colour waves folding through each other. Loops without a jump.
// type: pattern
// arrow: right
uniform float scale;      // 4 0.5 20    how many waves across
uniform float speed;      // 1 1 6       whole turns in one loop
uniform float warm;       // 0.5 0 1     0 = violet and blue, 1 = orange and gold
uniform float sound;      // 0 0 2       the level of the sound brightens it
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y * scale;
    float a = TAU * kLoop * floor(speed + 0.5);
    float v = sin(p.x * 1.3 + a) + sin(p.y * 1.7 - a) + sin((p.x + p.y) * 0.9 + a)
            + sin(length(p + vec2(sin(a), cos(a)) * 1.5) * 2.0 - a);
    v = 0.5 + 0.125 * v;
    vec3 cold = 0.5 + 0.5 * cos(TAU * (v + vec3(0.55, 0.65, 0.85)));
    vec3 hot = 0.5 + 0.5 * cos(TAU * (v * 0.7 + vec3(0.0, 0.12, 0.3)));
    vec3 col = mix(cold * mix(vec3(1.0), colour_a * 1.4, 0.4), hot * mix(vec3(1.0), colour_b * 1.3, 0.4), warm);
    col *= 1.0 + sound * kLevel;
    fragColor = vec4(col * kMask(kUV), 1.0);
}
