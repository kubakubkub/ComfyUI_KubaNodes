// about: a turning kaleidoscope of coloured shards. Loops without a jump.
// type: pattern
// arrow: centre
uniform float mirrors;    // 8 2 24      how many mirror segments
uniform float detail;     // 3 1 12      how busy the pattern is
uniform float turn;       // 1 0 4       whole turns in one loop
uniform float warm;       // 0.5 0 1
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float a = TAU * kLoop;
    float r = length(p), th = atan(p.y, p.x) + a * floor(turn + 0.5) / max(1.0, floor(mirrors));
    float seg = TAU / max(2.0, floor(mirrors));
    th = abs(mod(th, seg) - seg * 0.5);
    vec2 q = r * vec2(cos(th), sin(th)) * detail;
    float v = sin(q.x * 3.0 + a) * sin(q.y * 4.0 - a) + sin((q.x + q.y) * 2.0 + a * 2.0) * 0.5 + sin(r * detail * 5.0 - a * 2.0) * 0.5;
    float edge = smoothstep(0.0, 0.08, abs(fract(v * 1.2) - 0.5));
    float tone = 0.5 + 0.5 * sin(v * 2.2 + warm * 3.0);
    vec3 col = mix(mix(vec3(0.06, 0.04, 0.12), colour_a, smoothstep(0.0, 0.6, tone)), mix(colour_b, vec3(1.0, 0.9, 0.75), warm * 0.5), smoothstep(0.55, 1.0, tone)) * (0.25 + 0.75 * edge);
    fragColor = vec4(col * kMask(kUV), 1.0);
}
