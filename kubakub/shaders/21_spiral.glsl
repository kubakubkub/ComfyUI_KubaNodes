// about: a turning spiral, hypnotic. Loops without a jump.
uniform float arms;       // 3 1 16      arms of the spiral
uniform float tight;      // 6 0 30      how tightly it winds
uniform float width;      // 0.5 0.05 0.95
uniform float soft;       // 0.2 0 1
uniform float turn;       // 1 -4 4      whole turns in one loop (negative = the other way)
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float r = length(p), th = atan(p.y, p.x) / TAU;
    float n = max(1.0, floor(arms));
    float x = th * n + log(r + 0.02) * tight * 0.25 - kLoop * floor(turn + 0.5 * sign(turn));
    float d = abs(fract(x) - 0.5) * 2.0;
    float v = 1.0 - smoothstep(width * (1.0 - soft), width + 1e-4, d);
    vec3 col = mix(LAV, ORG, clamp(r * 1.4, 0.0, 1.0)) * v * smoothstep(0.0, 0.03, r);
    fragColor = vec4(col * kMask(kUV), 1.0);
}
