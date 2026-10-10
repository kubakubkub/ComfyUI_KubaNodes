// about: a maze of quarter circles that rearranges itself, with light running along the paths. Loops without a jump.
uniform float tiles;      // 8 2 40      tiles across the picture height
uniform float line;       // 0.12 0.02 0.4 how thick the paths are
uniform float flow;       // 2 1 8       how fast the light runs along
uniform float warm;       // 0.5 0 1
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);
const float TAU = 6.2831853;
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = fragCoord / iResolution.y * tiles;
    vec2 id = floor(p), f = fract(p);
    if (hash1(id) > 0.5) f.x = 1.0 - f.x;
    float d1 = abs(length(f) - 0.5), d2 = abs(length(f - 1.0) - 0.5);
    float d = min(d1, d2);
    vec2 c = d1 < d2 ? f : f - 1.0;
    float along = atan(c.y, c.x) / TAU * 4.0;
    float path = smoothstep(line, line - 0.03, d);
    float run = 0.5 + 0.5 * sin(TAU * (along * 2.0 + kLoop * floor(flow + 0.5)));
    vec3 col = mix(LAV, ORG, warm) * path * (0.25 + 0.75 * run) + vec3(1.0) * path * pow(run, 8.0) * 0.5;
    fragColor = vec4(col * kMask(kUV), 1.0);
}
