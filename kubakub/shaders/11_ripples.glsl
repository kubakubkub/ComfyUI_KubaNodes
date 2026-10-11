// about: rings of water spreading from a few drops and crossing each other; a drawn arrow's start is the first drop. Loops without a jump.
// type: pattern
uniform float drops;      // 4 1 8       how many drops
uniform float waves;      // 14 2 60     rings between a drop and the edge
uniform float speed;      // 2 1 8       whole rings that leave a drop in one loop
uniform float contrast;   // 1.5 0.5 5
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
const float TAU = 6.2831853;
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 asp = vec2(iResolution.x / iResolution.y, 1.0);
    vec2 p = kUV * asp;
    float sum = 0.0;
    int n = int(clamp(floor(drops), 1.0, 8.0));
    for (int i = 0; i < 8; i++) {
        if (i >= n) break;
        vec2 c = (i == 0 && kVector > 0.5 ? kFrom : vec2(hash1(vec2(float(i), 1.7)), hash1(vec2(float(i), 9.2)))) * asp;
        float d = length(p - c);
        sum += sin(d * waves * TAU / asp.x - TAU * kLoop * floor(speed + 0.5) + float(i)) / (1.0 + d * 3.0);
    }
    float v = 0.5 + 0.5 * sum / sqrt(float(n));
    v = pow(clamp(v, 0.0, 1.0), contrast);
    vec3 col = mix(vec3(0.03, 0.05, 0.12), mix(colour_a, vec3(0.8, 0.95, 1.0), v), v);
    fragColor = vec4(col * kMask(kUV), 1.0);
}
