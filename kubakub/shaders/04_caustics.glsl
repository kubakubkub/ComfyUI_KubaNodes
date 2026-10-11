// about: light on the floor of a pool: bright moving ridges. Loops without a jump.
// type: pattern
// arrow: right
uniform float scale;      // 5 1 30     how small the ripples are
uniform float sharp;      // 6 1 20     how thin the bright lines are
uniform float speed;      // 1 1 6      whole turns of the water in one loop
uniform float tint;       // 0.5 0 1    0 = cold blue, 1 = warm gold
uniform float sound;      // 0 0 2      the level of the sound makes it brighter

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y * scale;
    float a = 6.2831853 * kLoop * max(1.0, floor(speed));
    vec2 q = p;
    float sum = 0.0;
    for (int i = 1; i <= 5; i++) {
        float fi = float(i);
        q = p + vec2(cos(a + q.y * 1.3 + fi * 1.7), sin(a + q.x * 1.1 - fi * 2.3)) * 0.6;
        float w = sin(q.x * fi * 0.9 + a) + sin(q.y * fi * 0.8 - a);
        sum += 1.0 / max(0.05, abs(w) * fi);
        p = mat2(0.8, -0.6, 0.6, 0.8) * p * 1.15;
    }
    float v = pow(clamp(sum / 22.0, 0.0, 1.0), sharp * 0.25) * (1.0 + sound * kLevel);
    vec3 cold = vec3(0.25, 0.65, 1.0), gold = vec3(1.0, 0.75, 0.35);
    vec3 col = mix(cold, gold, tint) * v + vec3(1.0) * pow(v, 3.0) * 0.5;
    fragColor = vec4(col * kMask(kUV), 1.0);
}
