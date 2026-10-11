// about: streaks of rain, or light falling down the wall, or along a drawn arrow. Loops without a jump.
// type: pattern
uniform float columns;    // 60 4 400    streaks across the picture
uniform float length_;    // 0.25 0.02 1 how long a streak is
uniform float speed;      // 2 1 8       how fast the fastest streaks fall
uniform float slant;      // 0 -1 1      wind
uniform float warm;       // 0.2 0 1     0 = cold blue, 1 = orange
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    if (kVector > 0.5) {                                      // the streaks fall the way the arrow points
        vec2 d = kDir, q = (kUV - 0.5) * vec2(iResolution.x / iResolution.y, 1.0);
        uv = vec2(dot(q, vec2(-d.y, d.x)), -dot(q, d)) + 0.5;
    }
    uv.x += uv.y * slant * 0.3;
    float c = floor(uv.x * columns);
    float h = hash1(vec2(c, 1.0));
    float v = floor(1.0 + h * floor(speed));                  // whole falls per loop: every streak loops
    float y = fract(uv.y + kLoop * v + hash1(vec2(c, 7.0)));
    float streak = smoothstep(length_, 0.0, y) * smoothstep(0.0, 0.02, y);
    float thin = smoothstep(0.5, 0.2, abs(fract(uv.x * columns) - 0.5)) * step(0.35, h);
    vec3 col = mix(vec3(0.55, 0.75, 1.0), colour_b, warm) * streak * thin * (0.4 + 0.6 * h);
    fragColor = vec4(col * kMask(kUV), 1.0);
}
