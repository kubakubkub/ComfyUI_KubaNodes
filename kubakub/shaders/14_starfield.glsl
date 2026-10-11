// about: stars flying towards you. Loops without a jump.
// type: space
uniform float stars;      // 1 0.2 4     how many stars
uniform float speed;      // 1 1 6       how many layers pass in one loop
uniform float streak;     // 0.4 0 1     how long their tails are
uniform float colour;     // 0.5 0 1     0 = white, 1 = violet and orange stars
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 col = vec3(0.0);
    const float LAYERS = 8.0;
    for (float i = 0.0; i < LAYERS; i++) {
        float z = fract(i / LAYERS + kLoop * floor(speed + 0.5) / 1.0);
        float size = mix(26.0, 1.5, z) / max(0.2, stars);
        float fade = smoothstep(0.0, 0.25, z) * smoothstep(1.0, 0.85, z);
        vec2 q = p * size + i * 31.7;
        vec2 id = floor(q), f = fract(q) - 0.5;
        float h = hash1(id);
        vec2 off = (vec2(hash1(id + 3.1), hash1(id + 7.9)) - 0.5) * 0.7;
        vec2 d = f - off;
        vec2 dir = normalize(p + 1e-5);
        float along = dot(d, dir), across = dot(d, vec2(-dir.y, dir.x));
        float len = 1.0 + streak * 14.0 * z;
        float s = smoothstep(0.06, 0.0, length(vec2(along / len, across))) * step(0.72, h);
        vec3 tint = mix(vec3(1.0), mix(colour_a, colour_b, hash1(id + 11.3)) * 1.5, colour);
        col += tint * s * fade * (0.4 + 0.6 * z);
    }
    fragColor = vec4(col * kMask(kUV), 1.0);
}
