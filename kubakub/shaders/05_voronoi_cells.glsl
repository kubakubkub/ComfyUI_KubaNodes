// about: cells with glowing borders that slowly trade places. Loops without a jump.
// type: pattern
uniform float cells;      // 8 2 60     cells across the picture height
uniform float border;     // 0.06 0.005 0.5  how thick the glowing borders are
uniform float move;       // 0.35 0 0.5 how far the cell centres wander
uniform float fill;       // 0.25 0 1   how bright the inside of a cell is
uniform float sound;      // 0 0 2      a kick lights the borders

vec2 hash2(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + 33.33);
    return fract((q.xx + q.yz) * q.zy);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = fragCoord / iResolution.y * cells;
    vec2 ip = floor(p), fp = fract(p);
    float a = 6.2831853 * kLoop;
    float d1 = 8.0, d2 = 8.0;
    vec2 id = vec2(0.0);
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        vec2 g = vec2(float(x), float(y));
        vec2 h = hash2(ip + g);
        vec2 o = 0.5 + move * vec2(cos(a + h.x * 6.2831853), sin(a + h.y * 6.2831853));
        float d = length(g + o - fp);
        if (d < d1) { d2 = d1; d1 = d; id = ip + g; } else if (d < d2) { d2 = d; }
    }
    float edge = 1.0 - smoothstep(0.0, border, d2 - d1);
    float tone = hash2(id).x;
    vec3 inside = mix(vec3(0.63, 0.53, 0.72), vec3(0.95, 0.54, 0.35), tone) * fill * (0.4 + 0.6 * tone);
    vec3 col = inside + vec3(1.0, 0.9, 0.8) * edge * (1.0 + sound * kLow);
    fragColor = vec4(col * kMask(kUV), 1.0);
}
