// about: an endless flight over a glowing grid towards a striped sun. Loops without a jump; a beat can flash the lines.
uniform float horizon;    // 0.5 0.1 0.9  where the horizon sits, 0 = bottom, 1 = top
uniform float lanes;      // 1.5 0.3 8    how wide the squares are (more = smaller)
uniform float speed;      // 4 1 24       squares that pass in one loop
uniform float sun;        // 0.22 0 0.6   size of the sun, 0 = none
uniform float beat;       // 0 0 1
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float asp = iResolution.x / iResolution.y;
    vec2 p = vec2((uv.x - 0.5) * asp, uv.y - horizon);
    vec3 col;
    if (p.y < 0.0) {
        float z = 0.25 / -p.y;
        vec2 g = vec2(p.x * z * lanes * 2.0, z * lanes + kLoop * floor(speed + 0.5));
        vec2 w = fwidth(g) * 1.5;
        vec2 l = smoothstep(vec2(0.47) - w, vec2(0.5), abs(fract(g) - 0.5));
        float line = max(l.x, l.y) * (1.0 + beat * kBeat);
        float fog = smoothstep(0.0, 0.25, -p.y);
        col = mix(vec3(0.05, 0.02, 0.1), LAV * 0.35, 1.0 - fog) + mix(ORG, LAV, fog) * line * mix(0.3, 1.0, fog);
    } else {
        col = mix(ORG * 0.9, vec3(0.08, 0.03, 0.16), smoothstep(0.0, 0.45, p.y));
        float d = length(vec2(p.x, p.y - sun * 0.45));
        float disc = smoothstep(sun, sun - 0.006, d);
        float bars = step(0.35, fract(p.y / max(sun, 1e-3) * 4.0 - kLoop * 2.0)) + smoothstep(sun * 0.35, sun * 0.6, p.y);
        col = mix(col, mix(ORG, vec3(1.0, 0.92, 0.6), smoothstep(0.0, sun, p.y)), disc * clamp(bars, 0.0, 1.0));
        col += ORG * 0.5 * smoothstep(0.08, 0.0, p.y);
    }
    fragColor = vec4(col * kMask(uv), 1.0);
}
