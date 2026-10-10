// about: everything that moves in the clip on image0 leaves a fading trail that drifts, turns or zooms away. Needs a clip (a moving pattern without one).
// optional: image0
uniform float fade;       // 0.93 0.5 0.995 how long a trail lasts (near 1 = long)
uniform float zoom;       // 1.01 0.9 1.1  the trail grows (above 1) or shrinks away
uniform float turn;       // 0 -3 3        degrees the trail turns per frame
uniform float drift_y;    // 0 -8 8        pixels the trail rises per frame
uniform float tint;       // 0.3 0 1       the trail shifts towards violet as it ages
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);
const float TAU = 6.2831853;
vec3 src(vec2 uv) {
    if (iChannelResolution[0].x > 1.5) return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb;
    vec2 p = (uv - 0.5) * vec2(iResolution.x / iResolution.y, 1.0);
    float a = TAU * kLoop;
    vec2 c = vec2(0.45 * cos(a), 0.25 * sin(a * 2.0));
    float ring = smoothstep(0.25, 0.2, abs(fract(length(p - c) * 5.0 - kLoop * 4.0) - 0.5));
    float bar = smoothstep(0.06, 0.04, abs(p.y - 0.3 * sin(a * 3.0 + p.x * 4.0)));
    return mix(LAV, ORG, 0.5 + 0.5 * sin(a + p.x * 5.0)) * ring * smoothstep(0.5, 0.1, length(p - c)) + vec3(1.0, 0.9, 0.8) * bar;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    vec2 asp = vec2(iResolution.x / iResolution.y, 1.0);
    vec2 p = (uv - 0.5) * asp / zoom;
    float a = radians(turn);
    p = mat2(cos(a), -sin(a), sin(a), cos(a)) * p;
    vec2 back = p / asp + 0.5 - vec2(0.0, drift_y / iResolution.y);
    vec3 old = kPrev(back).rgb * fade * step(0.0, back.x) * step(back.x, 1.0) * step(0.0, back.y) * step(back.y, 1.0);
    old = mix(old, old.grb * vec3(0.9, 0.75, 1.1) + old * 0.1, tint * 0.15);
    vec3 col = iFrame == 0 ? src(uv) : max(src(uv), old);
    fragColor = vec4(col * kMask(uv), 1.0);
}
