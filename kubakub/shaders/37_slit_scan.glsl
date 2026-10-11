// about: slit scan: a slit travels across and paints what the clip on image0 shows at that moment, so moving things bend and stretch. Needs a clip (a moving pattern without one).
// type: time
// optional: image0
uniform float passes;     // 1 1 12      times the slit crosses in one loop
uniform float upright;    // 1 0 1       1 = the slit travels left to right, 0 = top to bottom
uniform float show;       // 0.6 0 1     how bright the slit itself is drawn
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
const float TAU = 6.2831853;
vec3 src(vec2 uv) {
    if (iChannelResolution[0].x > 1.5) return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb;
    vec2 p = (uv - 0.5) * vec2(iResolution.x / iResolution.y, 1.0);
    float a = TAU * kLoop;
    vec2 c = vec2(0.45 * cos(a), 0.25 * sin(a * 2.0));
    float ring = smoothstep(0.25, 0.2, abs(fract(length(p - c) * 5.0 - kLoop * 4.0) - 0.5));
    float bar = smoothstep(0.06, 0.04, abs(p.y - 0.3 * sin(a * 3.0 + p.x * 4.0)));
    return mix(colour_a, colour_b, 0.5 + 0.5 * sin(a + p.x * 5.0)) * ring * smoothstep(0.5, 0.1, length(p - c)) + vec3(1.0, 0.9, 0.8) * bar;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float n = max(1.0, floor(passes));
    float along = upright > 0.5 ? uv.x : 1.0 - uv.y;
    float pos = fract(kLoop * n);
    float stepSize = n / max(1.0, kSeconds * iFrameRate) * 1.5 + 1.5 / (upright > 0.5 ? iResolution.x : iResolution.y);
    float behind = fract(pos - along);                 // 0 just behind the slit, growing with the time since it passed
    bool fresh = iFrame == 0 || behind < stepSize;
    vec3 col = fresh ? src(uv) : kPrev(uv).rgb;
    float slit = smoothstep(stepSize * 1.5, 0.0, behind) * show;
    fragColor = vec4((col + vec3(1.0, 0.9, 0.8) * slit * 0.6 * (fresh ? 0.0 : 1.0) + vec3(slit * 0.25 * (fresh ? 1.0 : 0.0))) * kMask(uv), 1.0);
}
