// about: the bright parts of the picture on image0 bleed light into their surroundings, like lamps in fog or a soft-focus lens. The loop or a sound can swell the glow.
// type: lens
uniform float level;      // 0.6 0 1       how bright a part must be to glow
uniform float radius;     // 0.05 0 0.3    how far the light bleeds, as a part of the picture's height
uniform float strength;   // 1.2 0 5
uniform vec3 glow_colour; // #ffffff   the colour of the glow
uniform float breathe;    // 0 0 1         how much the glow goes away and comes back within one loop
uniform float sound;      // 0 0 2         the low end of a sound swells it
uniform float taps;       // 56 8 128      samples per pixel: more is smoother and slower
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.y / iResolution.x, 1.0);
    int n = int(clamp(taps, 8.0, 128.0));
    vec3 sum = vec3(0.0);
    for (int i = 0; i < n; i++) {
        float f = (float(i) + 0.5) / float(n), a = float(i) * 2.39996323;
        vec3 s = texture(iChannel0, clamp(uv + vec2(cos(a), sin(a)) * sqrt(f) * radius * asp, 0.0, 1.0)).rgb;
        sum += max(s - level, 0.0) / max(1.0 - level, 1e-3) * (1.0 - 0.6 * f);
    }
    float k = strength * (1.0 - breathe * (0.5 - 0.5 * cos(TAU * kLoop))) * (1.0 + sound * kLow * 2.0);
    fragColor = vec4((texture(iChannel0, uv).rgb + sum / float(n) * glow_colour * k) * kMask(uv), 1.0);
}
