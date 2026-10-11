// about: the outlines of the picture on image0 drawn in glowing light; a pulse runs through them.
// type: look
uniform float thick;      // 1.5 0.5 8   how thick the outlines are, in pixels
uniform float gain;       // 3 0.5 20    how strong
uniform float keep;       // 0.15 0 1    how much of the picture stays visible underneath
uniform float pulse;      // 0.6 0 1     a wave of brightness running along the field
uniform float sound;      // 0 0 3       the level of the sound adds glow
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour

float lum(vec2 uv) { return dot(texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb, vec3(0.299, 0.587, 0.114)); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, e = thick / iResolution.xy;
    float gx = lum(uv + vec2(e.x, 0.0)) - lum(uv - vec2(e.x, 0.0)) + 0.5 * (lum(uv + e) - lum(uv + vec2(-e.x, e.y)) + lum(uv + vec2(e.x, -e.y)) - lum(uv - e));
    float gy = lum(uv + vec2(0.0, e.y)) - lum(uv - vec2(0.0, e.y)) + 0.5 * (lum(uv + e) - lum(uv + vec2(e.x, -e.y)) + lum(uv + vec2(-e.x, e.y)) - lum(uv - e));
    float edge = clamp(length(vec2(gx, gy)) * gain, 0.0, 1.0);
    float x = fract(kField(uv) - kLoop);
    float wave = mix(1.0, smoothstep(0.5, 0.0, min(x, 1.0 - x) * 2.0) * 1.6 + 0.15, pulse);
    vec3 col = texture(iChannel0, uv).rgb * keep + mix(colour_a, colour_b, kField(uv)) * edge * wave * (1.5 + sound * kLevel);
    fragColor = vec4(col * kMask(uv), 1.0);
}
