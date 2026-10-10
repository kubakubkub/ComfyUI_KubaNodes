// about: the picture on image0 as an architect's blueprint: white outlines on blue, a drawing grid, and a pen of light that redraws it.
uniform float thick;      // 1.2 0.5 6   outline width in pixels
uniform float gain;       // 4 0.5 20    how many outlines show
uniform float grid;       // 24 0 120    grid squares across, 0 = none
uniform float pen;        // 0.6 0 1     the line of light that runs along the field
uniform float paper;      // 0.7 0 1     0 = black, 1 = blueprint blue
vec3 pic(vec2 uv) { return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, e = thick / iResolution.xy;
    float gx = lum(pic(uv + vec2(e.x, 0.0))) - lum(pic(uv - vec2(e.x, 0.0)));
    float gy = lum(pic(uv + vec2(0.0, e.y))) - lum(pic(uv - vec2(0.0, e.y)));
    float edge = clamp(length(vec2(gx, gy)) * gain, 0.0, 1.0);
    vec2 g = fragCoord / iResolution.x * grid;
    float fine = grid > 0.5 ? max(smoothstep(0.06, 0.0, abs(fract(g.x) - 0.5) - 0.44), smoothstep(0.06, 0.0, abs(fract(g.y) - 0.5) - 0.44)) : 0.0;
    float bold = grid > 0.5 ? max(smoothstep(0.02, 0.0, abs(fract(g.x / 4.0) - 0.5) - 0.48), smoothstep(0.02, 0.0, abs(fract(g.y / 4.0) - 0.5) - 0.48)) : 0.0;
    float x = fract(kField(uv) - kLoop);
    float light = mix(1.0, 0.35 + 1.4 * smoothstep(0.25, 0.0, min(x, 1.0 - x) * 2.0), pen);
    vec3 blue = mix(vec3(0.0), vec3(0.06, 0.2, 0.48), paper);
    vec3 col = blue + vec3(0.55, 0.75, 1.0) * (fine * 0.12 + bold * 0.18) + vec3(0.9, 0.96, 1.0) * edge * light;
    fragColor = vec4(col * kMask(uv), 1.0);
}
