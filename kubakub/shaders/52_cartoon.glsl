// about: the picture on image0 as a cartoon: a few flat tones, black ink outlines and hatching in the dark parts that drifts. High contrast, made for architecture.
// type: look
uniform float tones;      // 5 2 10      how many flat tones between dark and bright
uniform float contrast;   // 1.3 0.5 4   pushes dark and bright apart before the tones are cut
uniform float ink;        // 1 0 1       how black the outlines are
uniform float thick;      // 1.5 0.5 6   outline width in pixels
uniform float detail;     // 2 0.5 20    how many outlines show
uniform float hatch;      // 0.5 0 1     hatching in the dark tones
uniform float lines;      // 120 20 400  hatching lines across the picture
uniform float drift;      // 2 0 12      lines the hatching moves in one loop
uniform float two_colours; // 0 0 1      0 = the picture's own colours, 1 = dark, lavender, orange
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
vec3 pic(vec2 uv) { return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float lumAt(vec2 uv) { return lum(pic(uv)); }
vec3 ramp(float t) {
    return t < 0.5 ? mix(vec3(0.05, 0.04, 0.07), colour_a, t * 2.0) : mix(colour_a, mix(colour_b, vec3(1.0), 0.35), t * 2.0 - 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, e = thick / iResolution.xy;
    vec3 c = pic(uv);
    float l = lum(c);
    float n = max(2.0, floor(tones + 0.5));
    float q = floor(clamp((l - 0.5) * contrast + 0.5, 0.0, 1.0) * (n - 1.0) + 0.5) / (n - 1.0);
    vec3 hue = c / max(max(c.r, max(c.g, c.b)), 0.02);                       // the colour without its brightness
    vec3 flat_ = mix(hue * mix(0.1, 1.0, q), ramp(q), two_colours);          // the tone, in the picture's own colour
    float gx = lumAt(uv + vec2(e.x, 0.0)) - lumAt(uv - vec2(e.x, 0.0)) + 0.5 * (lumAt(uv + e) - lumAt(uv + vec2(-e.x, e.y)) + lumAt(uv + vec2(e.x, -e.y)) - lumAt(uv - e));
    float gy = lumAt(uv + vec2(0.0, e.y)) - lumAt(uv - vec2(0.0, e.y)) + 0.5 * (lumAt(uv + e) - lumAt(uv + vec2(e.x, -e.y)) + lumAt(uv + vec2(-e.x, e.y)) - lumAt(uv - e));
    float edge = smoothstep(0.35, 0.75, length(vec2(gx, gy)) * detail);
    float x = (uv.x * iResolution.x / iResolution.y + uv.y) * lines * 0.5 + kLoop * floor(drift + 0.5);
    float w = fwidth(x) * 1.2;
    float stripe = smoothstep(0.5 + w, 0.5 - w, abs(fract(x) - 0.5) * 2.0);
    float dark = 1.0 - smoothstep(0.2, 0.45, q);                             // only the darkest tones are hatched
    vec3 col = flat_ * (1.0 - stripe * dark * hatch * 0.7);
    col = mix(col, vec3(0.0), edge * ink);
    fragColor = vec4(clamp(col, 0.0, 1.0) * kMask(uv), 1.0);
}
