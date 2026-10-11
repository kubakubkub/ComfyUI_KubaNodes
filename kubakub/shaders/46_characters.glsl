// about: the picture on image0 typed in characters, dense signs where it is bright, like text-mode art; the signs flicker.
// type: look
uniform float columns;    // 110 20 320  characters across the picture
uniform float colour;     // 0.6 0 1     0 = terminal green, 1 = the picture's colours
uniform float flicker;    // 6 0 40      times a character changes in one loop
uniform float contrast;   // 1.3 0.5 3
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}
vec3 pic(vec2 uv) { return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

const int GLYPH[10] = int[10](0, 4, 131200, 14336, 459200, 4357252, 22511061, 11512810, 33412991, 33554431);
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = vec2(columns, floor(columns * iResolution.y / iResolution.x * 0.75 + 0.5));
    vec2 g = kUV * res, id = floor(g), f = fract(g);
    vec3 c = pic((id + 0.5) / res);
    float v = clamp((lum(c) - 0.5) * contrast + 0.5, 0.0, 0.999);
    float jit = hash1(id + floor(fract(kLoop) * floor(flicker)) * 7.0) - 0.5;
    int k = int(clamp(floor(v * 10.0 + jit * 1.2 * step(0.5, flicker)), 0.0, 9.0));
    vec2 cell = floor(f * vec2(6.0, 7.0)) - vec2(0.0, 1.0);        // 5 x 5 dots with a gap to the right and below
    float on = 0.0;
    if (cell.x >= 0.0 && cell.x < 5.0 && cell.y >= 0.0 && cell.y < 5.0) on = float((GLYPH[k] >> (int(cell.x) + 5 * int(cell.y))) & 1);
    vec3 ink = mix(vec3(0.25, 1.0, 0.45) * (0.4 + 0.6 * v), c / max(lum(c), 0.08) * (0.35 + 0.65 * v), colour);
    fragColor = vec4(ink * on * kMask(kUV), 1.0);
}
