// about: the picture on image0 as on an old game console: big pixels, a handful of colours, dither patterns.
uniform float pixels;     // 160 16 640  pixels across the picture
uniform float palette;    // 0 0 4       0 = four greens, 1 = cyan magenta white, 2 = amber, 3 = eight colours, 4 = 64 colours
uniform float dither;     // 0.8 0 1     how much the dot pattern is used to fake more shades
uniform float shimmer;    // 0 0 16      times the dither pattern jumps in one loop
uniform float contrast;   // 1.2 0.5 3
vec3 pic(vec2 uv) { return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

float bayer(vec2 p) {                              // the classic 4 x 4 dither pattern, 0 .. 1
    ivec2 q = ivec2(mod(p, 4.0));
    int m[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
    return (float(m[q.x + q.y * 4]) + 0.5) / 16.0;
}
vec3 shade4(float v, vec3 a, vec3 b, vec3 c, vec3 d) { return v < 0.25 ? a : v < 0.5 ? b : v < 0.75 ? c : d; }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = vec2(pixels, floor(pixels * iResolution.y / iResolution.x + 0.5));
    vec2 cell = floor(kUV * res);
    vec3 c = pic((cell + 0.5) / res);
    c = clamp((c - 0.5) * contrast + 0.5, 0.0, 1.0);
    float jump = floor(fract(kLoop) * floor(shimmer));
    float t = (bayer(cell + vec2(jump, jump * 3.0)) - 0.5) * dither;
    float v = clamp(lum(c) + t * 0.33, 0.0, 0.999);
    int pal = int(floor(palette + 0.5));
    vec3 col;
    if (pal == 0) col = shade4(v, vec3(0.06, 0.22, 0.06), vec3(0.19, 0.38, 0.19), vec3(0.55, 0.67, 0.06), vec3(0.61, 0.74, 0.06));
    else if (pal == 1) col = shade4(v, vec3(0.0), vec3(0.33, 1.0, 1.0), vec3(1.0, 0.33, 1.0), vec3(1.0));
    else if (pal == 2) col = shade4(v, vec3(0.05, 0.02, 0.0), vec3(0.45, 0.2, 0.02), vec3(0.95, 0.54, 0.1), vec3(1.0, 0.85, 0.5));
    else if (pal == 3) col = step(0.5, c + t);
    else col = floor(clamp(c + t * 0.33, 0.0, 0.999) * 4.0) / 3.0;
    fragColor = vec4(col * kMask(kUV), 1.0);
}
