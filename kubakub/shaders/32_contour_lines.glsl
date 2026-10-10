// about: the picture on image0 as height lines, like a map of its light and dark; the lines climb. Loops without a jump.
uniform float lines;      // 10 2 60     how many lines between dark and bright
uniform float thin;       // 0.12 0.03 0.9 how thin the lines are
uniform float speed;      // 1 0 6       levels the lines climb in one loop
uniform float smooth_;    // 4 0 16      how much the picture is smoothed first, in pixels
uniform float keep;       // 0 0 1       how much of the picture stays visible underneath
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);
vec3 pic(vec2 uv) { return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

float soft(vec2 uv) {
    vec2 e = smooth_ / iResolution.xy;
    return (lum(pic(uv)) * 2.0 + lum(pic(uv + vec2(e.x, 0.0))) + lum(pic(uv - vec2(e.x, 0.0))) + lum(pic(uv + vec2(0.0, e.y))) + lum(pic(uv - vec2(0.0, e.y)))
          + 0.5 * (lum(pic(uv + e)) + lum(pic(uv - e)) + lum(pic(uv + vec2(e.x, -e.y))) + lum(pic(uv + vec2(-e.x, e.y))))) / 8.0;
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float h = soft(uv);
    float x = h * lines - kLoop * floor(speed + 0.5);
    float w = fwidth(x) * 1.2;
    float v = smoothstep(thin + w, thin - w, abs(fract(x) - 0.5) * 2.0) * smoothstep(0.0, 0.004, w);      // no line on a flat area
    vec3 col = pic(uv) * keep + mix(LAV, ORG, smoothstep(0.2, 0.8, h)) * v;
    fragColor = vec4(col * kMask(uv), 1.0);
}
