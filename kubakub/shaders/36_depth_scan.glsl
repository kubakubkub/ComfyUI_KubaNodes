// about: a sheet of light that travels through the depth of the picture on image0 (a depth pass on image1, else its brightness). Loops without a jump.
// type: relief
// optional: image1
uniform float sheets;     // 1 1 12      sheets travelling at once
uniform float width;      // 0.12 0.01 1 how thick a sheet is
uniform float passes;     // 1 -6 6      times it travels through in one loop (negative = towards you)
uniform float keep;       // 0.25 0 1    how much of the picture stays visible
uniform float warm;       // 0.5 0 1
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
vec3 pic(vec2 uv) { return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float depthAt(vec2 uv) {                          // a depth pass on image1 when there is one, else how bright the picture is
    return iChannelResolution[1].x > 1.5 ? texture(iChannel1, clamp(uv, 0.0, 1.0)).r : lum(pic(uv));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float d = depthAt(uv);
    float x = fract(d * max(1.0, floor(sheets)) - kLoop * floor(passes + 0.5 * sign(passes)));
    float near = min(x, 1.0 - x) * 2.0;
    float body = smoothstep(width, 0.0, near);
    float rim = smoothstep(width * 0.25 + fwidth(x) * 2.0, 0.0, near);
    vec3 c = pic(uv);
    vec3 col = c * keep + (c * 1.5 + mix(colour_a, colour_b, warm) * 0.5) * body + vec3(1.0) * rim * 0.7;
    fragColor = vec4(col * kMask(uv), 1.0);
}
