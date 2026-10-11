// about: the picture on image0 as a glowing wire mesh, its lines bent by depth (a depth pass on image1, else by brightness). A wave of light runs through.
// type: relief
// optional: image1
uniform float rows;       // 40 4 200    lines from top to bottom
uniform float columns;    // 0 0 200     lines from left to right, 0 = none
uniform float relief;     // 0.08 0 0.4  how far depth pushes the lines up
uniform float thin;       // 0.12 0.02 0.8
uniform float wave;       // 0.7 0 1     a wave of brightness through the depth
uniform float keep;       // 0 0 1       how much of the picture stays visible underneath
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
vec3 pic(vec2 uv) { return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float depthAt(vec2 uv) {                          // a depth pass on image1 when there is one, else how bright the picture is
    return iChannelResolution[1].x > 1.5 ? texture(iChannel1, clamp(uv, 0.0, 1.0)).r : lum(pic(uv));
}

float lineOf(float x, float t) { float w = fwidth(x) * 1.2; return smoothstep(t + w, t - w, abs(fract(x) - 0.5) * 2.0); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float v = 0.0, hSeen = 0.0;
    for (int i = 0; i < 6; i++) {                      // a line belongs to the place it was pushed up from: look a few steps down
        float back = float(i) / 5.0 * relief;
        vec2 q = vec2(uv.x, uv.y - back);
        float h = depthAt(q);
        float y = (q.y + h * relief) * rows;
        float hit = lineOf(y, thin) * smoothstep(0.03, 0.0, abs((q.y + h * relief) - uv.y) - 0.5 / rows);
        if (hit > v) { v = hit; hSeen = h; }
    }
    float h0 = depthAt(uv);
    if (columns > 0.5) { float c = lineOf((uv.x + (h0 - 0.5) * relief * 0.3) * floor(columns), thin * 0.6); if (c > v) { v = c; hSeen = h0; } }
    float x = fract(hSeen - kLoop);
    float glow = mix(1.0, 0.25 + 1.5 * smoothstep(0.35, 0.0, min(x, 1.0 - x) * 2.0), wave);
    vec3 col = pic(uv) * keep + mix(colour_a, colour_b, hSeen) * v * glow;
    fragColor = vec4(col * kMask(uv), 1.0);
}
