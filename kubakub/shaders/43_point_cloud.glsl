// about: the picture on image0 as a cloud of points that sways in depth (a depth pass on image1, else by brightness), like a 3D scan.
// type: relief
// optional: image1
uniform float points;     // 90 10 400   points across the picture
uniform float size;       // 0.95 0.1 1.5 how big a point is
uniform float sway;       // 0.04 0 0.3  how far near and far points move apart as the view swings
uniform float depth_size; // 0.6 0 1     near points are bigger
uniform float glow;       // 0.8 0 2     near points are brighter
const float TAU = 6.2831853;
vec3 pic(vec2 uv) { return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float depthAt(vec2 uv) {                          // a depth pass on image1 when there is one, else how bright the picture is
    return iChannelResolution[1].x > 1.5 ? texture(iChannel1, clamp(uv, 0.0, 1.0)).r : lum(pic(uv));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.x / iResolution.y, 1.0);
    vec2 view = sway * vec2(sin(TAU * kLoop), 0.4 * sin(TAU * kLoop * 2.0));
    vec2 g = uv * asp * points, base = floor(g);
    vec3 col = vec3(0.0);
    float top = -1.0;
    for (int j = -2; j <= 2; j++)
    for (int i = -2; i <= 2; i++) {
        vec2 id = base + vec2(float(i), float(j));
        vec2 c = (id + 0.5) / points / asp;
        if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0) continue;
        float d = depthAt(c);
        vec2 at = c + (d - 0.5) * view;
        float r = size * mix(1.0, 0.35 + 1.1 * d, depth_size) * 0.5 / points;
        float dist = length((uv - at) * asp);
        float dot_ = smoothstep(r, r * 0.6, dist);
        vec3 pc = pic(c) * (1.0 + glow * d);
        if (dot_ > 0.0 && d > top) { top = d; col = pc * dot_; }
    }
    fragColor = vec4(col * kMask(uv), 1.0);
}
