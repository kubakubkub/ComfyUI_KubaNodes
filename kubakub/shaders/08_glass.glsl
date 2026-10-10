// about: the picture on image0 seen through rippled glass. A normal pass on image1 is the glass; without one it is made of ripples.
// optional: image1
uniform float bend;      // 0.04 0 0.3   how strongly the glass bends the picture
uniform float ripples;    // 6 1 40       size of the built-in ripples (no normal pass connected)
uniform float split;      // 0.3 0 1      colour fringes on the edges
uniform float shine;      // 0.4 0 2      highlights on the glass
uniform float flow;       // 1 0 4        whole turns of the ripples in one loop (built-in ripples)

float height(vec2 p, float a) {
    float h = 0.0;
    for (int i = 1; i <= 4; i++) {
        float fi = float(i);
        vec2 d = vec2(cos(fi * 2.1), sin(fi * 2.1));
        h += sin(dot(p, d) * fi + a * (mod(fi, 2.0) * 2.0 - 1.0) + fi * 1.3) / fi;
    }
    return h;
}

vec3 glassNormal(vec2 uv) {
    if (iChannelResolution[1].x > 1.5) {                  // a normal pass is connected
        vec3 n = texture(iChannel1, uv).xyz * 2.0 - 1.0;
        return normalize(n + vec3(0.0, 0.0, 1e-4));
    }
    vec2 p = (uv - 0.5) * vec2(iResolution.x / iResolution.y, 1.0) * ripples;
    float a = 6.2831853 * kLoop * floor(flow + 0.5);
    float e = 0.01;
    float hx = height(p + vec2(e, 0.0), a) - height(p - vec2(e, 0.0), a);
    float hy = height(p + vec2(0.0, e), a) - height(p - vec2(0.0, e), a);
    return normalize(vec3(-hx, -hy, e * 12.0));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    vec3 n = glassNormal(uv);
    vec2 off = n.xy * bend;
    float r = texture(iChannel0, clamp(uv + off * (1.0 + split), 0.0, 1.0)).r;
    float g = texture(iChannel0, clamp(uv + off, 0.0, 1.0)).g;
    float b = texture(iChannel0, clamp(uv + off * (1.0 - split), 0.0, 1.0)).b;
    vec3 light = normalize(vec3(0.4, 0.6, 0.7));
    float spec = pow(max(dot(n, light), 0.0), 40.0) * shine;
    float rim = pow(1.0 - n.z, 2.0) * shine * 0.5;
    fragColor = vec4((vec3(r, g, b) + spec + rim) * kMask(kUV), 1.0);
}
