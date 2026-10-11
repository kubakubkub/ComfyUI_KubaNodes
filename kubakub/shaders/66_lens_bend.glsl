// about: the picture on image0 through a strong lens: it bulges like a fish eye or pinches in, with colour fringes towards the corners. The loop can let it breathe.
// type: lens
uniform float bend;       // 0.35 -1 1     above 0 it bulges (fish eye), below 0 it pinches in
uniform float fringe;     // 0.012 0 0.08  how far the colours come apart towards the corners
uniform float zoom;       // 1 0.5 3
uniform float dark;       // 0.3 0 1       how much darker the corners are
uniform float breathe;    // 0 0 1         how far the bend swings within one loop
uniform float sound;      // 0 0 1         the low end of a sound pushes the bulge
const float TAU = 6.2831853;
vec3 pic(vec2 uv) { vec2 q = clamp(uv, 0.0, 1.0); return texture(iChannel0, q).rgb * step(distance(q, uv), 1e-5); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.x / iResolution.y, 1.0);
    vec2 d = (uv - 0.5) * asp;
    float b = bend + breathe * 0.5 * sin(TAU * kLoop) + sound * kLow * 0.5;
    float corner = dot(0.5 * asp, 0.5 * asp), r2 = dot(d, d);
    float k = (1.0 + b * r2 / corner) / (1.0 + max(b, 0.0)) / zoom;      // the corners stay inside the picture
    vec2 p = d * k / asp;
    float f = fringe * r2 / corner;
    vec3 c = vec3(pic(0.5 + p * (1.0 + f)).r, pic(0.5 + p).g, pic(0.5 + p * (1.0 - f)).b);
    fragColor = vec4(c * (1.0 - dark * r2 / corner) * kMask(uv), 1.0);
}
