// about: the picture on image0 seen through moving water: rings run out from a point and bend it, with a glint on their crests; a drawn arrow's start is the point. Loops without a jump.
// type: lens
uniform float waves;      // 12 1 60       rings from the point to the edge
uniform float amount;     // 0.008 0 0.08  how far they bend the picture
uniform float speed;      // 2 1 12        rings that pass a place in one loop
uniform float cx;         // 0.5 0 1       the point (a drawn arrow's start comes first)
uniform float cy;         // 0.5 0 1
uniform float shine;      // 0.25 0 1      the glint on the crests
uniform float sound;      // 0 0 2         the low end of a sound makes the waves higher
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.x / iResolution.y, 1.0);
    vec2 c = kVector > 0.5 ? kFrom : vec2(cx, cy);
    vec2 d = (uv - c) * asp;
    float r = length(d);
    float w = sin(TAU * (r * waves - floor(speed + 0.5) * kLoop));
    float k = amount * (1.0 + sound * kLow * 2.0) * smoothstep(0.0, 0.08, r);
    vec3 col = texture(iChannel0, clamp(uv + normalize(d + 1e-6) / asp * w * k, 0.0, 1.0)).rgb;
    col += shine * pow(max(w, 0.0), 12.0) * smoothstep(0.0, 0.08, r) * (0.4 + 0.6 * dot(col, vec3(0.333)));
    fragColor = vec4(col * kMask(uv), 1.0);
}
