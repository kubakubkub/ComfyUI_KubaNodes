// about: a light circles the picture on image0 and everything that stands out throws a hard shadow (a depth pass on image1, else its brightness). A fake shadow, no 3D needed. Loops without a jump.
// type: relief
// optional: image1
uniform float sun;        // 0.1 0.01 1.5 how high the light stands: low = long shadows
uniform float reach;      // 0.12 0.01 0.5 how far a shadow can fall, in picture heights
uniform float turns;      // 1 0 6       times the light circles in one loop (0 = it stands at angle)
uniform float angle;      // 135 0 360   where the light comes from, in degrees (0 = from the right, 90 = from above)
uniform float dark;       // 0.75 0 1    how dark the shadow is
uniform float soft;       // 0.1 0 1     0 = a hard edge
uniform float shade;      // 0.25 0 1    the sides turned away from the light get darker too
uniform float tint;       // 0.3 0 1     how much lavender is in the shadow
uniform vec3 colour_a;    // #a187b7   the colour of the shade
const float TAU = 6.2831853;
vec3 pic(vec2 uv) { return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float depthAt(vec2 uv) {                          // a depth pass on image1 when there is one, else how bright the picture is
    return iChannelResolution[1].x > 1.5 ? texture(iChannel1, clamp(uv, 0.0, 1.0)).r : lum(pic(uv));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float a = radians(angle) + TAU * kLoop * floor(turns + 0.5);
    vec2 to = vec2(cos(a), sin(a)) * vec2(iResolution.y / iResolution.x, 1.0);      // towards the light, in picture space
    float h0 = depthAt(uv), lit = 1.0;
    for (int i = 1; i <= 32; i++) {                    // walk towards the light: something higher than the ray shades this place
        float t = float(i) / 32.0 * reach;
        float over = depthAt(uv + to * t) - (h0 + t * sun) - 0.006;      // a depth pass in 8 bit has steps: do not let them throw shadows
        lit = min(lit, 1.0 - smoothstep(0.0, 0.004 + soft * 0.06, over));
    }
    vec2 e = 8.0 / iResolution.xy;                    // a wide step: the steps of an 8 bit depth pass would show as stripes
    vec2 slope = vec2(depthAt(uv + vec2(e.x, 0.0)) - depthAt(uv - vec2(e.x, 0.0)), depthAt(uv + vec2(0.0, e.y)) - depthAt(uv - vec2(0.0, e.y)));
    float facing = clamp(-dot(slope, vec2(cos(a), sin(a))) * 2.5, -1.0, 1.0);      // falling towards the light = lit
    vec3 c = pic(uv) * (1.0 + shade * 0.5 * facing);
    vec3 shadow = mix(c * (1.0 - dark), colour_a * lum(c) * (1.0 - dark), tint);
    fragColor = vec4(clamp(mix(shadow, c, lit), 0.0, 1.0) * kMask(uv), 1.0);
}
