// about: the picture on image0 graded like a photograph: exposure, contrast, colour, warmth, and where black and white sit. A still look; the loop can let the light breathe.
// type: lens
uniform float exposure;   // 0 -3 3        stops brighter or darker
uniform float contrast;   // 1 0.2 3
uniform float saturation; // 1 0 3         0 = grey
uniform float warmth;     // 0 -1 1        towards candle light, or towards a blue sky
uniform float tint;       // 0 -1 1        towards magenta, or towards green
uniform float black;      // 0 0 0.5       what counts as black
uniform float white;      // 1 0.5 1       what counts as white
uniform float gamma;      // 1 0.3 3       the middle tones: above 1 lifts them
uniform float breathe;    // 0 0 2         stops the exposure swings within one loop
uniform float sound;      // 0 0 2         a loud sound brightens it
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    vec3 c = texture(iChannel0, uv).rgb;
    c = (c - black) / max(white - black, 1e-3);
    c *= exp2(exposure + breathe * 0.5 * sin(TAU * kLoop) + sound * kLevel);
    c *= vec3(1.0 + 0.3 * warmth + 0.1 * tint, 1.0 - 0.2 * tint, 1.0 - 0.3 * warmth + 0.1 * tint);
    c = (c - 0.5) * contrast + 0.5;
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(l), c, saturation);
    c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / max(gamma, 0.05)));
    fragColor = vec4(c * kMask(uv), 1.0);
}
