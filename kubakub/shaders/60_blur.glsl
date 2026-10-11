// about: the picture on image0 out of focus: a soft, even blur. The loop can pull the focus, a loud sound sharpens it.
// type: lens
uniform float radius;     // 0.02 0 0.15   how far it blurs, as a part of the picture's height
uniform float breathe;    // 0 0 1         how much the blur goes away and comes back within one loop
uniform float sound;      // 0 0 2         a loud sound sharpens it
uniform float taps;       // 48 8 128      samples per pixel: more is smoother and slower
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.y / iResolution.x, 1.0);
    float r = radius * (1.0 - breathe * (0.5 - 0.5 * cos(TAU * kLoop))) / (1.0 + sound * kLevel * 3.0);
    int n = int(clamp(taps, 8.0, 128.0));
    vec3 sum = vec3(0.0);
    for (int i = 0; i < n; i++) {                     // a spiral of samples over a disc
        float f = (float(i) + 0.5) / float(n), a = float(i) * 2.39996323;
        sum += texture(iChannel0, clamp(uv + vec2(cos(a), sin(a)) * sqrt(f) * r * asp, 0.0, 1.0)).rgb;
    }
    fragColor = vec4(sum / float(n) * kMask(uv), 1.0);
}
