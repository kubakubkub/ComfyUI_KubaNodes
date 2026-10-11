// about: the picture on image0 rushing out of a point, like a zoom during a long exposure; a drawn arrow's start is the point. The loop or a beat can kick it.
// type: lens
uniform float amount;     // 0.2 0 1       how far it streaks towards the point
uniform float spin;       // 0 -90 90      degrees it also turns while it streaks
uniform float cx;         // 0.5 0 1       the point (a drawn arrow's start comes first)
uniform float cy;         // 0.5 0 1
uniform float pulse;      // 0 0 1         how much the streaks go away and come back within one loop
uniform float beat;       // 0 0 1         a beat adds a kick
const float TAU = 6.2831853;
const int N = 40;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.x / iResolution.y, 1.0);
    vec2 c = kVector > 0.5 ? kFrom : vec2(cx, cy);
    float k = (1.0 - pulse * (0.5 - 0.5 * cos(TAU * kLoop))) * (1.0 + beat * kBeat * 2.0);
    vec3 sum = vec3(0.0);
    for (int i = 0; i < N; i++) {
        float t = (float(i) + 0.5) / float(N), a = radians(spin) * t * k;
        vec2 d = (uv - c) * asp * (1.0 - amount * k * t);
        d = mat2(cos(a), -sin(a), sin(a), cos(a)) * d;
        sum += texture(iChannel0, clamp(c + d / asp, 0.0, 1.0)).rgb;
    }
    fragColor = vec4(sum / float(N) * kMask(uv), 1.0);
}
