// about: the picture on image0 smeared in one direction, as if it rushed past; a drawn arrow sets the direction. The loop or a beat can kick it.
// type: lens
uniform float length_;    // 0.06 0 0.5    how far it smears, as a part of the picture's width
uniform float angle;      // 0 -180 180    degrees (a drawn arrow comes first)
uniform float pulse;      // 0 0 1         how much the smear goes away and comes back within one loop
uniform float beat;       // 0 0 1         a beat adds a kick of smear
const float TAU = 6.2831853;
const int N = 36;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    vec2 dir = kVector > 0.5 ? kDir : vec2(cos(radians(angle)), sin(radians(angle)));
    float len = length_ * (1.0 - pulse * (0.5 - 0.5 * cos(TAU * kLoop))) * (1.0 + beat * kBeat * 2.0);
    vec2 step_ = dir * len * iResolution.x / iResolution.xy;
    vec3 sum = vec3(0.0);
    for (int i = 0; i < N; i++)
        sum += texture(iChannel0, clamp(uv + step_ * ((float(i) + 0.5) / float(N) - 0.5), 0.0, 1.0)).rgb;
    fragColor = vec4(sum / float(N) * kMask(uv), 1.0);
}
