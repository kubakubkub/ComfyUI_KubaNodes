// about: the bright parts of the picture on image0 melt downwards (or along a drawn arrow) in long streaks, like sorted pixels; the amount breathes with the loop.
// type: look
uniform float length_;    // 0.35 0.02 1 how long the streaks are
uniform float level;      // 0.6 0 1     how bright a place has to be to start a streak
uniform float breathe;    // 0.25 0 0.5  how much the level swings in one loop
uniform float sideways;   // 0 0 1       1 = streaks run to the right
uniform float sound;      // 0 0 1       a loud sound makes more streaks
const float TAU = 6.2831853;
vec3 pic(vec2 uv) { return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    vec2 dir = sideways > 0.5 ? vec2(-1.0, 0.0) : vec2(0.0, 1.0);        // where a streak comes from
    if (kVector > 0.5) dir = -kDir * vec2(iResolution.y / iResolution.x, 1.0);
    float lv = clamp(level + breathe * sin(TAU * kLoop) - sound * kLevel * 0.4, 0.0, 1.0);
    vec3 col = pic(uv);
    const int STEPS = 64;
    for (int i = 1; i <= STEPS; i++) {
        float t = float(i) / float(STEPS);
        vec2 q = uv + dir * t * length_;
        if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) break;
        vec3 c = pic(q);
        if (lum(c) > lv) { col = mix(c, col, t * t); break; }
    }
    fragColor = vec4(col * kMask(uv), 1.0);
}
