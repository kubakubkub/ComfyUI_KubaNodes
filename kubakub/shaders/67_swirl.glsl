// about: the picture on image0 wound up around a point like cream in coffee, and back again; a drawn arrow's start is the point. Loops without a jump.
// type: lens
uniform float angle;      // 200 -720 720  degrees it is wound up in the middle
uniform float size;       // 0.7 0.05 2    how far the swirl reaches
uniform float cx;         // 0.5 0 1       the point (a drawn arrow's start comes first)
uniform float cy;         // 0.5 0 1
uniform float turn;       // 1 0 4         how many times it winds up and back in one loop (0 = it holds still)
uniform float sound;      // 0 0 2         the low end of a sound winds it further
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.x / iResolution.y, 1.0);
    vec2 c = kVector > 0.5 ? kFrom : vec2(cx, cy);
    vec2 d = (uv - c) * asp;
    float n = floor(turn + 0.5);
    float k = (n < 0.5 ? 1.0 : sin(TAU * kLoop * n)) * (1.0 + sound * kLow);
    float t = max(0.0, 1.0 - length(d) / size);
    float a = radians(angle) * t * t * k;
    d = mat2(cos(a), -sin(a), sin(a), cos(a)) * d;
    fragColor = vec4(texture(iChannel0, clamp(c + d / asp, 0.0, 1.0)).rgb * kMask(uv), 1.0);
}
