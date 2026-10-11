// about: the picture on image0 in a turning kaleidoscope: a wedge of it mirrored around a point; a drawn arrow's start is the point. Loops without a jump.
// type: lens
uniform float mirrors;    // 6 1 16        how many mirrored pairs around the point
uniform float turn;       // 1 0 8         wedges it turns on in one loop (0 = it holds still)
uniform float zoom;       // 1 0.3 4
uniform float cx;         // 0.5 0 1       the point (a drawn arrow's start comes first)
uniform float cy;         // 0.5 0 1
uniform float slide;      // 0 0 1         how far the picture slides through the wedge and back in one loop
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.x / iResolution.y, 1.0);
    vec2 c = kVector > 0.5 ? kFrom : vec2(cx, cy);
    vec2 d = (uv - c) * asp;
    float seg = TAU / (2.0 * floor(clamp(mirrors, 1.0, 16.0) + 0.5));
    float a = atan(d.y, d.x) + 2.0 * seg * floor(turn + 0.5) * kLoop;
    a = abs(mod(a, 2.0 * seg) - seg);
    vec2 q = length(d) / zoom * vec2(cos(a), sin(a)) + vec2(slide * 0.25 * sin(TAU * kLoop), 0.0);
    vec2 src = c + q / asp;
    src = 1.0 - abs(1.0 - mod(src, 2.0));               // past an edge the picture comes back mirrored
    fragColor = vec4(texture(iChannel0, src).rgb * kMask(uv), 1.0);
}
