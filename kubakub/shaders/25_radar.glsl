// about: a sweeping beam like a radar or a lighthouse, with a fading tail; a drawn arrow's start is its centre. Loops without a jump.
uniform float beams;      // 1 1 12      how many beams
uniform float tail;       // 0.35 0.02 1 how long the tail is
uniform float cx;         // 0.5 0 1     the centre, 0 = left edge, 1 = right edge
uniform float cy;         // 0.5 0 1     0 = bottom, 1 = top
uniform float rings;      // 4 0 20      range rings, 0 = none
uniform float turn;       // 1 -4 4      whole turns in one loop
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - (kVector > 0.5 ? kFrom : vec2(cx, cy)) * iResolution.xy) / iResolution.y;
    float r = length(p), th = atan(p.y, p.x) / TAU;
    float n = max(1.0, floor(beams));
    float x = fract((th - kLoop * floor(turn + 0.5 * sign(turn))) * n);
    float t = smoothstep(1.0 - tail, 1.0, x);
    float beam = t * t * 0.7 + smoothstep(0.985, 1.0, x) * 0.8;
    float grid = rings > 0.5 ? smoothstep(0.47, 0.5, abs(fract(r * floor(rings) * 2.0) - 0.5)) * 0.25 : 0.0;
    vec3 col = mix(LAV, ORG, smoothstep(0.6, 1.0, x)) * beam + LAV * grid * (0.3 + t);
    fragColor = vec4(col * kMask(kUV), 1.0);
}
