// about: frames within frames within frames: an endless zoom into the wall. Loops without a jump.
// type: pattern
// arrow: centre
uniform float depth;      // 2 1 6       frames you pass in one loop (whole numbers, x 2)
uniform float gap;        // 1.6 1.15 4  how much bigger each frame is than the one inside it
uniform float border;     // 0.18 0.02 0.6 how thick a frame is
uniform float round_;     // 0 0 1       0 = rectangles, 1 = circles
uniform float turn;       // 0 -40 40    degrees each frame is turned against the next
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float asp = iResolution.x / iResolution.y;
    float n = 2.0 * max(1.0, floor(depth));
    float lg = log(gap);
    float d0 = mix(max(abs(p.x) / asp, abs(p.y)), length(p), round_);
    float level = floor(log(max(d0, 1e-5)) / lg + kLoop * n);
    float a = radians(turn) * (level - kLoop * n);
    vec2 q = mat2(cos(a), -sin(a), sin(a), cos(a)) * p;
    float d = mix(max(abs(q.x) / asp, abs(q.y)), length(q), round_);
    float x = fract(log(max(d, 1e-5)) / lg + kLoop * n);
    float frame = smoothstep(border, border - 0.03, x) + smoothstep(1.0 - 0.02, 1.0, x);
    float parity = mod(level, 2.0);
    vec3 wall = mix(colour_a * 0.3, colour_b * 0.3, parity) * (0.5 + 0.5 * x);
    vec3 col = mix(wall, mix(colour_b, colour_a, parity), clamp(frame, 0.0, 1.0));
    col *= smoothstep(0.0, 0.05, d0);
    fragColor = vec4(col * kMask(kUV), 1.0);
}
