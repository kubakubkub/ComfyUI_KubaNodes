// about: the picture on image0 as tiles that change size with the loop or the sound.
// type: look
uniform float tiles;      // 40 2 300    tiles across the picture
uniform float breathe;    // 0.5 0 1     how much the tile size swings in one loop
uniform float gap;        // 0.1 0 0.6   the dark line between tiles
uniform float round_;     // 0 0 1       0 = squares, 1 = dots
uniform float sound;      // 0 0 2       a kick makes the tiles bigger
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.x / iResolution.y, 1.0);
    float n = max(2.0, tiles * (1.0 - breathe * 0.5 * (0.5 + 0.5 * sin(TAU * kLoop))) / (1.0 + sound * kLow));
    vec2 g = uv * asp * n, id = floor(g), f = fract(g) - 0.5;
    vec3 c = texture(iChannel0, clamp((id + 0.5) / n / asp, 0.0, 1.0)).rgb;
    float d = mix(max(abs(f.x), abs(f.y)), length(f) * 0.9, round_);
    float m = smoothstep(0.5 - gap * 0.5, 0.5 - gap * 0.5 - 0.04, d);
    fragColor = vec4(c * m * kMask(uv), 1.0);
}
