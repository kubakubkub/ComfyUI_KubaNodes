// about: the picture on image0 rolled into a little planet, its ground in the middle and its sky all around (or the other way: a tunnel). It turns; loops without a jump.
// type: lens
uniform float amount;     // 1 0 1         0 = the picture as it is, 1 = the planet
uniform float turn;       // 1 0 6         turns in one loop (0 = it holds still)
uniform float zoom;       // 1 0.3 3
uniform float tunnel;     // 0 0 1         0 = the ground in the middle, 1 = the sky in the middle
uniform float mirror;     // 1 0 1         1 = the picture is mirrored so the planet has no seam
uniform float curve;      // 1 0.3 3       above 1 gives the ground more room
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.x / iResolution.y, 1.0);
    vec2 p = (uv - 0.5) * asp * 2.0 / zoom;
    float r = pow(clamp(length(p), 0.0, 1.0), curve);
    float a = fract(atan(p.y, p.x) / TAU + 0.75 + floor(turn + 0.5) * kLoop);
    a = mix(a, abs(a * 2.0 - 1.0), step(0.5, mirror));
    vec2 src = vec2(a, mix(r, 1.0 - r, step(0.5, tunnel)));
    vec3 planet = texture(iChannel0, clamp(src, 0.0, 1.0)).rgb * smoothstep(1.0, 0.985, length(p) * mix(0.0, 1.0, step(tunnel, 0.5)));
    fragColor = vec4(mix(texture(iChannel0, uv).rgb, planet, amount) * kMask(uv), 1.0);
}
