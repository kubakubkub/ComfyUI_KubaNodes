// about: the picture on image0 with darkened corners and film grain, as through an old lens. The grain moves; everything else holds still.
// type: lens
uniform float amount;     // 0.7 0 1       how dark the corners get
uniform float size;       // 0.75 0.1 1.5  how far the clear middle reaches
uniform float soft;       // 0.5 0.02 1    how gently it falls off
uniform float round;      // 0.6 0 1       0 = it follows the frame, 1 = a circle
uniform vec3 edge;        // #000000   the colour of the corners
uniform float grain;      // 0.08 0 0.5    how much film grain
uniform float grain_size; // 1.5 1 8       pixels per grain
uniform float beat;       // 0 0 1         a beat opens the vignette for a moment
float hash(vec3 p) { p = fract(p * vec3(0.1031, 0.1030, 0.0973)); p += dot(p, p.yzx + 33.33); return fract((p.x + p.y) * p.z); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    vec3 c = texture(iChannel0, uv).rgb;
    vec2 d = (uv - 0.5) * 2.0;
    d.x *= mix(1.0, iResolution.x / iResolution.y, round);
    float v = smoothstep(size, size + soft, length(d) / mix(1.0, 1.25, round)) * amount * (1.0 - beat * kBeat);
    float frame = floor(fract(kLoop) * max(kSeconds * iFrameRate, 1.0));      // a new grain at every frame, the same one where the loop closes
    float g = hash(vec3(floor((fragCoord + kShift) / max(grain_size, 1.0)), frame)) - 0.5;
    c += g * grain * (1.0 - 0.6 * dot(c, vec3(0.333)));
    fragColor = vec4(mix(c, edge, v) * kMask(uv), 1.0);
}
