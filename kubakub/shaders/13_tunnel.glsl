// about: a flight through a striped tunnel. Loops without a jump; a beat can flash the rings.
uniform float rings;      // 3 1 12      rings passing in one loop
uniform float segments;   // 12 2 48     stripes around the tunnel
uniform float twist;      // 0.5 -3 3    how much the stripes spiral
uniform float fog;        // 0.6 0 1     how dark the far end is
uniform float beat;       // 0 0 1       how much brighter on a beat
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float r = max(length(p), 1e-4), a = atan(p.y, p.x) / TAU;
    float z = 0.25 / r;
    float u = z + kLoop * floor(rings + 0.5);
    float seg = floor(segments + 0.5);
    float s = a * seg + floor(twist * 2.0 + 0.5) * 0.5 * floor(u * 2.0) / 1.0 * 0.0 + twist * z;
    float tile = step(0.5, fract(u * 2.0)) * 2.0 - 1.0;
    float stripe = step(0.5, fract(s));
    float v = tile * (stripe * 2.0 - 1.0) * 0.5 + 0.5;
    float ring = smoothstep(0.06, 0.0, abs(fract(u * 2.0) - 0.5) - 0.44);
    vec3 col = mix(LAV * 0.25, ORG, v) + vec3(1.0) * ring * (0.4 + beat * kBeat);
    col *= mix(1.0, smoothstep(0.0, 0.6, r), fog);
    fragColor = vec4(col * kMask(kUV), 1.0);
}
