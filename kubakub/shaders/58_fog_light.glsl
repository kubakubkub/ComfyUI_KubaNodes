// about: a beam of light swinging through drifting fog, like a projector in a hazy hall. Raymarched through the fog. Loops without a jump; a sound brightens the beam.
// type: space
uniform float density;    // 0.6 0.05 1.5 how thick the fog is
uniform float scale;      // 1.4 0.4 5   bigger = finer banks of fog
uniform float beam;       // 1 0 3       how bright the beam is
uniform float spread;     // 0.35 0.05 1 how wide the beam opens
uniform float rays;       // 9 0 30      shafts inside the beam (0 = an even cone)
uniform float sweep;      // 1 0 4       times the beam swings from side to side in one loop (0 = it stands still)
uniform float warm;       // 0.5 0 1
uniform float sound;      // 0 0 2
uniform float steps;      // 40 12 120   more = smoother fog, slower
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
const float TAU = 6.2831853;
float hash3(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
float noise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    vec2 o = vec2(0.0, 1.0);
    return mix(mix(mix(hash3(i + o.xxx), hash3(i + o.yxx), f.x), mix(hash3(i + o.xyx), hash3(i + o.yyx), f.x), f.y),
               mix(mix(hash3(i + o.xxy), hash3(i + o.yxy), f.x), mix(hash3(i + o.xyy), hash3(i + o.yyy), f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 3; i++) { v += a * noise(p); p = p * 2.03 + 11.7; a *= 0.5; }
    return v;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float a = TAU * kLoop, sw = floor(sweep + 0.5);
    vec3 ro = vec3(0.0, 0.0, 3.0), rd = kCamera(fragCoord, ro, vec3(0.0), 1.6);
    vec3 lamp = vec3(0.0, 1.9, -0.4);                 // above the picture
    vec3 ld = normalize(vec3(0.75 * sin(a * sw), -1.0, 0.3 * cos(a * sw) + 0.1));
    vec3 u1 = normalize(cross(ld, vec3(0.0, 0.0, 1.0))), u2 = cross(ld, u1);
    vec3 drift = 0.6 * vec3(cos(a), 0.5 * sin(a), sin(a));      // the fog moves in a circle: the loop closes
    vec3 glow = mix(mix(colour_a, colour_b, warm), vec3(1.0, 0.95, 0.9), 0.35) * beam * (1.0 + sound * kLevel);
    float n = clamp(steps, 12.0, 120.0), dt = 6.0 / n, edge = cos(spread * 0.9), count = floor(rays + 0.5);
    float t = dt * hash3(vec3(fragCoord, 1.0)), through = 1.0;      // every pixel starts a little off: no bands
    vec3 col = vec3(0.0);
    for (int i = 0; i < 120; i++) {
        if (float(i) >= n || through < 0.02) break;
        vec3 p = ro + rd * t, v = p - lamp;
        float d = length(v), fogHere = density * (0.12 + 0.88 * smoothstep(0.3, 0.75, fbm(p * scale + drift)));
        vec3 vn = v / max(d, 1e-4);
        float cone = smoothstep(edge, mix(edge, 1.0, 0.6), dot(vn, ld));
        float shaft = count > 0.5 ? 0.35 + 0.65 * pow(0.5 + 0.5 * sin(atan(dot(vn, u2), dot(vn, u1)) * count), 2.0) : 1.0;
        col += through * fogHere * (glow * cone * shaft * 14.0 / (0.5 + 0.3 * d * d) + colour_a * 0.1) * dt;
        through *= exp(-fogHere * dt * 0.8);
        t += dt;
    }
    fragColor = vec4(clamp(col, 0.0, 1.0) * kMask(kUV), 1.0);
}
