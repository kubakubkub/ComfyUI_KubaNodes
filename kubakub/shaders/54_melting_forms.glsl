// about: soft 3D forms that melt into each other and out of a wall, lit from the side with real shadows. Raymarched. Loops without a jump; the low end of a sound makes them swell. Writes a depth.
// type: space
uniform float forms;      // 5 1 8       how many forms
uniform float size;       // 0.4 0.1 0.9
uniform float melt;       // 0.4 0 1     how far they flow into each other and into the wall
uniform float blocks;     // 0.4 0 1     0 = all balls, 1 = all rounded blocks
uniform float light;      // 40 0 360    where the light comes from, in degrees (0 = from the right, 90 = from above)
uniform float sound;      // 0 0 2       a kick makes them swell
uniform float steps;      // 64 16 160   more = cleaner edges, slower
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
const float TAU = 6.2831853, WALL = -1.0;
float gTone, gForms;                              // set by map: the colour mix of the nearest forms, and how far they are
float hash1(float n) { return fract(sin(n * 127.1) * 43758.5453); }

float map(vec3 p) {                               // the distance from p to the nearest surface
    float a = TAU * kLoop, asp = iResolution.x / iResolution.y;
    int n = int(clamp(floor(forms + 0.5), 1.0, 8.0));
    float s = size * (1.0 + sound * kLow * 0.35), k = 0.02 + melt * 0.6;
    float d = 1e3, tone = 0.0, sum = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i >= n) break;
        float fi = float(i);
        vec3 c = vec3((hash1(fi + 0.3) - 0.5) * 1.5 * asp, (hash1(fi + 1.7) - 0.5) * 1.3, WALL + 0.25 + hash1(fi + 2.9) * 0.6);
        vec3 r = vec3(0.25 + 0.3 * hash1(fi + 4.1), 0.2 + 0.3 * hash1(fi + 5.3), 0.25);
        float t1 = floor(hash1(fi + 6.7) * 2.0) + 1.0, t2 = floor(hash1(fi + 7.9) * 2.0) + 1.0;
        vec3 q = p - c - r * vec3(cos(a * t1 + fi * 2.1), sin(a * t2 + fi * 1.3), sin(a + fi * 0.7));
        float si = s * (0.6 + 0.6 * hash1(fi + 9.1)), di;
        if (hash1(fi + 11.3) < blocks) {
            q.xy *= kRot(a * t1 + fi);
            q.xz *= kRot(a * t2);
            di = sdRoundBox(q, vec3(si * 0.8), si * 0.25);
        } else di = sdSphere(q, si);
        float w = exp(-6.0 * max(di, 0.0));
        tone += w * hash1(fi + 13.7);
        sum += w;
        d = opSmoothUnion(d, di, k);
    }
    gTone = tone / max(sum, 1e-4);
    gForms = d;
    return opSmoothUnion(d, p.z - WALL, k * 0.6);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 ro = vec3(0.0, 0.0, 2.2), rd = kCamera(fragCoord, ro, vec3(0.0, 0.0, WALL), 2.0);
    float t = kMarch(ro, rd, 8.0, int(clamp(steps, 16.0, 160.0)));
    vec3 col = vec3(0.0);
    if (t > 0.0) {
        vec3 p = ro + rd * t, n = kNormal(p);
        map(p);
        vec3 base = mix(mix(colour_a, colour_b, smoothstep(0.25, 0.75, gTone)), vec3(0.16, 0.14, 0.18), smoothstep(0.02, 0.3, gForms));
        vec3 L = normalize(vec3(cos(radians(light)), sin(radians(light)), 0.8));
        float ao = kAO(p, n), dif = max(dot(n, L), 0.0) * kShadow(p + n * 0.01, L, 5.0, 10.0);
        float rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
        float spec = pow(max(dot(reflect(-L, n), -rd), 0.0), 24.0) * dif;
        col = base * (0.16 * ao + dif) + colour_a * rim * 0.2 * ao + vec3(1.0, 0.95, 0.9) * spec * 0.25;
        kDepth = (p.z - WALL) / 1.7;              // black = the wall, white = what stands furthest out
    }
    fragColor = vec4(sqrt(clamp(col, 0.0, 1.0)) * kMask(kUV), 1.0);
}
