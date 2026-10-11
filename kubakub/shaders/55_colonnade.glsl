// about: an endless walk between two rows of columns under an open sky; the sun lays their shadows across the floor. Raymarched. Loops without a jump. Writes a depth.
// type: space
// arrow: centre
uniform float bays;       // 2 1 8       columns you pass in one loop
uniform float width;      // 1.5 0.8 4   how far the rows stand apart
uniform float thick;      // 0.22 0.08 0.45 how thick a column is
uniform float arches;     // 1 0 1       0 = a straight beam on the columns, 1 = round arches
uniform float sun;        // 35 5 85     how high the sun stands, in degrees: low = long shadows
uniform float sway;       // 0.3 0 1     how far the view drifts from side to side
uniform float fog;        // 0.5 0 1     how much the far end fades into the sky
uniform float sound;      // 0 0 2       the low end of a sound brightens the sun
uniform float steps;      // 96 24 200   more = cleaner edges, slower
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
const float TAU = 6.2831853, BAY = 2.0, FLOOR = -1.0, TOP = 1.1;

float map(vec3 p) {                               // the distance from p to the nearest surface
    float ax = abs(p.x) - width, z = p.z - BAY * round(p.z / BAY);      // both rows, and every bay, are one
    float d = p.y - FLOOR;
    d = min(d, sdCylinder(vec3(ax, p.y - (FLOOR + TOP) * 0.5, z), (TOP - FLOOR) * 0.5, thick));
    d = min(d, sdRoundBox(vec3(ax, p.y - FLOOR - 0.07, z), vec3(thick * 1.35, 0.07, thick * 1.35), 0.02));
    d = min(d, sdRoundBox(vec3(ax, p.y - TOP + 0.06, z), vec3(thick * 1.3, 0.06, thick * 1.3), 0.02));
    float zh = p.z - BAY * (floor(p.z / BAY) + 0.5), r = BAY * 0.5 - thick * 1.05;      // the opening between two columns
    float slab = sdBox(vec3(ax, p.y - TOP - 0.9, 0.0), vec3(thick * 0.9, 0.9, 1.0));
    float arch = max(slab, r - length(vec2(p.y - TOP, zh)));
    float beam = sdBox(vec3(ax, p.y - TOP - 0.2, 0.0), vec3(thick * 1.1, 0.2, 1.0));
    return min(d, mix(beam, arch, arches));
}

vec3 sky(vec3 rd) {
    return mix(mix(colour_b, vec3(1.0, 0.9, 0.8), 0.35), colour_a * 0.75, smoothstep(-0.05, 0.6, rd.y));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float a = TAU * kLoop, far = 44.0;
    vec3 ro = vec3(sway * 0.45 * width * sin(a), 0.03 * sin(2.0 * a), -kLoop * floor(bays + 0.5) * BAY);
    vec3 rd = kCamera(fragCoord, ro, ro + vec3(-0.25 * sway * sin(a), 0.12, -1.0), 1.5);
    float t = kMarch(ro, rd, far, int(clamp(steps, 24.0, 200.0)));
    vec3 back = sky(rd), col = back;
    if (t > 0.0) {
        vec3 p = ro + rd * t, n = kNormal(p);
        float s = radians(clamp(sun, 5.0, 85.0));
        vec3 L = normalize(vec3(0.8 * cos(s), sin(s), -0.6 * cos(s)));
        vec3 stone = mix(vec3(0.74, 0.69, 0.66), colour_a, 0.18);
        if (p.y < FLOOR + 0.01) {                      // the floor: two tones of tiles, one bay is four of them
            vec2 g = floor(p.xz * 2.0);
            stone *= mod(g.x + g.y, 2.0) < 0.5 ? 0.62 : 0.9;
        }
        float ao = kAO(p, n), dif = max(dot(n, L), 0.0) * kShadow(p + n * 0.01, L, 12.0, 24.0);
        col = stone * (colour_b * 1.6 * dif * (1.0 + sound * kLow) + colour_a * 0.4 * ao * (0.6 + 0.4 * n.y));
        col = mix(col, back, fog * (1.0 - exp(-0.0035 * t * t)));
        kDepth = kDepthOf(t, 0.5, 14.0);
    }
    fragColor = vec4(sqrt(clamp(col, 0.0, 1.0)) * kMask(kUV), 1.0);
}
