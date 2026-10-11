// about: the picture on image0 as a real relief: what stands out (a depth pass on image1, else its brightness) rises out of the wall, the view leans from side to side so near parts hide far ones, and a circling light throws true shadows. Raymarched. Loops without a jump. Writes a depth.
// type: relief
// optional: image1
uniform float height;     // 0.12 0 0.5  how far the forms stand out, in picture heights
uniform float sway;       // 0.5 0 2     how far the view leans to the side
uniform float nod;        // 0.25 0 2    ... and up and down
uniform float sun;        // 0.5 0.05 2  how high the light stands: low = long shadows
uniform float turns;      // 1 0 6       times the light circles in one loop (0 = it stands at angle)
uniform float angle;      // 135 0 360   where the light comes from, in degrees (0 = from the right, 90 = from above)
uniform float dark;       // 0.7 0 1     how dark the shadow is
uniform float soft;       // 0.15 0 1    0 = a hard edge
uniform float shade;      // 0.6 0 1     slopes turned to the light get brighter, the others darker
uniform float steps;      // 64 16 200   more = cleaner edges where near parts hide far ones, slower
const float TAU = 6.2831853;
vec3 pic(vec2 uv) { return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float depthAt(vec2 uv) {                          // a depth pass on image1 when there is one, else how bright the picture is
    return iChannelResolution[1].x > 1.5 ? texture(iChannel1, clamp(uv, 0.0, 1.0)).r : lum(pic(uv));
}
float hAt(vec2 uv) { return height * depthAt(uv); }      // how high the surface stands over the wall here

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.y / iResolution.x, 1.0);
    float a = TAU * kLoop, top = max(height, 1e-4), n = clamp(steps, 16.0, 200.0), dz = top / n, z = top;
    vec2 lean = vec2(sway * sin(a), nod * sin(2.0 * a)) * asp;      // the view ray moves this far sideways per height
    vec2 q = uv;
    for (int i = 0; i < 200; i++) {                    // walk down the ray from the highest a form can be to the wall
        if (float(i) >= n) break;
        z -= dz;
        q = uv + (z - 0.5 * top) * lean;
        if (hAt(q) >= z) {                            // it went into the surface on this step: halve in on the place
            float lo = z, hi = z + dz;
            for (int j = 0; j < 6; j++) {
                float mid = 0.5 * (lo + hi);
                if (hAt(uv + (mid - 0.5 * top) * lean) >= mid) lo = mid; else hi = mid;
            }
            z = 0.5 * (lo + hi);
            break;
        }
    }
    z = max(z, 0.0);
    q = uv + (z - 0.5 * top) * lean;

    float la = radians(angle) + a * floor(turns + 0.5), lit = 1.0, reach = min(top / max(sun, 0.05), 0.6);
    vec2 to = vec2(cos(la), sin(la));
    for (int i = 1; i <= 32; i++) {                    // walk towards the light: something higher than the ray shades this place
        float t = float(i) / 32.0 * reach;
        float over = hAt(q + to * asp * t) - (z + t * sun) - 0.02 * top;      // a depth pass in 8 bit has steps: do not let them throw shadows
        lit = min(lit, 1.0 - smoothstep(0.0, (0.02 + soft * 0.25) * top, over));
    }
    vec2 e = 4.0 / iResolution.xy;                    // a wide step: the steps of an 8 bit depth pass would show as stripes
    vec2 slope = vec2(hAt(q + vec2(e.x, 0.0)) - hAt(q - vec2(e.x, 0.0)), hAt(q + vec2(0.0, e.y)) - hAt(q - vec2(0.0, e.y))) / (2.0 * e.y);
    vec3 L = normalize(vec3(to, sun));
    float facing = max(dot(normalize(vec3(-slope, 1.0)), L), 0.0) / L.z;      // 1 on the flat wall
    vec3 c = pic(q) * mix(1.0, clamp(facing, 0.0, 1.8), shade);
    c *= mix(1.0 - dark, 1.0, lit);
    kDepth = z / top;
    fragColor = vec4(clamp(c, 0.0, 1.0) * kMask(uv), 1.0);
}
