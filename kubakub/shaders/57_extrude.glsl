// about: the regions rise out of the wall as blocks with sloped edges and throw shadows on it (connect the field of kubakub mask field, edge distance); a picture on image0 is painted on them. Raymarched. Loops without a jump; a beat can lift them. Writes a depth.
// type: regions
// optional: image0
// sample: field
uniform float height;     // 0.1 0 0.4   how far the blocks stand out, in picture heights
uniform float bevel;      // 0.3 0.01 1  how far in from the outline the slope reaches (1 = up to the middle)
uniform float round_;     // 0.5 0 1     0 = a straight slope, 1 = a round pillow
uniform float rise;       // 1 0 4       times the blocks rise and sink in one loop (0 = they stay up)
uniform float wave;       // 0 0 2       the middle of a region rises later than its outline
uniform float sway;       // 0.5 0 2     how far the view leans to the side
uniform float sun;        // 0.8 0.05 2  how high the light stands: low = long shadows
uniform float turns;      // 1 0 6       times the light circles in one loop (0 = it stands at angle)
uniform float angle;      // 135 0 360   where the light comes from, in degrees (0 = from the right, 90 = from above)
uniform float dark;       // 0.6 0 1     how dark the shadow is
uniform float shade;      // 0.6 0 1     slopes turned to the light get brighter, the others darker
uniform float beat;       // 0 0 1       how much higher on a beat
uniform float steps;      // 64 16 200   more = cleaner edges, slower
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
const float TAU = 6.2831853;
float hAt(vec2 uv) {                              // how high the surface stands over the wall here
    float f = kField(clamp(uv, 0.0, 1.0)), x = clamp(f / max(bevel, 1e-3), 0.0, 1.0);
    float form = mix(x, sqrt(1.0 - (1.0 - x) * (1.0 - x)), round_);
    float up = rise > 0.5 ? 0.5 - 0.5 * cos(TAU * (kLoop * floor(rise + 0.5) - wave * f)) : 1.0;
    return height * form * up * (1.0 + beat * kBeat) * kMask(clamp(uv, 0.0, 1.0));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.y / iResolution.x, 1.0);
    float a = TAU * kLoop, top = max(height * (1.0 + beat), 1e-4), n = clamp(steps, 16.0, 200.0), dz = top / n, z = top;
    vec2 lean = vec2(sway * sin(a), 0.4 * sway * sin(2.0 * a)) * asp;      // the view ray moves this far sideways per height
    vec2 q = uv;
    for (int i = 0; i < 200; i++) {                    // walk down the ray from the highest a block can be to the wall
        if (float(i) >= n) break;
        z -= dz;
        q = uv + z * lean;                            // the wall stays where it is, the tops move
        if (hAt(q) >= z) {                            // it went into the surface on this step: halve in on the place
            float lo = z, hi = z + dz;
            for (int j = 0; j < 6; j++) {
                float mid = 0.5 * (lo + hi);
                if (hAt(uv + mid * lean) >= mid) lo = mid; else hi = mid;
            }
            z = 0.5 * (lo + hi);
            break;
        }
    }
    z = max(z, 0.0);
    q = uv + z * lean;

    float la = radians(angle) + a * floor(turns + 0.5), lit = 1.0, reach = min(top / max(sun, 0.05), 0.6);
    vec2 to = vec2(cos(la), sin(la));
    for (int i = 1; i <= 32; i++) {                    // walk towards the light: something higher than the ray shades this place
        float t = float(i) / 32.0 * reach;
        lit = min(lit, 1.0 - smoothstep(0.0, 0.04 * top, hAt(q + to * asp * t) - (z + t * sun) - 0.02 * top));
    }
    vec2 e = 2.0 / iResolution.xy;
    vec2 slope = vec2(hAt(q + vec2(e.x, 0.0)) - hAt(q - vec2(e.x, 0.0)), hAt(q + vec2(0.0, e.y)) - hAt(q - vec2(0.0, e.y))) / (2.0 * e.y);
    vec3 L = normalize(vec3(to, sun));
    float facing = max(dot(normalize(vec3(-slope, 1.0)), L), 0.0) / L.z;      // 1 on the flat wall
    float on = smoothstep(0.0, 0.03 * top, hAt(q));
    vec3 c = iChannelResolution[0].x > 1.5 ? texture(iChannel0, clamp(q, 0.0, 1.0)).rgb
                                           : mix(vec3(0.2, 0.18, 0.22), mix(colour_a, colour_b, kField(clamp(q, 0.0, 1.0))), on);
    c *= mix(1.0, clamp(facing, 0.0, 1.8), shade) * mix(1.0 - dark, 1.0, lit);
    kDepth = z / top;
    fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
