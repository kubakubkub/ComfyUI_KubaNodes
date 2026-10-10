// about: the picture on image0 on an old tube television: a curved glass, scan lines, coloured dots and a bar that rolls through.
uniform float lines;      // 240 60 1080 scan lines
uniform float curve;      // 0.12 0 0.5  how much the glass bulges
uniform float dots;       // 0.35 0 1    the red green blue dots of the tube
uniform float roll;       // 1 0 6       times the dark bar rolls through in one loop
uniform float split;      // 0.002 0 0.02 colours a little apart
uniform float noise;      // 0.06 0 0.5  snow
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV * 2.0 - 1.0;
    uv *= 1.0 + curve * dot(uv.yx, uv.yx) * vec2(0.5, 0.6);
    vec2 q = uv * 0.5 + 0.5;
    float inside = step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
    vec3 c = vec3(texture(iChannel0, clamp(q + vec2(split, 0.0), 0.0, 1.0)).r, texture(iChannel0, clamp(q, 0.0, 1.0)).g,
                  texture(iChannel0, clamp(q - vec2(split, 0.0), 0.0, 1.0)).b);
    float scan = 0.72 + 0.28 * sin(q.y * lines * 3.14159 * 2.0);
    float px = mod(fragCoord.x, 3.0);
    vec3 triad = mix(vec3(1.0), vec3(px < 1.0 ? 1.4 : 0.7, px >= 1.0 && px < 2.0 ? 1.4 : 0.7, px >= 2.0 ? 1.4 : 0.7), dots);
    float bar = 1.0 - 0.25 * smoothstep(0.12, 0.0, abs(fract(q.y + kLoop * floor(roll + 0.5)) - 0.5) - 0.3) * step(0.5, roll);
    float snow = (hash1(fragCoord + floor(fract(kLoop) * 240.0) * 13.0) - 0.5) * noise;
    float vig = smoothstep(1.45, 0.55, length(uv));
    vec3 col = (c * scan * triad * bar + snow) * vig * inside;
    fragColor = vec4(col * kMask(kUV), 1.0);
}
