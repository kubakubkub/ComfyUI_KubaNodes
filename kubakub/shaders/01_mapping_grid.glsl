// about: a line-up grid for the projector: squares, the centre cross, circles and the diagonals; a scan line runs through.
// type: regions
uniform float cells;      // 12 2 64   squares across the picture
uniform float line;       // 2 0.5 12  line width in pixels
uniform float scan;       // 1 0 1     the running scan line, 0 = off

float stroke(float d, float w) { return 1.0 - smoothstep(w * 0.5, w * 0.5 + 1.0, abs(d)); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = iResolution.xy;
    vec2 p = fragCoord - 0.5 * res;
    float cell = res.x / max(2.0, floor(cells));
    vec2 g = abs(mod(p + 0.5 * cell, cell) - 0.5 * cell);
    float grid = max(stroke(g.x, line), stroke(g.y, line));
    float centre = max(stroke(p.x, line * 2.0), stroke(p.y, line * 2.0));
    float r = length(p);
    float rings = max(stroke(r - 0.25 * res.y, line), stroke(r - 0.5 * res.y, line));
    float k = res.y / res.x;
    float diag = max(stroke((p.y - p.x * k) / sqrt(1.0 + k * k), line), stroke((p.y + p.x * k) / sqrt(1.0 + k * k), line));
    vec2 e = min(fragCoord, res - fragCoord);
    float border = 1.0 - step(line * 2.0, min(e.x, e.y));
    vec3 col = vec3(0.35) * grid;
    col = max(col, vec3(0.63, 0.53, 0.72) * max(rings, diag));
    col = max(col, vec3(0.95, 0.54, 0.35) * centre);
    col = max(col, vec3(1.0) * border);
    float sweep = stroke(fragCoord.x - kLoop * res.x, line * 3.0) * scan;
    col = max(col, vec3(1.0) * sweep);
    fragColor = vec4(col, 1.0);
}
