# ── Example effect plugins ────────────────────────────────────────────────────
#
# A gallery of out-of-tree effects registered through the public API. Each is a pure
# callback — no kernel, no graph node, no CPU/GPU split — and once registered it is
# in the editor's Add-effect menu, keyframable, and callable over MCP.
#
#   include("examples/example_plugins.jl")
#
# The categories:
#   • Pointwise, no params — invert
#   • Pointwise, one param — sepia, posterize
#   • Pointwise, multiple params (namespaced keys) — levels
#   • Stencil (n×m neighborhood) — edges, emboss

import VideoEditor as VE
using VideoEditor.GeometryBasics: Vec3f   # idempotent even if Vec3f is already in scope (e.g. GLMakie)

# luma of a linear-ish RGB triple
luma(v) = 0.299f0 * v[1] + 0.587f0 * v[2] + 0.114f0 * v[3]

# ── Pointwise, no params ──────────────────────────────────────────────────────
VE.registerplugin!(:invert, "Invert", VE.FxParam[],
    p -> VE.Pointwise((c, uv) -> Vec3f(1.0f0, 1.0f0, 1.0f0) - c))

# ── Pointwise, one param ──────────────────────────────────────────────────────
VE.registerplugin!(:sepia, "Sepia", [VE.FxParam(:amount, "Amount"; max = 1.0, default = 0.8)],
    p -> (a = Float32(p.amount);
          VE.Pointwise() do c, uv
              y = luma(c)
              c * (1.0f0 - a) + Vec3f(y * 1.07f0, y * 0.74f0, y * 0.43f0) * a
          end))

VE.registerplugin!(:posterize, "Posterize", [VE.FxParam(:levels, "Levels"; min = 2, max = 16, default = 5)],
    p -> (n = Float32(round(p.levels));
          VE.Pointwise((c, uv) -> Vec3f(round(c[1] * n), round(c[2] * n), round(c[3] * n)) / n)))

# ── Pointwise, multiple params (→ namespaced keys :levels_black / :levels_white) ─
VE.registerplugin!(:levels, "Levels",
    [VE.FxParam(:black, "Black point"; max = 1.0, default = 0.0),
     VE.FxParam(:white, "White point"; max = 1.0, default = 1.0)],
    p -> (b = Float32(p.black); w = Float32(max(p.white, p.black + 1.0f-3)); s = 1.0f0 / (w - b);
          VE.Pointwise((c, uv) -> Vec3f(clamp((c[1] - b) * s, 0.0f0, 1.0f0),
                                        clamp((c[2] - b) * s, 0.0f0, 1.0f0),
                                        clamp((c[3] - b) * s, 0.0f0, 1.0f0)))))

# ── Stencil (neighborhood) ────────────────────────────────────────────────────
# Sobel edge magnitude on luma.
VE.registerplugin!(:edges, "Edges", VE.FxParam[],
    p -> VE.Stencil(1) do sample, r, uv
        g(di, dj) = luma(sample(di, dj))
        gx = g(1, -1) + 2.0f0 * g(1, 0) + g(1, 1) - g(-1, -1) - 2.0f0 * g(-1, 0) - g(-1, 1)
        gy = g(-1, 1) + 2.0f0 * g(0, 1) + g(1, 1) - g(-1, -1) - 2.0f0 * g(0, -1) - g(1, -1)
        m = sqrt(gx * gx + gy * gy)
        Vec3f(m, m, m)
    end)

# Directional emboss: grey + the diagonal luma slope.
VE.registerplugin!(:emboss, "Emboss", [VE.FxParam(:strength, "Strength"; min = 1, max = 6, default = 2)],
    p -> (k = Float32(p.strength);
          VE.Stencil(1) do sample, r, uv
              e = 0.5f0 + k * (luma(sample(1, 1)) - luma(sample(-1, -1)))
              Vec3f(e, e, e)
          end))

@info "registered example plugins: invert, sepia, posterize, levels, edges, emboss"
