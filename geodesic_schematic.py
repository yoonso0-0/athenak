"""
Schematic figure of the geodesic sphere discretization used by SphericalGrid.
Replicates the C++ construction in geodesic_grid.cpp exactly.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})


# ── replicate geodesic_grid.cpp construction ──────────────────────────────────

def build_geodesic_grid(nlevel):
    sin_ang = 2.0 / np.sqrt(5.0)
    cos_ang = 1.0 / np.sqrt(5.0)

    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([sin_ang, 0.0, cos_ang])
    p3 = np.array([sin_ang * np.cos(0.2 * np.pi),  sin_ang * np.sin(0.2 * np.pi),  -cos_ang])
    p4 = np.array([sin_ang * np.cos(-0.4 * np.pi), sin_ang * np.sin(-0.4 * np.pi),  cos_ang])
    p5 = np.array([sin_ang * np.cos(-0.2 * np.pi), sin_ang * np.sin(-0.2 * np.pi), -cos_ang])
    p6 = np.array([0.0, 0.0, -1.0])

    NL = nlevel
    anorm = np.zeros((5, 2 + NL, 2 + 2 * NL, 3))
    apnorm = np.array([[0., 0., 1.], [0., 0., -1.]])

    row_index = 1
    for l in range(NL):
        col_index = 1
        for m in range(l, NL):
            v = ((m - l + 1) * p2 + (NL - m - 1) * p1 + l * p4) / NL
            anorm[0, row_index, col_index] = v / np.linalg.norm(v)
            col_index += 1
        for m in range(NL - l, NL):
            v = ((NL - l) * p2 + (m - NL + l + 1) * p5 + (NL - m - 1) * p4) / NL
            anorm[0, row_index, col_index] = v / np.linalg.norm(v)
            col_index += 1
        for m in range(l, NL):
            v = ((m - l + 1) * p3 + (NL - m - 1) * p2 + l * p5) / NL
            anorm[0, row_index, col_index] = v / np.linalg.norm(v)
            col_index += 1
        for m in range(NL - l, NL):
            v = ((NL - l) * p3 + (m - NL + l + 1) * p6 + (NL - m - 1) * p5) / NL
            anorm[0, row_index, col_index] = v / np.linalg.norm(v)
            col_index += 1
        row_index += 1

    for ptch in range(1, 5):
        ang = ptch * 0.4 * np.pi
        c, s = np.cos(ang), np.sin(ang)
        for l in range(1, 1 + NL):
            for m in range(1, 1 + 2 * NL):
                x0, y0, z0 = anorm[0, l, m]
                anorm[ptch, l, m] = [x0 * c + y0 * s, y0 * c - x0 * s, z0]

    # Collect unique grid points (poles + interior of all 5 patches)
    pts_list = [apnorm[0], apnorm[1]]
    for ptch in range(5):
        for l in range(1, 1 + NL):
            for m in range(1, 1 + 2 * NL):
                pts_list.append(anorm[ptch, l, m].copy())
    pts = np.array(pts_list)
    pts_unique = np.unique(np.round(pts, 8), axis=0)
    return pts_unique


# ── icosahedron vertices and edges ────────────────────────────────────────────

def icosahedron_edges():
    sin_ang = 2.0 / np.sqrt(5.0)
    cos_ang = 1.0 / np.sqrt(5.0)
    north = np.array([0., 0., 1.])
    south = np.array([0., 0., -1.])
    upper = [np.array([sin_ang * np.cos(k * 0.4 * np.pi),
                        sin_ang * np.sin(k * 0.4 * np.pi), cos_ang]) for k in range(5)]
    lower = [np.array([sin_ang * np.cos(k * 0.4 * np.pi + 0.2 * np.pi),
                        sin_ang * np.sin(k * 0.4 * np.pi + 0.2 * np.pi), -cos_ang])
             for k in range(5)]
    edges = []
    for k in range(5):
        edges.append((north,    upper[k]))           # N pole to upper ring
        edges.append((south,    lower[k]))           # S pole to lower ring
        edges.append((upper[k], upper[(k+1) % 5]))  # upper ring
        edges.append((lower[k], lower[(k+1) % 5]))  # lower ring
        edges.append((upper[k], lower[k-1]))         # upper to lower (left)
        edges.append((upper[k], lower[k]))           # upper to lower (right)
    return edges


# ── build grids ───────────────────────────────────────────────────────────────

pts2 = build_geodesic_grid(2)
pts4 = build_geodesic_grid(4)


# ── Figure layout: 1 row × 3 columns ─────────────────────────────────────────

fig = plt.figure(figsize=(13, 5))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(
    1, 3,
    figure=fig,
    left=0.02, right=0.98,
    top=0.88, bottom=0.12,
    wspace=0.10,
)

ax3d_lo  = fig.add_subplot(gs[0, 0], projection="3d")
ax3d_hi  = fig.add_subplot(gs[0, 1], projection="3d")
ax_patch = fig.add_subplot(gs[0, 2])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panels (a) and (b): 3-D geodesic spheres
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def arc(A, B, n=20):
    """Great-circle arc from A to B on unit sphere."""
    t = np.linspace(0, 1, n)
    pts = np.outer(1 - t, A) + np.outer(t, B)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


ELEV, AZIM = 22, 30  # shared camera angles

def view_direction():
    """Unit vector pointing from origin toward the camera."""
    e, a = np.radians(ELEV), np.radians(AZIM)
    return np.array([np.cos(e) * np.cos(a),
                     np.cos(e) * np.sin(a),
                     np.sin(e)])

def is_front(mid):
    """Return True if the midpoint faces the camera (dot product > 0)."""
    return np.dot(mid / np.linalg.norm(mid), view_direction()) > 0


def draw_geodesic_3d(ax, pts_unique, nlevel, title, pt_color, front_alpha):
    ax.set_facecolor("white")

    # Transparent sphere surface
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    ax.plot_surface(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color="#AED6F1", alpha=0.15, linewidth=0, antialiased=True, zorder=0)

    # ── thin subdivision edges via convex hull (front hemisphere only) ────────
    hull = ConvexHull(pts_unique)
    drawn = set()
    for simplex in hull.simplices:
        for i in range(3):
            a, b = int(simplex[i]), int(simplex[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            if key in drawn:
                continue
            drawn.add(key)
            mid = (pts_unique[a] + pts_unique[b]) / 2
            if not is_front(mid):
                continue
            seg = arc(pts_unique[a], pts_unique[b])
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                    "-", color="#5D6D7E", lw=0.7, alpha=front_alpha, zorder=2)

    # ── thick icosahedron base edges (front hemisphere only) ──────────────────
    for A, B in icosahedron_edges():
        mid = (A + B) / 2
        if not is_front(mid):
            continue
        seg = arc(A, B, n=40)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                "-", color="black", lw=2.2, alpha=0.92, zorder=4,
                solid_capstyle="round")

    # ── grid points (front hemisphere only) ───────────────────────────────────
    vd = view_direction()
    mask = pts_unique @ vd > 0
    front_pts = pts_unique[mask]
    ax.scatter(front_pts[:, 0], front_pts[:, 1], front_pts[:, 2],
               s=20 if nlevel <= 2 else 7,
               color=pt_color, zorder=5, depthshade=False, linewidths=0)

    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=4)
    ax.view_init(elev=ELEV, azim=AZIM)


draw_geodesic_3d(ax3d_lo, pts2, 2,
                 r"(a) $n_{\rm lev}=2$,  $N_\Omega=42$",
                 pt_color="#C0392B", front_alpha=0.99)
draw_geodesic_3d(ax3d_hi, pts4, 4,
                 r"(b) $n_{\rm lev}=4$,  $N_\Omega=162$",
                 pt_color="#1A5276", front_alpha=0.99)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel (c): Single patch sub-grid, nlev=3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ax_patch.set_facecolor("white")
ax_patch.set_title(r"(c) Single patch, $n_{\rm lev}=3$", fontsize=11)

NL = 3

# Rebuild anorm for nlev=3 to get patch 0 coords
def build_patch0(nlevel):
    sin_ang = 2.0 / np.sqrt(5.0)
    cos_ang = 1.0 / np.sqrt(5.0)
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([sin_ang, 0.0, cos_ang])
    p3 = np.array([sin_ang * np.cos(0.2 * np.pi),  sin_ang * np.sin(0.2 * np.pi),  -cos_ang])
    p4 = np.array([sin_ang * np.cos(-0.4 * np.pi), sin_ang * np.sin(-0.4 * np.pi),  cos_ang])
    p5 = np.array([sin_ang * np.cos(-0.2 * np.pi), sin_ang * np.sin(-0.2 * np.pi), -cos_ang])
    p6 = np.array([0.0, 0.0, -1.0])
    NL = nlevel
    anorm = np.zeros((2 + NL, 2 + 2 * NL, 3))
    row_index = 1
    for l in range(NL):
        col_index = 1
        for m in range(l, NL):
            v = ((m - l + 1) * p2 + (NL - m - 1) * p1 + l * p4) / NL
            anorm[row_index, col_index] = v / np.linalg.norm(v)
            col_index += 1
        for m in range(NL - l, NL):
            v = ((NL - l) * p2 + (m - NL + l + 1) * p5 + (NL - m - 1) * p4) / NL
            anorm[row_index, col_index] = v / np.linalg.norm(v)
            col_index += 1
        for m in range(l, NL):
            v = ((m - l + 1) * p3 + (NL - m - 1) * p2 + l * p5) / NL
            anorm[row_index, col_index] = v / np.linalg.norm(v)
            col_index += 1
        for m in range(NL - l, NL):
            v = ((NL - l) * p3 + (m - NL + l + 1) * p6 + (NL - m - 1) * p5) / NL
            anorm[row_index, col_index] = v / np.linalg.norm(v)
            col_index += 1
        row_index += 1
    return anorm

an3 = build_patch0(NL)

patch_pts = np.zeros((NL, 2 * NL, 2))
for l in range(NL):
    for m in range(2 * NL):
        v = an3[l + 1, m + 1]
        patch_pts[l, m, 0] = np.degrees(np.arctan2(v[1], v[0]))
        patch_pts[l, m, 1] = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))

for l in range(NL):
    for m in range(2 * NL):
        A = patch_pts[l, m]
        if m + 1 < 2 * NL:
            B = patch_pts[l, m + 1]
            ax_patch.plot([A[0], B[0]], [A[1], B[1]], "-", color="#2C3E50", lw=1.0, alpha=1)
        if l + 1 < NL:
            C = patch_pts[l + 1, m]
            ax_patch.plot([A[0], C[0]], [A[1], C[1]], "-", color="#2C3E50", lw=1.0, alpha=1)
            if m + 1 < 2 * NL:
                D = patch_pts[l + 1, m + 1]
                ax_patch.plot([A[0], D[0]], [A[1], D[1]], "--", color="#7F8C8D", lw=0.6, alpha=0.5)

for l in range(NL):
    for m in range(2 * NL):
        A = patch_pts[l, m]
        ax_patch.scatter(A[0], A[1], s=45,
                         color=plt.cm.plasma(l / (NL - 0.5)),
                         zorder=5, linewidths=0.4, edgecolors="black")

ax_patch.set_xlabel(r"Longitude $\phi$ [deg]", fontsize=10)
ax_patch.set_ylabel(r"Latitude $\theta$ [deg]", fontsize=10)
ax_patch.tick_params(labelsize=8)

A = patch_pts[1, 3]
B = patch_pts[1, 4]
C = patch_pts[2, 3]
cx, cy = (A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3
ax_patch.annotate("cell\n(Voronoi)", xy=(cx, cy),
                  xytext=(cx + 5, cy - 5),
                  arrowprops=dict(arrowstyle="->", color="gray", lw=0.9),
                  fontsize=8, color="#555555",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.7))


# ── global title and save ─────────────────────────────────────────────────────

fig.suptitle(
    r"Geodesic sphere discretization:  $N_\Omega = 10\,n_{\rm lev}^2 + 2$ angular zones"
    "\n"
    r"Five-patch icosahedral grid  [SphericalGrid, AthenaK]",
    fontsize=13, y=1.01,
)

outfile = "/home/ykim7/athenak/geodesic_schematic.pdf"
fig.savefig(outfile, dpi=180, bbox_inches="tight")
fig.savefig(outfile.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved  {outfile}")
