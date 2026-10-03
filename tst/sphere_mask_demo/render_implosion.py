#!/usr/bin/env python3
"""
Render an MP4 of the density field for the 300x300, ppm4/hllc, radius=0.075 octant-domain
[0,0.3]^2 sphere_mask implosion demo, five variants side by side: baseline (no sphere_mask),
dirichlet1 (interior fixed to dens=1.0/pres=1.0/vel=0, matching the implosion's exterior
state), dirichlet2 (same but vel1=2.0), reflecting, absorbing.

Usage (after `run_demo.py implosion` has produced output):
    python3 tst/sphere_mask_demo/render_implosion.py

Reads:
    tst/sphere_mask_demo/output_<variant>/bin/*.bin
Writes:
    tst/sphere_mask_demo/sphere_mask_implosion.mp4
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "tst", "test_suite", "nr"))
import sphere_mask_utils as smu  # noqa: E402

RADIUS = 0.075
VARIANTS = ["baseline", "dirichlet1", "dirichlet2", "reflecting", "absorbing"]
TITLES = {
    "baseline": "Baseline (no sphere_mask)",
    "dirichlet1": "Dirichlet 1 (dens=1.0,pres=1.0,vel=0)",
    "dirichlet2": "Dirichlet 2 (dens=1.0,pres=1.0,vel1=2.0)",
    "reflecting": "Reflecting",
    "absorbing": "Absorbing",
}


def list_bins(variant):
    d = os.path.join(HERE, f"output_{variant}", "bin")
    return sorted(glob.glob(os.path.join(d, "*.hydro_w.*.bin")))


def main():
    bin_files = {bc: list_bins(bc) for bc in VARIANTS}
    for bc, files in bin_files.items():
        print(f"{bc}: {len(files)} frames")
    nframes = min(len(f) for f in bin_files.values())
    if nframes == 0:
        sys.exit("no .bin output found -- run `run_demo.py implosion` first")

    # Per-variant fixed colorbar ranges. Baseline has no masked interior, so its
    # range is set to cover the implosion's own d_in=0.125 -- d_out=1.0 range instead.
    VRANGE = {
        "baseline": (0.1, 1.5),
        "dirichlet1": (0.5, 1.3),
        "dirichlet2": (0.5, 3.0),
        "reflecting": (0.5, 1.3),
        "absorbing": (0.5, 1.3),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes.flat[-1].axis("off")
    images = []
    for ax, bc in zip(axes.flat, VARIANTS):
        d0 = smu.read_bin(bin_files[bc][0])
        m, x, y, dx, dy = smu.origin_meshblock(d0)
        extent = [x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2]
        dens0 = d0["mb_data"]["dens"][m][0]
        vmin, vmax = VRANGE[bc]
        im = ax.imshow(dens0, origin="lower", extent=extent, cmap="viridis",
                        vmin=vmin, vmax=vmax)
        if bc != "baseline":
            ax.add_patch(Circle((0, 0), RADIUS, fill=False, edgecolor="red",
                                 linewidth=1.5, linestyle="--"))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(TITLES[bc])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="density")
        images.append((im, m))

    suptitle = fig.suptitle("")
    fig.tight_layout()

    writer = FFMpegWriter(fps=20, metadata={"title": "sphere_mask implosion demo"})
    outpath = os.path.join(HERE, "sphere_mask_implosion.mp4")
    with writer.saving(fig, outpath, dpi=130):
        for f in range(nframes):
            t = None
            for (im, m), bc in zip(images, VARIANTS):
                d = smu.read_bin(bin_files[bc][f])
                t = d["time"]
                im.set_data(d["mb_data"]["dens"][m][0])
            suptitle.set_text(
                f"LW implosion, [0,0.3]^2, ppm4/hllc, 300x300, sphere_mask "
                f"radius={RADIUS}, t={t:.3f}")
            writer.grab_frame()
            if f % 20 == 0:
                print(f"frame {f}/{nframes}")

    print("saved", outpath)


if __name__ == "__main__":
    main()
