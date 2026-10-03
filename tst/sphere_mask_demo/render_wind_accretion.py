#!/usr/bin/env python3
"""
Render an MP4 comparing the Spherical Wind sphere_mask BC used two ways: outward wind
(vel_r=+1.0) vs. inward accretion (vel_r=-1.0), everything else held fixed. Both runs
use the spherical blast-wave pgen (src/pgen/fluids/blast.cpp) purely as a way to set up
a uniform ambient medium at rest -- outer_radius is smaller than the sphere_mask radius,
so the blast's own perturbation is entirely swallowed by the mask and never reaches the
exterior flow (see inputs/hydro/sphere_mask_wind_demo.athinput).

Top row: density. Bottom row: radial velocity v_r = (x*vx + y*vy)/r (diverging colormap,
red=outward/blue=inward), which makes the wind-driven forward shock and the converging
accretion flow immediately visually distinct.

Usage (after `run_demo.py wind_accretion` has produced output):
    python3 tst/sphere_mask_demo/render_wind_accretion.py

Reads:
    tst/sphere_mask_demo/output_<variant>/bin/*.bin
Writes:
    tst/sphere_mask_demo/sphere_mask_wind_accretion.mp4
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

RADIUS = 0.05
VARIANTS = ["wind", "accretion"]
TITLES = {"wind": "Spherical Wind (vel_r=+1.0)", "accretion": "Accretion (vel_r=-1.0)"}
# Both density panels share one color scale (covers the full range seen across both
# runs/frames) so the two variants are directly comparable -- this makes it visually
# obvious just how much larger the wind's density perturbation is than the accretion's.
DENS_RANGE_SHARED = (0.4, 1.5)
VR_RANGE = {"wind": (-1.8, 1.8), "accretion": (-1.1, 1.1)}


def list_bins(variant):
    d = os.path.join(HERE, f"output_{variant}", "bin")
    return sorted(glob.glob(os.path.join(d, "*.hydro_w.*.bin")))


def radial_velocity(d, m, x, y):
    vx = d["mb_data"]["velx"][m][0]
    vy = d["mb_data"]["vely"][m][0]
    xx, yy = np.meshgrid(x, y)
    rr = np.maximum(np.sqrt(xx**2 + yy**2), 1e-12)
    return (xx*vx + yy*vy)/rr


def main():
    bin_files = {bc: list_bins(bc) for bc in VARIANTS}
    for bc, files in bin_files.items():
        print(f"{bc}: {len(files)} frames")
    nframes = min(len(f) for f in bin_files.values())
    if nframes == 0:
        sys.exit("no .bin output found -- run `run_demo.py wind_accretion` first")

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    dens_images, vr_images = [], []
    for col, bc in enumerate(VARIANTS):
        d0 = smu.read_bin(bin_files[bc][0])
        m, x, y, dx, dy = smu.origin_meshblock(d0)
        extent = [x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2]

        ax = axes[0, col]
        dens0 = d0["mb_data"]["dens"][m][0]
        vmin, vmax = DENS_RANGE_SHARED
        im = ax.imshow(dens0, origin="lower", extent=extent, cmap="viridis",
                        vmin=vmin, vmax=vmax)
        ax.add_patch(Circle((0, 0), RADIUS, fill=False, edgecolor="red",
                             linewidth=1.5, linestyle="--"))
        ax.set_title(TITLES[bc])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="density")
        dens_images.append((im, m))

        ax = axes[1, col]
        vr0 = radial_velocity(d0, m, x, y)
        vrmin, vrmax = VR_RANGE[bc]
        imv = ax.imshow(vr0, origin="lower", extent=extent, cmap="RdBu_r",
                         vmin=vrmin, vmax=vrmax)
        ax.add_patch(Circle((0, 0), RADIUS, fill=False, edgecolor="black",
                             linewidth=1.5, linestyle="--"))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(imv, ax=ax, fraction=0.046, pad=0.04, label="v_r")
        vr_images.append((imv, m, x, y))

    suptitle = fig.suptitle("")
    fig.tight_layout()

    writer = FFMpegWriter(fps=15, metadata={"title": "sphere_mask wind/accretion demo"})
    outpath = os.path.join(HERE, "sphere_mask_wind_accretion.mp4")
    with writer.saving(fig, outpath, dpi=130):
        for f in range(nframes):
            t = None
            for (im, m), bc in zip(dens_images, VARIANTS):
                d = smu.read_bin(bin_files[bc][f])
                t = d["time"]
                im.set_data(d["mb_data"]["dens"][m][0])
            for (imv, m, x, y), bc in zip(vr_images, VARIANTS):
                d = smu.read_bin(bin_files[bc][f])
                imv.set_data(radial_velocity(d, m, x, y))
            suptitle.set_text(
                f"Uniform ambient (dens=1, pres=1) + spherical_wind sphere_mask "
                f"radius={RADIUS}, t={t:.3f}")
            writer.grab_frame()
            if f % 10 == 0:
                print(f"frame {f}/{nframes}")

    print("saved", outpath)


if __name__ == "__main__":
    main()
