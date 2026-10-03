#!/usr/bin/env python3
"""
Ad-hoc (non-pytest) demo combining the point-particle 1/r gravity source term
(<hydro_srcterms>/point_particle_gravity_at_center, src/srcterms/srcterms.cpp) with the
spherical inner mask (<sphere_mask>, src/hydro/hydro_sphere_mask.cpp).

Runs inputs/hydro/sphere_mask_gravity_blast.athinput as-is (edit that file, or pass
command-line overrides below, to change the test) and plots density + radial velocity
at a few snapshots so the gravitational infall and its reflection off the masked sphere
are visible directly. This is a qualitative demo, not a regression test.

Usage (from anywhere; build build_blast first -- see README.md in this directory):
    python3 tst/sphere_mask_demo/gravity_blast.py [--tlim 1.0] [extra overrides...]

Any extra arguments are passed straight through to athena as additional input
overrides, e.g.:
    python3 tst/sphere_mask_demo/gravity_blast.py sphere_mask/radius=0.3

Output: tst/sphere_mask_demo/output_gravity_blast/bin/*.bin (gitignored, *.bin) and
        tst/sphere_mask_demo/sphere_mask_gravity_blast.png
"""

import argparse
import glob
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "tst", "test_suite", "nr"))
import sphere_mask_utils as smu  # noqa: E402

INPUT_FILE = os.path.join(REPO_ROOT, "inputs", "hydro", "sphere_mask_gravity_blast.athinput")
DEFAULT_ATHENA = os.path.join(REPO_ROOT, "build_blast_omp", "src", "athena")


def run(tlim, athena, demo_dir, out_dir, extra_overrides, num_threads):
    bin_dir = os.path.join(out_dir, "bin")
    for f in glob.glob(os.path.join(bin_dir, "*.hydro_w.*.bin")):
        os.remove(f)
    cmd = [athena, "-i", INPUT_FILE]
    if tlim is not None:
        cmd.append(f"time/tlim={tlim}")
    cmd += ["-d", out_dir] + extra_overrides
    if num_threads is not None:
        cmd.append(f"--kokkos-num-threads={num_threads}")
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=demo_dir)
    return sorted(glob.glob(os.path.join(bin_dir, "*.hydro_w.*.bin")))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tlim", type=float, default=None,
                         help="override time/tlim (default: whatever the input file says)")
    parser.add_argument("--radius", type=float, default=None,
                         help="override sphere_mask/radius (also used to draw the circle)")
    parser.add_argument("--athena", default=DEFAULT_ATHENA,
                         help="path to a build_blast-configured athena executable")
    parser.add_argument("--num-threads", type=int, default=os.cpu_count(),
                         help="Kokkos OpenMP threads (--kokkos-num-threads); "
                              "requires an OpenMP-enabled build (default: build_blast_omp, "
                              "all cores)")
    parser.add_argument("overrides", nargs="*",
                         help="extra athinput overrides passed through to athena")
    args = parser.parse_args()

    if not os.path.isfile(args.athena):
        msg = (f"athena executable not found at {args.athena} -- build it first, see "
               f"README.md in this directory, or pass --athena")
        sys.exit(msg)

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(demo_dir, "output_gravity_blast")
    os.makedirs(out_dir, exist_ok=True)

    extra_overrides = list(args.overrides)
    if args.radius is not None:
        extra_overrides.append(f"sphere_mask/radius={args.radius}")

    binfiles = run(args.tlim, args.athena, demo_dir, out_dir, extra_overrides, args.num_threads)

    # radius used to draw the mask circle: explicit --radius, else parsed from the
    # input file's <sphere_mask>/radius line
    radius = args.radius
    if radius is None:
        with open(INPUT_FILE) as f:
            in_block = False
            for line in f:
                s = line.strip()
                if s.startswith("<"):
                    in_block = s.startswith("<sphere_mask>")
                    continue
                if in_block and s.startswith("radius"):
                    radius = float(s.split("=")[1].split("#")[0].strip())
                    break

    # pick 4 snapshots spread through the run: t=0 and 3 more evenly spaced through
    # the rest, biased toward the end where the infall/shock has developed
    n = len(binfiles)
    idxs = sorted(set([0] + [round(f * (n - 1)) for f in (1/3, 2/3, 1.0)]))
    dumps = [smu.read_bin(binfiles[i]) for i in idxs]

    fig, axes = plt.subplots(1, len(dumps), figsize=(5*len(dumps), 5))
    if len(dumps) == 1:
        axes = [axes]
    dens_max = max(d["mb_data"]["dens"][smu.origin_meshblock(d)[0]][0].max() for d in dumps)
    for ax, d in zip(axes, dumps):
        m, x, y, dx, dy = smu.origin_meshblock(d)
        dens = d["mb_data"]["dens"][m][0]
        extent = [x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2]
        im = ax.imshow(dens, origin="lower", extent=extent, cmap="inferno",
                        vmin=0.0, vmax=dens_max)
        circle = Circle((0, 0), radius, fill=False, edgecolor="cyan",
                         linewidth=1.5, linestyle="--")
        ax.add_patch(circle)
        ax.set_title(f"t={d['time']:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="density")

    fig.suptitle("Point-particle 1/r gravity + spherical inner mask "
                  f"(radius={radius:g}, reflecting), 2D blast pgen, uniform ambient")
    fig.tight_layout()
    outpath = os.path.join(demo_dir, "sphere_mask_gravity_blast.png")
    fig.savefig(outpath, dpi=130)
    print("saved", outpath)


if __name__ == "__main__":
    main()
