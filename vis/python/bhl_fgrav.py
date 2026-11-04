#! /usr/bin/env python

"""
Cumulative gravitational force fgrav(<r) of the gas on the GM=1 point mass, from
the bin dumps of the bhl_wind_tunnel problem generator.

For each dump, at each radius r on a grid of linear bin edges,

    fgrav_d(<r) = sum over cells with r_min <= r_cell < r of dens dV x_d / r_cell^3,

with x and r_cell the cell center, as in CellSums in
src/pgen/tests/bhl_wind_tunnel.cpp. By default r_min is <sphere_mask>/radius,
so the integral covers the gas outside the mask, and the largest r is the
distance from the point mass to the nearest domain face that is not a mirror
plane, so every shell is complete. Edit NBINS, R_MIN and R_MAX below to change
the grid.

If the mesh covers only half (or a quarter, ...) of the domain behind reflecting
faces at x_d = 0, the sums are those of the full domain, as in the user history
output: they are multiplied by the number of mirror images, and the force
component normal to each mirror plane is zero.

Writes one table per dump, <dump>.fgrav.txt (columns r, fgrav1, fgrav2,
fgrav3), and one figure, <basename>.fgrav.png, with the curves of all dumps
colored by time.

Usage:
  python bhl_fgrav.py run_dir/bin/BHLWindTunnel3D.hydro_w.*.bin [-o OUTDIR] [-j N]

The dumps must hold full MeshBlocks (no slices, no ghost zones) with a 'dens'
variable (hydro_w or hydro_u), one file per dump. Only the density of the
MeshBlocks that reach the radius grid is read from each file. Dumps are single
precision, so results agree with the double-precision .hst columns to ~1e-6
relative.
"""

# Python modules
import argparse
import concurrent.futures
import contextlib
import functools
import os

import numpy as np

# Radius grid: NBINS linear bins from R_MIN to R_MAX. None selects the default:
# R_MIN = <sphere_mask>/radius, R_MAX = distance to the nearest non-mirror face.
NBINS = 100
R_MIN = None
R_MAX = None

# Cells processed at once; bounds the memory of the temporary arrays (about
# 100 bytes per cell).
CHUNK_CELLS = 1 << 21


def get_param(header, block, key):
    """Return <block>/key from the input file stored in a bin header."""
    current = None
    for line in header:
        if line.startswith("<"):
            current = line.strip("<>")
            continue
        if current == block and "=" in line:
            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip()
    raise KeyError(f"no parameter <{block}>/{key} in the bin header")


def read_dump(filename):
    """
    Read a bin dump (layout as in src/outputs/binary.cpp): the input file stored
    in its header, the time and cycle, and the bounds and density of each
    MeshBlock. The density is a read-only memory map of the file, so only the
    MeshBlocks that are indexed are read, and the other variables never are.
    """
    with open(filename, "rb") as fp:
        if fp.readline().split()[-1:] != [b"version=1.1"]:
            raise TypeError(f"{filename} is not an AthenaK bin file, version 1.1")
        npreheader = int(fp.readline().split(b"=")[-1])
        preheader = {}
        for _ in range(npreheader - 1):
            key, value = fp.readline().decode().split("=")
            preheader[key.strip()] = value.strip()
        nvars = int(fp.readline().split(b"=")[-1])
        var_names = fp.readline().decode().split()[1:]
        header_size = int(fp.readline().split(b"=")[-1])
        header = [
            line.split("#")[0].strip()
            for line in fp.read(header_size).decode().split("\n")
        ]
        header = [line for line in header if line]
        offset = fp.tell()
        first_index = np.frombuffer(fp.read(24), dtype=np.int32)

    if "dens" not in var_names:
        raise ValueError(
            f"{filename} has no 'dens' variable (it has {' '.join(var_names)}); "
            "output hydro_w or hydro_u"
        )
    nx = [int(get_param(header, "meshblock", f"nx{d + 1}")) for d in range(3)]
    if nx[2] == 1:
        raise ValueError("bhl_fgrav.py requires a 3D mesh")
    if len(first_index) != 6:
        raise ValueError(f"{filename} holds no MeshBlocks")

    # Every MeshBlock record has the same size: 10 int32, 6 coordinates, then
    # nvars arrays of the output cells.
    nout = first_index[1::2] - first_index[0::2] + 1
    loc_type = np.float64 if preheader["size of location"] == "8" else np.float32
    var_type = np.float64 if preheader["size of variable"] == "8" else np.float32
    record = np.dtype([
        ("index", np.int32, 6),
        ("logical", np.int32, 4),
        ("geom", loc_type, 6),
        ("data", var_type, (nvars, nout[2], nout[1], nout[0])),
    ])
    nmb, extra = divmod(os.path.getsize(filename) - offset, record.itemsize)
    if extra != 0:
        raise ValueError(f"{filename} does not hold a whole number of MeshBlocks")
    blocks = np.memmap(filename, dtype=record, mode="r", offset=offset, shape=(nmb,))

    nghost = int(get_param(header, "mesh", "nghost"))
    whole = [0, nx[0] - 1, 0, nx[1] - 1, 0, nx[2] - 1]
    if np.any(blocks["index"] - nghost != whole):
        raise ValueError(
            f"{filename} holds partial MeshBlocks (a slice, or ghost zones); "
            "output whole MeshBlocks without ghost zones"
        )

    return {
        "header": header,
        "time": float(preheader["time"]),
        "cycle": int(preheader["cycle"]),
        # x1min, x1max, x2min, x2max, x3min, x3max of each MeshBlock
        "geom": np.array(blocks["geom"], dtype=np.float64),
        # [MeshBlock, k, j, i], not yet read
        "dens": blocks["data"][:, var_names.index("dens")],
    }


def mesh_folds(header):
    """
    Mirror planes of the mesh, as flux_fold in the problem generator: +1 (-1) if
    the mesh covers x_d >= 0 (x_d <= 0) behind a reflecting face at x_d = 0,
    else 0. Also returns the distance from the origin to the nearest face that
    is not a mirror plane.
    """
    folds = [0, 0, 0]
    extent = np.inf
    for d in range(3):
        lo = float(get_param(header, "mesh", f"x{d + 1}min"))
        hi = float(get_param(header, "mesh", f"x{d + 1}max"))
        fold_lo = lo == 0.0 and get_param(header, "mesh", f"ix{d + 1}_bc") == "reflect"
        fold_hi = hi == 0.0 and get_param(header, "mesh", f"ox{d + 1}_bc") == "reflect"
        folds[d] = 1 if fold_lo else (-1 if fold_hi else 0)
        if not fold_lo:
            extent = min(extent, -lo)
        if not fold_hi:
            extent = min(extent, hi)
    return folds, extent


def radius_edges(header):
    """Linear bin edges from R_MIN to R_MAX, with the defaults filled in."""
    r_mask = float(get_param(header, "sphere_mask", "radius"))
    softening = float(get_param(header, "hydro_srcterms", "gravity_softening_length"))
    _, extent = mesh_folds(header)
    r_min = r_mask if R_MIN is None else R_MIN
    r_max = extent if R_MAX is None else R_MAX
    if r_min < softening:
        raise ValueError(
            f"R_MIN = {r_min} is inside the softening length {softening}, where "
            "the force is not 1/r^2"
        )
    if not r_max > r_min:
        raise ValueError(f"R_MAX = {r_max} must exceed R_MIN = {r_min}")
    return np.linspace(r_min, r_max, NBINS + 1)


def cumulative_fgrav(dump, edges):
    """
    Return fgrav[k, d] = fgrav_d(<edges[k]), with fgrav[0] = 0, summed over the
    cells with edges[0] <= r_cell < edges[k]; edges must be increasing, with
    edges[0] > 0.
    """
    geom = dump["geom"]
    dens = dump["dens"]
    nbins = len(edges) - 1
    n3, n2, n1 = dens.shape[1:]

    # Only MeshBlocks with a cell center in edges[0] <= r < edges[-1] are read.
    # The centers lie within the MeshBlock bounds, so within [rnear, rfar] of the
    # origin; the slack keeps any MeshBlock that roundoff could put in range.
    lo, hi = geom[:, 0::2], geom[:, 1::2]
    rnear2 = np.sum(np.clip(0.0, lo, hi) ** 2, axis=1)
    rfar2 = np.sum(np.maximum(lo**2, hi**2), axis=1)
    used = np.flatnonzero(
        (rnear2 < (1.0 + 1e-10) * edges[-1] ** 2)
        & (rfar2 > (1.0 - 1e-10) * edges[0] ** 2)
    )

    # Shell sums; cells outside the grid go to the extra bin nbins, dropped below.
    shells = np.zeros((3, nbins + 1))
    frac = [(np.arange(n) + 0.5) / n for n in (n1, n2, n3)]
    shapes = [(-1, 1, 1, n1), (-1, 1, n2, 1), (-1, n3, 1, 1)]
    chunk = max(1, CHUNK_CELLS // (n1 * n2 * n3))
    for start in range(0, len(used), chunk):
        mbs = used[start:start + chunk]
        # cell centers as CellCenterX in cell_locations.hpp, each shaped to
        # broadcast over [MeshBlock, k, j, i]
        x = []
        for d in range(3):
            xmin = geom[mbs, 2 * d, None]
            xmax = geom[mbs, 2 * d + 1, None]
            xc = (frac[d] * xmax - frac[d] * xmin) - (0.5 * xmax - 0.5 * xmin) \
                + (0.5 * xmin + 0.5 * xmax)
            x.append(xc.reshape(shapes[d]))
        vol = np.prod((hi[mbs] - lo[mbs]) / (n1, n2, n3), axis=1).reshape(-1, 1, 1, 1)
        r2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2
        r = np.sqrt(r2)
        # bin k holds edges[k] <= r < edges[k+1]
        ibin = np.searchsorted(edges, r.ravel(), side="right") - 1
        ibin[ibin < 0] = nbins
        # a cell center at the origin is outside the grid (edges[0] > 0), so its
        # inf and nan go to the dropped bin
        with np.errstate(divide="ignore", invalid="ignore"):
            fac = vol * dens[mbs] / (r2 * r)
            for d in range(3):
                shells[d] += np.bincount(
                    ibin, weights=(fac * x[d]).ravel(), minlength=nbins + 1
                )

    folds, _ = mesh_folds(dump["header"])
    nimages = 2 ** sum(f != 0 for f in folds)
    fgrav = np.zeros((nbins + 1, 3))
    fgrav[1:] = nimages * np.cumsum(shells[:, :nbins].T, axis=0)
    for d in range(3):
        if folds[d] != 0:
            fgrav[:, d] = 0.0
    return fgrav


def write_table(filename, binfile, dump, edges, fgrav):
    r_min = edges[0]
    with open(filename, "w") as f:
        f.write(
            f"# Cumulative gravitational force on the GM=1 point mass from {binfile}\n"
        )
        f.write(f"# time = {dump['time']:.10e}, cycle = {dump['cycle']}\n")
        f.write(
            f"# fgrav_d(<r) = sum of dens dV x_d / r_cell^3 over cells with "
            f"{r_min:g} <= r_cell < r\n"
        )
        f.write("# [1]=r [2]=fgrav1 [3]=fgrav2 [4]=fgrav3\n")
        np.savetxt(f, np.column_stack([edges, fgrav]), fmt="%.10e")


def process_dump(binfile, edges, outdir):
    """Write the table of one dump; return its time, the curve and the table name."""
    dump = read_dump(binfile)
    fgrav = cumulative_fgrav(dump, edges)
    tabdir = os.path.dirname(binfile) if outdir is None else outdir
    stem = os.path.basename(binfile)
    if stem.endswith(".bin"):
        stem = stem[: -len(".bin")]
    tabfile = os.path.join(tabdir, stem + ".fgrav.txt")
    write_table(tabfile, binfile, dump, edges, fgrav)
    return dump["time"], fgrav, tabfile


def plot_curves(filename, times, edges, curves, folds):
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    tmin, tmax = min(times), max(times)
    if tmin == tmax:
        tmin, tmax = tmin - 0.5, tmax + 0.5
    norm = matplotlib.colors.Normalize(vmin=tmin, vmax=tmax)
    cmap = matplotlib.colormaps["viridis"]
    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(6.4, 8.0), layout="constrained"
    )
    for d, ax in enumerate(axes):
        for t, fgrav in zip(times, curves):
            ax.plot(edges, fgrav[:, d], color=cmap(norm(t)), lw=1.0)
        ax.axhline(0.0, color="0.6", lw=0.5)
        ax.set_ylabel(rf"$F_{{\mathrm{{grav}},{d + 1}}}(<r)$")
        if folds[d] != 0:
            ax.text(
                0.5, 0.5, rf"zero by symmetry ($x_{d + 1}=0$ mirror plane)",
                transform=ax.transAxes, ha="center", va="center", color="0.4",
            )
    axes[-1].set_xlabel(r"$r$")
    axes[-1].set_xlim(edges[0], edges[-1])
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="time"
    )
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main(**kwargs):
    binfiles = kwargs["files"]
    outdir = kwargs["outdir"]
    jobs = kwargs["jobs"]

    # the radius grid and mirror planes are fixed by the mesh, so the first dump
    # sets them for all
    header = read_dump(binfiles[0])["header"]
    edges = radius_edges(header)
    folds, _ = mesh_folds(header)
    basename = get_param(header, "job", "basename")

    work = functools.partial(process_dump, edges=edges, outdir=outdir)
    times = []
    curves = []
    parallel = jobs > 1
    with concurrent.futures.ProcessPoolExecutor(jobs) if parallel else \
            contextlib.nullcontext() as pool:
        results = pool.map(work, binfiles) if parallel else map(work, binfiles)
        for binfile, (time, fgrav, tabfile) in zip(binfiles, results):
            print(f"{binfile}: t = {time:.5g}, "
                  f"fgrav(<{edges[-1]:.4g}) = {fgrav[-1]} -> {tabfile}")
            times.append(time)
            curves.append(fgrav)

    order = np.argsort(times)
    times = [times[n] for n in order]
    curves = [curves[n] for n in order]
    plotdir = os.path.dirname(binfiles[0]) if outdir is None else outdir
    plotfile = os.path.join(plotdir, basename + ".fgrav.png")
    plot_curves(plotfile, times, edges, curves, folds)
    print(f"plot -> {plotfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cumulative gravitational force fgrav(<r) of the gas on the "
        "point mass in bhl_wind_tunnel bin dumps."
    )
    parser.add_argument("files", nargs="+", help="bin dumps, one file per dump")
    parser.add_argument(
        "-o", "--outdir", default=None,
        help="directory for the tables and the plot (default: next to the dumps)",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="number of dumps processed in parallel (default: 1)",
    )
    main(**vars(parser.parse_args()))
