"""
Helpers for the sphere_mask regression tests.

Includes a minimal, dependency-free reader for AthenaK's native ".bin" MeshBlock dumps
(a trimmed copy of the parsing logic in vis/python/bin_convert.py's read_binary(), which
cannot be imported directly here since that module unconditionally imports h5py), plus
a bilinear/trilinear mirror-point re-implementation used to independently check the
Reflecting/Absorbing sphere_mask output against the same physics the C++ kernel uses,
and a MeshBlock stitcher used to compare runs that differ only in <meshblock> size.
"""

import numpy as np


def read_bin(filename):
    """Read one AthenaK .bin MeshBlock dump into a dict of per-MeshBlock arrays."""
    filedata = {}
    with open(filename, "rb") as fp:
        fp.seek(0, 2)
        filesize = fp.tell()
        fp.seek(0, 0)

        code_header = fp.readline().split()
        if len(code_header) < 1 or code_header[0] != b"Athena":
            raise TypeError("unknown or unsupported AthenaK .bin file format")

        pheader_count = int(fp.readline().split(b"=")[-1])
        pheader = {}
        for _ in range(pheader_count - 1):
            key, val = [x.strip() for x in fp.readline().decode("utf-8").split("=")]
            pheader[key] = val
        time = float(pheader["time"])
        locsizebytes = int(pheader["size of location"])
        varsizebytes = int(pheader["size of variable"])

        nvars = int(fp.readline().split(b"=")[-1])
        var_list = [v.decode("utf-8") for v in fp.readline().split()[1:]]
        header_size = int(fp.readline().split(b"=")[-1])
        fp.read(header_size)  # skip <mesh>/<meshblock>/etc. input-file header block

        locfmt = np.float64 if locsizebytes == 8 else np.float32
        varfmt = np.float64 if varsizebytes == 8 else np.float32

        mb_geometry = []
        mb_data = {var: [] for var in var_list}
        while fp.tell() < filesize:
            mb_index = np.frombuffer(fp.read(24), dtype=np.int32)
            nx1_out = (mb_index[1] - mb_index[0]) + 1
            nx2_out = (mb_index[3] - mb_index[2]) + 1
            nx3_out = (mb_index[5] - mb_index[4]) + 1
            fp.read(16)  # mb_logical (gid/level/lx1/lx2/lx3-ish), unused here
            mb_geometry.append(np.frombuffer(fp.read(6*locsizebytes), dtype=locfmt))
            count = nx1_out*nx2_out*nx3_out*nvars
            data = np.fromfile(fp, dtype=varfmt, count=count)
            data = data.reshape(nvars, nx3_out, nx2_out, nx1_out)
            for vari, var in enumerate(var_list):
                mb_data[var].append(data[vari])

    filedata["time"] = time
    filedata["var_names"] = var_list
    filedata["n_mbs"] = len(mb_geometry)
    filedata["mb_geometry"] = np.array(mb_geometry)  # [x1min,x1max,x2min,x2max,..]
    filedata["mb_data"] = mb_data
    return filedata


def origin_meshblock(filedata):
    """Return (mb_index, x, y, dx, dy) for the MeshBlock whose active zone contains the
    coordinate origin, using the same half-open [min,max) convention as the C++ code."""
    geom = filedata["mb_geometry"]
    for m in range(filedata["n_mbs"]):
        x1min, x1max, x2min, x2max = geom[m, 0], geom[m, 1], geom[m, 2], geom[m, 3]
        if x1min <= 0.0 < x1max and x2min <= 0.0 < x2max:
            nx2, nx1 = filedata["mb_data"][filedata["var_names"][0]][m].shape[-2:]
            dx = (x1max - x1min)/nx1
            dy = (x2max - x2min)/nx2
            x = x1min + (np.arange(nx1)+0.5)*dx
            y = x2min + (np.arange(nx2)+0.5)*dy
            return m, x, y, dx, dy
    raise AssertionError("no MeshBlock in this .bin dump contains the origin")


def bilinear(arr2d, x0, dx, y0, dy, xm, ym, radius=None):
    """Bilinear-interpolate a cell-centered 2D array at (xm,ym). If radius is given,
    exclude stencil centers inside that radius and renormalize the remaining weights."""
    ny, nx = arr2d.shape
    xi = (xm - x0)/dx
    yi = (ym - y0)/dy
    i0 = int(np.floor(xi))
    j0 = int(np.floor(yi))
    fx = xi-i0
    fy = yi-j0
    i0 = min(max(i0, 0), nx-2)
    j0 = min(max(j0, 0), ny-2)
    value = 0.0
    weight_sum = 0.0
    for dj in (0, 1):
        for di in (0, 1):
            xc = x0 + (i0+di)*dx
            yc = y0 + (j0+dj)*dy
            wx = (1.0-fx) if di == 0 else fx
            wy = (1.0-fy) if dj == 0 else fy
            if radius is not None:
                if np.hypot(xc, yc) < radius:
                    continue
            weight = wx*wy
            if weight == 0.0:
                continue
            value += weight*arr2d[j0+dj, i0+di]
            weight_sum += weight
    assert weight_sum > 0.0, "mirror stencil has no exterior cell"
    return value/weight_sum


def expected_mirror_state(dens, velx, vely, eint, x0, dx, y0, dy, x, y, radius, bc):
    """Re-implements the Hydro::MaskSphere mirror-point formula (see
    src/hydro/hydro_sphere_mask.cpp) in Python, for cross-checking one interior cell."""
    r = np.hypot(x, y)
    r_safe = max(r, 1e-12*radius)
    s = (2.0*radius - r_safe)/r_safe
    xm, ym = x*s, y*s
    dm = bilinear(dens, x0, dx, y0, dy, xm, ym, radius)
    vxm = bilinear(velx, x0, dx, y0, dy, xm, ym, radius)
    vym = bilinear(vely, x0, dx, y0, dy, xm, ym, radius)
    em = bilinear(eint, x0, dx, y0, dy, xm, ym, radius)
    if bc == "reflecting":
        rhatx, rhaty = x/r_safe, y/r_safe
        vdotr = vxm*rhatx + vym*rhaty
        vx = vxm - 2*vdotr*rhatx
        vy = vym - 2*vdotr*rhaty
    else:  # absorbing
        vx, vy = 0.0, 0.0
    return dm, vx, vy, em


def origin_meshblock_3d(filedata):
    """3D analog of origin_meshblock(): returns (m, x, y, z, dx, dy, dz) for the
    MeshBlock whose active zone contains the origin along all three axes."""
    geom = filedata["mb_geometry"]
    for m in range(filedata["n_mbs"]):
        x1min, x1max = geom[m, 0], geom[m, 1]
        x2min, x2max = geom[m, 2], geom[m, 3]
        x3min, x3max = geom[m, 4], geom[m, 5]
        if x1min <= 0.0 < x1max and x2min <= 0.0 < x2max and x3min <= 0.0 < x3max:
            nz, ny, nx = filedata["mb_data"][filedata["var_names"][0]][m].shape
            dx = (x1max - x1min)/nx
            dy = (x2max - x2min)/ny
            dz = (x3max - x3min)/nz
            x = x1min + (np.arange(nx)+0.5)*dx
            y = x2min + (np.arange(ny)+0.5)*dy
            z = x3min + (np.arange(nz)+0.5)*dz
            return m, x, y, z, dx, dy, dz
    raise AssertionError("no MeshBlock in this .bin dump contains the origin")


def trilinear(arr3d, x0, dx, y0, dy, z0, dz, xm, ym, zm, radius=None):
    """Trilinear-interpolate a cell-centered 3D array at (xm,ym,zm). If radius is
    given, exclude stencil centers inside that radius and renormalize the weights."""
    nz, ny, nx = arr3d.shape
    xi = (xm - x0)/dx
    i0 = int(np.floor(xi))
    fx = xi-i0
    yi = (ym - y0)/dy
    j0 = int(np.floor(yi))
    fy = yi-j0
    zi = (zm - z0)/dz
    k0 = int(np.floor(zi))
    fz = zi-k0
    i0 = min(max(i0, 0), nx-2)
    j0 = min(max(j0, 0), ny-2)
    k0 = min(max(k0, 0), nz-2)

    value = 0.0
    weight_sum = 0.0
    for dk in (0, 1):
        for dj in (0, 1):
            for di in (0, 1):
                xc = x0 + (i0+di)*dx
                yc = y0 + (j0+dj)*dy
                zc = z0 + (k0+dk)*dz
                wx = (1.0-fx) if di == 0 else fx
                wy = (1.0-fy) if dj == 0 else fy
                wz = (1.0-fz) if dk == 0 else fz
                if radius is not None:
                    if np.sqrt(xc*xc + yc*yc + zc*zc) < radius:
                        continue
                weight = wx*wy*wz
                if weight == 0.0:
                    continue
                value += weight*arr3d[k0+dk, j0+dj, i0+di]
                weight_sum += weight
    assert weight_sum > 0.0, "mirror stencil has no exterior cell"
    return value/weight_sum


def expected_mirror_state_3d(
        dens, velx, vely, velz, eint, x0, dx, y0, dy, z0, dz, x, y, z, radius, bc):
    """3D analog of expected_mirror_state(), including the x3 component."""
    r = np.sqrt(x*x + y*y + z*z)
    r_safe = max(r, 1e-12*radius)
    s = (2.0*radius - r_safe)/r_safe
    xm, ym, zm = x*s, y*s, z*s
    dm = trilinear(dens, x0, dx, y0, dy, z0, dz, xm, ym, zm, radius)
    vxm = trilinear(velx, x0, dx, y0, dy, z0, dz, xm, ym, zm, radius)
    vym = trilinear(vely, x0, dx, y0, dy, z0, dz, xm, ym, zm, radius)
    vzm = trilinear(velz, x0, dx, y0, dy, z0, dz, xm, ym, zm, radius)
    em = trilinear(eint, x0, dx, y0, dy, z0, dz, xm, ym, zm, radius)
    if bc == "reflecting":
        rhatx, rhaty, rhatz = x/r_safe, y/r_safe, z/r_safe
        vdotr = vxm*rhatx + vym*rhaty + vzm*rhatz
        vx = vxm - 2*vdotr*rhatx
        vy = vym - 2*vdotr*rhaty
        vz = vzm - 2*vdotr*rhatz
    else:  # absorbing
        vx, vy, vz = 0.0, 0.0, 0.0
    return dm, vx, vy, vz, em


def stitch(filename):
    """Assemble every MeshBlock of a .bin dump into one global array per variable.

    The mesh is uniform and static, so two runs that differ only in <meshblock>/nx
    describe the same cells and can be compared directly once stitched. Returns
    (time, n_meshblocks, {var: array[nk,nj,ni]}).
    """
    d = read_bin(filename)
    geom = d["mb_geometry"]
    nk, nj, ni = d["mb_data"][d["var_names"][0]][0].shape
    x1min, x2min, x3min = geom[:, 0].min(), geom[:, 2].min(), geom[:, 4].min()
    dx = (geom[0, 1] - geom[0, 0])/ni
    dy = (geom[0, 3] - geom[0, 2])/nj
    dz = (geom[0, 5] - geom[0, 4])/nk
    shape = (int(round((geom[:, 5].max() - x3min)/dz)),
             int(round((geom[:, 3].max() - x2min)/dy)),
             int(round((geom[:, 1].max() - x1min)/dx)))
    out = {}
    for v in d["var_names"]:
        g = np.full(shape, np.nan)
        for m in range(d["n_mbs"]):
            i0 = int(round((geom[m, 0] - x1min)/dx))
            j0 = int(round((geom[m, 2] - x2min)/dy))
            k0 = int(round((geom[m, 4] - x3min)/dz))
            g[k0:k0+nk, j0:j0+nj, i0:i0+ni] = d["mb_data"][v][m]
        assert not np.isnan(g).any(), f"{filename}: gaps after stitching {v}"
        out[v] = g
    return d["time"], d["n_mbs"], out


def max_rel_diff(file_a, file_b):
    """Largest relative difference between two .bin dumps over all variables.

    Used to assert that splitting the masked sphere across MeshBlocks reproduces the
    single-MeshBlock answer. Returns (n_mbs_a, n_mbs_b, max_rel_diff, worst_var).
    """
    ta, na, a = stitch(file_a)
    tb, nb, b = stitch(file_b)
    assert abs(ta - tb) < 1e-12, f"dumps are at different times: {ta} vs {tb}"
    worst, wvar = 0.0, None
    for v in a:
        assert a[v].shape == b[v].shape, f"{v}: shape {a[v].shape} vs {b[v].shape}"
        err = np.abs(a[v] - b[v]).max()/max(np.abs(a[v]).max(), 1e-30)
        if err > worst:
            worst, wvar = err, v
    return na, nb, worst, wvar
