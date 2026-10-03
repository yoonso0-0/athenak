"""
Spherical inner mask test for non-relativistic hydro, across MPI ranks.

The sphere_mask design is deliberately fully local: at most one MeshBlock (on at most
one rank) ever owns the coordinate origin, so MaskSphere is a no-op everywhere else and
no MPI communication is required at runtime. This test decomposes the same 2D Liska &
Wendroff implosion used by test_nr_sphere_mask_cpu.py into 4 MeshBlocks and checks that
running it on 1 rank vs. 4 ranks produces bit-for-bit identical output -- a mismatch
would indicate a decomposition-dependent bug in the origin-MeshBlock lookup or the
mirror-point interpolation.
"""

import glob
import os

import numpy as np
import pytest
import test_suite.testutils as testutils
import test_suite.nr.sphere_mask_utils as smu

_INPUT = "inputs/sphere_mask_multirank.athinput"
_BASENAME = "SphereMaskMultirank"
_RADIUS = 0.03


def _remove_bin_outputs(basename):
    for f in glob.glob(f"bin/{basename}.hydro_w.*.bin"):
        os.remove(f)


def _latest_bin(basename):
    files = sorted(glob.glob(f"bin/{basename}.hydro_w.*.bin"))
    assert files, f"no .bin output found for {basename}"
    return files[-1]


@pytest.mark.parametrize("nranks", [1, 4])
def test_run(nranks):
    """The origin-owning MeshBlock must still pin/mirror correctly no matter how many
    ranks the 4-MeshBlock mesh is spread across."""
    basename = f"{_BASENAME}{nranks}"
    _remove_bin_outputs(basename)
    try:
        results = testutils.mpi_run(
            _INPUT, [f"job/basename={basename}"], threads=nranks)
        assert results, f"sphere_mask multirank test run failed for {nranks} ranks."

        d = smu.read_bin(_latest_bin(basename))
        m, x, y, dx, dy = smu.origin_meshblock(d)
        dens = d["mb_data"]["dens"][m][0]
        velx = d["mb_data"]["velx"][m][0]
        vely = d["mb_data"]["vely"][m][0]

        X, Y = np.meshgrid(x, y)
        mask = np.hypot(X, Y) < _RADIUS
        assert mask.sum() > 0, "no interior cells found -- check mesh/radius setup"
        assert np.allclose(dens[mask], 2.0, atol=1e-5), (
            f"{nranks}-rank run: interior density not pinned")
        assert np.allclose(velx[mask], 0.0, atol=1e-10)
        assert np.allclose(vely[mask], 0.0, atol=1e-10)
    finally:
        _remove_bin_outputs(basename)
        testutils.cleanup()


def test_rank_count_determinism():
    """1-rank and 4-rank runs of the same 4-MeshBlock mesh must agree bit-for-bit,
    since sphere_mask never communicates across ranks (see hydro_sphere_mask.cpp)."""
    base1, base4 = f"{_BASENAME}1", f"{_BASENAME}4"
    _remove_bin_outputs(base1)
    _remove_bin_outputs(base4)
    try:
        assert testutils.mpi_run(_INPUT, [f"job/basename={base1}"], threads=1)
        assert testutils.mpi_run(_INPUT, [f"job/basename={base4}"], threads=4)

        d1 = smu.read_bin(_latest_bin(base1))
        d4 = smu.read_bin(_latest_bin(base4))
        assert d1["n_mbs"] == d4["n_mbs"] == 4

        # match up MeshBlocks between the two runs by their physical origin corner,
        # since MPI rank assignment can reorder them
        def key(geom_row):
            return (round(geom_row[0], 10), round(geom_row[2], 10))

        order1 = sorted(range(4), key=lambda i: key(d1["mb_geometry"][i]))
        order4 = sorted(range(4), key=lambda i: key(d4["mb_geometry"][i]))

        for var in d1["var_names"]:
            for i1, i4 in zip(order1, order4):
                a1 = d1["mb_data"][var][i1]
                a4 = d4["mb_data"][var][i4]
                assert np.array_equal(a1, a4), (
                    f"variable {var} differs between 1-rank and 4-rank runs "
                    "-- sphere_mask should be fully decomposition-independent")
    finally:
        _remove_bin_outputs(base1)
        _remove_bin_outputs(base4)
        testutils.cleanup()
