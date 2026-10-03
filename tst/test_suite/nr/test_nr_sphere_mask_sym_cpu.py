"""
Spherical inner mask test for non-relativistic hydro, symmetric domain.

The main sphere_mask design (see test_nr_sphere_mask_cpu.py) requires that the single
MeshBlock owning the coordinate origin also contain the full 2*radius mirror-point
neighborhood of the origin. The original tests exercise this on an octant domain
(origin at a corner), where the mesh only extends in the positive direction along each
axis. This file checks the same requirement on a domain symmetric about the origin
([-0.3,0.3]^2), where the mesh extends in *both* directions along each axis, so
InitSphereMask's containment check must require 2*radius+margin of clearance on both
sides -- not just skip the negative side because it happened not to be needed before.

test_run mirrors test_nr_sphere_mask_cpu.py::test_run exactly (single MeshBlock spanning
the whole domain, so containment holds trivially) but on the symmetric domain, where the
implosion problem's diagonal interface passes right through the sphere, giving an
exterior state that is inhomogeneous across the masked region -- a stronger check on the
mirror-point interpolation than the octant case, where the region near the origin starts
out uniform.

test_quadrant_split_matches_single_block covers the layout a 2^N-MeshBlocks-per-axis
root grid always produces on a symmetric domain: block boundaries meeting exactly AT
the origin, cutting the masked sphere into quadrants. Every block masks its own quadrant
BEFORE the halo exchange, so SendU/RecvU carry finished values and no block ever has to
reconstruct a neighbour's masked cells. All four BCs must therefore reproduce the
single-MeshBlock answer cell for cell.

That matters more than it sounds: masked cells are overwritten every stage, so their own
evolution is irrelevant, but their values ARE the boundary condition -- they are read by
the reconstruction stencil of the UNMASKED cells just outside r=radius. Masking after the
exchange instead leaves a neighbour's copy one stage stale and perturbs the real solution
in a thin shell where the block boundary cuts the sphere's surface, which then advects
across the whole domain. These tests are what pins that down.

test_fatal_block_too_small_for_mirror covers the one layout constraint that survives:
reflecting/absorbing interpolate at a mirror point up to 2*radius from the origin, which
must land in the same block as the masked cell it belongs to. An 8x8 root grid here gives
each block only 0.075 of reach against 2*radius+margin=0.109, so InitSphereMask must
reject it -- while the pointwise BCs, which interpolate nothing, still run.
"""

import glob
import os

import numpy as np
import pytest
import test_suite.testutils as testutils
import test_suite.nr.sphere_mask_utils as smu

_RADIUS = 0.05
_SM_DENS = 2.0
_SM_EINT = 20.0


def _latest_bin(basename):
    files = sorted(glob.glob(f"bin/{basename}.hydro_w.*.bin"))
    assert files, f"no .bin output found for {basename}"
    return files[-1]


def _remove_bin_outputs(basename):
    for f in glob.glob(f"bin/{basename}.hydro_w.*.bin"):
        os.remove(f)


@pytest.mark.parametrize("bc", ["dirichlet", "reflecting", "absorbing"])
def test_run(bc):
    """Run one BC variant on the symmetric domain and check every interior
    (r<radius) cell against the analytic (dirichlet) or independently re-derived
    (reflecting/absorbing) state."""
    input_file = f"inputs/sphere_mask_{bc}_sym.athinput"
    basename = f"SphereMask{bc.capitalize()}Sym"
    _remove_bin_outputs(basename)
    try:
        results = testutils.run(input_file, [f"job/basename={basename}"])
        assert results, f"sphere_mask symmetric-domain {bc} test run failed."

        d = smu.read_bin(_latest_bin(basename))
        assert d["n_mbs"] == 1, "this test expects a single MeshBlock spanning the mesh"
        m, x, y, dx, dy = smu.origin_meshblock(d)
        dens = d["mb_data"]["dens"][m][0]
        velx = d["mb_data"]["velx"][m][0]
        vely = d["mb_data"]["vely"][m][0]
        eint = d["mb_data"]["eint"][m][0]

        X, Y = np.meshgrid(x, y)
        R = np.hypot(X, Y)
        mask = R < _RADIUS
        assert mask.sum() > 0, "no interior cells found -- check mesh/radius setup"

        if bc == "dirichlet":
            assert np.allclose(dens[mask], _SM_DENS, atol=1e-5), (
                "dirichlet: interior density not pinned")
            assert np.allclose(eint[mask], _SM_EINT, atol=1e-4), (
                "dirichlet: interior internal energy not pinned")
            assert np.allclose(velx[mask], 0.0, atol=1e-10), (
                "dirichlet: interior velx not pinned to zero")
            assert np.allclose(vely[mask], 0.0, atol=1e-10), (
                "dirichlet: interior vely not pinned to zero")
        else:
            if bc == "absorbing":
                assert np.all(velx[mask] == 0.0) and np.all(vely[mask] == 0.0), (
                    "absorbing: interior velocity must be exactly zero")
            max_err = 0.0
            for j in range(len(y)):
                for i in range(len(x)):
                    if not mask[j, i]:
                        continue
                    dm, vxe, vye, em = smu.expected_mirror_state(
                        dens, velx, vely, eint, x[0], dx, y[0], dy,
                        x[i], y[j], _RADIUS, bc)
                    max_err = max(max_err, abs(dm - dens[j, i]), abs(em - eint[j, i]),
                                  abs(vxe - velx[j, i]), abs(vye - vely[j, i]))
            assert max_err < 1e-2, (
                f"symmetric-domain {bc}: interior state does not match the "
                f"mirror-point formula (max_err={max_err:g})")
    finally:
        _remove_bin_outputs(basename)
        testutils.cleanup()


@pytest.mark.parametrize("bc", ["reflecting", "absorbing"])
def test_mirror_bc_applied_at_initialization(bc):
    """The t=0 output must already contain the mirror mask.  This specifically guards
    the initialization path: no RK stage is allowed to repair the state before it is
    checked.  A mask-disabled t=0 run supplies the unmodified state from which the
    expected mirror values are independently reconstructed."""
    input_file = f"inputs/sphere_mask_{bc}_sym.athinput"
    baseline_name = f"SphereMask{bc.capitalize()}InitBaseline"
    masked_name = f"SphereMask{bc.capitalize()}InitMasked"
    _remove_bin_outputs(baseline_name)
    _remove_bin_outputs(masked_name)
    try:
        testutils.run(input_file, ["sphere_mask/enabled=false", "time/nlim=0",
                                   f"job/basename={baseline_name}"])
        baseline = smu.read_bin(_latest_bin(baseline_name))

        testutils.run(input_file, ["time/nlim=0", f"job/basename={masked_name}"])
        masked = smu.read_bin(_latest_bin(masked_name))

        mb, x, y, dx, dy = smu.origin_meshblock(baseline)
        mm, xm, ym, _, _ = smu.origin_meshblock(masked)
        assert np.array_equal(x, xm) and np.array_equal(y, ym)

        base = {v: baseline["mb_data"][v][mb][0] for v in baseline["var_names"]}
        got = {v: masked["mb_data"][v][mm][0] for v in masked["var_names"]}
        X, Y = np.meshgrid(x, y)
        interior = np.hypot(X, Y) < _RADIUS
        assert interior.sum() > 0

        max_err = 0.0
        for j, i in zip(*np.nonzero(interior)):
            expected = smu.expected_mirror_state(
                base["dens"], base["velx"], base["vely"], base["eint"],
                x[0], dx, y[0], dy, x[i], y[j], _RADIUS, bc)
            actual = (got["dens"][j, i], got["velx"][j, i],
                      got["vely"][j, i], got["eint"][j, i])
            max_err = max(max_err, *(abs(a-b) for a, b in zip(actual, expected)))
        assert max_err < 1e-4, (
            f"{bc}: t=0 state was not masked before evolution (max_err={max_err:g})")
    finally:
        _remove_bin_outputs(baseline_name)
        _remove_bin_outputs(masked_name)
        testutils.cleanup()


@pytest.mark.parametrize("bc", ["reflecting", "absorbing"])
def test_mirror_bc_ignores_interior_state(bc):
    """Changing only the disposable pre-mask state at r<R must not change the mirror
    boundary state. This catches interpolation stencils that read an interior corner."""
    input_file = f"inputs/sphere_mask_{bc}_sym.athinput"
    name_a = f"SphereMask{bc.capitalize()}InteriorA"
    name_b = f"SphereMask{bc.capitalize()}InteriorB"
    _remove_bin_outputs(name_a)
    _remove_bin_outputs(name_b)
    try:
        common = ["time/nlim=0", f"problem/sphere_override_radius={_RADIUS}"]
        testutils.run(input_file, common + ["problem/sphere_override_dens=0.02",
                                           "problem/sphere_override_pres=0.03",
                                           f"job/basename={name_a}"])
        testutils.run(input_file, common + ["problem/sphere_override_dens=20.0",
                                           "problem/sphere_override_pres=30.0",
                                           f"job/basename={name_b}"])

        a = smu.read_bin(_latest_bin(name_a))
        b = smu.read_bin(_latest_bin(name_b))
        ma, x, y, _, _ = smu.origin_meshblock(a)
        mb, xb, yb, _, _ = smu.origin_meshblock(b)
        assert np.array_equal(x, xb) and np.array_equal(y, yb)
        interior = np.hypot(*np.meshgrid(x, y)) < _RADIUS
        assert interior.sum() > 0
        for var in ("dens", "velx", "vely", "velz", "eint"):
            va = a["mb_data"][var][ma][0]
            vb = b["mb_data"][var][mb][0]
            assert np.array_equal(va[interior], vb[interior]), (
                f"{bc}: masked {var} depends on the pre-mask interior state")
    finally:
        _remove_bin_outputs(name_a)
        _remove_bin_outputs(name_b)
        testutils.cleanup()


@pytest.mark.parametrize("bc", ["reflecting", "absorbing"])
def test_mirror_bc_rejects_origin_centered_cell(bc):
    """A mirror direction is undefined for a cell centered exactly at r=0."""
    try:
        with pytest.raises(RuntimeError):
            testutils.run(
                f"inputs/sphere_mask_{bc}_sym.athinput",
                ["mesh/nx1=63", "mesh/nx2=63",
                 "meshblock/nx1=63", "meshblock/nx2=63", "time/nlim=0"])
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("bc", ["reflecting", "absorbing"])
def test_full_domain_matches_reflecting_half_domain(bc):
    """For reflection-symmetric, nonuniform initial data, evolving x1>=0 with a
    reflecting boundary at x1=0 must reproduce the positive-x1 half of the full-domain
    solution.  The comparison includes the masked cells cut by the symmetry plane."""
    input_file = "inputs/sphere_mask_reflecting_sym.athinput"
    full_name = f"SphereMask{bc.capitalize()}Full2D"
    half_name = f"SphereMask{bc.capitalize()}Half2D"
    _remove_bin_outputs(full_name)
    _remove_bin_outputs(half_name)
    common = [f"sphere_mask/bc={bc}", "problem/radial_jump_radius=0.08",
              "mesh/nx1=32", "mesh/nx2=32", "time/nlim=6"]
    try:
        testutils.run(input_file, common + ["meshblock/nx1=32", "meshblock/nx2=32",
                                            f"job/basename={full_name}"])
        testutils.run(input_file, common + ["mesh/x1min=0.0", "mesh/nx1=16",
                                            "meshblock/nx1=16", "meshblock/nx2=32",
                                            "mesh/ix1_bc=reflect",
                                            f"job/basename={half_name}"])

        t_full, _, full = smu.stitch(_latest_bin(full_name))
        t_half, _, half = smu.stitch(_latest_bin(half_name))
        assert abs(t_full - t_half) < 1e-12
        assert t_full > 0.0 and np.ptp(full["dens"]) > 0.1, (
            "comparison must use an evolved, nonuniform solution")
        for var in full:
            expected = full[var][..., full[var].shape[-1]//2:]
            assert expected.shape == half[var].shape
            err = np.max(np.abs(expected-half[var]))
            scale = max(np.max(np.abs(expected)), 1.0)
            assert err/scale < 1e-10, (
                f"2D {bc}: reflecting half-domain disagrees with full domain in {var} "
                f"(max scaled difference={err/scale:g})")
    finally:
        _remove_bin_outputs(full_name)
        _remove_bin_outputs(half_name)
        testutils.cleanup()


_BCS = ["dirichlet", "spherical_wind", "reflecting", "absorbing"]
_INPUT = "inputs/sphere_mask_dirichlet_sym.athinput"
_BASENAME = "SphereMaskDirichletSym"


def _stitched_run(bc, nblk):
    """Run the symmetric-domain problem with bc on an (64/nblk)^2 root grid and return
    its dump stitched back into one global array."""
    testutils.run(_INPUT, [f"sphere_mask/bc={bc}",
                           f"meshblock/nx1={nblk}", f"meshblock/nx2={nblk}"])
    out = smu.stitch(_latest_bin(_BASENAME))
    _remove_bin_outputs(_BASENAME)
    return out


@pytest.mark.parametrize("bc", _BCS)
@pytest.mark.parametrize("nblk", [32, 16])
def test_quadrant_split_matches_single_block(bc, nblk):
    """Splitting the sphere across MeshBlocks that meet at the origin must give the same
    answer as masking it inside one block, for every BC (see module docstring).
    nblk=32/16 is a 2x2 / 4x4 root grid; both leave each block enough reach for the
    mirror BCs, so all four are exercised."""
    try:
        t_w, n_w, a = _stitched_run(bc, 64)
        t_s, n_s, b = _stitched_run(bc, nblk)
        assert n_w == 1 and n_s == (64//nblk)**2, (
            f"expected 1 vs {(64//nblk)**2} MeshBlocks, got {n_w} vs {n_s}")
        assert abs(t_w - t_s) < 1e-12, f"dumps at different times: {t_w} vs {t_s}"
        worst = max(np.abs(a[v] - b[v]).max()/max(np.abs(a[v]).max(), 1e-30) for v in a)
        # agreement is bitwise or at the roundoff floor; the tolerance only allows for a
        # different order of operations in the ghost exchange, never a stale ghost cell,
        # which shows up three or more orders of magnitude above this
        assert worst < 1e-10, (
            f"{bc}: {n_s}-MeshBlock quadrant split does not reproduce the "
            f"single-MeshBlock answer (max rel diff={worst:g})")
    finally:
        _remove_bin_outputs(_BASENAME)
        testutils.cleanup()


@pytest.mark.parametrize("bc", ["reflecting", "absorbing"])
def test_fatal_block_too_small_for_mirror(bc):
    """An 8x8 root grid leaves each block 0.075 of reach from the origin, short of
    2*radius+margin=0.109, so the mirror-point stencil would leave the MeshBlock.
    InitSphereMask must reject that at startup (see module docstring)."""
    try:
        with pytest.raises(RuntimeError):
            testutils.run(_INPUT, ["meshblock/nx1=8", "meshblock/nx2=8",
                                   f"sphere_mask/bc={bc}"])
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("bc", ["dirichlet", "spherical_wind"])
def test_pointwise_bc_allows_tiny_blocks(bc):
    """The converse of the test above: the pointwise BCs interpolate nothing, so they
    place no constraint at all on the MeshBlock layout and must run the same 8x8 grid."""
    try:
        results = testutils.run(_INPUT, ["meshblock/nx1=8", "meshblock/nx2=8",
                                         f"sphere_mask/bc={bc}"])
        assert results, f"sphere_mask {bc} must accept MeshBlocks smaller than the sphere"
    finally:
        _remove_bin_outputs(_BASENAME)
        testutils.cleanup()
