"""
Spherical inner mask test for non-relativistic hydro, 3D.

3D analog of test_nr_sphere_mask_cpu.py: same corner-anchored octant-domain Liska &
Wendroff implosion, extruded into x3 and driven with reflect BCs on all three inner
faces, exercising the trilinear mirror-point interpolation and k-direction indexing
paths that the 2D test cannot reach.

For a masked cell with r within about one cell width of radius, the interpolation stencil
around its exterior mirror point may straddle the spherical surface. Hydro::MaskSphere
excludes interior cells and renormalizes the remaining multilinear weights, so the
imposed state depends only on the current exterior flow.

test_octant_split_matches_single_block covers the 2x2x2 layout a 2^N-MeshBlocks-per-axis
root grid produces on a domain symmetric about the origin: the eight block corners meet
AT the origin, so the masked sphere is cut into octants. Each block masks its own octant
BEFORE the halo exchange, so SendU/RecvU carry finished values and no block ever has to
reconstruct a neighbour's masked cells; all four BCs must then reproduce the
single-MeshBlock answer. This is the sharpest of the sphere_mask tests -- with the mask
applied only at the end of the stage instead, the same comparison fails at the 1e-2 level
in every variable, seeded in a thin shell just outside r=radius and advected from there
across most of the domain.

test_fatal_block_too_small_for_mirror covers the one layout constraint that survives:
reflecting/absorbing interpolate at a mirror point up to 2*radius from the origin, which
must land in the same MeshBlock as the masked cell. An 8x8x8 root grid leaves each block
0.075 of reach against 2*radius+margin=0.1125, so it must be rejected at startup, while
the pointwise BCs still run it.
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


def _remove_bin_outputs(basename):
    for f in glob.glob(f"bin/{basename}.hydro_w.*.bin"):
        os.remove(f)


def _latest_bin(basename):
    files = sorted(glob.glob(f"bin/{basename}.hydro_w.*.bin"))
    assert files, f"no .bin output found for {basename}"
    return files[-1]


@pytest.mark.parametrize("bc", ["dirichlet", "reflecting", "absorbing"])
def test_run(bc):
    """Run one 3D BC variant and check every interior (r<radius) cell."""
    input_file = f"inputs/sphere_mask_{bc}_3d.athinput"
    basename = f"SphereMask{bc.capitalize()}3D"
    _remove_bin_outputs(basename)
    try:
        results = testutils.run(input_file, [f"job/basename={basename}"])
        assert results, f"sphere_mask 3D {bc} test run failed."

        d = smu.read_bin(_latest_bin(basename))
        m, x, y, z, dx, dy, dz = smu.origin_meshblock_3d(d)
        dens = d["mb_data"]["dens"][m]
        velx = d["mb_data"]["velx"][m]
        vely = d["mb_data"]["vely"][m]
        velz = d["mb_data"]["velz"][m]
        eint = d["mb_data"]["eint"][m]

        Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")
        mask = np.sqrt(Xg**2 + Yg**2 + Zg**2) < _RADIUS
        assert mask.sum() > 0, "no interior cells found -- check mesh/radius setup"

        if bc == "dirichlet":
            assert np.allclose(dens[mask], _SM_DENS, atol=1e-5), (
                "dirichlet: interior density not pinned")
            assert np.allclose(eint[mask], _SM_EINT, atol=1e-4), (
                "dirichlet: interior internal energy not pinned")
            for v in (velx, vely, velz):
                assert np.allclose(v[mask], 0.0, atol=1e-10), (
                    "dirichlet: interior velocity not pinned to zero")
        else:
            if bc == "absorbing":
                for v in (velx, vely, velz):
                    assert np.all(v[mask] == 0.0), (
                        "absorbing: interior velocity must be exactly zero")
            max_err = 0.0
            for k in range(len(z)):
                for j in range(len(y)):
                    for i in range(len(x)):
                        if not mask[k, j, i]:
                            continue
                        dm, vxe, vye, vze, em = smu.expected_mirror_state_3d(
                            dens, velx, vely, velz, eint, x[0], dx, y[0], dy, z[0], dz,
                            x[i], y[j], z[k], _RADIUS, bc)
                        max_err = max(max_err,
                                      abs(dm - dens[k, j, i]), abs(em - eint[k, j, i]),
                                      abs(vxe - velx[k, j, i]), abs(vye - vely[k, j, i]),
                                      abs(vze - velz[k, j, i]))
            assert max_err < 1e-4, (
                f"3D {bc}: interior state does not match the mirror-point formula "
                f"(max_err={max_err:g})")
    finally:
        _remove_bin_outputs(basename)
        testutils.cleanup()


def test_multi_meshblock():
    """With 8 MeshBlocks (2x2x2) instead of one, InitSphereMask must pick out exactly
    the corner block owning the origin and MaskSphere must leave the other 7 untouched
    -- this exercises the origin-MeshBlock lookup itself, distinct from the single-block
    tests above where that lookup is trivial."""
    basename = "SphereMaskMultirank3D"
    _remove_bin_outputs(basename)
    try:
        results = testutils.run(
            "inputs/sphere_mask_multirank_3d.athinput", [f"job/basename={basename}"])
        assert results, "sphere_mask 3D multi-MeshBlock test run failed."

        d = smu.read_bin(_latest_bin(basename))
        assert d["n_mbs"] == 8
        m, x, y, z, dx, dy, dz = smu.origin_meshblock_3d(d)

        dens = d["mb_data"]["dens"][m]
        velx = d["mb_data"]["velx"][m]
        vely = d["mb_data"]["vely"][m]
        velz = d["mb_data"]["velz"][m]
        eint = d["mb_data"]["eint"][m]
        Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")
        mask = np.sqrt(Xg**2 + Yg**2 + Zg**2) < 0.03
        assert mask.sum() > 0, "no interior cells found in the origin MeshBlock"
        assert np.allclose(dens[mask], _SM_DENS, atol=1e-5), (
            "multi-MeshBlock: interior density not pinned in the origin block")
        assert np.allclose(eint[mask], _SM_EINT, atol=1e-4)
        for v in (velx, vely, velz):
            assert np.allclose(v[mask], 0.0, atol=1e-10)

        # no other MeshBlock should ever be touched by the mask
        for mm in range(d["n_mbs"]):
            if mm == m:
                continue
            other_dens = d["mb_data"]["dens"][mm]
            other_eint = d["mb_data"]["eint"][mm]
            spuriously_pinned = np.any(
                (other_dens == _SM_DENS) & (other_eint == _SM_EINT))
            assert not spuriously_pinned, (
                f"MeshBlock {mm} (geometry {d['mb_geometry'][mm]}) was masked, but "
                "only the origin-owning MeshBlock should ever be touched")
    finally:
        _remove_bin_outputs(basename)
        testutils.cleanup()


@pytest.mark.parametrize("bc", ["reflecting", "absorbing"])
def test_full_domain_matches_reflecting_half_domain(bc):
    """3D full/half-domain equivalence for a spherical, nonuniform solution.  This
    exercises a symmetry plane that cuts the spherical mask while retaining all three
    dimensions in the mirror interpolation and hydro update."""
    input_file = "inputs/sphere_mask_octant_split_3d.athinput"
    full_name = f"SphereMask{bc.capitalize()}Full3D"
    half_name = f"SphereMask{bc.capitalize()}Half3D"
    _remove_bin_outputs(full_name)
    _remove_bin_outputs(half_name)
    common = [f"sphere_mask/bc={bc}", "problem/radial_jump_radius=0.10",
              "mesh/nx1=24", "mesh/nx2=24", "mesh/nx3=24", "time/nlim=4"]
    try:
        testutils.run(input_file, common + ["meshblock/nx1=24", "meshblock/nx2=24",
                                            "meshblock/nx3=24",
                                            f"job/basename={full_name}"])
        testutils.run(input_file, common + ["mesh/x1min=0.0", "mesh/nx1=12",
                                            "meshblock/nx1=12", "meshblock/nx2=24",
                                            "meshblock/nx3=24", "mesh/ix1_bc=reflect",
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
                f"3D {bc}: reflecting half-domain disagrees with full domain in {var} "
                f"(max scaled difference={err/scale:g})")
    finally:
        _remove_bin_outputs(full_name)
        _remove_bin_outputs(half_name)
        testutils.cleanup()


_BCS = ["dirichlet", "spherical_wind", "reflecting", "absorbing"]
_INPUT = "inputs/sphere_mask_octant_split_3d.athinput"
_SPLIT_BASENAME = "SphereMaskOctantSplit3D"
_BLK = ["meshblock/nx1", "meshblock/nx2", "meshblock/nx3"]


def _stitched_run(bc, nblk):
    """Run the octant-split problem with bc on an (48/nblk)^3 root grid and return its
    dump stitched back into one global array."""
    testutils.run(_INPUT, [f"sphere_mask/bc={bc}"] + [f"{k}={nblk}" for k in _BLK])
    out = smu.stitch(_latest_bin(_SPLIT_BASENAME))
    _remove_bin_outputs(_SPLIT_BASENAME)
    return out


@pytest.mark.parametrize("bc", _BCS)
@pytest.mark.parametrize("nblk", [24, 12])
def test_octant_split_matches_single_block(bc, nblk):
    """The masked sphere split across 2x2x2 (or 4x4x4) MeshBlocks meeting at the origin
    must give the same answer as masking it inside a single MeshBlock, for every BC."""
    try:
        t_w, n_w, a = _stitched_run(bc, 48)
        t_s, n_s, b = _stitched_run(bc, nblk)
        assert n_w == 1 and n_s == (48//nblk)**3, (
            f"expected 1 vs {(48//nblk)**3} MeshBlocks, got {n_w} vs {n_s}")
        assert abs(t_w - t_s) < 1e-12, f"dumps at different times: {t_w} vs {t_s}"
        # Use an absolute scale for fields that are analytically zero. Dividing the
        # ~1e-16 velocity roundoff floor by another ~1e-16 number reports a meaningless
        # O(1) relative error even when density/eint agree bitwise.
        worst = max(np.abs(a[v] - b[v]).max()/max(np.abs(a[v]).max(), 1.0) for v in a)
        assert worst < 1e-10, (
            f"{bc}: {n_s}-MeshBlock octant split does not reproduce the single-MeshBlock "
            f"answer (max rel diff={worst:g})")
    finally:
        _remove_bin_outputs(_SPLIT_BASENAME)
        testutils.cleanup()


@pytest.mark.parametrize("bc", ["reflecting", "absorbing"])
def test_fatal_block_too_small_for_mirror(bc):
    """An 8x8x8 root grid leaves each block 0.075 of reach from the origin, short of
    2*radius+margin=0.1125, so the mirror-point stencil would leave the MeshBlock.
    InitSphereMask must reject that at startup (see module docstring)."""
    try:
        with pytest.raises(RuntimeError):
            testutils.run(_INPUT, [f"sphere_mask/bc={bc}"] +
                          [f"{k}=6" for k in _BLK])
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("bc", ["dirichlet", "spherical_wind"])
def test_pointwise_bc_allows_tiny_blocks(bc):
    """The pointwise BCs interpolate nothing, so they place no constraint on the
    MeshBlock layout and must run the same 8x8x8 grid the mirror BCs reject."""
    try:
        results = testutils.run(_INPUT, [f"sphere_mask/bc={bc}"] +
                                [f"{k}=6" for k in _BLK])
        assert results, f"sphere_mask {bc} must accept MeshBlocks smaller than the sphere"
    finally:
        _remove_bin_outputs(_SPLIT_BASENAME)
        testutils.cleanup()
