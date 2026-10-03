"""
Spherical inner mask test for non-relativistic hydro.

A spherical mask of fixed radius, centered at the coordinate origin, overwrites the
primitive (and consistently the conserved) state in every cell with r < radius, every
stage, so the sphere's surface behaves like an effective boundary condition. This test
drives a 2D Liska & Wendroff implosion (a converging shock toward the corner at the
origin) and checks each of the four supported variants:
  - dirichlet:       interior pinned exactly to the prescribed fixed state
  - reflecting:      interior matches the mirror-point interpolation of the exterior
                      state, with the radial velocity component flipped
  - absorbing:        same mirrored density/pressure, but velocity forced to zero
  - spherical_wind:   interior density/pressure pinned as with dirichlet, but velocity
                       is a radial field of fixed magnitude vel_r (checked separately
                       in test_spherical_wind, since it has no fixed Cartesian vector)

It also checks that the two fatal startup validations required by the fully-local (no
MPI) design actually fire: sphere_mask + AMR, and a MeshBlock too small to contain the
2*radius mirror-point neighborhood of the origin.
"""

import glob
import os

import numpy as np
import pytest
import test_suite.testutils as testutils
import test_suite.nr.sphere_mask_utils as smu

_RADIUS = 0.03
_SM_DENS = 2.0
_SM_EINT = 20.0


def _latest_bin(basename):
    files = sorted(glob.glob(f"bin/{basename}.hydro_w.*.bin"))
    assert files, f"no .bin output found for {basename}"
    return files[-1]


def _origin_fields(fname):
    d = smu.read_bin(fname)
    m, x, y, dx, dy = smu.origin_meshblock(d)
    fields = {v: d["mb_data"][v][m][0] for v in d["var_names"]}
    return fields, x, y, dx, dy


def _remove_bin_outputs(basename):
    for f in glob.glob(f"bin/{basename}.hydro_w.*.bin"):
        os.remove(f)


@pytest.mark.parametrize("bc", ["dirichlet", "reflecting", "absorbing"])
def test_run(bc):
    """Run one BC variant and check every interior (r<radius) cell against the
    analytic (dirichlet) or independently re-derived (reflecting/absorbing) state."""
    input_file = f"inputs/sphere_mask_{bc}.athinput"
    basename = f"SphereMask{bc.capitalize()}"
    _remove_bin_outputs(basename)
    try:
        results = testutils.run(input_file, [f"job/basename={basename}"])
        assert results, f"sphere_mask {bc} test run failed."

        fields, x, y, dx, dy = _origin_fields(_latest_bin(basename))
        dens = fields["dens"]
        velx = fields["velx"]
        vely = fields["vely"]
        eint = fields["eint"]

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
            # float32 .bin output vs. double-precision re-derivation: expect ~1e-4
            assert max_err < 1e-2, (
                f"{bc}: interior state does not match the mirror-point formula "
                f"(max_err={max_err:g})")
    finally:
        _remove_bin_outputs(basename)
        testutils.cleanup()


def test_spherical_wind():
    """spherical_wind pins interior density/pressure like dirichlet, but velocity
    should be a radial field v = vel_r * rhat rather than a fixed vector."""
    vel_r = 0.4
    basename = "SphereMaskSphericalWind"
    _remove_bin_outputs(basename)
    try:
        results = testutils.run(
            "inputs/sphere_mask_spherical_wind.athinput", [f"job/basename={basename}"])
        assert results, "sphere_mask spherical_wind test run failed."

        fields, x, y, dx, dy = _origin_fields(_latest_bin(basename))
        dens = fields["dens"]
        velx = fields["velx"]
        vely = fields["vely"]
        eint = fields["eint"]

        X, Y = np.meshgrid(x, y)
        R = np.hypot(X, Y)
        mask = R < _RADIUS
        assert mask.sum() > 0, "no interior cells found -- check mesh/radius setup"

        assert np.allclose(dens[mask], _SM_DENS, atol=1e-5), (
            "spherical_wind: interior density not pinned")
        assert np.allclose(eint[mask], _SM_EINT, atol=1e-4), (
            "spherical_wind: interior internal energy not pinned")

        Rsafe = np.where(R > 0.0, R, 1.0)
        expected_velx = vel_r*X/Rsafe
        expected_vely = vel_r*Y/Rsafe
        assert np.allclose(velx[mask], expected_velx[mask], atol=1e-3), (
            "spherical_wind: interior velx does not match the radial-wind formula")
        assert np.allclose(vely[mask], expected_vely[mask], atol=1e-3), (
            "spherical_wind: interior vely does not match the radial-wind formula")
    finally:
        _remove_bin_outputs(basename)
        testutils.cleanup()


_BAD_INPUTS = [
    ("inputs/sphere_mask_bad_amr.athinput", "AMR"),
    ("inputs/sphere_mask_bad_block.athinput", "too-small MeshBlock"),
]


@pytest.mark.parametrize("input_file,label", _BAD_INPUTS)
def test_fatal_checks(input_file, label):
    """sphere_mask's startup validation must reject configurations that would break
    its fully-local, no-MPI design (see hydro_sphere_mask.cpp:InitSphereMask)."""
    try:
        with pytest.raises(RuntimeError):
            testutils.run(input_file)
    finally:
        testutils.cleanup()
