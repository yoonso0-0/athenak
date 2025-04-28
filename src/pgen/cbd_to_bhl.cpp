
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "kokkos/core/src/Kokkos_Timer.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "radiation/radiation.hpp"
#include "src/srcterms/turb_driver.hpp"

#include "kokkos/core/src/Kokkos_Timer.hpp"

#include <iostream>

#include <Kokkos_Random.hpp>

// prototypes for functions used internally to this pgen
namespace {

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct cbd_to_bhl_pgen pgen, Real x1,
                                         Real x2, Real x3, Real *pr,
                                         Real *ptheta, Real *pphi);

// Useful container for physical parameters
struct cbd_to_bhl_pgen {
  Real spin;             // black hole spin
  Real dexcise, pexcise; // excision parameters

  Real gamma_adi; // EOS parameters

  // Remnant Kick parameters
  Real v_kick_x;
  Real v_kick_y;
  Real v_kick_z;

  // Radius mask for computing gravitational drag
  Real grav_drag_mask_rmax;
};

static cbd_to_bhl_pgen cbd_to_bhl_converter;

} // namespace

// Prototypes for user-defined BCs and history functions
// void BhlAccretionBoundary(Mesh *pm);
void BhlAccretionHistory(HistoryData *pdata, Mesh *pm);

//------------------------------------------------------------------------------
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (global_variable::my_rank == 0) {
    std::cout << " ==================================================="
              << std::endl;
    std::cout << "\n  Running Newtonian CBD -> GRMHD BHL conversion\n"
              << std::endl;
    std::cout << " ==================================================="
              << std::endl;
  }

  //
  // YK: this pgen file doesn't really do anything, other than forcing
  // coordinate type gr=true. Rescaling/mapping MHD primitive variables and
  // coordinates are handled by other parts of the code.
  //

  if (not restart) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "\n You should be restarting from a rst file " << std::endl;
    exit(EXIT_FAILURE);
  }

  // if (pmbp->pcoord->is_general_relativistic) {
  //   std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
  //             << std::endl
  //             << "\n You cannot restart from non-GR coord" << std::endl;
  //   exit(EXIT_FAILURE);
  // }

  if (pmbp->prad != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " Radiation should not be enabled" << std::endl;
  }

  // Overwrite coordinate information
  pmbp->pcoord->is_general_relativistic = true;

  // User boundary function
  // @YK : Don't use user BC now
  // user_bcs_func = BhlAccretionBoundary;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;

  // coord.bh_spin = pin->GetReal("cbd_to_bhl_mapping", "spin");
  // coord.bh_excise = pin->GetBoolean("cbd_to_bhl_mapping", "excise");
  // coord.dexcise = pin->GetReal("cbd_to_bhl_mapping", "dexcise");
  // coord.pexcise = pin->GetReal("cbd_to_bhl_mapping", "pexcise");
  cbd_to_bhl_converter.spin = coord.bh_spin;
  // cbd_to_bhl_converter.dexcise = coord.dexcise;
  // cbd_to_bhl_converter.pexcise = coord.pexcise;

  cbd_to_bhl_converter.v_kick_x =
      pin->GetReal("cbd_to_bhl_mapping", "v_kick_x");
  cbd_to_bhl_converter.v_kick_y =
      pin->GetReal("cbd_to_bhl_mapping", "v_kick_y");
  cbd_to_bhl_converter.v_kick_z =
      pin->GetReal("cbd_to_bhl_mapping", "v_kick_z");

  const Real vx = cbd_to_bhl_converter.v_kick_x;
  const Real vy = cbd_to_bhl_converter.v_kick_y;
  const Real vz = cbd_to_bhl_converter.v_kick_z;
  const Real v_squared = SQR(vx) + SQR(vy) + SQR(vz);

  if (pmbp->prad != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " Radiation should not be enabled" << std::endl;
  }

  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  const Real r_plus = 1.0 + sqrt(1.0 - SQR(cbd_to_bhl_converter.spin));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, r_plus));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 2.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 3.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 4.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 5.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 10.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 20.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 50.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 100.0));
  user_hist_func = BhlAccretionHistory;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_; // @YK : u0 is evolved vars, w0 is prim vars
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
  }

  // Get ideal gas EOS data
  if (pmbp->phydro != nullptr) {
    cbd_to_bhl_converter.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  } else if (pmbp->pmhd != nullptr) {
    cbd_to_bhl_converter.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  }
  Real gamma_minus_one = cbd_to_bhl_converter.gamma_adi - 1.0;

  // Gravitational drag - Mask
  cbd_to_bhl_converter.grav_drag_mask_rmax =
      pin->GetReal("cbd_to_bhl_mapping", "grav_drag_mask_rmax");

  //****************************************************************
  if (global_variable::my_rank == 0) {
    std::cout << " BH spin = " << coord.bh_spin << std::endl;
    std::cout << " Horizon radius = " << r_plus << std::endl;
    std::cout << " EOS gamma = " << cbd_to_bhl_converter.gamma_adi << std::endl;
    std::cout << " Kick v^i = (" << vx << ", " << vy << ", " << vz << ")"
              << std::endl;
  }
  // exit(EXIT_FAILURE);
  //****************************************************************

  return;
}

//
// @YK : Some free functions defined below
//
namespace {

//----------------------------------------------------------------------------------------
// Function to calculate time component of contravariant four velocity in BL
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: u_t

KOKKOS_INLINE_FUNCTION
static Real CalculateCovariantUT(struct cbd_to_bhl_pgen pgen, Real r,
                                 Real sin_theta, Real l) {
  // Compute BL metric components
  Real sigma = SQR(r) + SQR(pgen.spin) * (1.0 - SQR(sin_theta));
  Real g_00 = -1.0 + 2.0 * r / sigma;
  Real g_03 = -2.0 * pgen.spin * r / sigma * SQR(sin_theta);
  Real g_33 = (SQR(r) + SQR(pgen.spin) +
               2.0 * SQR(pgen.spin) * r / sigma * SQR(sin_theta)) *
              SQR(sin_theta);

  // Compute time component of covariant BL 4-velocity
  Real u_t = -sqrt(
      fmax((SQR(g_03) - g_00 * g_33) / (g_33 + 2.0 * l * g_03 + SQR(l) * g_00),
           0.0));
  return u_t;
}

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct cbd_to_bhl_pgen pgen, Real x1,
                                         Real x2, Real x3, Real *pr,
                                         Real *ptheta, Real *pphi) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = fmax((sqrt(SQR(rad) - SQR(pgen.spin) +
                      sqrt(SQR(SQR(rad) - SQR(pgen.spin)) +
                           4.0 * SQR(pgen.spin) * SQR(x3))) /
                 sqrt(2.0)),
                1.0);
  *pr = r;
  *ptheta = (fabs(x3 / r) < 1.0) ? acos(x3 / r) : acos(copysign(1.0, x3));
  *pphi = atan2(r * x2 - pgen.spin * x1, pgen.spin * x2 + r * x1) -
          pgen.spin * r / (SQR(r) - 2.0 * r + SQR(pgen.spin));
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired
// coordinates Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist
//   coordinates x1,x2,x3: Cartesian Kerr-Schild coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired
//   coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct cbd_to_bhl_pgen pgen, Real a0_bl, Real a1_bl,
                            Real a2_bl, Real a3_bl, Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = fmax((sqrt(SQR(rad) - SQR(pgen.spin) +
                      sqrt(SQR(SQR(rad) - SQR(pgen.spin)) +
                           4.0 * SQR(pgen.spin) * SQR(x3))) /
                 sqrt(2.0)),
                1.0);
  Real delta = SQR(r) - 2.0 * r + SQR(pgen.spin);
  *pa0 = a0_bl + 2.0 * r / delta * a1_bl;
  *pa1 = a1_bl * ((r * x1 + pgen.spin * x2) / (SQR(r) + SQR(pgen.spin)) -
                  x2 * pgen.spin / delta) +
         a2_bl * x1 * x3 / r *
             sqrt((SQR(r) + SQR(pgen.spin)) / (SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ((r * x2 - pgen.spin * x1) / (SQR(r) + SQR(pgen.spin)) +
                  x1 * pgen.spin / delta) +
         a2_bl * x2 * x3 / r *
             sqrt((SQR(r) + SQR(pgen.spin)) / (SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3 / r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2)) / (SQR(r) + SQR(pgen.spin)));
  return;
}

} // namespace

//----------------------------------------------------------------------------------------
// Function for computing accretion fluxes through constant spherical KS radius
// surfaces

void BhlAccretionHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  // extract BH parameters
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  // set nvars, adiabatic index, primitive array w0, and field array bcc0 if
  // is_mhd
  int nvars;
  Real gamma;
  bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    gamma = pmbp->phydro->peos->eos_data.gamma;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    gamma = pmbp->pmhd->peos->eos_data.gamma;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }

  // extract grids, number of radii, number of fluxes, and history appending
  // index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  // int nflux = (is_mhd) ? 7 : 3;  // @YK : magnetic flux, momentum flux (3)
  int nflux = 10; // @YK : Mdot, Edot, Ldot, Phi, Momentum drag (3),
                  // Gravitational drag (3)

  // set number of and names of history variables for hydro or mhd

  // Quantities to be computed at different radii
  //  (1) mass accretion rate
  //  (2) energy flux
  //  (3) angular momentum flux
  //  (4) magnetic flux
  //  (5,6,7) Momentum flux in x, y, z
  //  (8,9,10) Gravitational drag in x, y, z
  //
  pdata->nhist = nradii * nflux;

  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int g = 0; g < nradii; ++g) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();
    pdata->label[nflux * g + 0] = "mdot_" + rad_str;
    pdata->label[nflux * g + 1] = "edot_" + rad_str;
    pdata->label[nflux * g + 2] = "ldot_" + rad_str;
    if (is_mhd) {
      pdata->label[nflux * g + 3] = "phi_" + rad_str;

      pdata->label[nflux * g + 4] = "Fmx_" + rad_str;
      pdata->label[nflux * g + 5] = "Fmy_" + rad_str;
      pdata->label[nflux * g + 6] = "Fmz_" + rad_str;

      pdata->label[nflux * g + 7] = "Fgx_" + rad_str;
      pdata->label[nflux * g + 8] = "Fgy_" + rad_str;
      pdata->label[nflux * g + 9] = "Fgz_" + rad_str;
    }
  }

  // go through angles at each radii:
  DualArray2D<Real> interpolated_bcc; // needed for MHD
  for (int g = 0; g < nradii; ++g) {
    // zero fluxes at this radius
    pdata->hdata[nflux * g + 0] = 0.0;
    pdata->hdata[nflux * g + 1] = 0.0;
    pdata->hdata[nflux * g + 2] = 0.0;
    if (is_mhd) {
      pdata->hdata[nflux * g + 3] = 0.0;
      pdata->hdata[nflux * g + 4] = 0.0;
      pdata->hdata[nflux * g + 5] = 0.0;
      pdata->hdata[nflux * g + 6] = 0.0;

      pdata->hdata[nflux * g + 7] = 0.0;
      pdata->hdata[nflux * g + 8] = 0.0;
      pdata->hdata[nflux * g + 9] = 0.0;
    }

    // interpolate primitives (and cell-centered magnetic fields iff mhd)
    if (is_mhd) {
      grids[g]->InterpolateToSphere(3, bcc0_);
      Kokkos::realloc(interpolated_bcc, grids[g]->nangles, 3);
      Kokkos::deep_copy(interpolated_bcc, grids[g]->interp_vals);
      interpolated_bcc.template modify<DevExeSpace>();
      interpolated_bcc.template sync<HostMemSpace>();
    }
    grids[g]->InterpolateToSphere(nvars, w0_);

    // compute fluxes
    for (int n = 0; n < grids[g]->nangles; ++n) {
      // extract coordinate data at this angle
      Real r = grids[g]->radius;
      Real theta = grids[g]->polar_pos.h_view(n, 0);
      Real phi = grids[g]->polar_pos.h_view(n, 1);
      Real x1 = grids[g]->interp_coord.h_view(n, 0);
      Real x2 = grids[g]->interp_coord.h_view(n, 1);
      Real x3 = grids[g]->interp_coord.h_view(n, 2);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1, x2, x3, flat, spin, glower, gupper);

      // extract interpolated primitives
      Real &int_dn = grids[g]->interp_vals.h_view(n, IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n, IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n, IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n, IVZ);
      Real &int_ie = grids[g]->interp_vals.h_view(n, IEN);

      // extract interpolated field components (iff is_mhd)
      Real int_bx = 0.0, int_by = 0.0, int_bz = 0.0;
      if (is_mhd) {
        int_bx = interpolated_bcc.h_view(n, IBX);
        int_by = interpolated_bcc.h_view(n, IBY);
        int_bz = interpolated_bcc.h_view(n, IBZ);
      }

      // Compute interpolated u^\mu in CKS
      Real q = glower[1][1] * int_vx * int_vx +
               2.0 * glower[1][2] * int_vx * int_vy +
               2.0 * glower[1][3] * int_vx * int_vz +
               glower[2][2] * int_vy * int_vy +
               2.0 * glower[2][3] * int_vy * int_vz +
               glower[3][3] * int_vz * int_vz;
      Real alpha = sqrt(-1.0 / gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor / alpha;
      Real u1 = int_vx - alpha * lor * gupper[0][1];
      Real u2 = int_vy - alpha * lor * gupper[0][2];
      Real u3 = int_vz - alpha * lor * gupper[0][3];

      // Lower vector indices
      Real u_0 = glower[0][0] * u0 + glower[0][1] * u1 + glower[0][2] * u2 +
                 glower[0][3] * u3;
      Real u_1 = glower[1][0] * u0 + glower[1][1] * u1 + glower[1][2] * u2 +
                 glower[1][3] * u3;
      Real u_2 = glower[2][0] * u0 + glower[2][1] * u1 + glower[2][2] * u2 +
                 glower[2][3] * u3;
      Real u_3 = glower[3][0] * u0 + glower[3][1] * u1 + glower[3][2] * u2 +
                 glower[3][3] * u3;

      // Calculate 4-magnetic field (returns zero if not MHD)
      Real b0 = u_1 * int_bx + u_2 * int_by + u_3 * int_bz;
      Real b1 = (int_bx + b0 * u1) / u0;
      Real b2 = (int_by + b0 * u2) / u0;
      Real b3 = (int_bz + b0 * u3) / u0;

      // compute b_\mu in CKS and b_sq (returns zero if not MHD)
      Real b_0 = glower[0][0] * b0 + glower[0][1] * b1 + glower[0][2] * b2 +
                 glower[0][3] * b3;
      Real b_1 = glower[1][0] * b0 + glower[1][1] * b1 + glower[1][2] * b2 +
                 glower[1][3] * b3;
      Real b_2 = glower[2][0] * b0 + glower[2][1] * b1 + glower[2][2] * b2 +
                 glower[2][3] * b3;
      Real b_3 = glower[3][0] * b0 + glower[3][1] * b1 + glower[3][2] * b2 +
                 glower[3][3] * b3;
      Real b_sq = b0 * b_0 + b1 * b_1 + b2 * b_2 + b3 * b_3;

      // Transform CKS 4-velocity and 4-magnetic field to spherical KS
      Real a2 = SQR(spin);
      Real rad2 = SQR(x1) + SQR(x2) + SQR(x3);
      Real r2 = SQR(r);
      Real sth = sin(theta);
      Real sph = sin(phi);
      Real cph = cos(phi);
      Real drdx = r * x1 / (2.0 * r2 - rad2 + a2);
      Real drdy = r * x2 / (2.0 * r2 - rad2 + a2);
      Real drdz = (r * x3 + a2 * x3 / r) / (2.0 * r2 - rad2 + a2);
      // contravariant r component of 4-velocity
      Real ur = drdx * u1 + drdy * u2 + drdz * u3;
      // contravariant r component of 4-magnetic field (returns zero if not MHD)
      Real br = drdx * b1 + drdy * b2 + drdz * b3;
      // covariant phi component of 4-velocity
      Real u_ph = (-r * sph - spin * cph) * sth * u_1 +
                  (r * cph - spin * sph) * sth * u_2;
      // covariant phi component of 4-magnetic field (returns zero if not MHD)
      Real b_ph = (-r * sph - spin * cph) * sth * b_1 +
                  (r * cph - spin * sph) * sth * b_2;

      // integration params
      Real &domega = grids[g]->solid_angles.h_view(n);
      Real sqrtmdet = (r2 + SQR(spin * cos(theta)));

      // temporary variables
      Real wtot = int_dn + gamma * int_ie + b_sq;
      Real ptot = (gamma - 1.0) * int_ie + 0.5 * b_sq;

      // compute mass flux
      pdata->hdata[nflux * g + 0] += -1.0 * int_dn * ur * sqrtmdet * domega;

      // compute energy flux
      Real t1_0 = wtot * ur * u_0 - br * b_0;
      // YK : there was a minus sign here, but removed it. T^r_t describes the
      // energy "inflow"
      pdata->hdata[nflux * g + 1] += t1_0 * sqrtmdet * domega;

      // compute angular momentum flux
      Real t1_3 = wtot * ur * u_ph - br * b_ph;
      pdata->hdata[nflux * g + 2] += t1_3 * sqrtmdet * domega;

      if (is_mhd) {
        // compute magnetic flux
        pdata->hdata[nflux * g + 3] +=
            0.5 * fabs(br * u0 - b0 * ur) * sqrtmdet * domega;

        // Compute momentum drag
        Real tr_1 = (wtot * ur * u_1) + (ptot * drdx) - (br * b_1);
        Real tr_2 = (wtot * ur * u_2) + (ptot * drdy) - (br * b_2);
        Real tr_3 = (wtot * ur * u_3) + (ptot * drdz) - (br * b_3);
        pdata->hdata[nflux * g + 4] += tr_1 * sqrtmdet * domega;
        pdata->hdata[nflux * g + 5] += tr_2 * sqrtmdet * domega;
        pdata->hdata[nflux * g + 6] += tr_3 * sqrtmdet * domega;
      }
    }

    //
    // Compute gravitational drag
    //
    // pdata->hdata[nflux * (nradii - 1) + 7] = 0.0;
    // pdata->hdata[nflux * (nradii - 1) + 8] = 0.0;
    // pdata->hdata[nflux * (nradii - 1) + 9] = 0.0;

    // loop over all MeshBlocks in this pack
    auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
    int is = indcs.is;
    int js = indcs.js;
    int ks = indcs.ks;
    const int nmkji = (pmbp->nmb_thispack) * indcs.nx3 * indcs.nx2 * indcs.nx1;
    const int nkji = indcs.nx3 * indcs.nx2 * indcs.nx1;
    const int nji = indcs.nx2 * indcs.nx1;
    auto &size = pm->pmb_pack->pmb->mb_size;

    const Real min_radius = grids[g]->radius;
    auto bhl = cbd_to_bhl_converter;

    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce(
        "gravitational_drag", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
        KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
          // compute m,k,j,i indices of thread and call function
          int m = (idx) / nkji;
          int k = (idx - m * nkji) / nji;
          int j = (idx - m * nkji - k * nji) / indcs.nx1;
          int i = (idx - m * nkji - k * nji - j * indcs.nx1) + is;
          k += ks;
          j += js;

          // Volume & Cell-centered coordinates & 1/r^3
          Real vol =
              size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;

          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j - js, indcs.nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

          Real r2 = SQR(x1v) + SQR(x2v) + SQR(x3v);

          // @YK : we can set the integrand to zero inside our outside a certain
          // radius.
          if (r2 > SQR(min_radius) and r2 < SQR(bhl.grav_drag_mask_rmax)) {
            Real one_over_r3 = 1.0 / (r2 * sqrt(r2));

            // Gravitational drag
            array_sum::GlobalSum grav_drag;
            grav_drag.the_array[0] =
                vol * w0_(m, IDN, k, j, i) * x1v * one_over_r3;
            grav_drag.the_array[1] =
                vol * w0_(m, IDN, k, j, i) * x2v * one_over_r3;
            grav_drag.the_array[2] =
                vol * w0_(m, IDN, k, j, i) * x3v * one_over_r3;

            // sum into parallel reduce
            mb_sum += grav_drag;
          }
        },
        Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));
    Kokkos::fence();

    // store data into hdata array
    pdata->hdata[nflux * g + 7] += sum_this_mb.the_array[0];
    pdata->hdata[nflux * g + 8] += sum_this_mb.the_array[1];
    pdata->hdata[nflux * g + 9] += sum_this_mb.the_array[2];
  }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}
