
#include <math.h>
#include <stdio.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm> // max(), max_element(), min(), min_element()
#include <iomanip>
#include <iostream> // endl
#include <limits>   // numeric_limits::max()
#include <memory>
#include <sstream> // stringstream
#include <string>  // c_str(), string
#include <vector>

#include "athena.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "radiation/radiation.hpp"
#include "src/srcterms/turb_driver.hpp"

#include <iostream>

namespace {

// Useful container for physical parameters
struct bhl_pgen {
  Real spin;             // black hole spin
  Real dexcise, pexcise; // excision parameters

  Real gamma_adi; // EOS parameters

  // Radius mask for computing gravitational drag
  Real grav_drag_mask_rmax;
};

static bhl_pgen bhl_accretion;

} // namespace

// Prototypes for user-defined BCs and history functions
void BhlAccretionHistory(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (not restart) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "\n You should be restarting from a rst file " << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->prad != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " Radiation should not be enabled" << std::endl;
  }

  if (!pmbp->pcoord->is_general_relativistic) {
    std::cout
        << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl
        << " This executable can only be run when GR defined in <coord> block"
        << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &coord = pmbp->pcoord->coord_data;
  bhl_accretion.spin = coord.bh_spin;

  bhl_accretion.grav_drag_mask_rmax =
      pin->GetReal("problem", "grav_drag_mask_rmax");

  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  const Real r_plus = 1.0 + sqrt(1.0 - SQR(coord.bh_spin));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, r_plus));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 2.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 3.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 4.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 5.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 10.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 20.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 50.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 100.0));
  user_hist_func = BhlAccretionHistory;

  // return if restart
  if (restart)
    return;

  return;
}

namespace {} // namespace

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
    auto bhl = bhl_accretion;

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
