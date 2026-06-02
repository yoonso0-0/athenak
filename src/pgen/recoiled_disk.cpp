//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//
//

// C++ headers
#include <cmath>
#include <iostream>
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

// Athena++ headers
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen.hpp"
#include "srcterms/srcterms.hpp"

namespace {

struct recoiled_disk_pgen {
  // surface density profile
  Real r_in;
  Real r_out;
  Real a_in;
  Real a_out;
  Real exponent_p;

  //
  bool constant_scale_height;
  Real scale_height_parameter;
  bool add_rotation_corrections;

  //
  Real gravity_softening_length;
  Real density_floor;

  //
  bool add_kick;
  Real kick_magnitude;
  Real kick_angle;
};

recoiled_disk_pgen recoiled_disk;

} // namespace

// Prototypes for user-defined BCs and history functions
void HydroNoInflow(Mesh *pm);
void RecoiledDiskHistory(HistoryData *pdata, Mesh *pm);

KOKKOS_INLINE_FUNCTION
Real compute_softened_inv_r(const Real radius, const Real h,
                            const Real one_over_h);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//
//  \brief Problem Generator for the recoiled thin disk (@YK)
//

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pcoord->is_general_relativistic ||
      pmbp->pcoord->is_dynamical_relativistic ||
      pmbp->pcoord->is_special_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " Recoiled disk problem cannot be run with relativity "
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // @YK : setting up boundary condition
  // user_bcs_func = HydroNoInflow;

  //
  user_hist_func = RecoiledDiskHistory;

  // return if restart
  if (restart)
    return;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  Real gm1;
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
    gm1 = (pmbp->phydro->peos->eos_data.gamma) - 1.0;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
    gm1 = (pmbp->pmhd->peos->eos_data.gamma) - 1.0;
  }

  recoiled_disk.r_in = pin->GetReal("problem", "r_in");
  recoiled_disk.r_out = pin->GetReal("problem", "r_out");
  recoiled_disk.a_in = pin->GetReal("problem", "a_in");
  recoiled_disk.a_out = pin->GetReal("problem", "a_out");
  recoiled_disk.exponent_p = pin->GetReal("problem", "exponent_p");

  recoiled_disk.constant_scale_height =
      pin->GetBoolean("problem", "constant_scale_height");
  recoiled_disk.scale_height_parameter =
      pin->GetReal("problem", "scale_height_parameter");
  recoiled_disk.add_rotation_corrections =
      pin->GetBoolean("problem", "add_rotation_corrections");

  recoiled_disk.gravity_softening_length =
      pin->GetReal("hydro_srcterms", "gravity_softening_length");

  recoiled_disk.density_floor = pin->GetReal("problem", "density_floor");

  recoiled_disk.add_kick = pin->GetBoolean("problem", "add_kick");
  recoiled_disk.kick_magnitude = pin->GetReal("problem", "kick_magnitude");
  recoiled_disk.kick_angle = pin->GetReal("problem", "kick_angle");

  auto pfloor = pin->GetReal("hydro", "pfloor");

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  const int nmkji = (pmbp->nmb_thispack) * indcs.nx3 * indcs.nx2 * indcs.nx1;
  const int nkji = indcs.nx3 * indcs.nx2 * indcs.nx1;
  const int nji = indcs.nx2 * indcs.nx1;

  //
  auto rec_disk = recoiled_disk;

  const Real h = rec_disk.gravity_softening_length;
  const Real one_over_h = 1.0 / h;

  Real min_dx1, min_dx2, min_dx3;
  Kokkos::parallel_reduce(
      "recoiled_disk_3d", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &local_min_dx1, Real &local_min_dx2,
                    Real &local_min_dx3) {
        int m = (idx) / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / indcs.nx1;
        int i = (idx - m * nkji - k * nji - j * indcs.nx1) + is;
        k += ks;
        j += js;

        const Real x1v = CellCenterX(i - is, indcs.nx1, size.d_view(m).x1min,
                                     size.d_view(m).x1max);
        const Real x2v = CellCenterX(j - js, indcs.nx2, size.d_view(m).x2min,
                                     size.d_view(m).x2max);
        const Real x3v = CellCenterX(k - ks, indcs.nx3, size.d_view(m).x3min,
                                     size.d_view(m).x3max);

        // cylindrical radius
        const Real r_cylindrical = std::sqrt(x1v * x1v + x2v * x2v);
        // const Real r_spherical = std::sqrt(x1v * x1v + x2v * x2v + x3v *
        // x3v);

        // soften (regularize) the radius near r=0
        const Real inv_softened_r_cylindrical =
            compute_softened_inv_r(r_cylindrical, h, one_over_h);
        const Real softened_r_cylindrical = 1.0 / inv_softened_r_cylindrical;

        // compute surface density profile
        Real Sigma = pow(softened_r_cylindrical, rec_disk.exponent_p);

        // Define alias variables
        const Real R = softened_r_cylindrical;
        const Real one_over_R = inv_softened_r_cylindrical;
        const Real r_in = rec_disk.r_in;
        const Real r_out = rec_disk.r_out;
        const Real a_in = rec_disk.a_in;
        const Real a_out = rec_disk.a_out;

        // apply inner and outer truncation
        const Real f_in = 1.0 / (1.0 + exp(-(R - r_in) / a_in));
        const Real f_out = 1.0 / (1.0 + exp(+(R - r_out) / a_out));
        Sigma *= f_in * f_out;

        const Real dLogSigma_dLogR = rec_disk.exponent_p +
                                     (R / a_in) * (1.0 - f_in) -
                                     (R / a_out) * (1.0 - f_out);

        Real aspect_ratio, H;
        if (rec_disk.constant_scale_height) {
          H = rec_disk.scale_height_parameter;
          aspect_ratio = H / R;
        } else {
          aspect_ratio = rec_disk.scale_height_parameter;
          H = aspect_ratio * R;
        }

        const Real dens =
            Sigma * exp(-0.5 * SQR(x3v / H)) / (H * sqrt(2.0 * M_PI));
        const Real sound_speed_squared =
            inv_softened_r_cylindrical * SQR(H * inv_softened_r_cylindrical);
        const Real pressure = dens * sound_speed_squared;

        // rotation profile
        Real vphi_squared =
            inv_softened_r_cylindrical; // leading order (Keplerian rotation)
        if (rec_disk.add_rotation_corrections) {
          vphi_squared += -1.5 * SQR(x3v * inv_softened_r_cylindrical) *
                          inv_softened_r_cylindrical;
          if (rec_disk.constant_scale_height) {
            vphi_squared += sound_speed_squared * (-3.0 + dLogSigma_dLogR);
          } else {
            vphi_squared +=
                sound_speed_squared * (-2.0 + SQR(x3v / H) + dLogSigma_dLogR);
          }
        }
        // const Real vphi = sqrt(vphi_squared);
        const Real vphi = sqrt(fmax(0.0, vphi_squared));

        Real vx, vy, vz;
        vx = -vphi * x2v * inv_softened_r_cylindrical;
        vy = vphi * x1v * inv_softened_r_cylindrical;
        vz = 0.0;

        if (rec_disk.add_kick) {
          vx -= rec_disk.kick_magnitude * cos(rec_disk.kick_angle);
          vz = -rec_disk.kick_magnitude * sin(rec_disk.kick_angle);
        }

        if (dens > rec_disk.density_floor) {

          u0_(m, IDN, k, j, i) = dens;
          u0_(m, IM1, k, j, i) = dens * vx;
          u0_(m, IM2, k, j, i) = dens * vy;
          u0_(m, IM3, k, j, i) = dens * vz;

          u0_(m, IEN, k, j, i) =
              pressure / gm1 + 0.5 * dens * (SQR(vx) + SQR(vy) + SQR(vz));
        } else {
          u0_(m, IDN, k, j, i) = rec_disk.density_floor;
          u0_(m, IM1, k, j, i) = 0;
          u0_(m, IM2, k, j, i) = 0;
          u0_(m, IM3, k, j, i) = 0;
          u0_(m, IEN, k, j, i) = pfloor / gm1;
        }

        // reduction: track minimum cell spacing per axis across all blocks
        local_min_dx1 = fmin(local_min_dx1, size.d_view(m).dx1);
        local_min_dx2 = fmin(local_min_dx2, size.d_view(m).dx2);
        local_min_dx3 = fmin(local_min_dx3, size.d_view(m).dx3);
      },
      Kokkos::Min<Real>(min_dx1), Kokkos::Min<Real>(min_dx2),
      Kokkos::Min<Real>(min_dx3));

#if MPI_PARALLEL_ENABLED
  Real min_dxs[3] = {min_dx1, min_dx2, min_dx3};
  MPI_Allreduce(MPI_IN_PLACE, min_dxs, 3, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
  min_dx1 = min_dxs[0];
  min_dx2 = min_dxs[1];
  min_dx3 = min_dxs[2];
#endif

  if (global_variable::my_rank == 0) {
    std::cout << "\n Gravitational softening of 1/r : "
              << "\n   - softening length h  = " << h
              << "\n   - min dx1 (x) = " << min_dx1
              << "   ->    h/dx1 = " << h / min_dx1
              << "\n   - min dx2 (y) = " << min_dx2
              << "   ->    h/dx2 = " << h / min_dx2
              << "\n   - min dx3 (z) = " << min_dx3
              << "   ->    h/dx3 = " << h / min_dx3 << std::endl;
  }

  return;
}

KOKKOS_INLINE_FUNCTION
Real compute_softened_inv_r(const Real radius, const Real h,
                            const Real one_over_h) {
  if (radius >= h) {
    return 1.0 / radius;
  } else {
    const Real u = radius * one_over_h;
    const Real u2 = SQR(u);
    // Horner's method for polynomial evaluation
    if (radius < 0.5 * h) {
      return one_over_h *
             (2.8 + u2 * (-5.333333333333333 + u2 * (9.6 - 6.4 * u)));
    } else {
      return (-0.06666666666666667 +
              u * (3.2 +
                   u2 * (-10.666666666666666 +
                         u * (16.0 + u * (-9.6 + 2.1333333333333333 * u))))) /
             radius;
    }
  }
}

//
// @YK: History variables
//
void RecoiledDiskHistory(HistoryData *pdata, Mesh *pm) {
  // MeshBlockPack *pmbp = pm->pmb_pack;

  pdata->nhist = 7;
  pdata->label[0] = "M";
  pdata->label[1] = "Lx";
  pdata->label[2] = "Ly";
  pdata->label[3] = "Lz";
  pdata->label[4] = "Eint";
  pdata->label[5] = "KE";
  pdata->label[6] = "PE";

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int nx1 = indcs.nx1;
  int js = indcs.js;
  int nx2 = indcs.nx2;
  int ks = indcs.ks;
  int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack) * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;

  // capture class variables for kernel
  auto &u0_ = pm->pmb_pack->phydro->u0;
  auto &w0_ = pm->pmb_pack->phydro->w0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhist_ = pdata->nhist;

  auto rec_disk = recoiled_disk;
  const Real h = rec_disk.gravity_softening_length;
  const Real one_over_h = 1.0 / h;

  array_sum::GlobalSum sum_this_mb;

  Kokkos::parallel_reduce(
      "recoiled_disk_history", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
        // compute n,k,j,i indices of thread
        int m = (idx) / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        int i = (idx - m * nkji - k * nji - j * nx1) + is;
        k += ks;
        j += js;

        Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;

        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        const Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        const Real x2v = CellCenterX(j - js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        const Real x3v = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

        array_sum::GlobalSum hvars;

        // total mass
        hvars.the_array[0] = vol * u0_(m, IDN, k, j, i);

        // total angular momentum
        const Real px = u0_(m, IM1, k, j, i);
        const Real py = u0_(m, IM2, k, j, i);
        const Real pz = u0_(m, IM3, k, j, i);
        hvars.the_array[1] = vol * (x2v * pz - x3v * py);
        hvars.the_array[2] = vol * (x3v * px - x1v * pz);
        hvars.the_array[3] = vol * (x1v * py - x2v * px);

        // total internal energy
        hvars.the_array[4] = vol * w0_(m, IEN, k, j, i);

        // total kinetic energy
        hvars.the_array[5] =
            0.5 * vol * u0_(m, IDN, k, j, i) *
            (SQR(w0_(m, IVX, k, j, i)) + SQR(w0_(m, IVY, k, j, i)) +
             SQR(w0_(m, IVZ, k, j, i)));

        // total gravitational potential energy
        const Real radius = std::sqrt(x1v * x1v + x2v * x2v + x3v * x3v);
        const Real softened_inv_r =
            compute_softened_inv_r(radius, h, one_over_h);
        hvars.the_array[6] = -vol * u0_(m, IDN, k, j, i) * softened_inv_r;

        // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
        for (int n = nhist_; n < NHISTORY_VARIABLES; ++n) {
          hvars.the_array[n] = 0.0;
        }

        // sum into parallel reduce
        mb_sum += hvars;
      },
      Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));

  // store data into hdata array
  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }

  return;
}

//
// @YK: Hydro no-inflow boundary condition
//
//  (Apr 2026) Only supports for hydro case. Look gr_torus.cpp for possible
//  extension to MHD or radiation
//
void HydroNoInflow(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pm->pmb_pack->phydro != nullptr) {
    u0_ = pm->pmb_pack->phydro->u0;
    w0_ = pm->pmb_pack->phydro->w0;
  } else if (pm->pmb_pack->pmhd != nullptr) {
    u0_ = pm->pmb_pack->pmhd->u0;
    w0_ = pm->pmb_pack->pmhd->w0;

    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " MHD should not be enabled for this problem! " << std::endl;
    exit(EXIT_FAILURE);
  }
  int nmb = pm->pmb_pack->nmb_thispack;
  int nvar = u0_.extent_int(1);

  // Determine if radiation is enabled
  const bool is_radiation_enabled = (pm->pmb_pack->prad != nullptr);
  if (is_radiation_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " Radiation should not be enabled for this problem! "
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // ConsToPrim over all X1 ghost zones *and* at the innermost/outermost
  // X1-active zones of Meshblocks, even if Meshblock face is not at the edge of
  // computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, is - ng, is, 0,
                                           (n2 - 1), 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, ie, ie + ng, 0,
                                           (n2 - 1), 0, (n3 - 1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " MHD should not be enabled for this problem! " << std::endl;
    exit(EXIT_FAILURE);
  }
  // Set X1-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for(
      "noinflow_hydro_x1", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), 0,
      (n3 - 1), 0, (n2 - 1), KOKKOS_LAMBDA(int m, int n, int k, int j) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
          for (int i = 0; i < ng; ++i) {
            if (n == (IVX)) {
              w0_(m, n, k, j, is - i - 1) = fmin(0.0, w0_(m, n, k, j, is));
            } else {
              w0_(m, n, k, j, is - i - 1) = w0_(m, n, k, j, is);
            }
          }
        }
        if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
          for (int i = 0; i < ng; ++i) {
            if (n == (IVX)) {
              w0_(m, n, k, j, ie + i + 1) = fmax(0.0, w0_(m, n, k, j, ie));
            } else {
              w0_(m, n, k, j, ie + i + 1) = w0_(m, n, k, j, ie);
            }
          }
        }
      });

  // PrimToCons on X1 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, is - ng, is - 1, 0,
                                           (n2 - 1), 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, ie + 1, ie + ng, 0,
                                           (n2 - 1), 0, (n3 - 1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " MHD should not be enabled for this problem! " << std::endl;
    exit(EXIT_FAILURE);
  }

  // X2-Boundary
  // Set X2-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for(
        "noinflow_field_x2", DevExeSpace(), 0, (nmb - 1), 0, (n3 - 1), 0,
        (n1 - 1), KOKKOS_LAMBDA(int m, int k, int i) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) {
            for (int j = 0; j < ng; ++j) {
              b0.x1f(m, k, js - j - 1, i) = b0.x1f(m, k, js, i);
              if (i == n1 - 1) {
                b0.x1f(m, k, js - j - 1, i + 1) = b0.x1f(m, k, js, i + 1);
              }
              b0.x2f(m, k, js - j - 1, i) = b0.x2f(m, k, js, i);
              b0.x3f(m, k, js - j - 1, i) = b0.x3f(m, k, js, i);
              if (k == n3 - 1) {
                b0.x3f(m, k + 1, js - j - 1, i) = b0.x3f(m, k + 1, js, i);
              }
            }
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
            for (int j = 0; j < ng; ++j) {
              b0.x1f(m, k, je + j + 1, i) = b0.x1f(m, k, je, i);
              if (i == n1 - 1) {
                b0.x1f(m, k, je + j + 1, i + 1) = b0.x1f(m, k, je, i + 1);
              }
              b0.x2f(m, k, je + j + 2, i) = b0.x2f(m, k, je + 1, i);
              b0.x3f(m, k, je + j + 1, i) = b0.x3f(m, k, je, i);
              if (k == n3 - 1) {
                b0.x3f(m, k + 1, je + j + 1, i) = b0.x3f(m, k + 1, je, i);
              }
            }
          }
        });
  }
  // ConsToPrim over all X2 ghost zones *and* at the innermost/outermost
  // X2-active zones of Meshblocks, even if Meshblock face is not at the edge of
  // computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, 0, (n1 - 1),
                                           js - ng, js, 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, 0, (n1 - 1), je,
                                           je + ng, 0, (n3 - 1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " MHD should not be enabled for this problem! " << std::endl;
    exit(EXIT_FAILURE);
  }
  // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for(
      "noinflow_hydro_x2", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), 0,
      (n3 - 1), 0, (n1 - 1), KOKKOS_LAMBDA(int m, int n, int k, int i) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) {
          for (int j = 0; j < ng; ++j) {
            if (n == (IVY)) {
              w0_(m, n, k, js - j - 1, i) = fmin(0.0, w0_(m, n, k, js, i));
            } else {
              w0_(m, n, k, js - j - 1, i) = w0_(m, n, k, js, i);
            }
          }
        }
        if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
          for (int j = 0; j < ng; ++j) {
            if (n == (IVY)) {
              w0_(m, n, k, je + j + 1, i) = fmax(0.0, w0_(m, n, k, je, i));
            } else {
              w0_(m, n, k, je + j + 1, i) = w0_(m, n, k, je, i);
            }
          }
        }
      });

  // PrimToCons on X2 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, 0, (n1 - 1), js - ng,
                                           js - 1, 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, 0, (n1 - 1), je + 1,
                                           je + ng, 0, (n3 - 1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " MHD should not be enabled for this problem! " << std::endl;
    exit(EXIT_FAILURE);
  }

  // X3-Boundary
  // Set X3-BCs on b0 if Meshblock face is at the edge of computational domain

  // ConsToPrim over all X3 ghost zones *and* at the innermost/outermost
  // X3-active zones of Meshblocks, even if Meshblock face is not at the edge of
  // computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, 0, (n1 - 1), 0,
                                           (n2 - 1), ks - ng, ks);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, 0, (n1 - 1), 0,
                                           (n2 - 1), ke, ke + ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " MHD should not be enabled for this problem! " << std::endl;
    exit(EXIT_FAILURE);
  }
  // Set X3-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for(
      "noinflow_hydro_x3", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), 0,
      (n2 - 1), 0, (n1 - 1), KOKKOS_LAMBDA(int m, int n, int j, int i) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user) {
          for (int k = 0; k < ng; ++k) {
            if (n == (IVZ)) {
              w0_(m, n, ks - k - 1, j, i) = fmin(0.0, w0_(m, n, ks, j, i));
            } else {
              w0_(m, n, ks - k - 1, j, i) = w0_(m, n, ks, j, i);
            }
          }
        }
        if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user) {
          for (int k = 0; k < ng; ++k) {
            if (n == (IVZ)) {
              w0_(m, n, ke + k + 1, j, i) = fmax(0.0, w0_(m, n, ke, j, i));
            } else {
              w0_(m, n, ke + k + 1, j, i) = w0_(m, n, ke, j, i);
            }
          }
        }
      });

  // PrimToCons on X3 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, 0, (n1 - 1), 0, (n2 - 1),
                                           ks - ng, ks - 1);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, 0, (n1 - 1), 0, (n2 - 1),
                                           ke + 1, ke + ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " MHD should not be enabled for this problem! " << std::endl;
    exit(EXIT_FAILURE);
  }

  return;
}
