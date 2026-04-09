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
  Real r_cavity;
  Real r_out;
  Real truncation_length_out;
  Real power_exponent_p;
  Real delta0;

  //
  Real scale_height;

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
    std::cout
        << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl
        << "GR torus problem can only be run when GR defined in <coord> block"
        << std::endl;
    exit(EXIT_FAILURE);
  }

  // @YK : setting up boundary condition
  user_bcs_func = HydroNoInflow;

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

  recoiled_disk.r_cavity = pin->GetReal("problem", "r_cavity");
  recoiled_disk.r_out = pin->GetReal("problem", "r_out");
  recoiled_disk.truncation_length_out =
      pin->GetReal("problem", "truncation_length_out");
  recoiled_disk.power_exponent_p = pin->GetReal("problem", "power_exponent_p");
  recoiled_disk.delta0 = pin->GetReal("problem", "delta0");

  recoiled_disk.scale_height = pin->GetReal("problem", "scale_height");

  recoiled_disk.add_kick = pin->GetBoolean("problem", "add_kick");
  recoiled_disk.kick_magnitude = pin->GetReal("problem", "kick_magnitude");
  recoiled_disk.kick_angle = pin->GetReal("problem", "kick_angle");

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // const int nmkji = (pmbp->nmb_thispack) * indcs.nx3 * indcs.nx2 * indcs.nx1;
  //   const int nkji = indcs.nx3 * indcs.nx2 * indcs.nx1;
  //   const int nji = indcs.nx2 * indcs.nx1;

  //
  auto rec_disk = recoiled_disk;

  par_for(
      "recoiled_disk_3d", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke,
      js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(i - is, nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2v = CellCenterX(j - js, nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        int nx3 = indcs.nx3;
        Real x3v = CellCenterX(k - ks, nx3, x3min, x3max);

        const Real r_cylindrical = std::sqrt(x1v * x1v + x2v * x2v);
        const Real r_squared = x1v * x1v + x2v * x2v + x3v * x3v;
        const Real radius = std::sqrt(r_squared);
        const Real omega_kepler =
            1 / (r_cylindrical * std::sqrt(r_cylindrical));

        // mass density distribution
        const Real delta0 = rec_disk.delta0;

        Real dens;

        if (r_squared < 1.0e-10) {
          dens = delta0;
        } else {
          dens = (1.0 - delta0) *
                     exp(-pow(rec_disk.r_cavity / r_cylindrical, 12.0)) +
                 delta0;
        }

        // Outer truncation
        dens *= 1 - 1.0 / (1 + exp(-(r_cylindrical - rec_disk.r_out) /
                                   rec_disk.truncation_length_out));

        // scale height, sound speed
        dens *= std::exp(-0.5 * SQR(x3v) / SQR(rec_disk.scale_height));
        const Real sound_speed = rec_disk.scale_height * omega_kepler;

        // velocity profile
        Real vx, vy, vz;

        if (r_squared < 1.0e-10) {
          vx = 0.0;
          vy = 0.0;
          vz = 0.0;

        } else {
          const Real v_kepler = 1 / r_cylindrical;
          const Real vphi = v_kepler;

          vx = -vphi * x2v / r_cylindrical;
          vy = vphi * x1v / r_cylindrical;
          vz = 0.0;
        }

        if (rec_disk.add_kick) {
          vx -= rec_disk.kick_magnitude * std::cos(rec_disk.kick_angle);
          vz = -rec_disk.kick_magnitude * std::sin(rec_disk.kick_angle);
        }

        u0_(m, IDN, k, j, i) = dens;
        u0_(m, IM1, k, j, i) = dens * vx;
        u0_(m, IM2, k, j, i) = dens * vy;
        u0_(m, IM3, k, j, i) = dens * vz;

        const Real pressure = dens * SQR(sound_speed);

        u0_(m, IEN, k, j, i) =
            pressure / gm1 +
            0.5 * (SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                   SQR(u0_(m, IM3, k, j, i)));
      });

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

//
// @YK: History variables
//
void RecoiledDiskHistory(HistoryData *pdata, Mesh *pm) {
  // MeshBlockPack *pmbp = pm->pmb_pack;

  pdata->nhist = 3;
  pdata->label[0] = "Lx";
  pdata->label[1] = "Ly";
  pdata->label[2] = "Lz";

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

  array_sum::GlobalSum sum_this_mb;

  Kokkos::parallel_reduce(
      "gravitational_drag", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
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

        // angular momentum
        array_sum::GlobalSum hvars;

        const Real px = u0_(m, IM1, k, j, i);
        const Real py = u0_(m, IM2, k, j, i);
        const Real pz = u0_(m, IM3, k, j, i);

        // l_x
        hvars.the_array[0] = vol * (x2v * pz - x3v * py);
        hvars.the_array[1] = vol * (x3v * px - x1v * pz);
        hvars.the_array[2] = vol * (x1v * py - x2v * px);

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
