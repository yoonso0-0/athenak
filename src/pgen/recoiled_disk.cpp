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
