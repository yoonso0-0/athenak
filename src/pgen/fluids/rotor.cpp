//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rotor.cpp
//  \brief Problem generator for a 2D MHD rotor: a dense disk in rigid rotation
//  embedded in a static, uniform ambient medium threaded by a uniform field
//  along x1.
//
//  Initial conditions (defaults):
//    ambient:   d = 1, P = 1, v = 0
//    r <= r0:   d = 10, v = (-omega*y, omega*x, 0), with r0 = 0.1, omega = 9.95
//    everywhere B = (1, 0, 0)
//  Works in Newtonian or special-relativistic MHD (<coord> special_rel = true);
//  in SR this is the relativistic rotor of Del Zanna et al. (2003), with max
//  |v| = 0.995. The disk edge is sharp (no taper). The setup is invariant under
//  rotation by pi about the origin, so cell centers are computed with the
//  bitwise-antisymmetric form.
//
// REFERENCES:
//   D. Balsara & D. Spicer, "A staggered mesh algorithm using high order Godunov
//   fluxes to ensure solenoidal magnetic fields in MHD simulations", JCP, 149,
//   270 (1999)
//   L. Del Zanna, N. Bucciantini & P. Londrillo, "An efficient shock-capturing
//   central-type scheme for multidimensional relativistic flows II.
//   Magnetohydrodynamics", A&A, 400, 397 (2003)
//========================================================================================

// C++ headers
#include <cmath>
#include <iostream> // endl

// Athena++ headers
#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//  \brief Problem Generator for the 2D MHD rotor

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart)
    return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Rotor test can only be run in MHD, but no <mhd> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!(pmy_mesh_->two_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Rotor test must be run in 2D (nx2 > 1, nx3 = 1)" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd->peos->eos_data.is_ideal == false) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Rotor test requires an ideal (adiabatic) EOS" << std::endl;
    exit(EXIT_FAILURE);
  }

  // problem parameters
  Real d_amb = pin->GetOrAddReal("problem", "d_amb", 1.0);
  Real p_amb = pin->GetOrAddReal("problem", "p_amb", 1.0);
  Real d_rot = pin->GetOrAddReal("problem", "d_rot", 10.0);
  Real omega = pin->GetOrAddReal("problem", "omega", 9.95);
  Real r0 = pin->GetOrAddReal("problem", "r0", 0.1);
  Real bx = pin->GetOrAddReal("problem", "bx", 1.0);
  if (pmbp->pcoord->is_special_relativistic && omega * r0 >= 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Rotor edge speed omega*r0 = " << omega * r0
              << " must be < 1 in SR" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;

  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &size = pmbp->pmb->mb_size;

  // mesh bounds and root-level cell counts, used to compute cell centers that
  // are bitwise antisymmetric about the mesh center
  auto &mblev = pmbp->pmb->mb_lev;
  int root_lev = pmy_mesh_->root_level;
  int nx1_root = pmy_mesh_->mesh_indcs.nx1;
  int nx2_root = pmy_mesh_->mesh_indcs.nx2;
  Real x1min_mesh = pmy_mesh_->mesh_size.x1min;
  Real x1max_mesh = pmy_mesh_->mesh_size.x1max;
  Real x2min_mesh = pmy_mesh_->mesh_size.x2min;
  Real x2max_mesh = pmy_mesh_->mesh_size.x2max;

  // in SR, primitive velocities are the spatial components of the 4-velocity
  // u^i = W v^i
  bool is_sr = pmbp->pcoord->is_special_relativistic;
  auto &w0 = pmbp->pmhd->w0;
  auto &bcc0 = pmbp->pmhd->bcc0;

  par_for(
      "pgen_rotor", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je,
      is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        int lev = mblev.d_view(m);
        Real &x1min = size.d_view(m).x1min;
        int ntot1 = nx1_root << (lev - root_lev);
        Real x1v =
            SymmetricCellCenterX(i - is, x1min, x1min_mesh, x1max_mesh, ntot1);

        Real &x2min = size.d_view(m).x2min;
        int ntot2 = nx2_root << (lev - root_lev);
        Real x2v =
            SymmetricCellCenterX(j - js, x2min, x2min_mesh, x2max_mesh, ntot2);

        Real rad = std::sqrt(SQR(x1v) + SQR(x2v));

        Real den = d_amb;
        Real v1 = 0.0;
        Real v2 = 0.0;
        if (rad <= r0) {
          den = d_rot;
          v1 = -omega * x2v;
          v2 = omega * x1v;
        }
        Real lor = 1.0;
        if (is_sr) {
          lor = 1.0 / std::sqrt(1.0 - (SQR(v1) + SQR(v2)));
        }

        w0(m, IDN, k, j, i) = den;
        w0(m, IVX, k, j, i) = lor * v1;
        w0(m, IVY, k, j, i) = lor * v2;
        w0(m, IVZ, k, j, i) = 0.0;
        w0(m, IEN, k, j, i) = p_amb / gm1;

        // uniform face-centered field; div(B)=0 trivially
        b0.x1f(m, k, j, i) = bx;
        b0.x2f(m, k, j, i) = 0.0;
        b0.x3f(m, k, j, i) = 0.0;
        if (i == ie)
          b0.x1f(m, k, j, i + 1) = bx;
        if (j == je)
          b0.x2f(m, k, j + 1, i) = 0.0;
        if (k == ke)
          b0.x3f(m, k + 1, j, i) = 0.0;
        bcc0(m, IBX, k, j, i) = bx;
        bcc0(m, IBY, k, j, i) = 0.0;
        bcc0(m, IBZ, k, j, i) = 0.0;
      });

  // Convert primitives to conserved
  pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);

  return;
}
