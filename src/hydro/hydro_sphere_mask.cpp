//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_sphere_mask.cpp
//! \brief Implements a static spherical mask centered at the coordinate origin that
//! overwrites the primitive (and consistently the conserved) hydro state in every cell
//! with r < radius, every stage, so the sphere's surface behaves like an effective
//! Dirichlet, Reflecting, Absorbing, or Spherical Wind (prescribed radial
//! inflow/outflow) boundary condition as seen by the exterior flow. Restricted to
//! Newtonian (non-SR, non-GR) Hydro.
//!
//! The mask is fully local: no extra MPI, no cross-rank communication, and no reads
//! outside the MeshBlock a masked cell lives in. The only global requirement is a
//! non-adaptive mesh, so the set of blocks holding masked cells is fixed for the run.
//!
//! The sphere may be split across MeshBlocks. Every block whose active zone comes within
//! radius of the origin masks its own share of the sphere, so a root grid whose block
//! boundaries meet at the origin -- the 2^N-blocks-per-axis layouts, on a domain
//! symmetric about 0 -- cuts the sphere into 2/4/8 pieces handled independently and in
//! parallel. Two things make that work.
//!
//! First, the mirror map used by Reflecting/Absorbing,
//!     x -> x * (2*radius - r)/r,
//! has (2*radius - r)/r > 1 for every masked cell (r < radius): it pushes the point
//! radially OUTWARD along its own ray, preserving the sign of every component. A masked
//! cell therefore mirrors into the same octant that it sits in itself. The interpolation
//! stencil is made one-sided at coordinate planes, so it likewise never reads the
//! neighbouring octant when a MeshBlock boundary lies on a plane through the origin.
//!
//! Second, the mask is applied BEFORE the halo exchange (Hydro::MaskSpherePre, scheduled
//! right after HydroSrcTerms; see hydro_tasks.cpp). A block's ghost zone holds its
//! neighbours' share of the sphere, and those ghost values are not inert: they are read
//! by the reconstruction stencil of the UNMASKED cells just outside r=radius, which is
//! precisely how the sphere acts on the exterior flow. Masking only at the end of the
//! stage would leave those copies one stage stale and contaminate the real solution in a
//! thin shell where a block boundary cuts the sphere's surface. Masking first makes
//! SendU/RecvU carry the finished values, so nothing downstream ever has to reconstruct
//! a neighbour's masked cells.
//!
//! The same ordering matters at PHYSICAL boundaries, not just internal ones.
//! ApplyPhysicalBCs runs after MaskSpherePre, so a reflect/outflow boundary that cuts the
//! masked sphere -- as it does on the octant domains where the origin sits at a corner of
//! the mesh -- now fills its ghost cells from cells that have already been masked.
//! Applying the mask only at the end of the stage instead reflects updated-but-unmasked
//! cells into those ghosts while the cells they mirror are pinned, an inconsistency worth
//! about 1% in every variable on the 3D octant test problems.
//!
//! A second pass (Hydro::MaskSphere) then runs at the end of the stage, for the pointwise
//! BCs only, so w0 holds the prescribed state exactly rather than a round-trip through
//! PrimToCons/ConsToPrim. Reflecting/Absorbing must not take it; see MaskSphere below.
//!
//! What a split costs is a one-sided reach requirement, checked per block: a block
//! holding masked cells must extend at least 2*radius from the origin along each
//! direction it actually owns cells in (a block with a face on the origin owns nothing on
//! the far side and so needs nothing there). The factor of 2 is tight: a masked cell near
//! the axis at r->0 mirrors out to r'->2*radius, and the whole bilinear/trilinear stencil
//! around that point must be in the block's active zone. This applies to
//! Reflecting/Absorbing only. Dirichlet and Spherical Wind set the masked state from the
//! cell's own position alone, never interpolating, so they place no constraint on the
//! block layout at all.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "hydro.hpp"

namespace hydro {

// Return a cell center through the global mesh coordinates rather than a MeshBlock's
// local endpoints. Equivalent cells then have bitwise-identical coordinates for every
// MeshBlock decomposition. The rounded offset is exact for the required static mesh.
KOKKOS_INLINE_FUNCTION
Real SphereMaskCellCenter(int i, int is, Real block_min, Real dx,
                          Real mesh_min, Real mesh_max) {
  int global_n = static_cast<int>(floor((mesh_max-mesh_min)/dx + 0.5));
  int block_offset = static_cast<int>(floor((block_min-mesh_min)/dx + 0.5));
  return CellCenterX(block_offset+i-is, global_n, mesh_min, mesh_max);
}

KOKKOS_INLINE_FUNCTION
Real SphereMaskWeight(int i, int is, Real block_min, Real dx,
                      Real mesh_min, Real xm) {
  int block_offset = static_cast<int>(floor((block_min-mesh_min)/dx + 0.5));
  int global_i = block_offset+i-is;
  Real xi = (xm-mesh_min)/dx - 0.5;
  int global_i0 = static_cast<int>(floor(xi));
  Real frac = xi-static_cast<Real>(global_i0);
  if (global_i == global_i0) { return 1.0-frac; }
  if (global_i == global_i0+1) { return frac; }
  return 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real SphereMaskInterp()
//! \brief Interpolates hydro variable n at a mirror point from the surrounding 2/4/8
//! cell centers. It excludes every center inside the mask and renormalizes the retained
//! multilinear weights. Thus the effective boundary state depends only on the physical
//! exterior flow. Global cell-index weights make the result independent of how the
//! static mesh is partitioned into MeshBlocks. The caller guarantees at least one cell.

KOKKOS_INLINE_FUNCTION
Real SphereMaskInterp(const DvceArray5D<Real> &w0_, int m, int n,
                       int k0, int k1, int j0, int j1, int i0, int i1,
                       bool multi_d, bool three_d,
                       int is, int js, int ks, Real x1min, Real x2min, Real x3min,
                       Real dx1, Real dx2, Real dx3, Real xm1, Real xm2, Real xm3,
                       Real rad, Real mx1min, Real mx1max, Real mx2min, Real mx2max,
                       Real mx3min, Real mx3max) {
  Real interp = 0.0;
  Real weight_sum = 0.0;
  int nk = three_d ? 2 : 1;
  int nj = multi_d ? 2 : 1;
  for (int dk = 0; dk < nk; ++dk) {
    int kk = (dk == 0) ? k0 : k1;
    Real x3 = three_d ? SphereMaskCellCenter(kk, ks, x3min, dx3, mx3min, mx3max) : 0.0;
    Real wz = three_d ? SphereMaskWeight(kk, ks, x3min, dx3, mx3min, xm3) : 1.0;
    for (int dj = 0; dj < nj; ++dj) {
      int jj = (dj == 0) ? j0 : j1;
      Real x2 = multi_d ? SphereMaskCellCenter(jj, js, x2min, dx2, mx2min, mx2max) : 0.0;
      Real wy = multi_d ? SphereMaskWeight(jj, js, x2min, dx2, mx2min, xm2) : 1.0;
      for (int di = 0; di < 2; ++di) {
        int ii = (di == 0) ? i0 : i1;
        Real x1 = SphereMaskCellCenter(ii, is, x1min, dx1, mx1min, mx1max);
        Real wx = SphereMaskWeight(ii, is, x1min, dx1, mx1min, xm1);
        Real weight = wx*wy*wz;
        if (weight > 0.0 && x1*x1 + x2*x2 + x3*x3 >= rad*rad) {
          // Do not even read excluded cells: zero*NaN would still contaminate interp.
          interp += weight*w0_(m,n,kk,jj,ii);
          weight_sum += weight;
        }
      }
    }
  }
  return interp/weight_sum;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::InitSphereMask()
//! \brief Parses the <sphere_mask> input block, collects the MeshBlocks on this rank
//! that hold masked cells, and validates that the feature can run fully locally: mesh
//! must be non-adaptive, and (for Reflecting/Absorbing) each of those blocks must reach
//! 2*radius+margin from the origin along every direction it extends in.

void Hydro::InitSphereMask(ParameterInput *pin) {
  if (pmy_pack->pcoord->is_special_relativistic ||
      pmy_pack->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<sphere_mask> is only implemented for Newtonian "
              << "(non-SR, non-GR) Hydro" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  sphere_mask_radius = pin->GetReal("sphere_mask","radius");
  if (!(sphere_mask_radius > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<sphere_mask>/radius must be positive, got "
              << sphere_mask_radius << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string bc_str = pin->GetString("sphere_mask","bc");
  if (bc_str.compare("dirichlet") == 0) {
    sphere_mask_bc = SphereMaskBC::dirichlet;
  } else if (bc_str.compare("reflecting") == 0) {
    sphere_mask_bc = SphereMaskBC::reflecting;
  } else if (bc_str.compare("absorbing") == 0) {
    sphere_mask_bc = SphereMaskBC::absorbing;
  } else if (bc_str.compare("spherical_wind") == 0) {
    sphere_mask_bc = SphereMaskBC::spherical_wind;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<sphere_mask>/bc = '" << bc_str << "' not recognized; "
              << "must be one of dirichlet|reflecting|absorbing|spherical_wind"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  sm_needs_mirror = (sphere_mask_bc == SphereMaskBC::reflecting ||
                     sphere_mask_bc == SphereMaskBC::absorbing);

  // Dirichlet target state (unused for reflecting/absorbing/spherical_wind)
  sm_dens = pin->GetOrAddReal("sphere_mask","dens",0.0);
  sm_vel1 = pin->GetOrAddReal("sphere_mask","vel1",0.0);
  sm_vel2 = pin->GetOrAddReal("sphere_mask","vel2",0.0);
  sm_vel3 = pin->GetOrAddReal("sphere_mask","vel3",0.0);
  sm_eint = pin->GetOrAddReal("sphere_mask","eint",0.0);
  // spherical_wind target radial velocity (unused otherwise); sign: >0 outward (wind),
  // <0 inward (accretion)
  sm_velr = pin->GetOrAddReal("sphere_mask","vel_r",0.0);
  if (sphere_mask_bc == SphereMaskBC::dirichlet ||
      sphere_mask_bc == SphereMaskBC::spherical_wind) {
    if (!(sm_dens > 0.0)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<sphere_mask>/dens must be positive for bc=" << bc_str
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (peos->eos_data.is_ideal && !(sm_eint > 0.0)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<sphere_mask>/eint must be positive for bc=" << bc_str
                << " with an ideal-gas EOS" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // require a static mesh: with AMR the origin's owning MeshBlock (and its neighbors)
  // can change at any time, breaking the single-block-containment guarantee below
  if (pmy_pack->pmesh->adaptive) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<sphere_mask> requires a non-adaptive (static) mesh"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  bool multi_d = pmy_pack->pmesh->multi_d;
  bool three_d = pmy_pack->pmesh->three_d;
  auto &msz = pmy_pack->pmesh->mesh_size;

  // sanity check the origin actually lies in the mesh's domain along every active axis
  bool origin_in_mesh = (msz.x1min <= 0.0 && msz.x1max > 0.0) &&
      (!multi_d || (msz.x2min <= 0.0 && msz.x2max > 0.0)) &&
      (!three_d || (msz.x3min <= 0.0 && msz.x3max > 0.0));
  if (!origin_in_mesh) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<sphere_mask> is enabled but the coordinate origin does "
              << "not lie within the mesh domain" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Collect every MeshBlock on this rank that holds at least one masked cell, i.e. whose
  // active zone comes within radius of the origin. Blocks are selected by the distance
  // from the origin to their (closed) bounding box, so a block whose face or corner lies
  // exactly on the origin is included along with its neighbours across that boundary:
  // each then masks its own half/quadrant/octant of the sphere. No half-open tie-break
  // is needed or wanted -- the pieces are disjoint because the cells are.
  std::vector<int> mask_mbs;
  auto &mb_size = pmy_pack->pmb->mb_size;
  int nmb = pmy_pack->nmb_thispack;
  for (int m = 0; m < nmb; ++m) {
    auto &sz = mb_size.h_view(m);
    // per-axis distance from the origin to the block's bounding box (0 if it straddles 0)
    Real d1 = std::max(std::max(sz.x1min, -sz.x1max), 0.0);
    Real d2 = multi_d ? std::max(std::max(sz.x2min, -sz.x2max), 0.0) : 0.0;
    Real d3 = three_d ? std::max(std::max(sz.x3min, -sz.x3max), 0.0) : 0.0;
    if (d1*d1 + d2*d2 + d3*d3 < sphere_mask_radius*sphere_mask_radius) {
      mask_mbs.push_back(m);
    }
  }
  nmb_mask = static_cast<int>(mask_mbs.size());
  if (nmb_mask == 0) { return; }  // no part of the sphere on this rank: nothing to do

  // The mirror map has no radial direction at an exact origin cell. Reject that layout
  // for the two mirror BCs rather than choose an arbitrary, symmetry-breaking direction.
  if (sm_needs_mirror) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    for (int n = 0; n < nmb_mask; ++n) {
      auto &sz = mb_size.h_view(mask_mbs[n]);
      bool x1_zero = false, x2_zero = !multi_d, x3_zero = !three_d;
      for (int i = indcs.is; i <= indcs.ie; ++i) {
        x1_zero = x1_zero ||
            (SphereMaskCellCenter(i, indcs.is, sz.x1min, sz.dx1,
                                  msz.x1min, msz.x1max) == 0.0);
      }
      for (int j = indcs.js; multi_d && j <= indcs.je; ++j) {
        x2_zero = x2_zero ||
            (SphereMaskCellCenter(j, indcs.js, sz.x2min, sz.dx2,
                                  msz.x2min, msz.x2max) == 0.0);
      }
      for (int k = indcs.ks; three_d && k <= indcs.ke; ++k) {
        x3_zero = x3_zero ||
            (SphereMaskCellCenter(k, indcs.ks, sz.x3min, sz.dx3,
                                  msz.x3min, msz.x3max) == 0.0);
      }
      if (x1_zero && x2_zero && x3_zero) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<sphere_mask>: bc=" << bc_str << " is undefined "
                  << "for a cell centered exactly at the coordinate origin. Use an "
                  << "even number of cells across the origin or a pointwise bc "
                  << "(dirichlet|spherical_wind)." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  // Reflecting/Absorbing read a mirror point up to 2*radius from the origin. Because the
  // mirror map never crosses a coordinate plane (see the notes at the top), that point is
  // always in the same MeshBlock as the cell it belongs to -- provided the block reaches
  // far enough. So require each masked block to extend 2*radius+margin from the origin
  // along every direction it actually owns cells in. A block with a face on the origin
  // owns nothing on the far side and is asked for nothing there, which is exactly what
  // lets a 2^N root grid split the sphere: each piece only ever mirrors into its own
  // octant. Dirichlet/Spherical Wind interpolate nothing and are unconstrained.
  if (sm_needs_mirror) {
    for (int n = 0; n < nmb_mask; ++n) {
      int m = mask_mbs[n];
      auto &sz = mb_size.h_view(m);
      Real margin = std::max(sz.dx1, std::max(multi_d ? sz.dx2 : 0.0,
                                               three_d ? sz.dx3 : 0.0));
      Real reach = 2.0*sphere_mask_radius + margin;
      // "owns cells on this side" means a cell CENTER past the origin, so test the block
      // edge against half a cell: an edge sitting on 0 up to roundoff owns nothing there
      bool ok = true;
      if (sz.x1max >  0.5*sz.dx1) { ok = ok && (sz.x1max >=  reach); }
      if (sz.x1min < -0.5*sz.dx1) { ok = ok && (sz.x1min <= -reach); }
      if (multi_d) {
        if (sz.x2max >  0.5*sz.dx2) { ok = ok && (sz.x2max >=  reach); }
        if (sz.x2min < -0.5*sz.dx2) { ok = ok && (sz.x2min <= -reach); }
      }
      if (three_d) {
        if (sz.x3max >  0.5*sz.dx3) { ok = ok && (sz.x3max >=  reach); }
        if (sz.x3min < -0.5*sz.dx3) { ok = ok && (sz.x3min <= -reach); }
      }
      if (!ok) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<sphere_mask>: bc=" << bc_str << " interpolates the "
                  << "exterior state at a mirror point up to 2*radius from the origin, "
                  << "but gid=" << pmy_pack->pmb->mb_gid.h_view(m) << " holds masked "
                  << "cells without extending at least 2*radius+margin=" << reach
                  << " from the origin along every direction it owns cells in, so that "
                  << "stencil would leave the MeshBlock. Use a larger/coarser MeshBlock, "
                  << "a smaller radius, or a pointwise bc (dirichlet|spherical_wind), "
                  << "which places no constraint on the block layout." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  // publish the block list to the device
  Kokkos::realloc(sphere_mask_mbs, nmb_mask);
  for (int n = 0; n < nmb_mask; ++n) { sphere_mask_mbs.h_view(n) = mask_mbs[n]; }
  sphere_mask_mbs.template modify<HostMemSpace>();
  sphere_mask_mbs.template sync<DevExeSpace>();

}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Hydro::MaskSpherePre()
//! \brief First of the stage's two mask passes, run before the halo exchange so that
//! SendU/RecvU carry finished masked values into every neighbour's ghost zone. This is
//! the pass that makes the masked state correct; see the notes at the top of this file.
//! Active zone only -- ghost cells are about to be overwritten by the exchange, or,
//! outside the mesh, by ApplyPhysicalBCs.

TaskStatus Hydro::MaskSpherePre(Driver *pdrive, int stage) {
  return MaskSphereImpl(true);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Hydro::MaskSphere()
//! \brief Second mask pass, at the end of the stage after the final ConToPrim, for the
//! pointwise BCs only. MaskSpherePre already put the right values in u0 and the exchange
//! distributed them, but ConToPrim then regenerated w0 from u0, so the prescribed
//! primitives arrived via a PrimToCons/ConsToPrim round-trip. Re-running a BC that
//! depends only on position restores them exactly, in the ghost zone as well.
//!
//! Reflecting/Absorbing deliberately skip this redundant pass: MaskSpherePre already
//! stored their reconstructed state consistently in u0 and w0, and the final ConToPrim
//! restores w0 from that u0. The pointwise BCs rerun only to preserve their prescribed
//! primitive values exactly through the PrimToCons/ConsToPrim round trip.

TaskStatus Hydro::MaskSphere(Driver *pdrive, int stage) {
  if (sm_needs_mirror) { return TaskStatus::complete; }
  return MaskSphereImpl(false);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Hydro::MaskSphereImpl()
//! \brief Shared body of the two passes above. Overwrites w0 (and consistently u0) for
//! every cell with r < sphere_mask_radius, in each MeshBlock on this rank that holds such
//! cells (a no-op everywhere else). When the sphere is split across blocks each block
//! masks its own piece independently, which reproduces masking it in one block because
//! the mirror map never crosses a block boundary.

TaskStatus Hydro::MaskSphereImpl(bool pre) {
  if (!use_sphere_mask || nmb_mask == 0) {
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
  bool multi_d = pmy_pack->pmesh->multi_d;
  bool three_d = pmy_pack->pmesh->three_d;

  auto &size = pmy_pack->pmb->mb_size;
  auto &w0_ = w0;
  auto &u0_ = u0;
  auto &eos = peos->eos_data;
  auto &mask_mbs_ = sphere_mask_mbs;
  int nmask = nmb_mask;
  Real rad = sphere_mask_radius;
  int ng = indcs.ng;
  auto &msz = pmy_pack->pmesh->mesh_size;
  Real mx1min = msz.x1min, mx1max = msz.x1max;
  Real mx2min = msz.x2min, mx2max = msz.x2max;
  Real mx3min = msz.x3min, mx3max = msz.x3max;
  SphereMaskBC bc = sphere_mask_bc;
  Real dens0 = sm_dens, v10 = sm_vel1, v20 = sm_vel2, v30 = sm_vel3, eint0 = sm_eint;
  Real velr0 = sm_velr;

  // The end-of-stage pass also pins ghost cells lying inside the mesh, so a block's ghost
  // copy of a neighbour's piece of the sphere holds the prescribed state exactly and not
  // a PrimToCons/ConsToPrim round-trip of it. It is only ever reached for the pointwise
  // BCs, which each block can evaluate anywhere from the cell's position alone. The
  // pre-exchange pass stays in the active zone: its ghost cells are about to be replaced
  // by SendU/RecvU regardless.
  bool fill_ghosts = !pre;
  int il = fill_ghosts ? is-ng : is;
  int iu = fill_ghosts ? ie+ng : ie;
  int jl = (fill_ghosts && multi_d) ? js-ng : js;
  int ju = (fill_ghosts && multi_d) ? je+ng : je;
  int kl = (fill_ghosts && three_d) ? ks-ng : ks;
  int ku = (fill_ghosts && three_d) ? ke+ng : ke;

  par_for("sphere_mask", DevExeSpace(), 0, nmask-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
    const int m = mask_mbs_.d_view(n);
    Real &x1min = size.d_view(m).x1min;
    Real &x2min = size.d_view(m).x2min;
    Real &x3min = size.d_view(m).x3min;

    Real dx1 = size.d_view(m).dx1;
    Real dx2 = multi_d ? size.d_view(m).dx2 : 0.0;
    Real dx3 = three_d ? size.d_view(m).dx3 : 0.0;
    Real x1 = SphereMaskCellCenter(i, is, x1min, dx1, mx1min, mx1max);
    Real x2 = multi_d ? SphereMaskCellCenter(j, js, x2min, dx2, mx2min, mx2max) : 0.0;
    Real x3 = three_d ? SphereMaskCellCenter(k, ks, x3min, dx3, mx3min, mx3max) : 0.0;
    Real rr = sqrt(x1*x1 + x2*x2 + x3*x3);
    if (rr >= rad) { return; }

    // ghost cells outside the global mesh are filled by a physical boundary condition
    // (reflect/outflow/periodic/user), not by a neighbouring MeshBlock: leave them to it
    if (x1 < mx1min || x1 > mx1max) { return; }
    if (multi_d && (x2 < mx2min || x2 > mx2max)) { return; }
    if (three_d && (x3 < mx3min || x3 > mx3max)) { return; }

    Real dens_new, v1_new, v2_new, v3_new, eint_new = 0.0;

    if (bc == SphereMaskBC::dirichlet || bc == SphereMaskBC::spherical_wind) {
      dens_new = dens0;
      if (bc == SphereMaskBC::dirichlet) {
        v1_new = v10;
        v2_new = v20;
        v3_new = v30;
      } else {
        // radial velocity field of fixed magnitude velr0 (sign: >0 outward/wind,
        // <0 inward/accretion); direction is along the local position vector
        Real rr_safe = fmax(rr, 1.0e-12*rad);
        Real rhat1 = x1/rr_safe;
        Real rhat2 = multi_d ? x2/rr_safe : 0.0;
        Real rhat3 = three_d ? x3/rr_safe : 0.0;
        v1_new = velr0*rhat1;
        v2_new = velr0*rhat2;
        v3_new = velr0*rhat3;
      }
      if (eos.is_ideal) { eint_new = eint0; }
    } else {
      // mirror point: x_mirror = x * (2*radius - r)/r, applied to active dims only
      Real rr_safe = fmax(rr, 1.0e-12*rad);
      Real s = (2.0*rad - rr_safe)/rr_safe;
      Real xm1 = x1*s;
      Real xm2 = multi_d ? x2*s : 0.0;
      Real xm3 = three_d ? x3*s : 0.0;

      int ioff = static_cast<int>(floor((x1min-mx1min)/dx1 + 0.5));
      Real xi1 = (xm1 - mx1min)/dx1 - 0.5;
      int i0 = static_cast<int>(floor(xi1)) - ioff;
      i0 = (i0 < 0) ? 0 : ((i0 > nx1-2) ? nx1-2 : i0);
      int i0b = i0+is, i1b = i0+1+is;

      int j0b = j, j1b = j;
      if (multi_d) {
        int joff = static_cast<int>(floor((x2min-mx2min)/dx2 + 0.5));
        Real xi2 = (xm2 - mx2min)/dx2 - 0.5;
        int j0 = static_cast<int>(floor(xi2)) - joff;
        j0 = (j0 < 0) ? 0 : ((j0 > nx2-2) ? nx2-2 : j0);
        j0b = j0+js; j1b = j0+1+js;
      }

      int k0b = k, k1b = k;
      if (three_d) {
        int koff = static_cast<int>(floor((x3min-mx3min)/dx3 + 0.5));
        Real xi3 = (xm3 - mx3min)/dx3 - 0.5;
        int k0 = static_cast<int>(floor(xi3)) - koff;
        k0 = (k0 < 0) ? 0 : ((k0 > nx3-2) ? nx3-2 : k0);
        k0b = k0+ks; k1b = k0+1+ks;
      }

      Real dens_m = SphereMaskInterp(w0_, m, IDN, k0b, k1b, j0b, j1b, i0b, i1b,
          multi_d, three_d, is, js, ks, x1min, x2min, x3min, dx1, dx2, dx3,
          xm1, xm2, xm3, rad, mx1min, mx1max, mx2min, mx2max, mx3min, mx3max);
      Real v1_m = SphereMaskInterp(w0_, m, IVX, k0b, k1b, j0b, j1b, i0b, i1b,
          multi_d, three_d, is, js, ks, x1min, x2min, x3min, dx1, dx2, dx3,
          xm1, xm2, xm3, rad, mx1min, mx1max, mx2min, mx2max, mx3min, mx3max);
      Real v2_m = SphereMaskInterp(w0_, m, IVY, k0b, k1b, j0b, j1b, i0b, i1b,
          multi_d, three_d, is, js, ks, x1min, x2min, x3min, dx1, dx2, dx3,
          xm1, xm2, xm3, rad, mx1min, mx1max, mx2min, mx2max, mx3min, mx3max);
      Real v3_m = SphereMaskInterp(w0_, m, IVZ, k0b, k1b, j0b, j1b, i0b, i1b,
          multi_d, three_d, is, js, ks, x1min, x2min, x3min, dx1, dx2, dx3,
          xm1, xm2, xm3, rad, mx1min, mx1max, mx2min, mx2max, mx3min, mx3max);
      Real eint_m = eos.is_ideal ? SphereMaskInterp(w0_, m, IEN, k0b, k1b, j0b, j1b,
          i0b, i1b, multi_d, three_d, is, js, ks, x1min, x2min, x3min,
          dx1, dx2, dx3, xm1, xm2, xm3, rad, mx1min, mx1max, mx2min, mx2max,
          mx3min, mx3max) : 0.0;

      dens_new = dens_m;
      eint_new = eint_m;
      if (bc == SphereMaskBC::reflecting) {
        Real rhat1 = x1/rr_safe;
        Real rhat2 = multi_d ? x2/rr_safe : 0.0;
        Real rhat3 = three_d ? x3/rr_safe : 0.0;
        Real vdotr = v1_m*rhat1 + v2_m*rhat2 + v3_m*rhat3;
        v1_new = v1_m - 2.0*vdotr*rhat1;
        v2_new = v2_m - 2.0*vdotr*rhat2;
        v3_new = v3_m - 2.0*vdotr*rhat3;
      } else {  // absorbing
        v1_new = 0.0;
        v2_new = 0.0;
        v3_new = 0.0;
      }
    }

    w0_(m,IDN,k,j,i) = dens_new;
    w0_(m,IVX,k,j,i) = v1_new;
    w0_(m,IVY,k,j,i) = v2_new;
    w0_(m,IVZ,k,j,i) = v3_new;

    // keep u0 consistent with the overwritten w0 for this masked cell, so it doesn't
    // go stale before the next stage's CopyCons
    HydCons1D ucons;
    if (eos.is_ideal) {
      w0_(m,IEN,k,j,i) = eint_new;
      HydPrim1D wprim;
      wprim.d = dens_new; wprim.vx = v1_new; wprim.vy = v2_new; wprim.vz = v3_new;
      wprim.e = eint_new;
      SingleP2C_IdealHyd(wprim, ucons);
    } else {
      ucons.d = dens_new;
      ucons.mx = dens_new*v1_new;
      ucons.my = dens_new*v2_new;
      ucons.mz = dens_new*v3_new;
    }
    u0_(m,IDN,k,j,i) = ucons.d;
    u0_(m,IM1,k,j,i) = ucons.mx;
    u0_(m,IM2,k,j,i) = ucons.my;
    u0_(m,IM3,k,j,i) = ucons.mz;
    if (eos.is_ideal) { u0_(m,IEN,k,j,i) = ucons.e; }
  });

  return TaskStatus::complete;
}

} // namespace hydro
