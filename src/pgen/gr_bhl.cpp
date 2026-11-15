//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_bhl.cpp
//! \brief GR hydro BHL accretion onto a Kerr black hole in Cartesian KS
//! coordinates (G = c = M = 1), with the wind entering at outer_x1.
//!
//! <problem> parameters rho_inf, v_inf, cs_inf (relativistic sound speed) set
//! the initial state and the outer_x1 ghost cells: rho = rho_inf and
//! uu^x = -v_inf/sqrt(1-v_inf^2), where uu^i = u^i + beta^i*u^t. Other faces
//! use primitive diode outflow. Passive scalars are zero.
//!
//! History (user_hist=true, 3D), one file per tag:
//!   flux0, flux1, ...: horizon, then flux_radii (default "12, 24").
//!     mdot = -int rho*u^r dA, edot = int T^r_t dA, ldot = int T^r_phi dA,
//!     Pdot_i = int T^r_i dA.
//!   volume0, ...: volume_rmin (default horizon) < r_KS < volume_radii[n]
//!     (default flux_radii, capped by volume_integral_rmax), active cells only.
//!     Gr_i = int (T^{mu nu}/2) d_i g_{mu nu} d^3x (GR force on gas),
//!     Nw_i = int rho*x_i/|x|^3 d^3x (Newtonian force on hole),
//!     T^t_i = int T^t_i d^3x.
//! Angular sampling: flux_nlev (default 5); interpolation: flux_ninterp
//! (default 2*nghost). sqrt(-g) = 1 in Cartesian KS.
//!
//! Symmetry: a reflecting face at z = 0 (any spin) or y = 0 (a = 0 only) is a
//! mirror plane. Spheres sample mirror images, volume sums are scaled, and
//! components odd under the reflection are zero, so all values are full-domain.
//! The headers record symmetry_factor (2 or 4).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

namespace Kokkos {
// Nine components keep the volume reducer's CUDA scratch allocation small.
template <> struct reduction_identity<array_sum::array_type<Real, 9>> {
  KOKKOS_FORCEINLINE_FUNCTION
  static array_sum::array_type<Real, 9> sum() {
    return array_sum::array_type<Real, 9>();
  }
};
} // namespace Kokkos

namespace {

struct BhlParameters {
  Real rho_inf, eint_inf, uu_inf;
  Real volume_rmin;
  std::vector<Real> volume_radii;
  // +1 (-1) if the mesh covers x_d >= 0 (<= 0) with a reflecting face at x_d = 0
  int fold[3];
  int symmetry_factor;
};
BhlParameters bhl;

void BhlBoundary(Mesh *pm);
void BhlHistory(HistoryData *pdata, Mesh *pm);

[[noreturn]] void Fatal(const std::string &message) {
  std::cerr << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

std::vector<Real> ParseRadii(const std::string &name, std::string list) {
  std::replace(list.begin(), list.end(), ',', ' ');
  std::istringstream stream(list);
  std::vector<Real> radii;
  while (true) {
    stream >> std::ws;
    if (stream.eof())
      break;
    Real r;
    if (!(stream >> r) || !std::isfinite(r) || r <= 0.0) {
      Fatal("<problem>/" + name + " must contain finite, positive radii.");
    }
    radii.push_back(r);
  }
  return radii;
}

// Recover u^mu and u_mu from AthenaK's projected spatial primitive uu^i.
KOKKOS_INLINE_FUNCTION
void FourVelocity(const Real glower[][4], const Real gupper[][4],
                  const Real uu[3], Real u[4], Real ucov[4]) {
  Real q = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j)
      q += glower[i + 1][j + 1] * uu[i] * uu[j];
  }
  const Real alpha = sqrt(-1.0 / gupper[0][0]);
  const Real lorentz = sqrt(1.0 + q);
  u[0] = lorentz / alpha;
  for (int i = 1; i < 4; ++i)
    u[i] = uu[i - 1] - alpha * lorentz * gupper[0][i];
  for (int i = 0; i < 4; ++i) {
    ucov[i] = 0.0;
    for (int j = 0; j < 4; ++j)
      ucov[i] += glower[i][j] * u[j];
  }
}
} // namespace

//----------------------------------------------------------------------------------------
// Set parameters and callbacks on both fresh starts and restarts.
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Validate the fluid, metric, and excision configuration.
  auto *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic || pmbp->phydro == nullptr ||
      pmbp->pmhd != nullptr || pmbp->prad != nullptr || pmbp->pz4c != nullptr) {
    Fatal("gr_bhl requires fixed-metric GR hydro, without MHD or radiation.");
  }
  const auto &coord = pmbp->pcoord->coord_data;
  const auto eos = pmbp->phydro->peos->eos_data;
  if (!eos.is_ideal || !std::isfinite(eos.gamma) ||
      eos.gamma <= 1.0 || eos.gamma > 2.0) {
    Fatal("gr_bhl requires an ideal-gas internal-energy EOS with 1 < gamma <= "
          "2.");
  }
  if (coord.is_minkowski || !coord.bh_excise || !std::isfinite(coord.bh_spin) ||
      std::abs(coord.bh_spin) > 1.0) {
    Fatal("gr_bhl requires a Kerr metric with |a| <= 1 and excise=true.");
  }
  const Real horizon = 1.0 + std::sqrt(1.0 - SQR(coord.bh_spin));
  if (!std::isfinite(coord.rexcise) || coord.rexcise <= 0.0 ||
      coord.rexcise > horizon || !std::isfinite(coord.dexcise) ||
      !std::isfinite(coord.pexcise) || coord.dexcise <= 0.0 ||
      coord.pexcise <= 0.0) {
    Fatal("Require 0 < r_excise <= r_horizon and positive excision "
          "density/pressure.");
  }

  // Derive the internal energy and projected velocity of the asymptotic wind.
  bhl.rho_inf = pin->GetReal("problem", "rho_inf");
  const Real v_inf = pin->GetReal("problem", "v_inf");
  const Real cs_inf = pin->GetReal("problem", "cs_inf");
  const Real gm1 = eos.gamma - 1.0;
  if (!std::isfinite(bhl.rho_inf) || bhl.rho_inf <= 0.0 ||
      !std::isfinite(v_inf) || v_inf < 0.0 || v_inf >= 1.0 ||
      !std::isfinite(cs_inf) || cs_inf <= 0.0 || SQR(cs_inf) >= gm1) {
    Fatal("Require rho_inf > 0, 0 <= v_inf < 1, and 0 < cs_inf^2 < gamma-1.");
  }
  bhl.eint_inf = bhl.rho_inf * SQR(cs_inf) / (eos.gamma * (gm1 - SQR(cs_inf)));
  bhl.uu_inf = -v_inf / std::sqrt(1.0 - SQR(v_inf));
  user_bcs_func = BhlBoundary;
  user_hist_func = BhlHistory;

  // Register one history tag per flux surface or integration volume.
  spherical_grids.clear();
  user_hist_tags.clear();
  bhl.volume_radii.clear();
  if (user_hist) {
    if (!pmy_mesh_->three_d)
      Fatal("Spherical BHL history diagnostics require 3D.");
    const int old_nlev = pin->GetOrAddInteger("problem", "sphere_nlev", 5);
    const int nlev = pin->GetOrAddInteger("problem", "flux_nlev", old_nlev);
    const int ninterp = pin->GetOrAddInteger("problem", "flux_ninterp", -1);
    if (nlev < 1 ||
        (ninterp != -1 &&
         (ninterp < 1 || ninterp > 2 * pmy_mesh_->mb_indcs.ng + 1))) {
      Fatal("Require flux_nlev >= 1 and flux_ninterp=-1 or 1..2*nghost+1.");
    }
    const std::string flux_list =
        pin->GetOrAddString("problem", "flux_radii", "12, 24");
    std::vector<Real> flux_radii{horizon};
    const auto extra = ParseRadii("flux_radii", flux_list);
    flux_radii.insert(flux_radii.end(), extra.begin(), extra.end());
    // Transverse mirror planes; the wind axis x is never one. Spheres may cross
    // them, since points beyond are sampled at their mirror images.
    const auto &size = pmy_mesh_->mesh_size;
    const Real lo[3] = {size.x1min, size.x2min, size.x3min};
    const Real hi[3] = {size.x1max, size.x2max, size.x3max};
    Real extent_xy = std::numeric_limits<Real>::max(), extent_z = extent_xy;
    bhl.symmetry_factor = 1;
    for (int d = 0; d < 3; ++d) {
      const bool fold_lo = d > 0 && lo[d] == 0.0 &&
                           pmy_mesh_->mesh_bcs[2 * d] == BoundaryFlag::reflect;
      const bool fold_hi = d > 0 && hi[d] == 0.0 &&
                           pmy_mesh_->mesh_bcs[2 * d + 1] == BoundaryFlag::reflect;
      bhl.fold[d] = fold_lo ? 1 : (fold_hi ? -1 : 0);
      bhl.symmetry_factor *= (bhl.fold[d] != 0) ? 2 : 1;
      Real &extent = (d < 2) ? extent_xy : extent_z;
      if (!fold_lo)
        extent = std::min(extent, -lo[d]);
      if (!fold_hi)
        extent = std::min(extent, hi[d]);
    }
    for (std::size_t n = 0; n < flux_radii.size(); ++n) {
      const Real radius = flux_radii[n];
      if (std::hypot(radius, coord.bh_spin) >= extent_xy ||
          radius >= extent_z || radius < horizon) {
        Fatal("Flux spheres must lie at or outside the horizon and inside "
              "the full mesh.");
      }
      if (std::find(flux_radii.begin(), flux_radii.begin() + n, radius) !=
          flux_radii.begin() + n) {
        Fatal("flux_radii must not duplicate a surface, including the horizon "
              "(flux0).");
      }
      spherical_grids.push_back(
          std::make_unique<SphericalGrid>(pmbp, nlev, radius, ninterp));
      if (bhl.symmetry_factor > 1)
        spherical_grids.back()->FoldInterpolationCoordinates(bhl.fold);
      user_hist_tags.push_back("flux" + std::to_string(n));
    }

    // Volume masks share an inner radius and may have a common outer cutoff.
    bhl.volume_rmin = pin->GetOrAddReal("problem", "volume_rmin", horizon);
    bhl.volume_radii =
        ParseRadii("volume_radii",
                   pin->GetOrAddString("problem", "volume_radii", flux_list));
    if (!std::isfinite(bhl.volume_rmin) || bhl.volume_rmin < horizon) {
      Fatal("volume_rmin must be finite and at or outside the horizon.");
    }
    const Real rmax = pin->GetOrAddReal(
        "problem", "volume_integral_rmax",
        bhl.volume_radii.empty() ? horizon
                                 : *std::max_element(bhl.volume_radii.begin(),
                                                     bhl.volume_radii.end()));
    if (!std::isfinite(rmax) || rmax <= bhl.volume_rmin) {
      Fatal(
          "volume_integral_rmax must be finite and greater than volume_rmin.");
    }
    for (std::size_t n = 0; n < bhl.volume_radii.size(); ++n) {
      bhl.volume_radii[n] = std::min(bhl.volume_radii[n], rmax);
      if (bhl.volume_radii[n] <= bhl.volume_rmin) {
        Fatal("Each volume radius must exceed volume_rmin after the outer "
              "cutoff.");
      }
      user_hist_tags.push_back("volume" + std::to_string(n));
    }
  }

  if (global_variable::my_rank == 0) {
    std::cout << "GR BHL wind: rho_inf=" << bhl.rho_inf << " v_inf=" << v_inf
              << " cs_inf=" << cs_inf << " eint_inf=" << bhl.eint_inf
              << std::endl;
  }
  if (restart)
    return;

  // Initialize active cells; excised cells use the prescribed floor state.
  const auto indcs = pmy_mesh_->mb_indcs;
  auto w0 = pmbp->phydro->w0;
  const int nhydro = pmbp->phydro->nhydro;
  const int nvars = nhydro + pmbp->phydro->nscalars;
  const Real rho_inf = bhl.rho_inf, eint_inf = bhl.eint_inf,
             uu_inf = bhl.uu_inf;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const auto excision = pmbp->pcoord->excision_floor;
  par_for(
      "bhl_initial_state", DevExeSpace(), 0, pmbp->nmb_thispack - 1, ks,
      indcs.ke, js, indcs.je, is, indcs.ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const bool excised = excision(m, k, j, i);
        w0(m, IDN, k, j, i) = excised ? coord.dexcise : rho_inf;
        w0(m, IEN, k, j, i) = excised ? coord.pexcise / gm1 : eint_inf;
        w0(m, IVX, k, j, i) = excised ? 0.0 : uu_inf;
        w0(m, IVY, k, j, i) = 0.0;
        w0(m, IVZ, k, j, i) = 0.0;
        for (int n = nhydro; n < nvars; ++n)
          w0(m, n, k, j, i) = 0.0;
      });
  pmbp->phydro->peos->PrimToCons(w0, pmbp->phydro->u0, is, indcs.ie, js,
                                 indcs.je, ks, indcs.ke);
}

namespace {

//----------------------------------------------------------------------------------------
// Sweep each direction, including transverse ghosts, to fill faces, edges, and
// corners. Recover primitives on source planes, then convert user ghosts at
// their own locations.
void BhlBoundary(Mesh *pm) {
  auto *phydro = pm->pmb_pack->phydro;
  const auto indcs = pm->mb_indcs;
  const auto size = pm->pmb_pack->pmb->mb_size;
  const auto coord = pm->pmb_pack->pcoord->coord_data;
  const auto mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  auto w0 = phydro->w0;
  auto u0 = phydro->u0;
  const int nhydro = phydro->nhydro;
  const int nvars = nhydro + phydro->nscalars;
  const Real gamma = phydro->peos->eos_data.gamma;
  const Real rho_inf = bhl.rho_inf, eint_inf = bhl.eint_inf,
             uu_inf = bhl.uu_inf;
  const int starts[3] = {indcs.is, indcs.js, indcs.ks};
  const int ends[3] = {indcs.ie, indcs.je, indcs.ke};
  const int ncells[3] = {indcs.nx1 + 2 * indcs.ng,
                         indcs.nx2 > 1 ? indcs.nx2 + 2 * indcs.ng : 1,
                         indcs.nx3 > 1 ? indcs.nx3 + 2 * indcs.ng : 1};
  const int ndim = pm->three_d ? 3 : (pm->two_d ? 2 : 1);
  for (int dir = 0; dir < ndim; ++dir) {
    const int inner = 2 * dir, outer = inner + 1;
    if (pm->mesh_bcs[inner] != BoundaryFlag::user &&
        pm->mesh_bcs[outer] != BoundaryFlag::user)
      continue;
    // Transverse ghosts can adjoin another MeshBlock while still lying on this
    // physical boundary, so every sweep must include them.
    int lower[3] = {0, 0, 0};
    int upper[3] = {ncells[0] - 1, ncells[1] - 1, ncells[2] - 1};

    // Refresh both source planes before constructing the ghost states.
    for (int side = 0; side < 2; ++side) {
      if (pm->mesh_bcs[inner + side] != BoundaryFlag::user)
        continue;
      const int plane = side == 0 ? starts[dir] : ends[dir];
      lower[dir] = upper[dir] = plane;
      phydro->peos->ConsToPrim(u0, w0, false, lower[0], upper[0], lower[1],
                               upper[1], lower[2], upper[2]);
    }

    lower[dir] = 0;
    upper[dir] = ncells[dir] - 1;
    const int start = starts[dir], end = ends[dir];
    par_for(
        "bhl_boundary", DevExeSpace(), 0, pm->pmb_pack->nmb_thispack - 1,
        lower[2], upper[2], lower[1], upper[1], lower[0], upper[0],
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          const int index = dir == 0 ? i : (dir == 1 ? j : k);
          if (index >= start && index <= end)
            return;
          const bool low = index < start;
          if (mb_bcs.d_view(m, low ? inner : outer) != BoundaryFlag::user)
            return;
          int si = i, sj = j, sk = k;
          if (dir == 0)
            si = low ? start : end;
          if (dir == 1)
            sj = low ? start : end;
          if (dir == 2)
            sk = low ? start : end;

          // Prescribe the upstream wind; copy and clip the other boundary
          // states.
          const bool inject = dir == 0 && !low;
          for (int n = 0; n < nvars; ++n) {
            Real value = inject ? 0.0 : w0(m, n, sk, sj, si);
            if (inject) {
              value = 0.0;
              if (n == IDN)
                value = rho_inf;
              if (n == IEN)
                value = eint_inf;
              if (n == IVX)
                value = uu_inf;
            } else if (n == IVX + dir) {
              value = low ? fmin(0.0, value) : fmax(0.0, value);
            }
            w0(m, n, k, j, i) = value;
          }

          // Conserved variables must use the metric at the destination ghost
          // cell.
          const Real x =
              CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                          size.d_view(m).x1max);
          const Real y =
              CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                          size.d_view(m).x2max);
          const Real z =
              CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                          size.d_view(m).x3max);
          Real glower[4][4], gupper[4][4];
          ComputeMetricAndInverse(x, y, z, false, coord.bh_spin, glower,
                                  gupper);
          HydPrim1D w;
          w.d = w0(m, IDN, k, j, i);
          w.e = w0(m, IEN, k, j, i);
          w.vx = w0(m, IVX, k, j, i);
          w.vy = w0(m, IVY, k, j, i);
          w.vz = w0(m, IVZ, k, j, i);
          HydCons1D u;
          SingleP2C_IdealGRHyd(glower, gupper, w, gamma, u);
          u0(m, IDN, k, j, i) = u.d;
          u0(m, IEN, k, j, i) = u.e;
          u0(m, IM1, k, j, i) = u.mx;
          u0(m, IM2, k, j, i) = u.my;
          u0(m, IM3, k, j, i) = u.mz;
          for (int n = nhydro; n < nvars; ++n)
            u0(m, n, k, j, i) = u.d * w0(m, n, k, j, i);
        });
  }
}

//----------------------------------------------------------------------------------------
// Integrate mass, energy, angular momentum, and Cartesian momentum fluxes on
// one sphere.
void SurfaceHistory(HistoryData *pdata, Mesh *pm) {
  auto *phydro = pm->pmb_pack->phydro;
  auto &grid = *pm->pgen->spherical_grids[pdata->user_index];
  const Real spin = pm->pmb_pack->pcoord->coord_data.bh_spin;
  const Real gamma = phydro->peos->eos_data.gamma;
  pdata->nhist = 6;
  const char *labels[] = {"mdot", "edot", "ldot", "Pdot_x", "Pdot_y", "Pdot_z"};
  for (int n = 0; n < pdata->nhist; ++n)
    pdata->label[n] = labels[n];
  pdata->symmetry_factor = bhl.symmetry_factor;
  pdata->header_params = {{"radius", grid.radius},
                          {"nangles", static_cast<Real>(grid.nangles)},
                          {"flux_ninterp", static_cast<Real>(grid.ninterp)}};

  grid.InterpolateToSphere(phydro->nhydro, phydro->w0);
  const Real r = grid.radius;
  for (int n = 0; n < grid.nangles; ++n) {
    const Real rho = grid.interp_vals.h_view(n, IDN);
    const Real eint = grid.interp_vals.h_view(n, IEN);
    // A folded point was sampled at its mirror image: use the original
    // position and flip the velocity normal to each mirror plane crossed.
    const Real theta = grid.polar_pos.h_view(n, 0);
    const Real phi = grid.polar_pos.h_view(n, 1);
    const Real pos[3] = {(r * cos(phi) - spin * sin(phi)) * sin(theta),
                         (r * sin(phi) + spin * cos(phi)) * sin(theta),
                         r * cos(theta)};
    const Real x = pos[0], y = pos[1], z = pos[2];
    Real uu[3] = {grid.interp_vals.h_view(n, IVX),
                  grid.interp_vals.h_view(n, IVY),
                  grid.interp_vals.h_view(n, IVZ)};
    for (int d = 1; d < 3; ++d) {
      if (bhl.fold[d] * pos[d] < 0.0)
        uu[d] = -uu[d];
    }
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x, y, z, false, spin, glower, gupper);
    Real u[4], ucov[4];
    FourVelocity(glower, gupper, uu, u, ucov);

    // Project onto the spherical KS radial direction and apply the area
    // measure.
    const Real denom = 2.0 * SQR(r) - (SQR(x) + SQR(y) + SQR(z)) + SQR(spin);
    const Real dr[3] = {r * x / denom, r * y / denom,
                        (r + SQR(spin) / r) * z / denom};
    const Real ur = dr[0] * u[1] + dr[1] * u[2] + dr[2] * u[3];
    const Real area = (SQR(r) + SQR(spin * cos(grid.polar_pos.h_view(n, 0)))) *
                      grid.solid_angles.h_view(n);
    const Real enthalpy = rho + gamma * eint;

    pdata->hdata[0] -= rho * ur * area;
    pdata->hdata[1] += enthalpy * ur * ucov[0] * area;
    pdata->hdata[2] += enthalpy * ur * (-y * ucov[1] + x * ucov[2]) * area;
    for (int d = 0; d < 3; ++d) {
      pdata->hdata[3 + d] +=
          (enthalpy * ur * ucov[d + 1] + (gamma - 1.0) * eint * dr[d]) * area;
    }
  }
}

//----------------------------------------------------------------------------------------
// Integrate GR forces, Newtonian force estimates, and momentum over one radial
// mask.
void VolumeHistory(HistoryData *pdata, Mesh *pm, const int index) {
  const Real rmin = bhl.volume_rmin, rmax = bhl.volume_radii[index];
  const auto indcs = pm->mb_indcs;
  const auto size = pm->pmb_pack->pmb->mb_size;
  const Real spin = pm->pmb_pack->pcoord->coord_data.bh_spin;
  const Real gamma = pm->pmb_pack->phydro->peos->eos_data.gamma;
  const auto w0 = pm->pmb_pack->phydro->w0;
  pdata->nhist = 9;
  const char *labels[] = {"Gr_x", "Gr_y",  "Gr_z",  "Nw_x", "Nw_y",
                          "Nw_z", "T^t_x", "T^t_y", "T^t_z"};
  for (int n = 0; n < pdata->nhist; ++n)
    pdata->label[n] = labels[n];
  pdata->symmetry_factor = bhl.symmetry_factor;
  pdata->header_params = {{"rin", rmin}, {"rout", rmax}};

  using VolumeSum = array_sum::array_type<Real, 9>;
  VolumeSum sum;
  const int nji = indcs.nx2 * indcs.nx1;
  const int nkji = indcs.nx3 * nji;
  Kokkos::parallel_reduce(
      "bhl_volume_history",
      Kokkos::RangePolicy<>(DevExeSpace(), 0,
                            pm->pmb_pack->nmb_thispack * nkji),
      KOKKOS_LAMBDA(const int idx, VolumeSum &accum) {
        const int m = idx / nkji;
        const int k = (idx - m * nkji) / nji + indcs.ks;
        const int j = ((idx - m * nkji) % nji) / indcs.nx1 + indcs.js;
        const int i = idx % indcs.nx1 + indcs.is;
        const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                   size.d_view(m).x1min, size.d_view(m).x1max);
        const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                   size.d_view(m).x2min, size.d_view(m).x2max);
        const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                   size.d_view(m).x3min, size.d_view(m).x3max);
        const Real rad2 = SQR(x) + SQR(y) + SQR(z);
        const Real rks2 =
            0.5 * (rad2 - SQR(spin) +
                   sqrt(SQR(rad2 - SQR(spin)) + 4.0 * SQR(spin * z)));
        if (rks2 <= SQR(rmin) || rks2 >= SQR(rmax))
          return;

        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x, y, z, false, spin, glower, gupper);
        const Real uu[3] = {w0(m, IVX, k, j, i), w0(m, IVY, k, j, i),
                            w0(m, IVZ, k, j, i)};
        Real u[4], ucov[4];
        FourVelocity(glower, gupper, uu, u, ucov);
        const Real rho = w0(m, IDN, k, j, i), eint = w0(m, IEN, k, j, i);
        const Real enthalpy = rho + gamma * eint,
                   pressure = (gamma - 1.0) * eint;
        Real dg[3][4][4];
        ComputeMetricDerivatives(x, y, z, false, spin, dg[0], dg[1], dg[2]);
        const Real vol =
            size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;
        const Real pos[3] = {x, y, z};
        for (int d = 0; d < 3; ++d) {
          Real source = 0.0;
          // Contract the symmetric stress tensor with the metric derivative.
          // Off-diagonal terms occur twice, canceling the overall factor of
          // 1/2.
          for (int mu = 0; mu < 4; ++mu) {
            for (int nu = mu; nu < 4; ++nu) {
              const Real stress =
                  enthalpy * u[mu] * u[nu] + pressure * gupper[mu][nu];
              source += (mu == nu ? 0.5 : 1.0) * dg[d][mu][nu] * stress;
            }
          }
          accum.the_array[d] += vol * source;
          accum.the_array[3 + d] += vol * rho * pos[d] / (rad2 * sqrt(rad2));
          accum.the_array[6 + d] += vol * enthalpy * u[0] * ucov[d + 1];
        }
      },
      sum);
  // Column n is the (n % 3) component: zero if odd under a mirror plane, else
  // scaled to the full domain.
  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = (bhl.fold[n % 3] != 0)
                          ? 0.0
                          : bhl.symmetry_factor * sum.the_array[n];
  }
}

//----------------------------------------------------------------------------------------
// The tag order is all flux surfaces followed by all integration volumes.
void BhlHistory(HistoryData *pdata, Mesh *pm) {
  static_assert(NHISTORY_VARIABLES >= 9,
                "BHL history needs at least nine columns");
  for (int n = 0; n < NHISTORY_VARIABLES; ++n)
    pdata->hdata[n] = 0.0;
  const int nflux = static_cast<int>(pm->pgen->spherical_grids.size());
  if (pdata->user_index < nflux) {
    SurfaceHistory(pdata, pm);
  } else {
    VolumeHistory(pdata, pm, pdata->user_index - nflux);
  }
}
} // namespace
