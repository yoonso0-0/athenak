//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bhl_wind_tunnel.cpp
//! \brief Uniform wind tunnel for Bondi-Hoyle-Lyttleton (BHL) accretion,
//! Newtonian hydro.
//!
//! A uniform flow of density dens_inf and speed vel_inf is directed along +x1
//! past the GM=1 point mass from
//! <hydro_srcterms>/point_particle_gravity_at_center. The singular center is
//! replaced by the spherical inner mask (<sphere_mask>); all bc options except
//! dirichlet are allowed. The mask is a boundary condition, not a sink, so the
//! net inward mass flow is measured as minus the outward mass flux through a
//! sphere outside the mask radius. With an injected wind, this is the net flow,
//! not the gross inflow rate.
//!
//! The initial state is the uniform wind, evolved until the wake settles. The
//! length scale is the accretion radius R_B = 2GM/(vel_inf^2 + cs_inf^2).
//! Startup prints the upstream and mask-wind states and the Bondi-Hoyle rate
//! estimate (and the Bondi rate for gamma < 5/3). These are
//! reference values for accretion without an injected mask wind.
//!
//! Ideal and isothermal EOS are supported (isothermal takes cs from
//! <hydro>/iso_sound_speed). The mesh may be 2D (a cheap qualitative pilot,
//! still with the 3D 1/r potential) or 3D. Faces flagged 'user' are held at the
//! upstream state, e.g. ix1_bc=user with ox1_bc=outflow.
//!
//! An optional passive scalar (<hydro>/nscalars=1) tags the gas injected by the
//! mask: s = 0 upstream and in the initial state, while the mask holds s =
//! <sphere_mask>/scalar for bc=spherical_wind (set it to 1, so s is the mass
//! fraction of wind material) and s = 0 for reflecting/absorbing.
//!
//! References: Hoyle & Lyttleton 1939, PCPS, 35, 405; Bondi & Hoyle 1944,
//! MNRAS, 104, 273; Edgar 2004, NewAR, 48, 843.

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
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "srcterms/srcterms.hpp"

namespace {

struct WindTunnelData {
  Real dens_inf;  // upstream density
  Real cs_inf;    // upstream sound speed
  Real eint_inf;  // upstream internal energy density (ideal EOS only)
  Real vel_inf;   // upstream speed, directed along +x1
  Real R_B;       // accretion radius 2GM/(vel_inf^2+cs_inf^2), with GM=1
  Real cons_mom1; // upstream x1 momentum density
  Real cons_etot; // upstream total energy density (ideal EOS only)
  bool is_ideal;
};

WindTunnelData wind_tunnel;

// Coordinate planes through the point mass that are reflecting mesh faces, for
// history output on a mesh covering a symmetric part of the domain: +1 (-1) if
// the mesh covers x_d >= 0 (x_d <= 0) with a reflecting face at x_d = 0, else
// 0.
int symmetry_fold[3] = {0, 0, 0};

// Number of reflected copies represented by the full-domain user histories.
int HistorySymmetryFactor() {
  int nimages = 1;
  for (int d = 0; d < 3; ++d)
    nimages *= (symmetry_fold[d] != 0) ? 2 : 1;
  return nimages;
}

// User history files 0..nflux_files-1 are the spherical fluxes (one per
// SphericalGrid); the next volume_radii.size() contain the same integrals as
// the domain history, restricted to sphere_mask_radius <= r < volume_radii[n].
// An optional last file holds totals outside the mask (<problem>/domain_hist).
int nflux_files = 0;
std::vector<Real> volume_radii;

// Entries shared by the volume and domain histories, in domain column order.
enum CellIntegral {
  kMass,
  kScalarMass,
  kMomentumExcessX1,
  kScalarMomentumExcessX1,
  kGravityForceX1,
  kScalarGravityForceX1,
  kWindDragX1,
  kOutflowDragX1,
  kNumCellIntegrals
};

[[noreturn]] void Fatal(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

// Parse <problem>/name, a comma-separated list of numbers, e.g. 0.3, 0.5, 1.0
std::vector<Real> ParseRadii(ParameterInput *pin, const std::string &name) {
  std::string list = pin->GetString("problem", name);
  std::replace(list.begin(), list.end(), ',', ' ');
  std::stringstream stream(list);
  std::vector<Real> radii;
  Real value;
  while (stream >> value)
    radii.push_back(value);
  if (!stream.eof() || radii.empty()) {
    Fatal("<problem>/" + name + " must be a comma-separated list of numbers");
  }
  return radii;
}

// Overwrite one boundary cell with the uniform upstream state.
KOKKOS_INLINE_FUNCTION
void SetUpstream(const WindTunnelData p, const DvceArray5D<Real> &u0,
                 const int m, const int k, const int j, const int i,
                 const int nhydro, const int nscalars) {
  u0(m, IDN, k, j, i) = p.dens_inf;
  u0(m, IM1, k, j, i) = p.cons_mom1;
  u0(m, IM2, k, j, i) = 0.0;
  u0(m, IM3, k, j, i) = 0.0;
  if (p.is_ideal)
    u0(m, IEN, k, j, i) = p.cons_etot;
  for (int n = nhydro; n < nhydro + nscalars; ++n)
    u0(m, n, k, j, i) = 0.0;
}

// Hold every face flagged 'user' at the uniform upstream state. Conserved
// variables are written directly, since converting a whole array from
// primitives at this point would overwrite active cells with stale stage data.
void FixedWindBoundary(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *phydro = pmbp->phydro;
  if (phydro == nullptr)
    return;

  auto &indcs = pm->mb_indcs;
  const int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2 * ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  const int nmb = pmbp->nmb_thispack;
  const int nhydro = phydro->nhydro;
  const int nscalars = phydro->nscalars;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &u0 = phydro->u0;
  const WindTunnelData p = wind_tunnel;

  par_for(
      "bhl_wind_bc_x1", DevExeSpace(), 0, nmb - 1, 0, n3 - 1, 0, n2 - 1, 0,
      ng - 1, KOKKOS_LAMBDA(int m, int k, int j, int g) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
          SetUpstream(p, u0, m, k, j, g, nhydro, nscalars);
        }
        if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
          SetUpstream(p, u0, m, k, j, ie + 1 + g, nhydro, nscalars);
        }
      });

  par_for(
      "bhl_wind_bc_x2", DevExeSpace(), 0, nmb - 1, 0, n3 - 1, 0, ng - 1, 0,
      n1 - 1, KOKKOS_LAMBDA(int m, int k, int g, int i) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) {
          SetUpstream(p, u0, m, k, g, i, nhydro, nscalars);
        }
        if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
          SetUpstream(p, u0, m, k, je + 1 + g, i, nhydro, nscalars);
        }
      });

  if (pm->three_d) {
    par_for(
        "bhl_wind_bc_x3", DevExeSpace(), 0, nmb - 1, 0, ng - 1, 0, n2 - 1, 0,
        n1 - 1, KOKKOS_LAMBDA(int m, int g, int j, int i) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user) {
            SetUpstream(p, u0, m, g, j, i, nhydro, nscalars);
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user) {
            SetUpstream(p, u0, m, ke + 1 + g, j, i, nhydro, nscalars);
          }
        });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SphericalFluxHistory()
//! \brief User history: net outward fluxes through the sphere of radius
//! <problem>/flux_radii[pdata->user_index] centered on the point mass (so
//! accretion appears as a negative mass flux). Columns: mass flux \oint dens
//! v.rhat dA and the three components of the momentum flux \oint (dens v_i
//! v.rhat + pres rhat_i) dA, pressure included; with a passive scalar, also the
//! scalar mass flux \oint dens s v.rhat dA. Called once per radius; each
//! gets its own history file, <basename>.user.flux<n>.hst, n = 0, 1, ... in
//! flux_radii order. If the mesh covers only half (or a quarter, ...) of the
//! domain behind reflecting faces at x_d = 0, the part of the sphere outside
//! the mesh is sampled at the mirror image points and the fluxes are those of
//! the full sphere.

void SphericalFluxHistory(HistoryData *pdata, Mesh *pm) {
  auto &grid = pm->pgen->spherical_grids[pdata->user_index];
  auto *phydro = pm->pmb_pack->phydro;
  auto &eos = phydro->peos->eos_data;
  const int nhydro = phydro->nhydro;
  const int nscalars = phydro->nscalars;

  pdata->nhist = 4 + nscalars;
  pdata->label[0] = "mflux";
  pdata->label[1] = "pflux1";
  pdata->label[2] = "pflux2";
  pdata->label[3] = "pflux3";
  if (nscalars > 0)
    pdata->label[4] = "sflux";

  // Each rank interpolates onto the angles it owns and zeros the rest; the
  // history output then sums over ranks.
  grid->InterpolateToSphere(nhydro + nscalars, phydro->w0);
  const Real r = grid->radius;
  Real flux[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (int n = 0; n < grid->nangles; ++n) {
    const Real theta = grid->polar_pos.h_view(n, 0);
    const Real phi = grid->polar_pos.h_view(n, 1);
    const Real rhat[3] = {sin(theta) * cos(phi), sin(theta) * sin(phi),
                          cos(theta)};
    const Real dens = grid->interp_vals.h_view(n, IDN);
    Real vel[3] = {grid->interp_vals.h_view(n, IVX),
                   grid->interp_vals.h_view(n, IVY),
                   grid->interp_vals.h_view(n, IVZ)};
    // For a point outside the simulated domain, interpolation used its mirror
    // image. Restore the velocity component normal to each reflection plane.
    for (int d = 0; d < 3; ++d) {
      if (symmetry_fold[d] * rhat[d] < 0.0)
        vel[d] = -vel[d];
    }
    const Real pres = eos.is_ideal
                          ? (eos.gamma - 1.0) * grid->interp_vals.h_view(n, IEN)
                          : dens * SQR(eos.iso_cs);
    const Real vr = vel[0] * rhat[0] + vel[1] * rhat[1] + vel[2] * rhat[2];
    const Real dA = SQR(r) * grid->solid_angles.h_view(n);
    flux[0] += dens * vr * dA;
    for (int d = 0; d < 3; ++d) {
      flux[1 + d] += (dens * vel[d] * vr + pres * rhat[d]) * dA;
    }
    if (nscalars > 0)
      flux[4] += dens * grid->interp_vals.h_view(n, nhydro) * vr * dA;
  }
  for (int n = 0; n < pdata->nhist; ++n)
    pdata->hdata[n] = flux[n];

  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n)
    pdata->hdata[n] = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void CellSums()
//! \brief Volume integrals over the cells whose centers lie in rin <= r < rout,
//! using conserved variables. The CellIntegral entries have the integrands
//! documented in VolumeHistory; scalar-dependent entries are zero without a
//! passive scalar. s is read as the conserved scalar density divided by dens,
//! with no clipping or selection by tracer value.
//!
//! The force is that of the gas on the GM=1 point mass. The histories use
//! rin = sphere_mask_radius >= gravity_softening_length, where the kernel is
//! Newtonian.
//! Only active cells inside the mesh are integrated. Cell centers determine
//! shell membership, so spherical boundaries are resolved to the cell scale.
//! Both volume and domain histories exclude mask cells.
//!
//! These are sums on this MPI rank; HistoryOutput sums them across ranks.
//! Mirror symmetry scales each integral by HistorySymmetryFactor(). The x1
//! force and momentum entries cancel if x1 = 0 is a reflection plane.

void CellSums(Mesh *pm, const Real rin, const Real rout,
              Real sums[kNumCellIntegrals]) {
  auto *phydro = pm->pmb_pack->phydro;
  auto &u0 = phydro->u0;
  const int nhydro = phydro->nhydro;
  const bool has_scalar = phydro->nscalars > 0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  const int nmkji = pm->pmb_pack->nmb_thispack * nkji;
  const Real mom1_inf = wind_tunnel.cons_mom1;
  const Real dens_inf = wind_tunnel.dens_inf;

  array_sum::GlobalSum sum_this_rank;
  Kokkos::parallel_reduce(
      "bhl_cell_sums", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &sum) {
        int m = idx / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        int i = idx - m * nkji - k * nji - j * nx1;
        const Real x1 =
            CellCenterX(i, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
        const Real x2 =
            CellCenterX(j, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        const Real x3 =
            CellCenterX(k, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
        const Real r2 = SQR(x1) + SQR(x2) + SQR(x3);
        const Real r = sqrt(r2);
        if (r >= rin && r < rout) {
          const Real vol =
              size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;
          const Real dens = u0(m, IDN, k + ks, j + js, i + is);
          const Real dm = vol * dens;
          sum.the_array[kMass] += dm;
          sum.the_array[kGravityForceX1] += (dm / (r2 * r)) * x1;
          const Real dpx =
              vol * (u0(m, IM1, k + ks, j + js, i + is) - mom1_inf);
          sum.the_array[kMomentumExcessX1] += dpx;
          if (has_scalar) {
            const Real scalar_dens = u0(m, nhydro, k + ks, j + js, i + is);
            const Real s = scalar_dens / dens;
            const Real scalar_dm = vol * scalar_dens;
            sum.the_array[kScalarMass] += scalar_dm;
            sum.the_array[kScalarGravityForceX1] += scalar_dm * x1 / (r2 * r);
            sum.the_array[kScalarMomentumExcessX1] += s * dpx;
            sum.the_array[kOutflowDragX1] +=
                vol * s * (dens - dens_inf) * x1 / (r2 * r);
            sum.the_array[kWindDragX1] +=
                vol * (1.0 - s) * (dens - dens_inf) * x1 / (r2 * r);
          }
        }
      },
      Kokkos::Sum<array_sum::GlobalSum>(sum_this_rank));

  const Real nimages = HistorySymmetryFactor();
  for (int n = 0; n < kNumCellIntegrals; ++n) {
    const bool cancels = symmetry_fold[0] != 0 && n != kMass && n != kScalarMass;
    sums[n] = cancels ? 0.0 : nimages * sum_this_rank.the_array[n];
  }
}

//----------------------------------------------------------------------------------------
//! \fn void VolumeHistory()
//! \brief Shared volume and domain history integrals. Each
//! <basename>.user.vol<n>.hst uses rout = <problem>/volume_radii[n];
//! <basename>.user.domain.hst uses the whole mesh outside the mask.
//! All integrals use active cells whose centers satisfy
//! sphere_mask_radius <= r < rout and include mirror symmetry scaling.
//!
//! Notation: dens = rho; s = outflow tracer mass fraction; mom1 = rho v1;
//! mom1_inf = dens_inf vel_inf. The force kernel is x1/r^3 with GM = 1;
//! positive force acts on the central mass toward +x1.
//!
//! Columns in output order:
//!
//!   mass       = \int dens dV
//!                Integrand: total gas mass density.
//!
//!   scal-0     = \int dens s dV
//!                Integrand: mass density of tracer-tagged outflow material.
//!
//!   dPx        = \int (mom1 - mom1_inf) dV
//!                Integrand: x1 momentum density minus the upstream value.
//!
//!   dPx_s      = \int s (mom1 - mom1_inf) dV
//!                Integrand: the same momentum excess weighted by s.
//!
//!   fgrav1     = \int dens x1/r^3 dV
//!                Integrand: total gas density times the x1 force kernel.
//!
//!   fgrav1_s   = \int dens s x1/r^3 dV
//!                Integrand: tracer-tagged density times the x1 force kernel.
//!
//!   DF_wind    = \int (1-s) (dens - dens_inf) x1/r^3 dV
//!                Integrand: density excess over upstream, weighted by the
//!                ambient fraction (1-s), times the x1 force kernel.
//!
//!   DF_outflow = \int s (dens - dens_inf) x1/r^3 dV
//!                Integrand: density excess over upstream, weighted by the
//!                outflow fraction s, times the x1 force kernel.
//!
//! DF_wind + DF_outflow = \int (dens - dens_inf) x1/r^3 dV.
//! Scalar-dependent columns are omitted without a passive scalar.
//! Mass and momentum can change through fluxes across the domain and mask
//! boundaries. dPx and dPx_s are momentum differences, not time derivatives.

void VolumeHistory(HistoryData *pdata, Mesh *pm, const Real rout) {
  const bool has_scalar = pm->pmb_pack->phydro->nscalars > 0;
  Real sums[kNumCellIntegrals];
  CellSums(pm, pm->pmb_pack->phydro->sphere_mask_radius, rout, sums);

  pdata->nhist = 0;
  auto add_column = [pdata](const char *label, const Real value) {
    const int n = pdata->nhist++;
    pdata->label[n] = label;
    pdata->hdata[n] = value;
  };
  add_column("mass", sums[kMass]);
  if (has_scalar)
    add_column("scal-0", sums[kScalarMass]);
  add_column("dPx", sums[kMomentumExcessX1]);
  if (has_scalar)
    add_column("dPx_s", sums[kScalarMomentumExcessX1]);
  add_column("fgrav1", sums[kGravityForceX1]);
  if (has_scalar) {
    add_column("fgrav1_s", sums[kScalarGravityForceX1]);
    add_column("DF_wind", sums[kWindDragX1]);
    add_column("DF_outflow", sums[kOutflowDragX1]);
  }

  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n)
    pdata->hdata[n] = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void DomainHistory()
//! \brief The same columns and integrands as VolumeHistory, over the entire
//! mesh outside the mask, in <basename>.user.domain.hst.

void DomainHistory(HistoryData *pdata, Mesh *pm) {
  VolumeHistory(pdata, pm, std::numeric_limits<Real>::max());
}

//----------------------------------------------------------------------------------------
//! \fn void BHLUserHistory()
//! \brief Dispatches the user history files, see nflux_files.

void BHLUserHistory(HistoryData *pdata, Mesh *pm) {
  // Volume sums are multiplied by this factor. Fluxes instead reconstruct the
  // full sphere by sampling mirror images; they need no further multiplier.
  pdata->symmetry_factor = HistorySymmetryFactor();
  const int nvol_files = volume_radii.size();
  if (pdata->user_index < nflux_files) {
    SphericalFluxHistory(pdata, pm);
  } else if (pdata->user_index < nflux_files + nvol_files) {
    VolumeHistory(pdata, pm, volume_radii[pdata->user_index - nflux_files]);
  } else {
    DomainHistory(pdata, pm);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PrintSetup()
//! \brief Startup summary on rank 0: the upstream flow, the mask wind, the
//! passive scalar, the accretion rate estimates, and the user history files.

void PrintSetup(hydro::Hydro *phydro, const std::vector<std::string> &tags,
                const std::vector<std::unique_ptr<SphericalGrid>> &grids) {
  auto &eos = phydro->peos->eos_data;
  const WindTunnelData &p = wind_tunnel;
  std::cout << std::endl
            << "BHL wind tunnel (GM = 1)" << std::endl
            << "- R_B = " << p.R_B
            << ", mach_inf = " << p.vel_inf / p.cs_inf << std::endl
            << "- dens_inf = " << p.dens_inf << std::endl
            << "- vel_inf = " << p.vel_inf << std::endl
            << "- cs_inf = " << p.cs_inf << std::endl;

  if (phydro->sphere_mask_bc == hydro::Hydro::SphereMaskBC::spherical_wind) {
    const Real cs_mask = p.is_ideal ? sqrt(eos.gamma * (eos.gamma - 1.0) *
                                           phydro->sm_eint / phydro->sm_dens)
                                    : eos.iso_cs;
    const Real R_0 = (phydro->sm_velr / p.vel_inf) *
                     sqrt(phydro->sm_dens / p.dens_inf);
    std::cout << std::endl
              << "Spherical wind from the mask" << std::endl
              << "- R_0 = " << R_0 << ", mach_mask = "
              << phydro->sm_velr / cs_mask << std::endl
              << "- dens = " << phydro->sm_dens << std::endl
              << "- vel_r = " << phydro->sm_velr << std::endl
              << "- cs = " << cs_mask << std::endl;
  }

  if (phydro->nscalars > 0) {
    const bool wind =
        phydro->sphere_mask_bc == hydro::Hydro::SphereMaskBC::spherical_wind;
    std::cout << std::endl
              << "Passive scalar: s = 0 upstream, s = "
              << (wind ? phydro->sm_scalar : 0.0) << " in the mask"
              << std::endl;
  }

  const Real gm2rho = p.dens_inf; // (GM)^2 dens_inf, with GM=1
  std::cout << std::endl
            << "Accretion rate estimates:" << std::endl
            << "- mdot_bhl = "
            << 4.0 * M_PI * gm2rho / pow(SQR(p.vel_inf) + SQR(p.cs_inf), 1.5)
            << std::endl;
  // Print the Bondi expression for gamma < 5/3 (gamma = 1 for isothermal gas).
  const Real gamma = p.is_ideal ? eos.gamma : 1.0;
  const Real q = 5.0 - 3.0 * gamma;
  if (q > 0.0) {
    const Real lambda = p.is_ideal
                            ? 0.25 * pow(2.0 / q, 0.5 * q / (gamma - 1.0))
                            : 0.25 * exp(1.5);
    std::cout << "- mdot_bondi = "
              << 4.0 * M_PI * lambda * gm2rho / (SQR(p.cs_inf) * p.cs_inf)
              << std::endl;
  }

  if (!tags.empty()) {
    std::cout << std::endl
              << "User history files <basename>.user.<tag>.hst:" << std::endl;
    for (int n = 0; n < static_cast<int>(tags.size()); ++n) {
      std::cout << "- " << tags[n] << ": ";
      if (n < nflux_files) {
        std::cout << "r = " << grids[n]->radius;
      } else if (n < nflux_files + static_cast<int>(volume_radii.size())) {
        std::cout << phydro->sphere_mask_radius
                  << " <= r < " << volume_radii[n - nflux_files];
      } else {
        std::cout << "outside the mask";
      }
      std::cout << std::endl;
    }
    std::cout << "- symmetry_factor = " << HistorySymmetryFactor()
              << " (recorded on the second header line)" << std::endl;
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::BHLWindTunnel()
//! \brief Initialize a uniform wind flowing past a masked, gravitating GM=1
//! point mass.

void ProblemGenerator::BHLWindTunnel(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *phydro = pmbp->phydro;

  if (phydro == nullptr || pmbp->pmhd != nullptr) {
    Fatal("BHL wind tunnel requires a <hydro> block and no <mhd> block");
  }
  if (pmbp->pcoord->is_special_relativistic ||
      pmbp->pcoord->is_general_relativistic) {
    Fatal("BHL wind tunnel cannot be used with relativistic coordinates");
  }
  if (!pmy_mesh_->multi_d) {
    Fatal("BHL wind tunnel requires a 2D or 3D Cartesian mesh");
  }
  if (phydro->psrc == nullptr ||
      !phydro->psrc->point_particle_gravity_at_center) {
    Fatal("BHL wind tunnel requires <hydro_srcterms>/"
          "point_particle_gravity_at_center=true");
  }
  if (!phydro->use_sphere_mask) {
    Fatal("BHL wind tunnel requires <sphere_mask>/enabled=true");
  }
  if (phydro->sphere_mask_bc == hydro::Hydro::SphereMaskBC::dirichlet) {
    Fatal("BHL wind tunnel does not allow <sphere_mask>/bc=dirichlet, which "
          "pins a "
          "fixed Cartesian velocity inside the mask; use absorbing, "
          "reflecting, or "
          "spherical_wind");
  }
  if (phydro->nscalars > 1) {
    Fatal("BHL wind tunnel supports at most one passive scalar, "
          "<hydro>/nscalars = 0 or 1");
  }

  auto &eos = phydro->peos->eos_data;
  wind_tunnel.is_ideal = eos.is_ideal;
  wind_tunnel.dens_inf = pin->GetOrAddReal("problem", "dens_inf", 1.0);
  if (pin->DoesParameterExist("problem", "mach_inf")) {
    Fatal("<problem>/mach_inf has been replaced by <problem>/vel_inf, the "
          "upstream "
          "speed in code units");
  }
  wind_tunnel.vel_inf = pin->GetOrAddReal("problem", "vel_inf", 2.0);
  if (wind_tunnel.is_ideal) {
    if (!(eos.gamma > 1.0)) {
      Fatal("BHL wind tunnel requires gamma > 1 for an ideal-gas EOS");
    }
    wind_tunnel.cs_inf = pin->GetOrAddReal("problem", "cs_inf", 1.0);
  } else {
    wind_tunnel.cs_inf = eos.iso_cs;
  }

  if (!(wind_tunnel.dens_inf > 0.0)) {
    Fatal("BHL wind tunnel requires <problem>/dens_inf > 0");
  }
  if (!(wind_tunnel.cs_inf > 0.0)) {
    Fatal("BHL wind tunnel requires a positive upstream sound speed");
  }
  if (!(wind_tunnel.vel_inf > 0.0)) {
    Fatal("BHL wind tunnel requires <problem>/vel_inf > 0");
  }

  wind_tunnel.R_B =
      2.0 / (SQR(wind_tunnel.vel_inf) + SQR(wind_tunnel.cs_inf));
  if (wind_tunnel.is_ideal) {
    const Real pres_inf =
        wind_tunnel.dens_inf * SQR(wind_tunnel.cs_inf) / eos.gamma;
    wind_tunnel.eint_inf = pres_inf / (eos.gamma - 1.0);
  } else {
    wind_tunnel.eint_inf = 0.0;
  }
  wind_tunnel.cons_mom1 = wind_tunnel.dens_inf * wind_tunnel.vel_inf;
  wind_tunnel.cons_etot = wind_tunnel.eint_inf +
                          0.5 * wind_tunnel.dens_inf * SQR(wind_tunnel.vel_inf);

  // Require the mask radius to be smaller than R_B and confine gravitational
  // softening to the mask, so the force outside it is exactly Newtonian.
  if (!(phydro->sphere_mask_radius < wind_tunnel.R_B)) {
    Fatal("<sphere_mask>/radius must be smaller than the accretion radius "
          "2GM/(vel_inf^2+cs_inf^2); reduce the radius, vel_inf, or cs_inf");
  }
  if (!(phydro->psrc->softening_length > 0.0 &&
        phydro->psrc->softening_length <= 1.0)) {
    Fatal("<hydro_srcterms>/gravity_softening_length must be positive and no "
          "larger "
          "than 1.0");
  }
  if (phydro->psrc->softening_length > phydro->sphere_mask_radius) {
    Fatal("gravity_softening_length must be no larger than the mask radius");
  }

  user_bcs_func = FixedWindBoundary;

  // Optional histories: flux<n> for spherical fluxes, vol<n> for the domain
  // integrals restricted by radius, and domain for totals outside the mask.
  // BHLUserHistory dispatches files in that order.
  if (user_hist) {
    if (!pmy_mesh_->three_d) {
      Fatal("<problem>/user_hist (spherical and volume integrals) requires a "
            "3D mesh");
    }
    const bool has_flux = pin->DoesParameterExist("problem", "flux_radii");
    const bool has_volume = pin->DoesParameterExist("problem", "volume_radii");
    const bool has_domain =
        pin->GetOrAddBoolean("problem", "domain_hist", false);
    if (!has_flux && !has_volume && !has_domain) {
      Fatal("<problem>/user_hist=true requires <problem>/flux_radii, "
            "<problem>/volume_radii and/or <problem>/domain_hist=true");
    }
    // A face at x_d = 0 with a reflecting BC is a symmetry plane: the integrals
    // cover the full domain by mirroring (see SphericalFluxHistory). A flux
    // sphere must otherwise fit inside the mesh.
    auto &msize = pmy_mesh_->mesh_size;
    const Real lo[3] = {msize.x1min, msize.x2min, msize.x3min};
    const Real hi[3] = {msize.x1max, msize.x2max, msize.x3max};
    Real extent = std::numeric_limits<Real>::max();
    for (int d = 0; d < 3; ++d) {
      const bool fold_lo =
          (lo[d] == 0.0 && pmy_mesh_->mesh_bcs[2 * d] == BoundaryFlag::reflect);
      const bool fold_hi = (hi[d] == 0.0 && pmy_mesh_->mesh_bcs[2 * d + 1] ==
                                                BoundaryFlag::reflect);
      symmetry_fold[d] = fold_lo ? 1 : (fold_hi ? -1 : 0);
      if (!fold_lo)
        extent = std::min(extent, -lo[d]);
      if (!fold_hi)
        extent = std::min(extent, hi[d]);
    }

    if (has_flux) {
      // geodesic grid level of the flux spheres: 10*nlev^2 + 2 points each
      const int flux_nlev = pin->GetOrAddInteger("problem", "flux_nlev", 5);
      if (flux_nlev < 1)
        Fatal("<problem>/flux_nlev must be at least 1");
      for (const Real flux_radius : ParseRadii(pin, "flux_radii")) {
        if (!(flux_radius > phydro->sphere_mask_radius &&
              flux_radius < extent)) {
          Fatal(
              "each <problem>/flux_radii entry must lie between the mask "
              "radius and "
              "the distance from the origin to the nearest domain boundary (a "
              "boundary at coordinate 0 only counts as a symmetry plane if it "
              "is "
              "reflecting)");
        }
        spherical_grids.push_back(
            std::make_unique<SphericalGrid>(pmbp, flux_nlev, flux_radius));
        spherical_grids.back()->FoldInterpolationCoordinates(symmetry_fold);
        user_hist_tags.push_back("flux" +
                                 std::to_string(user_hist_tags.size()));
      }
    }
    nflux_files = user_hist_tags.size();

    if (has_volume) {
      volume_radii = ParseRadii(pin, "volume_radii");
      for (std::size_t n = 0; n < volume_radii.size(); ++n) {
        if (!(volume_radii[n] > phydro->sphere_mask_radius)) {
          Fatal("each <problem>/volume_radii entry must exceed "
                "<sphere_mask>/radius");
        }
        user_hist_tags.push_back("vol" + std::to_string(n));
      }
    }

    if (has_domain)
      user_hist_tags.push_back("domain");
    user_hist_func = BHLUserHistory;
  }

  if (global_variable::my_rank == 0)
    PrintSetup(phydro, user_hist_tags, spherical_grids);

  if (restart)
    return;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const int nhydro = phydro->nhydro;
  const int nscalars = phydro->nscalars;
  auto &w0 = phydro->w0;
  const WindTunnelData p = wind_tunnel;

  par_for(
      "bhl_wind_tunnel_init", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        w0(m, IDN, k, j, i) = p.dens_inf;
        w0(m, IVX, k, j, i) = p.vel_inf;
        w0(m, IVY, k, j, i) = 0.0;
        w0(m, IVZ, k, j, i) = 0.0;
        if (p.is_ideal)
          w0(m, IEN, k, j, i) = p.eint_inf;
        for (int n = nhydro; n < nhydro + nscalars; ++n)
          w0(m, n, k, j, i) = 0.0;
      });

  phydro->peos->PrimToCons(w0, phydro->u0, is, ie, js, je, ks, ke);
}
