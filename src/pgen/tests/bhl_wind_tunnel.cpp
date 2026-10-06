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
//! accretion rate must be measured as minus the net outward mass flux through a
//! shell outside the mask radius.
//!
//! The initial state is the uniform wind, evolved until the wake settles. The
//! length scale is the accretion radius r_acc = 2GM/(vel_inf^2 + cs_inf^2).
//! Startup prints the upstream and mask-wind states and the Hoyle-Lyttleton and
//! Bondi-Hoyle rate estimates (and the Bondi rate where it exists, gamma <
//! 5/3); measured rates should lie between them and depend on the mask radius
//! until it is small compared with r_acc.
//!
//! Ideal and isothermal EOS are supported (isothermal takes cs from
//! <hydro>/iso_sound_speed). The mesh may be 2D (a cheap qualitative pilot,
//! still with the 3D 1/r potential) or 3D. Faces flagged 'user' are held at the
//! upstream state, e.g. ix1_bc=user with ox1_bc=outflow.
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
  Real r_acc;     // accretion radius 2GM/(vel_inf^2+cs_inf^2), with GM=1
  Real cons_mom1; // upstream conserved state, precomputed for the boundary
                  // kernels
  Real cons_etot;
  bool is_ideal;
};

WindTunnelData wind_tunnel;

// Coordinate planes through the point mass that are reflecting mesh faces, for
// history output on a mesh covering a symmetric part of the domain: +1 (-1) if
// the mesh covers x_d >= 0 (x_d <= 0) with a reflecting face at x_d = 0, else
// 0.
int flux_fold[3] = {0, 0, 0};

// User history files 0..nflux_files-1 are the spherical fluxes (one per
// SphericalGrid); the next volume_radii.size() are the gravitational force of
// the gas in the shells volume_rmin <= r < volume_radii[n] on the point mass;
// an optional last file holds totals over the whole domain outside the mask
// (<problem>/domain_hist).
int nflux_files = 0;
std::vector<Real> volume_radii;
Real volume_rmin = 1.0;

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
//! v.rhat + pres rhat_i) dA, pressure included. Called once per radius; each
//! gets its own history file, <basename>.user.flux<n>.hst, n = 0, 1, ... in
//! flux_radii order. If the mesh covers only half (or a quarter, ...) of the
//! domain behind reflecting faces at x_d = 0, the part of the sphere outside
//! the mesh is sampled at the mirror image points and the fluxes are those of
//! the full sphere.

void SphericalFluxHistory(HistoryData *pdata, Mesh *pm) {
  auto &grid = pm->pgen->spherical_grids[pdata->user_index];
  auto *phydro = pm->pmb_pack->phydro;
  auto &eos = phydro->peos->eos_data;

  pdata->nhist = 4;
  pdata->label[0] = "mflux";
  pdata->label[1] = "pflux1";
  pdata->label[2] = "pflux2";
  pdata->label[3] = "pflux3";

  // Each rank interpolates onto the angles it owns and zeros the rest; the
  // history output then sums over ranks.
  grid->InterpolateToSphere(phydro->nhydro, phydro->w0);
  const Real r = grid->radius;
  Real flux[4] = {0.0, 0.0, 0.0, 0.0};
  for (int n = 0; n < grid->nangles; ++n) {
    const Real theta = grid->polar_pos.h_view(n, 0);
    const Real phi = grid->polar_pos.h_view(n, 1);
    const Real rhat[3] = {sin(theta) * cos(phi), sin(theta) * sin(phi),
                          cos(theta)};
    const Real dens = grid->interp_vals.h_view(n, IDN);
    Real vel[3] = {grid->interp_vals.h_view(n, IVX),
                   grid->interp_vals.h_view(n, IVY),
                   grid->interp_vals.h_view(n, IVZ)};
    // the value was interpolated at the mirror image of this point: flip its
    // normal velocity component
    for (int d = 0; d < 3; ++d) {
      if (flux_fold[d] * rhat[d] < 0.0)
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
  }
  for (int n = 0; n < pdata->nhist; ++n)
    pdata->hdata[n] = flux[n];

  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n)
    pdata->hdata[n] = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void CellSums()
//! \brief Volume integrals over the cells whose centers lie in rin <= r < rout,
//! from the conserved variables: sums[0] = mass \int dens dV; sums[1..3] =
//! gravitational force of the gas on the GM=1 point mass \int dens x/r^3 dV
//! (minus the force of the point mass on the gas), exactly Newtonian as rin >=
//! the softening length; sums[4..6] = change of the linear momentum from the
//! uniform upstream flow \int (mom - mom_inf) dV. Cells are included by the
//! position of their center, as the sphere mask does, so shell edges are
//! accurate to a cell, and only the part of the shell inside the mesh is
//! integrated. On a mesh covering a symmetric part of the domain (see
//! SphericalFluxHistory) the sums are those of the full domain: they are scaled
//! by the number of mirror images and the vector components normal to a
//! symmetry plane, which cancel between the images, are zero.

void CellSums(Mesh *pm, const Real rin, const Real rout, Real sums[7]) {
  auto &u0 = pm->pmb_pack->phydro->u0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  const int nmkji = pm->pmb_pack->nmb_thispack * nkji;
  const Real mom1_inf = wind_tunnel.cons_mom1;

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
          const Real dm = vol * u0(m, IDN, k + ks, j + js, i + is);
          const Real fac = dm / (r2 * r);
          sum.the_array[0] += dm;
          sum.the_array[1] += fac * x1;
          sum.the_array[2] += fac * x2;
          sum.the_array[3] += fac * x3;
          sum.the_array[4] +=
              vol * (u0(m, IM1, k + ks, j + js, i + is) - mom1_inf);
          sum.the_array[5] += vol * u0(m, IM2, k + ks, j + js, i + is);
          sum.the_array[6] += vol * u0(m, IM3, k + ks, j + js, i + is);
        }
      },
      Kokkos::Sum<array_sum::GlobalSum>(sum_this_rank));

  Real nimages = 1.0;
  for (int d = 0; d < 3; ++d)
    nimages *= (flux_fold[d] != 0) ? 2.0 : 1.0;
  sums[0] = nimages * sum_this_rank.the_array[0];
  for (int n = 1; n < 7; ++n) {
    sums[n] = (flux_fold[(n - 1) % 3] != 0)
                  ? 0.0
                  : nimages * sum_this_rank.the_array[n];
  }
}

//----------------------------------------------------------------------------------------
//! \fn void GravityForceHistory()
//! \brief User history: total gravitational force of the gas in the shell
//! volume_rmin <= r < rout on the GM=1 point mass, see CellSums. Called once
//! per entry of <problem>/volume_radii; each gets its own history file,
//! <basename>.user.vol<n>.hst.

void GravityForceHistory(HistoryData *pdata, Mesh *pm, const Real rout) {
  pdata->nhist = 3;
  pdata->label[0] = "fgrav1";
  pdata->label[1] = "fgrav2";
  pdata->label[2] = "fgrav3";

  Real sums[7];
  CellSums(pm, volume_rmin, rout, sums);
  for (int d = 0; d < 3; ++d)
    pdata->hdata[d] = sums[1 + d];

  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n)
    pdata->hdata[n] = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void DomainHistory()
//! \brief User history: totals over the whole domain outside the mask (the
//! cells the sphere mask does not overwrite), see CellSums, in
//! <basename>.user.domain.hst. Columns: the mass; the three components of the
//! gravitational force of the gas on the point mass; and the change of the
//! three linear momentum components from the initial uniform flow, which is
//! dens_inf*vel_inf*(volume) along x1. Mass and momentum cross the domain
//! boundaries, so only the change is meaningful, not the value itself.

void DomainHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 7;
  pdata->label[0] = "mass";
  pdata->label[1] = "fgrav1";
  pdata->label[2] = "fgrav2";
  pdata->label[3] = "fgrav3";
  pdata->label[4] = "dmom1";
  pdata->label[5] = "dmom2";
  pdata->label[6] = "dmom3";

  Real sums[7];
  CellSums(pm, pm->pmb_pack->phydro->sphere_mask_radius,
           std::numeric_limits<Real>::max(), sums);
  for (int n = 0; n < pdata->nhist; ++n)
    pdata->hdata[n] = sums[n];

  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n)
    pdata->hdata[n] = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void BHLUserHistory()
//! \brief Dispatches the user history files, see nflux_files.

void BHLUserHistory(HistoryData *pdata, Mesh *pm) {
  const int nvol_files = volume_radii.size();
  if (pdata->user_index < nflux_files) {
    SphericalFluxHistory(pdata, pm);
  } else if (pdata->user_index < nflux_files + nvol_files) {
    GravityForceHistory(pdata, pm,
                        volume_radii[pdata->user_index - nflux_files]);
  } else {
    DomainHistory(pdata, pm);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PrintSetup()
//! \brief Startup summary on rank 0: the upstream flow, the mask wind, the
//! accretion rate estimates, and the user history files.

void PrintSetup(hydro::Hydro *phydro, const std::vector<std::string> &tags,
                const std::vector<std::unique_ptr<SphericalGrid>> &grids) {
  auto &eos = phydro->peos->eos_data;
  const WindTunnelData &p = wind_tunnel;
  std::cout << std::endl
            << "BHL wind tunnel (GM = 1) : r_acc = " << p.r_acc
            << ", mach_inf = " << p.vel_inf / p.cs_inf << std::endl
            << "- dens_inf = " << p.dens_inf << std::endl
            << "- vel_inf = " << p.vel_inf << std::endl
            << "- cs_inf = " << p.cs_inf << std::endl;

  if (phydro->sphere_mask_bc == hydro::Hydro::SphereMaskBC::spherical_wind) {
    const Real cs_mask = p.is_ideal ? sqrt(eos.gamma * (eos.gamma - 1.0) *
                                           phydro->sm_eint / phydro->sm_dens)
                                    : eos.iso_cs;
    std::cout << std::endl
              << "Spherical wind from the mask: mach_mask = "
              << phydro->sm_velr / cs_mask << std::endl
              << "- dens = " << phydro->sm_dens << std::endl
              << "- vel_r = " << phydro->sm_velr << std::endl
              << "- cs = " << cs_mask << std::endl;
  }

  const Real gm2rho = p.dens_inf; // (GM)^2 dens_inf, with GM=1
  std::cout << std::endl
            << "Accretion rate estimates:" << std::endl
            << "- mdot_bhl = "
            << 4.0 * M_PI * gm2rho / pow(SQR(p.vel_inf) + SQR(p.cs_inf), 1.5)
            << std::endl
            << "- mdot_hl = "
            << 4.0 * M_PI * gm2rho / (SQR(p.vel_inf) * p.vel_inf) << std::endl;
  // The Bondi rate exists only where the spherical transonic solution does.
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
        std::cout << "fluxes through the sphere r = " << grids[n]->radius;
      } else if (n < nflux_files + static_cast<int>(volume_radii.size())) {
        std::cout << "gravitational force of the shell " << volume_rmin
                  << " <= r < " << volume_radii[n - nflux_files];
      } else {
        std::cout
            << "mass, gravitational force and momentum change of the domain "
            << "outside the mask";
      }
      std::cout << std::endl;
    }
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

  wind_tunnel.r_acc =
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

  // The mask must be small compared with the accretion radius for the captured
  // column to be resolved at all; the gravitational field must be exactly 1/r^2
  // outside the mask.
  if (!(phydro->sphere_mask_radius < wind_tunnel.r_acc)) {
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

  // Optional history output, one file per radius: mass and momentum fluxes
  // through the spheres in flux_radii, and the gravitational force of the gas
  // in the shells volume_rmin <= r < volume_radii[n] on the point mass; and, if
  // domain_hist is set, one file of totals over the whole domain outside the
  // mask.
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
      flux_fold[d] = fold_lo ? 1 : (fold_hi ? -1 : 0);
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
        spherical_grids.back()->FoldInterpolationCoordinates(flux_fold);
        user_hist_tags.push_back("flux" +
                                 std::to_string(user_hist_tags.size()));
      }
    }
    nflux_files = user_hist_tags.size();

    if (has_volume) {
      volume_rmin = pin->GetOrAddReal("problem", "volume_rmin", 1.0);
      if (!(volume_rmin >= phydro->psrc->softening_length)) {
        Fatal("<problem>/volume_rmin must be at least "
              "gravity_softening_length, so "
              "that the force in the shell is exactly Newtonian");
      }
      volume_radii = ParseRadii(pin, "volume_radii");
      for (std::size_t n = 0; n < volume_radii.size(); ++n) {
        if (!(volume_radii[n] > volume_rmin)) {
          Fatal("each <problem>/volume_radii entry must exceed "
                "<problem>/volume_rmin");
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
