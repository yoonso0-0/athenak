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
//! replaced by the spherical inner mask (<sphere_mask>); bc=reflecting and
//! bc=spherical_wind are allowed. The mask is a boundary condition, not a sink,
//! so the net inward mass flow is measured as minus the outward mass flux
//! through a sphere outside the mask radius. With an injected wind, this is the
//! net flow, not the gross inflow rate.
//!
//! The initial state is the uniform wind, evolved until the wake settles. The
//! upstream state is set by <problem>/accretion_radius
//! R_B = 2GM/(vel_inf^2 + cs_inf^2) and <problem>/mach_inf = vel_inf/cs_inf,
//! from which vel_inf and cs_inf are computed. Startup prints the upstream and
//! mask-wind states and the Bondi-Hoyle rate estimate (and the Bondi rate for
//! gamma < 5/3). The rate estimates are reference values for accretion without
//! an injected mask wind.
//!
//! For bc=spherical_wind, <problem>/mask_vel_r_over_vel_inf sets the radial
//! velocity of the mask wind in units of vel_inf, overriding
//! <sphere_mask>/vel_r, and <problem>/mask_dens_over_dens_inf sets its density
//! in units of dens_inf, overriding <sphere_mask>/dens. For an ideal-gas EOS,
//! <problem>/mach_spherical_wind = |vel_r|/cs_mask sets the mask internal
//! energy density, overriding <sphere_mask>/eint.
//!
//! Ideal and isothermal EOS are supported. For isothermal gas the computed
//! cs_inf overrides <hydro>/iso_sound_speed. The mesh is 3D, or 2D for quick
//! qualitative tests (still with the 3D 1/r potential). Faces flagged 'user'
//! are held at the upstream state, e.g. ix1_bc=user with ox1_bc=outflow.
//!
//! An optional passive scalar (<hydro>/nscalars=1) tags the gas injected by the
//! mask: s = 0 upstream and in the initial state, while the mask holds s =
//! <sphere_mask>/scalar for bc=spherical_wind (set it to 1, so s is the mass
//! fraction of wind material) and s = 0 for reflecting.
//!
//! References: Hoyle & Lyttleton 1939, PCPS, 35, 405; Bondi & Hoyle 1944,
//! MNRAS, 104, 273; Edgar 2004, NewAR, 48, 843.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
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

[[noreturn]] void Fatal(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

// Replace <block>/name, already read into field, by value (derived from source):
// in field and in the stored parameters, so that parameter dumps and restart
// files record the value actually used. Rank 0 prints a notice if the value read
// differs, unless it is 0 (the default of the mask state when not given).
// SetReal keeps 6 significant digits, so a restart reads back a value within
// ~5e-6 of the one computed here: the notice tolerance must be looser than that.
void ReplaceParameter(ParameterInput *pin, const std::string &block,
                      const std::string &name, const std::string &source,
                      const Real value, Real &field) {
  const Real notice_tol = 1.0e-5;
  if (global_variable::my_rank == 0 && field != 0.0 &&
      fabs(field - value) > notice_tol * fabs(value)) {
    std::cout << "BHL wind tunnel: <" << block << ">/" << name << " = " << field
              << " replaced by " << value << " from " << source << std::endl;
  }
  field = value;
  pin->SetReal(block, name, value);
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
//! \fn void PrintSetup()
//! \brief Startup summary on rank 0: the upstream flow, the mask wind, the
//! passive scalar and the accretion rate estimates.

void PrintSetup(hydro::Hydro *phydro) {
  auto &eos = phydro->peos->eos_data;
  const WindTunnelData &p = wind_tunnel;
  std::cout << std::endl
            << "BHL wind tunnel (GM = 1)" << std::endl
            << "- R_B = " << p.R_B << ", mach_inf = " << p.vel_inf / p.cs_inf
            << std::endl
            << "- dens_inf = " << p.dens_inf << std::endl
            << "- vel_inf = " << p.vel_inf << std::endl
            << "- cs_inf = " << p.cs_inf << std::endl;

  if (phydro->sphere_mask_bc == hydro::Hydro::SphereMaskBC::spherical_wind) {
    const Real cs_mask = p.is_ideal ? sqrt(eos.gamma * (eos.gamma - 1.0) *
                                           phydro->sm_eint / phydro->sm_dens)
                                    : eos.iso_cs;
    // Ram-pressure standoff of a free outward wind of constant speed vel_r from
    // the mask surface, where dens(r) = sm_dens (r_mask/r)^2:
    // dens(R_0) vel_r^2 = dens_inf vel_inf^2, so R_0 = r_mask sqrt(ram_ratio)
    // with ram_ratio the ram-pressure ratio at the mask surface. Without an
    // outward wind there is no standoff (R_0 = 0).
    const Real ram_ratio = (phydro->sm_velr > 0.0)
                               ? (phydro->sm_dens * SQR(phydro->sm_velr)) /
                                     (p.dens_inf * SQR(p.vel_inf))
                               : 0.0;
    const Real R_0 = phydro->sphere_mask_radius * sqrt(ram_ratio);
    // mach_mask = vel_r/cs_mask is signed (negative for inflow), unlike
    // <problem>/mach_spherical_wind = |vel_r|/cs_mask.
    std::cout << std::endl
              << "Spherical wind from the mask" << std::endl
              << "- R_0 = " << R_0
              << ", mach_mask = " << phydro->sm_velr / cs_mask << std::endl
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
          "pins a fixed Cartesian velocity inside the mask; use reflecting or "
          "spherical_wind");
  }
  if (phydro->nscalars > 1) {
    Fatal("BHL wind tunnel supports at most one passive scalar, "
          "<hydro>/nscalars = 0 or 1");
  }

  auto &eos = phydro->peos->eos_data;
  wind_tunnel.is_ideal = eos.is_ideal;
  wind_tunnel.dens_inf = pin->GetOrAddReal("problem", "dens_inf", 1.0);
  wind_tunnel.R_B = pin->GetOrAddReal("problem", "accretion_radius", 0.4);
  const Real mach_inf = pin->GetOrAddReal("problem", "mach_inf", 2.0);
  if (!(wind_tunnel.dens_inf > 0.0)) {
    Fatal("BHL wind tunnel requires <problem>/dens_inf > 0");
  }
  if (!(wind_tunnel.R_B > 0.0)) {
    Fatal("BHL wind tunnel requires <problem>/accretion_radius > 0");
  }
  if (!(mach_inf > 0.0)) {
    Fatal("BHL wind tunnel requires <problem>/mach_inf > 0");
  }
  if (wind_tunnel.is_ideal && !(eos.gamma > 1.0)) {
    Fatal("BHL wind tunnel requires gamma > 1 for an ideal-gas EOS");
  }

  // Invert R_B = 2GM/(vel_inf^2 + cs_inf^2), with vel_inf = mach_inf*cs_inf
  // and GM = 1.
  wind_tunnel.cs_inf = sqrt(2.0 / (wind_tunnel.R_B * (1.0 + SQR(mach_inf))));
  wind_tunnel.vel_inf = mach_inf * wind_tunnel.cs_inf;
  // eint_inf = p_inf/(gamma-1), with p_inf = dens_inf cs_inf^2/gamma for the
  // adiabatic sound speed cs_inf.
  wind_tunnel.eint_inf =
      wind_tunnel.is_ideal ? wind_tunnel.dens_inf * SQR(wind_tunnel.cs_inf) /
                                 (eos.gamma * (eos.gamma - 1.0))
                           : 0.0;
  wind_tunnel.cons_mom1 = wind_tunnel.dens_inf * wind_tunnel.vel_inf;
  wind_tunnel.cons_etot = wind_tunnel.eint_inf +
                          0.5 * wind_tunnel.dens_inf * SQR(wind_tunnel.vel_inf);

  // An isothermal sound speed is fixed by R_B and mach_inf, so it overrides
  // <hydro>/iso_sound_speed.
  if (!wind_tunnel.is_ideal) {
    ReplaceParameter(pin, "hydro", "iso_sound_speed",
                     "<problem>/accretion_radius and mach_inf",
                     wind_tunnel.cs_inf, eos.iso_cs);
  }

  // Optionally set the mask wind relative to the upstream state. This replaces
  // the <sphere_mask> values Hydro has already read into sm_velr, sm_dens and
  // sm_eint; the mask reads those each time it is applied.
  for (const char *name : {"mask_vel_r_over_vel_inf", "mask_dens_over_dens_inf",
                           "mach_spherical_wind"}) {
    if (pin->DoesParameterExist("problem", name) &&
        phydro->sphere_mask_bc != hydro::Hydro::SphereMaskBC::spherical_wind) {
      Fatal(std::string("<problem>/") + name +
            " requires <sphere_mask>/bc=spherical_wind");
    }
  }
  if (pin->DoesParameterExist("problem", "mask_vel_r_over_vel_inf")) {
    ReplaceParameter(pin, "sphere_mask", "vel_r",
                     "<problem>/mask_vel_r_over_vel_inf",
                     pin->GetReal("problem", "mask_vel_r_over_vel_inf") *
                         wind_tunnel.vel_inf,
                     phydro->sm_velr);
  }
  if (pin->DoesParameterExist("problem", "mask_dens_over_dens_inf")) {
    ReplaceParameter(pin, "sphere_mask", "dens",
                     "<problem>/mask_dens_over_dens_inf",
                     pin->GetReal("problem", "mask_dens_over_dens_inf") *
                         wind_tunnel.dens_inf,
                     phydro->sm_dens);
  }
  // <problem>/mach_spherical_wind = |vel_r|/cs_mask sets the mask internal
  // energy density from the final mask density and radial velocity above,
  // through cs_mask^2 = gamma*(gamma-1)*eint/dens.
  if (pin->DoesParameterExist("problem", "mach_spherical_wind")) {
    if (!wind_tunnel.is_ideal) {
      Fatal(
          "<problem>/mach_spherical_wind requires an ideal-gas EOS: for an "
          "isothermal EOS the mask sound speed is cs_inf, so vel_r alone sets "
          "the Mach number of the spherical wind");
    }
    const Real mach_wind = pin->GetReal("problem", "mach_spherical_wind");
    if (!(mach_wind > 0.0)) {
      Fatal("<problem>/mach_spherical_wind must be positive");
    }
    if (phydro->sm_velr == 0.0) {
      Fatal("<problem>/mach_spherical_wind requires a nonzero mask radial "
            "velocity");
    }
    const Real cs_mask = fabs(phydro->sm_velr) / mach_wind;
    ReplaceParameter(
        pin, "sphere_mask", "eint", "<problem>/mach_spherical_wind",
        phydro->sm_dens * SQR(cs_mask) / (eos.gamma * (eos.gamma - 1.0)),
        phydro->sm_eint);
  }
  // Validate the final mask state now, before PrintSetup derives the mask sound
  // speed from it (Driver::InitBoundaryValuesAndPrimitives checks it again).
  phydro->CheckSphereMaskState();

  // Confine gravitational softening to the mask, so the force outside it is
  // exactly Newtonian. The mask radius relative to R_B is left to the user.
  const Real softening = phydro->psrc->softening_length;
  if (!(softening > 0.0 && softening <= phydro->sphere_mask_radius)) {
    Fatal("<hydro_srcterms>/gravity_softening_length must be positive and no "
          "larger than <sphere_mask>/radius");
  }

  user_bcs_func = FixedWindBoundary;

  if (global_variable::my_rank == 0)
    PrintSetup(phydro);

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
