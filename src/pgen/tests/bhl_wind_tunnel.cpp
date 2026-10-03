//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bhl_wind_tunnel.cpp
//! \brief Uniform wind tunnel for Bondi-Hoyle-Lyttleton (BHL) accretion, Newtonian hydro.
//!
//! A uniform flow of density dens_inf and Mach number mach_inf is directed along +x1 past
//! the GM=1 point mass from <hydro_srcterms>/point_particle_gravity_at_center. The
//! singular center is replaced by the spherical inner mask (<sphere_mask>); all bc
//! options except dirichlet are allowed. The mask is a boundary condition, not a sink, so
//! the accretion rate must be measured as an inward mass flux through a shell outside the
//! mask radius.
//!
//! The initial state is the uniform wind, evolved until the wake settles. The length
//! scale is the accretion radius r_acc = 2GM/(vel_inf^2 + cs_inf^2). Startup prints the
//! Hoyle-Lyttleton and Bondi-Hoyle rate estimates (and the Bondi rate where it exists,
//! gamma < 5/3); measured rates should lie between them and depend on the mask radius
//! until it is small compared with r_acc.
//!
//! Ideal and isothermal EOS are supported (isothermal takes cs from
//! <hydro>/iso_sound_speed). The mesh may be 2D (a cheap qualitative pilot, still with
//! the 3D 1/r potential) or 3D. Faces flagged 'user' are held at the upstream state, e.g.
//! ix1_bc=user with ox1_bc=outflow.
//!
//! References: Hoyle & Lyttleton 1939, PCPS, 35, 405; Bondi & Hoyle 1944, MNRAS,
//! 104, 273; Edgar 2004, NewAR, 48, 843.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "srcterms/srcterms.hpp"

namespace {

struct WindTunnelData {
  Real dens_inf;     // upstream density
  Real cs_inf;       // upstream sound speed
  Real eint_inf;     // upstream internal energy density (ideal EOS only)
  Real mach_inf;     // upstream Mach number
  Real vel_inf;      // upstream speed, directed along +x1
  Real r_acc;        // accretion radius 2GM/(vel_inf^2+cs_inf^2), with GM=1
  Real cons_mom1;    // upstream conserved state, precomputed for the boundary kernels
  Real cons_etot;
  bool is_ideal;
};

WindTunnelData wind_tunnel;

[[noreturn]] void Fatal(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

// Overwrite one boundary cell with the uniform upstream state.
KOKKOS_INLINE_FUNCTION
void SetUpstream(const WindTunnelData p, const DvceArray5D<Real> &u0, const int m,
                 const int k, const int j, const int i, const int nhydro,
                 const int nscalars) {
  u0(m,IDN,k,j,i) = p.dens_inf;
  u0(m,IM1,k,j,i) = p.cons_mom1;
  u0(m,IM2,k,j,i) = 0.0;
  u0(m,IM3,k,j,i) = 0.0;
  if (p.is_ideal) u0(m,IEN,k,j,i) = p.cons_etot;
  for (int n=nhydro; n<nhydro+nscalars; ++n) u0(m,n,k,j,i) = 0.0;
}

// Hold every face flagged 'user' at the uniform upstream state.  Conserved variables are
// written directly, since converting a whole array from primitives at this point would
// overwrite active cells with stale stage data.
void FixedWindBoundary(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *phydro = pmbp->phydro;
  if (phydro == nullptr) return;

  auto &indcs = pm->mb_indcs;
  const int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  const int nmb = pmbp->nmb_thispack;
  const int nhydro = phydro->nhydro;
  const int nscalars = phydro->nscalars;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &u0 = phydro->u0;
  const WindTunnelData p = wind_tunnel;

  par_for("bhl_wind_bc_x1", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(int m, int k, int j, int g) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      SetUpstream(p, u0, m, k, j, g, nhydro, nscalars);
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      SetUpstream(p, u0, m, k, j, ie+1+g, nhydro, nscalars);
    }
  });

  par_for("bhl_wind_bc_x2", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, ng-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int k, int g, int i) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) {
      SetUpstream(p, u0, m, k, g, i, nhydro, nscalars);
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
      SetUpstream(p, u0, m, k, je+1+g, i, nhydro, nscalars);
    }
  });

  if (pm->three_d) {
    par_for("bhl_wind_bc_x3", DevExeSpace(), 0, nmb-1, 0, ng-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(int m, int g, int j, int i) {
      if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user) {
        SetUpstream(p, u0, m, g, j, i, nhydro, nscalars);
      }
      if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user) {
        SetUpstream(p, u0, m, ke+1+g, j, i, nhydro, nscalars);
      }
    });
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::BHLWindTunnel()
//! \brief Initialize a uniform wind flowing past a masked, gravitating GM=1 point mass.

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
    Fatal("BHL wind tunnel does not allow <sphere_mask>/bc=dirichlet, which pins a "
          "fixed Cartesian velocity inside the mask; use absorbing, reflecting, or "
          "spherical_wind");
  }

  auto &eos = phydro->peos->eos_data;
  wind_tunnel.is_ideal = eos.is_ideal;
  wind_tunnel.dens_inf = pin->GetOrAddReal("problem", "dens_inf", 1.0);
  wind_tunnel.mach_inf = pin->GetOrAddReal("problem", "mach_inf", 2.0);
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
  if (!(wind_tunnel.mach_inf > 0.0)) {
    Fatal("BHL wind tunnel requires <problem>/mach_inf > 0");
  }

  wind_tunnel.vel_inf = wind_tunnel.mach_inf*wind_tunnel.cs_inf;
  wind_tunnel.r_acc = 2.0/(SQR(wind_tunnel.vel_inf) + SQR(wind_tunnel.cs_inf));
  if (wind_tunnel.is_ideal) {
    const Real pres_inf = wind_tunnel.dens_inf*SQR(wind_tunnel.cs_inf)/eos.gamma;
    wind_tunnel.eint_inf = pres_inf/(eos.gamma - 1.0);
  } else {
    wind_tunnel.eint_inf = 0.0;
  }
  wind_tunnel.cons_mom1 = wind_tunnel.dens_inf*wind_tunnel.vel_inf;
  wind_tunnel.cons_etot = wind_tunnel.eint_inf
                          + 0.5*wind_tunnel.dens_inf*SQR(wind_tunnel.vel_inf);

  // The mask must be small compared with the accretion radius for the captured column to
  // be resolved at all; the gravitational field must be exactly 1/r^2 outside the mask.
  if (!(phydro->sphere_mask_radius < wind_tunnel.r_acc)) {
    Fatal("<sphere_mask>/radius must be smaller than the accretion radius "
          "2GM/(vel_inf^2+cs_inf^2); reduce the radius or the upstream Mach number");
  }
  if (!(phydro->psrc->softening_length > 0.0 && phydro->psrc->softening_length <= 1.0)) {
    Fatal("<hydro_srcterms>/gravity_softening_length must be positive and no larger "
          "than 1.0");
  }
  if (phydro->psrc->softening_length > phydro->sphere_mask_radius) {
    Fatal("gravity_softening_length must be no larger than the mask radius");
  }

  user_bcs_func = FixedWindBoundary;
  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const int nhydro = phydro->nhydro;
  const int nscalars = phydro->nscalars;
  auto &w0 = phydro->w0;
  const WindTunnelData p = wind_tunnel;

  par_for("bhl_wind_tunnel_init", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0(m,IDN,k,j,i) = p.dens_inf;
    w0(m,IVX,k,j,i) = p.vel_inf;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    if (p.is_ideal) w0(m,IEN,k,j,i) = p.eint_inf;
    for (int n=nhydro; n<nhydro+nscalars; ++n) w0(m,n,k,j,i) = 0.0;
  });

  phydro->peos->PrimToCons(w0, phydro->u0, is, ie, js, je, ks, ke);

  if (global_variable::my_rank == 0) {
    const Real gm2rho = wind_tunnel.dens_inf;  // (GM)^2 dens_inf, with GM=1
    const Real mdot_hl = 4.0*M_PI*gm2rho/(SQR(wind_tunnel.vel_inf)*wind_tunnel.vel_inf);
    const Real mdot_bhl = 4.0*M_PI*gm2rho
                          /pow(SQR(wind_tunnel.vel_inf) + SQR(wind_tunnel.cs_inf), 1.5);
    std::cout << "BHL wind tunnel (GM=1): mach_inf=" << wind_tunnel.mach_inf
              << "  vel_inf=" << wind_tunnel.vel_inf
              << "  cs_inf=" << wind_tunnel.cs_inf
              << "  r_acc=" << wind_tunnel.r_acc
              << "  r_mask/r_acc="
              << phydro->sphere_mask_radius/wind_tunnel.r_acc << std::endl;
    if (phydro->sphere_mask_bc == hydro::Hydro::SphereMaskBC::spherical_wind) {
      // v_esc at the mask surface sets whether the injected wind escapes or falls back
      std::cout << "BHL wind tunnel mask wind: dens=" << phydro->sm_dens
                << "  eint=" << phydro->sm_eint
                << "  vel_r=" << phydro->sm_velr
                << "  v_esc(r_mask)=" << sqrt(2.0/phydro->sphere_mask_radius)
                << std::endl;
    }
    std::cout << "BHL wind tunnel rate estimates: mdot_bhl=" << mdot_bhl
              << "  mdot_hl=" << mdot_hl;
    // The Bondi rate exists only where the spherical transonic solution does.
    const Real gamma = wind_tunnel.is_ideal ? eos.gamma : 1.0;
    const Real q = 5.0 - 3.0*gamma;
    if (q > 0.0) {
      const Real lambda = wind_tunnel.is_ideal ? 0.25*pow(2.0/q, 0.5*q/(gamma - 1.0))
                                               : 0.25*exp(1.5);
      const Real mdot_b = 4.0*M_PI*lambda*gm2rho
                          /(SQR(wind_tunnel.cs_inf)*wind_tunnel.cs_inf);
      std::cout << "  mdot_bondi=" << mdot_b;
    }
    std::cout << std::endl;
  }
}
