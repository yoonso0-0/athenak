//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file newtonian_spherical.cpp
//! \brief Steady, 3D Newtonian Bondi accretion and Parker wind test problems.
//!
//! Both solutions are initialized on a Cartesian mesh around the GM=1 point mass supplied
//! by <hydro_srcterms>/point_particle_gravity_at_center.  A spherical mask replaces the
//! singular central region and analytic states are imposed on every user boundary.
//!
//! Select the solution with <problem>/pgen_name:
//!   newtonian_bondi  - transonic accretion, ideal or isothermal EOS
//!   parker_wind      - transonic isothermal outflow
//!
//! For an isothermal EOS, Bondi inflow and Parker outflow share the same Mach-number
//! equation when radius is normalized by the sonic radius r_s=GM/(2 c_s^2):
//!
//!   M^2 - ln(M^2) = 4 ln(r/r_s) + 4 r_s/r - 3.
//!
//! The subsonic branch is used outside r_s for accretion and inside r_s for a wind, so
//! the branch choice reverses between the two flows.  Ideal-gas Bondi accretion instead
//! solves the Bernoulli integral for density.
//!
//! References: Bondi 1952, MNRAS, 112, 195; Parker 1958, ApJ, 128, 664.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "srcterms/srcterms.hpp"

namespace {

struct SphericalFlowData {
  Real dens_inf;       // Bondi density at infinity
  Real cs_inf;         // isothermal sound speed or Bondi sound speed at infinity
  Real pres_inf;       // Bondi pressure at infinity (ideal EOS only)
  Real gamma;          // adiabatic index; one for an isothermal EOS
  Real r_bondi;        // GM/cs_inf^2, with GM=1
  Real lambda;         // dimensionless Bondi accretion eigenvalue
  Real sonic_radius;
  Real sonic_density;  // used directly by both isothermal solutions
  Real r_floor;        // analytic solution is evaluated at max(r,r_floor)
  Real mask_radius;
  Real mask_density;   // Parker state imposed by spherical_wind
  Real mask_velocity;
  bool is_ideal;
  bool is_wind;
  bool reset_ic = false;
};

SphericalFlowData spherical_flow;

[[noreturn]] void Fatal(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

// The transonic isothermal solution is solved for q=ln(M^2).  This remains conditioned
// on the exponentially subsonic branch and avoids a device Lambert-W implementation.
KOKKOS_INLINE_FUNCTION
Real IsothermalMach(const Real radius, const Real sonic_radius, const bool wind) {
  const Real x = radius/sonic_radius;
  if (fabs(x - 1.0) < 1.0e-12) return 1.0;

  Real rhs = 4.0*log(x) + 4.0/x - 3.0;
  rhs = fmax(rhs, 1.0);
  Real qlo, qhi;
  const bool subsonic = wind ? (x < 1.0) : (x > 1.0);
  if (subsonic) {
    qlo = -rhs;
    qhi = 0.0;
  } else {
    qlo = 0.0;
    qhi = log(rhs) + 1.0;
  }

  for (int n = 0; n < 80; ++n) {
    const Real qmid = 0.5*(qlo + qhi);
    const Real residual = exp(qmid) - qmid - rhs;
    if (subsonic) {
      if (residual > 0.0) {
        qlo = qmid;
      } else {
        qhi = qmid;
      }
    } else {
      if (residual > 0.0) {
        qhi = qmid;
      } else {
        qlo = qmid;
      }
    }
  }
  return exp(0.25*(qlo + qhi));
}

// Bernoulli residual for ideal-gas Bondi flow after eliminating the inflow speed using
// mass conservation.  Here x=r/r_B and y=ln(rho/rho_inf).
KOKKOS_INLINE_FUNCTION
Real BondiResidual(const SphericalFlowData p, const Real x, const Real y) {
  const Real vt = p.lambda*exp(-y)/(x*x);
  const Real gm1 = p.gamma - 1.0;
  const Real enthalpy = (exp(gm1*y) - 1.0)/gm1;
  return 0.5*vt*vt + enthalpy - 1.0/x;
}

KOKKOS_INLINE_FUNCTION
Real IdealBondiDensity(const SphericalFlowData p, const Real x) {
  const Real y_sonic =
      (2.0*log(p.lambda) - 4.0*log(x))/(p.gamma + 1.0);

  if (BondiResidual(p, x, y_sonic) >= 0.0) return exp(y_sonic);

  const bool subsonic = (x >= p.sonic_radius/p.r_bondi);
  Real ylo = y_sonic;
  Real yhi = y_sonic;
  Real dy = 1.0;
  for (int n = 0; n < 64; ++n) {
    if (subsonic) {
      yhi = y_sonic + dy;
      if (BondiResidual(p, x, yhi) > 0.0) break;
    } else {
      ylo = y_sonic - dy;
      if (BondiResidual(p, x, ylo) > 0.0) break;
    }
    dy *= 2.0;
  }

  for (int n = 0; n < 200; ++n) {
    const Real ymid = 0.5*(ylo + yhi);
    if ((yhi - ylo) <= 1.0e-14*(1.0 + fabs(ymid))) break;
    const bool below = (BondiResidual(p, x, ymid) < 0.0);
    if (below == subsonic) {
      ylo = ymid;
    } else {
      yhi = ymid;
    }
  }
  return exp(0.5*(ylo + yhi));
}

KOKKOS_INLINE_FUNCTION
void SphericalPrimitive(const SphericalFlowData p, const Real x1, const Real x2,
                        const Real x3, Real &dens, Real &eint,
                        Real &vel1, Real &vel2, Real &vel3) {
  const Real radius = sqrt(x1*x1 + x2*x2 + x3*x3);
  const Real eval_radius = fmax(radius, p.r_floor);
  Real velr;

  if (!p.is_ideal) {
    const Real x = eval_radius/p.sonic_radius;
    const Real mach = IsothermalMach(eval_radius, p.sonic_radius, p.is_wind);
    dens = p.sonic_density/(mach*x*x);
    velr = (p.is_wind ? 1.0 : -1.0)*p.cs_inf*mach;
  } else {
    const Real x = eval_radius/p.r_bondi;
    const Real rhot = IdealBondiDensity(p, x);
    dens = p.dens_inf*rhot;
    eint = p.pres_inf*pow(rhot, p.gamma)/(p.gamma - 1.0);
    velr = -p.cs_inf*p.lambda/(x*x*rhot);
  }

  const Real inv_radius = (radius > 0.0) ? 1.0/radius : 0.0;
  vel1 = velr*x1*inv_radius;
  vel2 = velr*x2*inv_radius;
  vel3 = velr*x3*inv_radius;
}

// Write analytic conserved states directly into boundary cells.  Converting a complete
// array from primitives here would overwrite active cells with stale stage data.
void FixedSphericalBoundary(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *phydro = pmbp->phydro;
  if (phydro == nullptr) return;

  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = indcs.nx2 + 2*ng;
  const int n3 = indcs.nx3 + 2*ng;
  const int nmb = pmbp->nmb_thispack;
  const int nhydro = phydro->nhydro;
  const int nscalars = phydro->nscalars;
  auto &size = pmbp->pmb->mb_size;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &u0 = phydro->u0;
  const SphericalFlowData p = spherical_flow;

  par_for("spherical_flow_bc_x1", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(int m, int k, int j, int g) {
    const Real x2 = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                                size.d_view(m).x2max);
    const Real x3 = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                                size.d_view(m).x3max);
    Real dens, eint = 0.0, v1, v2, v3;
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      const int i = g;
      const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                  size.d_view(m).x1max);
      SphericalPrimitive(p, x1, x2, x3, dens, eint, v1, v2, v3);
      u0(m,IDN,k,j,i) = dens;
      u0(m,IM1,k,j,i) = dens*v1;
      u0(m,IM2,k,j,i) = dens*v2;
      u0(m,IM3,k,j,i) = dens*v3;
      if (p.is_ideal) u0(m,IEN,k,j,i) = eint + 0.5*dens*(v1*v1 + v2*v2 + v3*v3);
      for (int n=nhydro; n<nhydro+nscalars; ++n) u0(m,n,k,j,i) = 0.0;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      const int i = ie + 1 + g;
      const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                  size.d_view(m).x1max);
      SphericalPrimitive(p, x1, x2, x3, dens, eint, v1, v2, v3);
      u0(m,IDN,k,j,i) = dens;
      u0(m,IM1,k,j,i) = dens*v1;
      u0(m,IM2,k,j,i) = dens*v2;
      u0(m,IM3,k,j,i) = dens*v3;
      if (p.is_ideal) u0(m,IEN,k,j,i) = eint + 0.5*dens*(v1*v1 + v2*v2 + v3*v3);
      for (int n=nhydro; n<nhydro+nscalars; ++n) u0(m,n,k,j,i) = 0.0;
    }
  });

  par_for("spherical_flow_bc_x2", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, ng-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int k, int g, int i) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real x3 = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                                size.d_view(m).x3max);
    Real dens, eint = 0.0, v1, v2, v3;
    if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) {
      const int j = g;
      const Real x2 = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                                  size.d_view(m).x2max);
      SphericalPrimitive(p, x1, x2, x3, dens, eint, v1, v2, v3);
      u0(m,IDN,k,j,i) = dens;
      u0(m,IM1,k,j,i) = dens*v1;
      u0(m,IM2,k,j,i) = dens*v2;
      u0(m,IM3,k,j,i) = dens*v3;
      if (p.is_ideal) u0(m,IEN,k,j,i) = eint + 0.5*dens*(v1*v1 + v2*v2 + v3*v3);
      for (int n=nhydro; n<nhydro+nscalars; ++n) u0(m,n,k,j,i) = 0.0;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
      const int j = je + 1 + g;
      const Real x2 = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                                  size.d_view(m).x2max);
      SphericalPrimitive(p, x1, x2, x3, dens, eint, v1, v2, v3);
      u0(m,IDN,k,j,i) = dens;
      u0(m,IM1,k,j,i) = dens*v1;
      u0(m,IM2,k,j,i) = dens*v2;
      u0(m,IM3,k,j,i) = dens*v3;
      if (p.is_ideal) u0(m,IEN,k,j,i) = eint + 0.5*dens*(v1*v1 + v2*v2 + v3*v3);
      for (int n=nhydro; n<nhydro+nscalars; ++n) u0(m,n,k,j,i) = 0.0;
    }
  });

  par_for("spherical_flow_bc_x3", DevExeSpace(), 0, nmb-1, 0, ng-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int g, int j, int i) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real x2 = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                                size.d_view(m).x2max);
    Real dens, eint = 0.0, v1, v2, v3;
    if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user) {
      const int k = g;
      const Real x3 = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                                  size.d_view(m).x3max);
      SphericalPrimitive(p, x1, x2, x3, dens, eint, v1, v2, v3);
      u0(m,IDN,k,j,i) = dens;
      u0(m,IM1,k,j,i) = dens*v1;
      u0(m,IM2,k,j,i) = dens*v2;
      u0(m,IM3,k,j,i) = dens*v3;
      if (p.is_ideal) u0(m,IEN,k,j,i) = eint + 0.5*dens*(v1*v1 + v2*v2 + v3*v3);
      for (int n=nhydro; n<nhydro+nscalars; ++n) u0(m,n,k,j,i) = 0.0;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user) {
      const int k = ke + 1 + g;
      const Real x3 = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                                  size.d_view(m).x3max);
      SphericalPrimitive(p, x1, x2, x3, dens, eint, v1, v2, v3);
      u0(m,IDN,k,j,i) = dens;
      u0(m,IM1,k,j,i) = dens*v1;
      u0(m,IM2,k,j,i) = dens*v2;
      u0(m,IM3,k,j,i) = dens*v3;
      if (p.is_ideal) u0(m,IEN,k,j,i) = eint + 0.5*dens*(v1*v1 + v2*v2 + v3*v3);
      for (int n=nhydro; n<nhydro+nscalars; ++n) u0(m,n,k,j,i) = 0.0;
    }
  });
}

} // namespace

void SphericalFlowErrors(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::NewtonianSphericalFlow()
//! \brief Initialize Newtonian Bondi accretion or an isothermal Parker wind.

void ProblemGenerator::NewtonianSphericalFlow(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *phydro = pmbp->phydro;

  if (phydro == nullptr || pmbp->pmhd != nullptr) {
    Fatal("Newtonian spherical flow requires a <hydro> block and no <mhd> block");
  }
  if (pmbp->pcoord->is_special_relativistic ||
      pmbp->pcoord->is_general_relativistic) {
    Fatal("Newtonian spherical flow cannot be used with relativistic coordinates");
  }
  if (!pmy_mesh_->three_d) {
    Fatal("Newtonian spherical flow requires a 3D Cartesian mesh");
  }
  if (phydro->psrc == nullptr ||
      !phydro->psrc->point_particle_gravity_at_center) {
    Fatal("Newtonian spherical flow requires <hydro_srcterms>/"
          "point_particle_gravity_at_center=true");
  }
  if (!phydro->use_sphere_mask) {
    Fatal("Newtonian spherical flow requires <sphere_mask>/enabled=true");
  }

  const std::string pgen_name = pin->GetString("problem", "pgen_name");
  spherical_flow.is_wind = (pgen_name == "parker_wind");
  if (!spherical_flow.is_wind && pgen_name != "newtonian_bondi") {
    Fatal("NewtonianSphericalFlow received unsupported pgen_name='" + pgen_name + "'");
  }

  auto &eos = phydro->peos->eos_data;
  spherical_flow.is_ideal = eos.is_ideal;
  spherical_flow.mask_radius = phydro->sphere_mask_radius;

  if (spherical_flow.is_wind) {
    if (spherical_flow.is_ideal) {
      Fatal("Parker wind requires <hydro>/eos=isothermal");
    }
    if (phydro->sphere_mask_bc != hydro::Hydro::SphereMaskBC::spherical_wind) {
      Fatal("Parker wind requires <sphere_mask>/bc=spherical_wind");
    }
    spherical_flow.gamma = 1.0;
    spherical_flow.cs_inf = eos.iso_cs;
    spherical_flow.dens_inf = 0.0;
    spherical_flow.pres_inf = 0.0;
    spherical_flow.r_bondi = 1.0/SQR(spherical_flow.cs_inf);
    spherical_flow.sonic_radius = 0.5*spherical_flow.r_bondi;
    spherical_flow.sonic_density =
        pin->GetOrAddReal("problem", "sonic_density", 1.0);
    spherical_flow.lambda = 0.0;
    spherical_flow.r_floor = spherical_flow.mask_radius;
  } else {
    if (phydro->sphere_mask_bc != hydro::Hydro::SphereMaskBC::absorbing) {
      Fatal("Newtonian Bondi accretion requires <sphere_mask>/bc=absorbing");
    }
    spherical_flow.dens_inf = pin->GetOrAddReal("problem", "dens_inf", 1.0);
    if (spherical_flow.is_ideal) {
      spherical_flow.gamma = eos.gamma;
      spherical_flow.cs_inf = pin->GetOrAddReal("problem", "cs_inf", 1.0);
      if (!(spherical_flow.gamma > 1.0)) {
        Fatal("Ideal-gas Newtonian Bondi accretion requires gamma > 1");
      }
    } else {
      spherical_flow.gamma = 1.0;
      spherical_flow.cs_inf = eos.iso_cs;
    }
    spherical_flow.r_bondi = 1.0/SQR(spherical_flow.cs_inf);
    const Real q = 5.0 - 3.0*spherical_flow.gamma;
    if (!(q > 0.0)) {
      Fatal("Newtonian Bondi accretion requires gamma < 5/3");
    }
    spherical_flow.sonic_radius = 0.25*q*spherical_flow.r_bondi;
    spherical_flow.pres_inf = spherical_flow.dens_inf*SQR(spherical_flow.cs_inf)
                              /spherical_flow.gamma;
    if (spherical_flow.is_ideal) {
      spherical_flow.lambda =
          0.25*pow(2.0/q, 0.5*q/(spherical_flow.gamma - 1.0));
      spherical_flow.sonic_density = 0.0;
    } else {
      spherical_flow.lambda = 0.25*exp(1.5);
      spherical_flow.sonic_density = spherical_flow.dens_inf*exp(1.5);
    }
    spherical_flow.r_floor =
        pin->GetOrAddReal("problem", "r_floor", spherical_flow.mask_radius);
  }

  if (!(spherical_flow.cs_inf > 0.0) ||
      !(spherical_flow.sonic_radius > 0.0) ||
      !(spherical_flow.mask_radius < spherical_flow.sonic_radius)) {
    Fatal("Newtonian spherical flow requires positive sound speed and mask radius "
          "strictly inside the sonic radius");
  }
  if (spherical_flow.is_wind && !(spherical_flow.sonic_density > 0.0)) {
    Fatal("Parker wind requires <problem>/sonic_density > 0");
  }
  if (!spherical_flow.is_wind && !(spherical_flow.dens_inf > 0.0)) {
    Fatal("Newtonian Bondi accretion requires <problem>/dens_inf > 0");
  }
  if (!(spherical_flow.r_floor >= spherical_flow.mask_radius)) {
    Fatal("<problem>/r_floor must be at least <sphere_mask>/radius");
  }
  if (!(phydro->psrc->softening_length > 0.0) ||
      phydro->psrc->softening_length > spherical_flow.mask_radius) {
    Fatal("gravity_softening_length must be positive and no larger than the mask radius");
  }

  if (spherical_flow.is_wind) {
    const Real mach = IsothermalMach(spherical_flow.mask_radius,
                                      spherical_flow.sonic_radius, true);
    const Real x = spherical_flow.mask_radius/spherical_flow.sonic_radius;
    spherical_flow.mask_density = spherical_flow.sonic_density/(mach*x*x);
    spherical_flow.mask_velocity = spherical_flow.cs_inf*mach;

    // Hydro parses the required positive placeholders before the pgen runs.  Replace
    // them so the masked state is continuous with the analytic exterior.
    phydro->sm_dens = spherical_flow.mask_density;
    phydro->sm_velr = spherical_flow.mask_velocity;
    pin->SetReal("sphere_mask", "dens", spherical_flow.mask_density);
    pin->SetReal("sphere_mask", "vel_r", spherical_flow.mask_velocity);
  }

  user_bcs_func = FixedSphericalBoundary;
  pgen_final_func = SphericalFlowErrors;
  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const int nhydro = phydro->nhydro;
  const int nscalars = phydro->nscalars;
  auto &size = pmbp->pmb->mb_size;
  auto &w0 = phydro->w0;
  const SphericalFlowData p = spherical_flow;

  par_for("newtonian_spherical_init", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real x2 = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                                size.d_view(m).x2max);
    const Real x3 = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                                size.d_view(m).x3max);
    Real dens, eint = 0.0, v1, v2, v3;
    SphericalPrimitive(p, x1, x2, x3, dens, eint, v1, v2, v3);
    w0(m,IDN,k,j,i) = dens;
    w0(m,IVX,k,j,i) = v1;
    w0(m,IVY,k,j,i) = v2;
    w0(m,IVZ,k,j,i) = v3;
    if (p.is_ideal) w0(m,IEN,k,j,i) = eint;
    for (int n=nhydro; n<nhydro+nscalars; ++n) w0(m,n,k,j,i) = 0.0;
  });

  if (spherical_flow.reset_ic) {
    phydro->peos->PrimToCons(w0, phydro->u1, is, ie, js, je, ks, ke);
  } else {
    phydro->peos->PrimToCons(w0, phydro->u0, is, ie, js, je, ks, ke);
  }

  if (global_variable::my_rank == 0 && !spherical_flow.reset_ic) {
    if (spherical_flow.is_wind) {
      const Real mdot = 4.0*M_PI*SQR(spherical_flow.sonic_radius)
                        *spherical_flow.sonic_density*spherical_flow.cs_inf;
      std::cout << "Isothermal Parker wind (GM=1): r_sonic="
                << spherical_flow.sonic_radius
                << "  rho_sonic=" << spherical_flow.sonic_density
                << "  mdot=" << mdot
                << "  mask_density=" << spherical_flow.mask_density
                << "  mask_velocity=" << spherical_flow.mask_velocity << std::endl;
    } else {
      const Real mdot = 4.0*M_PI*spherical_flow.lambda*spherical_flow.dens_inf
                        *spherical_flow.cs_inf*SQR(spherical_flow.r_bondi);
      std::cout << "Newtonian Bondi accretion (GM=1): r_bondi="
                << spherical_flow.r_bondi
                << "  r_sonic=" << spherical_flow.sonic_radius
                << "  lambda=" << spherical_flow.lambda
                << "  mdot=" << mdot
                << "  r_floor=" << spherical_flow.r_floor << std::endl;
    }
  }
}

// Measure drift from the analytic initial state.  As in other AthenaK steady-solution
// tests, the reference is regenerated into u1 at finalization.
void SphericalFlowErrors(ParameterInput *pin, Mesh *pm) {
  spherical_flow.reset_ic = true;
  pm->pgen->NewtonianSphericalFlow(pin, false);
  pm->pgen->OutputErrors(pin, pm);
}
