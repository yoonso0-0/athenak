//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file collision_debris.cpp
//! \brief Hydrodynamic collision debris accretion in Cartesian Kerr-Schild
//! coordinates. Adapted from fluids/gr_torus.cpp. Units are G = c = M_BH = 1.
//!
//! With <problem>/user_hist=true, write <basename>.user.flux<n>.hst on the
//! outer horizon (flux0) and on each comma-separated flux_radii entry
//! (flux1, ...; default: 12, 24). flux_nlev sets
//! the angular sampling, 10*nlev^2+2 points (default 5); flux_ninterp sets the
//! interpolation points per axis (-1 selects 2*nghost). Headers record the
//! radius, angular sampling, and interpolation stencil for each surface.
//! A reflecting boundary at z=0 permits either equatorial half-domain. Flux
//! histories reconstruct the full sphere and record symmetry_factor=2.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "units/units.hpp"

namespace {
void NoInflowBoundary(Mesh *pm);
void AccretionFluxes(HistoryData *pdata, Mesh *pm);

// +1/-1 samples the upper/lower half-domain through reflection at z=0.
// Only the equatorial plane is a reflection symmetry for a spinning Kerr
// metric.
int equatorial_fold = 0;

[[noreturn]] void Fatal(const std::string &message) {
  std::cerr << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

// Parse a comma-separated list of reals (same convention as bhl_wind_tunnel).
std::vector<Real> ParseRealList(const std::string &name, std::string list) {
  std::replace(list.begin(), list.end(), ',', ' ');
  std::istringstream stream(list);
  std::vector<Real> values;
  while (true) {
    stream >> std::ws;
    if (stream.eof())
      break;
    Real value;
    if (!(stream >> value)) {
      Fatal("<problem>/" + name +
            " must be a comma-separated list of numbers.");
    }
    values.push_back(value);
  }
  if (values.empty()) {
    Fatal("<problem>/" + name + " must be a comma-separated list of numbers.");
  }
  return values;
}

// The outer horizon is always the first surface (flux0); flux_radii lists the
// additional radii.
std::vector<Real> ParseFluxRadii(ParameterInput *pin, const Real r_horizon) {
  std::vector<Real> radii{r_horizon};
  const auto extra = ParseRealList(
      "flux_radii", pin->GetOrAddString("problem", "flux_radii", "12, 24"));
  radii.insert(radii.end(), extra.begin(), extra.end());
  return radii;
}

// Collision parameters, converted to code units (G = c = M_BH = 1) except
// smbh_mass, which stays in solar masses as the physical scale.
struct DebrisParameters {
  Real smbh_mass;     // M_BH / M_sun
  Real t_minus_tcoll; // t - t_coll in GM_BH/c^3 (input in days)
  Real t_days;        // t - t_coll in days (as input)
  Real x_coll[3];     // collision site, Cartesian KS x, y, z in GM_BH/c^2
  Real v_coll[3];     // Cartesian KS components in units of c (input in km/s)
  Real vr_peak; // peak debris radial velocity in units of c (input in km/s)
  Real density_scale; // physical density [g/cm^3] of code density 1
  // Total debris mass in code units, density_scale * r_g^3, so that the
  // integral of code rho over code volume equals it (input in M_sun).
  Real m_total;
  Real mu; // mean molecular weight
  // Inputs as given, for the startup report.
  Real v_coll_kms[3], vr_peak_kms, m_total_msun;
  // Code gas pressure per code density per kelvin, k_B / (mu m_u c^2):
  // p_code = rho_code * T[K] * kelvin_to_code (density_scale cancels).
  Real kelvin_to_code;
};

DebrisParameters ReadDebrisParameters(ParameterInput *pin) {
  using units::Units;
  DebrisParameters debris{};
  debris.smbh_mass = pin->GetReal("problem", "smbh_mass");
  const Real t_days = pin->GetReal("problem", "t_minus_tcoll");
  const auto x_coll =
      ParseRealList("x_coll", pin->GetString("problem", "x_coll"));
  const auto v_kms =
      ParseRealList("v_coll", pin->GetString("problem", "v_coll"));
  if (!std::isfinite(debris.smbh_mass) || debris.smbh_mass <= 0.0 ||
      !std::isfinite(t_days) || t_days <= 0.0) {
    Fatal("<problem>/smbh_mass and t_minus_tcoll must be positive and "
          "finite.");
  }
  debris.t_days = t_days;
  if (x_coll.size() != 3) {
    Fatal("<problem>/x_coll must have exactly three components (r_g).");
  }
  for (int d = 0; d < 3; ++d) {
    if (!std::isfinite(x_coll[d]))
      Fatal("<problem>/x_coll must be finite.");
    debris.x_coll[d] = x_coll[d];
  }
  if (v_kms.size() != 3) {
    Fatal("<problem>/v_coll must have exactly three components (km/s).");
  }

  constexpr Real seconds_per_day = 86400.0;
  const Real c_kms = Units::speed_of_light_cgs / Units::km_s_cgs;
  const Real t_g = Units::grav_constant_cgs * debris.smbh_mass *
                   Units::msun_cgs / std::pow(Units::speed_of_light_cgs, 3);
  debris.t_minus_tcoll = t_days * seconds_per_day / t_g;
  Real v2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    debris.v_coll_kms[d] = v_kms[d];
    debris.v_coll[d] = v_kms[d] / c_kms;
    v2 += SQR(debris.v_coll[d]);
  }
  if (!std::isfinite(v2) || v2 >= 1.0) {
    Fatal("<problem>/v_coll must be finite and subluminal.");
  }
  debris.vr_peak_kms = pin->GetReal("problem", "vr_peak");
  debris.vr_peak = debris.vr_peak_kms / c_kms;
  if (!std::isfinite(debris.vr_peak) || debris.vr_peak <= 0.0 ||
      debris.vr_peak >= 1.0) {
    Fatal("<problem>/vr_peak must be positive, finite, and subluminal (km/s).");
  }
  debris.density_scale = pin->GetReal("problem", "density_scale");
  if (!std::isfinite(debris.density_scale) || debris.density_scale <= 0.0) {
    Fatal("<problem>/density_scale must be positive and finite (g/cm^3).");
  }
  const Real m_total_msun = pin->GetReal("problem", "M_total");
  debris.m_total_msun = m_total_msun;
  const Real r_g = t_g * Units::speed_of_light_cgs; // cm
  debris.m_total =
      m_total_msun * Units::msun_cgs / (debris.density_scale * r_g * r_g * r_g);
  if (!std::isfinite(debris.m_total) || debris.m_total <= 0.0) {
    Fatal("<problem>/M_total must be positive and finite (M_sun).");
  }
  debris.mu = pin->GetOrAddReal("problem", "mu", 0.6);
  if (!std::isfinite(debris.mu) || debris.mu <= 0.0) {
    Fatal("<problem>/mu must be positive and finite.");
  }
  debris.kelvin_to_code =
      Units::k_boltzmann_cgs / (debris.mu * Units::atomic_mass_unit_cgs *
                                SQR(Units::speed_of_light_cgs));
  return debris;
}

// Debris temperature [K] at distance r from the collision site. For now the
// profile is flat at t_debris = 42500 K * (t_minus_tcoll / 1 day)^-0.7.
KOKKOS_INLINE_FUNCTION
Real DebrisTemperature(const Real r, const Real r_break, const Real t_debris) {
  return t_debris;
}

// Alternative (currently unused): log-normal shell peaked at 2.2 r_break with
// peak t_debris, and at least 3e4 K inside 2 r_break. r = 0 gives
// log10 -> -inf and T -> 0, then the inner minimum.
KOKKOS_INLINE_FUNCTION
Real DebrisTemperatureLogNormal(const Real r, const Real r_break,
                                const Real t_debris) {
  Real temp_k = t_debris * exp(-12.5 * SQR(log10(r / (2.2 * r_break))));
  if (r < 2.0 * r_break)
    temp_k = fmax(temp_k, 3.0e4);
  return temp_k;
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Set up fixed Kerr hydro and initialize the floor background.
//! Compile with -D PROBLEM=hydro/collision_debris.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->pmhd != nullptr ||
      pmbp->prad != nullptr) {
    std::cerr
        << "### FATAL ERROR in " << __FILE__ << std::endl
        << "collision_debris requires hydrodynamics without MHD or radiation."
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!pmbp->pcoord->is_general_relativistic) {
    std::cerr << "### FATAL ERROR in " << __FILE__ << std::endl
              << "collision_debris requires general_rel = true in <coord> "
              << "and a fixed spacetime." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const auto coord = pmbp->pcoord->coord_data;
  if (coord.is_minkowski || !std::isfinite(coord.bh_spin) ||
      std::abs(coord.bh_spin) > 1.0) {
    std::cerr << "### FATAL ERROR in " << __FILE__ << std::endl
              << "collision_debris requires a Kerr metric with |a| <= 1."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *phydro = pmbp->phydro;
  const auto eos = phydro->peos->eos_data;
  if (!eos.is_ideal || !std::isfinite(eos.gamma) || eos.gamma <= 1.0) {
    std::cerr << "### FATAL ERROR in " << __FILE__ << std::endl
              << "collision_debris requires an ideal gas with gamma > 1."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Re-enroll callbacks and reconstruct diagnostic grids on restarts as well.
  user_bcs_func = NoInflowBoundary;
  user_hist_func = AccretionFluxes;
  if (user_hist) {
    if (!pmy_mesh_->three_d) {
      Fatal("<problem>/user_hist requires a 3D mesh for spherical flux "
            "integrals.");
    }
    const int nlev = pin->GetOrAddInteger("problem", "flux_nlev", 5);
    if (nlev < 1)
      Fatal("<problem>/flux_nlev must be at least 1.");
    const int ninterp = pin->GetOrAddInteger("problem", "flux_ninterp", -1);
    if ((ninterp < 1 && ninterp != -1) ||
        ninterp > 2 * pmy_mesh_->mb_indcs.ng + 1) {
      Fatal("<problem>/flux_ninterp must be -1 (2*nghost) or lie between "
            "1 and 2*nghost+1.");
    }
    const Real r_horizon = 1.0 + std::sqrt(1.0 - SQR(coord.bh_spin));
    const auto radii = ParseFluxRadii(pin, r_horizon);
    const auto &size = pmy_mesh_->mesh_size;
    const bool reflect_lower =
        size.x3min == 0.0 &&
        pmy_mesh_->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::reflect;
    const bool reflect_upper =
        size.x3max == 0.0 &&
        pmy_mesh_->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::reflect;
    equatorial_fold = reflect_lower ? 1 : (reflect_upper ? -1 : 0);
    for (std::size_t n = 0; n < radii.size(); ++n) {
      const Real radius = radii[n];
      // A constant-r Kerr-Schild surface extends to sqrt(r^2+a^2) in x/y.
      const Real equatorial_radius = std::hypot(radius, coord.bh_spin);
      if (!std::isfinite(radius) || radius <= 0.0 ||
          equatorial_radius >=
              std::min({-size.x1min, size.x1max, -size.x2min, size.x2max}) ||
          (!reflect_lower && radius >= -size.x3min) ||
          (!reflect_upper && radius >= size.x3max)) {
        Fatal("Each <problem>/flux_radii entry must be finite, positive, and "
              "its Kerr-Schild surface must fit inside the mesh, allowing "
              "reflection through an equatorial boundary at z=0.");
      }
      spherical_grids.push_back(
          std::make_unique<SphericalGrid>(pmbp, nlev, radius, ninterp));
      if (equatorial_fold != 0) {
        const int fold[3] = {0, 0, equatorial_fold};
        spherical_grids.back()->FoldInterpolationCoordinates(fold);
      }
      user_hist_tags.push_back("flux" + std::to_string(n));
    }
  }
  if (restart)
    return;

  const DebrisParameters debris = ReadDebrisParameters(pin);
  // Debris length scales from free expansion since the collision [r_g].
  const Real r_peak = debris.vr_peak * debris.t_minus_tcoll;
  const Real r_break = 0.1 * debris.vr_peak * debris.t_minus_tcoll;
  // Uniform core density inside r_break, 0.18 * M_total / (4 pi
  // r_break^3 / 3), in code units (physical = rho_core * density_scale).
  const Real rho_core =
      0.18 * 3.0 * debris.m_total / (4.0 * M_PI * r_break * r_break * r_break);
  // Debris temperature [K] (DebrisTemperature): flat for now at
  // T_shell = 42500 K * (t_minus_tcoll / 1 day)^-0.7.
  const Real t_shell = 42500.0 * std::pow(debris.t_days, -0.7);

  // Distance from x_coll where the r^-8 density tail reaches dfloor (0 if
  // the core itself is at or below dfloor). The debris must stay above the
  // density floor out to r_peak, the edge of the homologous flow.
  const Real r_dfloor = (rho_core > eos.dfloor)
                            ? r_break * std::pow(rho_core / eos.dfloor, 0.125)
                            : 0.0;
  const bool dfloor_inside_peak = r_dfloor < r_peak;

  // Startup report (rank 0): inputs, derived debris scales, and the debris
  // temperature where the density reaches dfloor next to the temperature the
  // <hydro> floors imply there, T_floor = (pfloor / dfloor) / kelvin_to_code.
  if (global_variable::my_rank == 0) {
    const Real t_floor = eos.pfloor / (eos.dfloor * debris.kelvin_to_code);
    std::ostringstream msg;
    msg << std::scientific << std::setprecision(4);
    // One "label  value  unit" row with aligned columns.
    auto row = [&msg](const std::string &label, const Real value,
                      const std::string &unit) {
      msg << "  " << std::left << std::setw(34) << label << std::right
          << std::setw(12) << value;
      if (!unit.empty())
        msg << "  " << unit;
      msg << std::endl;
    };
    auto vec_row = [&msg](const std::string &label, const Real v[3],
                          const std::string &unit) {
      msg << "  " << std::left << std::setw(34) << label << std::right
          << std::setw(12) << v[0] << std::setw(12) << v[1] << std::setw(12)
          << v[2] << "  " << unit << std::endl;
    };
    auto section = [&msg](const std::string &title) {
      msg << std::endl << " " << title << std::endl;
    };
    const std::string rule(76, '-');
    msg << std::endl
        << rule << std::endl
        << " collision_debris: initial debris setup" << std::endl
        << rule << std::endl;
    section("Input parameters <problem>:");
    row("smbh_mass", debris.smbh_mass, "M_sun");
    row("t_minus_tcoll", debris.t_days, "days");
    vec_row("x_coll (x, y, z)", debris.x_coll, "r_g");
    vec_row("v_coll (x, y, z)", debris.v_coll_kms, "km/s");
    row("vr_peak", debris.vr_peak_kms, "km/s");
    row("M_total", debris.m_total_msun, "M_sun");
    row("density_scale", debris.density_scale, "g/cm^3");
    row("mu", debris.mu, "");
    section("Derived debris scales (code units):");
    row("t_minus_tcoll", debris.t_minus_tcoll, "t_g");
    row("r_peak = vr_peak * t_minus_tcoll", r_peak, "r_g");
    row("r_break = 0.1 * r_peak", r_break, "r_g");
    row("rho_core", rho_core, "");
    row("M_total", debris.m_total, "density_scale * r_g^3");
    row("T_shell (flat debris temperature)", t_shell, "K");
    section("<hydro> floors:");
    row("dfloor", eos.dfloor, "");
    row("pfloor", eos.pfloor, "");
    if (rho_core > eos.dfloor) {
      const Real t_debris = DebrisTemperature(r_dfloor, r_break, t_shell);
      section("Where the debris density falls to dfloor:");
      row("distance from x_coll, r_dfloor", r_dfloor, "r_g");
      row("  in units of r_break", r_dfloor / r_break, "");
      row("  in units of r_peak", r_dfloor / r_peak, "");
      row("debris temperature T_debris", t_debris, "K");
      row("floor temperature T_floor", t_floor, "K  (pfloor/dfloor)");
      row("T_floor / T_debris", t_floor / t_debris, "");
    } else {
      section("rho_core <= dfloor: the debris lies entirely on the density "
              "floor.");
      row("floor temperature T_floor", t_floor, "K  (pfloor/dfloor)");
    }
    msg << std::fixed << std::setprecision(2)
        << "  (T = p mu m_u / (rho k_B), mu = " << debris.mu << ")" << std::endl
        << rule << std::endl
        << std::endl;
    std::cout << msg.str();
    if (dfloor_inside_peak) {
      std::cerr << std::scientific << std::setprecision(4)
                << "### FATAL ERROR in " << __FILE__ << std::endl
                << "The debris density reaches dfloor at r_dfloor = "
                << r_dfloor << " r_g from x_coll, inside r_peak = " << r_peak
                << " r_g." << std::endl
                << "Lower <hydro>/dfloor or <problem>/density_scale so that "
                << "r_dfloor >= r_peak." << std::endl;
    }
  }
  if (dfloor_inside_peak) {
    // Let rank 0 finish printing before any rank exits, otherwise the MPI
    // launcher can kill it first and the report and error are lost.
    std::cout << std::flush;
    std::cerr << std::flush;
#if MPI_PARALLEL_ENABLED
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    std::exit(EXIT_FAILURE);
  }

  const auto indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nhydro = phydro->nhydro;
  const int nvars = nhydro + phydro->nscalars;
  const Real gm1 = eos.gamma - 1.0;
  // Internal-energy-density floor, as in reconstruct/recon.hpp.
  const Real efloor = eos.pfloor / gm1;
  const auto &size = pmbp->pmb->mb_size;
  const auto &excised = pmbp->pcoord->excision_floor;
  auto &w0 = phydro->w0;

  par_for(
      "collision_debris_init", DevExeSpace(), 0, pmbp->nmb_thispack - 1, ks, ke,
      js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real x = CellCenterX(i - is, indcs.nx1, size.d_view(m).x1min,
                                   size.d_view(m).x1max);
        const Real y = CellCenterX(j - js, indcs.nx2, size.d_view(m).x2min,
                                   size.d_view(m).x2max);
        const Real z = CellCenterX(k - ks, indcs.nx3, size.d_view(m).x3min,
                                   size.d_view(m).x3max);
        // Static background: <coord> excision values inside the excised
        // region (as in gr_torus), <hydro> floors elsewhere. eint is the
        // internal energy density (primitive IEN).
        Real rho, eint;
        const bool is_excised = coord.bh_excise && excised(m, k, j, i);
        if (is_excised) {
          rho = coord.dexcise;
          eint = coord.pexcise / gm1;
        } else {
          rho = eos.dfloor;
          eint = efloor;
        }
        Real uu1 = 0.0, uu2 = 0.0, uu3 = 0.0;

        // Debris velocity: center-of-mass velocity v_coll everywhere outside
        // excision, plus homologous radial expansion from the collision site,
        // v^r = vr_peak * r / r_peak for r < r_peak, with r the coordinate
        // distance from x_coll, i.e. vr_peak * (x^i - x_coll^i) / r_peak.
        // The two add linearly (both << c). v^i is the normal-observer
        // 3-velocity; the primitive is u^i = W v^i with
        // W = 1/sqrt(1 - gamma_ij v^i v^j).
        // Debris density: uniform rho_core for r < r_break, and
        // rho_core * (r/r_break)^-8 outside, never below the <hydro> floor.
        // Debris pressure: ideal gas at the debris temperature above,
        // pgas = rho * T_K * kelvin_to_code, with eint = pgas/(gamma-1)
        // floored at efloor = pfloor/(gamma-1).
        const Real dx[3] = {x - debris.x_coll[0], y - debris.x_coll[1],
                            z - debris.x_coll[2]};
        const Real r = sqrt(SQR(dx[0]) + SQR(dx[1]) + SQR(dx[2]));
        if (!is_excised) {
          if (r < r_break) {
            rho = rho_core;
          } else {
            const Real s2 = SQR(r_break / r);
            rho = fmax(rho, rho_core * SQR(SQR(s2)));
          }
          const Real temp_k = DebrisTemperature(r, r_break, t_shell);
          eint = fmax(efloor, rho * temp_k * debris.kelvin_to_code / gm1);
          Real v[3];
          for (int d = 0; d < 3; ++d) {
            v[d] = debris.v_coll[d];
            if (r < r_peak)
              v[d] += debris.vr_peak * dx[d] / r_peak;
          }
          Real glower[4][4], gupper[4][4];
          ComputeMetricAndInverse(x, y, z, coord.is_minkowski, coord.bh_spin,
                                  glower, gupper);
          Real v2 = 0.0;
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
              v2 += glower[a + 1][b + 1] * v[a] * v[b];
          const Real lorentz = 1.0 / sqrt(1.0 - v2);
          uu1 = lorentz * v[0];
          uu2 = lorentz * v[1];
          uu3 = lorentz * v[2];
        }

        w0(m, IDN, k, j, i) = rho;
        w0(m, IEN, k, j, i) = eint;
        w0(m, IVX, k, j, i) = uu1;
        w0(m, IVY, k, j, i) = uu2;
        w0(m, IVZ, k, j, i) = uu3;
        for (int n = nhydro; n < nvars; ++n) {
          w0(m, n, k, j, i) = 0.0;
        }
      });

  phydro->peos->PrimToCons(w0, phydro->u0, is, ie, js, je, ks, ke);
}

namespace {

//----------------------------------------------------------------------------------------
//! \brief Apply the gr_torus diode boundary to primitives, then rebuild
//! conserved ghosts. This clips the normal-frame velocity, following gr_torus's
//! boundary convention.

void NoInflowBoundary(Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto *phydro = pmbp->phydro;
  const auto indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int nc[3] = {indcs.nx1 + 2 * ng,
                     (indcs.nx2 > 1) ? indcs.nx2 + 2 * ng : 1,
                     (indcs.nx3 > 1) ? indcs.nx3 + 2 * ng : 1};
  const int start[3] = {indcs.is, indcs.js, indcs.ks};
  const int end[3] = {indcs.ie, indcs.je, indcs.ke};
  const auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &u0 = phydro->u0;
  auto &w0 = phydro->w0;
  const int nvars = phydro->nhydro + phydro->nscalars;

  // Sweep faces in x1, x2, x3 order, including transverse ghosts to fill
  // corners.
  for (int axis = 0; axis < 3; ++axis) {
    if (axis > 0 && nc[axis] == 1)
      continue;
    for (int side = 0; side < 2; ++side) {
      const int face = 2 * axis + side;
      if (pm->mesh_bcs[face] != BoundaryFlag::user)
        continue;
      const int edge = (side == 0) ? start[axis] : end[axis];
      const int direction = (side == 0) ? -1 : 1;
      int lo[3] = {0, 0, 0};
      int hi[3] = {nc[0] - 1, nc[1] - 1, nc[2] - 1};
      lo[axis] = (side == 0) ? edge - ng : edge;
      hi[axis] = (side == 0) ? edge : edge + ng;

      // Refresh primitives in the boundary cell and ghost slab on every block,
      // as in gr_torus. Only blocks on a user face have their primitives
      // replaced.
      phydro->peos->ConsToPrim(u0, w0, false, lo[0], hi[0], lo[1], hi[1], lo[2],
                               hi[2]);
      const int na = (axis == 0) ? nc[1] : nc[0];
      const int nb = (axis == 2) ? nc[1] : nc[2];
      par_for(
          "collision_debris_boundary", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
          0, nvars - 1, 0, nb - 1, 0, na - 1,
          KOKKOS_LAMBDA(int m, int n, int b, int a) {
            if (mb_bcs.d_view(m, face) != BoundaryFlag::user)
              return;
            int src[3];
            src[0] = (axis == 0) ? edge : a;
            src[1] = (axis == 0) ? a : ((axis == 1) ? edge : b);
            src[2] = (axis == 2) ? edge : b;
            Real value = w0(m, n, src[2], src[1], src[0]);
            if (n == IVX + axis) {
              value = (side == 0) ? fmin(0.0, value) : fmax(0.0, value);
            }
            for (int g = 1; g <= ng; ++g) {
              int dst[3] = {src[0], src[1], src[2]};
              dst[axis] += direction * g;
              w0(m, n, dst[2], dst[1], dst[0]) = value;
            }
          });

      // Exclude the active boundary cell when converting back to conserved
      // form.
      lo[axis] = (side == 0) ? edge - ng : edge + 1;
      hi[axis] = (side == 0) ? edge - 1 : edge + ng;
      phydro->peos->PrimToCons(w0, u0, lo[0], hi[0], lo[1], hi[1], lo[2],
                               hi[2]);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Integrate hydro fluxes on constant spherical Kerr-Schild radius
//! surfaces. Signs match gr_torus: mdot = -int rho u^r dA, edot = -int T^r_t
//! dA, ldot = int T^r_phi dA. Energy includes rest mass.
//! Folded grids still integrate every full-sphere angle; no extra factor of
//! two.

void AccretionFluxes(HistoryData *pdata, Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto *phydro = pmbp->phydro;
  const Real spin = pmbp->pcoord->coord_data.bh_spin;
  const Real gamma = phydro->peos->eos_data.gamma;
  auto &grid = *pm->pgen->spherical_grids[pdata->user_index];
  constexpr int nflux = 3;
  static_assert(nflux <= NHISTORY_VARIABLES, "Too many accretion flux columns");
  pdata->nhist = nflux;
  pdata->label[0] = "mdot";
  pdata->label[1] = "edot";
  pdata->label[2] = "ldot";
  pdata->symmetry_factor = (equatorial_fold == 0) ? 1 : 2;
  pdata->header_params = {{"radius", grid.radius},
                          {"nangles", static_cast<Real>(grid.nangles)},
                          {"flux_ninterp", static_cast<Real>(grid.ninterp)}};
  for (int n = 0; n < NHISTORY_VARIABLES; ++n)
    pdata->hdata[n] = 0.0;
  grid.InterpolateToSphere(phydro->nhydro, phydro->w0);

  for (int n = 0; n < grid.nangles; ++n) {
    const Real r = grid.radius;
    const Real theta = grid.polar_pos.h_view(n, 0);
    const Real x = grid.interp_coord.h_view(n, 0);
    const Real y = grid.interp_coord.h_view(n, 1);
    // Interpolation samples the mirror point; evaluate the metric and flux at
    // the original point and restore the odd vertical four-velocity component.
    const bool mirrored = equatorial_fold * std::cos(theta) < 0.0;
    const Real z_sample = grid.interp_coord.h_view(n, 2);
    const Real z = mirrored ? -z_sample : z_sample;
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x, y, z, false, spin, glower, gupper);

    const Real rho = grid.interp_vals.h_view(n, IDN);
    const Real eint = grid.interp_vals.h_view(n, IEN);
    const Real uu[3] = {
        grid.interp_vals.h_view(n, IVX), grid.interp_vals.h_view(n, IVY),
        (mirrored ? -1.0 : 1.0) * grid.interp_vals.h_view(n, IVZ)};
    Real q = 0.0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j)
        q += glower[i + 1][j + 1] * uu[i] * uu[j];
    }
    const Real alpha = std::sqrt(-1.0 / gupper[0][0]);
    const Real lorentz = std::sqrt(1.0 + q);
    Real u[4] = {lorentz / alpha, 0.0, 0.0, 0.0};
    for (int i = 1; i < 4; ++i)
      u[i] = uu[i - 1] - alpha * lorentz * gupper[0][i];
    Real ucov[4] = {};
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j)
        ucov[i] += glower[i][j] * u[j];
    }

    const Real denom = 2.0 * SQR(r) - (SQR(x) + SQR(y) + SQR(z)) + SQR(spin);
    const Real ur =
        (r * x * u[1] + r * y * u[2] + (r + SQR(spin) / r) * z * u[3]) / denom;
    const Real u_phi = -y * ucov[1] + x * ucov[2];
    const Real area =
        (SQR(r) + SQR(spin * std::cos(theta))) * grid.solid_angles.h_view(n);
    const Real enthalpy = rho + gamma * eint;
    pdata->hdata[0] -= rho * ur * area;
    pdata->hdata[1] -= enthalpy * ur * ucov[0] * area;
    pdata->hdata[2] += enthalpy * ur * u_phi * area;
  }
}

} // namespace
