//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file boosteed_bh.cpp
//! \brief Problem generator to initialize rotational equilibrium tori in GR, using either
//! Fishbone-Moncrief (1976) or Chakrabarti (1985) ICs, specialized for cartesian
//! Kerr-Schild coordinates.  Based on gr_torus.cpp in Athena++, with edits by CJW and SR.
//! Simplified and implemented in Kokkos by JMS.

#include <stdio.h>
#include <math.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm> // max(), max_element(), min(), min_element()
#include <iomanip>
#include <iostream> // endl
#include <limits>   // numeric_limits::max()
#include <memory>
#include <sstream> // stringstream
#include <string>  // c_str(), string
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "src/srcterms/turb_driver.hpp"

#include "kokkos/core/src/Kokkos_Timer.hpp"

#include <iostream>

#include <Kokkos_Random.hpp>

// prototypes for functions used internally to this pgen
namespace
{

  KOKKOS_INLINE_FUNCTION
  static void GetBoyerLindquistCoordinates(struct bhl_pgen pgen,
                                           Real x1, Real x2, Real x3,
                                           Real *pr, Real *ptheta, Real *pphi);

  KOKKOS_INLINE_FUNCTION
  Real A1(struct bhl_pgen pgen, Real x1, Real x2, Real x3);
  KOKKOS_INLINE_FUNCTION
  Real A2(struct bhl_pgen pgen, Real x1, Real x2, Real x3);
  KOKKOS_INLINE_FUNCTION
  Real A3(struct bhl_pgen pgen, Real x1, Real x2, Real x3);

  // Useful container for physical parameters
  struct bhl_pgen
  {
    Real spin;             // black hole spin
    Real dexcise, pexcise; // excision parameters

    Real gamma_adi; // EOS parameters
    // Real rho_min, rho_pow, pgas_min, pgas_pow; // background parameters

    // BHL-accretion related parameters
    Real rho_inf; // Incoming wind density
    Real mach;    // Incoming wind Mach number
    Real Ra;      // Accretion radius (sets wind velocity)
    Real v_inf;   // Incoming wind speed; computed from `Ra` (accretion radius)
    Real cs_inf;  // Incoming wind sound speed; computed from `v_inf` and `mach`

    // Controlling the B field configuration
    Real sigma_inf; // Incoming wind magnetization
    Real magnetic_field_angle_yz;
    bool sigma_variation_initial;
    bool sigma_variation_inject_from_boundary;
    Real dB_over_B_mag_inf;

    // Temporary quantities required for primitive variable calculations
    Real e_inf;              // e = rho epsilon
    Real rho_times_h_inf;    // rho h
    Real pressure_inf;       // pressure
    Real W_inf;              // u^0 = W
    Real u1_prim_inf;        // u^1 = W v^1
    Real comoving_b2;        // b^2
    Real B_mag_inf;          // mag|B^i|
    Real total_pressure_inf; // p_gas + b^2/2

    Real arad; // radiation constant? -> ignore

    // Radius mask for computing gravitational drag
    Real grav_drag_mask_rmax;
  };

  static bhl_pgen bhl_accretion;

} // namespace

// Prototypes for user-defined BCs and history functions
void BhlAccretionBoundary(Mesh *pm);
void BhlAccretionHistory(HistoryData *pdata, Mesh *pm);
// @YK : "history functions" seems to be what we call 'reduction quantities' in spectre (e.g. time series)

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for either Fishbone-Moncrief or Chakrabarti torus in GR
//! Compile with '-D PROBLEM=gr_torus' to enroll as user-specific problem generator
//!  assumes x3 is axisymmetric direction
//
// @YK : update the docs above.
//       this seems like doing an actual initialization for domain volume?
//

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart)
{
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic)
  {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR torus problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // User boundary function
  // @YK : setting up boundary condition
  user_bcs_func = BhlAccretionBoundary;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;

  // Extract BH parameters
  bhl_accretion.spin = coord.bh_spin;
  const Real r_excise = coord.rexcise;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);

  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  const Real rflux =
      (is_radiation_enabled) ? ceil(r_excise + 1.0) : 1.0 + sqrt(1.0 - SQR(bhl_accretion.spin));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, rflux));
  // NOTE(@pdmullen): Enroll additional radii for flux analysis by
  // pushing back the grids vector with additional SphericalGrid instances
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 2.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 3.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 4.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 5.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 10.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 20.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 50.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 100.0));
  user_hist_func = BhlAccretionHistory;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_; // @YK : u0 is evolved vars, w0 is prim vars
  if (pmbp->phydro != nullptr)
  {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
  }
  else if (pmbp->pmhd != nullptr)
  {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
  }

  // Extract radiation parameters if enabled
  int nangles_;
  DualArray2D<Real> nh_c_;
  DvceArray6D<Real> norm_to_tet_, tet_c_, tetcov_c_;
  DvceArray5D<Real> i0_;
  if (is_radiation_enabled)
  {
    nangles_ = pmbp->prad->prgeo->nangles;
    nh_c_ = pmbp->prad->nh_c;
    norm_to_tet_ = pmbp->prad->norm_to_tet;
    tet_c_ = pmbp->prad->tet_c;
    tetcov_c_ = pmbp->prad->tetcov_c;
    i0_ = pmbp->prad->i0;
  }

  // Get ideal gas EOS data
  if (pmbp->phydro != nullptr)
  {
    bhl_accretion.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  }
  else if (pmbp->pmhd != nullptr)
  {
    bhl_accretion.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  }
  Real gm1 = bhl_accretion.gamma_adi - 1.0;

  // Get Radiation constant (if radiation enabled)
  if (pmbp->prad != nullptr)
  {
    bhl_accretion.arad = pmbp->prad->arad;
  }

  // Read problem-specific parameters from input file global parameters
  bhl_accretion.mach = pin->GetReal("problem", "mach");
  bhl_accretion.Ra = pin->GetReal("problem", "Ra");
  bhl_accretion.rho_inf = pin->GetReal("problem", "rho_inf");

  bhl_accretion.v_inf = 1. / sqrt(0.5 * bhl_accretion.Ra);
  bhl_accretion.cs_inf = bhl_accretion.v_inf / bhl_accretion.mach;

  // excision parameters
  bhl_accretion.dexcise = coord.dexcise;
  bhl_accretion.pexcise = coord.pexcise;

  // Magnetic field - related variables
  bhl_accretion.sigma_inf = pin->GetReal("problem", "sigma_inf");
  bhl_accretion.magnetic_field_angle_yz = pin->GetReal("problem", "magnetic_field_angle_yz");
  bhl_accretion.sigma_variation_initial = pin->GetBoolean("turb_driving", "sigma_variation_initial");
  bhl_accretion.sigma_variation_inject_from_boundary = pin->GetBoolean("turb_driving", "sigma_variation_inject_from_boundary");
  bhl_accretion.dB_over_B_mag_inf = pin->GetReal("turb_driving", "dB_over_B_mag_inf");

  // Gravitational drag - Mask
  bhl_accretion.grav_drag_mask_rmax = pin->GetReal("problem", "grav_drag_mask_rmax");

  //  ---------------------------------------
  //    Compute auxiliary variables
  //  ---------------------------------------

  bhl_accretion.e_inf = bhl_accretion.rho_inf * SQR(bhl_accretion.cs_inf) / (bhl_accretion.gamma_adi - 1. - SQR(bhl_accretion.cs_inf)) / bhl_accretion.gamma_adi;
  bhl_accretion.rho_times_h_inf = bhl_accretion.rho_inf + bhl_accretion.gamma_adi * bhl_accretion.e_inf;
  bhl_accretion.pressure_inf = bhl_accretion.e_inf * (bhl_accretion.gamma_adi - 1.0);
  bhl_accretion.W_inf = 1.0 / sqrt(1.0 - SQR(bhl_accretion.v_inf));
  bhl_accretion.u1_prim_inf = bhl_accretion.W_inf * bhl_accretion.v_inf;
  bhl_accretion.comoving_b2 = bhl_accretion.sigma_inf * bhl_accretion.rho_times_h_inf;
  bhl_accretion.B_mag_inf = bhl_accretion.W_inf * sqrt(bhl_accretion.comoving_b2);
  bhl_accretion.total_pressure_inf = bhl_accretion.pressure_inf + 0.5 * bhl_accretion.comoving_b2;

  if (global_variable::my_rank == 0)
  {
    std::cout << " **** Initializing BHL variables **** " << std::endl;
    std::cout << "  v_inf        = " << bhl_accretion.v_inf << std::endl;
    std::cout << "  rho_inf      = " << bhl_accretion.rho_inf << std::endl;
    std::cout << "  cs_inf       = " << bhl_accretion.cs_inf << std::endl;
    std::cout << "  e_inf        = " << bhl_accretion.e_inf << std::endl;
    std::cout << "  pressure_inf = " << bhl_accretion.pressure_inf << std::endl;
    std::cout << "" << std::endl;
    std::cout << "  B_mag_inf    = " << bhl_accretion.B_mag_inf << std::endl;
    std::cout << "  Sigma_inf    = " << bhl_accretion.sigma_inf << std::endl;
    std::cout << "  Beta_inf     = " << bhl_accretion.pressure_inf / (0.5 * bhl_accretion.comoving_b2) << std::endl;
    std::cout << "  Initial B variation  = " << bhl_accretion.sigma_variation_initial << std::endl;
    std::cout << "  Inject B variation from boundary  = " << bhl_accretion.sigma_variation_inject_from_boundary << std::endl;
  }

  // return if restart
  if (restart)
    return;

  auto bhl = bhl_accretion;
  auto &size = pmbp->pmb->mb_size;

  const auto incoming_By = bhl.B_mag_inf * sin(bhl.magnetic_field_angle_yz);
  const auto incoming_Bz = bhl.B_mag_inf * cos(bhl.magnetic_field_angle_yz);

  //  ---------------------------------------
  //      initialize hydro primitive variables
  //  ---------------------------------------

  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  Real ptotmax = std::numeric_limits<float>::min();
  const int nmkji = (pmbp->nmb_thispack) * indcs.nx3 * indcs.nx2 * indcs.nx1;
  const int nkji = indcs.nx3 * indcs.nx2 * indcs.nx1;
  const int nji = indcs.nx2 * indcs.nx1;

  // auto &bcc_ = pmbp->pmhd->bcc0;

  if (global_variable::my_rank == 0)
  {
    std::cout << " **** Initializing BHL hydro profile **** " << std::endl;
  }

  Kokkos::parallel_reduce(
      "initial_hydro_profile", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &max_ptot) {
        //
        // @YK : memo on index notations
        //
        // `m` : index of a MeshBlock.
        // `nkji` : value of (Nx x Ny x Nz) in a MeshBlock
        // `nmkji' : `nkji` times number of MeshBlocks in the current MeshBlockPack
        //
        // `idx` : Looks like a single integer index for a whole (giant) array of
        //         quantities stored in this meshblockpack..
        //

        // compute m,k,j,i indices of thread and call function
        int m = (idx) / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / indcs.nx1;
        int i = (idx - m * nkji - k * nji - j * indcs.nx1) + is;
        k += ks;
        j += js;

        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j - js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

        // Extract metric and inverse
        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                                glower, gupper);

        // Calculate Boyer-Lindquist coordinates of cell
        Real r, theta, phi;
        GetBoyerLindquistCoordinates(bhl, x1v, x2v, x3v, &r, &theta, &phi);
        Real sin_theta = sin(theta);
        Real cos_theta = cos(theta);
        Real sin_phi = sin(phi);
        Real cos_phi = cos(phi);

        Real rho_init, e_init;

        if (r > 1.0)
        {
          rho_init = bhl.rho_inf;
          e_init = bhl.e_inf;
          // e_init = (bhl.total_pressure_inf - magnetic_pressure) / gm1;
        }
        else
        {
          rho_init = bhl.dexcise;
          e_init = bhl.pexcise / gm1;
        }

        // Set hydro primitive variables. Internal energy e_init is adjusted to
        // match the total pressure equilibrium.
        w0_(m, IDN, k, j, i) = rho_init;
        w0_(m, IEN, k, j, i) = e_init;

        w0_(m, IVX, k, j, i) = -bhl.u1_prim_inf;
        w0_(m, IVY, k, j, i) = 0.0;
        w0_(m, IVZ, k, j, i) = 0.0;

        // Compute total pressure (equal to gas pressure in non-radiating runs)
        Real ptot = gm1 * w0_(m, IEN, k, j, i);
        if (is_radiation_enabled)
          // ptot += urad / 3.0;
          max_ptot = fmax(ptot, max_ptot);
      },
      Kokkos::Max<Real>(ptotmax));

  //  ---------------------------------------
  //      initialize magnetic fields
  //  ---------------------------------------
  if (pmbp->pmhd != nullptr)
  {
    // compute vector potential over all faces
    // int ncells1 = indcs.nx1 + 2 * (indcs.ng);
    // int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * (indcs.ng)) : 1;
    // int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * (indcs.ng)) : 1;
    // DvceArray4D<Real> a1, a2, a3;
    // Kokkos::realloc(a1, nmb, ncells3, ncells2, ncells1);
    // Kokkos::realloc(a2, nmb, ncells3, ncells2, ncells1);
    // Kokkos::realloc(a3, nmb, ncells3, ncells2, ncells1);

    auto &b0 = pmbp->pmhd->b0;

    par_for(
        "pgen_b0", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          // Compute face-centered fields from curl(A).
          Real dx1 = size.d_view(m).dx1;
          Real dx2 = size.d_view(m).dx2;
          Real dx3 = size.d_view(m).dx3;

          b0.x1f(m, k, j, i) = 0;

          b0.x2f(m, k, j, i) = incoming_By;

          b0.x3f(m, k, j, i) = incoming_Bz;

          // Include extra face-component at edge of block in each direction
          if (i == ie)
          {
            b0.x1f(m, k, j, i + 1) = 0;
          }
          if (j == je)
          {
            b0.x2f(m, k, j + 1, i) = incoming_By;
          }
          if (k == ke)
          {
            b0.x3f(m, k + 1, j, i) = incoming_Bz;
          }
        });

    // Introduce random turbulence to the initial gas profile
    auto &pturb = pmbp->pturb;
    if (bhl.sigma_variation_initial)
    {
      auto dB_incoming = bhl.B_mag_inf * bhl.dB_over_B_mag_inf;

      // pturb->Initialize();
      pturb->InitializeModes(dB_incoming, false);

      // Required after initialization
      pturb->CopyForceTmpIntoForce();

      pturb->AssignInitialProfile();
    }

    // Compute cell-centered magnetic fields
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for(
        "pgen_bcc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          // cell-centered fields are simple linear average of face-centered fields
          Real &w_bx = bcc_(m, IBX, k, j, i);
          Real &w_by = bcc_(m, IBY, k, j, i);
          Real &w_bz = bcc_(m, IBZ, k, j, i);
          w_bx = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i + 1));
          w_by = 0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j + 1, i));
          w_bz = 0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k + 1, j, i));
        });
  }

  // Convert primitives to conserved
  if (pmbp->phydro != nullptr)
  {
    pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
  }
  else if (pmbp->pmhd != nullptr)
  {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
  }

  return;
}

//
// @YK : Some free functions defined below
//
namespace
{

  //----------------------------------------------------------------------------------------
  // Function to calculate time component of contravariant four velocity in BL
  // Inputs:
  //   r: radial Boyer-Lindquist coordinate
  //   sin_theta: sine of polar Boyer-Lindquist coordinate
  // Outputs:
  //   returned value: u_t

  KOKKOS_INLINE_FUNCTION
  static Real CalculateCovariantUT(struct bhl_pgen pgen, Real r, Real sin_theta, Real l)
  {
    // Compute BL metric components
    Real sigma = SQR(r) + SQR(pgen.spin) * (1.0 - SQR(sin_theta));
    Real g_00 = -1.0 + 2.0 * r / sigma;
    Real g_03 = -2.0 * pgen.spin * r / sigma * SQR(sin_theta);
    Real g_33 = (SQR(r) + SQR(pgen.spin) +
                 2.0 * SQR(pgen.spin) * r / sigma * SQR(sin_theta)) *
                SQR(sin_theta);

    // Compute time component of covariant BL 4-velocity
    Real u_t = -sqrt(fmax((SQR(g_03) - g_00 * g_33) / (g_33 + 2.0 * l * g_03 + SQR(l) * g_00), 0.0));
    return u_t;
  }

  //----------------------------------------------------------------------------------------
  // Function for returning corresponding Boyer-Lindquist coordinates of point
  // Inputs:
  //   x1,x2,x3: global coordinates to be converted
  // Outputs:
  //   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

  KOKKOS_INLINE_FUNCTION
  static void GetBoyerLindquistCoordinates(struct bhl_pgen pgen,
                                           Real x1, Real x2, Real x3,
                                           Real *pr, Real *ptheta, Real *pphi)
  {
    Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r = fmax((sqrt(SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad) - SQR(pgen.spin)) + 4.0 * SQR(pgen.spin) * SQR(x3))) / sqrt(2.0)), 1.0);
    *pr = r;
    *ptheta = (fabs(x3 / r) < 1.0) ? acos(x3 / r) : acos(copysign(1.0, x3));
    *pphi = atan2(r * x2 - pgen.spin * x1, pgen.spin * x2 + r * x1) -
            pgen.spin * r / (SQR(r) - 2.0 * r + SQR(pgen.spin));
    return;
  }

  //----------------------------------------------------------------------------------------
  // Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
  // Inputs:
  //   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
  //   x1,x2,x3: Cartesian Kerr-Schild coordinates of point
  // Outputs:
  //   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
  // Notes:
  //   Schwarzschild coordinates match Boyer-Lindquist when a = 0

  KOKKOS_INLINE_FUNCTION
  static void TransformVector(struct bhl_pgen pgen,
                              Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                              Real x1, Real x2, Real x3,
                              Real *pa0, Real *pa1, Real *pa2, Real *pa3)
  {
    Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r = fmax((sqrt(SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad) - SQR(pgen.spin)) + 4.0 * SQR(pgen.spin) * SQR(x3))) / sqrt(2.0)), 1.0);
    Real delta = SQR(r) - 2.0 * r + SQR(pgen.spin);
    *pa0 = a0_bl + 2.0 * r / delta * a1_bl;
    *pa1 = a1_bl * ((r * x1 + pgen.spin * x2) / (SQR(r) + SQR(pgen.spin)) - x2 * pgen.spin / delta) +
           a2_bl * x1 * x3 / r * sqrt((SQR(r) + SQR(pgen.spin)) / (SQR(x1) + SQR(x2))) -
           a3_bl * x2;
    *pa2 = a1_bl * ((r * x2 - pgen.spin * x1) / (SQR(r) + SQR(pgen.spin)) + x1 * pgen.spin / delta) +
           a2_bl * x2 * x3 / r * sqrt((SQR(r) + SQR(pgen.spin)) / (SQR(x1) + SQR(x2))) +
           a3_bl * x1;
    *pa3 = a1_bl * x3 / r -
           a2_bl * r * sqrt((SQR(x1) + SQR(x2)) / (SQR(r) + SQR(pgen.spin)));
    return;
  }

} // namespace

//----------------------------------------------------------------------------------------
//! \fn BhlAccretionBoundary
//  \brief Sets boundary condition on surfaces of computational domain

void BhlAccretionBoundary(Mesh *pm)
{
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  auto bhl = bhl_accretion;

  // FIXME : need to use correct formula.
  //
  // @YK : maybe okay if boundary is far away and the wind is slow?
  //
  const auto incoming_By = bhl.B_mag_inf * sin(bhl.magnetic_field_angle_yz);
  const auto incoming_Bz = bhl.B_mag_inf * cos(bhl.magnetic_field_angle_yz);

  // const Real time = pm->time;
  // const Real dt = pm->dt;
  // std::cout << " Boundary Condition :: time = " << time << std::endl;
  // std::cout << " Boundary Condition :: dt   = " << dt << std::endl;

  auto pturb = pm->pmb_pack->pturb;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pm->pmb_pack->phydro != nullptr)
  {
    u0_ = pm->pmb_pack->phydro->u0;
    w0_ = pm->pmb_pack->phydro->w0;
  }
  else if (pm->pmb_pack->pmhd != nullptr)
  {
    u0_ = pm->pmb_pack->pmhd->u0;
    w0_ = pm->pmb_pack->pmhd->w0;
  }
  int nmb = pm->pmb_pack->nmb_thispack;
  int nvar = u0_.extent_int(1);

  // Determine if radiation is enabled
  const bool is_radiation_enabled = (pm->pmb_pack->prad != nullptr);
  DvceArray5D<Real> i0_;
  int nang1;
  if (is_radiation_enabled)
  {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
  }

  // X1-Boundary
  // Set X1-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr)
  {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for(
        "noinflow_field_x1", DevExeSpace(), 0, (nmb - 1), 0, (n3 - 1), 0, (n2 - 1),
        KOKKOS_LAMBDA(int m, int k, int j) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user)
          {
            for (int i = 0; i < ng; ++i)
            {
              b0.x1f(m, k, j, is - i - 1) = b0.x1f(m, k, j, is);
              b0.x2f(m, k, j, is - i - 1) = b0.x2f(m, k, j, is);
              if (j == n2 - 1)
              {
                b0.x2f(m, k, j + 1, is - i - 1) = b0.x2f(m, k, j + 1, is);
              }
              b0.x3f(m, k, j, is - i - 1) = b0.x3f(m, k, j, is);
              if (k == n3 - 1)
              {
                b0.x3f(m, k + 1, j, is - i - 1) = b0.x3f(m, k + 1, j, is);
              }
            }
          }
          // @YK : this loop sets the B field of incoming wind
          if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user)
          {

            for (int i = 0; i < ng; ++i)
            {
              b0.x1f(m, k, j, ie + i + 2) = 0.; // b0.x1f(m,k,j,ie+1);

              b0.x2f(m, k, j, ie + i + 1) = incoming_By; // b0.x2f(m,k,j,ie);
              if (j == n2 - 1)
              {
                b0.x2f(m, k, j + 1, ie + i + 1) = incoming_By;
              }; // b0.x2f(m,k,j+1,ie);}

              b0.x3f(m, k, j, ie + i + 1) = incoming_Bz;
              if (k == n3 - 1)
              {
                b0.x3f(m, k + 1, j, ie + i + 1) = incoming_Bz;
              }
            }
          }
        });
  }

  if (bhl.sigma_variation_inject_from_boundary)
  {
    pturb->InjectTurbulenceOnOuterX1Boundary();
  }

  // ConsToPrim over all X1 ghost zones *and* at the innermost/outermost X1-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr)
  {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, is - ng, is, 0, (n2 - 1), 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, ie, ie + ng, 0, (n2 - 1), 0, (n3 - 1));
  }
  else if (pm->pmb_pack->pmhd != nullptr)
  {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_, b0, w0_, bcc, false, is - ng, is, 0, (n2 - 1), 0, (n3 - 1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_, b0, w0_, bcc, false, ie, ie + ng, 0, (n2 - 1), 0, (n3 - 1));
  }

  // Set X1-BCs on w0 if Meshblock face is at the edge of computational domain
  // auto &bcc = pm->pmb_pack->pmhd->bcc0;
  par_for(
      "noinflow_hydro_x1", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), 0, (n3 - 1), 0, (n2 - 1),
      KOKKOS_LAMBDA(int m, int n, int k, int j) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user)
        {
          for (int i = 0; i < ng; ++i)
          {
            if (n == (IVX))
            {
              w0_(m, n, k, j, is - i - 1) = fmin(0.0, w0_(m, n, k, j, is));
            }
            else
            {
              w0_(m, n, k, j, is - i - 1) = w0_(m, n, k, j, is);
            }
          }
        }
        // @YK : this loop sets the hydro variables
        if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user)
        {
          // Real bx = 0.0;
          // const auto by = bcc(m, 1, k, j, ie + 1);
          // const auto bz = bcc(m, 2, k, j, ie + 1);
          // const auto magnetic_pressure = 0.5 * (SQR(by) + SQR(bz)) / SQR(bhl.W_inf);

          for (int i = 0; i < ng; ++i)
          {
            if (n == (IVX))
            {
              // w0_(m, n, k, j, ie + i + 1) = fmax(0.0, w0_(m, n, k, j, ie));
              w0_(m, n, k, j, ie + i + 1) = -bhl.u1_prim_inf;
            }
            else
            {
              w0_(m, IDN, k, j, ie + i + 1) = bhl.rho_inf;
              w0_(m, IEN, k, j, ie + i + 1) = bhl.e_inf;
              // w0_(m, IEN, k, j, ie + i + 1) = (bhl.total_pressure_inf - magnetic_pressure) / (bhl.gamma_adi - 1.0);
              w0_(m, IVY, k, j, ie + i + 1) = 0;
              w0_(m, IVZ, k, j, ie + i + 1) = 0;
            }
          }
        }
      });
  if (is_radiation_enabled)
  {
    // Set X1-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for(
        "noinflow_rad_x1", DevExeSpace(), 0, (nmb - 1), 0, nang1, 0, (n3 - 1), 0, (n2 - 1),
        KOKKOS_LAMBDA(int m, int n, int k, int j) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user)
          {
            for (int i = 0; i < ng; ++i)
            {
              i0_(m, n, k, j, is - i - 1) = i0_(m, n, k, j, is);
            }
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user)
          {
            for (int i = 0; i < ng; ++i)
            {
              i0_(m, n, k, j, ie + i + 1) = i0_(m, n, k, j, ie);
            }
          }
        });
  }

  // PrimToCons on X1 ghost zones
  if (pm->pmb_pack->phydro != nullptr)
  {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, is - ng, is - 1, 0, (n2 - 1), 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, ie + 1, ie + ng, 0, (n2 - 1), 0, (n3 - 1));
  }
  else if (pm->pmb_pack->pmhd != nullptr)
  {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is - ng, is - 1, 0, (n2 - 1), 0, (n3 - 1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, ie + 1, ie + ng, 0, (n2 - 1), 0, (n3 - 1));
  }

  // Compute a new random force for the next time step
  if (bhl.sigma_variation_inject_from_boundary)
  {
    const auto dB_incoming = bhl.B_mag_inf * bhl.dB_over_B_mag_inf;
    pturb->InitializeModes(dB_incoming, true);
    pturb->AddForcing();
  }

  // X2-Boundary
  // Set X2-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr)
  {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for(
        "noinflow_field_x2", DevExeSpace(), 0, (nmb - 1), 0, (n3 - 1), 0, (n1 - 1),
        KOKKOS_LAMBDA(int m, int k, int i) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user)
          {
            for (int j = 0; j < ng; ++j)
            {
              b0.x1f(m, k, js - j - 1, i) = b0.x1f(m, k, js, i);
              if (i == n1 - 1)
              {
                b0.x1f(m, k, js - j - 1, i + 1) = b0.x1f(m, k, js, i + 1);
              }
              b0.x2f(m, k, js - j - 1, i) = b0.x2f(m, k, js, i);
              b0.x3f(m, k, js - j - 1, i) = b0.x3f(m, k, js, i);
              if (k == n3 - 1)
              {
                b0.x3f(m, k + 1, js - j - 1, i) = b0.x3f(m, k + 1, js, i);
              }
            }
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user)
          {
            for (int j = 0; j < ng; ++j)
            {
              b0.x1f(m, k, je + j + 1, i) = b0.x1f(m, k, je, i);
              if (i == n1 - 1)
              {
                b0.x1f(m, k, je + j + 1, i + 1) = b0.x1f(m, k, je, i + 1);
              }
              b0.x2f(m, k, je + j + 2, i) = b0.x2f(m, k, je + 1, i);
              b0.x3f(m, k, je + j + 1, i) = b0.x3f(m, k, je, i);
              if (k == n3 - 1)
              {
                b0.x3f(m, k + 1, je + j + 1, i) = b0.x3f(m, k + 1, je, i);
              }
            }
          }
        });
  }
  // ConsToPrim over all X2 ghost zones *and* at the innermost/outermost X2-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr)
  {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, 0, (n1 - 1), js - ng, js, 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, 0, (n1 - 1), je, je + ng, 0, (n3 - 1));
  }
  else if (pm->pmb_pack->pmhd != nullptr)
  {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_, b0, w0_, bcc, false, 0, (n1 - 1), js - ng, js, 0, (n3 - 1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_, b0, w0_, bcc, false, 0, (n1 - 1), je, je + ng, 0, (n3 - 1));
  }
  // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for(
      "noinflow_hydro_x2", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), 0, (n3 - 1), 0, (n1 - 1),
      KOKKOS_LAMBDA(int m, int n, int k, int i) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user)
        {
          for (int j = 0; j < ng; ++j)
          {
            if (n == (IVY))
            {
              w0_(m, n, k, js - j - 1, i) = fmin(0.0, w0_(m, n, k, js, i));
            }
            else
            {
              w0_(m, n, k, js - j - 1, i) = w0_(m, n, k, js, i);
            }
          }
        }
        if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user)
        {
          for (int j = 0; j < ng; ++j)
          {
            if (n == (IVY))
            {
              w0_(m, n, k, je + j + 1, i) = fmax(0.0, w0_(m, n, k, je, i));
            }
            else
            {
              w0_(m, n, k, je + j + 1, i) = w0_(m, n, k, je, i);
            }
          }
        }
      });
  if (is_radiation_enabled)
  {
    // Set X2-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for(
        "noinflow_rad_x2", DevExeSpace(), 0, (nmb - 1), 0, nang1, 0, (n3 - 1), 0, (n1 - 1),
        KOKKOS_LAMBDA(int m, int n, int k, int i) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user)
          {
            for (int j = 0; j < ng; ++j)
            {
              i0_(m, n, k, js - j - 1, i) = i0_(m, n, k, js, i);
            }
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user)
          {
            for (int j = 0; j < ng; ++j)
            {
              i0_(m, n, k, je + j + 1, i) = i0_(m, n, k, je, i);
            }
          }
        });
  }
  // PrimToCons on X2 ghost zones
  if (pm->pmb_pack->phydro != nullptr)
  {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, 0, (n1 - 1), js - ng, js - 1, 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, 0, (n1 - 1), je + 1, je + ng, 0, (n3 - 1));
  }
  else if (pm->pmb_pack->pmhd != nullptr)
  {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, 0, (n1 - 1), js - ng, js - 1, 0, (n3 - 1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, 0, (n1 - 1), je + 1, je + ng, 0, (n3 - 1));
  }

  // X3-Boundary
  // Set X3-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr)
  {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for(
        "noinflow_field_x3", DevExeSpace(), 0, (nmb - 1), 0, (n2 - 1), 0, (n1 - 1),
        KOKKOS_LAMBDA(int m, int j, int i) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user)
          {
            for (int k = 0; k < ng; ++k)
            {
              b0.x1f(m, ks - k - 1, j, i) = b0.x1f(m, ks, j, i);
              if (i == n1 - 1)
              {
                b0.x1f(m, ks - k - 1, j, i + 1) = b0.x1f(m, ks, j, i + 1);
              }
              b0.x2f(m, ks - k - 1, j, i) = b0.x2f(m, ks, j, i);
              if (j == n2 - 1)
              {
                b0.x2f(m, ks - k - 1, j + 1, i) = b0.x2f(m, ks, j + 1, i);
              }
              b0.x3f(m, ks - k - 1, j, i) = b0.x3f(m, ks, j, i);
            }
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user)
          {
            for (int k = 0; k < ng; ++k)
            {
              b0.x1f(m, ke + k + 1, j, i) = b0.x1f(m, ke, j, i);
              if (i == n1 - 1)
              {
                b0.x1f(m, ke + k + 1, j, i + 1) = b0.x1f(m, ke, j, i + 1);
              }
              b0.x2f(m, ke + k + 1, j, i) = b0.x2f(m, ke, j, i);
              if (j == n2 - 1)
              {
                b0.x2f(m, ke + k + 1, j + 1, i) = b0.x2f(m, ke, j + 1, i);
              }
              b0.x3f(m, ke + k + 2, j, i) = b0.x3f(m, ke + 1, j, i);
            }
          }
        });
  }
  // ConsToPrim over all X3 ghost zones *and* at the innermost/outermost X3-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr)
  {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, 0, (n1 - 1), 0, (n2 - 1), ks - ng, ks);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, 0, (n1 - 1), 0, (n2 - 1), ke, ke + ng);
  }
  else if (pm->pmb_pack->pmhd != nullptr)
  {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_, b0, w0_, bcc, false, 0, (n1 - 1), 0, (n2 - 1), ks - ng, ks);
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_, b0, w0_, bcc, false, 0, (n1 - 1), 0, (n2 - 1), ke, ke + ng);
  }
  // Set X3-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for(
      "noinflow_hydro_x3", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), 0, (n2 - 1), 0, (n1 - 1),
      KOKKOS_LAMBDA(int m, int n, int j, int i) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user)
        {
          for (int k = 0; k < ng; ++k)
          {
            if (n == (IVZ))
            {
              w0_(m, n, ks - k - 1, j, i) = fmin(0.0, w0_(m, n, ks, j, i));
            }
            else
            {
              w0_(m, n, ks - k - 1, j, i) = w0_(m, n, ks, j, i);
            }
          }
        }
        if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user)
        {
          for (int k = 0; k < ng; ++k)
          {
            if (n == (IVZ))
            {
              w0_(m, n, ke + k + 1, j, i) = fmax(0.0, w0_(m, n, ke, j, i));
            }
            else
            {
              w0_(m, n, ke + k + 1, j, i) = w0_(m, n, ke, j, i);
            }
          }
        }
      });
  if (is_radiation_enabled)
  {
    // Set X3-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for(
        "noinflow_rad_x3", DevExeSpace(), 0, (nmb - 1), 0, nang1, 0, (n2 - 1), 0, (n1 - 1),
        KOKKOS_LAMBDA(int m, int n, int j, int i) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user)
          {
            for (int k = 0; k < ng; ++k)
            {
              i0_(m, n, ks - k - 1, j, i) = i0_(m, n, ks, j, i);
            }
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user)
          {
            for (int k = 0; k < ng; ++k)
            {
              i0_(m, n, ke + k + 1, j, i) = i0_(m, n, ke, j, i);
            }
          }
        });
  }
  // PrimToCons on X3 ghost zones
  if (pm->pmb_pack->phydro != nullptr)
  {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, 0, (n1 - 1), 0, (n2 - 1), ks - ng, ks - 1);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, 0, (n1 - 1), 0, (n2 - 1), ke + 1, ke + ng);
  }
  else if (pm->pmb_pack->pmhd != nullptr)
  {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, 0, (n1 - 1), 0, (n2 - 1), ks - ng, ks - 1);
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, 0, (n1 - 1), 0, (n2 - 1), ke + 1, ke + ng);
  }

  return;
}

//----------------------------------------------------------------------------------------
// Function for computing accretion fluxes through constant spherical KS radius surfaces

void BhlAccretionHistory(HistoryData *pdata, Mesh *pm)
{
  MeshBlockPack *pmbp = pm->pmb_pack;

  // extract BH parameters
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  // set nvars, adiabatic index, primitive array w0, and field array bcc0 if is_mhd
  int nvars;
  Real gamma;
  bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr)
  {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    gamma = pmbp->phydro->peos->eos_data.gamma;
    w0_ = pmbp->phydro->w0;
  }
  else if (pmbp->pmhd != nullptr)
  {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    gamma = pmbp->pmhd->peos->eos_data.gamma;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }

  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  // int nflux = (is_mhd) ? 7 : 3;  // @YK : magnetic flux, momentum flux (3)
  int nflux = 10;  // @YK : Mdot, Edot, Ldot, Phi, Momentum drag (3), Gravitational drag (3)

  // set number of and names of history variables for hydro or mhd

  // Quantities to be computed at different radii
  //  (1) mass accretion rate
  //  (2) energy flux
  //  (3) angular momentum flux
  //  (4) magnetic flux
  //  (5,6,7) Momentum flux in x, y, z
  //  (8,9,10) Gravitational drag in x, y, z
  //
  pdata->nhist = nradii * nflux;

  if (pdata->nhist > NHISTORY_VARIABLES)
  {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int g = 0; g < nradii; ++g)
  {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();
    pdata->label[nflux * g + 0] = "mdot_" + rad_str;
    pdata->label[nflux * g + 1] = "edot_" + rad_str;
    pdata->label[nflux * g + 2] = "ldot_" + rad_str;
    if (is_mhd)
    {
      pdata->label[nflux * g + 3] = "phi_" + rad_str;

      pdata->label[nflux * g + 4] = "Fmx_" + rad_str;
      pdata->label[nflux * g + 5] = "Fmy_" + rad_str;
      pdata->label[nflux * g + 6] = "Fmz_" + rad_str;

      pdata->label[nflux * g + 7] = "Fgx_" + rad_str;
      pdata->label[nflux * g + 8] = "Fgy_" + rad_str;
      pdata->label[nflux * g + 9] = "Fgz_" + rad_str;
    }
  }

  // go through angles at each radii:
  DualArray2D<Real> interpolated_bcc; // needed for MHD
  for (int g = 0; g < nradii; ++g)
  {
    // zero fluxes at this radius
    pdata->hdata[nflux * g + 0] = 0.0;
    pdata->hdata[nflux * g + 1] = 0.0;
    pdata->hdata[nflux * g + 2] = 0.0;
    if (is_mhd){
      pdata->hdata[nflux * g + 3] = 0.0;
      pdata->hdata[nflux * g + 4] = 0.0;
      pdata->hdata[nflux * g + 5] = 0.0;
      pdata->hdata[nflux * g + 6] = 0.0;

      pdata->hdata[nflux * g + 7] = 0.0;
      pdata->hdata[nflux * g + 8] = 0.0;
      pdata->hdata[nflux * g + 9] = 0.0;
    }

    // interpolate primitives (and cell-centered magnetic fields iff mhd)
    if (is_mhd)
    {
      grids[g]->InterpolateToSphere(3, bcc0_);
      Kokkos::realloc(interpolated_bcc, grids[g]->nangles, 3);
      Kokkos::deep_copy(interpolated_bcc, grids[g]->interp_vals);
      interpolated_bcc.template modify<DevExeSpace>();
      interpolated_bcc.template sync<HostMemSpace>();
    }
    grids[g]->InterpolateToSphere(nvars, w0_);

    // compute fluxes
    for (int n = 0; n < grids[g]->nangles; ++n)
    {
      // extract coordinate data at this angle
      Real r = grids[g]->radius;
      Real theta = grids[g]->polar_pos.h_view(n, 0);
      Real phi = grids[g]->polar_pos.h_view(n, 1);
      Real x1 = grids[g]->interp_coord.h_view(n, 0);
      Real x2 = grids[g]->interp_coord.h_view(n, 1);
      Real x3 = grids[g]->interp_coord.h_view(n, 2);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1, x2, x3, flat, spin, glower, gupper);

      // extract interpolated primitives
      Real &int_dn = grids[g]->interp_vals.h_view(n, IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n, IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n, IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n, IVZ);
      Real &int_ie = grids[g]->interp_vals.h_view(n, IEN);

      // extract interpolated field components (iff is_mhd)
      Real int_bx = 0.0, int_by = 0.0, int_bz = 0.0;
      if (is_mhd)
      {
        int_bx = interpolated_bcc.h_view(n, IBX);
        int_by = interpolated_bcc.h_view(n, IBY);
        int_bz = interpolated_bcc.h_view(n, IBZ);
      }

      // Compute interpolated u^\mu in CKS
      Real q = glower[1][1] * int_vx * int_vx + 2.0 * glower[1][2] * int_vx * int_vy +
               2.0 * glower[1][3] * int_vx * int_vz + glower[2][2] * int_vy * int_vy +
               2.0 * glower[2][3] * int_vy * int_vz + glower[3][3] * int_vz * int_vz;
      Real alpha = sqrt(-1.0 / gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor / alpha;
      Real u1 = int_vx - alpha * lor * gupper[0][1];
      Real u2 = int_vy - alpha * lor * gupper[0][2];
      Real u3 = int_vz - alpha * lor * gupper[0][3];

      // Lower vector indices
      Real u_0 = glower[0][0] * u0 + glower[0][1] * u1 + glower[0][2] * u2 + glower[0][3] * u3;
      Real u_1 = glower[1][0] * u0 + glower[1][1] * u1 + glower[1][2] * u2 + glower[1][3] * u3;
      Real u_2 = glower[2][0] * u0 + glower[2][1] * u1 + glower[2][2] * u2 + glower[2][3] * u3;
      Real u_3 = glower[3][0] * u0 + glower[3][1] * u1 + glower[3][2] * u2 + glower[3][3] * u3;

      // Calculate 4-magnetic field (returns zero if not MHD)
      Real b0 = u_1 * int_bx + u_2 * int_by + u_3 * int_bz;
      Real b1 = (int_bx + b0 * u1) / u0;
      Real b2 = (int_by + b0 * u2) / u0;
      Real b3 = (int_bz + b0 * u3) / u0;

      // compute b_\mu in CKS and b_sq (returns zero if not MHD)
      Real b_0 = glower[0][0] * b0 + glower[0][1] * b1 + glower[0][2] * b2 + glower[0][3] * b3;
      Real b_1 = glower[1][0] * b0 + glower[1][1] * b1 + glower[1][2] * b2 + glower[1][3] * b3;
      Real b_2 = glower[2][0] * b0 + glower[2][1] * b1 + glower[2][2] * b2 + glower[2][3] * b3;
      Real b_3 = glower[3][0] * b0 + glower[3][1] * b1 + glower[3][2] * b2 + glower[3][3] * b3;
      Real b_sq = b0 * b_0 + b1 * b_1 + b2 * b_2 + b3 * b_3;

      // Transform CKS 4-velocity and 4-magnetic field to spherical KS
      Real a2 = SQR(spin);
      Real rad2 = SQR(x1) + SQR(x2) + SQR(x3);
      Real r2 = SQR(r);
      Real sth = sin(theta);
      Real sph = sin(phi);
      Real cph = cos(phi);
      Real drdx = r * x1 / (2.0 * r2 - rad2 + a2);
      Real drdy = r * x2 / (2.0 * r2 - rad2 + a2);
      Real drdz = (r * x3 + a2 * x3 / r) / (2.0 * r2 - rad2 + a2);
      // contravariant r component of 4-velocity
      Real ur = drdx * u1 + drdy * u2 + drdz * u3;
      // contravariant r component of 4-magnetic field (returns zero if not MHD)
      Real br = drdx * b1 + drdy * b2 + drdz * b3;
      // covariant phi component of 4-velocity
      Real u_ph = (-r * sph - spin * cph) * sth * u_1 + (r * cph - spin * sph) * sth * u_2;
      // covariant phi component of 4-magnetic field (returns zero if not MHD)
      Real b_ph = (-r * sph - spin * cph) * sth * b_1 + (r * cph - spin * sph) * sth * b_2;

      // integration params
      Real &domega = grids[g]->solid_angles.h_view(n);
      Real sqrtmdet = (r2 + SQR(spin * cos(theta)));

      // temporary variables
      Real wtot = int_dn + gamma * int_ie + b_sq;
      Real ptot = (gamma - 1.0) * int_ie + 0.5 * b_sq;

      // compute mass flux
      pdata->hdata[nflux * g + 0] += -1.0 * int_dn * ur * sqrtmdet * domega;

      // compute energy flux
      Real t1_0 = wtot * ur * u_0 - br * b_0;
      // YK : there was a minus sign here, but removed it. T^r_t describes the energy "inflow"
      pdata->hdata[nflux * g + 1] += t1_0 * sqrtmdet * domega;

      // compute angular momentum flux
      Real t1_3 = wtot * ur * u_ph - br * b_ph;
      pdata->hdata[nflux * g + 2] += t1_3 * sqrtmdet * domega;

      if (is_mhd)
      {
        // compute magnetic flux
        pdata->hdata[nflux * g + 3] += 0.5 * fabs(br * u0 - b0 * ur) * sqrtmdet * domega;

        // Compute momentum drag
        Real tr_1 = (wtot * ur * u_1) + (ptot * drdx) - (br * b_1);
        Real tr_2 = (wtot * ur * u_2) + (ptot * drdy) - (br * b_2);
        Real tr_3 = (wtot * ur * u_3) + (ptot * drdz) - (br * b_3);
        pdata->hdata[nflux * g + 4] += tr_1 * sqrtmdet * domega;
        pdata->hdata[nflux * g + 5] += tr_2 * sqrtmdet * domega;
        pdata->hdata[nflux * g + 6] += tr_3 * sqrtmdet * domega;
      }
    }

    //
    // Compute gravitational drag
    //
    // pdata->hdata[nflux * (nradii - 1) + 7] = 0.0;
    // pdata->hdata[nflux * (nradii - 1) + 8] = 0.0;
    // pdata->hdata[nflux * (nradii - 1) + 9] = 0.0;

    // loop over all MeshBlocks in this pack
    auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
    int is = indcs.is;
    int js = indcs.js;
    int ks = indcs.ks;
    const int nmkji = (pmbp->nmb_thispack) * indcs.nx3 * indcs.nx2 * indcs.nx1;
    const int nkji = indcs.nx3 * indcs.nx2 * indcs.nx1;
    const int nji = indcs.nx2 * indcs.nx1;
    auto &size = pm->pmb_pack->pmb->mb_size;

    const Real min_radius = grids[g]->radius;
    auto bhl = bhl_accretion;

    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce(
        "gravitational_drag", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
        KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
          // compute m,k,j,i indices of thread and call function
          int m = (idx) / nkji;
          int k = (idx - m * nkji) / nji;
          int j = (idx - m * nkji - k * nji) / indcs.nx1;
          int i = (idx - m * nkji - k * nji - j * indcs.nx1) + is;
          k += ks;
          j += js;

          // Volume & Cell-centered coordinates & 1/r^3
          Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;

          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j - js, indcs.nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

          Real r2 = SQR(x1v) + SQR(x2v) + SQR(x3v);

          // @YK : we can set the integrand to zero inside our outside a certain radius.
          if (r2 > SQR(min_radius) and r2 < SQR(bhl.grav_drag_mask_rmax))
          {
            Real one_over_r3 = 1.0 / (r2 * sqrt(r2));

            // Gravitational drag
            array_sum::GlobalSum grav_drag;
            grav_drag.the_array[0] = vol * w0_(m, IDN, k, j, i) * x1v * one_over_r3;
            grav_drag.the_array[1] = vol * w0_(m, IDN, k, j, i) * x2v * one_over_r3;
            grav_drag.the_array[2] = vol * w0_(m, IDN, k, j, i) * x3v * one_over_r3;

            // sum into parallel reduce
            mb_sum += grav_drag;
          }
        },
        Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));
    Kokkos::fence();

    // store data into hdata array
    pdata->hdata[nflux * g + 7] += sum_this_mb.the_array[0];
    pdata->hdata[nflux * g + 8] += sum_this_mb.the_array[1];
    pdata->hdata[nflux * g + 9] += sum_this_mb.the_array[2];
  }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n)
  {
    pdata->hdata[n] = 0.0;
  }

  return;
}
