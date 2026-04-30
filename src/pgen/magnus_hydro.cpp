
#include <math.h>
#include <stdio.h>

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

#include <Kokkos_Random.hpp>

#include "athena.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "kokkos/core/src/Kokkos_Timer.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "radiation/radiation.hpp"
#include "src/srcterms/turb_driver.hpp"

// prototypes for functions used internally to this pgen
namespace {

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bhl_pgen pgen, Real x1, Real x2,
                                         Real x3, Real *pr, Real *ptheta,
                                         Real *pphi);

// Useful container for physical parameters
struct bhl_pgen {
  Real spin;             // black hole spin
  Real dexcise, pexcise; // excision parameters

  Real gamma_adi; // EOS parameters
  // Real rho_min, rho_pow, pgas_min, pgas_pow; // background parameters

  // BHL-accretion related parameters
  Real rho_inf; // Incoming wind density
  // Real mach;    // Incoming wind Mach number
  // Real Ra;      // Accretion radius (sets wind velocity)
  Real v_inf;  // Incoming wind speed
  Real cs_inf; // Incoming wind sound speed

  // Controlling the B field configuration
  Real inv_plasma_beta_inf;
  Real B_mag_inf;               // Incoming wind magnetic field magnitude
  Real magnetic_field_angle_yz; // Angle of B in y-z plane (rad), measured from
                                // z-axis

  // Temporary quantities required for primitive variable calculations
  Real e_inf;           // e = rho epsilon
  Real rho_times_h_inf; // rho h
  Real pressure_inf;    // pressure
  Real W_inf;           // u^0 = W
  Real u1_prim_inf;     // u^1 = W v^1
  // Real comoving_b2;        // b^2
  // Real total_pressure_inf; // p_gas + b^2/2

  // Real u1_prim;
  // Real u2_prim;
  // Real u3_prim;

  // Real arad; // radiation constant? -> ignore

  // Radius mask for computing gravitational drag
  Real volume_integral_rmax;
};

static bhl_pgen bhl_accretion;

} // namespace

// Prototypes for user-defined BCs and history functions
void BhlAccretionBoundary(Mesh *pm);
void BhlAccretionHistory(HistoryData *pdata, Mesh *pm);
// @YK : "history functions" seems to be what we call 'reduction quantities' in
// spectre (e.g. time series)

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for either Fishbone-Moncrief or
//! Chakrabarti torus in GR Compile with '-D PROBLEM=gr_torus' to enroll as
//! user-specific problem generator
//!  assumes x3 is axisymmetric direction
//
// @YK : update the docs above.
//       this seems like doing an actual initialization for domain volume?
//

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic) {
    std::cout
        << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl
        << " This problem can only be run when GR defined in <coord> block"
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
  const Real rflux = (is_radiation_enabled)
                         ? ceil(r_excise + 1.0)
                         : 1.0 + sqrt(1.0 - SQR(bhl_accretion.spin));
  const int nlev_sphere = pin->GetInteger("problem", "sphere_nlev");
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, rflux));
  // Enroll additional radii
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 2.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 3.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 4.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 5.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 8.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 10.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 16.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 20.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 40.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 80.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 160.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 320.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 640.0));

  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 2.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 2.5));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 3.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 3.5));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 4.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 8.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 16.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, nlev_sphere, 32.0));

  user_hist_func = BhlAccretionHistory;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_; // @YK : u0 is evolved vars, w0 is prim vars
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
  }

  // Extract radiation parameters if enabled
  int nangles_;
  DualArray2D<Real> nh_c_;
  DvceArray6D<Real> norm_to_tet_, tet_c_, tetcov_c_;
  DvceArray5D<Real> i0_;
  if (is_radiation_enabled) {
    nangles_ = pmbp->prad->prgeo->nangles;
    nh_c_ = pmbp->prad->nh_c;
    norm_to_tet_ = pmbp->prad->norm_to_tet;
    tet_c_ = pmbp->prad->tet_c;
    tetcov_c_ = pmbp->prad->tetcov_c;
    i0_ = pmbp->prad->i0;
  }

  // Get ideal gas EOS data
  if (pmbp->phydro != nullptr) {
    bhl_accretion.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  } else if (pmbp->pmhd != nullptr) {
    bhl_accretion.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  }
  Real gm1 = bhl_accretion.gamma_adi - 1.0;

  // Get Radiation constant (if radiation enabled)
  if (pmbp->prad != nullptr) {
    // bhl_accretion.arad = pmbp->prad->arad;
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " This pgen does not support radidation " << std::endl;
    exit(EXIT_FAILURE);
  }

  // Read problem-specific parameters from input file global parameters
  bhl_accretion.rho_inf = pin->GetReal("problem", "rho_inf");
  bhl_accretion.v_inf = pin->GetReal("problem", "v_inf");
  bhl_accretion.cs_inf = pin->GetReal("problem", "cs_inf");
  if ((bhl_accretion.v_inf >= 1.0) or (bhl_accretion.cs_inf >= 1.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " v_inf and cs_inf should be smaller than 1.0 " << std::endl;
    exit(EXIT_FAILURE);
  }

  // excision parameters
  bhl_accretion.dexcise = coord.dexcise;
  bhl_accretion.pexcise = coord.pexcise;

  // Gravitational drag - Mask
  bhl_accretion.volume_integral_rmax =
      pin->GetReal("problem", "volume_integral_rmax");

  //  ---------------------------------------
  //    Compute auxiliary variables
  //  ---------------------------------------

  bhl_accretion.e_inf =
      bhl_accretion.rho_inf * SQR(bhl_accretion.cs_inf) /
      (bhl_accretion.gamma_adi - 1. - SQR(bhl_accretion.cs_inf)) /
      bhl_accretion.gamma_adi;
  bhl_accretion.rho_times_h_inf =
      bhl_accretion.rho_inf + bhl_accretion.gamma_adi * bhl_accretion.e_inf;
  bhl_accretion.pressure_inf =
      bhl_accretion.e_inf * (bhl_accretion.gamma_adi - 1.0);
  bhl_accretion.W_inf = 1.0 / sqrt(1.0 - SQR(bhl_accretion.v_inf));
  bhl_accretion.u1_prim_inf = bhl_accretion.W_inf * bhl_accretion.v_inf;
  // bhl_accretion.comoving_b2 =
  // bhl_accretion.sigma_inf * bhl_accretion.rho_times_h_inf;
  // bhl_accretion.B_mag_inf =
  // bhl_accretion.W_inf * sqrt(bhl_accretion.comoving_b2);
  // bhl_accretion.total_pressure_inf =
  // bhl_accretion.pressure_inf + 0.5 * bhl_accretion.comoving_b2;

  const Real mach_number = bhl_accretion.v_inf / bhl_accretion.cs_inf;

  // Magnetic field - related variables (MHD only)
  if (pmbp->pmhd != nullptr) {
    bhl_accretion.inv_plasma_beta_inf =
        pin->GetReal("problem", "inv_plasma_beta_inf");
    Real plasma_beta = 1.0 / bhl_accretion.inv_plasma_beta_inf;
    const Real comoving_b2 = 2.0 * bhl_accretion.pressure_inf / plasma_beta;
    bhl_accretion.B_mag_inf = bhl_accretion.W_inf * sqrt(comoving_b2);
    bhl_accretion.magnetic_field_angle_yz =
        pin->GetReal("problem", "magnetic_field_angle_yz");
  } else {
    bhl_accretion.inv_plasma_beta_inf = 0.0;
    bhl_accretion.B_mag_inf = 0.0;
    bhl_accretion.magnetic_field_angle_yz = 0.0;
  }

  if (global_variable::my_rank == 0) {
    // Save the original stream state
    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);

    std::cout << "========================================\n"
              << " **** Initializing BHL variables **** \n"
              << "========================================\n";

    const int name_width = 10;
    const int val_width = 12;

    // Apply ordinary floating-point format for the primary variables
    std::cout << std::fixed << std::setprecision(5);

    std::cout << "  " << std::left << std::setw(name_width) << "rho_inf"
              << " = " << std::right << std::setw(val_width)
              << bhl_accretion.rho_inf << '\n';

    std::cout << "  " << std::left << std::setw(name_width) << "cs_inf" << " = "
              << std::right << std::setw(val_width) << bhl_accretion.cs_inf
              << '\n';

    // The precision is adjusted to 3 specifically for the Mach number
    std::cout << "  " << std::left << std::setw(name_width) << "v_inf" << " = "
              << std::right << std::setw(val_width) << bhl_accretion.v_inf
              << "   (Mach = " << std::setprecision(3) << mach_number << ")\n";

    // Switching to scientific notation for the energy variable
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  " << std::left << std::setw(name_width) << "e_inf" << " = "
              << std::right << std::setw(val_width) << bhl_accretion.e_inf
              << '\n';

    if (pmbp->pmhd != nullptr) {
      std::cout << "  " << std::left << std::setw(name_width)
                << "inv_plasma_beta_inf" << " = " << std::right
                << std::setw(val_width) << bhl_accretion.inv_plasma_beta_inf
                << '\n';
      std::cout << "  " << std::left << std::setw(name_width) << "B_mag_inf"
                << " = " << std::right << std::setw(val_width)
                << bhl_accretion.B_mag_inf << '\n';
    }
    std::cout << "========================================\n" << std::flush;

    // Restore the original formatting state
    std::cout.copyfmt(old_state);
  }

  // return if restart
  if (restart)
    return;

  auto bhl = bhl_accretion;
  auto &size = pmbp->pmb->mb_size;

  // const auto incoming_By = bhl.B_mag_inf * sin(bhl.magnetic_field_angle_yz);
  // const auto incoming_Bz = bhl.B_mag_inf * cos(bhl.magnetic_field_angle_yz);

  //  ---------------------------------------
  //      initialize hydro primitive variables
  //  ---------------------------------------

  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  Real ptotmax = std::numeric_limits<float>::min();

  const int nmkji = (pmbp->nmb_thispack) * indcs.nx3 * indcs.nx2 * indcs.nx1;
  const int nkji = indcs.nx3 * indcs.nx2 * indcs.nx1;
  const int nji = indcs.nx2 * indcs.nx1;

  if (global_variable::my_rank == 0) {
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
        // `nmkji' : `nkji` times number of MeshBlocks in the current
        // MeshBlockPack
        //
        // `idx` : Looks like a single integer index for a whole (giant) array
        // of
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
        ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski,
                                coord.bh_spin, glower, gupper);

        // Calculate Boyer-Lindquist coordinates of cell
        Real r, theta, phi;
        GetBoyerLindquistCoordinates(bhl, x1v, x2v, x3v, &r, &theta, &phi);
        Real sin_theta = sin(theta);
        Real cos_theta = cos(theta);
        Real sin_phi = sin(phi);
        Real cos_phi = cos(phi);

        Real rho_init, e_init;
        Real u1_prim, u2_prim, u3_prim;

        if (r > r_excise) {
          rho_init = bhl.rho_inf;
          e_init = bhl.e_inf;
          // e_init = (bhl.total_pressure_inf - magnetic_pressure) / gm1;

          //
          // (Apr 27 2026) use ui_u0 as the input value
          //
          // Lapse and shift
          const Real alpha = 1.0 / sqrt(-gupper[0][0]);
          const Real beta1 = -gupper[0][1] / gupper[0][0];
          const Real beta2 = -gupper[0][2] / gupper[0][0];
          const Real beta3 = -gupper[0][3] / gupper[0][0];

          const Real u1_u0 = -bhl.v_inf;
          const Real u2_u0 = 0.0;
          const Real u3_u0 = 0.0;

          //
          // (Apr 27 2026) use ui as the input value
          //
          // Real u1_u0, u2_u0, u3_u0;
          // {
          //   const Real u1 = -bhl.v_inf;
          //   const Real u2 = 0.0;
          //   const Real u3 = 0.0;
          //   // Compute u^0 from normalization
          //   const Real a = glower[0][0];
          //   const Real b = 2.0 * (glower[0][1] * u1 + glower[0][2] * u2 +
          //                         glower[0][3] * u3);
          //   const Real c =
          //       1.0 + (glower[1][1] * u1 * u1 + 2.0 * glower[1][2] * u1 * u2
          //       +
          //              2.0 * glower[1][3] * u1 * u3 + glower[2][2] * u2 * u2
          //              + 2.0 * glower[2][3] * u2 * u3 + glower[3][3] * u3 *
          //              u3);
          //   // u^i/u^0
          //   const Real determinant = b * b - 4.0 * a * c;
          //   if (determinant < 0.0) {
          //     u1_u0 = 0.0;
          //     u2_u0 = 0.0;
          //     u3_u0 = 0.0;

          //     std::cout << " x = " << x1v << ", y = " << x2v << ", z = " <<
          //     x3v
          //               << " / u^0 does not exist" << std::endl;

          //   } else {
          //     const Real u0 = -2.0 * c / (b - sqrt(b * b - 4.0 * a * c));
          //     u1_u0 = u1 / u0;
          //     u2_u0 = u2 / u0;
          //     u3_u0 = u3 / u0;
          //   }
          // }

          const Real v1 = (u1_u0 + beta1) / alpha;
          const Real v2 = (u2_u0 + beta2) / alpha;
          const Real v3 = (u3_u0 + beta3) / alpha;

          const Real q = glower[1][1] * v1 * v1 + 2.0 * glower[1][2] * v1 * v2 +
                         2.0 * glower[1][3] * v1 * v3 + glower[2][2] * v2 * v2 +
                         2.0 * glower[2][3] * v2 * v3 + glower[3][3] * v3 * v3;

          // if (q > 1.0) {
          //   std::cout << " x = " << x1v << ", y = " << x2v << ", z = " << x3v
          //             << " / q = " << q << std::endl;
          // }

          const Real lorentz_W = 1.0 / sqrt(1.0 - q);

          u1_prim = lorentz_W * v1;
          u2_prim = lorentz_W * v2;
          u3_prim = lorentz_W * v3;

        } else {
          rho_init = bhl.dexcise;
          e_init = bhl.pexcise / gm1;
          u1_prim = 0.0;
          u2_prim = 0.0;
          u3_prim = 0.0;
        }

        // Set hydro primitive variables. Internal energy e_init is adjusted to
        // match the total pressure equilibrium.
        w0_(m, IDN, k, j, i) = rho_init;
        w0_(m, IEN, k, j, i) = e_init;

        w0_(m, IVX, k, j, i) = u1_prim;
        w0_(m, IVY, k, j, i) = u2_prim;
        w0_(m, IVZ, k, j, i) = u3_prim;

        // w0_(m, IVX, k, j, i) = -bhl.u1_prim_inf;
        // w0_(m, IVY, k, j, i) = 0.0;
        // w0_(m, IVZ, k, j, i) = 0.0;
      },
      Kokkos::Max<Real>(ptotmax));

  //  ---------------------------------------
  //      initialize magnetic fields
  //  ---------------------------------------
  if (pmbp->pmhd != nullptr) {
    const Real By =
        bhl_accretion.B_mag_inf * sin(bhl_accretion.magnetic_field_angle_yz);
    const Real Bz =
        bhl_accretion.B_mag_inf * cos(bhl_accretion.magnetic_field_angle_yz);

    auto &b0 = pmbp->pmhd->b0;
    par_for(
        "pgen_mhd_b0", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          b0.x1f(m, k, j, i) = 0.0;
          b0.x2f(m, k, j, i) = By;
          b0.x3f(m, k, j, i) = Bz;
          if (i == ie) {
            b0.x1f(m, k, j, i + 1) = 0.0;
          }
          if (j == je) {
            b0.x2f(m, k, j + 1, i) = By;
          }
          if (k == ke) {
            b0.x3f(m, k + 1, j, i) = Bz;
          }
        });

    auto &bcc0_ = pmbp->pmhd->bcc0;
    par_for(
        "pgen_mhd_bcc0", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          bcc0_(m, IBX, k, j, i) =
              0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i + 1));
          bcc0_(m, IBY, k, j, i) =
              0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j + 1, i));
          bcc0_(m, IBZ, k, j, i) =
              0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k + 1, j, i));
        });
  }

  // Convert primitives to conserved
  if (pmbp->phydro != nullptr) {
    pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
  } else if (pmbp->pmhd != nullptr) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
  }

  return;
}

//
// @YK : Some free functions defined below
//
namespace {

//----------------------------------------------------------------------------------------
// Function to calculate time component of contravariant four velocity in BL
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: u_t

KOKKOS_INLINE_FUNCTION
static Real CalculateCovariantUT(struct bhl_pgen pgen, Real r, Real sin_theta,
                                 Real l) {
  // Compute BL metric components
  Real sigma = SQR(r) + SQR(pgen.spin) * (1.0 - SQR(sin_theta));
  Real g_00 = -1.0 + 2.0 * r / sigma;
  Real g_03 = -2.0 * pgen.spin * r / sigma * SQR(sin_theta);
  Real g_33 = (SQR(r) + SQR(pgen.spin) +
               2.0 * SQR(pgen.spin) * r / sigma * SQR(sin_theta)) *
              SQR(sin_theta);

  // Compute time component of covariant BL 4-velocity
  Real u_t = -sqrt(
      fmax((SQR(g_03) - g_00 * g_33) / (g_33 + 2.0 * l * g_03 + SQR(l) * g_00),
           0.0));
  return u_t;
}

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bhl_pgen pgen, Real x1, Real x2,
                                         Real x3, Real *pr, Real *ptheta,
                                         Real *pphi) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = fmax((sqrt(SQR(rad) - SQR(pgen.spin) +
                      sqrt(SQR(SQR(rad) - SQR(pgen.spin)) +
                           4.0 * SQR(pgen.spin) * SQR(x3))) /
                 sqrt(2.0)),
                1.0);
  *pr = r;
  *ptheta = (fabs(x3 / r) < 1.0) ? acos(x3 / r) : acos(copysign(1.0, x3));
  *pphi = atan2(r * x2 - pgen.spin * x1, pgen.spin * x2 + r * x1) -
          pgen.spin * r / (SQR(r) - 2.0 * r + SQR(pgen.spin));
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired
// coordinates Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist
//   coordinates x1,x2,x3: Cartesian Kerr-Schild coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired
//   coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct bhl_pgen pgen, Real a0_bl, Real a1_bl,
                            Real a2_bl, Real a3_bl, Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = fmax((sqrt(SQR(rad) - SQR(pgen.spin) +
                      sqrt(SQR(SQR(rad) - SQR(pgen.spin)) +
                           4.0 * SQR(pgen.spin) * SQR(x3))) /
                 sqrt(2.0)),
                1.0);
  Real delta = SQR(r) - 2.0 * r + SQR(pgen.spin);
  *pa0 = a0_bl + 2.0 * r / delta * a1_bl;
  *pa1 = a1_bl * ((r * x1 + pgen.spin * x2) / (SQR(r) + SQR(pgen.spin)) -
                  x2 * pgen.spin / delta) +
         a2_bl * x1 * x3 / r *
             sqrt((SQR(r) + SQR(pgen.spin)) / (SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ((r * x2 - pgen.spin * x1) / (SQR(r) + SQR(pgen.spin)) +
                  x1 * pgen.spin / delta) +
         a2_bl * x2 * x3 / r *
             sqrt((SQR(r) + SQR(pgen.spin)) / (SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3 / r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2)) / (SQR(r) + SQR(pgen.spin)));
  return;
}

// inlined spherical Kerr-Schild r evaluated at CKS x1, x2, x3
KOKKOS_INLINE_FUNCTION
Real KSRX(const Real x1, const Real x2, const Real x3, const Real a) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  return sqrt((SQR(rad) - SQR(a) +
               sqrt(SQR(SQR(rad) - SQR(a)) + 4.0 * SQR(a) * SQR(x3))) /
              2.0);
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn BhlAccretionBoundary
//  \brief Sets boundary condition on surfaces of computational domain

void BhlAccretionBoundary(Mesh *pm) {
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

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pm->pmb_pack->phydro != nullptr) {
    u0_ = pm->pmb_pack->phydro->u0;
    w0_ = pm->pmb_pack->phydro->w0;
  } else if (pm->pmb_pack->pmhd != nullptr) {
    u0_ = pm->pmb_pack->pmhd->u0;
    w0_ = pm->pmb_pack->pmhd->w0;
  }
  int nmb = pm->pmb_pack->nmb_thispack;
  int nvar = u0_.extent_int(1);

  // Determine if radiation is enabled
  const bool is_radiation_enabled = (pm->pmb_pack->prad != nullptr);
  DvceArray5D<Real> i0_;
  int nang1;
  if (is_radiation_enabled) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
  }

  // X1-Boundary
  // Set X1-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for(
        "noinflow_field_x1", DevExeSpace(), 0, (nmb - 1), 0, (n3 - 1), 0,
        (n2 - 1), KOKKOS_LAMBDA(int m, int k, int j) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
            for (int i = 0; i < ng; ++i) {
              b0.x1f(m, k, j, is - i - 1) = b0.x1f(m, k, j, is);
              b0.x2f(m, k, j, is - i - 1) = b0.x2f(m, k, j, is);
              if (j == n2 - 1) {
                b0.x2f(m, k, j + 1, is - i - 1) = b0.x2f(m, k, j + 1, is);
              }
              b0.x3f(m, k, j, is - i - 1) = b0.x3f(m, k, j, is);
              if (k == n3 - 1) {
                b0.x3f(m, k + 1, j, is - i - 1) = b0.x3f(m, k + 1, j, is);
              }
            }
          }
          // @YK : this loop sets the B field of incoming wind
          if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {

            for (int i = 0; i < ng; ++i) {
              b0.x1f(m, k, j, ie + i + 2) = 0.; // b0.x1f(m,k,j,ie+1);

              b0.x2f(m, k, j, ie + i + 1) = incoming_By; // b0.x2f(m,k,j,ie);
              if (j == n2 - 1) {
                b0.x2f(m, k, j + 1, ie + i + 1) = incoming_By;
              }; // b0.x2f(m,k,j+1,ie);}

              b0.x3f(m, k, j, ie + i + 1) = incoming_Bz;
              if (k == n3 - 1) {
                b0.x3f(m, k + 1, j, ie + i + 1) = incoming_Bz;
              }
            }
          }
        });
  }

  // ConsToPrim over all X1 ghost zones *and* at the innermost/outermost
  // X1-active zones of Meshblocks, even if Meshblock face is not at the edge of
  // computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, is - ng, is, 0,
                                           (n2 - 1), 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, ie, ie + ng, 0,
                                           (n2 - 1), 0, (n3 - 1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_, b0, w0_, bcc, false, is - ng, is,
                                         0, (n2 - 1), 0, (n3 - 1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_, b0, w0_, bcc, false, ie, ie + ng,
                                         0, (n2 - 1), 0, (n3 - 1));
  }

  // Set X1-BCs on w0 if Meshblock face is at the edge of computational domain
  // auto &bcc = pm->pmb_pack->pmhd->bcc0;

  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &coord = pm->pmb_pack->pcoord->coord_data;

  par_for(
      "noinflow_hydro_x1", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), 0,
      (n3 - 1), 0, (n2 - 1), KOKKOS_LAMBDA(int m, int n, int k, int j) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
          for (int i = 0; i < ng; ++i) {
            if (n == (IVX)) {
              w0_(m, n, k, j, is - i - 1) = fmin(0.0, w0_(m, n, k, j, is));
            } else {
              w0_(m, n, k, j, is - i - 1) = w0_(m, n, k, j, is);
            }
          }
        }
        // @YK : this loop sets the hydro variables
        if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
          // Real bx = 0.0;
          // const auto by = bcc(m, 1, k, j, ie + 1);
          // const auto bz = bcc(m, 2, k, j, ie + 1);
          // const auto magnetic_pressure = 0.5 * (SQR(by) + SQR(bz)) /
          // SQR(bhl.W_inf);

          for (int i = 0; i < ng; ++i) {
            if (n == IDN) {
              w0_(m, n, k, j, ie + i + 1) = bhl.rho_inf;
            } else if (n == IEN) {
              w0_(m, IEN, k, j, ie + i + 1) = bhl.e_inf;
            } else {
              Real &x1min = size.d_view(m).x1min;
              Real &x1max = size.d_view(m).x1max;
              Real x1v = CellCenterX(indcs.nx1 + i, indcs.nx1, x1min, x1max);

              Real &x2min = size.d_view(m).x2min;
              Real &x2max = size.d_view(m).x2max;
              Real x2v = CellCenterX(j - js, indcs.nx2, x2min, x2max);

              Real &x3min = size.d_view(m).x3min;
              Real &x3max = size.d_view(m).x3max;
              Real x3v = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

              Real glower[4][4], gupper[4][4];
              ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski,
                                      coord.bh_spin, glower, gupper);

              const Real alpha = 1.0 / sqrt(-gupper[0][0]);
              const Real beta1 = -gupper[0][1] / gupper[0][0];
              const Real beta2 = -gupper[0][2] / gupper[0][0];
              const Real beta3 = -gupper[0][3] / gupper[0][0];

              const Real u1_u0 = -bhl.v_inf;
              const Real u2_u0 = 0.0;
              const Real u3_u0 = 0.0;

              const Real v1 = (u1_u0 + beta1) / alpha;
              const Real v2 = (u2_u0 + beta2) / alpha;
              const Real v3 = (u3_u0 + beta3) / alpha;

              const Real q =
                  glower[1][1] * v1 * v1 + 2.0 * glower[1][2] * v1 * v2 +
                  2.0 * glower[1][3] * v1 * v3 + glower[2][2] * v2 * v2 +
                  2.0 * glower[2][3] * v2 * v3 + glower[3][3] * v3 * v3;

              const Real lorentz_W = 1.0 / sqrt(1.0 - q);

              const Real u1_prim = lorentz_W * v1;
              const Real u2_prim = lorentz_W * v2;
              const Real u3_prim = lorentz_W * v3;

              // w0_(m, n, k, j, ie + i + 1) = fmax(0.0, w0_(m, n, k, j, ie));
              w0_(m, IVX, k, j, ie + i + 1) = u1_prim;
              w0_(m, IVY, k, j, ie + i + 1) = u2_prim;
              w0_(m, IVZ, k, j, ie + i + 1) = u3_prim;
            }
          }
        }
      });
  if (is_radiation_enabled) {
    // Set X1-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for(
        "noinflow_rad_x1", DevExeSpace(), 0, (nmb - 1), 0, nang1, 0, (n3 - 1),
        0, (n2 - 1), KOKKOS_LAMBDA(int m, int n, int k, int j) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
            for (int i = 0; i < ng; ++i) {
              i0_(m, n, k, j, is - i - 1) = i0_(m, n, k, j, is);
            }
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
            for (int i = 0; i < ng; ++i) {
              i0_(m, n, k, j, ie + i + 1) = i0_(m, n, k, j, ie);
            }
          }
        });
  }

  // PrimToCons on X1 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, is - ng, is - 1, 0,
                                           (n2 - 1), 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, ie + 1, ie + ng, 0,
                                           (n2 - 1), 0, (n3 - 1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is - ng, is - 1, 0,
                                         (n2 - 1), 0, (n3 - 1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, ie + 1, ie + ng, 0,
                                         (n2 - 1), 0, (n3 - 1));
  }

  // X2-Boundary
  // Set X2-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for(
        "noinflow_field_x2", DevExeSpace(), 0, (nmb - 1), 0, (n3 - 1), 0,
        (n1 - 1), KOKKOS_LAMBDA(int m, int k, int i) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) {
            for (int j = 0; j < ng; ++j) {
              b0.x1f(m, k, js - j - 1, i) = b0.x1f(m, k, js, i);
              if (i == n1 - 1) {
                b0.x1f(m, k, js - j - 1, i + 1) = b0.x1f(m, k, js, i + 1);
              }
              b0.x2f(m, k, js - j - 1, i) = b0.x2f(m, k, js, i);
              b0.x3f(m, k, js - j - 1, i) = b0.x3f(m, k, js, i);
              if (k == n3 - 1) {
                b0.x3f(m, k + 1, js - j - 1, i) = b0.x3f(m, k + 1, js, i);
              }
            }
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
            for (int j = 0; j < ng; ++j) {
              b0.x1f(m, k, je + j + 1, i) = b0.x1f(m, k, je, i);
              if (i == n1 - 1) {
                b0.x1f(m, k, je + j + 1, i + 1) = b0.x1f(m, k, je, i + 1);
              }
              b0.x2f(m, k, je + j + 2, i) = b0.x2f(m, k, je + 1, i);
              b0.x3f(m, k, je + j + 1, i) = b0.x3f(m, k, je, i);
              if (k == n3 - 1) {
                b0.x3f(m, k + 1, je + j + 1, i) = b0.x3f(m, k + 1, je, i);
              }
            }
          }
        });
  }
  // ConsToPrim over all X2 ghost zones *and* at the innermost/outermost
  // X2-active zones of Meshblocks, even if Meshblock face is not at the edge of
  // computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, 0, (n1 - 1),
                                           js - ng, js, 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, 0, (n1 - 1), je,
                                           je + ng, 0, (n3 - 1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_, b0, w0_, bcc, false, 0, (n1 - 1),
                                         js - ng, js, 0, (n3 - 1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_, b0, w0_, bcc, false, 0, (n1 - 1),
                                         je, je + ng, 0, (n3 - 1));
  }
  // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for(
      "noinflow_hydro_x2", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), 0,
      (n3 - 1), 0, (n1 - 1), KOKKOS_LAMBDA(int m, int n, int k, int i) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) {
          for (int j = 0; j < ng; ++j) {
            if (n == (IVY)) {
              w0_(m, n, k, js - j - 1, i) = fmin(0.0, w0_(m, n, k, js, i));
            } else {
              w0_(m, n, k, js - j - 1, i) = w0_(m, n, k, js, i);
            }
          }
        }
        if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
          for (int j = 0; j < ng; ++j) {
            if (n == (IVY)) {
              w0_(m, n, k, je + j + 1, i) = fmax(0.0, w0_(m, n, k, je, i));
            } else {
              w0_(m, n, k, je + j + 1, i) = w0_(m, n, k, je, i);
            }
          }
        }
      });
  if (is_radiation_enabled) {
    // Set X2-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for(
        "noinflow_rad_x2", DevExeSpace(), 0, (nmb - 1), 0, nang1, 0, (n3 - 1),
        0, (n1 - 1), KOKKOS_LAMBDA(int m, int n, int k, int i) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) {
            for (int j = 0; j < ng; ++j) {
              i0_(m, n, k, js - j - 1, i) = i0_(m, n, k, js, i);
            }
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
            for (int j = 0; j < ng; ++j) {
              i0_(m, n, k, je + j + 1, i) = i0_(m, n, k, je, i);
            }
          }
        });
  }
  // PrimToCons on X2 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, 0, (n1 - 1), js - ng,
                                           js - 1, 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, 0, (n1 - 1), je + 1,
                                           je + ng, 0, (n3 - 1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, 0, (n1 - 1), js - ng,
                                         js - 1, 0, (n3 - 1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, 0, (n1 - 1), je + 1,
                                         je + ng, 0, (n3 - 1));
  }

  // X3-Boundary
  // Set X3-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for(
        "noinflow_field_x3", DevExeSpace(), 0, (nmb - 1), 0, (n2 - 1), 0,
        (n1 - 1), KOKKOS_LAMBDA(int m, int j, int i) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user) {
            for (int k = 0; k < ng; ++k) {
              b0.x1f(m, ks - k - 1, j, i) = b0.x1f(m, ks, j, i);
              if (i == n1 - 1) {
                b0.x1f(m, ks - k - 1, j, i + 1) = b0.x1f(m, ks, j, i + 1);
              }
              b0.x2f(m, ks - k - 1, j, i) = b0.x2f(m, ks, j, i);
              if (j == n2 - 1) {
                b0.x2f(m, ks - k - 1, j + 1, i) = b0.x2f(m, ks, j + 1, i);
              }
              b0.x3f(m, ks - k - 1, j, i) = b0.x3f(m, ks, j, i);
            }
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user) {
            for (int k = 0; k < ng; ++k) {
              b0.x1f(m, ke + k + 1, j, i) = b0.x1f(m, ke, j, i);
              if (i == n1 - 1) {
                b0.x1f(m, ke + k + 1, j, i + 1) = b0.x1f(m, ke, j, i + 1);
              }
              b0.x2f(m, ke + k + 1, j, i) = b0.x2f(m, ke, j, i);
              if (j == n2 - 1) {
                b0.x2f(m, ke + k + 1, j + 1, i) = b0.x2f(m, ke, j + 1, i);
              }
              b0.x3f(m, ke + k + 2, j, i) = b0.x3f(m, ke + 1, j, i);
            }
          }
        });
  }
  // ConsToPrim over all X3 ghost zones *and* at the innermost/outermost
  // X3-active zones of Meshblocks, even if Meshblock face is not at the edge of
  // computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, 0, (n1 - 1), 0,
                                           (n2 - 1), ks - ng, ks);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, 0, (n1 - 1), 0,
                                           (n2 - 1), ke, ke + ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_, b0, w0_, bcc, false, 0, (n1 - 1),
                                         0, (n2 - 1), ks - ng, ks);
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_, b0, w0_, bcc, false, 0, (n1 - 1),
                                         0, (n2 - 1), ke, ke + ng);
  }
  // Set X3-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for(
      "noinflow_hydro_x3", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), 0,
      (n2 - 1), 0, (n1 - 1), KOKKOS_LAMBDA(int m, int n, int j, int i) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user) {
          for (int k = 0; k < ng; ++k) {
            if (n == (IVZ)) {
              w0_(m, n, ks - k - 1, j, i) = fmin(0.0, w0_(m, n, ks, j, i));
            } else {
              w0_(m, n, ks - k - 1, j, i) = w0_(m, n, ks, j, i);
            }
          }
        }
        if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user) {
          for (int k = 0; k < ng; ++k) {
            if (n == (IVZ)) {
              w0_(m, n, ke + k + 1, j, i) = fmax(0.0, w0_(m, n, ke, j, i));
            } else {
              w0_(m, n, ke + k + 1, j, i) = w0_(m, n, ke, j, i);
            }
          }
        }
      });
  if (is_radiation_enabled) {
    // Set X3-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for(
        "noinflow_rad_x3", DevExeSpace(), 0, (nmb - 1), 0, nang1, 0, (n2 - 1),
        0, (n1 - 1), KOKKOS_LAMBDA(int m, int n, int j, int i) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user) {
            for (int k = 0; k < ng; ++k) {
              i0_(m, n, ks - k - 1, j, i) = i0_(m, n, ks, j, i);
            }
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user) {
            for (int k = 0; k < ng; ++k) {
              i0_(m, n, ke + k + 1, j, i) = i0_(m, n, ke, j, i);
            }
          }
        });
  }
  // PrimToCons on X3 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, 0, (n1 - 1), 0, (n2 - 1),
                                           ks - ng, ks - 1);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, 0, (n1 - 1), 0, (n2 - 1),
                                           ke + 1, ke + ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, 0, (n1 - 1), 0,
                                         (n2 - 1), ks - ng, ks - 1);
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, 0, (n1 - 1), 0,
                                         (n2 - 1), ke + 1, ke + ng);
  }

  return;
}

//----------------------------------------------------------------------------------------
// Contract the upper-triangular stress-energy tensor T^{mu nu} (mu<=nu stored)
// against the metric derivative: S = 0.5 (partial g_{mu nu}) T^{mu nu}.
// Used for GR momentum source terms in BhlAccretionHistory.

KOKKOS_INLINE_FUNCTION
Real ContractSourceTerm(const Real dg[][4], const Real T[][4]) {
  return 0.5 * dg[0][0] * T[0][0] + dg[0][1] * T[0][1] + dg[0][2] * T[0][2] +
         dg[0][3] * T[0][3] + 0.5 * dg[1][1] * T[1][1] + dg[1][2] * T[1][2] +
         dg[1][3] * T[1][3] + 0.5 * dg[2][2] * T[2][2] + dg[2][3] * T[2][3] +
         0.5 * dg[3][3] * T[3][3];
}

//----------------------------------------------------------------------------------------
// Function for computing accretion fluxes through constant spherical KS radius
// surfaces

void BhlAccretionHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  // extract BH parameters
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  // set nvars, adiabatic index, primitive array w0, and field array bcc0 if
  // is_mhd
  int nvars;
  Real eos_gamma;
  bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    eos_gamma = pmbp->phydro->peos->eos_data.gamma;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    eos_gamma = pmbp->pmhd->peos->eos_data.gamma;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }

  // extract grids, number of radii, number of fluxes, and history appending
  // index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  //
  // @YK : expand history for Magnus experiment
  //
  // Common (15):
  //    * Mdot, Edot, Ldot
  //    * Momentum flux T^r_i integrated over the sphere (3)
  //    * GR source terms (3)
  //    * Newtonian gravitational force (3)
  //    * Total T^0_i contained in the volume (3)
  //  - remark: if MHD, quantities above are *total* (T_hydro + T_EM)
  //
  // iff MHD (+12):
  //    * Magnetic flux through the sphere
  //    * (only EM part) Edot, Ldot
  //    * (only EM part) Momentum flux T^r_i integrated over the sphere (3)
  //    * (only EM part) GR source terms (3)
  //    * (only EM part) Total T^0_i contained in the volume (3)
  //
  int nflux = (is_mhd) ? 27 : 15;

  // set number of and names of history variables for hydro or mhd
  pdata->nhist = nradii * nflux;

  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int g = 0; g < nradii; ++g) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();

    pdata->label[nflux * g + 0] = "mdot_" + rad_str;
    pdata->label[nflux * g + 1] = "edot_" + rad_str;
    pdata->label[nflux * g + 2] = "ldot_" + rad_str;

    pdata->label[nflux * g + 3] = "Pdot_x_" + rad_str;
    pdata->label[nflux * g + 4] = "Pdot_y_" + rad_str;
    pdata->label[nflux * g + 5] = "Pdot_z_" + rad_str;

    pdata->label[nflux * g + 6] = "Gr_x_" + rad_str;
    pdata->label[nflux * g + 7] = "Gr_y_" + rad_str;
    pdata->label[nflux * g + 8] = "Gr_z_" + rad_str;

    pdata->label[nflux * g + 9] = "Nw_x_" + rad_str;
    pdata->label[nflux * g + 10] = "Nw_y_" + rad_str;
    pdata->label[nflux * g + 11] = "Nw_z_" + rad_str;

    pdata->label[nflux * g + 12] = "T^t_x_" + rad_str;
    pdata->label[nflux * g + 13] = "T^t_y_" + rad_str;
    pdata->label[nflux * g + 14] = "T^t_z_" + rad_str;

    if (is_mhd) {
      pdata->label[nflux * g + 15] = "Phi_" + rad_str;
      pdata->label[nflux * g + 16] = "edot_em_" + rad_str;
      pdata->label[nflux * g + 17] = "ldot_em_" + rad_str;

      pdata->label[nflux * g + 18] = "Pdot_em_x_" + rad_str;
      pdata->label[nflux * g + 19] = "Pdot_em_y_" + rad_str;
      pdata->label[nflux * g + 20] = "Pdot_em_z_" + rad_str;

      pdata->label[nflux * g + 21] = "Gr_em_x_" + rad_str;
      pdata->label[nflux * g + 22] = "Gr_em_y_" + rad_str;
      pdata->label[nflux * g + 23] = "Gr_em_z_" + rad_str;

      pdata->label[nflux * g + 24] = "Tem^t_x_" + rad_str;
      pdata->label[nflux * g + 25] = "Tem^t_y_" + rad_str;
      pdata->label[nflux * g + 26] = "Tem^t_z_" + rad_str;
    }
  }

  // Extract global quantities for each radii
  DualArray2D<Real> interpolated_bcc; // needed for MHD

  for (int g = 0; g < nradii; ++g) {
    // initialize to zero
    for (int i = 0; i < nflux; ++i) {
      pdata->hdata[nflux * g + i] = 0.0;
    }

    // interpolate primitives (and cell-centered magnetic fields iff mhd)
    if (is_mhd) {
      grids[g]->InterpolateToSphere(3, bcc0_);
      Kokkos::realloc(interpolated_bcc, grids[g]->nangles, 3);
      Kokkos::deep_copy(interpolated_bcc, grids[g]->interp_vals);
      interpolated_bcc.template modify<DevExeSpace>();
      interpolated_bcc.template sync<HostMemSpace>();
    }
    grids[g]->InterpolateToSphere(nvars, w0_);

    //
    // compute surface integrals
    //
    for (int n = 0; n < grids[g]->nangles; ++n) {
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
      if (is_mhd) {
        int_bx = interpolated_bcc.h_view(n, IBX);
        int_by = interpolated_bcc.h_view(n, IBY);
        int_bz = interpolated_bcc.h_view(n, IBZ);
      }

      // Compute interpolated u^\mu in CKS
      Real q = glower[1][1] * int_vx * int_vx +
               2.0 * glower[1][2] * int_vx * int_vy +
               2.0 * glower[1][3] * int_vx * int_vz +
               glower[2][2] * int_vy * int_vy +
               2.0 * glower[2][3] * int_vy * int_vz +
               glower[3][3] * int_vz * int_vz;
      Real alpha = sqrt(-1.0 / gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor / alpha;
      Real u1 = int_vx - alpha * lor * gupper[0][1];
      Real u2 = int_vy - alpha * lor * gupper[0][2];
      Real u3 = int_vz - alpha * lor * gupper[0][3];

      // Lower vector indices
      Real u_0 = glower[0][0] * u0 + glower[0][1] * u1 + glower[0][2] * u2 +
                 glower[0][3] * u3;
      Real u_1 = glower[1][0] * u0 + glower[1][1] * u1 + glower[1][2] * u2 +
                 glower[1][3] * u3;
      Real u_2 = glower[2][0] * u0 + glower[2][1] * u1 + glower[2][2] * u2 +
                 glower[2][3] * u3;
      Real u_3 = glower[3][0] * u0 + glower[3][1] * u1 + glower[3][2] * u2 +
                 glower[3][3] * u3;

      // Calculate 4-magnetic field (returns zero if not MHD)
      Real b0 = u_1 * int_bx + u_2 * int_by + u_3 * int_bz;
      Real b1 = (int_bx + b0 * u1) / u0;
      Real b2 = (int_by + b0 * u2) / u0;
      Real b3 = (int_bz + b0 * u3) / u0;

      // compute b_\mu in CKS and b_sq (returns zero if not MHD)
      Real b_0 = glower[0][0] * b0 + glower[0][1] * b1 + glower[0][2] * b2 +
                 glower[0][3] * b3;
      Real b_1 = glower[1][0] * b0 + glower[1][1] * b1 + glower[1][2] * b2 +
                 glower[1][3] * b3;
      Real b_2 = glower[2][0] * b0 + glower[2][1] * b1 + glower[2][2] * b2 +
                 glower[2][3] * b3;
      Real b_3 = glower[3][0] * b0 + glower[3][1] * b1 + glower[3][2] * b2 +
                 glower[3][3] * b3;
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
      Real u_ph = (-r * sph - spin * cph) * sth * u_1 +
                  (r * cph - spin * sph) * sth * u_2;
      // covariant phi component of 4-magnetic field (returns zero if not MHD)
      Real b_ph = (-r * sph - spin * cph) * sth * b_1 +
                  (r * cph - spin * sph) * sth * b_2;

      // integration params
      Real &domega = grids[g]->solid_angles.h_view(n);
      Real sqrtmdet = (r2 + SQR(spin * cos(theta)));

      // temporary variables
      Real wtot = int_dn + eos_gamma * int_ie + b_sq;
      Real ptot = (eos_gamma - 1.0) * int_ie + 0.5 * b_sq;

      // mass accretion rate (note the minus sign)
      pdata->hdata[nflux * g + 0] += -1.0 * int_dn * ur * sqrtmdet * domega;

      // total energy (T^0_0) outflux. note the sign
      Real tr_0 = wtot * ur * u_0 - br * b_0;
      pdata->hdata[nflux * g + 1] += tr_0 * sqrtmdet * domega;

      // angular momentum flux (T^r_phi)
      Real tr_ph = wtot * ur * u_ph - br * b_ph;
      pdata->hdata[nflux * g + 2] += tr_ph * sqrtmdet * domega;

      // linear momentum flux (T^r_i)
      Real tr_1 = (wtot * ur * u_1) + (ptot * drdx) - (br * b_1);
      Real tr_2 = (wtot * ur * u_2) + (ptot * drdy) - (br * b_2);
      Real tr_3 = (wtot * ur * u_3) + (ptot * drdz) - (br * b_3);
      pdata->hdata[nflux * g + 3] += tr_1 * sqrtmdet * domega;
      pdata->hdata[nflux * g + 4] += tr_2 * sqrtmdet * domega;
      pdata->hdata[nflux * g + 5] += tr_3 * sqrtmdet * domega;

      if (is_mhd) {
        // magnetic flux
        pdata->hdata[nflux * g + 15] +=
            0.5 * fabs(br * u0 - b0 * ur) * sqrtmdet * domega;
        // Edot (EM part only)
        pdata->hdata[nflux * g + 16] +=
            (b_sq * ur * u_0 - br * b_0) * sqrtmdet * domega;
        // Ldot (EM part only)
        pdata->hdata[nflux * g + 17] +=
            (b_sq * ur * u_ph - br * b_ph) * sqrtmdet * domega;

        // linear momentum flux (T^r_i), (EM part only)
        Real tEMr_1 = (b_sq * ur * u_1) + (0.5 * b_sq * drdx) - (br * b_1);
        Real tEMr_2 = (b_sq * ur * u_2) + (0.5 * b_sq * drdy) - (br * b_2);
        Real tEMr_3 = (b_sq * ur * u_3) + (0.5 * b_sq * drdz) - (br * b_3);
        pdata->hdata[nflux * g + 18] += tEMr_1 * sqrtmdet * domega;
        pdata->hdata[nflux * g + 19] += tEMr_2 * sqrtmdet * domega;
        pdata->hdata[nflux * g + 20] += tEMr_3 * sqrtmdet * domega;
      }
    }

  } // end per-radius surface integral loop

  //
  // compute volume integrals in a single pass over all mesh cells
  //

  const int nvol = 15; // accumulator slots per radius

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  const int nmkji = (pmbp->nmb_thispack) * indcs.nx3 * indcs.nx2 * indcs.nx1;
  const int nkji = indcs.nx3 * indcs.nx2 * indcs.nx1;
  const int nji = indcs.nx2 * indcs.nx1;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto bhl = bhl_accretion;

  // copy shell inner radii to a device-accessible view
  Kokkos::View<Real *> d_radii("vol_radii", nradii);
  {
    auto h_radii = Kokkos::create_mirror_view(d_radii);
    for (int g = 0; g < nradii; ++g)
      h_radii(g) = grids[g]->radius;
    Kokkos::deep_copy(d_radii, h_radii);
  }

  // @YK : minimum shell boundary = inner edge of masked (excised) region
  Real r_outer = 1.0 + sqrt(1.0 - SQR(bhl.spin));
  // Real min_radius_all = grids[0]->radius;
  // for (int g = 1; g < nradii; ++g)
  //   min_radius_all = std::min(min_radius_all, grids[g]->radius);

  // Use a device View + atomic adds instead of parallel_reduce to avoid
  // exceeding CUDA L0 scratch memory limits (GlobalSum is 200*8=1600 bytes,
  // but CUDA limits reducer scratch to ~512 bytes).
  Kokkos::View<Real **> d_sum("vol_sums", nradii, nvol);
  Kokkos::deep_copy(d_sum, 0.0);
  Kokkos::parallel_for(
      "bhl_vol_integral", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx) {
        // compute m,k,j,i indices of thread and call function
        int m = (idx) / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / indcs.nx1;
        int i = (idx - m * nkji - k * nji - j * indcs.nx1) + is;
        k += ks;
        j += js;

        // Volume & Cell-centered coordinates
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

        // coordinate
        const Real r_ks = KSRX(x1v, x2v, x3v, bhl.spin);

        // skip cells within the outer horizon
        if (r_ks <= r_outer)
          return;

        // Below are copy from coordinate source terms (coordinates.cpp)

        // ------------------------------------
        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

        // compute derivatives of metric.
        Real dg_dx1[4][4], dg_dx2[4][4], dg_dx3[4][4];
        ComputeMetricDerivatives(x1v, x2v, x3v, flat, spin, dg_dx1, dg_dx2,
                                 dg_dx3);

        // Extract primitives
        const Real &rho = w0_(m, IDN, k, j, i);
        const Real &uu1 = w0_(m, IVX, k, j, i);
        const Real &uu2 = w0_(m, IVY, k, j, i);
        const Real &uu3 = w0_(m, IVZ, k, j, i);
        const Real &eint = w0_(m, IEN, k, j, i);
        // Real pgas = eos.IdealGasPressure(prim(m,IEN,k,j,i));

        // Calculate 4-velocity
        Real uu_sq = glower[1][1] * uu1 * uu1 + 2.0 * glower[1][2] * uu1 * uu2 +
                     2.0 * glower[1][3] * uu1 * uu3 + glower[2][2] * uu2 * uu2 +
                     2.0 * glower[2][3] * uu2 * uu3 + glower[3][3] * uu3 * uu3;
        Real alpha = sqrt(-1.0 / gupper[0][0]);
        Real gamma = sqrt(1.0 + uu_sq);
        Real u0 = gamma / alpha;
        Real u1 = uu1 - alpha * gamma * gupper[0][1];
        Real u2 = uu2 - alpha * gamma * gupper[0][2];
        Real u3 = uu3 - alpha * gamma * gupper[0][3];

        // lower vector indices
        Real u_1 = glower[1][0] * u0 + glower[1][1] * u1 + glower[1][2] * u2 +
                   glower[1][3] * u3;
        Real u_2 = glower[2][0] * u0 + glower[2][1] * u1 + glower[2][2] * u2 +
                   glower[2][3] * u3;
        Real u_3 = glower[3][0] * u0 + glower[3][1] * u1 + glower[3][2] * u2 +
                   glower[3][3] * u3;

        // calculate 4-magnetic field (returns zero if not MHD)
        Real bb1 = 0.0, bb2 = 0.0, bb3 = 0.0;
        if (is_mhd) {
          bb1 = bcc0_(m, IBX, k, j, i);
          bb2 = bcc0_(m, IBY, k, j, i);
          bb3 = bcc0_(m, IBZ, k, j, i);
        }
        Real b0 = u_1 * bb1 + u_2 * bb2 + u_3 * bb3;
        Real b1 = (bb1 + b0 * u1) / u0;
        Real b2 = (bb2 + b0 * u2) / u0;
        Real b3 = (bb3 + b0 * u3) / u0;

        // lower vector indices (returns zero if not MHD)
        Real b_0 = glower[0][0] * b0 + glower[0][1] * b1 + glower[0][2] * b2 +
                   glower[0][3] * b3;
        Real b_1 = glower[1][0] * b0 + glower[1][1] * b1 + glower[1][2] * b2 +
                   glower[1][3] * b3;
        Real b_2 = glower[2][0] * b0 + glower[2][1] * b1 + glower[2][2] * b2 +
                   glower[2][3] * b3;
        Real b_3 = glower[3][0] * b0 + glower[3][1] * b1 + glower[3][2] * b2 +
                   glower[3][3] * b3;
        Real b_sq = b_0 * b0 + b_1 * b1 + b_2 * b2 + b_3 * b3;

        // Calculate stress-energy tensor
        Real wtot = rho + eos_gamma * eint + b_sq;
        Real ptot = (eos_gamma - 1.0) * eint + 0.5 * b_sq;
        Real tt[4][4];
        tt[0][0] = wtot * u0 * u0 + ptot * gupper[0][0] - b0 * b0;
        tt[0][1] = wtot * u0 * u1 + ptot * gupper[0][1] - b0 * b1;
        tt[0][2] = wtot * u0 * u2 + ptot * gupper[0][2] - b0 * b2;
        tt[0][3] = wtot * u0 * u3 + ptot * gupper[0][3] - b0 * b3;
        tt[1][1] = wtot * u1 * u1 + ptot * gupper[1][1] - b1 * b1;
        tt[1][2] = wtot * u1 * u2 + ptot * gupper[1][2] - b1 * b2;
        tt[1][3] = wtot * u1 * u3 + ptot * gupper[1][3] - b1 * b3;
        tt[2][2] = wtot * u2 * u2 + ptot * gupper[2][2] - b2 * b2;
        tt[2][3] = wtot * u2 * u3 + ptot * gupper[2][3] - b2 * b3;
        tt[3][3] = wtot * u3 * u3 + ptot * gupper[3][3] - b3 * b3;

        // GR source terms: S_k = 0.5 (partial_k g_{mu nu}) T^{mu nu}
        Real s_1 = ContractSourceTerm(dg_dx1, tt);
        Real s_2 = ContractSourceTerm(dg_dx2, tt);
        Real s_3 = ContractSourceTerm(dg_dx3, tt);

        // EM stress-energy tensor and GR source terms (MHD only)
        Real sem_1 = 0.0, sem_2 = 0.0, sem_3 = 0.0;
        if (is_mhd) {
          Real ttem[4][4];
          ttem[0][0] = b_sq * u0 * u0 + 0.5 * b_sq * gupper[0][0] - b0 * b0;
          ttem[0][1] = b_sq * u0 * u1 + 0.5 * b_sq * gupper[0][1] - b0 * b1;
          ttem[0][2] = b_sq * u0 * u2 + 0.5 * b_sq * gupper[0][2] - b0 * b2;
          ttem[0][3] = b_sq * u0 * u3 + 0.5 * b_sq * gupper[0][3] - b0 * b3;
          ttem[1][1] = b_sq * u1 * u1 + 0.5 * b_sq * gupper[1][1] - b1 * b1;
          ttem[1][2] = b_sq * u1 * u2 + 0.5 * b_sq * gupper[1][2] - b1 * b2;
          ttem[1][3] = b_sq * u1 * u3 + 0.5 * b_sq * gupper[1][3] - b1 * b3;
          ttem[2][2] = b_sq * u2 * u2 + 0.5 * b_sq * gupper[2][2] - b2 * b2;
          ttem[2][3] = b_sq * u2 * u3 + 0.5 * b_sq * gupper[2][3] - b2 * b3;
          ttem[3][3] = b_sq * u3 * u3 + 0.5 * b_sq * gupper[3][3] - b3 * b3;
          sem_1 = ContractSourceTerm(dg_dx1, ttem);
          sem_2 = ContractSourceTerm(dg_dx2, ttem);
          sem_3 = ContractSourceTerm(dg_dx3, ttem);
        }

        // Gravitational force using approximate Newtonian formula
        const Real r2 = SQR(x1v) + SQR(x2v) + SQR(x3v);
        const Real one_over_r3 = 1.0 / (r2 * sqrt(r2));

        // accumulate into each shell that contains this cell
        for (int g = 0; g < nradii; ++g) {
          if (r_ks < d_radii(g)) {
            // GR source terms
            Kokkos::atomic_add(&d_sum(g, 0), vol * s_1);
            Kokkos::atomic_add(&d_sum(g, 1), vol * s_2);
            Kokkos::atomic_add(&d_sum(g, 2), vol * s_3);

            Kokkos::atomic_add(&d_sum(g, 3), vol * rho * x1v * one_over_r3);
            Kokkos::atomic_add(&d_sum(g, 4), vol * rho * x2v * one_over_r3);
            Kokkos::atomic_add(&d_sum(g, 5), vol * rho * x3v * one_over_r3);

            // T^0_i
            Kokkos::atomic_add(&d_sum(g, 6), vol * (wtot * u0 * u_1 - b0 * b_1));
            Kokkos::atomic_add(&d_sum(g, 7), vol * (wtot * u0 * u_2 - b0 * b_2));
            Kokkos::atomic_add(&d_sum(g, 8), vol * (wtot * u0 * u_3 - b0 * b_3));

            if (is_mhd) {
              // GR source terms (EM part only)
              Kokkos::atomic_add(&d_sum(g, 9),  vol * sem_1);
              Kokkos::atomic_add(&d_sum(g, 10), vol * sem_2);
              Kokkos::atomic_add(&d_sum(g, 11), vol * sem_3);

              // T^0_i (EM part only)
              Kokkos::atomic_add(&d_sum(g, 12), vol * (b_sq * u0 * u_1 - b0 * b_1));
              Kokkos::atomic_add(&d_sum(g, 13), vol * (b_sq * u0 * u_2 - b0 * b_2));
              Kokkos::atomic_add(&d_sum(g, 14), vol * (b_sq * u0 * u_3 - b0 * b_3));
            }
          }
        }
      });
  Kokkos::fence();

  // copy results back to host
  auto h_sum = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_sum);

  // store data into hdata array
  for (int g = 0; g < nradii; ++g) {
    pdata->hdata[nflux * g + 6]  += h_sum(g, 0);
    pdata->hdata[nflux * g + 7]  += h_sum(g, 1);
    pdata->hdata[nflux * g + 8]  += h_sum(g, 2);
    pdata->hdata[nflux * g + 9]  += h_sum(g, 3);
    pdata->hdata[nflux * g + 10] += h_sum(g, 4);
    pdata->hdata[nflux * g + 11] += h_sum(g, 5);
    pdata->hdata[nflux * g + 12] += h_sum(g, 6);
    pdata->hdata[nflux * g + 13] += h_sum(g, 7);
    pdata->hdata[nflux * g + 14] += h_sum(g, 8);
    if (is_mhd) {
      pdata->hdata[nflux * g + 21] += h_sum(g, 9);
      pdata->hdata[nflux * g + 22] += h_sum(g, 10);
      pdata->hdata[nflux * g + 23] += h_sum(g, 11);
      pdata->hdata[nflux * g + 24] += h_sum(g, 12);
      pdata->hdata[nflux * g + 25] += h_sum(g, 13);
      pdata->hdata[nflux * g + 26] += h_sum(g, 14);
    }
  }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}
