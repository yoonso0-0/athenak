#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.hpp
//  \brief definitions for Hydro class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "diffusion/sts_types.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class Viscosity;
class Conduction;
class SourceTerms;
class OrbitalAdvectionCC;
class ShearingBoxCC;
class Driver;

// constants that enumerate Hydro Riemann Solver options
enum class Hydro_RSolver {advect, llf, hlle, hllc, roe,    // non-relativistic
                          llf_sr, hlle_sr, hllc_sr,        // SR
                          llf_gr, hlle_gr};                // GR

//----------------------------------------------------------------------------------------
//! \struct HydroTaskIDs
//  \brief container to hold TaskIDs of all hydro tasks

struct HydroTaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID rkupdt;
  TaskID srctrms;
  TaskID c2p_mask;
  TaskID masksphere_pre;
  TaskID sendu_oa;
  TaskID recvu_oa;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID sendu_shr;
  TaskID recvu_shr;
  TaskID bcs;
  TaskID prol;
  TaskID c2p;
  TaskID masksphere;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
};

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class Hydro

class Hydro {
 public:
  Hydro(MeshBlockPack *ppack, ParameterInput *pin);
  ~Hydro();

  // data
  ReconstructionMethod recon_method;
  Hydro_RSolver rsolver_method;
  EquationOfState *peos;  // chosen EOS

  int nhydro;             // number of hydro variables (5/4 for ideal/isothermal EOS)
  int nscalars;           // number of passive scalars
  DvceArray5D<Real> u0;   // conserved variables
  DvceArray5D<Real> w0;   // primitive variables

  DvceArray5D<Real> coarse_u0;  // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceArray5D<Real> coarse_w0;  // primitive variables on 2x coarser grid (for SMR/AMR)

  // Boundary communication buffers and functions for u
  MeshBoundaryValuesCC *pbval_u;

  // Orbital advection and shearing box BCs
  OrbitalAdvectionCC *porb_u = nullptr;
  ShearingBoxCC *psbox_u = nullptr;

  // Object(s) for extra physics (viscosity, thermal conduction, srcterms)
  Viscosity *pvisc = nullptr;
  Conduction *pcond = nullptr;
  SourceTerms *psrc = nullptr;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables at intermediate step
  DvceArray5D<Real> u_sts0;   // conserved variables at start of STS sweep
  DvceArray5D<Real> u_sts1;   // previous STS stage state
  DvceArray5D<Real> u_sts2;   // second previous STS stage state
  DvceArray5D<Real> u_sts_rhs;  // cached first-stage RKL2 operator contribution
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  Real dtnew;

  bool has_explicit_viscosity = false;
  bool has_explicit_conduction = false;
  bool has_sts_viscosity = false;
  bool has_sts_conduction = false;
  bool has_any_sts_diffusion = false;

  // Global per-face L/R primitive buffers used by the split-kernel flux path
  // (PLM + LLF|HLLC). Shaped (nmb, nvars, nf3, nf2, nf1) where
  //   nf1 = nx1+1,
  //   nf2 = (nx2>1) ? nx2+1 : 1,
  //   nf3 = (nx3>1) ? nx3+1 : 1.
  // The buffer is sized to the max active face range across all three directions,
  // and reused sequentially per direction.  Indexing convention is
  //   wl(m, n, k-ks, j-js, i-is)
  // where for direction d the face-normal axis is face-indexed and the two
  // transverse axes are cell-indexed (origin-shifted by ks/js/is).
  DvceArray5D<Real> wl3d;
  DvceArray5D<Real> wr3d;

  // following used for FOFC
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  bool use_fofc = false;   // flag to enable FOFC
  DvceArray5D<Real> utest;  // scratch array for FOFC

  // following used for the spherical inner mask (effective BC on an internal sphere)
  // dirichlet:  fixed state (dens,vel1,vel2,vel3,pres), overwritten verbatim.
  // reflecting: exterior state mirrored through r=radius, radial velocity flipped.
  // absorbing:  same mirrored density/pressure as reflecting, velocity set to zero.
  // spherical_wind: fixed (dens,eint) as with dirichlet, but velocity is prescribed as
  //             a radial field of fixed magnitude vel_r (sign: >0 outward/wind,
  //             <0 inward/accretion) rather than a fixed Cartesian vector.
  enum class SphereMaskBC {dirichlet, reflecting, absorbing, spherical_wind};
  bool use_sphere_mask = false;    // flag to enable the spherical inner mask
  Real sphere_mask_radius = 0.0;   // radius R of the masked sphere, centered at origin
  SphereMaskBC sphere_mask_bc;     // which effective BC to emulate at r=R
  // Dirichlet/spherical_wind target state; eint (not pressure) is specified directly
  Real sm_dens, sm_vel1, sm_vel2, sm_vel3, sm_eint;
  Real sm_velr = 0.0;              // spherical_wind: prescribed radial velocity magnitude
  // true for Reflecting/Absorbing, which interpolate the exterior state at a mirror
  // point; false for the pointwise Dirichlet/Spherical Wind, which read no neighbours.
  // Set once by InitSphereMask and used to pick which of the two mask passes runs.
  bool sm_needs_mirror = false;
  // local pack indices of every MeshBlock on this rank holding masked cells (r<radius).
  // With the origin strictly inside one block this is that single block; when block
  // boundaries meet at the origin -- as they always do for a 2^N root grid on a domain
  // symmetric about 0 -- the sphere is cut into 2/4/8 pieces and each is masked by the
  // block owning it. Zero-length on ranks holding no part of the sphere.
  int nmb_mask = 0;
  DualArray1D<int> sphere_mask_mbs;

  // container to hold names of TaskIDs
  HydroTaskIDs id;

  // functions...
  void AssembleHydroTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_stagen_tl" list
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus InitRecvParabolic(Driver *d, int stage);
  // ...in "stagen_tl" list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  TaskStatus HydroSrcTerms(Driver *d, int stage);
  TaskStatus SendU_OA(Driver *d, int stage);
  TaskStatus RecvU_OA(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendU_Shr(Driver *d, int stage);
  TaskStatus RecvU_Shr(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus ConToPrimMask(Driver *d, int stage);
  TaskStatus MaskSphere(Driver *d, int stage);
  TaskStatus MaskSpherePre(Driver *d, int stage);
  // shared body of MaskSpherePre (pre=true) and MaskSphere (pre=false). Public because
  // nvcc rejects a KOKKOS_LAMBDA in a private or protected member function.
  TaskStatus MaskSphereImpl(bool pre);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus ClearSTSFlux(Driver *d, int stage);
  TaskStatus STSFluxes(Driver *d, int stage);
  TaskStatus STSUpdate(Driver *d, int stage);
  TaskStatus STSRefreshTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);  // also in Driver::Initialize

  // CalculateFluxes function templated over Riemann Solvers
  template <Hydro_RSolver T>
  void CalculateFluxes(Driver *d, int stage);

  // first-order flux correction
  void FOFC(Driver *d, int stage);

 private:
  void AddSelectedDiffusionFluxes(parabolic::DiffusionSelection selection);
  // parses <sphere_mask> input block and validates mesh/MeshBlock layout
  void InitSphereMask(ParameterInput *pin);
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
