#ifndef SRCTERMS_SRCTERMS_HPP_
#define SRCTERMS_SRCTERMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.hpp
//! \brief Data, functions, and classes to implement various source terms in the hydro
//! and/or MHD equations of motion.  Currently implemented:
//!  (1) constant (gravitational) acceleration - for RTI
//!  (2) shearing box in 2D (x-z), for both hydro and MHD
//!  (3) random forcing to drive turbulence - implemented in TurbulenceDriver class

#include <cmath>
#include <map>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \fn SinkIntPow
//! \brief x^n for integer n >= 0, by binary exponentiation. The sink kernel
//! exponent b is always an integer, and this is appreciably cheaper than
//! std::pow() with a floating-point exponent.

KOKKOS_INLINE_FUNCTION
Real SinkIntPow(Real x, int n) {
  Real result = 1.0;
  while (n > 0) {
    if (n & 1) {
      result *= x;
    }
    x *= x;
    n >>= 1;
  }
  return result;
}

//----------------------------------------------------------------------------------------
//! \fn SinkWindow
//! \brief the gas-removing sink's window function s(r) = exp(-(r/r_s)^b) of
//! Dittmann & Ryan (2022). Defined here rather than in srcterms.cpp so that
//! diagnostics computed elsewhere (e.g. the removal rates in a problem
//! generator's history function) use the same kernel as the source term itself
//! instead of a copy that can drift out of step with it.

KOKKOS_INLINE_FUNCTION
Real SinkWindow(Real radius, Real inv_r_s, int b) {
  return std::exp(-SinkIntPow(radius*inv_r_s, b));
}

//----------------------------------------------------------------------------------------
//! \class SourceTerms
//! \brief data and functions for physical source terms

class SourceTerms {
 public:
  SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~SourceTerms();

  // data
  // flags for various source terms
  bool const_accel;
  bool ism_cooling;
  bool rel_cooling;
  bool rad_beam;
  bool self_gravity;

  // @YK
  bool point_particle_gravity_at_center;
  bool gas_removing_sink;

  // new timestep
  Real dtnew;

  // data for constant accel
  Real const_accel_val;   // magnitude of accn
  int const_accel_dir;    // direction of accn

  // data for ISM cooling
  Real hrate;

  // data for relativistic cooling
  Real crate_rel;
  Real cpower_rel;

  // data for radiation beam source
  Real dii_dt;            // injection rate
  Real pos1, pos2, pos3;  // position of source
  Real dir1, dir2, dir3;  // direction of source
  Real width, spread;     // spatial width of source region, spread in angles

  // @YK: data for point-particle gravity
  Real softening_length;

  // @YK: data for the gas-removing sink at the coordinate origin
  Real sink_radius;    // r_s, characteristic radius of the removal kernel
  Real sink_rate;      // gamma, removal rate in units of Omega_K(r_s)
  int sink_kernel_b;   // b in s = exp(-(r/r_s)^b), always an integer
  // derived from the above in the constructor
  Real sink_removal_rate;  // gamma * Omega_K(r_s), the actual removal rate
  Real sink_r_cut;         // radius beyond which the kernel underflows to zero

  // @YK: running totals of what the sink has taken out of the domain, accumulated
  // inside GasRemovingSink with the integrator's per-stage source weight (see
  // Driver::src_wt). Element 0 is the removed mass; elements 1-3 are the angular
  // momentum about the origin that the removed mass was carrying.
  //
  // NOTE the torque-free sink removes NO angular momentum: the momentum it takes is
  // purely radial, so r x dp = 0 identically and the domain's total angular momentum
  // is untouched. Elements 1-3 are therefore the angular momentum that a *standard*
  // sink would have removed and that this prescription instead leaves behind in the
  // surviving gas, i.e. the torque the torque-free prescription applies to the disk.
  //
  // These are per-rank partial sums: each rank accumulates only its own MeshBlocks,
  // and the sum over ranks is taken by the MPI_Reduce in HistoryOutput::WriteOutputFile.
  static constexpr int nsink_accum = 4;
  DvceArray1D<Real> sink_accum;
  std::string sink_block;  // input block name, for writing the totals to restarts

  // functions
  void ApplySrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                     const Real bdt, const Real acc_dt, DvceArray5D<Real> &u0);
  void ApplySrcTerms(DvceArray5D<Real> &i0, const Real bdt);
  void ConstantAccel(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                     const Real bdt, DvceArray5D<Real> &u0);
  void ISMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real bdt, DvceArray5D<Real> &u0);
  void RelCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real bdt, DvceArray5D<Real> &u0);
  void SelfGravity(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                   const Real bdt, DvceArray5D<Real> &u0);
  void BeamSource(DvceArray5D<Real> &i0, const Real bdt);
  void NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos);

  void PointParticleGravity(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                            const Real bdt, DvceArray5D<Real> &u0);
  void GasRemovingSink(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                       const Real bdt, const Real acc_dt, DvceArray5D<Real> &u0);
  void SinkAccumOnHost(Real *vals);
  void StoreSinkAccumInInput(ParameterInput *pin);

private:
  MeshBlockPack *pmy_pack;
};

#endif  // SRCTERMS_SRCTERMS_HPP_
