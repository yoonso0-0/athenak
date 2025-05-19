//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_tasks.cpp
//! \brief functions that control MHD tasks stored in tasklists in MeshBlockPack

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box/shearing_box.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

#include "coordinates/cartesian_ks.hpp"
#include "eos/ideal_c2p_mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void MHD::AssembleMHDTasks
//! \brief Adds mhd tasks to appropriate task lists used by time integrators.
//! Called by MeshBlockPack::AddPhysics() function directly after MHD constructor
//! See comments Hydro::AssembleHydroTasks() function for more details.

void MHD::AssembleMHDTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  // assemble "before_timeintegrator" task list
  id.savest = tl["before_timeintegrator"]->AddTask(&MHD::SaveMHDState, this, none);

  // assemble "before_stagen" task list
  id.irecv = tl["before_stagen"]->AddTask(&MHD::InitRecv, this, none);

  // assemble "stagen" task list
  id.copyu     = tl["stagen"]->AddTask(&MHD::CopyCons, this, none);
  id.flux      = tl["stagen"]->AddTask(&MHD::Fluxes, this, id.copyu);
  id.sendf     = tl["stagen"]->AddTask(&MHD::SendFlux, this, id.flux);
  id.recvf     = tl["stagen"]->AddTask(&MHD::RecvFlux, this, id.sendf);
  id.rkupdt    = tl["stagen"]->AddTask(&MHD::RKUpdate, this, id.recvf);
  id.srctrms   = tl["stagen"]->AddTask(&MHD::MHDSrcTerms, this, id.rkupdt);
  id.sendu_oa  = tl["stagen"]->AddTask(&MHD::SendU_OA, this, id.srctrms);
  id.recvu_oa  = tl["stagen"]->AddTask(&MHD::RecvU_OA, this, id.sendu_oa);
  id.restu     = tl["stagen"]->AddTask(&MHD::RestrictU, this, id.recvu_oa);
  id.sendu     = tl["stagen"]->AddTask(&MHD::SendU, this, id.restu);
  id.recvu     = tl["stagen"]->AddTask(&MHD::RecvU, this, id.sendu);
  id.sendu_shr = tl["stagen"]->AddTask(&MHD::SendU_Shr, this, id.recvu);
  id.recvu_shr = tl["stagen"]->AddTask(&MHD::RecvU_Shr, this, id.sendu_shr);
  id.efld      = tl["stagen"]->AddTask(&MHD::CornerE, this, id.recvu_shr);
  id.efldsrc   = tl["stagen"]->AddTask(&MHD::EFieldSrc, this, id.efld);
  id.sende     = tl["stagen"]->AddTask(&MHD::SendE, this, id.efldsrc);
  id.recve     = tl["stagen"]->AddTask(&MHD::RecvE, this, id.sende);
  id.ct        = tl["stagen"]->AddTask(&MHD::CT, this, id.recve);
  id.sendb_oa  = tl["stagen"]->AddTask(&MHD::SendB_OA, this, id.ct);
  id.recvb_oa  = tl["stagen"]->AddTask(&MHD::RecvB_OA, this, id.sendb_oa);
  id.restb     = tl["stagen"]->AddTask(&MHD::RestrictB, this, id.recvb_oa);
  id.sendb     = tl["stagen"]->AddTask(&MHD::SendB, this, id.restb);
  id.recvb     = tl["stagen"]->AddTask(&MHD::RecvB, this, id.sendb);
  id.sendb_shr = tl["stagen"]->AddTask(&MHD::SendB_Shr, this, id.recvb);
  id.recvb_shr = tl["stagen"]->AddTask(&MHD::RecvB_Shr, this, id.sendb_shr);
  id.bcs       = tl["stagen"]->AddTask(&MHD::ApplyPhysicalBCs, this, id.recvb_shr);
  id.prol      = tl["stagen"]->AddTask(&MHD::Prolongate, this, id.bcs);
  id.c2p       = tl["stagen"]->AddTask(&MHD::ConToPrim, this, id.prol);
  id.newdt     = tl["stagen"]->AddTask(&MHD::NewTimeStep, this, id.c2p);

  // assemble "after_stagen" task list
  id.csend = tl["after_stagen"]->AddTask(&MHD::ClearSend, this, none);
  // although RecvFlux/U/E/B functions check that all recvs complete, add ClearRecv to
  // task list anyways to catch potential bugs in MPI communication logic
  id.crecv = tl["after_stagen"]->AddTask(&MHD::ClearRecv, this, id.csend);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SaveMHDState
//! \brief Copy primitives and bcc before step to enable computation of time derivatives,
//! for example to compute jcon in GRMHD.

TaskStatus MHD::SaveMHDState(Driver *pdrive, int stage) {
  if (wbcc_saved) {
    Kokkos::deep_copy(DevExeSpace(), wsaved, w0);
    Kokkos::deep_copy(DevExeSpace(), bccsaved, bcc0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI), and
//! initialize all boundary receive status flags to waiting (with or without MPI).  Note
//! this must be done for communication of BOTH conserved (cell-centered) and
//! face-centered fields AND their fluxes (with SMR/AMR).

TaskStatus MHD::InitRecv(Driver *pdrive, int stage) {
  // post receives for U
  TaskStatus tstat = pbval_u->InitRecv(nmhd+nscalars);
  if (tstat != TaskStatus::complete) return tstat;
  // post receives for B
  tstat = pbval_b->InitRecv(3);
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR post receives for fluxes of U, always post receives for fluxes of B
  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (stage >= 0) {
    // with SMR/AMR, post receives for fluxes of U
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_u->InitFluxRecv(nmhd+nscalars);
      if (tstat != TaskStatus::complete) return tstat;
    }
    // post receives for fluxes of B, which are used even with uniform grids
    tstat = pbval_b->InitFluxRecv(3);
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with orbital advection post receives for U and B
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = porb_u->InitRecv();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = porb_b->InitRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with shearing box boundaries caluclate x2-distance x1-boundaries have sheared and
  // with MPI post receives for U and B
  // only execute when (shearing box defined) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    Real qom = (psrc->qshear)*(psrc->omega0);
    Real time = pmy_pack->pmesh->time;
    if (stage == pdrive->nexp_stages) {
      time += pmy_pack->pmesh->dt;
    }
    tstat = psbox_u->InitRecv(qom, time);
    if (tstat != TaskStatus::complete) return tstat;
    tstat = psbox_b->InitRecv(qom, time);
    if (tstat != TaskStatus::complete) return tstat;
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::CopyCons
//! \brief Simple task list function that copies u0 --> u1, and b0 --> b1 in first stage

TaskStatus MHD::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
    Kokkos::deep_copy(DevExeSpace(), b1.x1f, b0.x1f);
    Kokkos::deep_copy(DevExeSpace(), b1.x2f, b0.x2f);
    Kokkos::deep_copy(DevExeSpace(), b1.x3f, b0.x3f);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::Fluxes
//! \brief Wrapper task list function that calls everything necessary to compute fluxes
//! of conserved variables

TaskStatus MHD::Fluxes(Driver *pdrive, int stage) {
  // select which calculate_flux function to call based on rsolver_method
  if (rsolver_method == MHD_RSolver::advect) {
    CalculateFluxes<MHD_RSolver::advect>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf) {
    CalculateFluxes<MHD_RSolver::llf>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlle) {
    CalculateFluxes<MHD_RSolver::hlle>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlld) {
    CalculateFluxes<MHD_RSolver::hlld>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf_sr) {
    CalculateFluxes<MHD_RSolver::llf_sr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlle_sr) {
    CalculateFluxes<MHD_RSolver::hlle_sr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf_gr) {
    CalculateFluxes<MHD_RSolver::llf_gr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlle_gr) {
    CalculateFluxes<MHD_RSolver::hlle_gr>(pdrive, stage);
  }

  // Add viscous, resistive, heat-flux, etc fluxes
  if (pvisc != nullptr) {
    pvisc->IsotropicViscousFlux(w0, pvisc->nu_iso, peos->eos_data, uflx);
  }
  if ((presist != nullptr) && (peos->eos_data.is_ideal)) {
    presist->OhmicEnergyFlux(b0, uflx);
  }
  if (pcond != nullptr) {
    pcond->AddHeatFlux(w0, peos->eos_data, uflx);
  }

  // call FOFC if necessary
  if (use_fofc) {
    FOFC(pdrive, stage);
  } else if (pmy_pack->pcoord->is_general_relativistic) {
    if (pmy_pack->pcoord->coord_data.bh_excise) {
      FOFC(pdrive, stage);
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendFlux
//! \brief Wrapper task list function to pack/send restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus MHD::SendFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel)  {
    tstat = pbval_u->PackAndSendFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvFlux
//! \brief Wrapper task list function to recv/unpack restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus MHD::RecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->RecvAndUnpackFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::MHDSrcTerms
//! \brief Wrapper task list function to apply source terms to conservative vars
//! Note source terms must be computed using only primitives (w0), as the conserved
//! variables (u0) have already been partially updated when this fn called.

TaskStatus MHD::MHDSrcTerms(Driver *pdrive, int stage) {
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);

  // Add source terms for various physics
  if (psrc->const_accel)  psrc->ConstantAccel(w0, peos->eos_data, beta_dt, u0);
  if (psrc->ism_cooling)  psrc->ISMCooling(w0, peos->eos_data, beta_dt, u0);
  if (psrc->rel_cooling)  psrc->RelCooling(w0, peos->eos_data, beta_dt, u0);
  if (psrc->shearing_box) psrc->ShearingBox(w0, bcc0, peos->eos_data, beta_dt, u0);

  // Add coordinate source terms in GR.  Again, must be computed with only primitives.
  if (pmy_pack->pcoord->is_general_relativistic &&
      !pmy_pack->pcoord->is_dynamical_relativistic) {
    pmy_pack->pcoord->CoordSrcTerms(w0, bcc0, peos->eos_data, beta_dt, u0);
  } else if (pmy_pack->pcoord->is_dynamical_relativistic) {
    pmy_pack->pdyngr->AddCoordTerms(w0, bcc0, beta_dt, u0, pmy_pack->pmesh->mb_indcs.ng);
  }

  // Add user source terms
  if (pmy_pack->pmesh->pgen->user_srcs) {
    (pmy_pack->pmesh->pgen->user_srcs_func)(pmy_pack->pmesh, beta_dt);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendU_OA
//! \brief Wrapper task list function to pack/send data for orbital advection

TaskStatus MHD::SendU_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = porb_u->PackAndSendCC(u0);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvU_OA
//! \brief Wrapper task list function to recv/unpack data for orbital advection

TaskStatus MHD::RecvU_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    Real qom = (psrc->qshear)*(psrc->omega0);
    tstat = porb_u->RecvAndUnpackCC(u0, recon_method, qom);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RestrictU
//! \brief Wrapper task list function to restrict conserved vars

TaskStatus MHD::RestrictU(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/AMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendU
//! \brief Wrapper task list function to pack/send cell-centered conserved variables

TaskStatus MHD::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvU
//! \brief Wrapper task list function to receive/unpack cell-centered conserved variables

TaskStatus MHD::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendU_Shr
//! \brief Wrapper task list function to pack/send data for shearing box boundaries

TaskStatus MHD::SendU_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = psbox_u->PackAndSendCC(u0, recon_method);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvU_Shr
//! \brief Wrapper task list function to recv/unpack data for shearing box boundaries
//! Orbital remap is performed in this step.

TaskStatus MHD::RecvU_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = psbox_u->RecvAndUnpackCC(u0);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::EFieldSrc
//! \brief Wrapper task list function to apply source terms to electric field

TaskStatus MHD::EFieldSrc(Driver *pdrive, int stage) {
  // only execute when (shearing box defined) AND (2D)
  if ((psrc->shearing_box) && (pmy_pack->pmesh->two_d)) {
    psrc->SBoxEField(b0, efld);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendE
//! \brief Wrapper task list function to pack/send fluxes of magnetic fields
//! (i.e. edge-centered electric field E) at MeshBlock boundaries. This is performed both
//! at MeshBlock boundaries at the same level (to keep magnetic flux in-sync on different
//! MeshBlocks), and at fine/coarse boundaries with SMR/AMR using restricted values of E.

TaskStatus MHD::SendE(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  tstat = pbval_b->PackAndSendFluxFC(efld);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvE
//! \brief Wrapper task list function to recv/unpack fluxes of magnetic fields
//! (i.e. edge-centered electric field E) at MeshBlock boundaries

TaskStatus MHD::RecvE(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  tstat = pbval_b->RecvAndUnpackFluxFC(efld);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendB_OA
//! \brief Wrapper task list function to pack/send data for orbital advection

TaskStatus MHD::SendB_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = porb_b->PackAndSendFC(b0);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvB_OA
//! \brief Wrapper task list function to recv/unpack data for orbital advection

TaskStatus MHD::RecvB_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    Real qom = (psrc->qshear)*(psrc->omega0);
    tstat = porb_b->RecvAndUnpackFC(b0, recon_method, qom);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendB
//! \brief Wrapper task list function to pack/send face-centered magnetic fields

TaskStatus MHD::SendB(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_b->PackAndSendFC(b0, coarse_b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvB
//! \brief Wrapper task list function to recv/unpack face-centered magnetic fields

TaskStatus MHD::RecvB(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_b->RecvAndUnpackFC(b0, coarse_b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendB_Shr
//! \brief Wrapper task list function to pack/send data for shearing box boundaries

TaskStatus MHD::SendB_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = psbox_b->PackAndSendFC(b0, recon_method);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvB_Shr
//! \brief Wrapper task list function to recv/unpack data for shearing box boundaries
//! Orbital remap is performed in this step.

TaskStatus MHD::RecvB_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = psbox_b->RecvAndUnpackFC(b0);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ApplyPhysicalBCs
//! \brief Wrapper task list function to call funtions that set physical and user BCs

TaskStatus MHD::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // physical BCs
  pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), u0);
  pbval_b->BFieldBCs((pmy_pack), (pbval_b->b_in), b0);

  // user BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive) variables
//! at fine/coarse bundaries with SMR/AMR

TaskStatus MHD::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    pbval_u->FillCoarseInBndryCC(u0, coarse_u0);
    pbval_b->FillCoarseInBndryFC(b0, coarse_b0);
    if (pmy_pack->pmesh->pmr->prolong_prims) {
      pbval_u->ConsToPrimCoarseBndry(coarse_u0, coarse_b0, coarse_w0);
      pbval_u->ProlongateCC(w0, coarse_w0);
      pbval_b->ProlongateFC(b0, coarse_b0);
      pbval_u->PrimToConsFineBndry(w0, b0, u0);
    } else {
      pbval_u->ProlongateCC(u0, coarse_u0);
      pbval_b->ProlongateFC(b0, coarse_b0);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ConToPrim
//! \brief Wrapper task list function to call ConsToPrim over entire mesh (including gz)

TaskStatus MHD::ConToPrim(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  peos->ConsToPrim(u0, b0, w0, bcc0, false, 0, n1m1, 0, n2m1, 0, n3m1);
  return TaskStatus::complete;
}

TaskStatus MHD::ConToPrimIdealNewtonianMhd(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2 * ng - 1;
  int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng - 1) : 0;

  const bool only_testfloors = false;
  const int il = 0;
  const int iu = n1m1;
  const int jl = 0;
  const int ju = n2m1;
  const int kl = 0;
  const int ku = n3m1;

  auto &cons = u0;
  auto &b = b0;
  auto &prim = w0;
  auto &bcc = bcc0;

  {
    int &nmhd = pmy_pack->pmhd->nmhd;
    int &nscal = pmy_pack->pmhd->nscalars;
    int &nmb = pmy_pack->nmb_thispack;
    auto &eos = peos->eos_data;
    auto &fofc_ = pmy_pack->pmhd->fofc;

    const int ni = (iu - il + 1);
    const int nji = (ju - jl + 1) * ni;
    const int nkji = (ku - kl + 1) * nji;
    const int nmkji = nmb * nkji;

    int nfloord_ = 0, nfloore_ = 0, nfloort_ = 0;
    Kokkos::parallel_reduce(
        "mhd_c2p", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
        KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumt) {
          int m = (idx) / nkji;
          int k = (idx - m * nkji) / nji;
          int j = (idx - m * nkji - k * nji) / ni;
          int i = (idx - m * nkji - k * nji - j * ni) + il;
          j += jl;
          k += kl;

          // load single state conserved variables
          MHDCons1D u;
          u.d = cons(m, IDN, k, j, i);
          u.mx = cons(m, IM1, k, j, i);
          u.my = cons(m, IM2, k, j, i);
          u.mz = cons(m, IM3, k, j, i);
          u.e = cons(m, IEN, k, j, i);

          // load cell-centered fields into conserved state
          // use input CC fields if only testing floors with FOFC
          if (only_testfloors) {
            u.bx = bcc(m, IBX, k, j, i);
            u.by = bcc(m, IBY, k, j, i);
            u.bz = bcc(m, IBZ, k, j, i);
            // else use simple linear average of face-centered fields
          } else {
            u.bx = 0.5 * (b.x1f(m, k, j, i) + b.x1f(m, k, j, i + 1));
            u.by = 0.5 * (b.x2f(m, k, j, i) + b.x2f(m, k, j + 1, i));
            u.bz = 0.5 * (b.x3f(m, k, j, i) + b.x3f(m, k + 1, j, i));
          }

          // if (m == 190 and i == 4 and j == 4 and k == 4) {
          //   std::cout << " IdealMhd:: before C2P (cons) " << std::endl;

          //   std::cout << " D   = " << u.d << std::endl;
          //   std::cout << " Mx  = " << u.mx << std::endl;
          //   std::cout << " My  = " << u.my << std::endl;
          //   std::cout << " Mz  = " << u.mz << std::endl;
          //   std::cout << " E   = " << u.e << std::endl;

          //   std::cout << " Bx = " << u.bx << std::endl;
          //   std::cout << " By = " << u.by << std::endl;
          //   std::cout << " Bz = " << u.bz << std::endl;

          //   std::cout << "\n" << std::endl;
          // }

          // call c2p function
          // (inline function in ideal_c2p_mhd.hpp file)
          HydPrim1D w;
          bool dfloor_used = false, efloor_used = false, tfloor_used = false;
          SingleC2P_IdealMHD(u, eos, w, dfloor_used, efloor_used, tfloor_used);

          // set FOFC flag and quit loop if this function called only to check
          // floors
          if (only_testfloors) {
            if (dfloor_used || efloor_used || tfloor_used) {
              fofc_(m, k, j, i) = true;
              sumd++; // use dfloor as counter for when either is true
            }
          } else {
            // update counter, reset conserved if floor was hit
            if (dfloor_used) {
              cons(m, IDN, k, j, i) = u.d;
              sumd++;
            }
            if (efloor_used) {
              cons(m, IEN, k, j, i) = u.e;
              sume++;
            }
            if (tfloor_used) {
              cons(m, IEN, k, j, i) = u.e;
              sumt++;
            }
            // store primitive state in 3D array
            prim(m, IDN, k, j, i) = w.d;
            prim(m, IVX, k, j, i) = w.vx;
            prim(m, IVY, k, j, i) = w.vy;
            prim(m, IVZ, k, j, i) = w.vz;
            prim(m, IEN, k, j, i) = w.e;
            // store cell-centered fields in 3D array
            bcc(m, IBX, k, j, i) = u.bx;
            bcc(m, IBY, k, j, i) = u.by;
            bcc(m, IBZ, k, j, i) = u.bz;
            // convert scalars (if any), always stored at end of cons and prim
            // arrays.
            for (int n = nmhd; n < (nmhd + nscal); ++n) {
              // apply scalar floor
              if (cons(m, n, k, j, i) < 0.0) {
                cons(m, n, k, j, i) = 0.0;
              }
              prim(m, n, k, j, i) = cons(m, n, k, j, i) / u.d;
            }

            // if (m == 190 and i == 4 and j == 4 and k == 4) {
            //   std::cout << " IdealMhd:: after C2P (prims) " << std::endl;

            //   std::cout << " dens = " << prim(m, IDN, k, j, i) << std::endl;
            //   std::cout << " v_x  = " << prim(m, IVX, k, j, i) << std::endl;
            //   std::cout << " v_y  = " << prim(m, IVY, k, j, i) << std::endl;
            //   std::cout << " v_z  = " << prim(m, IVZ, k, j, i) << std::endl;
            //   std::cout << " eint = " << prim(m, IEN, k, j, i) << std::endl;

            //   std::cout << " Bx = " << bcc(m, IBX, k, j, i) << std::endl;
            //   std::cout << " By = " << bcc(m, IBY, k, j, i) << std::endl;
            //   std::cout << " Bz = " << bcc(m, IBZ, k, j, i) << std::endl;

            //   std::cout << "\n" << std::endl;
            // }
          }
        },
        Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_),
        Kokkos::Sum<int>(nfloort_));

    // store appropriate counters
    if (only_testfloors) {
      pmy_pack->pmesh->ecounter.nfofc += nfloord_;
    } else {
      pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
      pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
      pmy_pack->pmesh->ecounter.neos_tfloor += nfloort_;
    }
  }

  return TaskStatus::complete;
}

// ===========================================================================
//  YK: this task manipulates primitive variables.
// 
TaskStatus MHD::RescaleAndAddRecoil(ParameterInput *pin) {

  if (global_variable::my_rank == 0) {
    std::cout << "  ** Rescaling primitives and adding recoil ...."
              << std::endl;
  }

  auto &pmbp = pmy_pack->pmesh->pmb_pack;

  // Parse a_over_rg from input
  const Real a_over_rg = pin->GetReal("cbd_to_bhl_mapping", "a_over_rg");

  // YK: fix this later. Need to talk with Elias on correct scaling
  // const Real rho0_over_rho_infty = 0.1;  // YK: rescale to have max 1.0 at the beginning..
  // const Real rho0_over_rho_infty = 1.0; // YK: do nothing?
  const Real rho0_over_rho_infty = pin->GetReal("cbd_to_bhl_mapping", "rho0_over_rho_infty");

  // Parse kick velocity from input file
  const Real vk_x = pin->GetReal("cbd_to_bhl_mapping", "v_kick_x");
  const Real vk_y = pin->GetReal("cbd_to_bhl_mapping", "v_kick_y");
  const Real vk_z = pin->GetReal("cbd_to_bhl_mapping", "v_kick_z");
  const Real v_squared = SQR(vk_x) + SQR(vk_y) + SQR(vk_z);

  // capture variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;

  auto &size = pmbp->pmb->mb_size;

  // MHD primitives
  auto &w0_ = pmbp->pmhd->w0;
  // MHD evolved vars
  auto &u0_ = pmbp->pmhd->u0;

  // Magnetic fields, face-centered and cell-centered
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc_ = pmbp->pmhd->bcc0;

  const auto &spin = coord.bh_spin;
  const auto &flat = coord.is_minkowski;

  const Real &gamma = pmy_pack->pmhd->peos->eos_data.gamma;

  par_for(
      "cbd_to_bhl", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je,
      is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(i - is, nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2v = CellCenterX(j - js, nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        int nx3 = indcs.nx3;
        Real x3v = CellCenterX(k - ks, nx3, x3min, x3max);

        // Compute metric coefficients of KS -- need later for P2C
        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

        // if (m == 190 and i == 4 and j == 4 and k == 4) {
        //   std::cout << i << ", " << j << ", " << k << std::endl;
        //   std::cout << nx1 << ", " << nx2 << ", " << nx3 << std::endl;
        //   std::cout << " dens = " << w0_(m, IDN, k, j, i) << std::endl;
        //   std::cout << " v_x  = " << w0_(m, IVX, k, j, i) << std::endl;
        //   std::cout << " v_y  = " << w0_(m, IVY, k, j, i) << std::endl;
        //   std::cout << " v_z  = " << w0_(m, IVZ, k, j, i) << std::endl;
        //   std::cout << " eint = " << w0_(m, IEN, k, j, i) << std::endl;

        //   std::cout << " Bx = " << bcc_(m, IBX, k, j, i) << std::endl;
        //   std::cout << " By = " << bcc_(m, IBY, k, j, i) << std::endl;
        //   std::cout << " Bz = " << bcc_(m, IBZ, k, j, i) << std::endl;

        //   std::cout << "\n" << std::endl;
        // }

        // Rescale hydro primitives
        w0_(m, IDN, k, j, i) *= rho0_over_rho_infty;
        w0_(m, IVX, k, j, i) /= std::sqrt(a_over_rg);
        w0_(m, IVY, k, j, i) /= std::sqrt(a_over_rg);
        w0_(m, IVZ, k, j, i) /= std::sqrt(a_over_rg);
        w0_(m, IEN, k, j, i) *= rho0_over_rho_infty / a_over_rg;

        // Cell-centered magnetic fields
        bcc_(m, IBX, k, j, i) *= std::sqrt(rho0_over_rho_infty / a_over_rg);
        bcc_(m, IBY, k, j, i) *= std::sqrt(rho0_over_rho_infty / a_over_rg);
        bcc_(m, IBZ, k, j, i) *= std::sqrt(rho0_over_rho_infty / a_over_rg);

        // Face-centered magnetic fields
        b0.x1f(m, k, j, i) *= std::sqrt(rho0_over_rho_infty / a_over_rg);
        b0.x2f(m, k, j, i) *= std::sqrt(rho0_over_rho_infty / a_over_rg);
        b0.x3f(m, k, j, i) *= std::sqrt(rho0_over_rho_infty / a_over_rg);
        // Include extra face-component at edges of the meshblock
        if (i == ie) {
          b0.x1f(m, k, j, i + 1) *= std::sqrt(rho0_over_rho_infty / a_over_rg);
        }
        if (j == je) {
          b0.x2f(m, k, j + 1, i) *= std::sqrt(rho0_over_rho_infty / a_over_rg);
        }
        if (k == ke) {
          b0.x3f(m, k + 1, j, i) *= std::sqrt(rho0_over_rho_infty / a_over_rg);
        }

        // Add kick
        w0_(m, IVX, k, j, i) -= vk_x;
        w0_(m, IVY, k, j, i) -= vk_y;
        w0_(m, IVZ, k, j, i) -= vk_z;

        // if (m == 190 and i == 4 and j == 4 and k == 4) {
        //   std::cout << i << ", " << j << ", " << k << std::endl;
        //   std::cout << nx1 << ", " << nx2 << ", " << nx3 << std::endl;
        //   std::cout << " dens = " << w0_(m, IDN, k, j, i) << std::endl;
        //   std::cout << " v_x  = " << w0_(m, IVX, k, j, i) << std::endl;
        //   std::cout << " v_y  = " << w0_(m, IVY, k, j, i) << std::endl;
        //   std::cout << " v_z  = " << w0_(m, IVZ, k, j, i) << std::endl;
        //   std::cout << " eint = " << w0_(m, IEN, k, j, i) << std::endl;

        //   std::cout << " Bx = " << bcc_(m, IBX, k, j, i) << std::endl;
        //   std::cout << " By = " << bcc_(m, IBY, k, j, i) << std::endl;
        //   std::cout << " Bz = " << bcc_(m, IBZ, k, j, i) << std::endl;
        // }

        // Update conservatives from the modified primitives
        MHDPrim1D w;
        w.d = w0_(m, IDN, k, j, i);
        w.vx = w0_(m, IVX, k, j, i);
        w.vy = w0_(m, IVY, k, j, i);
        w.vz = w0_(m, IVZ, k, j, i);
        w.e = w0_(m, IEN, k, j, i);
        w.bx = bcc_(m, IBX, k, j, i);
        w.by = bcc_(m, IBY, k, j, i);
        w.bz = bcc_(m, IBZ, k, j, i);

        HydCons1D u;
        SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);
        u0_(m, IDN, k, j, i) = u.d;
        u0_(m, IM1, k, j, i) = u.mx;
        u0_(m, IM2, k, j, i) = u.my;
        u0_(m, IM3, k, j, i) = u.mz;
        u0_(m, IEN, k, j, i) = u.e;

        // if (m == 190 and i == 4 and j == 4 and k == 4) {
        //   std::cout << " D   = " << u.d << std::endl;
        //   std::cout << " Mx  = " << u.mx << std::endl;
        //   std::cout << " My  = " << u.my << std::endl;
        //   std::cout << " Mz  = " << u.mz << std::endl;
        //   std::cout << " E   = " << u.e << std::endl;

        //   std::cout << " Bx = " << w.bx << std::endl;
        //   std::cout << " By = " << w.by << std::endl;
        //   std::cout << " Bz = " << w.bz << std::endl;
        // }
      });

  return TaskStatus::complete;
}
// ===========================================================================


//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed. Used in
//! TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears U, B, Flx_U, Flx_B, U_OA, B_OA, U_Shr, BShr
//! If (last stage)>stage>=(0): clears U, B, Flx_U, Flx_B,             U_Shr, B_Shr
//! If stage=(-1):              clears U, B
//! If stage=(-4):              clears                                 U_Shr, B_Shr

TaskStatus MHD::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat;
  if ((stage >= 0) || (stage == -1)) {
    // check sends of U complete
    TaskStatus tstat = pbval_u->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
    // check sends of B complete
    tstat = pbval_b->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check sends for fluxes of U complete.  Always check sends of E complete
  // do not check flux send for ICs (stage < 0)
  if (stage >= 0) {
    // with SMR/AMR check sends of restricted fluxes of U complete
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_u->ClearFluxSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
    // check sends of restricted fluxes of B complete even for uniform grids
    tstat = pbval_b->ClearFluxSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with orbital advection check sends for U and B complete
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = porb_u->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = porb_b->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with shearing box boundaries check sends of U and B complete
  // only execute when (shearing box defined) AND (stage>=0 or -4) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && ((stage >= 0) || (stage == -4)) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = psbox_u->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = psbox_b->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have completed. Used in
//! TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears U, B, Flx_U, Flx_B, U_OA, B_OA, U_Shr, BShr
//! If (last stage)>stage>=(0): clears U, B, Flx_U, Flx_B,             U_Shr, B_Shr
//! If stage=(-1):              clears U, B
//! If stage=(-4):              clears                                 U_Shr, B_Shr

TaskStatus MHD::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat;
  if ((stage >= 0) || (stage == -1)) {
    // check receives of U complete
    tstat = pbval_u->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
    // check receives of B complete
    tstat = pbval_b->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check recvs for fluxes of U complete.  Always check recvs of E complete
  // do not check flux receives when stage < 0 (i.e. ICs)
  if (stage >= 0) {
    // with SMR/AMR check receives of restricted fluxes of U complete
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_u->ClearFluxRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
    // with SMR/AMR check receives of restricted fluxes of B complete
    tstat = pbval_b->ClearFluxRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with orbital advection check receives of U and B are complete
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = porb_u->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = porb_b->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with shearing box boundaries check receives of U and B complete
  // only execute when (shearing box defined) AND (stage>=0 or -4) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && ((stage >= 0) || (stage == -4)) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = psbox_u->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = psbox_b->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RestrictB
//! \brief Wrapper function that restricts face-centered variables (magnetic field)

TaskStatus MHD::RestrictB(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/AMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictFC(b0, coarse_b0);
  }
  return TaskStatus::complete;
}

} // namespace mhd
