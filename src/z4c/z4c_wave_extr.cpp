//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_weyl_scalars.cpp
//  \brief implementation of functions in the Z4c class related to calculation of Weyl scalars

// C++ standard headers
//#include <iostream>
#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <fstream>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "globals.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"

namespace z4c {

//Factorial
Real fac(Real n){
 if(n==0 || n==1){
   return 1.0;
 }
 else{
   n=n*fac(n-1);
   return n;
 }
}

//Calculate spin weighted spherical harmonics sw=-2 using Wigner-d matrix notation see e.g. Eq II.7, II.8 in 0709.0093
void swsh(Real * ylmR, Real * ylmI, int l, int m, Real theta, Real phi){
  Real wignerd = 0;
  int k1,k2,k;
  k1 = std::max(0, m-2);
  k2 = std::min(l+m,l-2);
  for (k = k1; k<k2+1; ++k){
    wignerd += pow((-1),k)*sqrt(fac(l+m)*fac(l-m)*fac(l+2)*fac(l-2))*pow(std::cos(theta/2.0),2*l+m-2-2*k)
    *pow(std::sin(theta/2.0),2*k+2-m)/(fac(l+m-k)*fac(l-2-k)*fac(k)*fac(k+2-m));
  }
  *ylmR = sqrt((2*l+1)/(4*M_PI))*wignerd*std::cos(m*phi);
  *ylmI = sqrt((2*l+1)/(4*M_PI))*wignerd*std::sin(m*phi);
}
int LmIndex(int l,int m) {
    return l*l+m+l-4;
}
//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cWeyl(MeshBlockPack *pmbp)
// \brief compute the weyl scalars given the adm variables and matter state
//
// This function operates only on the interior points of the MeshBlock
void Z4c::WaveExtr(MeshBlockPack *pmbp) {

  #ifdef MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
  #endif
  // Spherical Grid for user-defined history
  auto &grids = pmbp->pz4c->spherical_grids;
  auto &u_weyl = pmbp->pz4c->u_weyl;
  auto &psi_out = pmbp->pz4c->psi_out;
  // TODO(@hzhu): add an mpi call here to fill in the ghost before interpolation to sphere

  // number of radii
  int nradii = grids.size();

  // maximum l; TODO(@hzhu): read in from input file
  int lmax = 8;
  bool bitant = true;

  Real ylmR,ylmI;
  for (int g=0; g<nradii; ++g) {
    // Interpolate Weyl scalars to the surface
    grids[g]->InterpolateToSphere(2, u_weyl);
    for (int l = 2; l < lmax+1; ++l) {
      for (int m = -l; m < l+1 ; ++m) {
        Real psilmR = 0.0;
        Real psilmI = 0.0;
          for (int ip = 0; ip < grids[g]->nangles; ++ip) {
            Real theta = grids[g]->polar_pos.h_view(ip,0);
            Real phi = grids[g]->polar_pos.h_view(ip,1);
            Real datareal = grids[g]->interp_vals.h_view(ip,0);
            Real dataim = grids[g]->interp_vals.h_view(ip,1);
            Real weight = grids[g]->solid_angles.h_view(ip);
            swsh(&ylmR,&ylmI,l,m,theta,phi);
            // The spherical harmonics transform as Y^s_{l m}( Pi-th, ph ) = (-1)^{l+s} Y^s_{l -m}(th, ph)
            // but the PoisitionPolar function returns theta \in [0,\pi], so these are correct for bitant.
            // With bitant, under reflection the imaginary part of the weyl scalar should pick a - sign,
            // which is accounted for here.
            Real bitant_z_fac = (bitant && theta > M_PI/2) ? -1 : 1;
            psilmR += datareal*weight*ylmR + bitant_z_fac*dataim*weight*ylmI;
            psilmI += bitant_z_fac*dataim*weight*ylmR - datareal*weight*ylmI;
          }
        psi_out(g,LmIndex(l,m),0) = psilmR;
        psi_out(g,LmIndex(l,m),1) = psilmI;
      }
    }
    if (g==0) {
      // Output file name
      std::string filename = "waveforms/out.tab";

      // Check if the file already exists
      std::ifstream fileCheck(filename);
      bool fileExists = fileCheck.good();
      fileCheck.close();

      // If the file doesn't exist, create it
      if (!fileExists) {
          std::ofstream createFile(filename);
          createFile.close();
      }
      
      // Open a file stream for writing
      std::ofstream outFile;

      // append mode
      outFile.open(filename, std::ios::out | std::ios::app);

      // first append time
      outFile << pmbp->pmesh->time << "\t";
      // append waveform
      for (int l = 2; l < lmax+1; ++l) {
        for (int m = -l; m < l+1 ; ++m) {
          outFile << psi_out(g,LmIndex(l,m),0) << '\t';
        }
      }
      outFile << '\n';
      // Close the file stream
      outFile.close();
    }
  }
  #ifdef MPI_PARALLEL_ENABLED
  }
  #endif
}


}
