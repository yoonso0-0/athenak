#ifndef COORDINATES_CELL_LOCATIONS_HPP_
#define COORDINATES_CELL_LOCATIONS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cell_locations.hpp
//  \brief functions to compute locations on a uniform Cartesian grid
// They provide functionality of the Coordinates class in the C++ version of the code.
// Very similar to cc_pos.c function in C version of the code (Athena4.2)
// Not incorporated in Coordinates class so that they can be used anywhere (for example
// to compute locations of MeshBlocks in Mesh).

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn void LeftEdgeX()
// returns x-posn of left edge of i^th cell where index range [0,N] maps to [xmin,xmax]
// returns ghost cell posn if i outside range [0,N] (e.g. i=-1 is x-posn of first ghost
// cell). Averages linear interpolation from each side to symmetrize r.o. error

KOKKOS_INLINE_FUNCTION
static Real LeftEdgeX(int ith, int n, Real xmin, Real xmax) {
  Real x = (static_cast<Real>(ith)) / (static_cast<Real>(n));
  return (x*xmax - x*xmin) - (0.5*xmax - 0.5*xmin) + (0.5*xmin + 0.5*xmax);
}

//----------------------------------------------------------------------------------------
//! \fn void SymmetricLeftEdgeX()
// returns x-posn of left edge of ith division where index range [0,n] maps to
// [gmin,gmax].
//
// Same position as LeftEdgeX to round-off, but exactly antisymmetric under reflection
// about the mesh center when the mesh is centered on zero (gmin = -gmax): edge ith and
// edge (n-ith) are related by an exact IEEE sign flip.  Used to set MeshBlock bounds, so
// that mirror-image MeshBlocks get exactly-negated bounds and hence a bitwise-identical
// dx.  LeftEdgeX cannot do this because it rounds ith/n independently at each edge,
// which leaves mirror blocks with dx differing by up to ~20 ulp -- an asymmetry that
// then enters every flux divergence, at every stage of every cycle.

KOKKOS_INLINE_FUNCTION
static Real SymmetricLeftEdgeX(int ith, int n, Real gmin, Real gmax) {
  Real dx = (gmax - gmin)/(static_cast<Real>(n));
  return 0.5*(gmin + gmax) + (static_cast<Real>(ith) - 0.5*static_cast<Real>(n))*dx;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenterX()
// returns cell-center posn of i^th cell where index range [0,N] maps to [xmin,xmax]
// returns ghost cell posn if i outside range [0,N] (e.g. i=-1 is cc-posn of first ghost
// cell). Averages linear interpolation from each side to symmetrize r.o. error

KOKKOS_INLINE_FUNCTION
static Real CellCenterX(int ith, int n, Real xmin, Real xmax) {
  Real x = (static_cast<Real>(ith) + 0.5) / (static_cast<Real>(n));
  return (x*xmax - x*xmin) - (0.5*xmax - 0.5*xmin) + (0.5*xmin + 0.5*xmax);
}

//----------------------------------------------------------------------------------------
//! \fn void SymmetricCellCenterX()
// returns cell-center posn of the ith cell of a MeshBlock whose left edge is at xmin,
// for a mesh spanning [gmin,gmax] with ntot cells in this direction at this level.
//
// Same position as CellCenterX to round-off, but exactly antisymmetric under reflection
// about the mesh center when the mesh is centered on zero (gmin = -gmax): mirror cells
// return values related by an exact IEEE sign flip, so cos()/even functions of the result
// agree bit-for-bit. CellCenterX cannot do this because it rounds (i+0.5)/n independently
// at each cell, and per-MeshBlock bounds carry their own ~1 ulp asymmetry; both are
// bypassed here by reconstructing the global cell index and using only the mesh bounds.

KOKKOS_INLINE_FUNCTION
static Real SymmetricCellCenterX(int ith, Real xmin, Real gmin, Real gmax, int ntot) {
  Real dx = (gmax - gmin)/(static_cast<Real>(ntot));
  int ioff = static_cast<int>((xmin - gmin)/dx + 0.5);
  Real xoff = (static_cast<Real>(ioff + ith) - 0.5*(static_cast<Real>(ntot) - 1.0))*dx;
  return 0.5*(gmin + gmax) + xoff;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenterIndex()
// returns i-index of cell containing x position

// TODO(@user): set trap if out-of-range

KOKKOS_INLINE_FUNCTION
static int CellCenterIndex(Real x, int n, Real xmin, Real xmax) {
  return static_cast<int>(((x-xmin)/(xmax-xmin))*static_cast<Real>(n));
}

#endif // COORDINATES_CELL_LOCATIONS_HPP_
