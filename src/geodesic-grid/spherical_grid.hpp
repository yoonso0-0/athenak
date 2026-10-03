#ifndef GEODESIC_GRID_SPHERICAL_GRID_HPP_
#define GEODESIC_GRID_SPHERICAL_GRID_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_grid.hpp
//  \brief definitions for SphericalGrid class

#include "athena.hpp"
#include "geodesic-grid/geodesic_grid.hpp"

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class SphericalGrid

class SphericalGrid: public GeodesicGrid {
 public:
    // Creates a geodesic grid with refinement level nlev and radius rad
    SphericalGrid(MeshBlockPack *pmy_pack, int nlev, Real rad, int ninterp = -1);
    ~SphericalGrid();

    Real radius;  // const radius for SphericalGrid
    int ninterp;  // number of interpolation points along each dimension
    DualArray2D<Real> interp_coord;  // Cartesian coordinates for grid points
    DualArray2D<Real> interp_vals;   // container for data interpolated to sphere
    void InterpolateToSphere(int nvars, DvceArray5D<Real>& val);  // interpolate to sphere
    // interpolate a range of variables to a sphere
    void InterpolateToSphere(int vs, int ve, DvceArray5D<Real>& val);
    // Reflect the interpolation points through the coordinate planes through the origin
    // where fold[d] != 0, onto the side with sign fold[d] of x_d, e.g. fold[2]=+1 maps
    // points with x3<0 to x3>0. For a mesh covering only that side whose face at x_d=0 is
    // a reflecting boundary, the full sphere can then still be sampled: the caller must
    // flip the sign of the vector components normal to the folded planes at the points
    // that were moved. polar_pos and solid_angles keep describing the unfolded sphere.
    void FoldInterpolationCoordinates(const int fold[3]);

 private:
    MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
    DualArray2D<int> interp_indcs;   // indices of MeshBlock and zones therein for interp
    DualArray3D<Real> interp_wghts;  // weights for interpolation
    void SetInterpolationCoordinates();  // set indexing for interpolation
    void SetInterpolationIndices();      // set indexing for interpolation
    void SetInterpolationWeights();      // set weights for interpolation
};

#endif // GEODESIC_GRID_SPHERICAL_GRID_HPP_
