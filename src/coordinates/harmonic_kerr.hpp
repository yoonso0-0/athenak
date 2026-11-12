#ifndef COORDINATES_HARMONIC_KERR_HPP_
#define COORDINATES_HARMONIC_KERR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file harmonic_kerr.hpp
//! \brief Kerr metric (M = 1) in horizon-penetrating Cartesian harmonic
//! coordinates of Cook & Scheel (1997, PRD 56, 4775).
//!
//! With Boyer-Lindquist radius r and R = r - M, the harmonic coordinates
//! (T,X,Y,Z) are
//!   X + iY = (R + ia) e^{i phi_KS} sin(theta),   Z = R cos(theta),
//!   T      = t_KS - 2M ln(R + R_H),              R_H = sqrt(M^2 - a^2),
//! i.e. the spatial coordinates are Cartesian Kerr-Schild (CKS) coordinates
//! built from the spheroidal radius R instead of r, and the slicing differs
//! from KS by a function of r. The horizon sits at R = R_H, and R = 0 (r = M)
//! is a coordinate singularity that lies inside the horizon (it must be
//! excised).
//!
//! The map to CKS coordinates (t,x^i) is
//!   x^i = X^i + M n^i,    t = T + 2M ln(R + R_H),
//!   n   = ((R X + a Y)/(R^2+a^2), (R Y - a X)/(R^2+a^2), Z/R),
//! where |n| = 1 and the CKS null covector evaluated at x^i is l = (1, n).
//! Pulling back g_CKS = eta + 2 H l l with Lambda^a_mu = dx^a/dX^mu gives
//!   g_00 = -1 + 2H
//!   g_0j = -f dR_j + 2H L_j
//!   g_ij = Lambda^k_i Lambda^k_j - f^2 dR_i dR_j + 2H L_i L_j
//! with f = 2M/(R + R_H), dR_j = dR/dX^j, L_j = n_j + f dR_j (using n.dn = 0),
//! Lambda^k_i = delta^k_i + M dn^k/dX^i, and H = M r R^2/(r^2 R^2 + a^2 Z^2).
//!
//! Metric derivatives are obtained exactly with forward-mode automatic
//! differentiation (dual numbers carrying the gradient with respect to X, Y,
//! Z).
//!
//! These functions are used in place of the Cartesian Kerr-Schild ones in
//! cartesian_ks.hpp when compiled with -DAthena_HARMONIC_KS=ON. Unlike CKS,
//! sqrt(-g) != 1 here, so the GR (M)HD conserved variables, fluxes, source
//! terms and face-centered magnetic fields are densitized by sqrt(-g) (see
//! SqrtMinusDetMetric()).

#include <math.h>

#include "athena.hpp"
#include "coordinates/adm.hpp"

namespace harmonic_kerr {

//----------------------------------------------------------------------------------------
//! \struct Dual
//! \brief forward-mode dual number: value and gradient with respect to (X,Y,Z)

struct Dual {
  Real v;
  Real d[3];
};

// builds a Dual from its value and gradient components
KOKKOS_INLINE_FUNCTION Dual MakeDual(const Real v, const Real d0,
                                     const Real d1, const Real d2) {
  Dual res;
  res.v = v;
  res.d[0] = d0;
  res.d[1] = d1;
  res.d[2] = d2;
  return res;
}

KOKKOS_INLINE_FUNCTION Real Value(const Real &a) { return a; }
KOKKOS_INLINE_FUNCTION Real Value(const Dual &a) { return a.v; }

// constant of the same scalar type as the second argument (zero gradient for
// Dual)
KOKKOS_INLINE_FUNCTION Real Constant(const Real v, const Real &) { return v; }
KOKKOS_INLINE_FUNCTION Dual Constant(const Real v, const Dual &) {
  return MakeDual(v, 0.0, 0.0, 0.0);
}

KOKKOS_INLINE_FUNCTION Dual operator+(const Dual &a, const Dual &b) {
  return MakeDual(a.v + b.v, a.d[0] + b.d[0], a.d[1] + b.d[1], a.d[2] + b.d[2]);
}
KOKKOS_INLINE_FUNCTION Dual operator+(const Dual &a, const Real b) {
  return MakeDual(a.v + b, a.d[0], a.d[1], a.d[2]);
}
KOKKOS_INLINE_FUNCTION Dual operator+(const Real a, const Dual &b) {
  return b + a;
}
KOKKOS_INLINE_FUNCTION Dual operator-(const Dual &a) {
  return MakeDual(-a.v, -a.d[0], -a.d[1], -a.d[2]);
}
KOKKOS_INLINE_FUNCTION Dual operator-(const Dual &a, const Dual &b) {
  return MakeDual(a.v - b.v, a.d[0] - b.d[0], a.d[1] - b.d[1], a.d[2] - b.d[2]);
}
KOKKOS_INLINE_FUNCTION Dual operator-(const Dual &a, const Real b) {
  return MakeDual(a.v - b, a.d[0], a.d[1], a.d[2]);
}
KOKKOS_INLINE_FUNCTION Dual operator-(const Real a, const Dual &b) {
  return MakeDual(a - b.v, -b.d[0], -b.d[1], -b.d[2]);
}
KOKKOS_INLINE_FUNCTION Dual operator*(const Dual &a, const Dual &b) {
  return MakeDual(a.v * b.v, a.d[0] * b.v + a.v * b.d[0],
                  a.d[1] * b.v + a.v * b.d[1], a.d[2] * b.v + a.v * b.d[2]);
}
KOKKOS_INLINE_FUNCTION Dual operator*(const Dual &a, const Real b) {
  return MakeDual(a.v * b, a.d[0] * b, a.d[1] * b, a.d[2] * b);
}
KOKKOS_INLINE_FUNCTION Dual operator*(const Real a, const Dual &b) {
  return b * a;
}
KOKKOS_INLINE_FUNCTION Dual operator/(const Dual &a, const Dual &b) {
  const Real ib = 1.0 / b.v;
  const Real q = a.v * ib;
  return MakeDual(q, (a.d[0] - q * b.d[0]) * ib, (a.d[1] - q * b.d[1]) * ib,
                  (a.d[2] - q * b.d[2]) * ib);
}
KOKKOS_INLINE_FUNCTION Dual operator/(const Dual &a, const Real b) {
  const Real ib = 1.0 / b;
  return MakeDual(a.v * ib, a.d[0] * ib, a.d[1] * ib, a.d[2] * ib);
}
KOKKOS_INLINE_FUNCTION Dual operator/(const Real a, const Dual &b) {
  const Real q = a / b.v;
  const Real s = -q / b.v;
  return MakeDual(q, s * b.d[0], s * b.d[1], s * b.d[2]);
}

//----------------------------------------------------------------------------------------
//! \fn Real SpheroidalRadius
//! \brief harmonic spheroidal radius R = r_BL - M at Cartesian harmonic
//! (X,Y,Z), i.e. the positive root of R^4 - (X^2+Y^2+Z^2-a^2) R^2 - a^2 Z^2 =
//! 0. Smoothly floored at a tiny value (as for CKS) to avoid division by zero
//! at the R = 0 coordinate singularity.

KOKKOS_INLINE_FUNCTION
Real SpheroidalRadius(const Real x, const Real y, const Real z, const Real a) {
  Real rad2 = SQR(x) + SQR(y) + SQR(z);
  Real rr = sqrt(
      (rad2 - SQR(a) + sqrt(SQR(rad2 - SQR(a)) + 4.0 * SQR(a) * SQR(z))) / 2.0);
  const Real eps = 1.0e-6;
  if (rr < eps) {
    rr = 0.5 * (eps + rr * rr / eps);
  }
  return rr;
}

// Spheroidal radius as a function of the scalar type: for Dual, the gradient
// follows from implicit differentiation of the quartic, evaluated at the
// (floored) radius.
KOKKOS_INLINE_FUNCTION
Real Radius(const Real &x, const Real &y, const Real &z, const Real a) {
  return SpheroidalRadius(x, y, z, a);
}

KOKKOS_INLINE_FUNCTION
Dual Radius(const Dual &x, const Dual &y, const Dual &z, const Real a) {
  const Real rr = SpheroidalRadius(x.v, y.v, z.v, a);
  const Real den = SQR(SQR(rr)) + SQR(a) * SQR(z.v);
  const Real gx = SQR(rr) * rr * x.v / den;
  const Real gy = SQR(rr) * rr * y.v / den;
  const Real gz = rr * (SQR(rr) + SQR(a)) * z.v / den;
  Dual res;
  res.v = rr;
  for (int k = 0; k < 3; ++k) {
    res.d[k] = gx * x.d[k] + gy * y.d[k] + gz * z.d[k];
  }
  return res;
}

//----------------------------------------------------------------------------------------
//! \fn void LowerMetric
//! \brief lower metric g_{mu nu} at harmonic (X,Y,Z), templated on scalar type
//! so the same expressions give the metric (Real) and its first derivatives
//! (Dual).
//!
//! The coordinate singularity R = 0 is the disk X^2+Y^2 <= a^2, Z = 0 (the
//! origin for a = 0), and close to it the metric is not even Lorentzian in
//! floating point. Points with R < R_floor (deep inside the horizon, and inside
//! the default excision radius R_H/2) are therefore evaluated at Z = +-R_floor
//! instead, which guarantees R >= R_floor since |Z| <= R. This only affects
//! excised cells, but keeps the metric finite there.

template <typename T>
KOKKOS_INLINE_FUNCTION void LowerMetric(const T &x, const T &y, const T &z_in,
                                        const bool minkowski, const Real a,
                                        T g[][4]) {
  const Real m = minkowski ? 0.0 : 1.0;             // black hole mass
  const Real rh = sqrt(fmax(SQR(m) - SQR(a), 0.0)); // horizon radius R_H

  const Real rfloor = fmax(0.1 * rh, 1.0e-3);
  T z = z_in;
  if (SpheroidalRadius(Value(x), Value(y), Value(z_in), a) < rfloor) {
    z = Constant((Value(z_in) >= 0.0) ? rfloor : -rfloor, z_in);
  }

  T rr = Radius(x, y, z, a);
  T rr2 = rr * rr;
  T q = rr2 + SQR(a);
  T den = rr2 * rr2 + SQR(a) * (z * z);

  // gradient of R
  T dr[3] = {rr2 * rr * x / den, rr2 * rr * y / den, rr * q * z / den};

  // unit vector n (spatial part of the CKS null covector) and its gradient
  // dn[i][j]
  T n[3] = {(rr * x + a * y) / q, (rr * y - a * x) / q, z / rr};
  T dn[3][3];
  for (int j = 0; j < 3; ++j) {
    dn[0][j] = (x - 2.0 * rr * n[0]) * dr[j] / q;
    dn[1][j] = (y - 2.0 * rr * n[1]) * dr[j] / q;
    dn[2][j] = -z * dr[j] / rr2;
  }
  dn[0][0] = dn[0][0] + rr / q;
  dn[0][1] = dn[0][1] + a / q;
  dn[1][0] = dn[1][0] - a / q;
  dn[1][1] = dn[1][1] + rr / q;
  dn[2][2] = dn[2][2] + 1.0 / rr;

  // spatial Jacobian Lambda^k_i = dx_CKS^k/dX^i
  T lam[3][3];
  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 3; ++i) {
      lam[k][i] = m * dn[k][i] + ((k == i) ? 1.0 : 0.0);
    }
  }

  // dt_CKS/dR and the CKS scalar H evaluated at the mapped point
  T f = (2.0 * m) / (rr + rh);
  T r = rr + m;
  T hh = m * r * rr2 / (r * r * rr2 + SQR(a) * (z * z));

  // L_j = l_a Lambda^a_j
  T ll[3];
  for (int j = 0; j < 3; ++j) {
    ll[j] = n[j] + f * dr[j];
  }

  g[0][0] = 2.0 * hh - 1.0;
  for (int j = 0; j < 3; ++j) {
    g[0][j + 1] = 2.0 * hh * ll[j] - f * dr[j];
    g[j + 1][0] = g[0][j + 1];
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      T gij =
          lam[0][i] * lam[0][j] + lam[1][i] * lam[1][j] + lam[2][i] * lam[2][j];
      gij = gij - f * f * dr[i] * dr[j] + 2.0 * hh * ll[i] * ll[j];
      g[i + 1][j + 1] = gij;
      g[j + 1][i + 1] = gij;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void InvertMetric
//! \brief upper metric from lower metric via the 3+1 split

KOKKOS_INLINE_FUNCTION
void InvertMetric(const Real glower[][4], Real gupper[][4]) {
  const Real det = adm::SpatialDet(glower[1][1], glower[1][2], glower[1][3],
                                   glower[2][2], glower[2][3], glower[3][3]);
  Real gi[3][3];
  adm::SpatialInv(1.0 / det, glower[1][1], glower[1][2], glower[1][3],
                  glower[2][2], glower[2][3], glower[3][3], &gi[0][0],
                  &gi[0][1], &gi[0][2], &gi[1][1], &gi[1][2], &gi[2][2]);
  gi[1][0] = gi[0][1];
  gi[2][0] = gi[0][2];
  gi[2][1] = gi[1][2];

  // shift beta^i = gamma^ij beta_j with beta_j = g_0j, and alpha^2 = beta^i
  // beta_i - g_00
  Real beta_u[3];
  for (int i = 0; i < 3; ++i) {
    beta_u[i] = gi[i][0] * glower[0][1] + gi[i][1] * glower[0][2] +
                gi[i][2] * glower[0][3];
  }
  Real alpha2 = beta_u[0] * glower[0][1] + beta_u[1] * glower[0][2] +
                beta_u[2] * glower[0][3] - glower[0][0];
  Real ialpha2 = 1.0 / alpha2;

  gupper[0][0] = -ialpha2;
  for (int i = 0; i < 3; ++i) {
    gupper[0][i + 1] = beta_u[i] * ialpha2;
    gupper[i + 1][0] = gupper[0][i + 1];
    for (int j = 0; j < 3; ++j) {
      gupper[i + 1][j + 1] = gi[i][j] - beta_u[i] * beta_u[j] * ialpha2;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeMetricAndInverse
//! \brief lower and upper metric in Cartesian harmonic Kerr coordinates

KOKKOS_INLINE_FUNCTION
void ComputeMetricAndInverse(Real x, Real y, Real z, bool minkowski, Real a,
                             Real glower[][4], Real gupper[][4]) {
  LowerMetric(x, y, z, minkowski, a, glower);
  InvertMetric(glower, gupper);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeMetricDerivatives
//! \brief partial derivatives of the lower metric with respect to X, Y, Z

KOKKOS_INLINE_FUNCTION
void ComputeMetricDerivatives(Real x, Real y, Real z, bool minkowski, Real a,
                              Real dg_dx1[][4], Real dg_dx2[][4],
                              Real dg_dx3[][4]) {
  const Dual xd = MakeDual(x, 1.0, 0.0, 0.0);
  const Dual yd = MakeDual(y, 0.0, 1.0, 0.0);
  const Dual zd = MakeDual(z, 0.0, 0.0, 1.0);
  Dual g[4][4];
  LowerMetric(xd, yd, zd, minkowski, a, g);
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      dg_dx1[mu][nu] = g[mu][nu].d[0];
      dg_dx2[mu][nu] = g[mu][nu].d[1];
      dg_dx3[mu][nu] = g[mu][nu].d[2];
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeADMDecomposition
//! \brief ADM variables of the (stationary) harmonic Kerr metric. Uses the same
//! sign convention as cks::ComputeADMDecomposition:
//!   K_ij = (D_i beta_j + D_j beta_i)/(2 alpha).

KOKKOS_INLINE_FUNCTION
void ComputeADMDecomposition(Real x, Real y, Real z, bool minkowski, Real a,
                             Real *alp, Real *betax, Real *betay, Real *betaz,
                             Real *psi4, Real *gxx, Real *gxy, Real *gxz,
                             Real *gyy, Real *gyz, Real *gzz, Real *Kxx,
                             Real *Kxy, Real *Kxz, Real *Kyy, Real *Kyz,
                             Real *Kzz) {
  const Dual xd = MakeDual(x, 1.0, 0.0, 0.0);
  const Dual yd = MakeDual(y, 0.0, 1.0, 0.0);
  const Dual zd = MakeDual(z, 0.0, 0.0, 1.0);
  Dual g4[4][4];
  LowerMetric(xd, yd, zd, minkowski, a, g4);

  // spatial metric, its inverse, and derivatives dg_ddd[k][i][j] = d_k gamma_ij
  Real g_dd[3][3], dg_ddd[3][3][3], beta_d[3], dbeta_dd[3][3];
  for (int i = 0; i < 3; ++i) {
    beta_d[i] = g4[0][i + 1].v;
    for (int k = 0; k < 3; ++k) {
      dbeta_dd[k][i] = g4[0][i + 1].d[k];
    }
    for (int j = 0; j < 3; ++j) {
      g_dd[i][j] = g4[i + 1][j + 1].v;
      for (int k = 0; k < 3; ++k) {
        dg_ddd[k][i][j] = g4[i + 1][j + 1].d[k];
      }
    }
  }
  const Real det = adm::SpatialDet(g_dd[0][0], g_dd[0][1], g_dd[0][2],
                                   g_dd[1][1], g_dd[1][2], g_dd[2][2]);
  Real g_uu[3][3];
  adm::SpatialInv(1.0 / det, g_dd[0][0], g_dd[0][1], g_dd[0][2], g_dd[1][1],
                  g_dd[1][2], g_dd[2][2], &g_uu[0][0], &g_uu[0][1], &g_uu[0][2],
                  &g_uu[1][1], &g_uu[1][2], &g_uu[2][2]);
  g_uu[1][0] = g_uu[0][1];
  g_uu[2][0] = g_uu[0][2];
  g_uu[2][1] = g_uu[1][2];

  Real beta_u[3];
  for (int i = 0; i < 3; ++i) {
    beta_u[i] = g_uu[i][0] * beta_d[0] + g_uu[i][1] * beta_d[1] +
                g_uu[i][2] * beta_d[2];
  }
  const Real alpha = sqrt(beta_u[0] * beta_d[0] + beta_u[1] * beta_d[1] +
                          beta_u[2] * beta_d[2] - g4[0][0].v);

  *alp = alpha;
  *betax = beta_u[0];
  *betay = beta_u[1];
  *betaz = beta_u[2];
  *psi4 = pow(det, 1. / 3.);
  *gxx = g_dd[0][0];
  *gxy = g_dd[0][1];
  *gxz = g_dd[0][2];
  *gyy = g_dd[1][1];
  *gyz = g_dd[1][2];
  *gzz = g_dd[2][2];

  // Christoffel symbols of the spatial metric
  Real gamma_udd[3][3][3];
  for (int d = 0; d < 3; ++d)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        gamma_udd[d][i][j] = 0.0;
        for (int e = 0; e < 3; ++e) {
          gamma_udd[d][i][j] +=
              0.5 * g_uu[d][e] *
              (dg_ddd[i][e][j] + dg_ddd[j][e][i] - dg_ddd[e][i][j]);
        }
      }

  // K_ij = (D_i beta_j + D_j beta_i)/(2 alpha)
  Real k_dd[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      Real dbij = dbeta_dd[i][j];
      Real dbji = dbeta_dd[j][i];
      for (int d = 0; d < 3; ++d) {
        dbij -= gamma_udd[d][i][j] * beta_d[d];
        dbji -= gamma_udd[d][j][i] * beta_d[d];
      }
      k_dd[i][j] = (dbij + dbji) / (2.0 * alpha);
    }
  *Kxx = k_dd[0][0];
  *Kxy = k_dd[0][1];
  *Kxz = k_dd[0][2];
  *Kyy = k_dd[1][1];
  *Kyz = k_dd[1][2];
  *Kzz = k_dd[2][2];
  return;
}

} // namespace harmonic_kerr

#endif // COORDINATES_HARMONIC_KERR_HPP_
