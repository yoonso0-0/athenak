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
//! COORDINATES
//! -----------
//! Let r and theta be the Boyer-Lindquist radius and polar angle, phi_in the
//! ingoing-Kerr azimuth, and t_KS the Kerr-Schild time. Cartesian Kerr-Schild
//! (CKS) coordinates (t, x, y, z) are
//!   x + iy = (r + ia) e^{i phi_in} sin(theta),   z = r cos(theta),   t = t_KS.
//! The harmonic coordinates (T, X, Y, Z) replace r by the spheroidal radius
//! R = r - M and shift the time:
//!   X + iY = (R + ia) e^{i phi_in} sin(theta),   Z = R cos(theta),
//!   T      = t_KS - 2M ln((R + R_H)/(2M)),       R_H = sqrt(M^2 - a^2).
//! Since R + R_H = r - r_- with r_- = M - R_H, this is
//! T = t_KS + 2M ln|2M/(r - r_-)|, Eq. (6.3) of Zhang et al. (2012,
//! arXiv:1208.3034), from Cook & Scheel Eqs. (19)-(20). The constant -2M ln(2M)
//! in t_KS = T + 2M ln(R + R_H) - 2M ln(2M) only shifts the origin of T; it
//! drops out of dt and hence of the metric.
//!
//! Consequences:
//!   - the horizon r = r_+ = M + R_H is at R = R_H,
//!   - R = 0 (r = M, inside the horizon) is a coordinate singularity: the disk
//!     X^2+Y^2 <= a^2, Z = 0 (the origin for a = 0). It must be excised.
//! Given (X, Y, Z), R is the positive root of the CKS relation for r with r
//! replaced by R:
//!   R^4 - (X^2+Y^2+Z^2-a^2) R^2 - a^2 Z^2 = 0.
//!
//! NOTATION
//! --------
//! CKS quantities are lower case and harmonic ones upper case: (t, x^p) vs
//! (T, X^i), r vs R. Indices show which system a component belongs to:
//!   CKS:       A, B (spacetime),   p, q (spatial)
//!   harmonic:  mu, nu (spacetime), i, j, k (spatial; time index written 0)
//! Spatial indices run over 1..3 in formulas and 0..2 in code. d_mu = d/dX^mu
//! is the partial derivative in harmonic coordinates (its index is lower).
//! An object keeps its symbol in both systems: l_A and l_mu are the CKS and
//! harmonic components of the same covector l. Arrays with one index of each
//! kind (Lambda^p_i, d_i n^p) are Jacobians, not tensors. delta^p_i is the
//! identity matrix pairing equal labels (x <-> X, y <-> Y, z <-> Z). The spin a
//! and the covector l are never used as indices.
//!
//! METRIC
//! ------
//! The CKS metric is g_AB = eta_AB + 2H l_A l_B with
//!   l_A = (l_t, l_p) = (1, n_p),
//!   n_p = ((r x + a y)/(r^2+a^2), (r y - a x)/(r^2+a^2), z/r),
//!   H   = M r^3/(r^4 + a^2 z^2).
//! n is a unit vector, delta^{pq} n_p n_q = 1, and l is null. Raising the
//! index of n leaves its components unchanged, n^p = n_p, both with eta
//! (eta^{pq} = delta^{pq}) and with g (for null l, g^{AB} l_B = eta^{AB} l_B,
//! so l^A = (-1, n^p)). The code therefore stores one array, nhat, used as n_p
//! or n^p. This relies on the Kerr-Schild form and does not hold for harmonic
//! components.
//!
//! The map from harmonic (T, X^i) to CKS (t, x^p) coordinates is
//!   x^p = X^p + M n^p,    t = T + 2M ln((R + R_H)/(2M)),
//! where X^p is the harmonic coordinate with the same label as x^p.
//! Since n_x + i n_y = e^{i phi_in} sin(theta) = (X + iY)/(R + ia) and
//! n_z = cos(theta) = Z/R, in harmonic variables
//!   n^p = ((R X + a Y)/(R^2+a^2), (R Y - a X)/(R^2+a^2), Z/R),
//!   H   = M r R^2/(r^2 R^2 + a^2 Z^2)   (r = R + M, z = Z r/R).
//!
//! Harmonic components follow from the pullback
//!   g_{mu nu} = (dx^A/dX^mu) (dx^B/dX^nu) g_AB,   l_mu = (dx^A/dX^mu) l_A,
//! with the Jacobian
//!   d_0 t = 1,   d_j t = f d_j R,   d_0 x^p = 0,   d_i x^p = Lambda^p_i,
//!   f = dt/dR = 2M/(R + R_H),   Lambda^p_i = delta^p_i + M d_i n^p.
//! This gives l_0 = 1, l_j = f d_j R + delta^p_j n_p (see LowerMetric()), and
//!   g_00 = -1 + 2H
//!   g_0j = -f d_j R + 2H l_j
//!   g_ij = delta_pq Lambda^p_i Lambda^q_j - f^2 d_i R d_j R + 2H l_i l_j.
//! The terms without H are the pullback of eta; their minus signs come from
//! eta_tt = -1.
//!
//! DERIVATIVES
//! -----------
//! The derivatives d_k g_{mu nu} are computed exactly (to roundoff) with
//! forward-mode automatic differentiation:
//!   - A Dual holds a value v and its gradient d[k] = d_k v.
//!   - Overloaded +, -, *, / apply the sum, product and quotient rules, so any
//!     expression built from them carries its exact gradient.
//!   - LowerMetric() is a template on the scalar type T (not the harmonic time
//!     T): T = Real gives the metric, and T = Dual with inputs seeded as
//!     X = (X;1,0,0), Y = (Y;0,1,0), Z = (Z;0,0,1) gives g[mu][nu].d[k] =
//!     d_k g_{mu nu}.
//!   - Apart from the seeds, the only gradient set by hand is that of R, which
//!     is computed with sqrt (no Dual overload); see Radius().
//! Unlike finite differencing, there is no step size and no truncation error.
//!
//! USAGE
//! -----
//! When compiled with -DAthena_HARMONIC_KS=ON, ComputeMetricAndInverse() and
//! ComputeMetricDerivatives() in cartesian_ks.hpp forward to the functions
//! here. Since sqrt(-g) != 1 (unlike CKS), the GR (M)HD conserved variables,
//! fluxes, source terms and face-centered magnetic fields are densitized by
//! sqrt(-g) (see SqrtMinusDetMetric()). Only the fixed-spacetime GR path is
//! supported: there is no ADM decomposition, so runs with <adm> or <z4c>
//! blocks abort at startup (see coordinates.cpp).

#include <math.h>

#include "athena.hpp"
#include "coordinates/adm.hpp"

namespace harmonic_kerr {

//----------------------------------------------------------------------------------------
//! \struct Dual
//! \brief forward-mode dual number: a value v and its gradient d[k] = d_k v
//! with respect to the harmonic coordinates (X,Y,Z).
//!
//! Formally v + d[0] e_X + d[1] e_Y + d[2] e_Z with e_i e_j = 0, so a product
//! of Duals keeps exactly the first-order (gradient) terms.

struct Dual {
  Real v;    // value
  Real d[3]; // gradient (d/dX, d/dY, d/dZ)
};

// builds a Dual from its value and gradient components
KOKKOS_INLINE_FUNCTION Dual MakeDual(const Real v, const Real d0, const Real d1,
                                     const Real d2) {
  Dual res;
  res.v = v;
  res.d[0] = d0;
  res.d[1] = d1;
  res.d[2] = d2;
  return res;
}

// plain value of a Real or Dual, for branch decisions (sign of Z, size of R)
KOKKOS_INLINE_FUNCTION Real Value(const Real &a) { return a; }
KOKKOS_INLINE_FUNCTION Real Value(const Dual &a) { return a.v; }

// v as a constant (zero gradient) of the same type as the second argument,
// whose value is ignored
KOKKOS_INLINE_FUNCTION Real Constant(const Real v, const Real &) { return v; }
KOKKOS_INLINE_FUNCTION Dual Constant(const Real v, const Dual &) {
  return MakeDual(v, 0.0, 0.0, 0.0);
}

//----------------------------------------------------------------------------------------
// Dual arithmetic. Each operator returns the value of the operation and its
// gradient from the corresponding differentiation rule. Real operands are
// constants (zero gradient).

// sum rule: d(a + b) = da + db
KOKKOS_INLINE_FUNCTION Dual operator+(const Dual &a, const Dual &b) {
  return MakeDual(a.v + b.v, a.d[0] + b.d[0], a.d[1] + b.d[1], a.d[2] + b.d[2]);
}
KOKKOS_INLINE_FUNCTION Dual operator+(const Dual &a, const Real b) {
  return MakeDual(a.v + b, a.d[0], a.d[1], a.d[2]);
}
KOKKOS_INLINE_FUNCTION Dual operator+(const Real a, const Dual &b) {
  return b + a;
}

// negation and difference: d(-a) = -da, d(a - b) = da - db
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

// product rule: d(a b) = da b + a db
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

// quotient rule: d(a/b) = (da - (a/b) db)/b
KOKKOS_INLINE_FUNCTION Dual operator/(const Dual &a, const Dual &b) {
  const Real inv_b = 1.0 / b.v;
  const Real a_over_b = a.v * inv_b;
  return MakeDual(a_over_b, (a.d[0] - a_over_b * b.d[0]) * inv_b,
                  (a.d[1] - a_over_b * b.d[1]) * inv_b,
                  (a.d[2] - a_over_b * b.d[2]) * inv_b);
}
KOKKOS_INLINE_FUNCTION Dual operator/(const Dual &a, const Real b) {
  const Real inv_b = 1.0 / b;
  return MakeDual(a.v * inv_b, a.d[0] * inv_b, a.d[1] * inv_b, a.d[2] * inv_b);
}
// d(a/b) = -(a/b^2) db for a constant numerator a
KOKKOS_INLINE_FUNCTION Dual operator/(const Real a, const Dual &b) {
  const Real a_over_b = a / b.v;
  const Real dquot_db = -a_over_b / b.v;
  return MakeDual(a_over_b, dquot_db * b.d[0], dquot_db * b.d[1],
                  dquot_db * b.d[2]);
}

//----------------------------------------------------------------------------------------
//! \fn Real SpheroidalRadius
//! \brief spheroidal radius R = r - M at harmonic (X,Y,Z). The quartic
//! R^4 - (X^2+Y^2+Z^2-a^2) R^2 - a^2 Z^2 = 0 is a quadratic in R^2, with
//! positive root
//!   R^2 = [ (rho^2 - a^2) + sqrt((rho^2 - a^2)^2 + 4 a^2 Z^2) ] / 2,
//!   rho^2 = X^2+Y^2+Z^2.
//! As r in cartesian_ks.hpp, R is smoothly floored at 1e-6 to avoid division
//! by zero at R = 0.

KOKKOS_INLINE_FUNCTION
Real SpheroidalRadius(const Real x_harm, const Real y_harm, const Real z_harm,
                      const Real a) {
  // rho^2 = X^2+Y^2+Z^2
  const Real rho2_harm = SQR(x_harm) + SQR(y_harm) + SQR(z_harm);
  Real rsph =
      sqrt((rho2_harm - SQR(a) +
            sqrt(SQR(rho2_harm - SQR(a)) + 4.0 * SQR(a) * SQR(z_harm))) /
           2.0);
  // smooth floor: below eps, R -> (eps + R^2/eps)/2, which matches R and its
  // slope at R = eps and is >= eps/2
  const Real eps = 1.0e-6;
  if (rsph < eps) {
    rsph = 0.5 * (eps + rsph * rsph / eps);
  }
  return rsph;
}

//----------------------------------------------------------------------------------------
//! \fn Radius
//! \brief spheroidal radius R for either scalar type.
//!
//! For Real, this is SpheroidalRadius(). For Dual, the value is the same, and
//! the gradient follows from implicit differentiation of
//!   F(R; X,Y,Z) = R^4 - (X^2+Y^2+Z^2-a^2) R^2 - a^2 Z^2 = 0,
//! dR/dX^i = -(dF/dX^i)/(dF/dR). Eliminating X^2+Y^2+Z^2 with F = 0 gives
//!   dR/dX = R^3 X / D,   dR/dY = R^3 Y / D,   dR/dZ = R (R^2 + a^2) Z / D,
//!   D = R^4 + a^2 Z^2,
//! which the chain rule combines with the gradients carried by the inputs
//! (the unit seeds, except for a clamped Z in LowerMetric()).

KOKKOS_INLINE_FUNCTION
Real Radius(const Real &x_harm, const Real &y_harm, const Real &z_harm,
            const Real a) {
  return SpheroidalRadius(x_harm, y_harm, z_harm, a);
}

KOKKOS_INLINE_FUNCTION
Dual Radius(const Dual &x_harm, const Dual &y_harm, const Dual &z_harm,
            const Real a) {
  const Real rsph = SpheroidalRadius(x_harm.v, y_harm.v, z_harm.v, a);
  // D = R^4 + a^2 Z^2
  const Real grad_den = SQR(SQR(rsph)) + SQR(a) * SQR(z_harm.v);
  const Real drsph_dx_harm = SQR(rsph) * rsph * x_harm.v / grad_den;
  const Real drsph_dy_harm = SQR(rsph) * rsph * y_harm.v / grad_den;
  const Real drsph_dz_harm = rsph * (SQR(rsph) + SQR(a)) * z_harm.v / grad_den;
  Dual res;
  res.v = rsph;
  // chain rule: res.d[k] = sum_i (dR/dX^i) * (input i).d[k]
  for (int k = 0; k < 3; ++k) {
    res.d[k] = drsph_dx_harm * x_harm.d[k] + drsph_dy_harm * y_harm.d[k] +
               drsph_dz_harm * z_harm.d[k];
  }
  return res;
}

//----------------------------------------------------------------------------------------
//! \fn void LowerMetric
//! \brief lower metric g_{mu nu} at harmonic (X,Y,Z): the metric for T = Real,
//! and the metric with its gradient d_k g_{mu nu} for T = Dual. Notation as in
//! the file header.
//!
//! The metric contains first derivatives of R and n (in f d_j R and
//! Lambda^p_i). They are written out as T-valued expressions (grad_rsph,
//! grad_nhat) instead of being read off from Dual gradients, so that for
//! T = Dual their own derivatives (second derivatives of R and n, needed in
//! d_k g_{mu nu}) are propagated too.
//!
//! Near R = 0 the metric is not even Lorentzian in floating point. Points with
//! R < R_floor = max(0.1 R_H, 1e-3) are therefore evaluated at Z = R_floor
//! (Z >= 0) or Z = -R_floor (Z < 0), which gives R >= R_floor since |Z| <= R.
//! For R_H > 2e-3 (a < 0.999998) such points are inside the default excision
//! radius R = R_H/2, so the clamp only keeps the metric finite in excised
//! cells. The clamped Z is a constant, so d_Z g_{mu nu} = 0 there.
//!
//! Variables and the symbols they hold:
//!   x_harm, y_harm, z_harm   X, Y, Z (harmonic coordinates of the mesh point)
//!   mass, r_horizon          M, R_H
//!   rsph, r_bl               R, r = R + M
//!   grad_rsph[j]             d_j R
//!   nhat[p]                  n^p = n_p (CKS components)
//!   grad_nhat[p][j]          d_j n^p
//!   jacobian[p][i]           Lambda^p_i
//!   dt_drsph                 f
//!   h_ks                     H
//!   l_harm[j]                l_j (harmonic components of l)
//! The CKS coordinates x^p are never formed; the CKS quantities needed (n^p,
//! H) are written directly in harmonic variables.

template <typename T>
KOKKOS_INLINE_FUNCTION void
LowerMetric(const T &x_harm, const T &y_harm, const T &z_harm_in,
            const bool minkowski, const Real a, T g[][4]) {
  const Real mass = minkowski ? 0.0 : 1.0;                    // M
  const Real r_horizon = sqrt(fmax(SQR(mass) - SQR(a), 0.0)); // R_H

  // clamp Z near the R = 0 singularity (see above)
  const Real r_floor = fmax(0.1 * r_horizon, 1.0e-3);
  T z_harm = z_harm_in;
  if (SpheroidalRadius(Value(x_harm), Value(y_harm), Value(z_harm_in), a) <
      r_floor) {
    z_harm =
        Constant((Value(z_harm_in) >= 0.0) ? r_floor : -r_floor, z_harm_in);
  }

  // spheroidal radius R and frequently used combinations
  T rsph = Radius(x_harm, y_harm, z_harm, a);              // R
  T rsph2 = rsph * rsph;                                   // R^2
  T rsph2_a2 = rsph2 + SQR(a);                             // R^2 + a^2
  T grad_den = rsph2 * rsph2 + SQR(a) * (z_harm * z_harm); // R^4 + a^2 Z^2

  // grad_rsph[j] = d_j R, a harmonic covector (see Radius())
  T grad_rsph[3] = {rsph2 * rsph * x_harm / grad_den,
                    rsph2 * rsph * y_harm / grad_den,
                    rsph * rsph2_a2 * z_harm / grad_den};

  // nhat[p] = n^p = n_p, the CKS components of the unit vector n. The comments
  // write n^p where n is the displacement in x^p = X^p + M n^p (grad_nhat,
  // jacobian) and n_p where it is the spatial part of l_A = (1, n_p) (l_harm).
  // The two agree, and so do their derivatives (see the file header).
  //
  // grad_nhat[p][j] = d_j n^p, with p a CKS component and j a harmonic
  // derivative direction (code index 0,1,2 = x,y,z for p and X,Y,Z for j).
  // Differentiating
  //   n^x = (R X + a Y)/(R^2+a^2),  n^y = (R Y - a X)/(R^2+a^2),  n^z = Z/R
  // gives a term through R (set in the loop) plus a term from the explicit
  // X, Y, Z dependence (added after the loop):
  //   d_j n^x = (X - 2R n^x) d_j R/(R^2+a^2)
  //             + (R delta_Xj + a delta_Yj)/(R^2+a^2)
  //   d_j n^y = (Y - 2R n^y) d_j R/(R^2+a^2)
  //             + (R delta_Yj - a delta_Xj)/(R^2+a^2)
  //   d_j n^z = -Z d_j R/R^2 + delta_Zj/R
  T nhat[3] = {(rsph * x_harm + a * y_harm) / rsph2_a2,
               (rsph * y_harm - a * x_harm) / rsph2_a2, z_harm / rsph};
  T grad_nhat[3][3];
  for (int j = 0; j < 3; ++j) {
    grad_nhat[0][j] = (x_harm - 2.0 * rsph * nhat[0]) * grad_rsph[j] / rsph2_a2;
    grad_nhat[1][j] = (y_harm - 2.0 * rsph * nhat[1]) * grad_rsph[j] / rsph2_a2;
    grad_nhat[2][j] = -z_harm * grad_rsph[j] / rsph2;
  }
  grad_nhat[0][0] = grad_nhat[0][0] + rsph / rsph2_a2;
  grad_nhat[0][1] = grad_nhat[0][1] + a / rsph2_a2;
  grad_nhat[1][0] = grad_nhat[1][0] - a / rsph2_a2;
  grad_nhat[1][1] = grad_nhat[1][1] + rsph / rsph2_a2;
  grad_nhat[2][2] = grad_nhat[2][2] + 1.0 / rsph;

  // spatial Jacobian of the map x^p = X^p + M n^p:
  //   jacobian[p][i] = Lambda^p_i = d_i x^p = delta^p_i + M d_i n^p
  T jacobian[3][3];
  for (int p = 0; p < 3; ++p) {
    for (int i = 0; i < 3; ++i) {
      jacobian[p][i] = mass * grad_nhat[p][i] + ((p == i) ? 1.0 : 0.0);
    }
  }

  // f = dt/dR = 2M/(R + R_H), from t = T + 2M ln((R + R_H)/(2M))
  T dt_drsph = (2.0 * mass) / (rsph + r_horizon);
  // Boyer-Lindquist radius r = R + M
  T r_bl = rsph + mass;
  // H = M r^3/(r^4 + a^2 z^2), written with z = Z r/R
  T h_ks =
      mass * r_bl * rsph2 / (r_bl * r_bl * rsph2 + SQR(a) * (z_harm * z_harm));

  // l_harm[j] = l_j: harmonic spatial components of the null covector l, whose
  // CKS components are l_A = (l_t, l_p) = (1, n_p). The pullback of 2H l_A l_B
  // is 2H l_mu l_nu.
  //
  // (1) Covector transformation:
  //       l_j = (dx^A/dX^j) l_A = (d_j t) l_t + (d_j x^p) l_p.
  //
  // (2) - Time part: t depends on X^j only through R, so d_j t = f d_j R, and
  //       with l_t = 1 the term is f d_j R.
  //     - Spatial part: d_j x^p = Lambda^p_j = delta^p_j + M d_j n^p, so
  //       l_p Lambda^p_j = delta^p_j n_p + M n_p d_j n^p.
  //
  // (3) Differentiating |n|^2 = delta_pq n^p n^q = 1 gives n_p d_j n^p = 0, so
  //     the last term vanishes:
  //       l_j = f d_j R + delta^p_j n_p.
  //     |n| = 1 holds identically in (X,Y,Z), to roundoff in R, because
  //       |n|^2 = (X^2+Y^2)/(R^2+a^2) + Z^2/R^2 = 1
  //     is equivalent to the quartic that defines R, so its derivatives vanish
  //     too and the term is zero also for T = Dual. The one exception, the
  //     smooth floor R < 1e-6 in SpheroidalRadius(), is never reached because
  //     of the Z clamp above.
  //
  // The time component l_0 = (d_0 t) l_t + (d_0 x^p) l_p = 1 enters g_00 and
  // g_0j below implicitly.
  //
  // In code, delta^p_j n_p is nhat[j]: the CKS component whose label matches
  // the harmonic index j.
  T l_harm[3];
  for (int j = 0; j < 3; ++j) {
    l_harm[j] = nhat[j] + dt_drsph * grad_rsph[j];
  }

  // assemble g_{mu nu} (see the file header)
  //   g_00 = -1 + 2H
  g[0][0] = 2.0 * h_ks - 1.0;
  //   g_0j = -f d_j R + 2H l_j
  for (int j = 0; j < 3; ++j) {
    g[0][j + 1] = 2.0 * h_ks * l_harm[j] - dt_drsph * grad_rsph[j];
    g[j + 1][0] = g[0][j + 1];
  }
  //   g_ij = delta_pq Lambda^p_i Lambda^q_j - f^2 d_i R d_j R + 2H l_i l_j
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      T gij = jacobian[0][i] * jacobian[0][j] +
              jacobian[1][i] * jacobian[1][j] + jacobian[2][i] * jacobian[2][j];
      gij = gij - dt_drsph * dt_drsph * grad_rsph[i] * grad_rsph[j] +
            2.0 * h_ks * l_harm[i] * l_harm[j];
      g[i + 1][j + 1] = gij;
      g[j + 1][i + 1] = gij;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void InvertMetric
//! \brief upper metric from lower metric via the 3+1 split.
//!
//! Writing g_00 = -alpha^2 + beta_i beta^i, g_0i = beta_i, g_ij = gamma_ij,
//! the inverse is
//!   g^00 = -1/alpha^2,  g^0i = beta^i/alpha^2,
//!   g^ij = gamma^ij - beta^i beta^j/alpha^2,
//! which only needs the inverse of the 3x3 spatial block.

KOKKOS_INLINE_FUNCTION
void InvertMetric(const Real glower[][4], Real gupper[][4]) {
  // inverse spatial metric gamma^ij
  const Real det_gamma =
      adm::SpatialDet(glower[1][1], glower[1][2], glower[1][3], glower[2][2],
                      glower[2][3], glower[3][3]);
  Real gamma_uu[3][3];
  adm::SpatialInv(1.0 / det_gamma, glower[1][1], glower[1][2], glower[1][3],
                  glower[2][2], glower[2][3], glower[3][3], &gamma_uu[0][0],
                  &gamma_uu[0][1], &gamma_uu[0][2], &gamma_uu[1][1],
                  &gamma_uu[1][2], &gamma_uu[2][2]);
  gamma_uu[1][0] = gamma_uu[0][1];
  gamma_uu[2][0] = gamma_uu[0][2];
  gamma_uu[2][1] = gamma_uu[1][2];

  // shift beta^i = gamma^ij beta_j with beta_j = g_0j
  Real beta_u[3];
  for (int i = 0; i < 3; ++i) {
    beta_u[i] = gamma_uu[i][0] * glower[0][1] + gamma_uu[i][1] * glower[0][2] +
                gamma_uu[i][2] * glower[0][3];
  }
  // lapse squared alpha^2 = beta^i beta_i - g_00
  const Real alpha2 = beta_u[0] * glower[0][1] + beta_u[1] * glower[0][2] +
                      beta_u[2] * glower[0][3] - glower[0][0];
  const Real inv_alpha2 = 1.0 / alpha2;

  gupper[0][0] = -inv_alpha2;
  for (int i = 0; i < 3; ++i) {
    gupper[0][i + 1] = beta_u[i] * inv_alpha2;
    gupper[i + 1][0] = gupper[0][i + 1];
    for (int j = 0; j < 3; ++j) {
      gupper[i + 1][j + 1] =
          gamma_uu[i][j] - beta_u[i] * beta_u[j] * inv_alpha2;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeMetricAndInverse
//! \brief lower and upper metric at harmonic (X,Y,Z) = (x_harm, y_harm, z_harm)

KOKKOS_INLINE_FUNCTION
void ComputeMetricAndInverse(Real x_harm, Real y_harm, Real z_harm,
                             bool minkowski, Real a, Real glower[][4],
                             Real gupper[][4]) {
  // T = Real: values only
  LowerMetric(x_harm, y_harm, z_harm, minkowski, a, glower);
  InvertMetric(glower, gupper);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeMetricDerivatives
//! \brief partial derivatives of the lower metric,
//! dg_dx1[mu][nu] = d_X g_{mu nu}, dg_dx2 = d_Y g, dg_dx3 = d_Z g,
//! from LowerMetric() with T = Dual and unit seeds.

KOKKOS_INLINE_FUNCTION
void ComputeMetricDerivatives(Real x_harm, Real y_harm, Real z_harm,
                              bool minkowski, Real a, Real dg_dx1[][4],
                              Real dg_dx2[][4], Real dg_dx3[][4]) {
  // seeds: d_k X = (1,0,0), d_k Y = (0,1,0), d_k Z = (0,0,1)
  const Dual x_harm_dual = MakeDual(x_harm, 1.0, 0.0, 0.0);
  const Dual y_harm_dual = MakeDual(y_harm, 0.0, 1.0, 0.0);
  const Dual z_harm_dual = MakeDual(z_harm, 0.0, 0.0, 1.0);
  Dual g[4][4];
  LowerMetric(x_harm_dual, y_harm_dual, z_harm_dual, minkowski, a, g);
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      dg_dx1[mu][nu] = g[mu][nu].d[0];
      dg_dx2[mu][nu] = g[mu][nu].d[1];
      dg_dx3[mu][nu] = g[mu][nu].d[2];
    }
  }
  return;
}

} // namespace harmonic_kerr

#endif // COORDINATES_HARMONIC_KERR_HPP_
