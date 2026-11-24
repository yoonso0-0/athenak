//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file metric_dump.cpp
//! \brief Standalone tool (not part of athena) that dumps the GR metric used
//! by the code at a list of points, for cross-checks against other codes.
//!
//! Build (in an athena build directory):  make metric_dump
//! The coordinates are those of that build: Cartesian Kerr-Schild, or
//! Cartesian harmonic if configured with -DAthena_HARMONIC_KS=ON (M = 1).
//! Evaluates the same inline functions the GR solver uses
//! (ComputeMetricAndInverse, ComputeMetricDerivatives, BoyerLindquistRadius in
//! coordinates/cartesian_ks.hpp).
//!
//! Usage:  metric_dump [-a spin] [-m] [-o dir] points.txt
//!   -a spin   black hole spin a/M (default 0)
//!   -m        Minkowski (flat) metric
//!   -o dir    write the points to dir/coords.txt ("x y z" per line) and each
//!             other column to its own file dir/<column>.txt (e.g.
//!             dir/g_tt.txt), one value per line in the order of the points,
//!             creating dir if needed. Default: all columns as one table on
//!             standard output.
//!   points    ASCII file, one point "x y z" per line; text after '#' and blank
//!             lines are ignored. "-" reads standard input.
//!
//! Columns (55 per point):
//!   x y z, r_BL, sqrt_mg = sqrt(-g),
//!   g_{mu nu} (g_..), g^{mu nu} (gu_..), d_x g_{mu nu} (dx_g_..),
//!   d_y g_{mu nu} (dy_g_..), d_z g_{mu nu} (dz_g_..),
//! each group of 10 in the order tt tx ty tz xx xy xz yy yz zz. sqrt(-g) is
//! computed from the dumped metric as sqrt(-det(gamma_ij)/g^00) in both builds
//! (SqrtMinusDetMetric returns exactly 1 for Kerr-Schild), so it also checks
//! the metric to roundoff.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cartesian_ks.hpp"

namespace {

[[noreturn]] void Fatal(const std::string &message) {
  std::cerr << "metric_dump: " << message << std::endl;
  std::exit(EXIT_FAILURE);
}

void Usage() {
  std::cerr << "usage: metric_dump [-a spin] [-m] [-o dir] points.txt"
            << std::endl;
  std::exit(EXIT_FAILURE);
}

// component order of the 10 independent entries of a symmetric 4x4 matrix
constexpr int kIdx[10][2] = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1},
                             {1, 2}, {1, 3}, {2, 2}, {2, 3}, {3, 3}};
constexpr int kNumColumns = 55;
// the first kNumCoords columns (x, y, z) share the file coords.txt with -o
constexpr int kNumCoords = 3;

// column names, in output order
std::vector<std::string> ColumnNames() {
  const char *label = "txyz";
  std::vector<std::string> names = {"x", "y", "z", "r_BL", "sqrt_mg"};
  for (const char *group : {"g_", "gu_", "dx_g_", "dy_g_", "dz_g_"}) {
    for (int n = 0; n < 10; ++n) {
      names.push_back(std::string(group) + label[kIdx[n][0]] +
                      label[kIdx[n][1]]);
    }
  }
  return names;
}

// header lines shared by the table and the per-column files
void PrintMetadata(FILE *out, const bool minkowski, const Real spin) {
  std::fprintf(out, "# AthenaK metric dump (M = 1)\n");
  std::fprintf(out, "# coordinates = %s\n",
               HARMONIC_KS_ENABLED ? "cartesian_harmonic"
                                   : "cartesian_kerr_schild");
  std::fprintf(out, "# minkowski = %s\n", minkowski ? "true" : "false");
  std::fprintf(out, "# a = %.17e\n", spin);
}

// all columns at one point
void EvaluatePoint(const Real x, const Real y, const Real z,
                   const bool minkowski, const Real spin,
                   Real row[kNumColumns]) {
  Real glower[4][4], gupper[4][4];
  Real dg[3][4][4];
  ComputeMetricAndInverse(x, y, z, minkowski, spin, glower, gupper);
  ComputeMetricDerivatives(x, y, z, minkowski, spin, dg[0], dg[1], dg[2]);
  // det g = -alpha^2 det(gamma_ij), alpha^2 = -1/g^00
  const Real det3 = adm::SpatialDet(glower[1][1], glower[1][2], glower[1][3],
                                    glower[2][2], glower[2][3], glower[3][3]);
  int col = 0;
  row[col++] = x;
  row[col++] = y;
  row[col++] = z;
  row[col++] = BoyerLindquistRadius(x, y, z, minkowski, spin);
  row[col++] = std::sqrt(-det3 / gupper[0][0]);
  for (int n = 0; n < 10; ++n)
    row[col++] = glower[kIdx[n][0]][kIdx[n][1]];
  for (int n = 0; n < 10; ++n)
    row[col++] = gupper[kIdx[n][0]][kIdx[n][1]];
  for (int k = 0; k < 3; ++k) {
    for (int n = 0; n < 10; ++n)
      row[col++] = dg[k][kIdx[n][0]][kIdx[n][1]];
  }
}

} // namespace

int main(int argc, char *argv[]) {
  Real spin = 0.0;
  bool minkowski = false;
  const char *points_file = nullptr;
  const char *out_dir = nullptr;
  for (int n = 1; n < argc; ++n) {
    if (std::strcmp(argv[n], "-a") == 0) {
      if (++n >= argc)
        Usage();
      char *end;
      spin = std::strtod(argv[n], &end);
      if (*end != '\0')
        Fatal(std::string("invalid spin '") + argv[n] + "'");
    } else if (std::strcmp(argv[n], "-m") == 0) {
      minkowski = true;
    } else if (std::strcmp(argv[n], "-o") == 0) {
      if (++n >= argc)
        Usage();
      out_dir = argv[n];
    } else if (points_file == nullptr) {
      points_file = argv[n];
    } else {
      Usage();
    }
  }
  if (points_file == nullptr)
    Usage();
  if (minkowski)
    spin = 0.0;

  std::ifstream file;
  std::istream *in = &std::cin;
  if (std::strcmp(points_file, "-") != 0) {
    file.open(points_file);
    if (!file.is_open()) {
      Fatal(std::string("cannot open '") + points_file + "'");
    }
    in = &file;
  }

  // with -o, coords.txt (x y z) and one file per other column; else one table
  // on stdout
  const std::vector<std::string> names = ColumnNames();
  std::vector<FILE *> outs; // outs[0]: coords, outs[col - kNumCoords + 1]: col
  if (out_dir != nullptr) {
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec)
      Fatal(std::string("cannot create directory '") + out_dir + "'");
    std::vector<std::string> file_names = {"coords"};
    file_names.insert(file_names.end(), names.begin() + kNumCoords,
                      names.end());
    for (const std::string &name : file_names) {
      const std::string path =
          (std::filesystem::path(out_dir) / (name + ".txt")).string();
      FILE *out = std::fopen(path.c_str(), "w");
      if (out == nullptr)
        Fatal("cannot open '" + path + "' for writing");
      PrintMetadata(out, minkowski, spin);
      std::fprintf(out, "# quantity = %s\n",
                   name == "coords" ? "x y z" : name.c_str());
      outs.push_back(out);
    }
  } else {
    PrintMetadata(stdout, minkowski, spin);
    std::fprintf(stdout, "# columns:");
    for (int col = 0; col < kNumColumns; ++col)
      std::fprintf(stdout, " [%d]%s", col + 1, names[col].c_str());
    std::fprintf(stdout, "\n");
  }

  std::string line;
  int lineno = 0;
  int npoints = 0;
  while (std::getline(*in, line)) {
    ++lineno;
    std::istringstream stream(line.substr(0, line.find('#')));
    stream >> std::ws;
    if (stream.eof())
      continue;
    Real x, y, z;
    std::string extra;
    if (!(stream >> x >> y >> z) || (stream >> extra)) {
      Fatal("line " + std::to_string(lineno) + " is not of the form 'x y z'");
    }
    Real row[kNumColumns];
    EvaluatePoint(x, y, z, minkowski, spin, row);
    if (out_dir != nullptr) {
      std::fprintf(outs[0], "%.17e %.17e %.17e\n", row[0], row[1], row[2]);
      for (int col = kNumCoords; col < kNumColumns; ++col)
        std::fprintf(outs[col - kNumCoords + 1], "%.17e\n", row[col]);
    } else {
      for (int col = 0; col < kNumColumns; ++col)
        std::fprintf(stdout, col == 0 ? "%.17e" : " %.17e", row[col]);
      std::fprintf(stdout, "\n");
    }
    ++npoints;
  }

  if (out_dir != nullptr) {
    for (FILE *out : outs)
      std::fclose(out);
    std::cerr << "metric_dump: wrote " << npoints << " points to "
              << outs.size() << " files in '" << out_dir << "'" << std::endl;
  }
  return 0;
}
