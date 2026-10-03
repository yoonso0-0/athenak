# Spherical inner mask -- visual demos

Ad-hoc, qualitative demos of the `<sphere_mask>` feature (`src/hydro/hydro_sphere_mask.cpp`)
and the point-particle gravity source term. They are **not** regression tests; the
pytest checks live in `tst/test_suite/nr/test_nr_sphere_mask_*.py`.

Each demo is a single input file in `inputs/hydro/`, run several ways through command-line
overrides, so there is nothing to keep in sync. Run output goes to
`tst/sphere_mask_demo/output_*/` and videos to `tst/sphere_mask_demo/*.mp4` (both
gitignored; videos need `ffmpeg`).

| Demo | Input | Executable | Run | Render |
|---|---|---|---|---|
| Implosion | `sphere_mask_implosion_demo.athinput` | `build/src/athena` | `run_demo.py implosion` | `render_implosion.py` |
| Wind / accretion | `sphere_mask_wind_demo.athinput` | `build_blast/src/athena` | `run_demo.py wind_accretion` | `render_wind_accretion.py` |
| Gravity + mask | `sphere_mask_gravity_blast.athinput` | `build_blast_omp/src/athena` | `gravity_blast.py` | (same script) |

`blast.cpp` is a `UserProblem`-only pgen (not selectable by `pgen_name` in the stock
build), so the wind and gravity demos need a separate build:

```
cmake -S . -B build_blast -D PROBLEM=fluids/blast -D Athena_ENABLE_MPI=OFF -D Athena_ENABLE_OPENMP=OFF
cmake --build build_blast -j
```

(`build_blast_omp` is the same with OpenMP on; `gravity_blast.py --athena PATH` and
`run_demo.py --athena PATH` accept any other executable.)

## Implosion (all BCs vs. no mask)

300x300 octant Liska & Wendroff implosion, ppm4/hllc, `radius=0.075`, run to `t=2`
with `dt=0.01` output. `run_demo.py implosion` runs five variants:

- `baseline`: `sphere_mask/enabled=false`
- `dirichlet1`: interior pinned to the implosion's exterior state, at rest
- `dirichlet2`: same with `sphere_mask/vel1=2.0`, a supersonic outflow from the "hole"
- `reflecting`, `absorbing`: mirror BCs

`render_implosion.py` puts the five density fields side by side. Mirror BCs show a faint
"star" pattern in the few cells nearest the sphere center. It is an artifact of the mirror
map `x_mirror = x*(2R - r)/r`, whose direction is ill-conditioned as `r -> 0`; it does not
affect the surface at `r ~ R` and cannot leak outward, because the interior is overwritten
every stage.

## Wind vs. accretion

`spherical_wind` BC with the interior `dens`/`eint` equal to the ambient state, so the
imposed radial velocity is the only difference between interior and exterior. Uniform
ambient medium from the blast pgen (`outer_radius` < mask radius, so its blast region is
fully overwritten by the mask). `run_demo.py wind_accretion` runs `vel_r=+1.0` (wind) and
`vel_r=-1.0` (accretion), both subsonic (Mach ~0.85), to `t=0.3`.

- **Wind**: a piston problem; the boundary advancing into gas at rest always shocks it, so
  a weaker, gradual forward shock forms (density from a ~0.5 hollow to a ~1.4 shell).
- **Accretion**: a converging rarefaction, spherically symmetric, density falling
  smoothly from 1.0 to ~0.6 at the sphere.
- `vel_r=+-3.0` (override `sphere_mask/vel_r=...`) gives a much stronger wind side (density
  0.18 to 5.3, shock reaches the walls at `t~0.29`); accretion barely changes.
- `sphere_mask/vel_r=0.0` should make the mask a complete no-op: the domain stays at the
  ambient state with max|velocity| at roundoff (~1e-16).

## Gravity + mask

`gravity_blast.py` runs the point-particle `1/r` gravity source term
(`<hydro_srcterms>/point_particle_gravity_at_center`, GM=1, softening 0.05) together with a
reflecting mask of radius 0.1 in a uniform ambient medium (`drat = prat = 1`, so no real
blast), `[-1,1]^2`, 400x400, ppm4, `t=1.5`. It writes `sphere_mask_gravity_blast.png`
(density and radial velocity snapshots): gas free-falls onto the point mass and piles up in
a thin shell at the mask surface, depleting the far field. Extra arguments are athena
overrides, e.g. `gravity_blast.py --tlim 2.0 --radius 0.3 hydro_srcterms/gravity_softening_length=0.1`.
The interior of the mask is not physically evolved; it mirrors the state just outside.
The committed PNG is a reference rendering and is overwritten on every run.
