
namespace {

// Useful container for physical parameters
struct bhl_collision_pgen {
  Real spin;             // black hole spin
  Real dexcise, pexcise; // excision parameters

  Real gamma_adi; // EOS parameters

  // Collision ejecta parameters

  // Radius mask for computing gravitational drag
  Real grav_drag_mask_rmax;
};

} // namespace
