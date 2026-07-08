// =============================================================================
// source_config.rs
//
// the Overlay -> KernelSet source dispatch (DISTILLED step 3c). an `Overlay`
// (symbi_hydro's source-monoid value: point_mass(gm,xm) + uniform_accel(g) + ..)
// is the user's declaration of WHAT sources act; this module decides HOW each
// runs and binds it into a substrate kernel set:
//
//   - prefer FUSED when the AOT layer baked a fused godunov for the family
//     (one launch covers div(F) + source + integrator).
//   - else fall back to the standalone ADDITIVE pass — proven bit-for-bit equal
//     to fused (godunov_with_fused_source + additive_source_equals_fused_evolve).
//
// so the execution strategy is a property of what's BAKED, not a user choice;
// the same Overlay runs fused on a baked family and additive on an unbaked one,
// with identical numerics either way.
//
// scope: external source families (point_mass, uniform_accel) — baked cartesian.
// the in-godunov metric/geometric source is separate (always fused, never an
// Overlay). the substrate binds the FIRST fused family (the documented
// single-family limit, simulation_laws::derive_fused_binding).
//
// usage:
//  let sub = configure_source(
//      IsoSubstrateKernelSet::<Mem, f64, 2>::new(cs, cfl, &sim.geom.allocated),
//      &point_mass(gm, vec![0.0, 0.0]), "iso", 2,
//  );
// =============================================================================

use symbi_hydro::Overlay;
use symbi_ir::algebra::Scalar;
use symbi_algebra::OrderedNumeric;
use symbi_xpu::MemorySpace;

use crate::regimes::substrate::IsoSubstrateKernelSet;
use crate::regimes::substrate_kernels::FusedSourceBinding;
use crate::regimes::substrate_newton::AdiabaticSubstrateKernelSet;

/// the source-execution seam a substrate kernel set exposes: route a binding to
/// EITHER the fused godunov or the additive pass. lets `configure_source` write
/// the fused-vs-additive decision ONCE, regime-generically.
pub trait ConfigureSource: Sized {
    /// fold the source into the godunov kernel (the fast, baked path).
    fn route_fused(self, binding: FusedSourceBinding) -> Self;
    /// run the source as a separate per-stage additive pass (the general path).
    fn route_additive(self, binding: FusedSourceBinding) -> Self;
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> ConfigureSource
    for IsoSubstrateKernelSet<Mem, Sc, D>
{
    fn route_fused(self, b: FusedSourceBinding) -> Self {
        self.with_fused_source(b)
    }
    fn route_additive(self, b: FusedSourceBinding) -> Self {
        self.with_additive_source(b)
    }
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> ConfigureSource
    for AdiabaticSubstrateKernelSet<Mem, Sc, D>
{
    fn route_fused(self, b: FusedSourceBinding) -> Self {
        self.with_fused_source(b)
    }
    fn route_additive(self, b: FusedSourceBinding) -> Self {
        self.with_additive_source(b)
    }
}

/// bind an `Overlay`'s source into `ks`, choosing fused-when-baked else additive
/// by consulting the AOT kernel registry. `prefix` is the regime kernel prefix
/// (`"iso"` / `"adiabatic"`), `d` the grid dimension.
pub fn configure_source<K: ConfigureSource>(ks: K, overlay: &Overlay, prefix: &str, d: usize) -> K {
    configure_source_with(ks, overlay, prefix, d, |name| {
        symbi_aot::kernel_by_name::<f64>(name).is_some()
    })
}

/// the testable core of [`configure_source`]: the bake check is injected as
/// `is_baked` so both branches (fused vs additive fallback) are exercisable
/// without needing an actually-unbaked family. an Overlay with no fused family
/// leaves `ks` unchanged (a source-free run).
pub fn configure_source_with<K: ConfigureSource>(
    ks: K,
    overlay: &Overlay,
    prefix: &str,
    d: usize,
    is_baked: impl Fn(&str) -> bool,
) -> K {
    // the substrate binds the FIRST fused family (single-family limit); a 2nd
    // family awaits a composite slug or a spec-list additive pass (parked).
    let Some(family) = overlay.fused.first() else {
        return ks;
    };
    let (slug, pairs) = family.into_binding_pair();
    let binding = FusedSourceBinding::from_pair((slug, pairs));
    let fused_kernel = format!("{prefix}_godunov_stage_with_{slug}_{d}d");
    if is_baked(&fused_kernel) {
        ks.route_fused(binding)
    } else {
        ks.route_additive(binding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_hydro::{point_mass, uniform_accel, Overlay};
    use symbi_xpu::HostMemory;
    use symbi_algebra::{Domain, Space};

    fn iso_set() -> IsoSubstrateKernelSet<HostMemory, f64, 2> {
        // a tiny alloc domain just to size the kernel-set scratch fields; the
        // configure_source decision is independent of grid size.
        let alloc = Domain::<2>::new([
            Space { name: "x", lo: 0, hi: 8 },
            Space { name: "y", lo: 0, hi: 8 },
        ]);
        IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(0.1, 0.4, &alloc)
    }

    #[test]
    fn picks_fused_when_baked() {
        // is_baked = true -> the binding folds into godunov; no additive pass.
        let ks = configure_source_with(iso_set(), &point_mass(1.0, vec![0.0, 0.0], 0.0), "iso", 2, |_| true);
        assert!(ks.fused_source.is_some(), "baked family must route fused");
        assert!(ks.additive_source.is_none(), "fused path must NOT also set additive");
        assert_eq!(ks.fused_source.as_ref().unwrap().source_id, "point_mass_grav");
    }

    #[test]
    fn falls_back_to_additive_when_unbaked() {
        // is_baked = false -> the SAME binding routes to the additive pass.
        let ks = configure_source_with(iso_set(), &point_mass(1.0, vec![0.0, 0.0], 0.0), "iso", 2, |_| false);
        assert!(ks.additive_source.is_some(), "unbaked family must route additive");
        assert!(ks.fused_source.is_none(), "additive fallback must NOT also set fused");
        assert_eq!(ks.additive_source.as_ref().unwrap().source_id, "point_mass_grav");
    }

    #[test]
    fn empty_overlay_leaves_set_source_free() {
        let ks = configure_source_with(iso_set(), &Overlay::none(), "iso", 2, |_| true);
        assert!(ks.fused_source.is_none() && ks.additive_source.is_none());
    }

    #[test]
    fn real_registry_bakes_point_mass_iso_2d() {
        // end-to-end against the ACTUAL AOT registry: point_mass IS baked for
        // iso 2d, so the production `configure_source` routes it fused — proves
        // the kernel-name format + the registry consult agree with the bake.
        let ks = configure_source(iso_set(), &point_mass(1.0, vec![0.0, 0.0], 0.0), "iso", 2);
        assert!(ks.fused_source.is_some(), "point_mass IS baked for iso 2d -> fused");
    }

    #[test]
    fn uniform_accel_routes_too() {
        // the other family resolves its own slug + scalars (g_ext_k).
        let ks = configure_source_with(iso_set(), &uniform_accel(vec![0.0, -1.0]), "iso", 2, |_| true);
        assert_eq!(ks.fused_source.as_ref().unwrap().source_id, "uniform_accel");
    }
}
