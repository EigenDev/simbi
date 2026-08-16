// =============================================================================
// named_call.rs
//
// the name-keyed CPU-kernel invocation for host + test code. resolves an emitted
// kernel and its buffer/scalar manifest by name, binds every buffer + scalar BY
// FIELD NAME (order-independent), then runs the structured slice ABI.
//
// this is the only sanctioned way for host code to call an emitted kernel. the
// positional `__raw(field0, field1, .., gamma)` form is codegen-internal: its
// arity changes whenever a builder adds/removes an input, silently breaking
// hand-written positional callers. binding by name turns that into a loud, named
// failure — a missing/extra field panics with the manifest's expected names — so
// adding a kernel input surfaces immediately at every stale caller.
//
// usage:
//  NamedKernel::new("rmhd_c2p_1d")
//      .input("cons.den", &den).input("cons.mom_0", &sx) // .. by manifest name
//      .output("prim.rho", &mut rho).output("prim.vel_0", &mut vx)
//      .grid(&[n]).dom_lo(&[0])
//      .scalar("gamma", gamma)
//      .run::<f64>();
// =============================================================================

use crate::{BufHandle, CpuField, CpuFieldMut, OrderedNumeric, Scalar, kernel_by_name};
use symbi_ir::{
    FieldBind, ScalarBind, ScalarRef, kernel_bindings_from_ir, kernel_scalar_params_typed_from_ir,
};

/// a buffer binding awaiting manifest ordering: its name + data handle + optional
/// explicit layout (`lo`/`extent`). default layout is the 1D contiguous case
/// (`lo = [0]`, `extent = [len]`) — the override is for staggered face/edge domains.
struct Binding<'a, S> {
    name: String,
    handle: BufHandle<'a, S>,
    layout: Option<(&'a [i32], &'a [u32])>,
}

/// name-keyed kernel-invocation builder. accumulate inputs/outputs/scalars/ints by
/// name in any order, then `run` reorders them against the kernel's manifest.
pub struct NamedKernel<'a, S> {
    name: &'a str,
    bufs: Vec<Binding<'a, S>>,
    scalars: Vec<(&'a str, S)>,
    ints: Vec<(&'a str, i32)>,
    grid: &'a [u32],
    dom_lo: &'a [i32],
}

impl<'a, S: Scalar + OrderedNumeric> NamedKernel<'a, S> {
    pub fn new(name: &'a str) -> Self {
        Self {
            name,
            bufs: Vec::new(),
            scalars: Vec::new(),
            ints: Vec::new(),
            grid: &[],
            dom_lo: &[],
        }
    }

    /// bind a read-only input buffer by its manifest field name (1D layout).
    pub fn input(mut self, name: &str, data: &'a [S]) -> Self {
        self.bufs.push(Binding {
            name: name.to_string(),
            handle: BufHandle::Host(data),
            layout: None,
        });
        self
    }

    /// bind an output (or in-place) buffer by its manifest field name (1D layout).
    pub fn output(mut self, name: &str, data: &'a mut [S]) -> Self {
        self.bufs.push(Binding {
            name: name.to_string(),
            handle: BufHandle::HostMut(data),
            layout: None,
        });
        self
    }

    /// bind a read-only input with an EXPLICIT layout (staggered / multi-axis domains).
    pub fn input_at(mut self, name: &str, data: &'a [S], lo: &'a [i32], extent: &'a [u32]) -> Self {
        self.bufs.push(Binding {
            name: name.to_string(),
            handle: BufHandle::Host(data),
            layout: Some((lo, extent)),
        });
        self
    }

    /// bind an output with an EXPLICIT layout (staggered / multi-axis domains).
    pub fn output_at(
        mut self,
        name: &str,
        data: &'a mut [S],
        lo: &'a [i32],
        extent: &'a [u32],
    ) -> Self {
        self.bufs.push(Binding {
            name: name.to_string(),
            handle: BufHandle::HostMut(data),
            layout: Some((lo, extent)),
        });
        self
    }

    pub fn scalar(mut self, name: &'a str, value: S) -> Self {
        self.scalars.push((name, value));
        self
    }

    pub fn int(mut self, name: &'a str, value: i32) -> Self {
        self.ints.push((name, value));
        self
    }

    pub fn grid(mut self, grid: &'a [u32]) -> Self {
        self.grid = grid;
        self
    }

    pub fn dom_lo(mut self, dom_lo: &'a [i32]) -> Self {
        self.dom_lo = dom_lo;
        self
    }

    /// resolve + reorder against the manifest, then run on the CPU. panics — with the
    /// manifest's expected names — if any declared buffer/scalar is unbound or extra.
    pub fn run(self) {
        let (kernel, ir) = kernel_by_name::<S>(self.name)
            .unwrap_or_else(|| panic!("no kernel '{}' in registry", self.name));
        // (field path, is_output) in canonical order: inputs then outputs. the manifest is typed
        // (FieldBind); this test harness binds buffers by their runtime-path string, so flatten
        // each bind back to its name here.
        let want: Vec<(String, bool)> = kernel_bindings_from_ir(ir)
            .into_iter()
            .map(|(f, is_out)| (f.name(), is_out))
            .collect();

        // reorder the provided buffers to match the manifest, by name. a manifest field
        // with no provided buffer (or vice versa) raises a named panic, not silent drift.
        // default layout is 1D contiguous (lo = [0], extent = [len]); the run-local layout
        // vectors live for the whole call, so the field views may borrow them.
        let mut provided = self.bufs;
        let mut los: Vec<[i32; 1]> = Vec::with_capacity(want.len());
        let mut exts: Vec<[u32; 1]> = Vec::with_capacity(want.len());
        let mut ordered: Vec<(BufHandle<'a, S>, Option<(&'a [i32], &'a [u32])>)> =
            Vec::with_capacity(want.len());
        for (field, is_out) in &want {
            // match by canonical field identity: the typed manifest
            // canonicalizes a buffer to one wire name (`prim.vel[0]`), but a caller may bind it
            // under the producer's secondary spelling (`prim.vel_0`). both parse to the same
            // FieldBind, so normalize both sides through it before comparing.
            let pos = provided
                .iter()
                .position(|b| FieldBind::from_path(&b.name) == FieldBind::from_path(field))
                .unwrap_or_else(|| {
                    let names: Vec<&str> = want.iter().map(|(n, _)| n.as_str()).collect();
                    panic!("kernel '{}': no buffer bound for manifest field '{field}'; expected {names:?}", self.name)
                });
            let b = provided.swap_remove(pos);
            let (len, is_mut) = match &b.handle {
                BufHandle::Host(d) => (d.len() as u32, false),
                BufHandle::HostMut(d) => (d.len() as u32, true),
            };
            assert_eq!(
                is_mut,
                *is_out,
                "kernel '{}': field '{field}' is {} in the manifest but bound as {}",
                self.name,
                if *is_out { "an output" } else { "an input" },
                if is_mut { "output" } else { "input" },
            );
            los.push([0]);
            exts.push([len]);
            ordered.push((b.handle, b.layout));
        }
        if !provided.is_empty() {
            let extra: Vec<&str> = provided.iter().map(|b| b.name.as_str()).collect();
            panic!(
                "kernel '{}': buffers bound but not in manifest: {extra:?}",
                self.name
            );
        }

        // build the slice-ABI inputs/outputs directly (the field views borrow the run-local
        // default layouts above; explicit layouts pass through). manifest order is inputs
        // first then outputs, so the two groups fill in their canonical sub-order.
        let mut inputs: Vec<CpuField<'_, S>> = Vec::new();
        let mut outputs: Vec<CpuFieldMut<'_, S>> = Vec::new();
        for (ii, (handle, layout)) in ordered.into_iter().enumerate() {
            let (lo, ext) = layout.unwrap_or((&los[ii], &exts[ii]));
            match handle {
                BufHandle::Host(d) => inputs.push(CpuField::from_layout(d, lo, ext)),
                BufHandle::HostMut(d) => outputs.push(CpuFieldMut::from_layout(d, lo, ext)),
            }
        }

        // route scalars by name into the (ints, floats) tails per the typed manifest. the manifest
        // is typed (`ScalarBind`); this harness binds scalars by their wire-name string, so flatten
        // each bind back to its name here.
        let scalar_kinds = kernel_scalar_params_typed_from_ir(ir); // (bind, is_int) in declared order
        let mut ints: Vec<i32> = Vec::new();
        let mut scalars: Vec<S> = Vec::new();
        for (bind, is_int) in &scalar_kinds {
            let name = bind.name();
            if *is_int {
                let v = self
                    .ints
                    .iter()
                    .find(|(n, _)| *n == name.as_str())
                    .map(|(_, v)| *v)
                    .unwrap_or_else(|| {
                        panic!("kernel '{}': missing int scalar '{name}'", self.name)
                    });
                ints.push(v);
            } else {
                let v = self
                    .scalars
                    .iter()
                    .find(|(n, _)| *n == name.as_str())
                    .map(|(_, v)| *v)
                    // static-mesh default: a test confined to a static mesh has zero mesh rates —
                    // exactly what production `motion_scalar` returns for a static
                    // mesh. so the harness supplies 0 for any unsupplied `mesh_*` scalar rather
                    // than forcing every static test to enumerate mesh_adot/mesh_vtrans/mesh_hdil.
                    // this closes the footgun where a kernel growing a moving-mesh scalar silently
                    // rotted every static test that hand-builds its scalar list by name.
                    //
                    // uniform-spacing default, same footgun class: `map_kind_{ax}` = 0 selects the
                    // uniform face map `x_lo + i*dx` — what every test written before spacing
                    // became a runtime scalar means. a log-spacing test supplies `map_kind_{ax}`
                    // = 1 explicitly (its analytic expectations fail loudly if it forgets).
                    //
                    // `x_lo_{ax}` joins them because a kernel that only DIFFERENCES widths reads
                    // the axis origin solely through the mapped arm of that selector; the uniform
                    // map bypasses that arm — so the origin is unread and any value serves. a
                    // kernel that positions a cell absolutely (a moving mesh, a curvilinear
                    // metric) reads it on every arm, and those tests bind it.
                    .or_else(|| {
                        matches!(
                            bind,
                            ScalarBind::Ref(
                                ScalarRef::Mesh(_)
                                    | ScalarRef::MapKind(_)
                                    | ScalarRef::MapParam(_)
                                    | ScalarRef::XLo(_),
                            )
                        )
                        .then_some(S::ZERO)
                    })
                    .unwrap_or_else(|| {
                        panic!("kernel '{}': missing float scalar '{name}'", self.name)
                    });
                scalars.push(v);
            }
        }

        kernel(
            &inputs,
            &mut outputs,
            self.grid,
            self.dom_lo,
            &ints,
            &scalars,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a rest-mass RHD state (v = 0, W = 1): D = rho, S = 0, tau = p/(gamma-1).
    // the name-keyed c2p must recover (rho, 0, p) — proving the harness binds the
    // real emitted kernel correctly (order-independent, by manifest name).
    #[test]
    fn named_kernel_binds_and_runs_the_real_c2p() {
        let n = 4usize;
        let den = vec![1.0f64; n];
        let mom = vec![0.0f64; n];
        let nrg = vec![1.5f64; n]; // p/(gamma-1) = 1/(5/3 - 1) = 1.5
        let (mut rho, mut vel, mut pre) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let grid = [n as u32];
        let dom = [0i32];
        // deliberately bind in a SCRAMBLED order to prove binding keys on field name; positional order is ignored.
        NamedKernel::new("rhd_c2p_1d")
            .output("prim.pre", &mut pre)
            .input("cons.nrg", &nrg)
            .output("prim.rho", &mut rho)
            .input("cons.den", &den)
            .output("prim.vel_0", &mut vel)
            .input("cons.mom_0", &mom)
            .grid(&grid)
            .dom_lo(&dom)
            .scalar("gamma", 5.0 / 3.0)
            .run();
        for ii in 0..n {
            assert!((rho[ii] - 1.0).abs() < 1e-9, "rho[{ii}] = {}", rho[ii]);
            assert!(vel[ii].abs() < 1e-9, "vel[{ii}] = {}", vel[ii]);
            assert!((pre[ii] - 1.0).abs() < 1e-6, "pre[{ii}] = {}", pre[ii]);
        }
    }

    // the whole point: an unbound manifest field raises a loud, named panic, not the
    // silent positional drift that `__raw` allowed (omit cons.nrg here).
    #[test]
    #[should_panic(expected = "no buffer bound for manifest field 'cons.nrg'")]
    fn missing_field_is_a_named_panic() {
        let n = 4usize;
        let den = vec![1.0f64; n];
        let mom = vec![0.0f64; n];
        let (mut rho, mut vel, mut pre) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let grid = [n as u32];
        let dom = [0i32];
        NamedKernel::new("rhd_c2p_1d")
            .input("cons.den", &den)
            .input("cons.mom_0", &mom) // cons.nrg omitted
            .output("prim.rho", &mut rho)
            .output("prim.vel_0", &mut vel)
            .output("prim.pre", &mut pre)
            .grid(&grid)
            .dom_lo(&dom)
            .scalar("gamma", 5.0 / 3.0)
            .run();
    }
}
