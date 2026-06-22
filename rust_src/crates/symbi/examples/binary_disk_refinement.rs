// =============================================================================
// examples/binary_disk_amr.rs
//
// 3d CIRCUMBINARY DISK on the static-refinement hierarchy: an equal-mass
// binary on a prescribed keplerian orbit (two accreting black holes — the
// sinks and the accretion diagnostics live on the FINE level, the coarse
// level carries gravity-only proxies), embedded in a near-keplerian gas disk
// with a tapered inner cavity and gaussian vertical structure. the fine level
// covers the binary + cavity edge; refluxing keeps the composite gas totals
// exact across the coarse-fine interface while the sinks drain.
//
// the domain is CENTERED AT THE ORIGIN (the prescribed binary advance rotates
// about it). per-level HDF5 checkpoints are written side by side
// (`{name}_L0_####.h5`, `{name}_L1_####.h5`) for scripts/plot_binary_disk_amr.py.
//
// usage (smoke — a few binary orbits at low resolution):
//   cargo run --release -p symbi --example binary_disk_refinement -- \
//       --n 48 --end-time 0.5 --n-checkpoints 5 --out output/binary_disk
//   uv run python3 scripts/plot_binary_disk_amr.py output/binary_disk/binary_disk_L0_final.h5
//
// usage (production — gpu, several orbits):
//   cargo run --release -p symbi --features cuda --example binary_disk_refinement -- \
//       --n 128 --end-time 10 --n-checkpoints 40 --out output/binary_disk
//
// problem-specific knobs (via `--key val`):
//   --bound <2.0>   domain half-width (extent = ±bound per axis)
//   --zoom <0.5>    fine-level half-width (the refined box = ±zoom)
//   --mtot <1.0>    binary total mass × G
//   --sep <0.2>     binary separation (orbit radius = sep/2 each)
//   --aspect <0.25> disk aspect ratio h = H/r (sets the temperature profile;
//                   under ~0.2 the vertical structure and the disk mach number
//                   outrun demo-scale grids — adiabatic + no-silent-floors
//                   needs thermal margin)
//   --sink <5.0>    sink rate; --racc <soft> accretion radius (keep >= soft so
//                   the sink covers the gravitational choke region)
//   --soft <0.08>   gravitational softening (keep >= ~2 fine cells)
//   --rcav <0.4>    cavity radius (the density taper scale, ~2*sep)
//   --prolong <ppm> coarse-fine prolongation order (pcm|plm|ppm)
//
// --end-time is interpreted in BINARY ORBITS (P = 2 pi sqrt(sep^3 / mtot)).
// =============================================================================

mod common;
use common::{run_hierarchy, BaseCli, Metadata};

use symbi::prelude::*;
// amr hierarchy + immersed-body types — outside the single-sim prelude surface.
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi_ib::{BinaryParams, Body, BodyCollection, BodyKind};

const TAU: f64 = 2.0 * std::f64::consts::PI;
const GAMMA: f64 = 5.0 / 3.0;

// feature-selected backend (GPU under `--features cuda`, else CPU) via the prelude's Default*.
type Sim = SimDefault<Newtonian, 3, Cartesian, IdealGas<f64>>;
type Kset = AdiabaticSubstrateKernelSet<DefaultMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, DefaultSpace, DefaultMemory, Kset>;

/// the disk primitive state at a physical point: a near-keplerian, locally
/// isothermal-profile disk (p = rho * cs(r)^2 / gamma with cs = h * v_k) with
/// an exponential inner-cavity taper, a gaussian vertical profile, and a
/// quiet ambient floor that dominates outside the disk body.
struct DiskProfile {
    mtot: f64,
    aspect: f64,
    rcav: f64,
    r_out: f64,
    rho0: f64,
    floor: f64,
    /// gravitational softening — the gas rotates at the circular velocity of
    /// the SOFTENED monopole, which is bounded at the origin, so every cell
    /// (cavity floor included) starts force-balanced instead of free-falling
    /// onto the sinks on the first kick.
    soft: f64,
    /// minimum vertical scale height (~2 fine cells): a gaussian thinner than
    /// the grid is a multi-decade density jump inside one cell — unresolvable.
    hz_min: f64,
}

impl DiskProfile {
    fn prim(&self, p: [f64; 3]) -> Prim<f64, 3> {
        let (x, y, z) = (p[0], p[1], p[2]);
        let rc = (x * x + y * y).sqrt().max(1e-12);
        let h_z = (self.aspect * rc.max(self.soft)).max(self.hz_min);

        // sigma ~ r^-1/2 with the cavity taper and an outer edge. the taper
        // is deliberately GENTLE (exp(-rcav/rc), not squared): a hard cavity
        // leaves floor-density gas where the point-mass kicks are strongest,
        // and one full-cfl step there drives the pressure negative. the sinks
        // carve the real cavity themselves.
        let taper = (-(self.rcav / rc)).exp();
        let edge = if rc < self.r_out { 1.0 } else { (-(rc - self.r_out) / (0.1 * self.r_out)).exp() };
        let vert = (-(z * z) / (2.0 * h_z * h_z)).exp();
        let rho = (self.rho0 * rc.powf(-0.5) * taper * edge * vert).max(self.floor);

        // circular velocity of the softened monopole: v_c^2 = M rc^2 /
        // (rc^2 + soft^2)^(3/2) — keplerian far out, -> 0 smoothly at the
        // origin. the temperature is the disk law (cs = aspect * v_k) plus a
        // HOT CAVITY term: a fraction of the local escape speed squared,
        // tapering off outside rcav. gas streaming into the binary potential
        // accelerates near-isentropically, so a cold cavity guarantees a
        // kinetic-dominated choke (e_int ~ 1% of total energy) whose internal
        // energy the energy-equation cancellation drains negative within a
        // fraction of an orbit — at ANY resolution and timestep. bounding the
        // inflow entropy bounds the mach number instead.
        let v_c2 = self.mtot * rc * rc / (rc * rc + self.soft * self.soft).powf(1.5);
        let v_c = v_c2.sqrt();
        let r_sph = (rc * rc + z * z).sqrt();
        let cs2_disk = self.aspect * self.aspect * self.mtot / rc.max(self.soft);
        let cs2_cav = 0.15 * self.mtot / (r_sph * r_sph + self.soft * self.soft).sqrt()
            * (-r_sph / self.rcav).exp();
        let cs2 = (cs2_disk + cs2_cav).max(0.01);
        let pre = rho * cs2 / GAMMA;

        // slightly sub-keplerian via the pressure-gradient factor.
        let v_phi = v_c * (1.0 - 1.5 * self.aspect * self.aspect).sqrt();
        let (vx, vy) = (-v_phi * y / rc, v_phi * x / rc);
        Prim { rho, vel: Tensor::new([vx, vy, 0.0]), pre }
    }
}

fn fill(sim: &Sim, disk: &DiskProfile) {
    let cnrg = sim.fields.cons.nrg_field().expect("adiabatic cons.nrg");
    for c in sim.geom.interior.iter() {
        let prim = disk.prim(sim.geom.centroid(c));
        let cons = symbi_hydro::regime::Regime::to_conserved(&sim.physics.regime, &sim.physics.eos, &prim);
        sim.fields.cons.den.view_mut().set(c, cons.den);
        for k in 0..3 {
            sim.fields.cons.mom[k].view_mut().set(c, cons.mom[k]);
        }
        cnrg.view_mut().set(c, cons.nrg);
    }
}

/// composite gas mass (coarse outside the coverage + fine interior).
fn composite_mass(hier: &Hier) -> f64 {
    let mut mass = 0.0;
    for lvl in &hier.levels {
        let vol: f64 = lvl.state.geom.dx.iter().product();
        let cov = lvl.coverage.as_ref();
        for c in lvl.state.geom.interior.iter() {
            if let Some(cov) = cov {
                if cov.contains(c) {
                    continue;
                }
            }
            mass += *lvl.state.fields.cons.den.view().at(c) * vol;
        }
    }
    mass
}

/// per-body (total_accreted, mdot) from the finest level's diagnostics.
fn accretion(hier: &Hier) -> Vec<(f64, f64)> {
    let bodies = &hier.levels.last().unwrap().state.immersed.as_ref().unwrap().bodies;
    (0..bodies.len())
        .map(|bb| match bodies.get(bb).kind {
            BodyKind::BlackHole { total_accreted_mass, accretion_rate, .. } => {
                (total_accreted_mass, accretion_rate)
            }
            _ => (0.0, 0.0),
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BaseCli::parse("binary_disk");

    let bound = cli.extra_f64("bound", 2.0);
    let zoom = cli.extra_f64("zoom", 0.5);
    let mtot = cli.extra_f64("mtot", 1.0);
    let sep = cli.extra_f64("sep", 0.2);
    let aspect = cli.extra_f64("aspect", 0.25);
    let sink = cli.extra_f64("sink", 5.0);
    let soft = cli.extra_f64("soft", 0.08);
    let racc = cli.extra_f64("racc", soft);
    let rcav = cli.extra_f64("rcav", 2.0 * sep);
    let [nx, ny, nz] = cli.n3();

    let dx = 2.0 * bound / nx as f64;
    let p_bin = TAU * (sep.powi(3) / mtot).sqrt();
    let t_final = cli.end_time * p_bin;

    assert!(
        soft >= 0.5 * dx,
        "softening {soft} under one fine cell ({}) — the first gravity kick goes supersonic",
        dx / 2.0
    );
    if soft < dx {
        eprintln!("[binary_disk] WARNING: softening {soft} under two fine cells — expect noisy sinks");
    }

    // ---- coarse level + disk ----
    let disk = DiskProfile {
        mtot,
        aspect,
        rcav,
        r_out: 0.75 * bound,
        rho0: 1.0,
        floor: 1e-5,
        soft,
        hz_min: dx,
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([nx, ny, nz])
        .origin([-bound; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial(|x| disk.prim(x))
        .build();
    let ck = Kset::new(GAMMA, cli.cfl, &coarse.geom.allocated);

    // ---- the refined box around the binary + cavity edge ----
    let regions = [RefinementRegion { x_lo: [-zoom; 3], x_hi: [zoom; 3] }];
    let prolong = match cli.extra_str("prolong", "ppm") {
        "pcm" => ProlongOrder::Pcm,
        "plm" => ProlongOrder::Plm,
        _ => ProlongOrder::Ppm,
    };
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, prolong, |s| {
        Kset::new(GAMMA, cli.cfl, &s.geom.allocated)
    })?;
    fill(&hier.levels[1].state, &disk);

    // ---- the prescribed keplerian binary (sinks on the fine level) ----
    let omega = (mtot / sep.powi(3)).sqrt();
    let v_orb = omega * (sep / 2.0);
    let bodies = BodyCollection::new()
        .add(Body::black_hole(
            0, Tensor::new([-sep / 2.0, 0.0, 0.0]), Tensor::new([0.0, -v_orb, 0.0]),
            mtot / 2.0, racc, soft, sink, 1.0, racc,
        ))
        .add(Body::black_hole(
            1, Tensor::new([sep / 2.0, 0.0, 0.0]), Tensor::new([0.0, v_orb, 0.0]),
            mtot / 2.0, racc, soft, sink, 1.0, racc,
        ))
        .with_binary_params(BinaryParams::new(mtot, sep, 0.0, 1.0))
        .as_binary();
    let mut hier = hier.with_bodies(bodies);

    // ---- evolve under the live progress widget (per-level checkpoints) ----
    let m0 = composite_mass(&hier);
    let setup = [
        ["Binary".to_string(), "M, a".to_string(), format!("{mtot}, {sep}")],
        ["".to_string(), "P".to_string(), format!("{p_bin:.4}")],
        ["".to_string(), "sink, racc, soft".to_string(), format!("{sink}, {racc}, {soft}")],
        ["".to_string(), "fine box".to_string(), format!("±{zoom}")],
    ];
    run_hierarchy(
        &mut hier, t_final, cli.n_checkpoints, &cli.out_dir, "binary_disk",
        p_bin, "P", "HLLE", &setup,
        |h| metadata_for(h, p_bin),
        |h| {
            let (m, acc) = (composite_mass(h), accretion(h));
            format!(
                "gas {:.6e} ({:+.3e})  mdot [{:.3e}, {:.3e}]  acc [{:.3e}, {:.3e}]",
                m, m - m0, acc[0].1, acc[1].1, acc[0].0, acc[1].0,
            )
        },
    )
}

fn metadata_for(hier: &Hier, p_bin: f64) -> Metadata {
    let acc = accretion(hier);
    Metadata::new()
        .with("problem", "binary_disk_amr")
        .with("p_bin", p_bin)
        .with("accreted_0", acc[0].0)
        .with("accreted_1", acc[1].0)
}
