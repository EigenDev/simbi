// =============================================================================
// boundary_policy.rs
//
// trait-based boundary condition policies for hydrodynamics.
// pure functional design with zero-cost policy composition.
//
// design philosophy:
//   - policies are pure transformations: (state, context) -> state
//   - compose via traits, not inheritance
//   - compile-time dispatch (zero runtime overhead)
//   - physics-agnostic base layer, physics-aware policies on top
//
// architecture:
//   1. base policies: outflow, reflect, periodic (geometry-only)
//   2. physics policies: hydro-specific transformations
//   3. composition: base ∘ physics = complete boundary
//
// inspired by c++ policy pattern but more compositional.
//
// usage:
//   let policy = HydroReflectPolicy { velocity_idx: 1 };
//   let ghost_state = policy.apply(edge_state, dim, side);
// =============================================================================

use super::state::Regime;

// =============================================================================
// boundary side and type
// =============================================================================

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BoundaryType {
    /// zero-gradient extrapolation (transmissive)
    Outflow,
    /// reflecting wall (velocity flip)
    Reflect,
    /// periodic wrapping
    Periodic,
    /// user-defined dynamic boundary
    Dynamic,
}

// =============================================================================
// boundary context (for dynamic boundaries, moving meshes)
// =============================================================================

#[derive(Debug, Copy, Clone)]
pub struct BoundaryContext<const RANK: usize> {
    pub dim: usize,
    pub side: Side,
    pub boundary_type: BoundaryType,
    pub time: f64,
    pub position: [f64; RANK],
    pub wall_velocity: Option<f64>,
}

impl<const RANK: usize> BoundaryContext<RANK> {
    pub fn simple(dim: usize, side: Side, boundary_type: BoundaryType) -> Self {
        Self {
            dim,
            side,
            boundary_type,
            time: 0.0,
            position: [0.0; RANK],
            wall_velocity: None,
        }
    }

    pub fn with_wall_velocity(mut self, velocity: f64) -> Self {
        self.wall_velocity = Some(velocity);
        self
    }

    pub fn with_position(mut self, position: [f64; RANK]) -> Self {
        self.position = position;
        self
    }
}

// =============================================================================
// core boundary policy trait
// =============================================================================

/// boundary policy trait - pure transformation of state values
pub trait BoundaryPolicy<T, const RANK: usize> {
    /// applies boundary condition to a single value
    fn apply(&self, edge_value: T, context: &BoundaryContext<RANK>) -> T;
}

// =============================================================================
// base geometric policies (physics-agnostic)
// =============================================================================

/// outflow (zero-gradient) - simply copies edge value to ghost
pub struct OutflowPolicy;

impl<T: Copy, const RANK: usize> BoundaryPolicy<T, RANK> for OutflowPolicy {
    #[inline]
    fn apply(&self, edge_value: T, _context: &BoundaryContext<RANK>) -> T {
        edge_value
    }
}

/// periodic boundary - not applied locally, needs global coordination
pub struct PeriodicPolicy;

impl<T: Copy, const RANK: usize> BoundaryPolicy<T, RANK> for PeriodicPolicy {
    #[inline]
    fn apply(&self, edge_value: T, _context: &BoundaryContext<RANK>) -> T {
        // periodic is handled by halo exchange, not local ghost filling
        edge_value
    }
}

// =============================================================================
// hydro-specific policies
// =============================================================================

/// reflecting boundary for primitive hydro states (rho, vel, pre)
/// flips velocity component normal to boundary
pub struct HydroReflectPolicy {
    /// index of first velocity component in state vector
    pub velocity_offset: usize,
}

impl HydroReflectPolicy {
    pub fn new(velocity_offset: usize) -> Self {
        Self { velocity_offset }
    }

    pub fn standard() -> Self {
        Self { velocity_offset: 1 }
    }
}

// 1d hydro reflect
impl BoundaryPolicy<[f64; 3], 1> for HydroReflectPolicy {
    fn apply(&self, mut edge_value: [f64; 3], context: &BoundaryContext<1>) -> [f64; 3] {
        let vel_idx = self.velocity_offset + context.dim;
        edge_value[vel_idx] = -edge_value[vel_idx];

        if let Some(v_wall) = context.wall_velocity {
            edge_value[vel_idx] += 2.0 * v_wall;
        }

        edge_value
    }
}

// 2d hydro reflect
impl BoundaryPolicy<[f64; 4], 2> for HydroReflectPolicy {
    fn apply(&self, mut edge_value: [f64; 4], context: &BoundaryContext<2>) -> [f64; 4] {
        let vel_idx = self.velocity_offset + context.dim;
        edge_value[vel_idx] = -edge_value[vel_idx];

        if let Some(v_wall) = context.wall_velocity {
            edge_value[vel_idx] += 2.0 * v_wall;
        }

        edge_value
    }
}

// 3d hydro reflect
impl BoundaryPolicy<[f64; 5], 3> for HydroReflectPolicy {
    fn apply(&self, mut edge_value: [f64; 5], context: &BoundaryContext<3>) -> [f64; 5] {
        let vel_idx = self.velocity_offset + context.dim;
        edge_value[vel_idx] = -edge_value[vel_idx];

        if let Some(v_wall) = context.wall_velocity {
            edge_value[vel_idx] += 2.0 * v_wall;
        }

        edge_value
    }
}

/// reflecting boundary for conserved hydro states (den, mom, nrg)
/// flips momentum component normal to boundary
pub struct ConservedReflectPolicy {
    pub momentum_offset: usize,
}

impl ConservedReflectPolicy {
    pub fn new(momentum_offset: usize) -> Self {
        Self { momentum_offset }
    }

    pub fn standard() -> Self {
        Self { momentum_offset: 1 }
    }
}

// 1d conserved reflect
impl BoundaryPolicy<[f64; 3], 1> for ConservedReflectPolicy {
    fn apply(&self, mut edge_value: [f64; 3], context: &BoundaryContext<1>) -> [f64; 3] {
        let mom_idx = self.momentum_offset + context.dim;
        edge_value[mom_idx] = -edge_value[mom_idx];

        if let Some(v_wall) = context.wall_velocity {
            edge_value[mom_idx] += 2.0 * edge_value[0] * v_wall;
        }

        edge_value
    }
}

// 2d conserved reflect
impl BoundaryPolicy<[f64; 4], 2> for ConservedReflectPolicy {
    fn apply(&self, mut edge_value: [f64; 4], context: &BoundaryContext<2>) -> [f64; 4] {
        let mom_idx = self.momentum_offset + context.dim;
        edge_value[mom_idx] = -edge_value[mom_idx];

        if let Some(v_wall) = context.wall_velocity {
            edge_value[mom_idx] += 2.0 * edge_value[0] * v_wall;
        }

        edge_value
    }
}

// 3d conserved reflect
impl BoundaryPolicy<[f64; 5], 3> for ConservedReflectPolicy {
    fn apply(&self, mut edge_value: [f64; 5], context: &BoundaryContext<3>) -> [f64; 5] {
        let mom_idx = self.momentum_offset + context.dim;
        edge_value[mom_idx] = -edge_value[mom_idx];

        if let Some(v_wall) = context.wall_velocity {
            edge_value[mom_idx] += 2.0 * edge_value[0] * v_wall;
        }

        edge_value
    }
}

// =============================================================================
// policy composition
// =============================================================================

/// composes two policies: applies first, then second
pub struct ComposedPolicy<P1, P2> {
    pub first: P1,
    pub second: P2,
}

impl<P1, P2, T, const RANK: usize> BoundaryPolicy<T, RANK> for ComposedPolicy<P1, P2>
where
    P1: BoundaryPolicy<T, RANK>,
    P2: BoundaryPolicy<T, RANK>,
{
    #[inline]
    fn apply(&self, edge_value: T, context: &BoundaryContext<RANK>) -> T {
        let intermediate = self.first.apply(edge_value, context);
        self.second.apply(intermediate, context)
    }
}

/// helper to compose policies
pub fn compose<P1, P2>(first: P1, second: P2) -> ComposedPolicy<P1, P2> {
    ComposedPolicy { first, second }
}

// =============================================================================
// dynamic boundary (user-defined function)
// =============================================================================

/// dynamic boundary using user-provided closure
pub struct DynamicPolicy<F> {
    pub func: F,
}

impl<F, T, const RANK: usize> BoundaryPolicy<T, RANK> for DynamicPolicy<F>
where
    F: Fn(T, &BoundaryContext<RANK>) -> T,
{
    #[inline]
    fn apply(&self, edge_value: T, context: &BoundaryContext<RANK>) -> T {
        (self.func)(edge_value, context)
    }
}

/// helper to create dynamic policy from closure
pub fn dynamic<F>(func: F) -> DynamicPolicy<F> {
    DynamicPolicy { func }
}

// =============================================================================
// field-wise boundary application
// =============================================================================

/// applies boundary policy to entire ghost region of a field
pub fn apply_boundary_1d<T, P>(
    field_data: &mut [T],
    nghosts: usize,
    dim: usize,
    side: Side,
    boundary_type: BoundaryType,
    policy: &P,
) where
    T: Copy,
    P: BoundaryPolicy<T, 1>,
{
    let n = field_data.len();
    let context = BoundaryContext::simple(dim, side, boundary_type);

    match side {
        Side::Left => {
            let edge_value = field_data[nghosts];
            for i in 0..nghosts {
                field_data[i] = policy.apply(edge_value, &context);
            }
        }
        Side::Right => {
            let edge_value = field_data[n - nghosts - 1];
            for i in (n - nghosts)..n {
                field_data[i] = policy.apply(edge_value, &context);
            }
        }
    }
}

// =============================================================================
// regime-specific convenience constructors
// =============================================================================

/// creates standard outflow policy (works for any type)
pub fn outflow<const RANK: usize>() -> OutflowPolicy {
    OutflowPolicy
}

/// creates standard reflecting boundary for primitive states
pub fn reflect_primitive<const RANK: usize>() -> HydroReflectPolicy {
    HydroReflectPolicy::standard()
}

/// creates standard reflecting boundary for conserved states
pub fn reflect_conserved<const RANK: usize>() -> ConservedReflectPolicy {
    ConservedReflectPolicy::standard()
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outflow_policy() {
        let policy = outflow::<1>();
        let context: BoundaryContext<1> =
            BoundaryContext::simple(0, Side::Left, BoundaryType::Outflow);

        let value = 42.0;
        let result = policy.apply(value, &context);

        assert_eq!(result, value);
    }

    #[test]
    fn test_hydro_reflect_policy() {
        let policy = reflect_primitive::<1>();
        let context: BoundaryContext<1> =
            BoundaryContext::simple(0, Side::Left, BoundaryType::Reflect);

        // state: [rho, vx, pre]
        let state = [1.0, 0.5, 1.0];
        let result = policy.apply(state, &context);

        assert_eq!(result[0], 1.0); // rho unchanged
        assert_eq!(result[1], -0.5); // vx flipped
        assert_eq!(result[2], 1.0); // pre unchanged
    }

    #[test]
    fn test_hydro_reflect_with_moving_wall() {
        let policy = reflect_primitive::<1>();
        let mut context: BoundaryContext<1> =
            BoundaryContext::simple(0, Side::Left, BoundaryType::Reflect);
        context.wall_velocity = Some(0.3);

        let state = [1.0, 0.5, 1.0];
        let result = policy.apply(state, &context);

        // v_ghost = -v_edge + 2*v_wall = -0.5 + 0.6 = 0.1
        assert!((result[1] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_conserved_reflect_policy() {
        let policy = reflect_conserved::<1>();
        let context: BoundaryContext<1> =
            BoundaryContext::simple(0, Side::Left, BoundaryType::Reflect);

        // state: [den, momx, nrg]
        let state = [2.0, 1.0, 5.0];
        let result = policy.apply(state, &context);

        assert_eq!(result[0], 2.0); // den unchanged
        assert_eq!(result[1], -1.0); // momx flipped
        assert_eq!(result[2], 5.0); // nrg unchanged
    }

    #[test]
    fn test_policy_composition() {
        let policy1 = OutflowPolicy;
        let policy2 = dynamic(|val: f64, _ctx: &BoundaryContext<1>| val * 2.0);
        let composed = compose(policy1, policy2);

        let context: BoundaryContext<1> =
            BoundaryContext::simple(0, Side::Left, BoundaryType::Outflow);
        let result = composed.apply(5.0, &context);

        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_dynamic_policy() {
        let policy = dynamic(
            |val: f64, ctx: &BoundaryContext<1>| {
                if ctx.time > 1.0 {
                    val * 2.0
                } else {
                    val
                }
            },
        );

        let mut context = BoundaryContext::simple(0, Side::Left, BoundaryType::Dynamic);

        context.time = 0.5;
        assert_eq!(policy.apply(10.0, &context), 10.0);

        context.time = 1.5;
        assert_eq!(policy.apply(10.0, &context), 20.0);
    }

    #[test]
    fn test_apply_boundary_1d_left() {
        let mut field = vec![0.0, 0.0, 5.0, 6.0, 7.0, 0.0, 0.0];
        let nghosts = 2;
        let policy = outflow::<1>();

        apply_boundary_1d(
            &mut field,
            nghosts,
            0,
            Side::Left,
            BoundaryType::Outflow,
            &policy,
        );

        assert_eq!(field[0], 5.0);
        assert_eq!(field[1], 5.0);
        assert_eq!(field[2], 5.0); // edge cell
    }

    #[test]
    fn test_apply_boundary_1d_right() {
        let mut field = vec![0.0, 0.0, 5.0, 6.0, 7.0, 0.0, 0.0];
        let nghosts = 2;
        let policy = outflow::<1>();

        apply_boundary_1d(
            &mut field,
            nghosts,
            0,
            Side::Right,
            BoundaryType::Outflow,
            &policy,
        );

        assert_eq!(field[4], 7.0); // edge cell
        assert_eq!(field[5], 7.0);
        assert_eq!(field[6], 7.0);
    }

    #[test]
    fn test_2d_reflect_policies() {
        // test that reflection works correctly in each dimension
        let policy = reflect_primitive::<2>();

        // reflect in x-direction (dim=0)
        let context_x = BoundaryContext::simple(0, Side::Left, BoundaryType::Reflect);
        let state = [1.0, 0.5, 0.3, 1.0]; // [rho, vx, vy, pre]
        let result = policy.apply(state, &context_x);
        assert_eq!(result[1], -0.5); // vx flipped
        assert_eq!(result[2], 0.3); // vy unchanged

        // reflect in y-direction (dim=1)
        let context_y = BoundaryContext::simple(1, Side::Left, BoundaryType::Reflect);
        let result = policy.apply(state, &context_y);
        assert_eq!(result[1], 0.5); // vx unchanged
        assert_eq!(result[2], -0.3); // vy flipped
    }
}
