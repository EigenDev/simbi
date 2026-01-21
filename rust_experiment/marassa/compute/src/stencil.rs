// =============================================================================
// stencil.rs
//
// compile-time stencil pattern generation and field access for finite-volume methods.
// provides const functions to compute neighbor offsets and runtime views that apply
// these patterns to fields.
//
// key design:
//   - const fn pattern generation (zero runtime cost)
//   - stencil views that sample fields using patterns
//   - generic over rank (1d, 2d, 3d)
//   - generic over stencil size (determined by reconstruction order)
//   - direction-aware (stencil extends along specified axis)
//
// usage example:
//   // 1. generate compile-time pattern
//   const PATTERN: [[i64; 1]; 3] = left_pattern(Reconstruction::PLM, 0);
//
//   // 2. create stencil view from field
//   let stencil = StencilView::new(field.view(), PATTERN);
//
//   // 3. sample neighboring cells
//   let neighbors = stencil.sample([i]); // returns [&T; 3]
//
//   // 4. or use convenience constructors
//   let left_stencil = StencilView::<_, 1, 3>::left(
//       field.view(),
//       Reconstruction::PLM,
//       0  // x-direction
//   );
//
//   // 5. create lazy computations from stencils
//   let reconstruction = stencil_computation(
//       field.view(),
//       PATTERN,
//       |samples| linear_interpolate(samples)  // custom reconstruction op
//   );
//
// typical workflow (riemann solver):
//   let rho_left = StencilView::<_, 1, 3>::left(rho.view(), Reconstruction::PLM, 0);
//   let rho_right = StencilView::<_, 1, 3>::right(rho.view(), Reconstruction::PLM, 0);
//
//   for i in domain.iter() {
//       let ul = reconstruct_plm(rho_left.sample(i));   // left state at interface
//       let ur = reconstruct_plm(rho_right.sample(i));  // right state at interface
//       let flux = hlle_solver(ul, ur);                 // riemann solver
//   }
// =============================================================================

use crate::domain::Domain;
use crate::field::FieldView;

/// reconstruction schemes supported by the stencil system.
/// determines the order of spatial accuracy and stencil width.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Reconstruction {
    /// piecewise constant method (first-order)
    /// stencil size: 1
    PCM,

    /// piecewise linear method (second-order)
    /// stencil size: 3 (cell + 2 neighbors)
    PLM,
}

/// returns the number of cells in a stencil for the given reconstruction order.
///
/// # examples
/// ```
/// use base::{Reconstruction, stencil_size};
///
/// assert_eq!(stencil_size(Reconstruction::PCM), 1);
/// assert_eq!(stencil_size(Reconstruction::PLM), 3);
/// ```
pub const fn stencil_size(rec: Reconstruction) -> usize {
    match rec {
        Reconstruction::PCM => 1,
        Reconstruction::PLM => 3,
    }
}

/// returns the formal order of accuracy for the reconstruction scheme.
pub const fn reconstruction_order(rec: Reconstruction) -> usize {
    match rec {
        Reconstruction::PCM => 1,
        Reconstruction::PLM => 2,
    }
}

/// generates the left reconstruction stencil pattern for a given direction.
///
/// the "left" stencil is used to reconstruct the state at the left side
/// of a cell interface (the "+" side in finite volume terminology).
///
/// # arguments
/// * `rec` - reconstruction scheme
/// * `direction` - spatial dimension index (0=x, 1=y, 2=z)
///
/// # returns
/// array of coordinate offsets relative to the interface position.
/// each offset is an n-dimensional array where only the specified
/// direction component is non-zero.
///
/// # examples
/// ```
/// use base::{Reconstruction, left_pattern};
///
/// // 1d plm left stencil in x-direction
/// const PATTERN: [[i64; 1]; 3] = left_pattern(Reconstruction::PLM, 0);
/// // produces: [[-2], [-1], [0]]
///
/// // 2d plm left stencil in y-direction
/// const PATTERN_2D: [[i64; 2]; 3] = left_pattern(Reconstruction::PLM, 1);
/// // produces: [[0, -2], [0, -1], [0, 0]]
/// ```
pub const fn left_pattern<const N: usize, const SIZE: usize>(
    rec: Reconstruction,
    direction: usize,
) -> [[i64; N]; SIZE] {
    assert!(direction < N, "direction index out of bounds");
    assert!(
        SIZE == stencil_size(rec),
        "size mismatch for reconstruction"
    );

    let mut pattern = [[0i64; N]; SIZE];

    match rec {
        Reconstruction::PCM => {
            // pcm: use cell to the left of interface
            pattern[0][direction] = -1;
        }
        Reconstruction::PLM => {
            // plm: use cells at i-2, i-1, i (left-biased)
            pattern[0][direction] = -2;
            pattern[1][direction] = -1;
            pattern[2][direction] = 0;
        }
    }

    pattern
}

/// generates the right reconstruction stencil pattern for a given direction.
///
/// the "right" stencil is used to reconstruct the state at the right side
/// of a cell interface (the "-" side in finite volume terminology).
///
/// # arguments
/// * `rec` - reconstruction scheme
/// * `direction` - spatial dimension index (0=x, 1=y, 2=z)
///
/// # returns
/// array of coordinate offsets relative to the interface position.
///
/// # examples
/// ```
/// use base::{Reconstruction, right_pattern};
///
/// // 1d plm right stencil in x-direction
/// const PATTERN: [[i64; 1]; 3] = right_pattern(Reconstruction::PLM, 0);
/// // produces: [[-1], [0], [1]]
///
/// // 2d plm right stencil in x-direction
/// const PATTERN_2D: [[i64; 2]; 3] = right_pattern(Reconstruction::PLM, 0);
/// // produces: [[-1, 0], [0, 0], [1, 0]]
/// ```
pub const fn right_pattern<const N: usize, const SIZE: usize>(
    rec: Reconstruction,
    direction: usize,
) -> [[i64; N]; SIZE] {
    assert!(direction < N, "direction index out of bounds");
    assert!(
        SIZE == stencil_size(rec),
        "size mismatch for reconstruction"
    );

    let mut pattern = [[0i64; N]; SIZE];

    match rec {
        Reconstruction::PCM => {
            // pcm: use cell to the right of interface
            pattern[0][direction] = 0;
        }
        Reconstruction::PLM => {
            // plm: use cells at i-1, i, i+1 (right-biased)
            pattern[0][direction] = -1;
            pattern[1][direction] = 0;
            pattern[2][direction] = 1;
        }
    }

    pattern
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_1d_patterns() {
        const LEFT: [[i64; 1]; 1] = left_pattern(Reconstruction::PCM, 0);
        const RIGHT: [[i64; 1]; 1] = right_pattern(Reconstruction::PCM, 0);

        assert_eq!(LEFT[0][0], -1);
        assert_eq!(RIGHT[0][0], 0);
    }

    #[test]
    fn plm_1d_patterns() {
        const LEFT: [[i64; 1]; 3] = left_pattern(Reconstruction::PLM, 0);
        const RIGHT: [[i64; 1]; 3] = right_pattern(Reconstruction::PLM, 0);

        // left: i-2, i-1, i
        assert_eq!(LEFT[0][0], -2);
        assert_eq!(LEFT[1][0], -1);
        assert_eq!(LEFT[2][0], 0);

        // right: i-1, i, i+1
        assert_eq!(RIGHT[0][0], -1);
        assert_eq!(RIGHT[1][0], 0);
        assert_eq!(RIGHT[2][0], 1);
    }

    #[test]
    fn plm_2d_x_direction() {
        const LEFT: [[i64; 2]; 3] = left_pattern(Reconstruction::PLM, 0);
        const RIGHT: [[i64; 2]; 3] = right_pattern(Reconstruction::PLM, 0);

        // x-direction: only first component varies
        assert_eq!(LEFT[0], [-2, 0]);
        assert_eq!(LEFT[1], [-1, 0]);
        assert_eq!(LEFT[2], [0, 0]);

        assert_eq!(RIGHT[0], [-1, 0]);
        assert_eq!(RIGHT[1], [0, 0]);
        assert_eq!(RIGHT[2], [1, 0]);
    }

    #[test]
    fn plm_2d_y_direction() {
        const LEFT: [[i64; 2]; 3] = left_pattern(Reconstruction::PLM, 1);
        const RIGHT: [[i64; 2]; 3] = right_pattern(Reconstruction::PLM, 1);

        // y-direction: only second component varies
        assert_eq!(LEFT[0], [0, -2]);
        assert_eq!(LEFT[1], [0, -1]);
        assert_eq!(LEFT[2], [0, 0]);

        assert_eq!(RIGHT[0], [0, -1]);
        assert_eq!(RIGHT[1], [0, 0]);
        assert_eq!(RIGHT[2], [0, 1]);
    }

    #[test]
    fn plm_3d_z_direction() {
        const LEFT: [[i64; 3]; 3] = left_pattern(Reconstruction::PLM, 2);
        const RIGHT: [[i64; 3]; 3] = right_pattern(Reconstruction::PLM, 2);

        // z-direction: only third component varies
        assert_eq!(LEFT[0], [0, 0, -2]);
        assert_eq!(LEFT[1], [0, 0, -1]);
        assert_eq!(LEFT[2], [0, 0, 0]);

        assert_eq!(RIGHT[0], [0, 0, -1]);
        assert_eq!(RIGHT[1], [0, 0, 0]);
        assert_eq!(RIGHT[2], [0, 0, 1]);
    }

    #[test]
    fn const_evaluation() {
        // verify patterns are truly const (compile-time)
        const _PCM_1D_LEFT: [[i64; 1]; 1] = left_pattern(Reconstruction::PCM, 0);
        const _PLM_2D_RIGHT: [[i64; 2]; 3] = right_pattern(Reconstruction::PLM, 1);
        const _PLM_3D_LEFT: [[i64; 3]; 3] = left_pattern(Reconstruction::PLM, 2);
    }

    #[test]
    #[should_panic(expected = "size mismatch")]
    fn wrong_size_panics() {
        // attempting to create plm pattern with pcm size should panic at compile time
        let _: [[i64; 1]; 1] = left_pattern(Reconstruction::PLM, 0);
    }
}

// =============================================================================
// stencil views: apply patterns to fields
// =============================================================================

/// non-owning view that samples a field using a stencil pattern.
/// provides access to multiple neighboring cells simultaneously.
pub struct StencilView<'a, T, const N: usize, const SIZE: usize> {
    field: FieldView<'a, T, N>,
    pattern: [[i64; N]; SIZE],
}

impl<'a, T, const N: usize, const SIZE: usize> StencilView<'a, T, N, SIZE> {
    /// creates a stencil view from a field view and offset pattern.
    pub fn new(field: FieldView<'a, T, N>, pattern: [[i64; N]; SIZE]) -> Self {
        Self { field, pattern }
    }

    /// creates a left reconstruction stencil view.
    pub fn left(field: FieldView<'a, T, N>, rec: Reconstruction, direction: usize) -> Self {
        let pattern = left_pattern(rec, direction);
        Self::new(field, pattern)
    }

    /// creates a right reconstruction stencil view.
    pub fn right(field: FieldView<'a, T, N>, rec: Reconstruction, direction: usize) -> Self {
        let pattern = right_pattern(rec, direction);
        Self::new(field, pattern)
    }

    /// samples the field at the stencil positions relative to coord.
    /// returns array of values at each stencil position.
    pub fn sample(&self, coord: [i64; N]) -> [&T; SIZE] {
        let mut result: [&T; SIZE] = unsafe { core::mem::zeroed() };

        for ii in 0..SIZE {
            let mut offset_coord = coord;
            for jj in 0..N {
                offset_coord[jj] = coord[jj] + self.pattern[ii][jj];
            }
            result[ii] = self.field.eval(offset_coord);
        }

        result
    }

    /// checks if the stencil is fully contained within the domain at coord.
    pub fn is_valid(&self, coord: [i64; N]) -> bool {
        let domain = self.field.domain();

        for ii in 0..SIZE {
            let mut offset_coord = coord;
            for jj in 0..N {
                offset_coord[jj] = coord[jj] + self.pattern[ii][jj];
            }

            if !domain.contains(offset_coord) {
                return false;
            }
        }

        true
    }

    /// returns the underlying field view.
    pub fn field(&self) -> &FieldView<'a, T, N> {
        &self.field
    }

    /// returns the stencil pattern.
    pub fn pattern(&self) -> &[[i64; N]; SIZE] {
        &self.pattern
    }

    /// returns the domain of the field.
    pub fn domain(&self) -> Domain<N> {
        self.field.domain()
    }
}

/// creates a computation that applies a stencil operation to a field.
/// useful for building reconstruction computations lazily.
pub fn stencil_computation<'a, T, U, const N: usize, const SIZE: usize, F>(
    field: FieldView<'a, T, N>,
    pattern: [[i64; N]; SIZE],
    op: F,
) -> crate::computation::Computation<U, N, impl Fn([i64; N]) -> U + 'a>
where
    T: Clone,
    F: Fn([&T; SIZE]) -> U + 'a,
{
    let domain = field.domain();
    let view = StencilView::new(field, pattern);

    crate::computation::from_fn(domain, move |coord| {
        let samples = view.sample(coord);
        op(samples)
    })
}

#[cfg(test)]
mod stencil_view_tests {
    use super::*;
    use crate::domain::Domain;
    use crate::field::Field;
    use xpu_core::Device;
    use xpu_host::CpuDevice;

    #[test]
    fn test_stencil_view_pcm() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);
        let mut field = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        // initialize: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let data: Vec<f64> = (0..10).map(|x| x as f64).collect();
        field.from_host(&data).unwrap();

        const PATTERN: [[i64; 1]; 1] = left_pattern(Reconstruction::PCM, 0);
        let view = StencilView::new(field.view(), PATTERN);

        // sample at position 5 with pattern [[-1]]
        let samples = view.sample([5]);
        assert_eq!(*samples[0], 4.0);
    }

    #[test]
    fn test_stencil_view_plm_1d() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);
        let mut field = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        let data: Vec<f64> = (0..10).map(|x| x as f64).collect();
        field.from_host(&data).unwrap();

        // left pattern: [[-2], [-1], [0]]
        const LEFT: [[i64; 1]; 3] = left_pattern(Reconstruction::PLM, 0);
        let left_view = StencilView::new(field.view(), LEFT);

        // sample at position 5
        let samples = left_view.sample([5]);
        assert_eq!(*samples[0], 3.0); // i-2
        assert_eq!(*samples[1], 4.0); // i-1
        assert_eq!(*samples[2], 5.0); // i

        // right pattern: [[-1], [0], [1]]
        const RIGHT: [[i64; 1]; 3] = right_pattern(Reconstruction::PLM, 0);
        let right_view = StencilView::new(field.view(), RIGHT);

        let samples = right_view.sample([5]);
        assert_eq!(*samples[0], 4.0); // i-1
        assert_eq!(*samples[1], 5.0); // i
        assert_eq!(*samples[2], 6.0); // i+1
    }

    #[test]
    fn test_stencil_view_plm_2d() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([5, 5]);
        let mut field = Field::<f64, _, 2>::zeros(&device, domain).unwrap();

        // initialize with row-major indexing
        let data: Vec<f64> = (0..25).map(|x| x as f64).collect();
        field.from_host(&data).unwrap();

        // x-direction stencil
        const PATTERN_X: [[i64; 2]; 3] = left_pattern(Reconstruction::PLM, 0);
        let view_x = StencilView::new(field.view(), PATTERN_X);

        // at [3, 2], pattern [[-2, 0], [-1, 0], [0, 0]] samples x-neighbors
        let samples = view_x.sample([3, 2]);
        // [1, 2], [2, 2], [3, 2] -> row-major: 1*5+2=7, 2*5+2=12, 3*5+2=17
        assert_eq!(*samples[0], 7.0);
        assert_eq!(*samples[1], 12.0);
        assert_eq!(*samples[2], 17.0);

        // y-direction stencil
        const PATTERN_Y: [[i64; 2]; 3] = left_pattern(Reconstruction::PLM, 1);
        let view_y = StencilView::new(field.view(), PATTERN_Y);

        // at [2, 3], pattern [[0, -2], [0, -1], [0, 0]] samples y-neighbors
        let samples = view_y.sample([2, 3]);
        // [2, 1], [2, 2], [2, 3] -> row-major: 2*5+1=11, 2*5+2=12, 2*5+3=13
        assert_eq!(*samples[0], 11.0);
        assert_eq!(*samples[1], 12.0);
        assert_eq!(*samples[2], 13.0);
    }

    #[test]
    fn test_stencil_view_validity() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);
        let field = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        const PATTERN: [[i64; 1]; 3] = left_pattern(Reconstruction::PLM, 0);
        let view = StencilView::new(field.view(), PATTERN);

        // valid in interior
        assert!(view.is_valid([5]));

        // invalid near boundaries (pattern needs i-2)
        assert!(!view.is_valid([0]));
        assert!(!view.is_valid([1]));

        // valid at boundary that pattern can reach
        assert!(view.is_valid([2]));
    }

    #[test]
    fn test_stencil_computation() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);
        let mut field = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        let data: Vec<f64> = (0..10).map(|x| x as f64).collect();
        field.from_host(&data).unwrap();

        const PATTERN: [[i64; 1]; 3] = left_pattern(Reconstruction::PLM, 0);

        // create lazy computation that computes average of stencil
        let comp = stencil_computation(field.view(), PATTERN, |samples: [&f64; 3]| {
            (samples[0] + samples[1] + samples[2]) / 3.0
        });

        // at position 5: average of [3.0, 4.0, 5.0] = 4.0
        assert_eq!(comp.eval([5]), 4.0);
    }

    #[test]
    fn test_left_right_constructors() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);
        let mut field = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        let data: Vec<f64> = (0..10).map(|x| x as f64).collect();
        field.from_host(&data).unwrap();

        let left_view = StencilView::<_, 1, 3>::left(field.view(), Reconstruction::PLM, 0);
        let samples = left_view.sample([5]);
        assert_eq!(*samples[0], 3.0);
        assert_eq!(*samples[1], 4.0);
        assert_eq!(*samples[2], 5.0);

        let right_view = StencilView::<_, 1, 3>::right(field.view(), Reconstruction::PLM, 0);
        let samples = right_view.sample([5]);
        assert_eq!(*samples[0], 4.0);
        assert_eq!(*samples[1], 5.0);
        assert_eq!(*samples[2], 6.0);
    }

    #[test]
    fn test_solver_workflow_example() {
        // demonstrates typical usage in a finite-volume solver
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);
        let mut rho = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        // initialize density field
        let data: Vec<f64> = (0..10).map(|x| (x as f64) * 0.5 + 1.0).collect();
        rho.from_host(&data).unwrap();

        // create left and right stencils for plm reconstruction
        let left = StencilView::<_, 1, 3>::left(rho.view(), Reconstruction::PLM, 0);
        let right = StencilView::<_, 1, 3>::right(rho.view(), Reconstruction::PLM, 0);

        // sample at interface i=5
        let ul_samples = left.sample([5]); // cells 3, 4, 5
        let ur_samples = right.sample([5]); // cells 4, 5, 6

        // verify samples
        assert_eq!(*ul_samples[0], 2.5); // i-2 = 3
        assert_eq!(*ul_samples[1], 3.0); // i-1 = 4
        assert_eq!(*ul_samples[2], 3.5); // i   = 5

        assert_eq!(*ur_samples[0], 3.0); // i-1 = 4
        assert_eq!(*ur_samples[1], 3.5); // i   = 5
        assert_eq!(*ur_samples[2], 4.0); // i+1 = 6

        // simple reconstruction (just use center value for demonstration)
        let ul = ul_samples[2]; // left state
        let ur = ur_samples[1]; // right state

        // simple flux computation (average)
        let flux = (ul + ur) / 2.0;
        assert_eq!(flux, 3.5);
    }

    #[test]
    fn test_2d_directional_stencils() {
        // demonstrates x and y direction stencils in 2d
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([5, 5]);
        let mut field = Field::<f64, _, 2>::zeros(&device, domain).unwrap();

        // initialize with i + j pattern
        let data: Vec<f64> = (0..25)
            .map(|idx| {
                let i = idx / 5;
                let j = idx % 5;
                (i + j) as f64
            })
            .collect();
        field.from_host(&data).unwrap();

        // x-direction stencil at [2, 2]
        let x_stencil = StencilView::<_, 2, 3>::left(field.view(), Reconstruction::PLM, 0);
        let x_samples = x_stencil.sample([2, 2]);
        // [0,2], [1,2], [2,2] -> values 2, 3, 4
        assert_eq!(*x_samples[0], 2.0);
        assert_eq!(*x_samples[1], 3.0);
        assert_eq!(*x_samples[2], 4.0);

        // y-direction stencil at [2, 2]
        let y_stencil = StencilView::<_, 2, 3>::left(field.view(), Reconstruction::PLM, 1);
        let y_samples = y_stencil.sample([2, 2]);
        // [2,0], [2,1], [2,2] -> values 2, 3, 4
        assert_eq!(*y_samples[0], 2.0);
        assert_eq!(*y_samples[1], 3.0);
        assert_eq!(*y_samples[2], 4.0);
    }
}
