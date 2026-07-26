use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DOMAIN_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DomainId(u64);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Space {
    pub name: &'static str,
    pub lo: isize,
    pub hi: isize,
}

impl Space {
    pub fn size(&self) -> usize {
        (self.hi - self.lo) as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Lo,
    Hi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Axis {
    pub fn index(self) -> usize {
        self as usize
    }
}

/// trait so methods accept both `Axis` and `usize`.
pub trait IntoAxis {
    fn into_axis(self) -> usize;
}

impl IntoAxis for Axis {
    fn into_axis(self) -> usize {
        self as usize
    }
}

impl IntoAxis for usize {
    fn into_axis(self) -> usize {
        self
    }
}

#[derive(Debug, Clone)]
pub struct Domain<const R: usize> {
    pub id: DomainId,
    pub spaces: [Space; R],
    strides: [usize; R],
}

impl<const R: usize> Domain<R> {
    pub fn new(spaces: [Space; R]) -> Self {
        for ii in 0..R {
            for jj in (ii + 1)..R {
                if spaces[ii].name == spaces[jj].name {
                    panic!("Domain: duplicate space name '{}'", spaces[ii].name);
                }
            }
        }
        // construction gate: every axis must satisfy hi >= lo. an inverted axis
        // (hi < lo) makes Space::size cast a negative isize to a ~1.8e19 usize,
        // poisoning size/volume/shape/Layout::len and every allocation path.
        // enforcing it here means every derived op (contract/slab/extend/...)
        // inherits the guard, since they all route through Domain::new.
        for ii in 0..R {
            assert!(
                spaces[ii].hi >= spaces[ii].lo,
                "Domain: empty space '{}' has hi ({}) < lo ({})",
                spaces[ii].name,
                spaces[ii].hi,
                spaces[ii].lo,
            );
        }
        let strides = Self::compute_strides(&spaces);
        Domain {
            id: DomainId(NEXT_DOMAIN_ID.fetch_add(1, Ordering::Relaxed)),
            spaces,
            strides,
        }
    }

    fn compute_strides(spaces: &[Space; R]) -> [usize; R] {
        // **physical-x-fastest convention**: axis 0 has stride 1, axis N has
        // stride = product of all lower-axis extents. matches standard CFD
        // row-major layout (`field[k][j][i]` with `i` fastest) when the caller pins
        // axis 0 to the physical fast axis (= simbi convention: x at axis 0).
        //
        // a kernel reading `field[coord[0]*stride[0] + ...]` with this layout
        // is GPU-coalesced: CUDA's standard `threadIdx.x → axis 0` mapping
        // means consecutive warp threads access consecutive bytes. **prior**
        // convention (axis R-1 fastest) had x at the slowest axis, so warp
        // threads accessed addresses `Ny*Nz` apart — uncoalesced reads cost
        // every gmem-touching kernel its memory bandwidth ceiling.
        //
        // THE formula lives once in `layout::strides_from_extent` — every view
        // (this, the runtime CpuField/DeviceView, symbi-grid's View) derives from
        // it, so they cannot drift apart.
        let extent: [usize; R] = std::array::from_fn(|a| spaces[a].size());
        let mut strides = [0usize; R];
        crate::layout::strides_from_extent(&extent, &mut strides);
        strides
    }

    pub const fn rank(&self) -> usize {
        R
    }

    /// per-axis sizes (for allocation). always positive.
    pub fn shape(&self) -> [usize; R] {
        std::array::from_fn(|a| self.spaces[a].size())
    }

    /// total number of points in the domain.
    pub fn volume(&self) -> usize {
        self.spaces.iter().map(|s| s.size()).product()
    }

    /// return a new domain contracted by `width` on each side of every axis.
    /// e.g., [0, 100).contract(2) -> [2, 98)
    pub fn contract(&self, width: isize) -> Domain<R> {
        Domain::new(std::array::from_fn(|a| Space {
            name: self.spaces[a].name,
            lo: self.spaces[a].lo + width,
            hi: self.spaces[a].hi - width,
        }))
    }

    /// restrict one axis to [lo, hi), keep all others unchanged.
    /// `dom.slab(Axis::X, (2, 5))` -> axis 0 becomes [2, 5), others untouched.
    pub fn slab(&self, axis: impl IntoAxis, range: impl IntoRange) -> Domain<R> {
        let axis = axis.into_axis();
        assert!(axis < R, "slab: axis {} out of range for rank {}", axis, R);
        let (lo, hi) = range.into_range();
        Domain::new(std::array::from_fn(|a| {
            if a == axis {
                Space {
                    name: self.spaces[a].name,
                    lo,
                    hi,
                }
            } else {
                self.spaces[a].clone()
            }
        }))
    }

    /// per-axis intersection. panics if result is empty on any axis.
    pub fn intersect(&self, other: &Domain<R>) -> Domain<R> {
        Domain::new(std::array::from_fn(|a| {
            let lo = self.spaces[a].lo.max(other.spaces[a].lo);
            let hi = self.spaces[a].hi.min(other.spaces[a].hi);
            assert!(
                hi > lo,
                "intersect: empty result on axis {} ('{}'), [{}, {}) ^ [{}, {})",
                a,
                self.spaces[a].name,
                self.spaces[a].lo,
                self.spaces[a].hi,
                other.spaces[a].lo,
                other.spaces[a].hi,
            );
            Space {
                name: self.spaces[a].name,
                lo,
                hi,
            }
        }))
    }

    /// per-axis bounding box (conservative union).
    pub fn hull(&self, other: &Domain<R>) -> Domain<R> {
        Domain::new(std::array::from_fn(|a| {
            let lo = self.spaces[a].lo.min(other.spaces[a].lo);
            let hi = self.spaces[a].hi.max(other.spaces[a].hi);
            Space {
                name: self.spaces[a].name,
                lo,
                hi,
            }
        }))
    }

    /// true if self and other overlap on every axis.
    pub fn overlaps(&self, other: &Domain<R>) -> bool {
        (0..R).all(|a| {
            self.spaces[a].lo < other.spaces[a].hi && other.spaces[a].lo < self.spaces[a].hi
        })
    }

    /// set difference: self \ other.
    /// decomposes into up to 3^R - 1 non-overlapping axis-aligned boxes.
    /// each box is the cartesian product of per-axis intervals chosen from
    /// {before, overlap, after}, excluding the all-overlap (center) box.
    /// empty sub-boxes are skipped.
    pub fn difference(&self, other: &Domain<R>) -> Vec<Domain<R>> {
        // if no overlap, self is the only region
        if !self.overlaps(other) {
            return vec![self.clone()];
        }

        // compute the overlap region (clamped intersection)
        let overlap: [(isize, isize); R] = std::array::from_fn(|a| {
            (
                self.spaces[a].lo.max(other.spaces[a].lo),
                self.spaces[a].hi.min(other.spaces[a].hi),
            )
        });

        // if overlap covers all of self, nothing remains
        let full_cover =
            (0..R).all(|a| overlap[a].0 <= self.spaces[a].lo && overlap[a].1 >= self.spaces[a].hi);
        if full_cover {
            return vec![];
        }

        // per-axis intervals: [before, overlap, after]
        // each is (lo, hi, valid)
        let intervals: [[(isize, isize, bool); 3]; R] = std::array::from_fn(|a| {
            let s = &self.spaces[a];
            [
                (s.lo, overlap[a].0, s.lo < overlap[a].0), // before
                (overlap[a].0, overlap[a].1, true),        // overlap
                (overlap[a].1, s.hi, overlap[a].1 < s.hi), // after
            ]
        });

        // enumerate all 3^R combinations, skip center (all index=1)
        let mut result = Vec::new();
        let mut idx = [0usize; R];

        loop {
            // skip center
            let is_center = idx.iter().all(|&i| i == 1);
            if !is_center {
                // check all intervals valid and non-empty
                let valid = (0..R).all(|a| intervals[a][idx[a]].2);
                if valid {
                    let dom = Domain::new(std::array::from_fn(|a| {
                        let (lo, hi, _) = intervals[a][idx[a]];
                        Space {
                            name: self.spaces[a].name,
                            lo,
                            hi,
                        }
                    }));
                    if dom.volume() > 0 {
                        result.push(dom);
                    }
                }
            }

            // increment base-3 counter
            let mut carry = true;
            for aa in 0..R {
                if carry {
                    idx[aa] += 1;
                    if idx[aa] < 3 {
                        carry = false;
                    } else {
                        idx[aa] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }

        result
    }

    /// minimal disjoint cover of `self \ other` — the GUILLOTINE difference.
    ///
    /// like [`difference`](Self::difference) it returns disjoint boxes whose
    /// union is exactly the set difference, but it cuts the shell with `2*R`
    /// half-open guillotine slabs, fewer pieces than the `3^R - 1` product cells. axis
    /// `ax` owns the two slabs lying outside the overlap on axis `ax`, taken at
    /// FULL extent on the higher axes (`aa > ax`) and CLIPPED to the overlap on
    /// the lower axes (`aa < ax`); a cell outside the overlap on several axes is
    /// therefore owned by its LOWEST such axis, which makes the slabs pairwise
    /// disjoint while still tiling `self \ other`.
    ///
    /// at most `2*R` boxes (vs up to `3^R - 1` for `difference`). use this when
    /// the cover is DISPATCHED (one kernel per box): the per-box launch cost
    /// makes the fewest-boxes partition the right one. use `difference` when the
    /// per-cell CLASSIFICATION of the shell matters (faces vs edges vs corners).
    pub fn guillotine_difference(&self, other: &Domain<R>) -> Vec<Domain<R>> {
        // no overlap -> self is untouched.
        if !self.overlaps(other) {
            return vec![self.clone()];
        }
        // the clamped overlap self ∩ other (per axis).
        let ov: [(isize, isize); R] = std::array::from_fn(|a| {
            (
                self.spaces[a].lo.max(other.spaces[a].lo),
                self.spaces[a].hi.min(other.spaces[a].hi),
            )
        });
        // overlap covers all of self -> nothing remains.
        if (0..R).all(|a| ov[a].0 <= self.spaces[a].lo && ov[a].1 >= self.spaces[a].hi) {
            return vec![];
        }
        let mut parts = Vec::with_capacity(2 * R);
        for ax in 0..R {
            // the two slabs outside the overlap on axis ax: [self.lo, ov.lo) and
            // [ov.hi, self.hi). either may be empty (overlap flush to that face).
            for (slo, shi) in [
                (self.spaces[ax].lo, ov[ax].0),
                (ov[ax].1, self.spaces[ax].hi),
            ] {
                if slo >= shi {
                    continue;
                }
                parts.push(Domain::new(std::array::from_fn(|aa| {
                    use std::cmp::Ordering;
                    let (lo, hi) = match aa.cmp(&ax) {
                        Ordering::Equal => (slo, shi),
                        // lower axes: owned by their own slabs -> clip to overlap.
                        Ordering::Less => (ov[aa].0, ov[aa].1),
                        // higher axes: full self extent.
                        Ordering::Greater => (self.spaces[aa].lo, self.spaces[aa].hi),
                    };
                    Space {
                        name: self.spaces[aa].name,
                        lo,
                        hi,
                    }
                })));
            }
        }
        parts
    }

    /// boundary slab: the `width`-thick layer at the lo or hi end of `axis`.
    pub fn boundary(&self, axis: impl IntoAxis, side: Side, width: isize) -> Domain<R> {
        let axis = axis.into_axis();
        assert!(
            axis < R,
            "boundary: axis {} out of range for rank {}",
            axis,
            R
        );
        let (lo, hi) = match side {
            Side::Lo => (self.spaces[axis].lo, self.spaces[axis].lo + width),
            Side::Hi => (self.spaces[axis].hi - width, self.spaces[axis].hi),
        };
        self.slab(axis, (lo, hi))
    }

    /// contract a single axis by `width` on each side.
    pub fn contract_axis(&self, axis: impl IntoAxis, width: isize) -> Domain<R> {
        let axis = axis.into_axis();
        assert!(
            axis < R,
            "contract_axis: axis {} out of range for rank {}",
            axis,
            R
        );
        Domain::new(std::array::from_fn(|a| {
            if a == axis {
                Space {
                    name: self.spaces[a].name,
                    lo: self.spaces[a].lo + width,
                    hi: self.spaces[a].hi - width,
                }
            } else {
                self.spaces[a].clone()
            }
        }))
    }

    /// extend one axis: grow lo by lo_delta (negative = expand left),
    /// grow hi by hi_delta (positive = expand right).
    /// `dom.extend(0, 0, 1)` adds one cell on the right of axis 0.
    /// `dom.extend(1, -1, 1)` adds one cell on each side of axis 1.
    pub fn extend(&self, axis: impl IntoAxis, lo_delta: isize, hi_delta: isize) -> Domain<R> {
        let axis = axis.into_axis();
        assert!(
            axis < R,
            "extend: axis {} out of range for rank {}",
            axis,
            R
        );
        Domain::new(std::array::from_fn(|a| {
            if a == axis {
                Space {
                    name: self.spaces[a].name,
                    lo: self.spaces[a].lo + lo_delta,
                    hi: self.spaces[a].hi + hi_delta,
                }
            } else {
                self.spaces[a].clone()
            }
        }))
    }

    /// expand one axis by n on BOTH sides.
    /// `dom.expand(0, 1)` adds 1 cell on left and 1 on right of axis 0.
    pub fn expand(&self, axis: impl IntoAxis, n: isize) -> Domain<R> {
        self.extend(axis.into_axis(), -n, n)
    }

    /// true if `point` lies within [lo, hi) on every axis.
    pub fn contains(&self, point: [isize; R]) -> bool {
        (0..R).all(|a| point[a] >= self.spaces[a].lo && point[a] < self.spaces[a].hi)
    }

    /// precomputed row-major strides.
    pub fn strides(&self) -> &[usize; R] {
        &self.strides
    }

    /// convert domain-coordinate point to flat index, via the one [`crate::layout::flat_offset`]
    /// value-path formula.
    pub fn flat_index(&self, point: [isize; R]) -> usize {
        crate::layout::flat_offset(
            point,
            std::array::from_fn(|a| self.spaces[a].lo),
            self.strides,
        )
    }

    /// convert flat index back to domain coordinates via divmod on strides.
    ///
    /// processes axes in DECREASING-stride order (highest axis first) so the
    /// divmod chain peels the slowest-varying axis off first. with the
    /// physical-x-fastest convention (`strides[0] == 1`, `strides[R-1]` =
    /// product of lower extents), that means iterating `R-1 → 0`.
    pub fn unflatten(&self, flat_idx: usize) -> [isize; R] {
        let mut point = [0isize; R];
        let mut remaining = flat_idx;
        for aa in (0..R).rev() {
            point[aa] = (remaining / self.strides[aa]) as isize + self.spaces[aa].lo;
            remaining %= self.strides[aa];
        }
        point
    }
}

// =============================================================================
// domain iteration
// =============================================================================

/// iterator over all coordinates in a domain, row-major order.
pub struct DomainIter<const R: usize> {
    lo: [isize; R],
    hi: [isize; R],
    current: [isize; R],
    done: bool,
}

impl<const R: usize> Iterator for DomainIter<R> {
    type Item = [isize; R];

    #[inline]
    fn next(&mut self) -> Option<[isize; R]> {
        if self.done {
            return None;
        }
        let result = self.current;
        // advance in row-major order (last axis fastest)
        let mut carry = true;
        for ax in (0..R).rev() {
            if carry {
                self.current[ax] += 1;
                if self.current[ax] < self.hi[ax] {
                    carry = false;
                } else {
                    self.current[ax] = self.lo[ax];
                }
            }
        }
        if carry {
            self.done = true;
        }
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            return (0, Some(0));
        }
        let mut remaining = 1usize;
        for ax in 0..R {
            remaining *= (self.hi[ax] - self.lo[ax]) as usize;
        }
        (remaining, Some(remaining))
    }
}

impl<const R: usize> Domain<R> {
    /// iterate all coordinates in the domain, row-major order.
    pub fn iter(&self) -> DomainIter<R> {
        let lo = std::array::from_fn(|ax| self.spaces[ax].lo);
        let hi = std::array::from_fn(|ax| self.spaces[ax].hi);
        let done = self.volume() == 0;
        DomainIter {
            lo,
            hi,
            current: lo,
            done,
        }
    }
}

impl<const R: usize> IntoIterator for &Domain<R> {
    type Item = [isize; R];
    type IntoIter = DomainIter<R>;
    fn into_iter(self) -> DomainIter<R> {
        self.iter()
    }
}

pub fn domain<const R: usize>(spaces: [Space; R]) -> Domain<R> {
    Domain::new(spaces)
}

/// a named index variable. use `.over(n)` or `.over((lo, hi))` to produce a Space.
#[derive(Debug, Clone, Copy)]
pub struct IndexName {
    pub name: &'static str,
}

impl IndexName {
    /// `index("i").over(n)` -> Space [0, n)
    /// `index("i").over((lo, hi))` -> Space [lo, hi)
    pub fn over(self, args: impl IntoRange) -> Space {
        let (lo, hi) = args.into_range();
        assert!(hi >= lo, "IndexName::over: hi ({}) < lo ({})", hi, lo);
        Space {
            name: self.name,
            lo,
            hi,
        }
    }
}

pub fn index(name: &'static str) -> IndexName {
    IndexName { name }
}

/// trait for `.over()` argument flexibility.
pub trait IntoRange {
    fn into_range(self) -> (isize, isize);
}

impl IntoRange for usize {
    fn into_range(self) -> (isize, isize) {
        (0, self as isize)
    }
}

impl IntoRange for isize {
    fn into_range(self) -> (isize, isize) {
        (0, self)
    }
}

impl IntoRange for (usize, usize) {
    fn into_range(self) -> (isize, isize) {
        (self.0 as isize, self.1 as isize)
    }
}

impl IntoRange for (isize, isize) {
    fn into_range(self) -> (isize, isize) {
        self
    }
}

impl IntoRange for i32 {
    fn into_range(self) -> (isize, isize) {
        (0, self as isize)
    }
}

impl IntoRange for u32 {
    fn into_range(self) -> (isize, isize) {
        (0, self as isize)
    }
}

impl IntoRange for (i32, i32) {
    fn into_range(self) -> (isize, isize) {
        (self.0 as isize, self.1 as isize)
    }
}

/// decompose index arrays into named components.
/// `let (ii, jj) = p.split();` for 2D, `let (ii, jj, kk) = p.split();` for 3D.
pub trait Split {
    type Output;
    fn split(self) -> Self::Output;
}

impl Split for [isize; 1] {
    type Output = isize;
    fn split(self) -> isize {
        self[0]
    }
}

impl Split for [isize; 2] {
    type Output = (isize, isize);
    fn split(self) -> (isize, isize) {
        (self[0], self[1])
    }
}

impl Split for [isize; 3] {
    type Output = (isize, isize, isize);
    fn split(self) -> (isize, isize, isize) {
        (self[0], self[1], self[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_basics() {
        let i = index("i");
        let j = index("j");
        let d = domain([i.over(10), j.over(20)]);
        assert_eq!(d.rank(), 2);
        assert_eq!(d.shape(), [10, 20]);
        assert_eq!(d.volume(), 200);
    }

    #[test]
    fn test_unique_ids() {
        let d1 = domain([index("x").over(5)]);
        let d2 = domain([index("x").over(5)]);
        assert_ne!(d1.id, d2.id);
    }

    #[test]
    #[should_panic(expected = "duplicate space name")]
    fn test_duplicate_space_panics() {
        let i = index("i");
        domain([i.over(10), i.over(20)]);
    }

    #[test]
    fn test_index_over_tuple() {
        let d = domain([index("i").over((2, 10))]);
        assert_eq!(d.shape(), [8]);
        assert_eq!(d.spaces[0].lo, 2);
        assert_eq!(d.spaces[0].hi, 10);
    }

    #[test]
    fn test_index_over_shorthand() {
        let d = domain([index("i").over(10)]);
        assert_eq!(d.shape(), [10]);
        assert_eq!(d.spaces[0].lo, 0);
        assert_eq!(d.spaces[0].hi, 10);
    }

    #[test]
    fn test_contract() {
        let d = domain([index("i").over(100), index("j").over(200)]);
        let c = d.contract(2);
        assert_eq!(c.spaces[0].lo, 2);
        assert_eq!(c.spaces[0].hi, 98);
        assert_eq!(c.spaces[1].lo, 2);
        assert_eq!(c.spaces[1].hi, 198);
        assert_eq!(c.shape(), [96, 196]);
    }

    #[test]
    fn test_negative_domain() {
        let d = domain([index("i").over((-2_isize, 102_isize))]);
        assert_eq!(d.shape(), [104]);
        assert_eq!(d.spaces[0].lo, -2);
        assert_eq!(d.spaces[0].hi, 102);
        let c = d.contract(2);
        assert_eq!(c.spaces[0].lo, 0);
        assert_eq!(c.spaces[0].hi, 100);
        assert_eq!(c.shape(), [100]);
    }

    #[test]
    #[should_panic(expected = "empty space")]
    fn test_contract_past_half_panics() {
        // size-2 axis contracted by 2 yields lo=2, hi=-2 (hi < lo). without the
        // construction guard, Space::size casts the negative isize to ~1.8e19,
        // poisoning every allocation-sizing path. the guard in Domain::new must
        // reject it at construction.
        let d = domain([index("i").over(2)]);
        d.contract(2);
    }

    #[test]
    fn test_strides_1d() {
        let d = domain([index("i").over(10)]);
        assert_eq!(d.strides(), &[1]);
    }

    #[test]
    fn test_strides_2d() {
        // shape [3, 4]: axis 0 fastest (stride 1), axis 1 stride = N_axis_0 = 3.
        let d = domain([index("i").over(3), index("j").over(4)]);
        assert_eq!(d.strides(), &[1, 3]);
    }

    #[test]
    fn test_strides_3d() {
        // shape [2, 3, 4]: axis 0 stride 1, axis 1 = 2, axis 2 = 2*3 = 6.
        let d = domain([index("i").over(2), index("j").over(3), index("k").over(4)]);
        assert_eq!(d.strides(), &[1, 2, 6]);
    }

    #[test]
    fn test_flat_index_1d() {
        let d = domain([index("i").over(10)]);
        assert_eq!(d.flat_index([0]), 0);
        assert_eq!(d.flat_index([5]), 5);
        assert_eq!(d.flat_index([9]), 9);
    }

    #[test]
    fn test_flat_index_2d() {
        // shape [3, 4], strides [1, 3]: flat = i*1 + j*3.
        let d = domain([index("i").over(3), index("j").over(4)]);
        assert_eq!(d.flat_index([0, 0]), 0);
        assert_eq!(d.flat_index([1, 0]), 1); // step in axis 0 → +1 (fastest)
        assert_eq!(d.flat_index([0, 1]), 3); // step in axis 1 → +3
        assert_eq!(d.flat_index([2, 3]), 11);
    }

    #[test]
    fn test_flat_index_with_offset() {
        let d = domain([
            index("i").over((-2_isize, 3_isize)),
            index("j").over((5_isize, 9_isize)),
        ]);
        // shape [5, 4], strides [1, 5]
        assert_eq!(d.strides(), &[1, 5]);
        assert_eq!(d.flat_index([-2, 5]), 0);
        assert_eq!(d.flat_index([-1, 5]), 1); // step axis 0
        assert_eq!(d.flat_index([-2, 6]), 5); // step axis 1
        assert_eq!(d.flat_index([2, 8]), 19); // (2-(-2))*1 + (8-5)*5 = 4 + 15
    }

    #[test]
    fn test_unflatten_1d() {
        let d = domain([index("i").over(10)]);
        assert_eq!(d.unflatten(0), [0]);
        assert_eq!(d.unflatten(5), [5]);
        assert_eq!(d.unflatten(9), [9]);
    }

    #[test]
    fn test_unflatten_2d() {
        // shape [3, 4], strides [1, 3]: flat = i + j*3.
        let d = domain([index("i").over(3), index("j").over(4)]);
        assert_eq!(d.unflatten(0), [0, 0]);
        assert_eq!(d.unflatten(1), [1, 0]); // step axis 0
        assert_eq!(d.unflatten(3), [0, 1]); // step axis 1
        assert_eq!(d.unflatten(11), [2, 3]);
    }

    #[test]
    fn test_unflatten_with_offset() {
        // shape [5, 4], strides [1, 5]: flat = (i+2) + (j-5)*5.
        let d = domain([
            index("i").over((-2_isize, 3_isize)),
            index("j").over((5_isize, 9_isize)),
        ]);
        assert_eq!(d.unflatten(0), [-2, 5]);
        assert_eq!(d.unflatten(1), [-1, 5]);
        assert_eq!(d.unflatten(5), [-2, 6]);
        assert_eq!(d.unflatten(19), [2, 8]);
    }

    #[test]
    fn test_flat_index_unflatten_roundtrip() {
        let d = domain([index("i").over(3), index("j").over(4), index("k").over(5)]);
        for flat in 0..d.volume() {
            let point: [isize; 3] = d.unflatten(flat);
            assert_eq!(
                d.flat_index(point),
                flat,
                "roundtrip failed for flat={}",
                flat
            );
        }
    }

    #[test]
    fn test_contract_preserves_strides() {
        let d = domain([index("i").over(100), index("j").over(200)]);
        let c = d.contract(2);
        // contracted shape [96, 196]: physical-x-fastest → strides [1, 96].
        assert_eq!(c.strides(), &[1, 96]);
    }

    #[test]
    fn test_slab_2d() {
        let d = domain([index("i").over(10), index("j").over(20)]);
        let s = d.slab(0, (2, 5));
        assert_eq!(s.spaces[0].lo, 2);
        assert_eq!(s.spaces[0].hi, 5);
        assert_eq!(s.spaces[1].lo, 0);
        assert_eq!(s.spaces[1].hi, 20);
        assert_eq!(s.shape(), [3, 20]);
    }

    #[test]
    fn test_slab_3d_axis_2() {
        let d = domain([
            index("i").over(10),
            index("j").over(10),
            index("k").over(10),
        ]);
        let s = d.slab(2, (0, 2));
        assert_eq!(s.spaces[0].lo, 0);
        assert_eq!(s.spaces[0].hi, 10);
        assert_eq!(s.spaces[2].lo, 0);
        assert_eq!(s.spaces[2].hi, 2);
        assert_eq!(s.shape(), [10, 10, 2]);
    }

    #[test]
    fn test_intersect_2d() {
        let a = domain([index("i").over((0, 10)), index("j").over((0, 20))]);
        let b = domain([index("i").over((3, 15)), index("j").over((5, 12))]);
        let c = a.intersect(&b);
        assert_eq!(c.spaces[0].lo, 3);
        assert_eq!(c.spaces[0].hi, 10);
        assert_eq!(c.spaces[1].lo, 5);
        assert_eq!(c.spaces[1].hi, 12);
    }

    #[test]
    #[should_panic(expected = "empty result")]
    fn test_intersect_empty_panics() {
        let a = domain([index("i").over((0, 5))]);
        let b = domain([index("i").over((10, 20))]);
        a.intersect(&b);
    }

    #[test]
    fn test_hull_2d() {
        let a = domain([index("i").over((2, 5)), index("j").over((3, 7))]);
        let b = domain([index("i").over((0, 8)), index("j").over((5, 10))]);
        let c = a.hull(&b);
        assert_eq!(c.spaces[0].lo, 0);
        assert_eq!(c.spaces[0].hi, 8);
        assert_eq!(c.spaces[1].lo, 3);
        assert_eq!(c.spaces[1].hi, 10);
    }

    #[test]
    fn test_boundary_lo() {
        let d = domain([index("i").over((-2, 102)), index("j").over(100)]);
        let b = d.boundary(0, Side::Lo, 2);
        assert_eq!(b.spaces[0].lo, -2);
        assert_eq!(b.spaces[0].hi, 0);
        assert_eq!(b.spaces[1].lo, 0);
        assert_eq!(b.spaces[1].hi, 100);
    }

    #[test]
    fn test_boundary_hi() {
        let d = domain([index("i").over((-2, 102)), index("j").over(100)]);
        let b = d.boundary(0, Side::Hi, 2);
        assert_eq!(b.spaces[0].lo, 100);
        assert_eq!(b.spaces[0].hi, 102);
        assert_eq!(b.spaces[1].lo, 0);
        assert_eq!(b.spaces[1].hi, 100);
    }

    #[test]
    fn test_contract_axis() {
        let d = domain([index("i").over(100), index("j").over(200)]);
        let c = d.contract_axis(0, 3);
        assert_eq!(c.spaces[0].lo, 3);
        assert_eq!(c.spaces[0].hi, 97);
        assert_eq!(c.spaces[1].lo, 0);
        assert_eq!(c.spaces[1].hi, 200);
    }

    #[test]
    fn test_contains() {
        let d = domain([index("i").over((-2, 5)), index("j").over((3, 10))]);
        assert!(d.contains([-2, 3]));
        assert!(d.contains([4, 9]));
        assert!(!d.contains([5, 3]));
        assert!(!d.contains([-2, 10]));
        assert!(!d.contains([-3, 5]));
    }

    #[test]
    fn test_extend() {
        let d = domain([index("i").over(10), index("j").over(20)]);
        // extend axis 0 by 1 on right
        let e = d.extend(0, 0, 1);
        assert_eq!(e.spaces[0].lo, 0);
        assert_eq!(e.spaces[0].hi, 11);
        assert_eq!(e.spaces[1].hi, 20); // unchanged
    }

    #[test]
    fn test_extend_both_sides() {
        let d = domain([index("i").over(10)]);
        let e = d.extend(0, -2, 3);
        assert_eq!(e.spaces[0].lo, -2);
        assert_eq!(e.spaces[0].hi, 13);
        assert_eq!(e.shape(), [15]);
    }

    #[test]
    fn test_expand() {
        let d = domain([index("i").over(10), index("j").over(20)]);
        let e = d.expand(1, 1);
        assert_eq!(e.spaces[0].lo, 0);
        assert_eq!(e.spaces[0].hi, 10); // unchanged
        assert_eq!(e.spaces[1].lo, -1);
        assert_eq!(e.spaces[1].hi, 21);
    }

    #[test]
    fn test_ct_face_domain() {
        // face domain: one extra cell in normal direction
        let interior = domain([
            index("i").over(64),
            index("j").over(64),
            index("k").over(64),
        ]);
        let x_face = interior.extend(0, 0, 1);
        assert_eq!(x_face.shape(), [65, 64, 64]);
        let y_face = interior.extend(1, 0, 1);
        assert_eq!(y_face.shape(), [64, 65, 64]);
    }

    #[test]
    fn test_ct_edge_domain() {
        // edge domain: extra in both transverse directions
        let interior = domain([
            index("i").over(64),
            index("j").over(64),
            index("k").over(64),
        ]);
        // E_x at x-edges: extra in y AND z
        let x_edge = interior.extend(1, 0, 1).extend(2, 0, 1);
        assert_eq!(x_edge.shape(), [64, 65, 65]);
        // E_z at z-edges: extra in x AND y
        let z_edge = interior.extend(0, 0, 1).extend(1, 0, 1);
        assert_eq!(z_edge.shape(), [65, 65, 64]);
    }

    #[test]
    fn test_ct_mhd_flux_domain() {
        // MHD flux domain: face + transverse ghost expansion
        let interior = domain([
            index("i").over(64),
            index("j").over(64),
            index("k").over(64),
        ]);
        // flux[0] (x-face): +1 in x, +1 ghost on each side of y and z
        let x_flux = interior.extend(0, 0, 1).expand(1, 1).expand(2, 1);
        assert_eq!(x_flux.shape(), [65, 66, 66]);
        assert_eq!(x_flux.spaces[1].lo, -1);
        assert_eq!(x_flux.spaces[1].hi, 65);
    }

    #[test]
    fn test_3d_ghost_zone_workflow() {
        // typical 3D ghost zone decomposition
        let ng = 2_isize;
        let n = 100_isize;
        let dom = domain([
            index("i").over((-ng, n + ng)),
            index("j").over((-ng, n + ng)),
            index("k").over((-ng, n + ng)),
        ]);

        let interior = dom.contract(ng);
        assert_eq!(interior.spaces[0].lo, 0);
        assert_eq!(interior.spaces[0].hi, n);

        // x-left ghost slab
        let x_lo = dom.boundary(0, Side::Lo, ng);
        assert_eq!(x_lo.shape(), [2, 104, 104]);

        // x-left face (full y-z extent, 1 cell thick at interior boundary)
        let x_lo_face = dom.slab(0, (0, 1));
        assert_eq!(x_lo_face.shape(), [1, 104, 104]);

        // corner: intersection of x-lo and y-lo ghost slabs
        let y_lo = dom.boundary(1, Side::Lo, ng);
        let corner = x_lo.intersect(&y_lo);
        assert_eq!(corner.shape(), [2, 2, 104]);
    }

    #[test]
    fn test_overlaps() {
        let a = domain([index("i").over((0, 10))]);
        let b = domain([index("i").over((5, 15))]);
        let c = domain([index("i").over((10, 20))]);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c)); // touching but not overlapping
    }

    #[test]
    fn test_difference_1d() {
        // [0, 10) \ [2, 8) = [0, 2) + [8, 10)
        let full = domain([index("i").over((0, 10))]);
        let inner = domain([index("i").over((2, 8))]);
        let diff = full.difference(&inner);
        assert_eq!(diff.len(), 2);
        let total: usize = diff.iter().map(|d| d.volume()).sum();
        assert_eq!(total, 4); // 2 + 2
    }

    #[test]
    fn test_difference_no_overlap() {
        let a = domain([index("i").over((0, 5))]);
        let b = domain([index("i").over((10, 20))]);
        let diff = a.difference(&b);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0].volume(), 5);
    }

    #[test]
    fn test_difference_full_cover() {
        let a = domain([index("i").over((2, 8))]);
        let b = domain([index("i").over((0, 10))]);
        let diff = a.difference(&b);
        assert_eq!(diff.len(), 0);
    }

    #[test]
    fn test_difference_2d_ghost_zones() {
        // 2D domain with 2-cell ghost layer
        let ng = 2_isize;
        let n = 10_isize;
        let full = domain([
            index("i").over((-ng, n + ng)),
            index("j").over((-ng, n + ng)),
        ]);
        let interior = full.contract(ng);
        let ghosts = full.difference(&interior);

        // 3^2 - 1 = 8 regions (4 faces + 4 corners)
        assert_eq!(ghosts.len(), 8);

        // total ghost volume = full - interior
        let ghost_vol: usize = ghosts.iter().map(|d| d.volume()).sum();
        assert_eq!(ghost_vol, full.volume() - interior.volume());
    }

    #[test]
    fn test_difference_3d_ghost_zones() {
        // 3D domain with 2-cell ghost layer
        let ng = 2_isize;
        let n = 10_isize;
        let full = domain([
            index("i").over((-ng, n + ng)),
            index("j").over((-ng, n + ng)),
            index("k").over((-ng, n + ng)),
        ]);
        let interior = full.contract(ng);
        let ghosts = full.difference(&interior);

        // 3^3 - 1 = 26 regions (6 faces + 12 edges + 8 corners)
        assert_eq!(ghosts.len(), 26);

        // volume conservation
        let ghost_vol: usize = ghosts.iter().map(|d| d.volume()).sum();
        assert_eq!(ghost_vol, full.volume() - interior.volume());
    }
}

// =============================================================================
// algebraic laws (axioms)
//
// the domain algebra is verified against its GROUND-TRUTH semantics: a `Domain`
// IS its set of integer lattice points, and every operation must agree with the
// corresponding set / lattice operation on those points. these are the axioms
// the rest of the codebase (fields, kernels, the amr ghost decomposition) is
// entitled to assume. each is checked over a stream of randomly generated boxes.
//
// the structure is the LATTICE OF AXIS-ALIGNED BOXES under inclusion:
//   * meet   = `intersect`  (the largest box inside both)            [partial: may be empty]
//   * join   = `hull`       (the smallest box containing both — the bounding box)
//   * `difference` is a DISJOINT PARTITION of the set difference A \ B
//   * `expand` / `contract_axis` are inverse on a single axis
//   * `flat_index` / `unflatten` are mutually inverse coordinate <-> offset maps
//
// NOTE on `hull`: it is the box-lattice join; it is not a set union, since boxes are not
// closed under union. so the law is the bounding-box law `A,B subseteq hull(A,B)`, and hull is
// the LEAST box with that property; hull(A,B) is not the cell set `cells(A) union cells(B)`.
// =============================================================================
#[cfg(test)]
mod laws {
    use super::*;
    use std::collections::HashSet;

    // ground truth: a domain denotes its set of integer cells.
    fn cells<const R: usize>(d: &Domain<R>) -> HashSet<[isize; R]> {
        d.iter().collect()
    }

    // deterministic splitmix-style prng — no external dep, no Date/rand (which
    // the workspace forbids in build-affecting code). boxes live in a small
    // coordinate window so the cell-set checks stay cheap while still exercising
    // disjoint / overlapping / nested / touching configurations.
    struct Rng(u64);
    impl Rng {
        fn bits(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn in_range(&mut self, lo: isize, hi: isize) -> isize {
            lo + (self.bits() % ((hi - lo) as u64)) as isize
        }
        // a box with each axis lo in [-4, 4) and size in [1, 5).
        fn box_r<const R: usize>(&mut self, names: [&'static str; R]) -> Domain<R> {
            Domain::new(std::array::from_fn(|a| {
                let lo = self.in_range(-4, 4);
                let size = self.in_range(1, 5);
                Space {
                    name: names[a],
                    lo,
                    hi: lo + size,
                }
            }))
        }
        fn point_near<const R: usize>(&mut self) -> [isize; R] {
            std::array::from_fn(|_| self.in_range(-6, 10))
        }
    }

    const N3: [&str; 3] = ["i", "j", "k"];
    const N2: [&str; 2] = ["i", "j"];
    const ITERS: usize = 4000;

    // AXIOM: intersect == set meet (when non-empty), commutative, idempotent.
    #[test]
    fn intersect_is_set_meet() {
        let mut rng = Rng(0x1234_5678);
        for _ in 0..ITERS {
            let a = rng.box_r(N3);
            let b = rng.box_r(N3);
            // idempotent on every box.
            assert_eq!(cells(&a.intersect(&a)), cells(&a));
            if a.overlaps(&b) {
                let m = a.intersect(&b);
                assert_eq!(cells(&m), &cells(&a) & &cells(&b), "intersect != set meet");
                // commutative.
                assert_eq!(cells(&m), cells(&b.intersect(&a)));
                // meet is a lower bound: m ⊆ a and m ⊆ b.
                assert!(cells(&m).is_subset(&cells(&a)));
                assert!(cells(&m).is_subset(&cells(&b)));
            } else {
                // disjoint boxes do not overlap — and `overlaps` agrees with the
                // set predicate exactly.
                assert!(
                    (&cells(&a) & &cells(&b)).is_empty(),
                    "overlaps() lied: sets DO meet"
                );
            }
        }
    }

    // AXIOM: hull is the box-lattice JOIN — least box containing both operands.
    #[test]
    fn hull_is_least_upper_bound() {
        let mut rng = Rng(0xA1B2_C3D4);
        for _ in 0..ITERS {
            let a = rng.box_r(N3);
            let b = rng.box_r(N3);
            let h = a.hull(&b);
            let (ca, cb, ch) = (cells(&a), cells(&b), cells(&h));
            // upper bound: contains both.
            assert!(ca.is_subset(&ch), "hull does not contain a");
            assert!(cb.is_subset(&ch), "hull does not contain b");
            // LEAST such box: every face is tight, so pulling any face in by one
            // must drop a cell of a or b (otherwise a smaller upper bound exists).
            for ax in 0..3 {
                for face in [h.extend(ax, 1, 0), h.extend(ax, 0, -1)] {
                    let cf = cells(&face);
                    assert!(
                        !ca.is_subset(&cf) || !cb.is_subset(&cf),
                        "hull not tight on axis {ax}: a smaller box still contains both"
                    );
                }
            }
            // idempotent.
            assert_eq!(cells(&a.hull(&a)), ca);
        }
    }

    // AXIOM: difference is a DISJOINT partition of the set difference A \ B.
    #[test]
    fn difference_is_a_disjoint_partition_of_set_difference() {
        let mut rng = Rng(0xDEAD_BEEF);
        for _ in 0..ITERS {
            let a = rng.box_r(N3);
            let b = rng.box_r(N3);
            let parts = a.difference(&b);

            // pairwise disjoint.
            for ii in 0..parts.len() {
                for jj in (ii + 1)..parts.len() {
                    assert!(!parts[ii].overlaps(&parts[jj]), "difference parts overlap");
                }
            }
            // disjoint => union cardinality == sum of volumes.
            let union: HashSet<[isize; 3]> = parts.iter().flat_map(|d| d.iter()).collect();
            let vol_sum: usize = parts.iter().map(|d| d.volume()).sum();
            assert_eq!(union.len(), vol_sum, "parts not disjoint by volume");

            // CORRECTNESS: union == set difference cells(a) \ cells(b).
            let expected: HashSet<[isize; 3]> = &cells(&a) - &cells(&b);
            assert_eq!(union, expected, "difference != set difference");

            // RECONSTRUCTION: (A \ B) ⊎ (A ∩ B) == A.
            let inter = &cells(&a) & &cells(&b);
            assert_eq!(
                &union | &inter,
                cells(&a),
                "A\\B and A∩B do not reconstruct A"
            );
        }
    }

    // AXIOM: guillotine_difference is ALSO a disjoint partition of A \ B (same
    // set as `difference`), but with at most 2*R boxes.
    #[test]
    fn guillotine_difference_is_minimal_disjoint_set_difference() {
        let mut rng = Rng(0x0DDB_A11);
        for _ in 0..ITERS {
            let a = rng.box_r(N3);
            let b = rng.box_r(N3);
            let parts = a.guillotine_difference(&b);

            // at most 2*R boxes (the minimality claim).
            assert!(
                parts.len() <= 2 * 3,
                "guillotine produced {} > 2R boxes",
                parts.len()
            );

            // pairwise disjoint.
            for ii in 0..parts.len() {
                for jj in (ii + 1)..parts.len() {
                    assert!(!parts[ii].overlaps(&parts[jj]), "guillotine parts overlap");
                }
            }
            let union: HashSet<[isize; 3]> = parts.iter().flat_map(|d| d.iter()).collect();
            let vol_sum: usize = parts.iter().map(|d| d.volume()).sum();
            assert_eq!(
                union.len(),
                vol_sum,
                "guillotine parts not disjoint by volume"
            );

            // SAME cell set as the maximal `difference` (both are A \ B).
            let expected: HashSet<[isize; 3]> = &cells(&a) - &cells(&b);
            assert_eq!(union, expected, "guillotine_difference != set difference");
            let maximal: HashSet<[isize; 3]> =
                a.difference(&b).iter().flat_map(|d| d.iter()).collect();
            assert_eq!(union, maximal, "guillotine and difference disagree");
        }
    }

    // AXIOM: expand and contract_axis are inverse on a single axis.
    #[test]
    fn expand_contract_are_inverse() {
        let mut rng = Rng(0x0F0F_0F0F);
        for _ in 0..ITERS {
            let d = rng.box_r(N2);
            for ax in 0..2 {
                for n in 1..=3isize {
                    let round = d.expand(ax, n).contract_axis(ax, n);
                    assert_eq!(
                        cells(&round),
                        cells(&d),
                        "expand∘contract != id (axis {ax}, n {n})"
                    );
                }
            }
        }
    }

    // AXIOM: `contains` is exactly set membership.
    #[test]
    fn contains_is_set_membership() {
        let mut rng = Rng(0xCAFE_F00D);
        for _ in 0..ITERS {
            let d = rng.box_r(N3);
            let set = cells(&d);
            let p = rng.point_near::<3>();
            assert_eq!(
                d.contains(p),
                set.contains(&p),
                "contains != membership at {p:?}"
            );
        }
    }

    // AXIOM: flat_index and unflatten are mutual inverses, and flat_index is a
    // bijection onto [0, volume).
    #[test]
    fn flat_index_unflatten_bijection() {
        let mut rng = Rng(0x5151_2727);
        for _ in 0..ITERS {
            let d = rng.box_r(N3);
            let mut seen = HashSet::new();
            for p in d.iter() {
                let f = d.flat_index(p);
                assert!(f < d.volume(), "flat index out of range");
                assert!(seen.insert(f), "flat_index not injective at {p:?}");
                assert_eq!(d.unflatten(f), p, "unflatten∘flat_index != id");
            }
            assert_eq!(seen.len(), d.volume(), "flat_index not onto [0, volume)");
        }
    }

    // TRAP PIN: `DomainIter` advances LAST-axis-fastest, but storage is
    // axis-0-fastest (`flat_index`). these orders differ for D>=2. this is
    // intentional (the iterator visits every cell exactly once — the bijection
    // law above proves that), but it means `dom.iter()` does NOT yield cells in
    // ascending `flat_index` order. any code that COLLECTS `iter()` into a
    // sequential buffer and then reads it back BY POSITION is transposed (the
    // checkpoint-restart bug was exactly this). this test makes the
    // divergence explicit and load-bearing: a future "fix" that silently aligns
    // the iterator to storage order must update this pin and audit collectors.
    #[test]
    fn iter_order_is_not_storage_order_for_d_ge_2() {
        // a non-square 2d box: iteration walks j (axis 1) fastest, storage walks
        // i (axis 0) fastest, so the produced flat-index sequence is not sorted.
        let d = domain([index("i").over(3), index("j").over(4)]);
        let order: Vec<usize> = d.iter().map(|p| d.flat_index(p)).collect();
        assert!(
            order.windows(2).any(|w| w[0] > w[1]),
            "iter() yielded storage order for D>=2 — collectors that read by position would silently transpose"
        );
        // but it is still a bijection onto [0, volume): every offset visited once.
        let seen: HashSet<usize> = order.iter().copied().collect();
        assert_eq!(
            seen.len(),
            d.volume(),
            "iter() must still visit every cell exactly once"
        );
    }
}
