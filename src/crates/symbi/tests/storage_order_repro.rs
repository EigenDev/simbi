// =============================================================================
// storage_order_repro.rs
//
// **diagnostic reproducer** — pin down which axis is the fastest-varying
// in memory, then derive what that means for CUDA coalescing and for the
// numpy/plotting convention. the asserts record the layout the code has, and
// print it, so the consequences below can be read off at a glance.
//
// what this verifies:
//   - the stride each axis of a `Domain<3>([Nx, Ny, Nz])` carries;
//   - the flat index `Domain::flat_index` and `View::flat_index` produce for a
//     known coord `[i, j, k]`;
//   - derived: which axis lives "next to itself" in memory (= the fastest);
//   - derived: the memory addresses a CUDA warp launched with `threadIdx.x`
//     mapped to axis 0 reads, and whether they are contiguous.
//
// run: cargo test -p symbi --test storage_order_repro -- --nocapture
// =============================================================================

use symbi_algebra::{Domain, Space, domain};

const NX: usize = 2; // physical x
const NY: usize = 3; // physical y
const NZ: usize = 4; // physical z

fn make_domain() -> Domain<3> {
    domain([
        Space {
            name: "x",
            lo: 0,
            hi: NX as isize,
        },
        Space {
            name: "y",
            lo: 0,
            hi: NY as isize,
        },
        Space {
            name: "z",
            lo: 0,
            hi: NZ as isize,
        },
    ])
}

#[test]
fn part1_stride_per_axis() {
    let d = make_domain();
    let s = d.strides();
    println!(
        "\npart 1 — Domain<3>([Nx={}, Ny={}, Nz={}]).strides() = {:?}",
        NX, NY, NZ, s,
    );
    println!("  -> axis 0 (physical x) has stride {} cells", s[0]);
    println!("  -> axis 1 (physical y) has stride {} cells", s[1]);
    println!("  -> axis 2 (physical z) has stride {} cells", s[2]);

    // physical-x-fastest convention: axis 0 -> stride 1, axis 2 -> stride Nx*Ny.
    assert_eq!(s[0], 1, "axis 0 (x) stride — must be 1 (fastest)");
    assert_eq!(s[1], NX, "axis 1 (y) stride — must be Nx");
    assert_eq!(s[2], NX * NY, "axis 2 (z) stride — must be Nx*Ny (slowest)");

    println!("  axis 0 (x) is the FASTEST-varying — standard CFD convention");
}

#[test]
fn part2_flat_index_for_known_coords() {
    let d = make_domain();
    let (nx, ny, nz) = (NX as isize, NY as isize, NZ as isize);

    println!("\npart 2 — flat_index for a few corners and edges:");
    for (i, j, k) in [
        (0, 0, 0),
        (0, 0, 1), // step in z
        (0, 0, nz - 1),
        (0, 1, 0), // step in y
        (1, 0, 0), // step in x
        (nx - 1, ny - 1, nz - 1),
    ] {
        let flat = d.flat_index([i, j, k]);
        println!("  ({i}, {j}, {k}) -> flat index {flat}");
    }

    // direct derivation of who's adjacent in memory.
    let flat_origin = d.flat_index([0, 0, 0]);
    let flat_z_plus = d.flat_index([0, 0, 1]);
    let flat_y_plus = d.flat_index([0, 1, 0]);
    let flat_x_plus = d.flat_index([1, 0, 0]);
    println!(
        "  -> moving one step in z costs {} flat slots",
        flat_z_plus - flat_origin,
    );
    println!(
        "  -> moving one step in y costs {} flat slots",
        flat_y_plus - flat_origin,
    );
    println!(
        "  -> moving one step in x costs {} flat slots",
        flat_x_plus - flat_origin,
    );

    assert_eq!(flat_x_plus - flat_origin, 1, "x is the cheapest step");
    assert_eq!(flat_y_plus - flat_origin, NX, "y costs Nx slots");
    assert_eq!(flat_z_plus - flat_origin, NX * NY, "z costs Nx*Ny slots");
}

#[test]
fn part3_what_a_cuda_warp_actually_reads() {
    // CUDA emit at crates/symbi-ir/src/backends/kernel.rs:142 does:
    //   _i0 = blockIdx.x*blockDim.x + threadIdx.x  -> cell axis 0 (= x)
    //   _i1 = blockIdx.y*blockDim.y + threadIdx.y  -> cell axis 1 (= y)
    //   _i2 = blockIdx.z*blockDim.z + threadIdx.z  -> cell axis 2 (= z)
    //
    // a single warp is 32 consecutive `threadIdx` values, so consecutive
    // threads in a warp map to consecutive ii (= cell.x) values.

    let d = make_domain();
    println!("\npart 3 — flat addresses 32 consecutive warp threads would read");
    println!("  (block dispatched at (j=0, k=0), threadIdx.x walks 0..nx)");

    let mut addrs = Vec::new();
    for warp_lane in 0..(NX as isize) {
        let i = warp_lane;
        let flat = d.flat_index([i, 0, 0]);
        addrs.push(flat);
        println!("    thread {warp_lane}: cell ({i}, 0, 0) -> flat = {flat}");
    }

    // pin: adjacent warp threads access ADJACENT memory (stride 1) ->
    // coalesced reads. this is the load-bearing assertion.
    if NX >= 2 {
        let stride_between_threads = addrs[1] - addrs[0];
        assert_eq!(
            stride_between_threads, 1,
            "stride between threadIdx.x = 0 and threadIdx.x = 1 — must be 1",
        );
        println!(
            "  adjacent warp threads access addresses {} element apart -> COALESCED",
            stride_between_threads,
        );
    }
}

#[test]
fn part4_what_numpy_imshow_sees() {
    // saved data file is a flat buffer of size Nx*Ny*Nz. for OT-style 2D
    // (Nz=1) the user reshapes to `(Ny, Nx)` in numpy (last-axis fastest ->
    // matches the axis-0 = x = fastest convention).
    //
    // matplotlib `imshow(arr)` puts numpy axis 0 as ROWS (vertical) and
    // axis 1 as COLUMNS (horizontal). with reshape (Ny, Nx):
    //   - numpy axis 0 = Ny -> vertical screen axis = physical y
    //   - numpy axis 1 = Nx -> horizontal screen axis = physical x
    // standard physics orientation, screen axes aligned with the physical ones.

    let nx = NX;
    let ny = NY;
    let mut buf = vec![0u32; nx * ny];
    // emit-side index formula: flat = i * stride[0] + j * stride[1]
    //                        = i * 1        + j * Nx
    for i in 0..nx {
        for j in 0..ny {
            let flat = i + j * nx;
            buf[flat] = (i * 100 + j) as u32;
        }
    }

    println!("\npart 4 — Nx={nx}, Ny={ny}, flat buffer (i*100 + j tags):");
    println!("  raw buffer order: {:?}", buf);
    println!();
    println!("  reshaped as `arr.reshape((Ny, Nx)) = arr.reshape(({ny}, {nx}))`:");
    for j in 0..ny {
        let row: Vec<u32> = (0..nx).map(|i| buf[i + j * nx]).collect();
        println!("    arr[{j}] = {:?}", row);
    }
    println!();
    println!("  matplotlib `imshow(arr)` puts arr[0] as the TOP row ->");
    println!("  -> numpy axis 0 = Ny = vertical screen axis = physical y");
    println!("  -> numpy axis 1 = Nx = horizontal screen axis = physical x");
    println!("  -> standard physics orientation, NO rotation.");
}
