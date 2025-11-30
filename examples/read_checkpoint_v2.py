"""
example usage of the new io reader.

demonstrates:
- loading checkpoints with Result types
- accessing metadata, mesh, fields
- handling MHD face-centered fields
- extracting interior (ghost-free) data
- working with multi-partition data
"""

from simbi.reader.io import get_base_fields, read_checkpoint


def example_basic_usage(filename: str):
    """basic checkpoint reading."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)

    # read checkpoint (returns Result)
    result = read_checkpoint(filename)

    # check for errors
    if result.is_err():
        print(f"Failed to read checkpoint: {result.error}")
        return

    # extract checkpoint
    checkpoint = result.value
    print(f"✓ Loaded checkpoint from {filename}")

    # access metadata
    meta = checkpoint.metadata
    print("\nMetadata:")
    print(f"  Time: {meta.time:.6e}")
    print(f"  Timestep: {meta.dt:.6e}")
    print(f"  Iteration: {meta.iteration}")
    print(f"  Gamma: {meta.gamma}")
    print(f"  Is MHD: {meta.is_mhd}")
    print(f"  Coord System: {meta.coord_system}")
    print(f"  Dimensions: {meta.dimensions}")

    # access base level
    base = checkpoint.base_level()
    print("\nBase Level:")
    print(f"  Level ID: {base.level_id}")
    print(f"  Partitions: {base.num_partitions}")

    # mesh info
    mesh = base.mesh
    print("\nMesh:")
    print(f"  Global cells: {mesh.global_cells}")
    print(f"  Halo radius: {mesh.halo_radius}")
    print(f"  Metric: {mesh.metric}")
    print(f"  Bounds: {mesh.bounds_min} to {mesh.bounds_max}")

    # access first partition
    if base.num_partitions > 0:
        partition = base.partitions[0]
        print("\nPartition 0:")
        print(f"  Device ID: {partition.device_id}")
        print(f"  Owned domain: {partition.owned_domain}")

        # primitive fields
        prims = partition.hydro.primitives
        print("\nPrimitive Fields:")
        for name, field in prims.items():
            print(f"  {name}: shape {field.shape}, domain {field.domain}")

        # check for magnetic fields
        if partition.hydro.has_magnetic:
            print("\nMagnetic Fields (Face-Centered):")
            for name, field in partition.hydro.magnetic.items():
                print(f"  {name}: shape {field.shape}, domain {field.domain}")

    print()


def example_field_extraction(filename: str):
    """extract and process field data."""
    print("=" * 80)
    print("EXAMPLE 2: Field Extraction")
    print("=" * 80)

    checkpoint = read_checkpoint(filename).unwrap()
    base = checkpoint.base_level()
    partition = base.partitions[0]

    # get field with ghosts
    rho_field = partition.hydro.primitives["rho"]
    print("Density field (with ghosts):")
    print(f"  Shape: {rho_field.shape}")
    print(f"  Domain: {rho_field.domain}")
    print(f"  Min: {rho_field.data.min():.6e}")
    print(f"  Max: {rho_field.data.max():.6e}")

    # extract interior (remove ghosts)
    halo = base.mesh.halo_radius
    rho_interior = rho_field.interior(halo)
    print("\nDensity field (interior only):")
    print(f"  Shape: {rho_interior.shape}")
    print(f"  Domain: {rho_interior.domain}")
    print(f"  Min: {rho_interior.data.min():.6e}")
    print(f"  Max: {rho_interior.data.max():.6e}")

    print()


def example_mhd_fields(filename: str):
    """work with face-centered magnetic fields."""
    print("=" * 80)
    print("EXAMPLE 3: MHD Face-Centered Fields")
    print("=" * 80)

    checkpoint = read_checkpoint(filename).unwrap()
    partition = checkpoint.base_level().partitions[0]

    if not partition.hydro.has_magnetic:
        print("Not an MHD simulation, skipping.")
        return

    mag = partition.hydro.magnetic

    # face-centered fields have different shapes
    print("Face-centered magnetic fields:")
    for name, field in mag.items():
        print(f"  {name}:")
        print(f"    Shape: {field.shape}")
        print(f"    Domain: {field.domain}")
        print(f"    Min: {field.data.min():.6e}, Max: {field.data.max():.6e}")

    # manually average to cell centers
    b1_faces = mag["b1"].data
    b1_cells = 0.5 * (b1_faces[..., 1:] + b1_faces[..., :-1])
    print("\nB1 averaged to cell centers:")
    print(f"  Shape: {b1_cells.shape}")
    print(f"  Min: {b1_cells.min():.6e}, Max: {b1_cells.max():.6e}")

    print()


def example_convenience_api(filename: str):
    """use convenience function for simple cases."""
    print("=" * 80)
    print("EXAMPLE 4: Convenience API")
    print("=" * 80)

    checkpoint = read_checkpoint(filename).unwrap()

    # get all fields as simple dict (single partition only)
    fields = get_base_fields(checkpoint, unpad=True)

    print("Fields extracted (interior only):")
    for name, arr in fields.items():
        print(f"  {name}: shape {arr.shape}, dtype {arr.dtype}")

    # now use fields like old API
    rho = fields["rho"]
    pressure = fields["p"]
    print("\nDensity stats:")
    print(f"  Mean: {rho.mean():.6e}")
    print(f"  Std: {rho.std():.6e}")

    print()


def example_error_handling(filename: str):
    """demonstrate error handling patterns."""
    print("=" * 80)
    print("EXAMPLE 5: Error Handling")
    print("=" * 80)

    # pattern 1: check is_err() / is_ok()
    result = read_checkpoint(filename)
    if result.is_err():
        print(f"Error: {result.error}")
        return
    checkpoint = result.value
    print("✓ Pattern 1: explicit error check")

    # pattern 2: unwrap() (raises on error)
    try:
        checkpoint = read_checkpoint(filename).unwrap()
        print("✓ Pattern 2: unwrap() succeeded")
    except RuntimeError as e:
        print(f"✗ Pattern 2: unwrap() raised: {e}")

    # pattern 3: unwrap_or (provide default)
    checkpoint = read_checkpoint("nonexistent.h5").unwrap_or(None)
    if checkpoint is None:
        print("✓ Pattern 3: unwrap_or returned None for missing file")

    # pattern 4: map (functional composition)
    fields = (
        read_checkpoint(filename)
        .map(lambda cp: get_base_fields(cp, unpad=True))
        .unwrap_or({})
    )
    print(f"✓ Pattern 4: map + unwrap_or returned {len(fields)} fields")

    print()


def example_amr_multi_level(filename: str):
    """work with multi-level AMR data."""
    print("=" * 80)
    print("EXAMPLE 6: Multi-Level AMR")
    print("=" * 80)

    checkpoint = read_checkpoint(filename).unwrap()

    if not checkpoint.has_refinement:
        print("Single-level simulation, skipping AMR example.")
        return

    print(f"Multi-level simulation with {checkpoint.num_levels} levels:")

    for level in checkpoint.levels:
        print(f"\nLevel {level.level_id}:")
        print(f"  Partitions: {level.num_partitions}")
        print(f"  Mesh resolution: {level.mesh.global_cells}")
        print(f"  Halo radius: {level.mesh.halo_radius}")

        for partition in level.partitions:
            print(f"    Partition {partition.partition_id}:")
            print(f"      Owned: {partition.owned_domain}")
            print(f"      Device: {partition.device_id}")

            # access field from this partition
            rho = partition.hydro.primitives["rho"]
            print(
                f"      rho: {rho.shape}, range [{rho.data.min():.2e}, {rho.data.max():.2e}]"
            )
            print(
                f"      rho: {rho.shape}, range [{rho.data.min():.2e}, {rho.data.max():.2e}]"
            )
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python read_checkpoint_v2.py <checkpoint_file.h5>")
        print("\nExamples:")
        print("  python read_checkpoint_v2.py data/checkpoint_0100.h5")
        sys.exit(1)

    filename = sys.argv[1]

    # run all examples
    example_basic_usage(filename)
    example_field_extraction(filename)
    example_mhd_fields(filename)
    example_convenience_api(filename)
    example_error_handling(filename)
    example_amr_multi_level(filename)

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
