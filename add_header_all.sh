#!/bin/bash

# Directory to search for .hpp files
DIR=$1

# Find all .hpp files
find "$DIR" -name "*.hpp" | while read -r filepath; do
    filename=$(basename "$filepath")
    echo "Processing $filename..."

    # Check if banner already exists
    if grep -q "SIMBI - Special Relativistic Magnetohydrodynamics Code" "$filepath"; then
        echo "  Banner already exists in $filepath, skipping"
        continue
    fi

    # Get current date and year
    DATE=$(date +%Y-%m-%d)
    YEAR=$(date +%Y)

    # Create temporary file with banner
    TMP_FILE=$(mktemp)

    # Write banner to temp file
    cat > "$TMP_FILE" << EOF
/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            $filename
 * @brief
 * @details
 *
 * @version         0.8.0
 * @date            $DATE
 * @author          Marcus DuPont
 * @email           marcus.dupont@princeton.edu
 *
 *==============================================================================
 * @build           Requirements & Dependencies
 *==============================================================================
 * @requires        C++20
 * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 * @platform        Linux, MacOS
 * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *
 *==============================================================================
 * @documentation   Reference & Notes
 *==============================================================================
 * @usage
 * @note
 * @warning
 * @todo
 * @bug
 * @performance
 *
 *==============================================================================
 * @testing        Quality Assurance
 *==============================================================================
 * @test
 * @benchmark
 * @validation
 *
 *==============================================================================
 * @history        Version History
 *==============================================================================
 * $DATE      v0.8.0      Initial implementation
 *
 *==============================================================================
 * @copyright (C) $YEAR Marcus DuPont. All rights reserved.
 *==============================================================================
 */

EOF

    # Append original content
    cat "$filepath" >> "$TMP_FILE"

    # Replace original file
    mv "$TMP_FILE" "$filepath"

    echo "  Added banner to $filepath"
done

echo "Done!"
