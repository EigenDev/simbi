#ifndef FMR_FLUX_CORRECTION_HPP
#define FMR_FLUX_CORRECTION_HPP

#include "compat.hpp"
#include "compute/field.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "hierarchy.hpp"
#include "level_mapping.hpp"

#include <cstdint>
#include <iostream>

namespace simbi::mesh::fmr {

    // returns the coarse-fine interface faces for flux correction
    template <std::uint64_t Dims>
    struct flux_interface_t {
        domain_t<Dims> coarse_face;   // face in coarse coordinates
        domain_t<Dims> fine_face;     // corresponding face in fine coordinates
        bool is_valid;
    };

    // get lower and upper interface faces in a given direction
    template <std::uint64_t Dims>
    vector_t<flux_interface_t<Dims>, 2> get_flux_interfaces(
        const level_mapping_t<Dims>& map,
        std::uint64_t flux_direction
    )
    {
        vector_t<flux_interface_t<Dims>, 2> interfaces;
        const auto& coverage     = map.coarse_staggered_coverage;
        const auto& coarse_faces = map.coarse_face_domains[flux_direction];

        // lower interface
        {
            auto& lower = interfaces[0];
            lower.is_valid =
                (coverage.start[flux_direction] >
                 coarse_faces.start[flux_direction]);

            if (lower.is_valid) {
                // coarse face: full transverse extent of coverage, 1 cell in
                // flux direction
                lower.coarse_face = coverage;   // base for transverse dims

                //  j_start, the logical flux index
                lower.coarse_face.start[flux_direction] =
                    coverage.start[flux_direction];

                lower.coarse_face.fin[flux_direction] =
                    coverage.start[flux_direction] + 1;

                // map coarse face corners to fine coordinates
                auto fine_start =
                    map.coarse_to_fine_face_base(lower.coarse_face.start);
                auto fine_end =
                    map.coarse_to_fine_face_base(lower.coarse_face.fin);

                // build fine face domain
                lower.fine_face.start = fine_start;
                lower.fine_face.fin   = fine_end;

                // in flux direction: make it 1 cell thick
                lower.fine_face.fin[flux_direction] =
                    fine_start[flux_direction] + 1;
            }
        }

        // upper interface
        {
            auto& upper = interfaces[1];
            upper.is_valid =
                (coverage.fin[flux_direction] <
                 coarse_faces.fin[flux_direction]);

            if (upper.is_valid) {
                // set up coarse face exactly at the end of coverage
                upper.coarse_face = coverage;   // base for transverse

                // This is j_end, the logical flux index
                upper.coarse_face.start[flux_direction] =
                    coverage.fin[flux_direction];

                upper.coarse_face.fin[flux_direction] =
                    coverage.fin[flux_direction] + 1;

                auto fine_start =
                    map.coarse_to_fine_face_base(upper.coarse_face.start);
                auto fine_end =
                    map.coarse_to_fine_face_base(upper.coarse_face.fin);

                upper.fine_face.start = fine_start;
                upper.fine_face.fin   = fine_end;

                // make it 1 cell thick in flux direction
                upper.fine_face.fin[flux_direction] =
                    fine_start[flux_direction] + 1;
            }
        }

        return interfaces;
    }

    // correct fluxes at a single coarse-fine interface
    template <typename T, std::uint64_t Dims>
    void correct_interface_fluxes(
        field_t<T, Dims>& coarse_flux,
        const field_t<T, Dims>& fine_flux,
        const flux_interface_t<Dims>& interface,
        const level_mapping_t<Dims>& map,
        std::uint64_t flux_direction
    )
    {
        if (!interface.is_valid) {
            return;
        }

        for (const auto& coarse_coord : interface.coarse_face) {
            auto fine_children =
                map.fine_face_children(coarse_coord, flux_direction);

            T sum{};
            real volume = 0;
            bool first  = true;

            for (const auto& fine_coord : fine_children) {
                if (fine_flux.domain().contains(fine_coord)) {
                    if (first) {
                        sum   = fine_flux(fine_coord);
                        first = false;
                    }
                    else {
                        sum = sum | add_gas(fine_flux(fine_coord));
                    }
                    volume += 1.0;
                }
            }

            if (volume > 0) {
                coarse_flux(coarse_coord) = sum / volume;
            }
        }
    }

    // correct fluxes in one direction for a level
    template <typename T, std::uint64_t Dims>
    void correct_fluxes_direction(
        field_t<T, Dims>& coarse_flux,
        const field_t<T, Dims>& fine_flux,
        const level_mapping_t<Dims>& map,
        std::uint64_t flux_direction
    )
    {
        auto interfaces = get_flux_interfaces(map, flux_direction);
        for (const auto& interface : interfaces) {
            correct_interface_fluxes(
                coarse_flux,
                fine_flux,
                interface,
                map,
                flux_direction
            );
        }
    }

    // correct all fluxes (all directions) for a level
    template <typename T, std::uint64_t Dims>
    void correct_level_fluxes(
        vector_t<field_t<T, Dims>, Dims>& coarse_fluxes,
        const vector_t<field_t<T, Dims>, Dims>& fine_fluxes,
        const level_mapping_t<Dims>& map
    )
    {
        // correct fluxes in each coordinate direction
        for (std::uint64_t dir = 0; dir < Dims; ++dir) {
            correct_fluxes_direction(
                coarse_fluxes[dir],
                fine_fluxes[dir],
                map,
                dir
            );
        }
    }

    // synchronize all flux interfaces in hierarchy
    template <typename T, std::uint64_t Dims, std::uint32_t MAX_LEVELS = 8>
    void synchronize_fluxes(
        vector_t<vector_t<field_t<T, Dims>, Dims>, MAX_LEVELS>& level_fluxes,
        const mesh_hierarchy_t<Dims>& hierarchy
    )
    {
        // work from finest to coarsest
        for (std::uint64_t lvl = hierarchy.num_levels - 1; lvl > 0; --lvl) {

            // create mapping for this level
            auto map = create_level_mapping(hierarchy, lvl);

            // correct fluxes
            correct_level_fluxes(level_fluxes[lvl - 1], level_fluxes[lvl], map);
        }
    }

    template <typename Sim>
    void
    diagnose_flux_correction(Sim& sim, std::uint64_t lvl, std::uint64_t dir)
    {
        if (lvl == 0) {
            return;
        }

        auto map        = create_level_mapping(sim.hierarchy(), lvl);
        auto interfaces = get_flux_interfaces(map, dir);

        std::cout << "\n=== Flux Interface Diagnostic (Level " << lvl
                  << ", Dir " << dir << ") ===\n";

        for (std::uint64_t i = 0; i < 2; ++i) {
            const auto& interface = interfaces[i];
            if (!interface.is_valid) {
                continue;
            }

            std::cout << (i == 0 ? "Lower" : "Upper") << " interface:\n";
            std::cout << "  Coarse face: " << interface.coarse_face << "\n";
            std::cout << "  Fine face: " << interface.fine_face << "\n";

            // Sample a face
            auto unos          = ones<Sim::dimensions, std::int64_t>();
            auto coarse_sample = interface.coarse_face.fin - unos;
            auto fine_sample   = interface.fine_face.fin - unos;

            auto& coarse_flux = sim.hydro(lvl - 1).flux[dir];
            auto& fine_flux   = sim.hydro(lvl).flux[dir];

            if (coarse_flux.domain().contains(coarse_sample) &&
                fine_flux.domain().contains(fine_sample)) {
                std::cout << "  Coarse flux at " << coarse_sample << ": "
                          << coarse_flux(coarse_sample) << "\n";
                std::cout << "  Fine flux at " << fine_sample << ": "
                          << fine_flux(fine_sample) << "\n";
            }
        }
    }

    template <typename Sim>
    void diagnose_flux_averaging(Sim& sim, std::uint64_t lvl, std::uint64_t dir)
    {
        if (lvl == 0) {
            return;
        }

        auto map        = create_level_mapping(sim.hierarchy(), lvl);
        auto interfaces = get_flux_interfaces(map, dir);

        // Check lower interface
        auto& interface = interfaces[0];
        if (!interface.is_valid) {
            return;
        }

        std::cout << "\n=== Detailed Flux Averaging (Level " << lvl << ", Dir "
                  << dir << ", Lower) ===\n";

        // Pick one coarse face to examine
        auto coarse_sample = interface.coarse_face.start;
        std::cout << "Coarse face at: " << coarse_sample << "\n";

        // Get all fine children
        auto fine_children = map.fine_face_children(coarse_sample, dir);
        std::cout << "Fine children domain: " << fine_children << "\n";

        auto& coarse_flux = sim.hydro(lvl - 1).flux[dir];
        auto& fine_flux   = sim.hydro(lvl).flux[dir];

        // Show each fine flux
        using T = typename Sim::conserved_t;
        T sum{};
        real count = 0;

        std::cout << "Fine fluxes being averaged:\n";
        for (const auto& fine_coord : fine_children) {
            if (fine_flux.domain().contains(fine_coord)) {
                auto flux_val = fine_flux(fine_coord);
                std::cout << "  [" << fine_coord << "]: " << flux_val << "\n";
                if (count == 0) {
                    sum = flux_val;
                }
                else {
                    sum = sum | add_gas(flux_val);
                }
                count += 1.0;
            }
            else {
                std::cout << "  [" << fine_coord << "]: NOT IN DOMAIN\n";
            }
        }

        if (count > 0) {
            auto averaged        = sum / count;
            auto coarse_original = coarse_flux(coarse_sample);

            std::cout << "Average of " << count << " fine fluxes: " << averaged
                      << "\n";
            std::cout << "Coarse flux (original): " << coarse_original << "\n";
            std::cout << "Coarse flux (should be): " << averaged << "\n";
        }
    }

    template <typename Sim>
    void check_flux_conservation(Sim& sim, std::uint64_t lvl)
    {
        if (lvl == 0) {
            return;
        }

        auto map = create_level_mapping(sim.hierarchy(), lvl);

        std::cout << "\n=== Flux Conservation Check (Level " << lvl
                  << ") ===\n";

        for (std::uint64_t dir = 0; dir < Sim::dimensions; ++dir) {
            auto interfaces = get_flux_interfaces(map, dir);

            for (std::uint64_t iface = 0; iface < 2; ++iface) {
                auto& interface = interfaces[iface];
                if (!interface.is_valid) {
                    continue;
                }

                auto& coarse_flux = sim.hydro(lvl - 1).flux[dir];
                auto& fine_flux   = sim.hydro(lvl).flux[dir];

                // Sum coarse fluxes at interface
                real coarse_flux_sum = 0.0;
                for (const auto& coord : interface.coarse_face) {
                    coarse_flux_sum += coarse_flux(coord)[0];   // density flux
                }

                // Sum fine fluxes at interface
                real fine_flux_sum = 0.0;
                for (const auto& coord : interface.fine_face) {
                    if (fine_flux.domain().contains(coord)) {
                        fine_flux_sum += fine_flux(coord)[0];
                    }
                }

                std::cout << "Dir " << dir << ", "
                          << (iface == 0 ? "Lower" : "Upper") << ":\n";
                std::cout << "  Coarse flux sum: " << coarse_flux_sum << "\n";
                std::cout << "  Fine flux sum: " << fine_flux_sum << "\n";
                std::cout << "  Ratio (should be ~" << (map.ratio * map.ratio)
                          << "): " << fine_flux_sum / coarse_flux_sum << "\n";
            }
        }
    }

    template <typename Sim>
    void comprehensive_upper_interface_check(
        Sim& sim,
        std::uint64_t lvl,
        std::uint64_t dir
    )
    {
        if (lvl == 0) {
            return;
        }

        auto map        = create_level_mapping(sim.hierarchy(), lvl);
        auto interfaces = get_flux_interfaces(map, dir);
        auto& upper     = interfaces[1];

        if (!upper.is_valid) {
            std::cout << "Level " << lvl << " Dir " << dir
                      << ": Upper interface NOT VALID\n";
            return;
        }

        std::cout << "\n=== Upper Interface Check (Level " << lvl << ", Dir "
                  << dir << ") ===\n";

        // 1. Check domains
        std::cout << "Coarse coverage: " << map.coarse_coverage << "\n";
        std::cout << "Fine active: " << map.fine_active << "\n";
        std::cout << "Coarse face domain: " << upper.coarse_face << "\n";
        std::cout << "Fine face domain: " << upper.fine_face << "\n";

        // 2. Check face coordinates exist
        auto& coarse_flux = sim.hydro(lvl - 1).flux[dir];
        auto& fine_flux   = sim.hydro(lvl).flux[dir];

        std::cout << "Coarse flux field domain: " << coarse_flux.domain()
                  << "\n";
        std::cout << "Fine flux field domain: " << fine_flux.domain() << "\n";

        // 3. Check a sample coarse face
        auto sample_coarse = upper.coarse_face.start;
        std::cout << "Sample coarse coord: " << sample_coarse << "\n";

        if (!coarse_flux.domain().contains(sample_coarse)) {
            std::cout
                << "ERROR: Sample coarse coord NOT in coarse flux domain!\n";
        }

        // 4. Check fine children
        auto fine_children = map.fine_face_children(sample_coarse, dir);
        std::cout << "Fine children domain: " << fine_children << "\n";

        std::uint64_t valid_count   = 0;
        std::uint64_t invalid_count = 0;

        for (const auto& fine_coord : fine_children) {
            if (fine_flux.domain().contains(fine_coord)) {
                valid_count++;
            }
            else {
                invalid_count++;
                if (invalid_count == 1) {
                    std::cout << "ERROR: Fine coord " << fine_coord
                              << " NOT in fine flux domain!\n";
                }
            }
        }

        std::cout << "Valid fine children: " << valid_count << "\n";
        std::cout << "Invalid fine children: " << invalid_count << "\n";

        std::uint64_t expected = 1;
        for (std::uint64_t d = 0; d < Sim::dimensions; ++d) {
            if (d != dir) {
                expected *= map.ratio;
            }
        }

        if (valid_count != expected) {
            std::cout << "ERROR: Expected " << expected
                      << " fine children, got " << valid_count << "!\n";
        }
        else {
            std::cout << "✓ Correct number of fine children\n";
        }

        // 5. Check flux values
        auto coarse_sample_flux = coarse_flux(sample_coarse);
        std::cout << "Coarse flux at sample: " << coarse_sample_flux << "\n";

        if (coarse_sample_flux[0] == 0.0) {
            std::cout << "WARNING: Coarse flux has zero density!\n";
        }

        // 6. Sample one fine flux
        auto sample_fine = fine_children.start;
        if (fine_flux.domain().contains(sample_fine)) {
            auto fine_sample_flux = fine_flux(sample_fine);
            std::cout << "Fine flux at sample: " << fine_sample_flux << "\n";

            if (fine_sample_flux[0] == 0.0) {
                std::cout << "WARNING: Fine flux has zero density!\n";
            }
        }
    }

    template <typename Sim>
    void check_interface_flux_balance(Sim& sim)
    {
        std::cout << "\n=== Interface Flux Balance Check ===\n";

        for (std::uint64_t lvl = 1; lvl < sim.num_levels(); ++lvl) {
            auto map = create_level_mapping(sim.hierarchy(), lvl);

            for (std::uint64_t dir = 0; dir < Sim::dimensions; ++dir) {
                auto interfaces   = get_flux_interfaces(map, dir);
                auto& coarse_flux = sim.hydro(lvl - 1).flux[dir];
                auto& fine_flux   = sim.hydro(lvl).flux[dir];

                for (std::uint64_t iface = 0; iface < 2; ++iface) {
                    auto& interface = interfaces[iface];
                    if (!interface.is_valid) {
                        continue;
                    }

                    // Sum over entire interface
                    real total_coarse_flux     = 0.0;
                    real total_fine_flux       = 0.0;
                    std::uint64_t coarse_count = 0;
                    for (const auto& coord : interface.coarse_face) {
                        total_coarse_flux += coarse_flux(coord)[0];
                        coarse_count += 1;
                    }

                    std::uint64_t fine_count = 0;
                    for (const auto& coord : interface.fine_face) {
                        fine_count += 1;
                        if (fine_flux.domain().contains(coord)) {
                            total_fine_flux += fine_flux(coord)[0];
                        }
                    }

                    real expected_ratio =
                        std::pow(map.ratio, Sim::dimensions - 1);
                    real actual_ratio = total_fine_flux / total_coarse_flux;
                    real error = std::abs(actual_ratio - expected_ratio) /
                                 expected_ratio;

                    std::cout << "Level " << lvl << " Dir " << dir << " "
                              << (iface == 0 ? "Lower" : "Upper") << ":\n";
                    std::cout << "Coarse faces counted: " << coarse_count
                              << "\n";
                    std::cout << "Fine faces counted: " << fine_count << "\n";
                    std::cout << "Expected fine count: "
                              << coarse_count * expected_ratio << "\n";
                    std::cout << "  Expected ratio: " << expected_ratio << "\n";
                    std::cout << "  Actual ratio: " << actual_ratio << "\n";
                    std::cout << "  Error: " << error * 100 << "%\n";

                    if (error > 1e-10) {
                        std::cout << "  ⚠️  WARNING: Flux mismatch!\n";
                    }
                    else {
                        std::cout << "  ✓ Flux balance verified\n";
                    }
                }
            }
        }
    }

}   // namespace simbi::mesh::fmr

#endif
