#ifndef FMR_CONFIG_HPP
#define FMR_CONFIG_HPP

#include "compat.hpp"              // for real type
#include "containers/vector.hpp"   // for vector_t
#include "domain/algebra.hpp"      // for domain_algebra
#include "domain/domain.hpp"       // for domain_t

#include <cstddef>   // for std::size_t
#include <cstdint>   // for std::uint64_t, std::uint64_t
#include <vector>    // for std::vector

namespace simbi::mesh::refinement {

    template <std::uint64_t Dims>
    struct refinement_region_t {
        // region in the coarse grid's index space
        domain_t<Dims> domain;
        // refinement ratio for this region
        std::uint64_t ratio;
        // level this region belongs to (1-based, level 0 is base grid)
        std::uint64_t level_id;
    };

    template <std::uint64_t Dims>
    struct fmr_config_t {
        // maximum number of refinement levels
        std::uint64_t max_levels{1};

        // collection of refinement regions
        std::vector<refinement_region_t<Dims>> regions;

        // buffer zone size around refinement regions
        std::uint64_t buffer_size{1};

        bool conservative_interpolation{true};

        // validate configuration
        bool validate() const
        {
            // ensure regions don't overlap
            for (std::size_t ii = 0; ii < regions.size(); ++ii) {
                for (std::size_t jj = ii + 1; jj < regions.size(); ++jj) {
                    if (!domain_algebra::intersection(
                             regions[ii].domain,
                             regions[jj].domain
                        )
                             .empty()) {
                        return false;   // overlapping regions found
                    }
                }
            }

            // ensure all level ids are valid
            for (const auto& region : regions) {
                if (region.level_id >= max_levels) {
                    return false;
                }
            }

            return true;
        }

        // add a new refinement region
        void add_region(
            const domain_t<Dims>& domain,
            const std::uint64_t ratio,
            std::uint64_t level_id
        )
        {
            // add buffer zone around region
            // [TODO]: quick and dirty - improve later
            auto buffered_domain = domain_algebra::expand(
                domain,
                ones<Dims, std::int64_t>() *
                    static_cast<std::int64_t>(buffer_size)
            );

            regions.push_back({buffered_domain, ratio, level_id});
        }

        // get all regions for a specific level
        std::vector<refinement_region_t<Dims>>
        get_level_regions(std::uint64_t level_id) const
        {
            std::vector<refinement_region_t<Dims>> level_regions;

            for (const auto& region : regions) {
                if (region.level_id == level_id) {
                    level_regions.push_back(region);
                }
            }

            return level_regions;
        }
    };

    // helper to set up nested refinement
    template <std::uint64_t Dims>
    fmr_config_t<Dims> create_nested_config(
        const std::vector<physical_region_t<Dims>>& regions,
        const std::vector<std::uint64_t>& ratios,
        const vector_t<real, Dims>& bounds_min,
        const vector_t<real, Dims>& bounds_max,
        const iarray<Dims>& base_resolution,
        bool conservative_interpolation = true,
        std::uint64_t buffer_size       = 1
    )
    {
        fmr_config_t<Dims> config;
        config.max_levels                 = regions.size() + 1;
        config.buffer_size                = buffer_size;
        config.conservative_interpolation = conservative_interpolation;

        // add each refinement level
        for (std::uint64_t level = 0; level < regions.size(); ++level) {
            auto index_domain = to_index_space(
                regions[level],
                bounds_min,
                bounds_max,
                base_resolution
            );
            config.add_region(index_domain, ratios[level], level + 1);
        }

        return config;
    }

}   // namespace simbi::mesh::refinement

#endif   // FMR_CONFIG_HPP
