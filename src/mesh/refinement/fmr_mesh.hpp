#ifndef FMR_MESH_HPP
#define FMR_MESH_HPP

#include "compat.hpp"                       // for real type
#include "containers/vector.hpp"            // for vector_t, iarray, etc
#include "domain/domain.hpp"                // for domain_t
#include "mesh/mesh_config.hpp"             // for mesh_config_t
#include "mesh/refinement/fmr_config.hpp"   // for fmr_config_t
#include "mesh/refinement/level.hpp"        // for level_t
#include "utility/enums.hpp"                // for Geometry

#include <cstddef>     // for std::size_t
#include <cstdint>     // for std::uint32_t, std::uint64_t
#include <stdexcept>   // for std::runtime_error, std::out_of_range
#include <vector>      // for std::vector

namespace simbi::mesh::refinement {

    template <std::uint64_t Dims, Geometry G>
    class fmr_mesh_t
    {
      public:
        // construct from base mesh and fmr configuration
        fmr_mesh_t(
            const mesh_config_t<Dims, G>& base_mesh,
            const fmr_config_t<Dims>& config
        )
            : base_mesh_(base_mesh), config_(config)
        {
            if (!config_.validate()) {
                throw std::runtime_error("invalid fmr configuration");
            }

            // initialize refinement levels
            initialize_levels();
        }

        // accessors
        const mesh_config_t<Dims, G>& base_mesh() const { return base_mesh_; }

        const level_t<Dims, G>& level(std::uint32_t level_id) const
        {
            if (level_id >= levels_.size()) {
                throw std::out_of_range("invalid refinement level");
            }
            return levels_[level_id];
        }

        std::uint32_t num_levels() const { return levels_.size(); }

        // check if a point needs refinement
        bool needs_refinement(
            const iarray<Dims>& coord,
            std::uint32_t current_level
        ) const
        {
            // check if point is in any refinement region of the next level
            for (const auto& region :
                 config_.get_level_regions(current_level + 1)) {
                if (region.domain.contains(coord)) {
                    return true;
                }
            }
            return false;
        }

        // get effective resolution at a point
        vector_t<real, Dims> get_dx(std::uint32_t level_id) const
        {
            if (level_id >= levels_.size()) {
                return base_mesh_.dx;
            }
            return levels_[level_id].dx;
        }

        // check if coordinate is on level boundary
        bool is_level_boundary(
            const iarray<Dims>& coord,
            std::uint32_t level_id
        ) const
        {
            if (level_id >= levels_.size()) {
                return false;
            }

            const auto& level = levels_[level_id];

            // check each direction
            for (std::uint32_t dd = 0; dd < Dims; ++dd) {
                // get neighboring coordinates
                auto left  = coord;
                auto right = coord;
                left[dd] -= 1;
                right[dd] += 1;

                // check if either neighbor is outside refinement region
                if (!level.parent_domain.contains(left) ||
                    !level.parent_domain.contains(right)) {
                    return true;
                }
            }
            return false;
        }

        void print_info() const
        {
            std::cout << "\nFMR Mesh Configuration:" << std::endl;
            std::cout << "=======================" << std::endl;
            std::cout << "Number of levels: " << config_.max_levels
                      << std::endl;
            std::cout << "Buffer size: " << config_.buffer_size << std::endl;
            std::cout << "Conservative interpolation: "
                      << (config_.conservative_interpolation ? "Yes" : "No")
                      << std::endl;

            // print info for each level
            for (std::uint64_t level = 0; level < config_.max_levels; ++level) {
                std::cout << "\nLevel " << level << ":" << std::endl;
                std::cout << "-------------" << std::endl;

                if (level > 0) {
                    auto level_regions = config_.get_level_regions(level);

                    for (const auto& region : level_regions) {
                        std::cout << "Refinement ratio: " << region.ratio << "x"
                                  << std::endl;
                        std::cout << "Index space: " << region.domain
                                  << std::endl;

                        // convert to physical space
                        auto phys_region = to_physical_space(
                            region.domain,
                            base_mesh_.bounds_min,
                            base_mesh_.bounds_max,
                            base_mesh_.shape
                        );

                        std::cout << "Physical space: [";
                        for (std::uint64_t d = 0; d < Dims; ++d) {
                            std::cout << phys_region.min[d] << ", "
                                      << phys_region.max[d];
                            if (d < Dims - 1) {
                                std::cout << "] × [";
                            }
                        }
                        std::cout << "]" << std::endl;
                    }
                }

                // print cell sizes
                auto dx = get_dx(level);
                std::cout << "Cell sizes: ";
                for (std::uint64_t d = 0; d < Dims; ++d) {
                    std::cout << dx[d];
                    if (d < Dims - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << std::endl;
            }
            std::cout << "=======================" << std::endl;
        }

      private:
        mesh_config_t<Dims, G> base_mesh_;
        fmr_config_t<Dims> config_;
        std::vector<level_t<Dims, G>> levels_;

        void initialize_levels()
        {
            // start with base level
            levels_.clear();
            levels_.reserve(config_.max_levels);

            // level 0 is the base mesh
            levels_.push_back(
                level_t<Dims, G>{
                  .level_id      = 0,
                  .mesh          = base_mesh_,
                  .parent_domain = base_mesh_.domain,
                  .ref_ratio     = 1,
                  .dx            = base_mesh_.dx
                }
            );

            // create remaining levels
            for (std::uint32_t ii = 1; ii < config_.max_levels; ++ii) {
                auto regions = config_.get_level_regions(ii);
                if (regions.empty()) {
                    continue;
                }

                // create level from parent
                const auto& parent = levels_[ii - 1];

                // merge all regions at this level
                domain_t<Dims> merged_domain = regions[0].domain;
                std::uint64_t ref_ratio      = regions[0].ratio;

                for (std::size_t jj = 1; jj < regions.size(); ++jj) {
                    merged_domain = domain_algebra::union_of(
                        merged_domain,
                        regions[jj].domain
                    );
                    // ensure consistent refinement ratios
                    if (regions[jj].ratio != ref_ratio) {
                        throw std::runtime_error(
                            "inconsistent refinement ratios at same level"
                        );
                    }
                }

                // create new level
                levels_.push_back(
                    level_t<Dims, G>::create(
                        ii,
                        parent.mesh,
                        merged_domain,
                        ref_ratio
                    )
                );
            }
        }
    };

}   // namespace simbi::mesh::refinement

#endif   // FMR_MESH_HPP
