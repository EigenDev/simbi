#ifndef GRID_AMR_FLUX_CORRECTION_HPP
#define GRID_AMR_FLUX_CORRECTION_HPP

#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "grid/amr/restriction.hpp"
#include "grid/connectivity.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "xpu/execution_space.hpp"
#include "xpu/executor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace simbi::grid::amr {

    using namespace simbi::compute;

    template <typename T, std::uint64_t Rank>
    class flux_register_t
    {
        using field_type = field_t<T, Rank>;

        // one register per face direction
        std::vector<std::unique_ptr<field_type>> registers_;
        domain_t<Rank>                           coarse_domain_;
        iarray<Rank>                             ratio_;

      public:
        flux_register_t(const domain_t<Rank>& domain, iarray<Rank> ratio)
            : coarse_domain_(domain), ratio_(ratio)
        {
            registers_.resize(2 * Rank);
        }

        // allocate buffer for a face (uses unified memory)
        void initialize_face(std::size_t dim, side_t side)
        {
            std::size_t idx = dim * 2 + static_cast<std::size_t>(side);
            if (registers_[idx]) {
                return;
            }

            // construct face domain (1 cell thick) inside coarse block
            domain_t<Rank> face = coarse_domain_;
            if (side == side_t::left) {
                face.fin[dim] = face.start[dim] + 1;
            }
            else {
                face.start[dim] = face.fin[dim] - 1;
            }

            registers_[idx] = std::make_unique<field_type>(face);
        }

        // access
        field_type* get_register(std::size_t dim, side_t side)
        {
            std::size_t idx = dim * 2 + static_cast<std::size_t>(side);
            return registers_[idx].get();
        }

        // zero all registers
        template <xpu::execution_space ExecutionSpace>
        void zero_all(xpu::executor_t<ExecutionSpace>& exec)
        {
            for (auto& reg : registers_) {
                if (reg) {
                    *reg = computation(*reg).map([] DUAL(const T&) { return T{}; }).with(exec);
                }
            }
        }

        // ---------------------------------------------------------------------
        // accumulation logic
        // ---------------------------------------------------------------------

        // coarse: R += -F * dt * area
        template <xpu::execution_space ExecutionSpace, typename Geometry>
        void accumulate_coarse(
            xpu::executor_t<ExecutionSpace>& exec,
            const field_type&                coarse_flux,
            const Geometry&                  geometry,
            std::size_t                      dim,
            side_t                           side,
            real                             dt
        )
        {
            auto* reg = get_register(dim, side);
            if (!reg) {
                return;
            }

            // we read from the coarse flux at the register's domain
            auto flux_slice = coarse_flux[reg->domain()];

            // perform additive update: reg = reg + (-flux * dt * area)
            auto current_val = computation(*reg);
            auto flux_comp   = computation(flux_slice);

            auto update = current_val.enum_map([dt, dim, side, geometry, flux_comp] DUAL(
                                                   const iarray<Rank>& coord,
                                                   const T&            curr
                                               ) {
                auto f = flux_comp(coord);
                // compute face area at the interface
                real area;
                if (side == side_t::left) {
                    area = geometry.face_area(coord, dim);
                }
                else {
                    auto rc = coord + unit_vectors::array_offset<Rank>(dim);
                    area    = geometry.face_area(rc, dim);
                }
                return curr + f * (-dt * area);
            });

            *reg = update.with(exec);
        }

        // fine: R += average(F * area) * dt
        template <xpu::execution_space ExecutionSpace, typename Geometry>
        void accumulate_fine(
            xpu::executor_t<ExecutionSpace>& exec,
            const field_type&                fine_flux,
            const Geometry&                  geometry,
            std::size_t                      dim,
            side_t                           side,
            real                             dt
        )
        {
            auto* reg = get_register(dim, side);
            if (!reg) {
                return;
            }

            // multiply fine flux by fine face area before restricting
            // this creates F * A at fine resolution
            auto fine_flux_area = computation(fine_flux).enum_map(
                [dim, side, geometry] DUAL(const iarray<Rank>& fine_coord, const T& f) -> T {
                    // fine_coord is in fine index space
                    // compute face area at fine resolution
                    real area;
                    if (side == side_t::left) {
                        area = geometry.face_area(fine_coord, dim);
                    }
                    else {
                        auto rc = fine_coord + unit_vectors::array_offset<Rank>(dim);
                        area    = geometry.face_area(rc, dim);
                    }
                    return f * area;
                }
            );

            // now restrict F*A to coarse domain (conservative averaging)
            auto restricted_flux_area = restrict(fine_flux_area, ratio_);

            // intersect with register
            auto face_flux_area = restricted_flux_area.at(reg->domain());

            // update: current + average(F*A) * dt
            auto current_val = computation(*reg);

            auto update =
                current_val.zip(face_flux_area, [dt] DUAL(const T& curr, const T& fa_avg) {
                    return curr + fa_avg * dt;
                });

            *reg = update.with(exec);
        }
    };

} // namespace simbi::grid::amr

#endif // GRID_AMR_FLUX_CORRECTION_HPP
