#ifndef GRID_AMR_FLUX_CORRECTION_HPP
#define GRID_AMR_FLUX_CORRECTION_HPP

#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "grid/amr/restriction.hpp"
#include "grid/connectivity.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "xpu/execution/execution_space.hpp"
#include "xpu/execution/executor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace simbi::grid::amr {

    using namespace simbi::compute;

    // -------------------------------------------------------------------------
    // coarse flux accumulation functor
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank, typename Geometry, typename FluxComp>
    struct coarse_flux_accumulate_t
    {
        real        dt;
        std::size_t dim;
        side_t      side;
        Geometry    geometry;
        FluxComp    flux_comp;

        DEV T operator()(const iarray<Rank>& coord, const T& curr) const
        {
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
        }
    };

    // -------------------------------------------------------------------------
    // fine flux area functor
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank, typename Geometry>
    struct fine_flux_area_t
    {
        std::size_t dim;
        side_t      side;
        Geometry    geometry;

        DEV T operator()(const iarray<Rank>& fine_coord, const T& f) const
        {
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
    };

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
                    *reg = reg->as_computation().map(fp::zero_func).with(exec);
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
            auto current_val = reg->as_computation();
            auto flux_comp   = flux_slice.as_computation();

            coarse_flux_accumulate_t<T, Rank, Geometry, decltype(flux_comp)>
                accumulate_op{dt, dim, side, geometry, flux_comp};

            auto update = current_val.enum_map(accumulate_op);

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
            fine_flux_area_t<T, Rank, Geometry> flux_area_op{dim, side, geometry};

            auto fine_flux_area = fine_flux.enum_map(flux_area_op);

            // now restrict F*A to coarse domain (conservative averaging)
            auto restricted_flux_area = restrict(fine_flux_area, ratio_);

            // intersect with register
            auto face_flux_area = restricted_flux_area.at(reg->domain());

            // update: current + average(F*A) * dt
            auto current_val = reg->as_computation();

            auto scaled_flux = face_flux_area.map(fp::scalar_multiply(dt));
            auto update      = current_val.zip(scaled_flux, fp::add_op);

            *reg = update.with(exec);
        }
    };

} // namespace simbi::grid::amr

#endif // GRID_AMR_FLUX_CORRECTION_HPP
