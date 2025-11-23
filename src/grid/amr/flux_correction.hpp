#ifndef GRID_AMR_FLUX_CORRECTION_HPP
#define GRID_AMR_FLUX_CORRECTION_HPP

#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "grid/amr/restriction.hpp"
#include "grid/connectivity.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/executor.hpp"

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
        domain_t<Rank> coarse_domain_;
        iarray<Rank> ratio_;

      public:
        flux_register_t(const domain_t<Rank>& domain, iarray<Rank> ratio)
            : coarse_domain_(domain), ratio_(ratio)
        {
            registers_.resize(2 * Rank);
        }

        // allocate buffer for a face
        void initialize_face(std::size_t dim, side_t side, het::locality_t loc)
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

            registers_[idx] = std::make_unique<field_type>(face, loc);
        }

        // access
        field_type* get_register(std::size_t dim, side_t side)
        {
            std::size_t idx = dim * 2 + static_cast<std::size_t>(side);
            return registers_[idx].get();
        }

        // ---------------------------------------------------------------------
        // accumulation logic
        // ---------------------------------------------------------------------

        // coarse: R = -F * dt
        void accumulate_coarse(
            het::exec::executor_t& exec,
            const field_type& coarse_flux,
            std::size_t dim,
            side_t side,
            real dt
        )
        {
            auto* reg = get_register(dim, side);
            if (!reg) {
                return;
            }

            // we read from the coarse flux at the register's domain
            // computation: r = -flux * dt
            auto flux_slice = coarse_flux[reg->domain()];

            // assignment triggers kernel
            *reg = computation(flux_slice)
                       .map([dt] DUAL(const T& f) { return f * (-dt); })
                       .with(exec);
        }

        // fine: R += average(F) * dt
        void accumulate_fine(
            het::exec::executor_t& exec,
            const field_type& fine_flux,
            std::size_t dim,
            side_t side,
            real dt
        )
        {
            auto* reg = get_register(dim, side);
            if (!reg) {
                return;
            }

            // restrict fine flux to coarse domain
            // this returns a computation defined on the coarse equivalent of
            // the fine domain
            auto restricted_flux = restrict(computation(fine_flux), ratio_);

            // intersect with register
            // the fine flux might cover the whole patch, but we only want the
            // face
            auto face_flux = restricted_flux.at(reg->domain());

            // update logic: current + new * dt
            // zip the current register value with the incoming flux
            auto current_val = computation(*reg);

            auto update = current_val.zip(
                face_flux,
                [dt] DUAL(const T& curr, const T& flux_avg) {
                    return curr + flux_avg * dt;
                }
            );

            *reg = update.with(exec);
        }
    };

}   // namespace simbi::grid::amr

#endif   // GRID_AMR_FLUX_CORRECTION_HPP
