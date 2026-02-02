// =============================================================================
// flux_correction.hpp
//
// amr flux correction (refluxing) implementation.
// defines the `flux_register_t` class, which accumulates flux differences at
// coarse-fine boundaries. also provides the `coarse_flux_accumulate_t` and
// `fine_flux_area_t` functors used to perform the refluxing operation,
// ensuring conservation across AMR levels.
//
// the reflux operation corrects conservation at coarse-fine interfaces:
//   1. coarse flux is subtracted: R -= F_coarse * dt * A_coarse
//   2. fine fluxes are summed:    R += sum(F_fine * A_fine) * dt
//   3. correction applied:        U_coarse += R / V_coarse
//
// note: fine flux uses summation (not averaging) because we need the total
// flux through all fine faces that comprise one coarse face. this is
// ratio^(Rank-1) fine faces, not ratio^Rank cells.
//
// usage:
//   flux_register.accumulate_coarse(...);
//   flux_register.accumulate_fine(...);
//   apply_flux_correction(coarse_field, flux_register, ...);
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"
#include "functional/fp.hpp"
#include "grid/connectivity.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "xpu/xpu.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace simbi::grid::amr {

    using namespace simbi::compute;

    // -------------------------------------------------------------------------
    // face restriction kernel (summation, not averaging)
    //
    // for flux correction, we need to sum fine face fluxes, not average them.
    // each coarse face is covered by ratio^(Rank-1) fine faces. the total flux
    // through the coarse face equals the sum of fluxes through all fine faces.
    // -------------------------------------------------------------------------
    template <typename FineComp, std::uint64_t Rank>
    struct restrict_face_sum_t
    {
        using value_type = std::remove_cv_t<std::remove_reference_t<typename FineComp::value_type>>;
        using argument_type                 = iarray<Rank>;
        static constexpr std::uint64_t rank = Rank;

        FineComp      fine_comp_;
        argument_type ratio_;
        std::size_t   face_dim_;

        DUAL restrict_face_sum_t(FineComp fine, argument_type ratio, std::size_t face_dim)
            : fine_comp_(std::move(fine)), ratio_(ratio), face_dim_(face_dim)
        {
        }

        DUAL value_type operator()(const argument_type& coarse_coord) const
        {
            using namespace structs;

            // compute fine base coordinate
            argument_type fine_base;
            for (std::uint64_t d = 0; d < Rank; ++d) {
                fine_base[d] = coarse_coord[d] * ratio_[d];
            }

            value_type sum{};
            bool       first = true;

            // sum over fine faces: loop over all dimensions except face_dim
            // for a face normal to dimension d, we have ratio^(Rank-1) fine faces
            if constexpr (Rank == 1) {
                // 1d: one fine face per coarse face (no transverse directions)
                sum = fine_comp_(fine_base);
            }
            else if constexpr (Rank == 2) {
                // 2d: ratio fine faces per coarse face
                std::size_t trans_dim = (face_dim_ == 0) ? 1 : 0;
                for (std::int64_t ii = 0; ii < ratio_[trans_dim]; ++ii) {
                    argument_type fc = fine_base;
                    fc[trans_dim] += ii;
                    if (first) {
                        sum   = fine_comp_(fc);
                        first = false;
                    }
                    else {
                        if constexpr (is_hydro_conserved_c<value_type>) {
                            sum = sum | add_gas(fine_comp_(fc));
                        }
                        else {
                            sum = sum + fine_comp_(fc);
                        }
                    }
                }
            }
            else if constexpr (Rank == 3) {
                // 3d: ratio^2 fine faces per coarse face
                std::size_t t0, t1;
                if (face_dim_ == 0) {
                    t0 = 1;
                    t1 = 2;
                }
                else if (face_dim_ == 1) {
                    t0 = 0;
                    t1 = 2;
                }
                else {
                    t0 = 0;
                    t1 = 1;
                }

                for (std::int64_t jj = 0; jj < ratio_[t1]; ++jj) {
                    for (std::int64_t ii = 0; ii < ratio_[t0]; ++ii) {
                        argument_type fc = fine_base;
                        fc[t0] += ii;
                        fc[t1] += jj;
                        if (first) {
                            sum   = fine_comp_(fc);
                            first = false;
                        }
                        else {
                            if constexpr (is_hydro_conserved_c<value_type>) {
                                sum = sum | add_gas(fine_comp_(fc));
                            }
                            else {
                                sum = sum + fine_comp_(fc);
                            }
                        }
                    }
                }
            }

            return sum;
        }
    };

    // helper to scale face domain down (for face restriction)
    template <std::uint64_t Rank>
    constexpr grid::domain_t<Rank>
    scale_face_domain_down(const grid::domain_t<Rank>& d, const iarray<Rank>& ratio)
    {
        auto start = d.start;
        auto fin   = d.fin;
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            start[ii] /= ratio[ii];
            fin[ii] /= ratio[ii];
            // handle rounding for non-aligned boundaries
            if (d.fin[ii] % ratio[ii] != 0) {
                fin[ii] += 1;
            }
        }
        return {start, fin};
    }

    // factory for face restriction (summation)
    template <typename Computation, std::uint64_t Rank>
    auto restrict_face(const Computation& fine, iarray<Rank> ratio, std::size_t face_dim)
    {
        auto coarse_domain = scale_face_domain_down(fine.domain(), ratio);
        auto kernel        = restrict_face_sum_t<Computation, Rank>(fine, ratio, face_dim);
        return computation_t<Rank, decltype(kernel)>{std::move(kernel), coarse_domain};
    }

    // -------------------------------------------------------------------------
    // coarse flux accumulation functor
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank, typename Geometry, typename FluxComp>
    struct coarse_flux_accumulate_t
    {
        real        dt;
        std::size_t dim;
        Geometry    geometry;
        FluxComp    flux_comp;

        DEV T operator()(const iarray<Rank>& coord, const T& curr) const
        {
            auto f    = flux_comp(coord);
            real area = geometry.labframe_face_area(coord, dim);
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
        Geometry    geometry;

        DEV T operator()(const iarray<Rank>& fine_coord, const T& f) const
        {
            real area = geometry.labframe_face_area(fine_coord, dim);
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
            for (std::size_t dd = 0; dd < Rank; ++dd) {
                initialize_face(dd, side_t::left);
                initialize_face(dd, side_t::right);
            }
        }

        // allocate buffer for a face (uses unified memory)
        void initialize_face(std::size_t dim, side_t side)
        {
            std::size_t idx = dim * 2 + static_cast<std::size_t>(side);
            if (registers_[idx]) {
                return;
            }

            // construct face domain (1 cell thick) at the coarse-fine interface.
            // flux[i] stores the flux at the left face of cell i, so:
            //   left boundary  -> face at index start (flux at left face of first cell)
            //   right boundary -> face at index fin   (flux at right face of last cell)
            domain_t<Rank> face = coarse_domain_;
            if (side == side_t::left) {
                face.fin[dim] = face.start[dim] + 1;
            }
            else {
                face.start[dim] = face.fin[dim];
                face.fin[dim]   = face.start[dim] + 1;
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
        template <xpu::execution_space_c ExecutionSpace>
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
        template <xpu::execution_space_c ExecutionSpace, typename Geometry>
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
                accumulate_op{dt, dim, geometry, flux_comp};

            auto update = current_val.enum_map(accumulate_op);

            *reg = update.with(exec);
        }

        // fine: R += sum(F * area) * dt
        // note: uses summation (not averaging) because we need total flux
        // through all fine faces that cover one coarse face
        template <xpu::execution_space_c ExecutionSpace, typename Geometry>
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
            fine_flux_area_t<T, Rank, Geometry> flux_area_op{dim, geometry};

            auto fine_flux_area = fine_flux.enum_map(flux_area_op);

            // sum F*A over fine faces (not average!)
            // each coarse face is covered by ratio^(Rank-1) fine faces
            auto summed_flux_area = restrict_face(fine_flux_area, ratio_, dim);

            // intersect with register domain
            auto face_flux_area = summed_flux_area.at(reg->domain());

            // update: current + sum(F*A) * dt
            auto current_val = reg->as_computation();

            auto scaled_flux = face_flux_area.map(fp::scalar_multiply(dt));
            auto update      = current_val.zip(scaled_flux, fp::add_op);

            *reg = update.with(exec);
        }
    };

} // namespace simbi::grid::amr
