#ifndef BODY_TRAITS_HPP
#define BODY_TRAITS_HPP

#include "build_options.hpp"

namespace simbi::ib::traits {
    template <typename T>
    class Gravitational
    {
      public:
        struct Params {
            T softening_length    = T(0.01);   // Softening length
            bool two_way_coupling = false;     // Apply reaction forces to body?
        };

        Gravitational(const Params& params = {}) : params_(params) {}

        // capabilities of the gravitational traits
        DUAL T softening_length() const { return params_.softening_length; }
        DUAL bool two_way_coupling() const { return params_.two_way_coupling; }

        // param accessor
        DUAL const Params& params() const { return params_; }
        DUAL Params& params() { return params_; }

      private:
        Params params_;
    };

    // elastic
    template <typename T>
    class Elastic
    {
      public:
        struct Params {
            T stiffness   = T(1000);
            T damping     = T(0.3);
            T rest_length = T(0.0);
        };

        Elastic(const Params& params = {}) : params_(params) {}

        // capabilities of the elastic traits
        DUAL T stiffness() const { return params_.stiffness; }
        DUAL T damping() const { return params_.damping; }
        DUAL T rest_length() const { return params_.rest_length; }

        // param accessor
        DUAL const Params& params() const { return params_; }
        DUAL Params& params() { return params_; }

      private:
        Params params_;
    };

    // viscous
    template <typename T>
    class Viscous
    {
      public:
        struct Params {
            T viscosity = T(0.01);
        };

        Viscous(const Params& params = {}) : params_(params) {}

        // capabilities of the viscous traits
        DUAL T viscosity() const { return params_.viscosity; }

        // param accessor
        DUAL const Params& params() const { return params_; }
        DUAL Params& params() { return params_; }

      private:
        Params params_;
    };

    // accretion
    template <typename T>
    class Accreting
    {
      public:
        struct Params {
            T accretion_efficiency    = T(0.01);
            T accretion_radius_factor = T(1.0);
        };

        Accreting(const Params& params = {}) : params_(params) {}

        // capabilities of the accreting traits
        DUAL T accretion_efficiency() const
        {
            return params_.accretion_efficiency;
        }

        DUAL T accretion_radius_factor() const
        {
            return params_.accretion_radius_factor;
        }

        DUAL T total_accreted_mass() const { return total_accreted_mass_; }
        DUAL T total_accreted_momentum() const
        {
            return total_accreted_momentum_;
        }
        DUAL T total_accreted_energy() const { return total_accreted_energy_; }

        // param accessor
        DUAL const Params& params() const { return params_; }
        DUAL Params& params() { return params_; }

        // add accreted mass, momentum, and energy
        DUAL void add_accreted_mass(T mass) { total_accreted_mass_ += mass; }

        DUAL void add_accreted_momentum(T momentum)
        {
            total_accreted_momentum_ += momentum;
        }

        DUAL void add_accreted_energy(T energy)
        {
            total_accreted_energy_ += energy;
        }

      private:
        Params params_;
        T total_accreted_mass_     = T(0);
        T total_accreted_momentum_ = T(0);
        T total_accreted_energy_   = T(0);
    };

    // rigid
    template <typename T>
    class Rigid
    {
      public:
        struct Params {
            T density                 = T(1.0);
            bool infinitely_rigid     = true;
            T restitution_coefficient = T(0.8);
        };

        Rigid(const Params& params = {}) : params_(params) {}

        // accessors
        DUAL T density() const { return params_.density; }
        DUAL bool infinitely_rigid() const { return params_.infinitely_rigid; }
        DUAL T restitution_coefficient() const
        {
            return params_.restitution_coefficient;
        }

        // param accessor
        DUAL const Params& params() const { return params_; }
        DUAL Params& params() { return params_; }

      private:
        Params params_;
    };

    // deformable material
    template <typename T>
    class Deformable
    {
      public:
        struct Params {
            T density                = T(1.0);
            T youngs_modulus         = T(1000);
            T poisson_ratio          = T(0.3);
            T yield_strength         = T(0.1);
            T failure_strain         = T(0.1);
            bool plastic_deformation = false;   // allow permanent deformation?
        };

        Deformable(const Params& params = {}) : params_(params) {}

        // accessors
        DUAL T youngs_modulus() const { return params_.youngs_modulus; }
        DUAL T& youngs_modulus() { return params_.youngs_modulus; }

        DUAL T poisson_ratio() const { return params_.poisson_ratio; }
        DUAL T& poisson_ratio() { return params_.poisson_ratio; }

        DUAL T yield_strength() const { return params_.yield_strength; }
        DUAL T& yield_strength() { return params_.yield_strength; }

        DUAL T failure_strain() const { return params_.failure_strain; }
        DUAL T& failure_strain() { return params_.failure_strain; }

        DUAL bool plastic_deformation() const
        {
            return params_.plastic_deformation;
        }
        DUAL bool& plastic_deformation() { return params_.plastic_deformation; }

        // Deformation state
        DUAL T max_deformation() const { return max_deformation_; }
        DUAL void update_deformation(T deformation)
        {
            max_deformation_ = std::max(max_deformation_, deformation);

            // Check if yield strength is exceeded
            if (params_.plastic_deformation &&
                max_deformation_ * params_.youngs_modulus >
                    params_.yield_strength) {
                permanently_deformed_ = true;
            }
        }

        DUAL bool is_permanently_deformed() const
        {
            return permanently_deformed_;
        }
        DUAL bool is_failed() const
        {
            return max_deformation_ > params_.failure_strain;
        }

        DUAL T stored_elastic_energy() const { return stored_elastic_energy_; }
        DUAL void set_stored_elastic_energy(T energy)
        {
            stored_elastic_energy_ = energy;
        }

        // Parameter access
        DUAL const Params& params() const { return params_; }
        DUAL Params& params() { return params_; }

      private:
        Params params_;
        T max_deformation_         = T(0);
        bool permanently_deformed_ = false;
        T stored_elastic_energy_   = T(0);
    };

}   // namespace simbi::ib::traits

#endif   // BODY_TRAITS_HPP
