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
        T softening_length() const { return params_.softening_length; }
        bool two_way_coupling() const { return params_.two_way_coupling; }

        // param accessor
        const Params& params() const { return params_; }
        Params& params() { return params_; }

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
        T stiffness() const { return params_.stiffness; }
        T damping() const { return params_.damping; }
        T rest_length() const { return params_.rest_length; }

        // param accessor
        const Params& params() const { return params_; }
        Params& params() { return params_; }

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
        T viscosity() const { return params_.viscosity; }

        // param accessor
        const Params& params() const { return params_; }
        Params& params() { return params_; }

      private:
        Params params_;
    };

    // accretion
    template <typename T>
    class Accreting
    {
      public:
        struct Params {
            T accretion_efficiency = T(0.01);
            T accretion_radius     = T(1.0);
        };

        Accreting(const Params& params = {}) : params_(params) {}

        // capabilities of the accreting traits
        T accretion_efficiency() const { return params_.accretion_efficiency; }

        T accretion_radius() const { return params_.accretion_radius; }

        T total_accreted_mass() const { return total_accreted_mass_; }
        T total_accreted_momentum() const { return total_accreted_momentum_; }
        T total_accreted_energy() const { return total_accreted_energy_; }

        // param accessor
        const Params& params() const { return params_; }
        Params& params() { return params_; }

        // add accreted mass, momentum, and energy
        void add_accreted_mass(T mass) { total_accreted_mass_ += mass; }

        void add_accreted_momentum(T momentum)
        {
            total_accreted_momentum_ += momentum;
        }

        void add_accreted_energy(T energy) { total_accreted_energy_ += energy; }

        void add_accreted_angular_momentum(T angular_momentum)
        {
            total_accreted_angular_momentum_ += angular_momentum;
        }

      private:
        Params params_;
        T total_accreted_mass_             = T(0);
        T total_accreted_momentum_         = T(0);
        T total_accreted_energy_           = T(0);
        T total_accreted_angular_momentum_ = T(0);
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
        T density() const { return params_.density; }
        bool infinitely_rigid() const { return params_.infinitely_rigid; }
        T restitution_coefficient() const
        {
            return params_.restitution_coefficient;
        }

        // param accessor
        const Params& params() const { return params_; }
        Params& params() { return params_; }

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
        T youngs_modulus() const { return params_.youngs_modulus; }
        T& youngs_modulus() { return params_.youngs_modulus; }

        T poisson_ratio() const { return params_.poisson_ratio; }
        T& poisson_ratio() { return params_.poisson_ratio; }

        T yield_strength() const { return params_.yield_strength; }
        T& yield_strength() { return params_.yield_strength; }

        T failure_strain() const { return params_.failure_strain; }
        T& failure_strain() { return params_.failure_strain; }

        bool plastic_deformation() const { return params_.plastic_deformation; }
        bool& plastic_deformation() { return params_.plastic_deformation; }

        // Deformation state
        T max_deformation() const { return max_deformation_; }
        void update_deformation(T deformation)
        {
            max_deformation_ = std::max(max_deformation_, deformation);

            // Check if yield strength is exceeded
            if (params_.plastic_deformation &&
                max_deformation_ * params_.youngs_modulus >
                    params_.yield_strength) {
                permanently_deformed_ = true;
            }
        }

        bool is_permanently_deformed() const { return permanently_deformed_; }
        bool is_failed() const
        {
            return max_deformation_ > params_.failure_strain;
        }

        T stored_elastic_energy() const { return stored_elastic_energy_; }
        void set_stored_elastic_energy(T energy)
        {
            stored_elastic_energy_ = energy;
        }

        // Parameter access
        const Params& params() const { return params_; }
        Params& params() { return params_; }

      private:
        Params params_;
        T max_deformation_         = T(0);
        bool permanently_deformed_ = false;
        T stored_elastic_energy_   = T(0);
    };

}   // namespace simbi::ib::traits

#endif   // BODY_TRAITS_HPP
