#include "state.hpp"
#include "srhd.hpp"
#include "newt.hpp"
namespace simbi
{
    namespace hydrostate
    {
        template <HydroRegime regime, int dim> 
        struct return_type_chooser;

        template<>
        struct return_type_chooser<HydroRegime::NEWTONIAN, 1> {
            using type = Newtonian<1>;
        };
        template<>
        struct return_type_chooser<HydroRegime::NEWTONIAN, 2> {
            using type = Newtonian<2>;
        };
        template<>
        struct return_type_chooser<HydroRegime::NEWTONIAN, 3> {
            using type = Newtonian<3>;
        };

        template<>
        struct return_type_chooser<HydroRegime::RELATIVISTC, 1> {
            using type = SRHD<1>;
        };
        template<>
        struct return_type_chooser<HydroRegime::RELATIVISTC, 2> {
            using type = SRHD<2>;
        };
        template<>
        struct return_type_chooser<HydroRegime::RELATIVISTC, 3> {
            using type = SRHD<3>;
        };


        // template<HydroRegime regime, int dim>
        // std::unique_ptr<HydroBase> create(
        //     const std::vector<std::vector<real>> &state,
        //     const InitialConditions &init_cond,
        //     const std::string &regime,
        //     const int dim)
        // {
        //     if (regime == "relativistic") {
        //         if (dim == 1) {
        //             return std::make_unique<SRHD<1>>(state, init_cond);
        //         } else if (dim == 2) {
        //             return std::make_unique<SRHD<2>>(state, init_cond);
        //         } else {
        //             return std::make_unique<SRHD<3>>(state, init_cond);
        //         }
        //     } else {
        //         if (dim == 1) {
        //             return std::make_unique<Newtonian<1>>(state, init_cond);
        //         } else if (dim == 2) {
        //             return std::make_unique<Newtonian<2>>(state, init_cond);
        //         } else {
        //             return std::make_unique<Newtonian<3>>(state, init_cond);
        //         }
        //     }
        // };

        template<HydroRegime regime, int dim>
        std::unique_ptr<return> create(
            const std::vector<std::vector<real>> &state,
            const InitialConditions &init_cond,
            const std::string &regime,
            const int dim)
        {
            if (regime == "relativistic") {
                if (dim == 1) {
                    return std::make_unique<SRHD<1>>(state, init_cond);
                } else if (dim == 2) {
                    return std::make_unique<SRHD<2>>(state, init_cond);
                } else {
                    return std::make_unique<SRHD<3>>(state, init_cond);
                }
            } else {
                if (dim == 1) {
                    return std::make_unique<Newtonian<1>>(state, init_cond);
                } else if (dim == 2) {
                    return std::make_unique<Newtonian<2>>(state, init_cond);
                } else {
                    return std::make_unique<Newtonian<3>>(state, init_cond);
                }
            }
        // };

        // template<>
        // std::unique_ptr<Newtonian<1>>  create<HydroRegime::NEWTONIAN, 1>(
        //     std::vector<std::vector<real>> &state,
        //     const InitialConditions &init_cond
        // );
        // template<>
        // std::unique_ptr<Newtonian<2>>  create<HydroRegime::NEWTONIAN, 2>(
        //     std::vector<std::vector<real>> &state,
        //     const InitialConditions &init_cond
        // );
        // template<>
        // std::unique_ptr<Newtonian<3>>  create<HydroRegime::NEWTONIAN, 3>(
        //     std::vector<std::vector<real>> &state,
        //     const InitialConditions &init_cond
        // );

        // template<>
        // std::unique_ptr<SRHD<1>>  create<HydroRegime::RELATIVISTC, 1>(
        //     std::vector<std::vector<real>> &state,
        //     const InitialConditions &init_cond
        // );
        // template<>
        // std::unique_ptr<SRHD<2>>  create<HydroRegime::RELATIVISTC, 2>(
        //     std::vector<std::vector<real>> &state,
        //     const InitialConditions &init_cond
        // );
        // template<>
        // std::unique_ptr<SRHD<3>>  create<HydroRegime::RELATIVISTC, 3>(
        //     std::vector<std::vector<real>> &state,
        //     const InitialConditions &init_cond
        // );
        
    } // namespace hydrostate
    
} // namespace simbi
