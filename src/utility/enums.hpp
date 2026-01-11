/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            enums.hpp
 *  * @brief           useful enums for whole codebase
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef ENUMS_HPP
#define ENUMS_HPP

#include "bimap.hpp"

#include <cstdint>

namespace simbi {
    enum class cellspacing_t {
        LINEAR,
        LOG,
    };

    enum class regime_t {
        NEWTONIAN,
        SRHD,
        RMHD,
        MHD,
    };

    enum class geometry_t {
        CARTESIAN,
        SPHERICAL,
        CYLINDRICAL,
        AXIS_CYLINDRICAL,
        PLANAR_CYLINDRICAL,
    };

    enum class accuracy_t {
        FIRST_ORDER,
        SECOND_ORDER,
    };

    enum class reconstruction_t {
        PCM,
        PLM,
    };

    enum class timestepping_t {
        EULER,
        RK2,
    };

    enum class solver_t {
        HLLD,
        HLLC,
        HLLE,
    };

    enum class wave_speed_estimate_t {
        SCHNEIDER_ET_AL_93,
        MIGNONE_AND_BODO_05,
        HUBER_AND_KISSMANN_2021,
        DAVIDSON,
    };

    enum class dir_t {
        N,
        E,
        S,
        W,
        SW,
        SE,
        NW,
        NE
    };

    enum class corner_t {
        NE,
        SE,
        SW,
        NW
    };

    enum class face_t {
        N,
        E,
        S,
        W
    };

    enum class plane_t {
        IJ,
        IK,
        JK
    };

    enum interface_t {
        LF,
        RF
    };

    enum class ct_algo_t {
        ZERO,
        CONTACT,
        ALPHA,
        MdZ,
    };

    enum class LIMITER {
        MINMOD,
        VAN_LEER
    };

    enum class shockwave_limiter_t {
        NONE,
        FLEISCHMANN,
        QUIRK,
    };

    enum class color_t {
        DEFAULT,
        BLACK,
        BLUE,
        LIGHT_GREY,
        DARK_GREY,
        LIGHT_RED,
        LIGHT_GREEN,
        LIGHT_YELLOW,
        LIGHT_BLUE,
        LIGHT_MAGENTA,
        LIGHT_CYAN,
        WHITE,
        RED,
        GREEN,
        YELLOW,
        CYAN,
        MAGENTA,
        BOLD,
        RESET,
    };

    enum class body_capability_t : uint32_t {
        NONE          = 0,
        GRAVITATIONAL = 1 << 0,
        ACCRETION     = 1 << 1,
        ELASTIC       = 1 << 2,
        DEFORMABLE    = 1 << 3,
        RIGID         = 1 << 4,
    };

    enum class subcycling_mode_t : uint32_t {
        NONE,
        STANDARD,
        MANUAL,
        ADAPTIVE,
    };

    constexpr inline body_capability_t operator|(body_capability_t lhs, body_capability_t rhs)
    {
        return static_cast<body_capability_t>(
            static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs)
        );
    }

    constexpr inline body_capability_t& operator|=(body_capability_t& lhs, body_capability_t rhs)
    {
        lhs = lhs | rhs;
        return lhs;
    }

    // component identifiers for magnetic field directions
    enum class magnetic_comp_t : std::uint64_t {
        I = 2, // B1 component
        J = 1, // B2 component
        K = 0  // B3 component
    };

    constexpr auto comp_wave_speed    = wave_speed_estimate_t::MIGNONE_AND_BODO_05;
    constexpr auto comp_ct_type       = ct_algo_t ::CONTACT;
    constexpr auto comp_slope_limiter = LIMITER::MINMOD;

    // register BiMaps for enum serialization and deserialization
    REGISTER_ENUM_BIMAP(
        timestepping_t,
        {timestepping_t::EULER, "rk1"},
        {timestepping_t::RK2, "rk2"}
    );

    REGISTER_ENUM_BIMAP(
        cellspacing_t,
        {cellspacing_t::LINEAR, "linear"},
        {cellspacing_t::LOG, "log"}
    );

    REGISTER_ENUM_BIMAP(
        regime_t,
        {regime_t::NEWTONIAN, "newtonian"},
        {regime_t::SRHD, "srhd"},
        {regime_t::RMHD, "srmhd"},
        {regime_t::MHD, "mhd"}
    );

    REGISTER_ENUM_BIMAP(
        geometry_t,
        {geometry_t::CARTESIAN, "cartesian"},
        {geometry_t::SPHERICAL, "spherical"},
        {geometry_t::CYLINDRICAL, "cylindrical"},
        {geometry_t::AXIS_CYLINDRICAL, "axis_cylindrical"},
        {geometry_t::PLANAR_CYLINDRICAL, "planar_cylindrical"}
    );

    REGISTER_ENUM_BIMAP(
        accuracy_t,
        {accuracy_t::FIRST_ORDER, "first_order"},
        {accuracy_t::SECOND_ORDER, "second_order"}
    );

    REGISTER_ENUM_BIMAP(
        reconstruction_t,
        {reconstruction_t::PCM, "pcm"},
        {reconstruction_t::PLM, "plm"},
    );

    REGISTER_ENUM_BIMAP(
        solver_t,
        {solver_t::HLLD, "hlld"},
        {solver_t::HLLC, "hllc"},
        {solver_t::HLLE, "hlle"},
    );

    REGISTER_ENUM_BIMAP(
        wave_speed_estimate_t,
        {wave_speed_estimate_t::SCHNEIDER_ET_AL_93, "schneider_et_al_93"},
        {wave_speed_estimate_t::MIGNONE_AND_BODO_05, "mignone_and_bodo_05"},
        {wave_speed_estimate_t::HUBER_AND_KISSMANN_2021, "huber_and_kissmann_2021"},
        {wave_speed_estimate_t::DAVIDSON, "davidson"}
    );

    REGISTER_ENUM_BIMAP(
        shockwave_limiter_t,
        {shockwave_limiter_t::NONE, "none"},
        {shockwave_limiter_t::FLEISCHMANN, "fleischmann"},
        {shockwave_limiter_t::QUIRK, "quirk"}
    );

    REGISTER_ENUM_BIMAP(LIMITER, {LIMITER::MINMOD, "minmod"}, {LIMITER::VAN_LEER, "van_leer"});

    REGISTER_ENUM_BIMAP(
        ct_algo_t,
        {ct_algo_t::ALPHA, "alpha"},
        {ct_algo_t::CONTACT, "contact"},
        {ct_algo_t::MdZ, "mdz"},
        {ct_algo_t::ZERO, "zero"}
    );

    REGISTER_ENUM_BIMAP(
        subcycling_mode_t,
        {subcycling_mode_t::NONE, "none"},
        {subcycling_mode_t::STANDARD, "standard"},
        {subcycling_mode_t::MANUAL, "manual"},
        {subcycling_mode_t::ADAPTIVE, "adaptive"}
    );
} // namespace simbi
#endif
