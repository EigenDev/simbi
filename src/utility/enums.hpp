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

#include <cstddef>
#include <cstdint>

namespace simbi {
    enum class Cellspacing {
        LINEAR,
        LOG,
    };

    enum class Regime {
        NEWTONIAN,
        SRHD,
        RMHD,
        MHD,
    };

    enum class Geometry {
        CARTESIAN,
        SPHERICAL,
        CYLINDRICAL,
        AXIS_CYLINDRICAL,
        PLANAR_CYLINDRICAL,
    };

    enum class Accuracy {
        FIRST_ORDER,
        SECOND_ORDER,
    };

    enum class Reconstruction {
        PCM,
        PLM,
    };

    enum class Timestepping {
        EULER,
        RK2,
    };

    enum class Solver {
        HLLD,
        HLLC,
        HLLE,
    };

    enum class TIMESTEP_TYPE {
        MINIMUM,
        ADAPTIVE,
    };

    enum class BoundaryCondition {
        REFLECTING = 0,
        OUTFLOW,
        DYNAMIC,
        PERIODIC,
    };

    enum class BoundaryFace {
        X1_INNER = 0,
        X1_OUTER = 1,
        X2_INNER = 2,
        X2_OUTER = 3,
        X3_INNER = 4,
        X3_OUTER = 5
    };

    enum class WaveSpeedEstimate {
        SCHNEIDER_ET_AL_93,
        MIGNONE_AND_BODO_05,
        HUBER_AND_KISSMANN_2021,
        DAVIDSON,
    };

    enum class CONS2PRIMTYPE {
        VOLUMETRIC,
        CHARGES,
    };

    enum class BlockAx {
        K,
        J,
        I
    };

    enum class Dir {
        N,
        E,
        S,
        W,
        SW,
        SE,
        NW,
        NE
    };

    enum class Corner {
        NE,
        SE,
        SW,
        NW
    };

    enum class Face {
        N,
        E,
        S,
        W
    };

    enum class Plane {
        IJ,
        IK,
        JK
    };

    enum Interface {
        LF,
        RF
    };

    namespace IJ {
        enum corner {
            NE,
            SE,
            SW,
            NW
        };
    };   // namespace IJ

    namespace IK {
        enum corner {
            NE,
            SE,
            SW,
            NW
        };
    };   // namespace IK

    namespace JK {
        enum corner {
            NE,
            SE,
            SW,
            NW
        };
    }   // namespace JK

    enum class CTAlgo {
        ZERO,
        CONTACT,
        ALPHA,
        MdZ,
    };

    enum class LIMITER {
        MINMOD,
        VAN_LEER
    };

    enum class ShockWaveLimiter {
        NONE,
        FLEISCHMANN,
        QUIRK,
    };

    enum class Color {
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

    enum class BodyCapability : uint32_t {
        NONE          = 0,
        GRAVITATIONAL = 1 << 0,
        ACCRETION     = 1 << 1,
        ELASTIC       = 1 << 2,
        DEFORMABLE    = 1 << 3,
        RIGID         = 1 << 4,

        // TODO: add more capabilities as needed
    };

    // component identifiers for magnetic field directions
    enum class magnetic_comp_t : std::uint64_t {
        I = 2,   // B1 component
        J = 1,   // B2 component
        K = 0    // B3 component
    };

    constexpr auto comp_wave_speed    = WaveSpeedEstimate::MIGNONE_AND_BODO_05;
    constexpr auto comp_ct_type       = CTAlgo::CONTACT;
    constexpr auto comp_slope_limiter = LIMITER::MINMOD;

    // register BiMaps for enum serialization and deserialization
    REGISTER_ENUM_BIMAP(
        Timestepping,
        {Timestepping::EULER, "rk1"},
        {Timestepping::RK2, "rk2"}
    );

    REGISTER_ENUM_BIMAP(
        Cellspacing,
        {Cellspacing::LINEAR, "linear"},
        {Cellspacing::LOG, "log"}
    );

    REGISTER_ENUM_BIMAP(
        Regime,
        {Regime::NEWTONIAN, "newtonian"},
        {Regime::SRHD, "srhd"},
        {Regime::RMHD, "srmhd"},
        {Regime::MHD, "mhd"}
    );

    REGISTER_ENUM_BIMAP(
        Geometry,
        {Geometry::CARTESIAN, "cartesian"},
        {Geometry::SPHERICAL, "spherical"},
        {Geometry::CYLINDRICAL, "cylindrical"},
        {Geometry::AXIS_CYLINDRICAL, "axis_cylindrical"},
        {Geometry::PLANAR_CYLINDRICAL, "planar_cylindrical"}
    );

    REGISTER_ENUM_BIMAP(
        Accuracy,
        {Accuracy::FIRST_ORDER, "first_order"},
        {Accuracy::SECOND_ORDER, "second_order"}
    );

    REGISTER_ENUM_BIMAP(
        Reconstruction,
        {Reconstruction::PCM, "pcm"},
        {Reconstruction::PLM, "plm"},
    );

    REGISTER_ENUM_BIMAP(
        Solver,
        {Solver::HLLD, "hlld"},
        {Solver::HLLC, "hllc"},
        {Solver::HLLE, "hlle"},
    );

    REGISTER_ENUM_BIMAP(
        BoundaryCondition,
        {BoundaryCondition::REFLECTING, "reflecting"},
        {BoundaryCondition::OUTFLOW, "outflow"},
        {BoundaryCondition::DYNAMIC, "dynamic"},
        {BoundaryCondition::PERIODIC, "periodic"}
    );

    REGISTER_ENUM_BIMAP(
        WaveSpeedEstimate,
        {WaveSpeedEstimate::SCHNEIDER_ET_AL_93, "schneider_et_al_93"},
        {WaveSpeedEstimate::MIGNONE_AND_BODO_05, "mignone_and_bodo_05"},
        {WaveSpeedEstimate::HUBER_AND_KISSMANN_2021, "huber_and_kissmann_2021"},
        {WaveSpeedEstimate::DAVIDSON, "davidson"}
    );

    REGISTER_ENUM_BIMAP(
        ShockWaveLimiter,
        {ShockWaveLimiter::NONE, "none"},
        {ShockWaveLimiter::FLEISCHMANN, "fleischmann"},
        {ShockWaveLimiter::QUIRK, "quirk"}
    );

    REGISTER_ENUM_BIMAP(
        LIMITER,
        {LIMITER::MINMOD, "minmod"},
        {LIMITER::VAN_LEER, "van_leer"}
    );

    REGISTER_ENUM_BIMAP(
        CTAlgo,
        {CTAlgo::ALPHA, "alpha"},
        {CTAlgo::CONTACT, "contact"},
        {CTAlgo::MdZ, "mdz"},
        {CTAlgo::ZERO, "zero"}
    );
}   // namespace simbi
#endif
