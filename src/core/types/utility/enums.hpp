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

    enum class Solver {
        HLLD,
        HLLC,
        HLLE,
        AUSM_PLUS,
        SLAU,
    };

    enum class TIMESTEP_TYPE {
        MINIMUM,
        ADAPTIVE,
    };

    enum class HLLCTYPE {
        CLASSICAL,
        // Apply the low-Mach HLLC fix found in Fleischmann et al 2020:
        // https://www.sciencedirect.com/science/article/pii/S0021999120305362
        FLEISCHMANN,
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
        NAIVE,
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

    enum class CTTYPE {
        ZERO,
        CONTACT,
        ALPHA,
        MdZ,
    };

    enum class LIMITER {
        MINMOD,
        VAN_LEER
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

    enum class VectorType : uint32_t {
        GENERAL       = 0,
        SPATIAL       = 1 << 0 | GENERAL,
        MAGNETIC      = 1 << 1 | GENERAL,
        MAGNETIC_FOUR = 1 << 2 | GENERAL,
        SPACETIME     = 1 << 3 | GENERAL,
    };

    constexpr auto comp_wave_speed    = WaveSpeedEstimate::MIGNONE_AND_BODO_05;
    constexpr auto comp_ct_type       = CTTYPE::CONTACT;
    constexpr auto comp_slope_limiter = LIMITER::MINMOD;
    constexpr auto comp_hllc_type     = HLLCTYPE::CLASSICAL;
}   // namespace simbi
#endif
