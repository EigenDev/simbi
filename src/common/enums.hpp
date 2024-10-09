/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       enums.hpp
 * @brief A place to house compilation enums for context switching in a sim
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */

#ifndef ENUMS_HPP
#define ENUMS_HPP

namespace simbi {
    enum class Cellspacing {
        LINSPACE,
        LOGSPACE,
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
        REFLECTING,
        OUTFLOW,
        INFLOW,
        PERIODIC,
    };

    enum class HydroRegime {
        RELATIVISTC,
        NEWTONIAN,
    };

    enum class WaveSpeeds {
        SCHNEIDER_ET_AL_93,
        MIGNONE_AND_BODO_05,
        HUBER_AND_KISSMANN_2021,
        NAIVE,
    };

    enum class CONS2PRIMTYPE {
        VOLUMETRIC,
        CHARGES,
    };

    enum class BlkAx {
        K,
        J,
        I
    };

    enum class Pos {
        center,
        left_face,
        right_face,
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
        ALPHA
    };

    enum class LIMITER {
        MINMOD,
        VAN_LEER
    };

    constexpr auto comp_wave_speed    = WaveSpeeds::MIGNONE_AND_BODO_05;
    constexpr auto comp_ct_type       = CTTYPE::CONTACT;
    constexpr auto comp_slope_limiter = LIMITER::VAN_LEER;
    constexpr auto comp_hllc_type     = HLLCTYPE::CLASSICAL;
}   // namespace simbi
#endif