/**
 * Houses all of the system configuration enum classes
 * 
*/

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "build_options.hpp"
// Precision preprocessing
#if FLOAT_PRECISION
typedef float real;
constexpr real tol_scale = 1e-6;
#else
typedef double real;
constexpr real tol_scale = 1e-12;
#endif 

#define my_max(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b;       \
})

#define my_min(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b;       \
})

constexpr int BLOCK_SIZE   = 64;
constexpr int BLOCK_SIZE2D = 16;
constexpr int BLOCK_SIZE3D = 4;

constexpr int MAX_ITER     = 50;

// autonomous self memeber alias
template<typename T>
struct Self
{
protected:
    typedef T self;
};

namespace simbi{
    enum class MemSide {
        Host,
        Dev,
    };

    enum class Cellspacing
    {
        LINSPACE,
        LOGSPACE,
    };

    enum class Geometry
    {
        SPHERICAL,
        CARTESIAN,
    };

    enum class Accuracy
    {
        FIRST_ORDER,
        SECOND_ORDER,
    };

    enum class Dimensions
    {
        ONE_D,
        TWO_D,
    };

    enum class Solver
    {
        HLLC,
        HLLE,
    };

    enum class WaveSpeeds
    {
        SCHNEIDER_ET_AL_93,
        MIGNONE_AND_BODO_05,
    };

    constexpr auto comp_wave_speed = WaveSpeeds::SCHNEIDER_ET_AL_93;
}

#endif