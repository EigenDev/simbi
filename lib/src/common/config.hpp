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

#define my_max3(a,b, c)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    __typeof__ (c) _c = (c); \
    _a > _b ? (_a > _c ? _a : _c) : _b > _c ? _b : _c ;   \
})

#define my_min(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b;       \
})

#define my_min3(a,b, c)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    __typeof__ (c) _c = (c); \
    _a < _b ? (_a < _c ? _a : _c) : _b < _c ? _b : _c ;   \
})

#include <functional>

// constexpr int BLOCK_SIZE   = 64;
// constexpr int BLOCK_SIZE2D = 16;
// constexpr int BLOCK_SIZE3D = 4;
// constexpr int MAX_ITER     = 50;

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
        CYLINDRICAL,
    };

    enum class Accuracy
    {
        FIRST_ORDER,
        SECOND_ORDER,
    };

    enum class Solver
    {
        HLLC,
        HLLE,
    };

    enum class BoundaryCondition
    {
        REFLECTING,
        OUTFLOW,
        INFLOW,
        PERIODIC,
    };

    enum class WaveSpeeds
    {
        SCHNEIDER_ET_AL_93,
        MIGNONE_AND_BODO_05,
    };

    constexpr auto comp_wave_speed = WaveSpeeds::MIGNONE_AND_BODO_05;
}

#endif