#ifndef  FAMILIAR_H
#define  FAMILIAR_H
#include "shapeshift.h"

namespace loup {
    template <typename Function, typename... Arguments>
    GPU_LAUNCHABLE
    void Kernel(Function f, Arguments... args)
    {
        f(args...);
    }

}
#endif