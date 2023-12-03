#ifndef PYOBJ_WRAPPER_HPP
#define PYOBJ_WRAPPER_HPP

#include "call_obj.h"   // cython helper file
#include <Python.h>

// Adapted from:
// https://stackoverflow.com/questions/39044063/pass-a-closure-from-cython-to-c
class PyObjWrapper
{
  public:
    // constructors and destructors mostly do reference counting
    PyObjWrapper(PyObject* o) : held(o) { Py_XINCREF(o); }

    PyObjWrapper(const PyObjWrapper& rhs) : PyObjWrapper(rhs.held)
    {   // C++11 onwards only
    }

    PyObjWrapper(PyObjWrapper&& rhs) : held(rhs.held) { rhs.held = 0; }

    // need no-arg constructor to stack allocate in Cython
    PyObjWrapper() : PyObjWrapper(nullptr) {}

    ~PyObjWrapper() { Py_XDECREF(held); }

    PyObjWrapper& operator=(const PyObjWrapper& rhs)
    {
        PyObjWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyObjWrapper& operator=(PyObjWrapper&& rhs)
    {
        held     = rhs.held;
        rhs.held = 0;
        return *this;
    }

    double operator()(double x) const
    {
        if (held) {   // nullptr check
            const auto res = call_obj(held, x);
            return res;
        }
        return 0;
    }

    double operator()(double x, double y) const
    {
        if (held) {   // nullptr check
            const auto res = call_obj2(held, x, y);
            return res;
        }
        return 0;
    }

    double operator()(double x, double y, double z) const
    {
        if (held) {   // nullptr check
            const auto res = call_obj3(held, x, y, z);
            return res;
        }
        return 0;
    }

    operator bool() const { return held; }

  private:
    PyObject* held;
};

#endif