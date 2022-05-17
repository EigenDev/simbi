#ifndef PYOBJ_WRAPPER_HPP
#define PYOBJ_WRAPPER_HPP

#include <Python.h>
#include "util/call_obj.h" // cython helper file

class PyObjWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyObjWrapper(PyObject* o): held(o) {
        Py_XINCREF(o);
    }

    PyObjWrapper(const PyObjWrapper& rhs): PyObjWrapper(rhs.held) { // C++11 onwards only
    }

    PyObjWrapper(PyObjWrapper&& rhs): held(rhs.held) {
        rhs.held = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyObjWrapper(): PyObjWrapper(nullptr) {
    }

    ~PyObjWrapper() {
        Py_XDECREF(held);
    }

    PyObjWrapper& operator=(const PyObjWrapper& rhs) {
        PyObjWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyObjWrapper& operator=(PyObjWrapper&& rhs) {
        held = rhs.held;
        rhs.held = 0;
        return *this;
    }

    double operator()(double x) {
        if (held) { // nullptr check 
            const auto res = call_obj(held, x); 
            return res;
        }
    }

    double operator()(double x, double y){
        if (held) { // nullptr check 
            const auto res = call_obj2(held, x, y);
            return res;  
        }
    }

    double operator()(double x, double y, double z){
        if (held) { // nullptr check 
           const auto res = call_obj3(held, x, y, z);
           return res;
        }
    }

private:
    PyObject* held;
};

#endif