#ifndef PYOBJ_WRAPPER_HPP
#define PYOBJ_WRAPPER_HPP

#include <Python.h>
#include <stdexcept>

class PyObjWrapper
{
  public:
    // Nullary constructor
    PyObjWrapper() : py_func(nullptr) {}

    PyObjWrapper(PyObject* py_func) : py_func(py_func)
    {
        if (py_func && !PyCallable_Check(py_func)) {
            throw std::invalid_argument("Object is not callable");
        }
        Py_XINCREF(py_func);   // Use Py_XINCREF to handle nullptr
    }

    PyObjWrapper(const PyObjWrapper& other) : py_func(other.py_func)
    {
        Py_XINCREF(py_func);   // Use Py_XINCREF to handle nullptr
    }

    PyObjWrapper(PyObjWrapper&& other) noexcept : py_func(other.py_func)
    {
        other.py_func = nullptr;
    }

    PyObjWrapper& operator=(const PyObjWrapper& other)
    {
        if (this != &other) {
            PyGILState_STATE gstate = PyGILState_Ensure();
            if (py_func) {
                Py_DECREF(py_func);
            }
            py_func = other.py_func;
            Py_XINCREF(py_func);   // Use Py_XINCREF to handle nullptr
            PyGILState_Release(gstate);
        }
        return *this;
    }

    PyObjWrapper& operator=(PyObjWrapper&& other) noexcept
    {
        if (this != &other) {
            PyGILState_STATE gstate = PyGILState_Ensure();
            if (py_func) {
                Py_DECREF(py_func);
            }
            py_func       = other.py_func;
            other.py_func = nullptr;
            PyGILState_Release(gstate);
        }
        return *this;
    }

    ~PyObjWrapper()
    {
        PyGILState_STATE gstate = PyGILState_Ensure();
        if (py_func) {
            Py_DECREF(py_func);
        }
        PyGILState_Release(gstate);
    }

    double operator()(double x) const
    {
        if (!py_func) {
            throw std::runtime_error("Python function is not set");
        }
        PyGILState_STATE gstate = PyGILState_Ensure();
        PyObject* args          = PyTuple_Pack(1, PyFloat_FromDouble(x));
        if (!args) {
            PyGILState_Release(gstate);
            throw std::runtime_error("Error creating arguments tuple");
        }
        PyObject* result = PyObject_CallObject(py_func, args);
        Py_DECREF(args);
        if (!result) {
            PyGILState_Release(gstate);
            throw std::runtime_error("Error calling Python function");
        }
        double ret = PyFloat_AsDouble(result);
        Py_DECREF(result);
        PyGILState_Release(gstate);
        return ret;
    }

    double operator()(double x, double t) const
    {
        if (!py_func) {
            throw std::runtime_error("Python function is not set");
        }
        PyGILState_STATE gstate = PyGILState_Ensure();
        PyObject* args =
            PyTuple_Pack(2, PyFloat_FromDouble(x), PyFloat_FromDouble(t));
        if (!args) {
            PyGILState_Release(gstate);
            throw std::runtime_error("Error creating arguments tuple");
        }
        PyObject* result = PyObject_CallObject(py_func, args);
        Py_DECREF(args);
        if (!result) {
            PyGILState_Release(gstate);
            throw std::runtime_error("Error calling Python function");
        }
        double ret = PyFloat_AsDouble(result);
        Py_DECREF(result);
        PyGILState_Release(gstate);
        return ret;
    }

    double operator()(double x, double y, double t) const
    {
        if (!py_func) {
            throw std::runtime_error("Python function is not set");
        }
        PyGILState_STATE gstate = PyGILState_Ensure();
        PyObject* args          = PyTuple_Pack(
            3,
            PyFloat_FromDouble(x),
            PyFloat_FromDouble(y),
            PyFloat_FromDouble(t)
        );
        if (!args) {
            PyGILState_Release(gstate);
            throw std::runtime_error("Error creating arguments tuple");
        }
        PyObject* result = PyObject_CallObject(py_func, args);
        Py_DECREF(args);
        if (!result) {
            PyGILState_Release(gstate);
            throw std::runtime_error("Error calling Python function");
        }
        double ret = PyFloat_AsDouble(result);
        Py_DECREF(result);
        PyGILState_Release(gstate);
        return ret;
    }

    double operator()(double x, double y, double z, double t) const
    {
        if (!py_func) {
            throw std::runtime_error("Python function is not set");
        }
        PyGILState_STATE gstate = PyGILState_Ensure();
        PyObject* args          = PyTuple_Pack(
            4,
            PyFloat_FromDouble(x),
            PyFloat_FromDouble(y),
            PyFloat_FromDouble(z),
            PyFloat_FromDouble(t)
        );
        if (!args) {
            PyGILState_Release(gstate);
            throw std::runtime_error("Error creating arguments tuple");
        }
        PyObject* result = PyObject_CallObject(py_func, args);
        Py_DECREF(args);
        if (!result) {
            PyGILState_Release(gstate);
            throw std::runtime_error("Error calling Python function");
        }
        double ret = PyFloat_AsDouble(result);
        Py_DECREF(result);
        PyGILState_Release(gstate);
        return ret;
    }

    // Conversion operator to bool
    operator bool() const { return py_func != nullptr; }

  private:
    PyObject* py_func;
};

#endif   // PYOBJ_WRAPPER_HPP