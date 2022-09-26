# distutils: language = c++
cdef public double call_obj(obj, double x):
    cdef double res = obj(x)
    return res 

cdef public double call_obj2(obj, double x, double y):
    cdef double res = obj(x, y)
    return res

cdef public double call_obj3(obj, double x, double y, double z):
    cdef double res = obj(x, y, z)
    return res 