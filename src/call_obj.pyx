cdef public double call_obj(obj, double t):
    cdef double res = obj(t)
    return res 

cdef public double call_obj2(obj, double x, double t):
    cdef double res = obj(x, t)
    return res 

cdef public double call_obj3(obj, double x, double y, double t):
    cdef double res = obj(x, y, t)
    return res

cdef public double call_obj4(obj, double x, double y, double z, double t):
    cdef double res = obj(x, y, z, t)
    return res 