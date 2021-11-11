try:
    from gpu_ext import *
    from .simbi import Hydro
except ImportError:
    print("The gpu extention not configured.")