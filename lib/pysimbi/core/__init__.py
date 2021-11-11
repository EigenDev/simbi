try:
    from .simbi import Hydro 
    from cpu_ext import *
except ImportError:
    print("CPU configuration not built")