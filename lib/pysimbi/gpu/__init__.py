try:
    from pysimbi.gpu import Hydro
except ImportError:
    print("The gpu module not configured. Try installing with the --gpu flag")