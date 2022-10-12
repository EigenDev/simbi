import numpy as np 
def calc_cell_volume1D(r: np.ndarray) -> np.ndarray:
    rvertices = np.sqrt(r[1:] * r[:-1])
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, r.shape, r[-1])
    rmean     = 0.75 * (rvertices[1:]**4 - rvertices[:-1]**4) / (rvertices[1:]**3 - rvertices[:-1]**3)
    dr        = rvertices[1:] - rvertices[:-1]
    return rmean * rmean * dr 

def calc_cell_volume2D(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    rr, thetta = np.meshgrid(r, theta)
    tvertices = 0.5 * (thetta[1:] + thetta[:-1])
    tvertices = np.insert(tvertices, 0, thetta[0], axis=0)
    tvertices = np.insert(tvertices, tvertices.shape[0], thetta[-1], axis=0)
    dcos      = np.cos(tvertices[:-1]) - np.cos(tvertices[1:])
    
    rvertices = np.sqrt(rr[:, 1:] * rr[:, :-1])
    rvertices = np.insert(rvertices,  0, rr[:, 0], axis=1)
    rvertices = np.insert(rvertices, rvertices.shape[1], rr[:, -1], axis=1)
    dr        = rvertices[:, 1:] - rvertices[:, :-1]
    return (2.0 * np.pi *  (1./3.) * (rvertices[:, 1:]**3 - rvertices[:, :-1]**3) *  dcos)

def print_problem_params(args, parser) -> None:
    print("\nProblem paramters:")
    print("="*90)
    for arg in vars(args):
        description = parser._option_string_actions[f'--{arg}'].help
        val = getattr(args, arg)
        if (isinstance(val ,float)):
            val = round(val, 3)
        val = str(val)
        print(f"{arg:.<30} {val:<15} {description}")