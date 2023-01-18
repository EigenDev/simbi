# Module to config the initial condition for the SIMBI
# hydro setup. From here on, I will fragment the code 
# to try and reduce the confusion between functions
import numpy as np 
import h5py 
import simbi.helpers as helpers 
import numpy.typing as npt
from .key_types import *

def load_checkpoint(model: Any, filename: str, dim: int, mesh_motion: bool) -> None:
    print(f"Loading from checkpoint: {filename}...", flush=True)
    volume_factor: Any = 1.0
    with h5py.File(filename, 'r+') as hf:
        t = 0
        ds = hf.get("sim_info")
        
        if dim == 1:
            rho              = hf.get("rho")[:]
            v                = hf.get("v")[:]
            p                = hf.get("p")[:]
            nx               = ds.attrs["nx"]
            model.start_time = ds.attrs["current_time"]
            x1max            = ds.attrs["x1max"]
            x1min            = ds.attrs["x1min"]
            ad_gamma         = ds.attrs["adiabatic_gamma"]
            model.ckpt_idx   = ds.attrs['chkpt_idx']
            
            if mesh_motion:
                nx_active = ds.attrs['xactive_zones']
                if ds.attrs['linspace']:
                    model.x1 = np.linspace(x1min, x1max, nx_active)
                else:
                    model.x1 = np.geomspace(x1min, x1max, nx_active)

                volume_factor = helpers.calc_cell_volume1D(x1=model.x1)
            
            h = 1. + ad_gamma*p/(rho*(ad_gamma - 1.0))
            
            W   = 1./np.sqrt(1. - v ** 2)
            model.D   = rho * W 
            model.S   = W**2 * rho*h*v
            model.tau = W**2 * rho*h - p - rho*W
            model.u   = np.array([model.D, model.S, model.tau])
            if mesh_motion:
                if ds.attrs['boundary_condition'] == 'periodic':
                    model.u   *= volume_factor
                else:
                    if ds.attrs['first_order']:
                        nghosts = 1 
                    else:
                        nghosts = 2 
                    model.u[:, nghosts:-nghosts] *= volume_factor
                    model.u[:, 0:nghosts]        *= volume_factor[0]
                    model.u[:, -nghosts: ]       *= volume_factor[-1]
        else:
            rho     = hf.get("rho")[:]
            v1      = hf.get("v1")[:]
            v2      = hf.get("v2")[:]
            p       = hf.get("p")[:]
            nx      = ds.attrs["nx"]
            ny      = ds.attrs["ny"]
            scalars = hf.get("chi")[:]

            model.start_time = ds.attrs["current_time"]
            x1max   = ds.attrs["x1max"]
            x1min   = ds.attrs["x1min"]
            x2max   = ds.attrs["x2max"]
            x2min   = ds.attrs["x2min"]

            ad_gamma = ds.attrs["adiabatic_gamma"]
            model.chkpt_idx = ds.attrs['chkpt_idx']
            
            if mesh_motion:
                nx_active = ds.attrs['xactive_zones']
                ny_active = ds.attrs['yactive_zones']
                if (nx_active == 0 or ny_active == 0):
                    nx_active = nx 
                    ny_active = ny 
                    if ds.attrs['boundary_condition'] != 'periodic':
                        if ds.attrs['first_order']:
                            nx_active -= 2
                            ny_active -= 2
                        else:
                            nx_active -= 4
                            ny_active -= 4
                
                try:
                    model.x1 = hf.get('x1')[:]
                    model.x2 = hf.get('x2')[:]
                except:
                    if ds.attrs['linspace']:
                        model.x1 = np.linspace(x1min, x1max, nx_active)
                        model.x2 = np.linspace(x1min, x2max, ny_active)
                    else:
                        model.x1 = np.geomspace(x1min, x1max, nx_active)
                        model.x2 = np.linspace(x2min,  x2max, ny_active)
                volume_factor = helpers.calc_cell_volume2D(x1=model.x1, x2=model.x2, coord_system=model.coord_system)
            rho     = rho.reshape(ny, nx)
            v1      = v1.reshape(ny, nx)
            v2      = v2.reshape(ny, nx)
            p       = p.reshape(ny, nx)
            scalars = scalars.reshape(ny,nx)
            
            if ds.attrs['regime'].decode("utf-8") == 'relativistic':
                h = 1. + ad_gamma*p/(rho*(ad_gamma - 1.0))
                if ds.attrs['using_gamma_beta']:
                    W = np.sqrt(1 + (v1*v1 + v2 * v2))
                    v1 /= W 
                    v2 /= W
                else:
                    W   = 1./np.sqrt(1. - (v1*v1 + v2*v2))
                D    = rho * W              
                S1   = W*W*rho*h*v1         
                S2   = W*W*rho*h*v2         
                tau  = W*W*rho*h - p - rho*W
                Dchi = D * scalars    
                model.u    = np.array([D, S1, S2, tau, Dchi])
            else:
                model.u    = np.array([rho, rho * v1, rho * v2, p / (ad_gamma - 1) + 0.5 * rho * (v1 ** 2 + v2 ** 2), rho * scalars])
            
            if mesh_motion:
                if ds.attrs['boundary_condition'] == 'periodic':
                    model.u   *= volume_factor
                else:
                    if ds.attrs['first_order']:
                        nghosts = 1 
                    else:
                        nghosts = 2 
                model.u[:, nghosts:-nghosts, nghosts:-nghosts] *= volume_factor
                model.u[:, nghosts:-nghosts, 0: nghosts]       *= volume_factor[:, 0].reshape(-1,1)
                model.u[:, nghosts:-nghosts, -nghosts: ]       *= volume_factor[:, -1].reshape(-1,1)
                model.u[:, 0: nghosts, nghosts:-nghosts]       *= volume_factor[0, :]
                model.u[:, -nghosts: , nghosts:-nghosts]       *= volume_factor[-1, :]
                for i in range(nghosts):
                    model.u[:, :,  (i + 0)] = model.u[:, :,  (nghosts + 0)]
                    model.u[:, :, -(i + 1)] = model.u[:, :, -(nghosts + 1)]
                    model.u[:,  (i + 0), :] = model.u[:,  (nghosts + 0), :]
                    model.u[:, -(i + 1), :] = model.u[:, -(nghosts + 1), :]
        

def initializeModel(model: Any, first_order: bool, volume_factor: Union[float, NDArray[Any]], passive_scalars: Union[npt.NDArray[Any], Any]) -> None:
    full_periodic = all(bc == 'periodic' for bc in model.boundary_conditions)
    if full_periodic:
        return
    
    if passive_scalars is not None:
        model.u[-1,...] = passive_scalars
    
    npad = ((0,0),) + tuple(tuple(val) for val in [[2 * (first_order + 1), 2 * (first_order + 1)]] * model.dimensionality) 
    model.u = np.pad(model.u, npad, 'edge')  * volume_factor