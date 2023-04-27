# Module to config the initial condition for the SIMBI
# hydro setup. From here on, I will fragment the code 
# to try and reduce the confusion between functions

import numpy as np 
import h5py 
import numpy.typing as npt
from ..key_types import *
from . import helpers

def flatten_fully(x: Any) -> Any:
    if any(dim == 1 for dim in x.shape):
        x = np.vstack(x)
        if len(x.shape) == 2 and x.shape[0] == 1:
            return x.flatten()
        return flatten_fully(x)
    else:
        return np.asanyarray(x) 
    
def load_checkpoint(model: Any, filename: str, dim: int, mesh_motion: bool) -> None:
    print(f"Loading from checkpoint: {filename}...", flush=True)
    setup: dict[str, Any] = {}
    volume_factor: Union[float, NDArray[numpy_float]] = 1.0
    with h5py.File(filename, 'r') as hf:         
        ds  = hf.get('sim_info')
        nx  = ds.attrs['nx'] or 1
        ny  = ds.attrs['ny'] if 'ny' in ds.attrs.keys() else 1
        nz  = ds.attrs['nz'] if 'nz' in ds.attrs.keys() else 1
        try:
            ndim = ds.attrs['dimensions']
        except KeyError:
            ndim = 1 + (ny > 1) + (nz > 1)
            
        setup['ad_gamma']     = ds.attrs['adiabatic_gamma']
        setup['regime']       = ds.attrs['regime'].decode('utf-8')
        setup['coord_system'] = ds.attrs['geometry'].decode('utf-8')
        setup['mesh_motion']  = ds.attrs['mesh_motion']
        setup['linspace']     = ds.attrs['linspace']
        if not (bcs := hf.get('boundary_conditions')):
            try:
                bcs = [ds.attrs['boundary_condition']]
            except KeyError:
                bcs = [b'outflow']
                
        full_periodic = all(bc.decode("utf-8") == 'periodic' for bc in bcs)
        #------------------------
        # Generate Mesh
        #------------------------
        arr_gen: Any = np.linspace if setup['linspace'] else np.geomspace
        funcs  = [arr_gen, np.linspace, np.linspace]
        mesh = {
            f'x{i+1}': hf.get(f'x{i+1}')[:] for i in range(ndim)
        }
        
        if ds.attrs['x1max'] > mesh['x1'][-1]:
            mesh['x1'] = arr_gen(ds.attrs['x1min'], ds.attrs['x1max'], ds.attrs['xactive_zones'])
        
        if setup['mesh_motion']:
            if ndim == 1 and setup['coord_system'] != 'cartesian':
                volume_factor = helpers.calc_cell_volume1D(
                    x1=mesh['x1'], 
                    coord_system=setup['coord_system']
                )
            elif ndim == 2:
                volume_factor = helpers.calc_cell_volume2D(
                    x1=mesh['x1'], 
                    x2=mesh['x2'],
                    coord_system=setup['coord_system']
                )
            elif ndim == 3:
                raise NotImplementedError()
                # volume_factor = helpers.calc_cell_volume3D(
                #     x1=mesh['x1'], 
                #     x2=mesh['x2'],
                #     x3=mesh['x3'],
                #     coord_system=setup['coord_system']
                # )
            
            if setup['coord_system'] != 'cartesian':
                npad = tuple(tuple(val) for val in [[((ds.attrs['first_order']^1) + 1), 
                                                     ((ds.attrs['first_order']^1) + 1)]] * ndim)
                volume_factor = np.pad(volume_factor, npad, 'edge')
        
        rho  = hf.get('rho')[:]
        v    = [(hf.get(f'v{dim}') or hf.get(f'v'))[:] for dim in range(1,ndim + 1)]
        p    = hf.get('p')[:]         
        chi  = (hf.get('chi') or np.zeros_like(rho))[:]
        rho = flatten_fully(rho.reshape(nz, ny, nx))
        v   = [flatten_fully(vel.reshape(nz, ny, nx)) for vel in v]
        p   = flatten_fully(p.reshape(nz, ny, nx))
        chi = flatten_fully(chi.reshape(nz, ny, nx))
        
        #-------------------------------
        # Load Fields
        #-------------------------------
        vsqr = np.sum(vel * vel for vel in v) # type: ignore
        if setup['regime'] == 'relativistic':
            if ds.attrs['using_gamma_beta']:
                W = (1 + vsqr) ** 0.5
                v     = [vel / W for vel in v]
                vsqr /= W**2 
            else:
                W = (1 - vsqr) ** (-0.5)
        else:
            W = 1
            
        if setup['regime'] == 'relativistic':
            h = 1.0 + setup['ad_gamma'] * p / (rho * (setup['ad_gamma'] - 1.0))
            e = rho * W * W * h - p - rho * W 
        else:
            h = 1.0 
            e = p / (setup['ad_gamma'] - 1.0) + 0.5 * rho * vsqr
            
        momentum         = np.asarray([rho * W * W * h * vel for vel in v])
        model.start_time = ds.attrs['current_time']
        model.x1         = mesh['x1']
        if ndim >= 2:
            model.x2 = mesh['x2']
        if ndim >= 3:
            model.x3 = mesh['x3']
            
        if ndim == 1:
            model.u = np.array([rho * W, *momentum, e]) * volume_factor
        else:
            model.u = np.array([rho * W, *momentum, e, rho * W * chi]) * volume_factor
        
        model.chkpt_idx = ds.attrs['chkpt_idx']
        
def initializeModel(model: Any, first_order: bool, volume_factor: Union[float, NDArray[Any]], passive_scalars: Union[npt.NDArray[Any], Any]) -> None:
    full_periodic = all(bc == 'periodic' for bc in model.boundary_conditions)
    if full_periodic:
        return
    
    if passive_scalars is not None:
        model.u[-1,...] = passive_scalars * model.u[0]
    
    npad = ((0,0),) + tuple(tuple(val) for val in [[((first_order^1) + 1),  ((first_order^1) + 1)]] * model.dimensionality) 
    model.u = np.pad(model.u  * volume_factor, npad, 'edge')