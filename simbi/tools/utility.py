# Utility functions for visualization scripts 
import h5py 
import astropy.constants as const 
import astropy.units as units
import numpy as np 
import argparse 
import matplotlib.pyplot as plt 
import os
from typing import Union, Any, Callable, Optional
from numpy.typing import NDArray 
from numpy import int64 as numpy_int, float64 as numpy_float, cast
from ..detail.helpers import find_nearest

# FONT SIZES
SMALL_SIZE   = 6
DEFAULT_SIZE = 10
BIGGER_SIZE  = 12

logically_curvlinear = ['spherical', 'cylindrical', 'planar_cylindrical']
logically_cartesian  = ['cartesian', 'axis_cylindrical']
#================================
#   constants of nature
#================================
R_0 = const.R_sun.cgs 
c   = const.c.cgs
m   = const.M_sun.cgs
 
rho_scale    = m / R_0 ** 3 
e_scale      = m * c **2
edens_scale  = e_scale / R_0**3
time_scale   = R_0 / c
mass_scale   = m

e_scale_bmk   = 1e53 * units.erg
rho_scale_bmk = 1.0 * const.m_p.cgs / units.cm**3
ell_scale     = (e_scale_bmk / rho_scale_bmk / const.c.cgs**2)**(1/3)
t_scale       = const.c.cgs * ell_scale

def calc_enthalpy(fields: dict[str, NDArray[numpy_float]]) -> Any:
    return 1.0 + fields['p']*fields['ad_gamma'] / (fields['rho'] * (fields['ad_gamma'] - 1.0))
    
def calc_lorentz_gamma(fields: dict[str, NDArray[numpy_float]]) -> Any:
    return (1.0 + fields['gamma_beta']**2)**0.5

def calc_beta(fields: dict[str, NDArray[numpy_float]]) -> Any:
    W = calc_lorentz_gamma(fields)
    return (1.0 - 1.0 / W**2)**0.5

def get_field_str(args: argparse.Namespace) -> Union[str, list[str]]:
    field_str_list = []
    for field in args.fields:
        if field == 'rho' or field == 'D':
            var = r'\rho' if field == 'rho' else 'D'
            if args.units:
                field_str_list.append( r'${}$ [g cm$^{{-3}}$]'.format(var))
            else:
                field_str_list.append( r'${}/{}_0$'.format(var,var))
            
        elif field in ['gamma_beta', 'u']:
            field_str_list.append( r'$\Gamma \beta$')
        elif field in ['gamma_beta_1', 'u1']:
            field_str_list.append( r'$\Gamma \beta_1$')
        elif field in ['gamma_beta_2', 'u2']:
            field_str_list.append( r'$\Gamma \beta_2$')
        elif field in ['gamma_beta_3', 'u3']:
            field_str_list.append( r'$\Gamma \beta_3$')
        elif field in ['energy', 'p']:
            if args.units:
                if field == 'energy':
                    field_str_list.append( r'$\tau [\rm erg \ cm^{{-3}}]$')
                else:
                    field_str_list.append( r'$p [\rm erg \ cm^{{-3}}]$')
            else:
                if field == 'energy':
                    field_str_list.append( r'$\tau/\tau_0$')
                else:
                    field_str_list.append( r'$p/p_0$')
        elif field == 'energy_rst':
            if args.units:
                field_str_list.append( r'$E \  [\rm erg \ cm^{-3}]$')
            else:
                field_str_list.append( r'$E / E_0$')
        elif field == 'chi':
            field_str_list.append( r'$\chi$')
        elif field == 'chi_dens':
            field_str_list.append( r'$D \cdot \chi$')
        elif field == 'T_eV':
            field_str_list.append("T [eV]" if args.units else "T")
        elif field == 'temperature':
            field_str_list.append("T [K]" if args.units else "T")
        elif field == 'mach':
            field_str_list.append('M')
        elif field == 'v1' or field == 'v':
            field_str_list.append('$v / v_0$')
        elif field == 'v2':
            field_str_list.append('$v_2 / v_0$')
        elif field == 'v3':
            field_str_list.append('$v_3 / v_0$')
        else:
            field_str_list.append(rf'${field}$')

    return field_str_list if len(args.fields) > 1 else field_str_list[0]

def unpad(arr: NDArray[numpy_float], pad_width: tuple[tuple[Any, ...], ...]) -> Any:
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return arr[tuple(slices)]

def flatten_fully(x: NDArray[numpy_float]) -> Any:
    if any(dim == 1 for dim in x.shape):
        x = np.vstack(x) #type: ignore
        if len(x.shape) == 2 and x.shape[0] == 1:
            return x.flatten()
        return flatten_fully(x)
    else:
        return np.asanyarray(x) 
    
def get_dimensionality(files: Union[list[str], dict[int, list[str]]]) -> int:
    dims = []
    all_equal: Callable[[list[int]], bool] = lambda x: x.count(x[0]) == len(x)
    ndim: int = 0
    if isinstance(files, dict):
        import itertools
        files = list(itertools.chain(*files.values()))
    
    files = list(filter(bool, files))
    for file in files:
        with h5py.File(file, 'r') as hf:
            ds  = hf.get('sim_info')
            try:
                ndim = ds.attrs['dimensions']
            except KeyError:
                ny   = ds.attrs['ny'] if 'ny' in ds.attrs.keys() else 1
                nz   = ds.attrs['nz'] if 'nz' in ds.attrs.keys() else 1  
                ndim = 1 + (ny > 1) + (nz > 1)
            dims += [ndim]
            if not all_equal(dims):
                raise ValueError("All simulation files require identical dimensionality")
    
    return ndim
    
            
def read_file(args: argparse.Namespace, filename: str, ndim: int) -> tuple[dict[str, Any], dict[str, Any],dict[str, Any]]:
    rho: Any 
    v: Any 
    p: Any 
    chi: Any 
    
    def try_read(dset: Union[h5py.AttributeManager, h5py.File], key: str, *, fall_back_key: str = 'None', fall_back: Any = None) -> Any:
        if isinstance(dset, h5py.File):
            try:
                res = dset.get(key)[:]
            except TypeError:
                res = fall_back
        else:
            try:
                res = dset.attrs[key]
            except KeyError:
                try:
                    res = dset.attrs[fall_back_key]
                except KeyError:
                    res = fall_back
        
        if isinstance(res, bytes):
            res = res.decode('utf-8')
        
        return res 
    
    setup = {}
    with h5py.File(filename, 'r') as hf: 
        ds  = hf.get('sim_info')
        rho = hf.get('rho')[:]
        v   = [(hf.get(f'v{dim}') or hf.get(f'v'))[:] for dim in range(1,ndim + 1)]
        p   = hf.get('p')[:]         
        chi = (hf.get('chi') or np.zeros_like(rho))[:]

        if not (bcs := hf.get('boundary_conditions')):
            try:
                bcs = [ds.attrs['boundary_condition']]
            except KeyError:
                bcs = [b'outflow']
                
        full_periodic = all(bc.decode("utf-8") == 'periodic' for bc in bcs)
        if 'no_cut' in vars(args).keys() and args.no_cut:
            full_periodic = True
                
        setup['first_order']  = try_read(ds, key='first_order', fall_back=False)
        nx                    = ds.attrs['nx'] if 'nx' in ds.attrs.keys() else 1
        ny                    = ds.attrs['ny'] if 'ny' in ds.attrs.keys() else 1
        nz                    = ds.attrs['nz'] if 'nz' in ds.attrs.keys() else 1
        
        setup['x1active']     = nx if full_periodic else nx - 2 * (1 + (setup['first_order']^1)) * (nx - 2 > 0)
        setup['x2active']     = ny if full_periodic else ny - 2 * (1 + (setup['first_order']^1)) * (ny - 2 > 0)
        setup['x3active']     = nz if full_periodic else nz - 2 * (1 + (setup['first_order']^1)) * (nz - 2 > 0)
        setup['time']         = ds.attrs['current_time']
        setup['linspace']     = ds.attrs['linspace']
        setup['ad_gamma']     = ds.attrs['adiabatic_gamma']
        setup['x1min']        = try_read(ds, 'x1min', fall_back_key='xmin', fall_back=0.0)
        setup['x1max']        = try_read(ds, 'x1max', fall_back_key='xmax', fall_back=0.0)
        setup['x2min']        = try_read(ds, 'x2min', fall_back_key='ymin', fall_back=0.0)
        setup['x2max']        = try_read(ds, 'x2max', fall_back_key='ymax', fall_back=0.0)
        setup['x3min']        = try_read(ds, 'x3min', fall_back_key='zmin', fall_back=0.0)
        setup['x3max']        = try_read(ds, 'x3max', fall_back_key='zmax', fall_back=0.0)
        setup['regime']       = try_read(ds, 'regime', fall_back='relativistic')
        setup['coord_system'] = try_read(ds, 'geometry', fall_back='spherical') 
        setup['mesh_motion']  = try_read(ds, key='mesh_motion', fall_back=False)
        setup['is_cartesian'] = setup['coord_system'] in logically_cartesian
        
        rho = flatten_fully(rho.reshape(nz, ny, nx))
        v   = [flatten_fully(vel.reshape(nz, ny, nx)) for vel in v]
        p   = flatten_fully(p.reshape(nz, ny, nx))
        chi = flatten_fully(chi.reshape(nz, ny, nx))
        if 'dt' in ds.attrs:
            setup['dt'] = ds.attrs['dt']
        else:
            setup['dt'] = ds.attrs['time_step']
            
        if not full_periodic:
            npad = tuple(tuple(val) for val in [[((setup['first_order']^1) + 1), ((setup['first_order']^1) + 1)]] * ndim) 
            rho  = unpad(rho, npad)
            v    = np.asanyarray([unpad(vel, npad) for vel in v])
            p    = unpad(p, npad)
            chi  = unpad(chi, npad)
        
        #-------------------------------
        # Load Fields
        #-------------------------------
        fields = {f'v{i+1}': v[i] for i in range(len(v))}
        fields['rho']          = rho
        fields['p']            = p
        fields['chi']          = chi
        fields['ad_gamma']     = setup['ad_gamma']
        
        vsqr = np.sum(vel * vel for vel in v) # type: ignore
        if setup['regime'] == 'relativistic':
            if ds.attrs['using_gamma_beta']:
                W = (1 + vsqr) ** 0.5
                fields.update({f'v{i+1}': v[i] / W for i in range(len(v))})
                vsqr /= W**2 
            else:
                W = (1 - vsqr) ** (-0.5)
        else:
            W = 1
        fields['gamma_beta']   = np.sqrt(vsqr) * W 
        fields['W'] = W 
        fields['W'] = W

        #------------------------
        # Generate Mesh
        #------------------------
        arr_gen: Union[Callable[..., Any], Callable[..., Any]]
        arr_gen = np.linspace if setup['linspace'] else np.geomspace
        funcs  = [arr_gen, np.linspace, np.linspace]
        mesh = { 
            f'x{i+1}': 
                try_read(hf, f'x{i+1}',
                    fall_back=funcs[i](
                        setup[f'x{i+1}min'], 
                        setup[f'x{i+1}max'], 
                        setup[f'x{i+1}active']
                    )  
                ) for i in range(ndim)
        }
        
        if setup['x1max'] > mesh['x1'][-1]:
            mesh['x1'] = arr_gen(setup['x1min'], setup['x1max'], setup['x1active'])
    
    return fields, setup, mesh 

def prims2var(fields: dict[str, NDArray[numpy_float]], var: str) -> Any:
    h = calc_enthalpy(fields)
    W = calc_lorentz_gamma(fields)
    if var == 'D':
        # Lab frame density
        return fields['rho'] * W
    elif var == 'S':
        # Lab frame momentum density
        return fields['rho'] * W**2 * calc_enthalpy(fields) * fields['v']
    elif var == 'energy':
        # Energy minus rest mass energy
        return fields['rho']*h*W**2 - fields['p'] - fields['rho']*W
    elif var == 'energy_rst':
        # Total Energy
        return fields['rho']*h*W**2 - fields['p']
    elif var == "temperature":
        a    = (4.0 * const.sigma_sb.cgs / c)
        T    = (3.0 * fields['p'] * edens_scale  / a)**0.25
        return T
    elif var == 'T_eV':
        a    = (4.0 * const.sigma_sb.cgs / c)
        T    = (3.0 * fields['p'] * edens_scale  / a)**0.25
        T_eV = (const.k_B.cgs * T).to(units.eV)
        return T_eV
    elif var == 'chi_dens':
        fields['chi'][fields['chi'] == 0] = 1.e-10
        return fields['chi'] * fields['rho'] * W
    elif var == 'gamma_beta_1':
        return W * fields['v1']
    elif var == 'gamma_beta_2':
        return W * fields['v2']
    elif var == 'gamma_beta_3':
        return W * fields['v3']
    elif var =='sp_enthalpy':
        # Specific enthalpy
        return h - 1.0  
    elif var == 'mach':
        beta2 = 1.0 - (1.0 + fields['gamma_beta']**2)**(-1)
        cs2   = fields['ad_gamma'] * fields['p'] / fields['rho'] / h
        return np.sqrt(beta2 / cs2)
    elif var == 'u1':
        return W * fields['v1']
    elif var == 'u2':
        return W * fields['v2']
    elif var == 'u3':
        return W * fields['v3']
    elif var == 'u':
        return fields['gamma_beta']

def get_colors(interval: NDArray[numpy_float], cmap: plt.cm, vmin: Optional[float] = None, vmax: Optional[float] = None) -> plt.cm:
    """
    Return array of rgba colors for a given matplotlib colormap
    
    Parameters
    -------------------------
    interval: interval range for colormarp min and max 
    cmap: the matplotlib colormap instnace
    vmin: minimum for colormap 
    vmax: maximum for colormap 
    
    Returns
    -------------------------
    arr: the colormap array generate by the user conditions
    """
    plt.Normalize(vmin, vmax)
    return cmap(interval)
    
def fill_below_intersec(x: NDArray[numpy_float], y: NDArray[numpy_float], constraint: float, color: float, axis: str) -> None:
    if axis == 'x':
        ind: int = find_nearest(x, constraint)[0]
    else:
        ind = find_nearest(y, constraint)[0]
    plt.fill_between(x[ind:],y[ind:], color=color, alpha=0.1, interpolate=True)
    
def get_file_list(inputs: str, sort: bool = False) -> Union[tuple[list[str], int], tuple[dict[int, list[str]], bool]]:
    files: list[str] = []
    file_dict: dict[int, list[str]] = {}
    dircount  = 0
    multidir = False
    isDirectory = False
    for idx, obj in enumerate(inputs):
        #check if path is a directory
        isDirectory = os.path.isdir(obj)
        
        if isDirectory:
            file_path = os.path.join(obj, '')
            if dircount == 0:
                files += sorted([file_path + f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))])
            else:
                multidir = True
                if dircount == 1:
                    file_dict[idx - 1] = files
                file_dict[idx]     = sorted([file_path + f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))])
            dircount += 1
        else:
            files += [file for file in inputs]
            break
    
    if not multidir:
        # sort by length of strings now
        if sort:
            files.sort(key=len, reverse=False)
        return files, len(files)
    else:
        any(file_dict[key].sort(key=len, reverse=False) for key in file_dict.keys())
        return file_dict, isDirectory