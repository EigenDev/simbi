#! /usr/bin/env python 

# Utility functions for visualization scripts 

from math import gamma
import h5py 
import astropy.constants as const 
import astropy.units as units
import numpy as np 
import argparse 
import matplotlib.pyplot as plt 
import os
from typing import Union

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
 
rho_scale    = m / (4./3. * np.pi * R_0 ** 3) 
e_scale      = m * c **2
edens_scale  = e_scale / (4./3. * np.pi * R_0**3)
time_scale   = R_0 / c

e_scale_bmk   = 1e53 * units.erg
rho_scale_bmk = 1.0 * const.m_p.cgs / units.cm**3
ell_scale     = (e_scale_bmk / rho_scale_bmk / const.c.cgs**2)**(1/3)
t_scale       = const.c.cgs * ell_scale


def calc_rverticies(r: np.ndarray) -> np.ndarray:
    rvertices = np.sqrt(r[1:] * r[:-1])
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, r.shape, r[-1])
    return rvertices 

def calc_theta_verticies(theta: np.ndarray) -> np.ndarray:
    tvertices = 0.5 * (theta[1:] + theta[:-1])
    tvertices = np.insert(tvertices, 0, theta[0], axis=0)
    tvertices = np.insert(tvertices, tvertices.shape[0], theta[-1], axis=0)
    return tvertices 

def calc_cell_volume1D(r: np.ndarray) -> np.ndarray:
    rvertices = np.sqrt(r[1:] * r[:-1])
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, r.shape, r[-1])
    return 4.0 * np.pi * (1./3.) * (rvertices[1:]**3 - rvertices[:-1]**3)

def calc_cell_volume2D(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    tvertices = 0.5 * (theta[1:] + theta[:-1])
    tvertices = np.insert(tvertices, 0, theta[0], axis=0)
    tvertices = np.insert(tvertices, tvertices.shape[0], theta[-1], axis=0)
    dcos      = np.cos(tvertices[:-1]) - np.cos(tvertices[1:])
    
    rvertices = np.sqrt(r[:, 1:] * r[:, :-1])
    rvertices = np.insert(rvertices,  0, r[:, 0], axis=1)
    rvertices = np.insert(rvertices, rvertices.shape[1], r[:, -1], axis=1)
    dr        = rvertices[:, 1:] - rvertices[:, :-1]
    
    return (2.0 * np.pi *  (1./3.) * (rvertices[:, 1:]**3 - rvertices[:, :-1]**3) *  dcos)

def calc_enthalpy(fields: dict) -> np.ndarray:
    return 1.0 + fields['p']*fields['ad_gamma'] / (fields['rho'] * (fields['ad_gamma'] - 1.0))
    
def calc_lorentz_gamma(fields: dict) -> np.ndarray:
    return (1.0 + fields['gamma_beta']**2)**0.5

def calc_beta(fields: dict) -> np.ndarray:
    W = calc_lorentz_gamma(fields)
    return (1.0 - 1.0 / W**2)**0.5

def get_field_str(args: argparse.ArgumentParser) -> str:
    field_str_list = []
    for field in args.fields:
        if field == 'rho' or field == 'D':
            var = r'\rho' if field == 'rho' else 'D'
            if args.units:
                field_str_list.append( r'${}$ [g cm$^{{-3}}$]'.format(var))
            else:
                field_str_list.append( r'${}/{}_0$'.format(var,var))
            
        elif field == 'gamma_beta':
            field_str_list.append( r'$\Gamma \beta$')
        elif field == 'gamma_beta_1':
            field_str_list.append( r'$\Gamma \beta_1$')
        elif field == 'gamma_beta_2':
            field_str_list.append( r'$\Gamma \beta_2$')
        elif field == 'energy' or field == 'p':
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
                field_str_list.append( r'$\tau + D \  [\rm erg \ cm^{-3}]$')
            else:
                field_str_list.append( r'$\tau + D')
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
        elif field == 'v':
            field_str_list.append('$v / v_0$')
        elif field == 'u1':
            field_str_list.append(r'$\Gamma \beta_1$')
        elif field == 'u2':
            field_str_list.append(r'$\Gamma \beta_2$')
        else:
            field_str_list.append(field)

    return field_str_list if len(args.fields) > 1 else field_str_list[0]

def calc_bfield_shock(fields: dict, eb: float = 0.1) -> np.ndarray:
    W = calc_lorentz_gamma(fields)
    comoving_density = fields['rho'] * W *  rho_scale
    return (32 * np.pi *  eb * comoving_density)**0.5 * W * const.c.cgs 

def unpad(arr, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return arr[tuple(slices)]

def flatten_fully(x):
    if any(dim == 1 for dim in x.shape):
        x = np.vstack(x)
        if len(x.shape) == 2 and x.shape[0] == 1:
            return x.flatten()
        return flatten_fully(x)
    else:
        return np.asarray(x) 
    
def get_dimensionality(files: list[str]) -> int:
    dims = []
    all_equal = lambda x: x.count(x[0]) == len(x)
    for file in files:
        with h5py.File(file, 'r') as hf:
            ds  = hf.get('sim_info')
            try:
                ndim = ds.attrs['dimensions']
            except KeyError:
                rho = hf.get('rho')[:]
                nx  = ds.attrs['nx'] or 1
                ny  = ds.attrs['ny'] if 'ny' in ds.attrs.keys() else 1
                nz  = ds.attrs['nz'] if 'nz' in ds.attrs.keys() else 1
                rho = rho.reshape(nz, ny, nx)
                rho = flatten_fully(rho)
                ndim = rho.ndim
            dims += [ndim]
            if not all_equal(dims):
                raise ValueError("All simulation files require identical dimensionality")
    
    return ndim
        
            
def read_file(args: argparse.ArgumentParser, filename: str, ndim: int) -> tuple[dict,dict,dict]:
    setup  = {}
    with h5py.File(filename, 'r') as hf: 
        ds  = hf.get('sim_info')
        rho = hf.get('rho')[:]
        v   = [(hf.get(f'v{dim}') or hf.get(f'v'))[:] for dim in range(1,ndim + 1)]
        p   = hf.get('p')[:]         
        chi = (hf.get('chi') or np.zeros_like(rho))[:]

        bcs = hf.get('boundary_conditions')[:]
        full_periodic = all(bc.decode("utf-8") == 'periodic' for bc in bcs)
        
        setup['time']          = ds.attrs['current_time']
        setup['linspace']      = ds.attrs['linspace']
        setup['regime']        = ds.attrs['regime'].decode("utf-8")
        setup['coord_system']  = ds.attrs['geometry'].decode('utf-8')
        setup['is_cartesian']  = setup['coord_system'] in logically_cartesian
        setup['x1']            = hf.get('x1')[:]
        setup['x2']            = (hf.get('x2') or np.zeros_like(setup['x1']))[:]
        setup['x3']            = (hf.get('x3') or np.zeros_like(setup['x1']))[:]
        setup['first_order']   = ds.attrs['first_order']
        setup['mesh_motion']   = ds.attrs['mesh_motion']
        nx = ds.attrs['nx'] or 1
        ny = ds.attrs['ny'] if 'ny' in ds.attrs.keys() else 1
        nz = ds.attrs['nz'] if 'nz' in ds.attrs.keys() else 1

        gamma       = ds.attrs['adiabatic_gamma']
        coord_sysem = ds.attrs['geometry'].decode('utf-8')
        
        rho = flatten_fully(rho.reshape(nz, ny, nx))
        v   = [flatten_fully(vel.reshape(nz, ny, nx)) for vel in v]
        p   = flatten_fully(p.reshape(nz, ny, nx))
        chi = flatten_fully(chi.reshape(nz, ny, nx))
        
        npad = tuple(tuple(val) for val in [[2 * (setup['first_order'] + 1), 2 * (setup['first_order'] + 1)]] * ndim) 
        rho  = unpad(rho, npad)
        v    = np.asarray([unpad(vel, npad) for vel in v])
        p    = unpad(p, npad)
        chi  = unpad(chi, npad)
        
        fields = {f'v{i+1}': v[i] for i in range(len(v))}
        fields['rho']          = rho
        fields['p']            = p
        fields['chi']          = chi
        fields['ad_gamma']     = gamma
        
        vsqr = np.sum(vel * vel for vel in v)
        W    = 1/np.sqrt(1.0 - vsqr) if setup['regime'] == 'relativistic' else 1
        fields['gamma_beta']   = np.sqrt(vsqr) * W 
    
    mesh = {}
    if ndim == 1:
        mesh['x1'] = setup['x1']
    elif ndim == 2:
        mesh['x1'], mesh['x2'] = np.meshgrid(setup['x1'], setup['x2'])
    else:
        mesh['x3'], mesh['x2'], mesh['x1'] = np.meshgrid(setup['x3'], setup['x2'], setup['x1'], indexing='ij')
    return fields, setup, mesh 

def prims2var(fields: dict, var: str) -> np.ndarray:
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
    elif var == 'gamma_beta':
        return W * fields['v']
    elif var == 'chi_dens':
        return fields['chi'] * fields['rho'] * W
    elif var == 'gamma_beta_1':
        return W * fields['v1']
    elif var == 'gamma_beta_2':
        return W * fields['v2']
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

def get_colors(interval: np.ndarray, cmap: plt.cm, vmin: float = None, vmax: float = None):
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
    norm = plt.Normalize(vmin, vmax)
    return cmap(interval)

def find_nearest(arr: list, val: float) -> Union[int, float]:
    arr = np.asarray(arr)
    idx = np.argmin(np.abs(arr - val))
    return idx, arr[idx]
    
def fill_below_intersec(x: np.ndarray, y: np.ndarray, constraint: float, color: float) -> None:
    ind = find_nearest(y, constraint)[0]
    plt.fill_between(x[ind:],y[ind:], color=color, alpha=0.1, interpolate=True)
    
def get_file_list(inputs: str) -> list:
    files = []
    file_dict = {}
    dircount  = 0
    multidir = False
    for idx, obj in enumerate(inputs):
        #check if path is a file
        isFile = os.path.isfile(obj)

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
        # files.sort(key=len, reverse=False)
        return files, len(files)
    else:
        [file_dict[key].sort(key=len, reverse=False) for key in file_dict.keys()]
        return file_dict, isDirectory