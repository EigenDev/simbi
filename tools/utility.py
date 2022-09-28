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

#================================
#   constants of nature
#================================
R_0 = const.R_sun.cgs 
c   = const.c.cgs
m   = const.M_sun.cgs
 
rho_scale  = m / (4./3. * np.pi * R_0 ** 3) 
e_scale    = m * c **2
edens_scale  = e_scale / (4./3. * np.pi * R_0**3)
time_scale = R_0 / c

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
        elif field == 'T_ev':
            field_str_list.append("T [eV]" if args.units else "T")
        elif field == 'temperature':
            field_str_list.append("T [K]" if args.units else "T")
        elif field == 'mach':
            field_str_list.append('M')
        else:
            field_str_list.append(field)

    return field_str_list if len(args.fields) > 1 else field_str_list[0]

def calc_lorentz_gamma(fields: dict) -> np.ndarray:
    return (1.0 + fields['gamma_beta']**2)**0.5

def calc_beta(fields: dict) -> np.ndarray:
    W = calc_lorentz_gamma(fields)
    return (1.0 - 1.0 / W**2)**0.5

def calc_bfield_shock(fields: dict, eb: float = 0.1) -> np.ndarray:
    W = calc_lorentz_gamma(fields)
    comoving_density = fields['rho'] * W *  rho_scale
    return (32 * np.pi *  eb * comoving_density)**0.5 * W * const.c.cgs 

def read_3d_file(args: argparse.ArgumentParser, filename: str) -> Union[dict,dict,dict]:
    setup  = {}
    fields = {}
    is_cartesian = False
    with h5py.File(filename, 'r') as hf: 
        ds  = hf.get('sim_info')
        rho = hf.get('rho')[:]
        v1  = hf.get('v1')[:]
        v2  = hf.get('v2')[:]
        v3  = hf.get('v3')[:]
        p   = hf.get('p')[:]
        chi = hf.get('chi')[:]
        t   = ds.attrs['current_time']
        
        x1max = ds.attrs['x1max']
        x1min = ds.attrs['x1min']
        x2max = ds.attrs['x2max']
        x2min = ds.attrs['x2min']
        x3min = ds.attrs['x3min']
        x3max = ds.attrs['x3max']
        
        # New checkpoint files, so check if new attributes were
        # implemented or not
        nx = ds.attrs['nx']
        ny = ds.attrs['ny']
        nz = ds.attrs['nz']

        gamma = ds.attrs['adiabatic_gamma']
            
        try:
            coord_sysem = ds.attrs['geometry'].decode('utf-8')
        except Exception as e:
            coord_sysem = 'spherical'
            
        try:
            is_linspace = ds.attrs['linspace']
        except:
            is_linspace = False
        
        setup['x1max'] = x1max 
        setup['x1min'] = x1min 
        setup['x2max'] = x2max 
        setup['x2min'] = x2min 
        setup['x3min'] = x3min 
        setup['x3max'] = x3max
        setup['time']  = t
        
        rho = rho.reshape(nz, ny, nx)
        v1  = v1.reshape(nz, ny, nx)
        v2  = v2.reshape(nz, ny, nx)
        v3  = v3.reshape(nz, ny, nx)
        p   = p.reshape(nz, ny, nx)
        chi = chi.reshape(nz, ny, nx)
        
        
        if args.forder:
            rho = rho[1:-1, 1: -1, 1: -1]
            v1  = v1 [1:-1, 1: -1, 1: -1]
            v2  = v2 [1:-1, 1: -1, 1: -1]
            v3  = v3 [1:-1, 1: -1, 1: -1]
            p   = p  [1:-1, 1: -1, 1: -1]
            chi = chi[1:-1, 1: -1, 1: -1]
            xactive = nx - 2
            yactive = ny - 2
            zactive = nz - 2
            setup['xactive'] = xactive
            setup['yactive'] = yactive
            setup['zactive'] = zactive
        else:
            rho = rho[2:-2, 2: -2, 2:-2]
            v1  = v1 [2:-2, 2: -2, 2:-2]
            v2  = v2 [2:-2, 2: -2, 2:-2]
            v3  = v3 [2:-2, 2: -2, 2:-2]
            p   = p  [2:-2, 2: -2, 2:-2]
            chi = chi[2:-2, 2: -2, 2:-2]
            xactive = nx - 4
            yactive = ny - 4
            zactive = nz - 4
            setup['xactive'] = xactive
            setup['yactive'] = yactive
            setup['zactive'] = zactive
        
        if is_linspace:
            setup['x1'] = np.linspace(x1min, x1max, xactive)
            setup['x2'] = np.linspace(x2min, x2max, yactive)
            setup['x3'] = np.linspace(x3min, x3max, zactive)
        else:
            setup['x1'] = np.logspace(np.log10(x1min), np.log10(x1max), xactive)
            setup['x2'] = np.linspace(x2min, x2max, yactive)
            setup['x3'] = np.linspace(x3min, x3max, zactive)
        
        if coord_sysem == 'cartesian':
            is_cartesian = True
        
        W = 1/np.sqrt(1.0 -(v1**2 + v2**2 + v3**2))
        beta = np.sqrt(v1**2 + v2**2 + v3**2)
        
        fields['rho']          = rho
        fields['v1']           = v1 
        fields['v2']           = v2 
        fields['v3']           = v3
        fields['p']            = p
        fields['chi']          = chi
        fields['gamma_beta']   = W*beta
        fields['ad_gamma']     = gamma
        setup['is_cartesian']  = is_cartesian
        
        
    znpts, ynpts, xnpts = rho.shape 
    
    mesh = {}
    if setup['is_cartesian']:
        yy, zz, xx = np.meshgrid(setup['x2'], setup['x3'], setup['x1'])
        mesh['xx'] = xx
        mesh['yy'] = yy
        mesh['zz'] = zz
    else:  
        tt, phii, rr = np.meshgrid(setup['x2'], setup['x3'], setup['x1'])
        mesh['theta'] = tt 
        mesh['rr']    = rr
        mesh['phii']  = phii
        mesh['r']     = setup['x1']
        mesh['th']    = setup['x2']
        mesh['phi']   = setup['x3']
        
    return fields, setup, mesh 

def read_2d_file(args: argparse.ArgumentParser, filename: str) -> Union[dict,dict,dict]:
    setup  = {}
    fields = {}
    is_cartesian = False
    with h5py.File(filename, 'r') as hf: 
        ds          = hf.get('sim_info')
        rho         = hf.get('rho')[:]
        v1          = hf.get('v1')[:]
        v2          = hf.get('v2')[:]
        p           = hf.get('p')[:]
        t           = ds.attrs['current_time']
        
        try:
            x1max = ds.attrs['x1max']
            x1min = ds.attrs['x1min']
            x2max = ds.attrs['x2max']
            x2min = ds.attrs['x2min']
        except:
            x1max = ds.attrs['xmax']
            x1min = ds.attrs['xmin']
            x2max = ds.attrs['ymax']
            x2min = ds.attrs['ymin']  
        
        # New checkpoint files, so check if new attributes were
        # implemented or not
        try:
            nx          = ds.attrs['nx']
            ny          = ds.attrs['ny']
        except:
            nx          = ds.attrs['NX']
            ny          = ds.attrs['NY']
        
        try:
            chi = hf.get('chi')[:]
        except:
            chi = np.zeros((ny, nx))
            
        try:
            gamma = ds.attrs['adiabatic_gamma']
        except:
            gamma = 4./3.
        
        # Check for garbage value
        if gamma < 1:
            gamma = 4./3. 
            
        try:
            coord_sysem = ds.attrs['geometry'].decode('utf-8')
        except Exception as e:
            coord_sysem = 'spherical'
            
        try:
            is_linspace = ds.attrs['linspace']
        except:
            is_linspace = False
        
        setup['x1max'] = x1max 
        setup['x1min'] = x1min 
        setup['x2max'] = x2max 
        setup['x2min'] = x2min 
        setup['time']  = t
        
        rho = rho.reshape(ny, nx)
        v1  = v1.reshape(ny, nx)
        v2  = v2.reshape(ny, nx)
        p   = p.reshape(ny, nx)
        chi = chi.reshape(ny, nx)
        
        
        if args.forder:
            rho = rho[1:-1, 1: -1]
            v1  = v1 [1:-1, 1: -1]
            v2  = v2 [1:-1, 1: -1]
            p   = p  [1:-1, 1: -1]
            chi = chi[1:-1, 1: -1]
            xactive = nx - 2
            yactive = ny - 2
            setup['xactive'] = xactive
            setup['yactive'] = yactive
        else:
            rho = rho[2:-2, 2: -2]
            v1  = v1 [2:-2, 2: -2]
            v2  = v2 [2:-2, 2: -2]
            p   = p  [2:-2, 2: -2]
            chi = chi[2:-2, 2: -2]
            xactive = nx - 4
            yactive = ny - 4
            setup['xactive'] = xactive
            setup['yactive'] = yactive
        
        if is_linspace:
            setup['x1'] = np.linspace(x1min, x1max, xactive)
            setup['x2'] = np.linspace(x2min, x2max, yactive)
        else:
            setup['x1'] = np.logspace(np.log10(x1min), np.log10(x1max), xactive)
            setup['x2'] = np.linspace(x2min, x2max, yactive)
        
        if coord_sysem == 'cartesian':
            is_cartesian = True
        
        # if (v1**2 + v2**2).any() >= 1.0:
        #     W = 0
        # else:
        #     W = 1/np.sqrt(1.0 -(v1**2 + v2**2))
        W = 1/np.sqrt(1.0 -(v1**2 + v2**2))
        beta = np.sqrt(v1**2 + v2**2)
        
        fields['rho']          = rho
        fields['v1']           = v1 
        fields['v2']           = v2 
        fields['p']            = p
        fields['chi']          = chi
        fields['gamma_beta']   = W*beta
        fields['ad_gamma']     = gamma
        setup['is_cartesian']  = is_cartesian
        
        
    ynpts, xnpts = rho.shape 
    mesh = {}
    if setup['is_cartesian']:
        xx, yy = np.meshgrid(setup['x1'], setup['x2'])
        mesh['xx'] = xx
        mesh['yy'] = yy
    else:      
        rr, tt = np.meshgrid(setup['x1'], setup['x2'])
        mesh['theta'] = tt 
        mesh['rr']    = rr
        mesh['r']     = setup['x1']
        mesh['th']    = setup['x2']
        
    return fields, setup, mesh 

def read_1d_file(filename: str) -> dict:
    is_linspace = False
    ofield = {}
    setups = {}
    mesh   = {}
    with h5py.File(filename, 'r') as hf:
        ds = hf.get('sim_info')
        
        rho         = hf.get('rho')[:]
        v           = hf.get('v')[:]
        p           = hf.get('p')[:]
        nx          = ds.attrs['Nx']
        t           = ds.attrs['current_time']
        try:
            x1max = ds.attrs['x1max']
            x1min = ds.attrs['x1min']
        except:
            x1max = ds.attrs['xmax']
            x1min = ds.attrs['xmin']

        try:
            is_linspace = ds.attrs['linspace']
        except:
            is_linspace = False
            
        rho = rho[2:-2]
        v   = v  [2:-2]
        p   = p  [2:-2]
        xactive = nx - 4
        
        try:
            using_gamma_beta = ds.attrs['using_gamma_beta']
        except:
            using_gamma_beta = False
            
        if using_gamma_beta:
            W = np.sqrt(1 + v * v)
            gamma_beta = v
        else:
            W = 1/np.sqrt(1 - v**2)
            gamma_beta = W * v
        
        a    = (4 * const.sigma_sb.cgs / c)
        k    = const.k_B.cgs
        T    = (3 * p * edens_scale  / a)**(1./4.)
        T_eV = (k * T).to(units.eV)
        
        h = 1.0 + 4/3 * p / (rho * (4/3 - 1))
        
        if is_linspace:
            mesh['r'] = np.linspace(x1min, x1max, xactive)
        else:
            mesh['r'] = np.logspace(np.log10(x1min), np.log10(x1max), xactive)
            
        setups['ad_gamma']    = 4./3.
        setups['time']        = t
        setups['linspace']    = is_linspace
        ofield['ad_gamma']    = 4./3.
        ofield['rho']         = rho
        ofield['v']           = v
        ofield['p']           = p
        ofield['W']           = W
        ofield['enthalpy']    = h
        ofield['gamma_beta']  = gamma_beta
        ofield['temperature'] = T_eV
        mesh['xlims']         = x1min, x1max
        
    return ofield, setups, mesh

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
    elif var == 'T_ev':
        a    = (4.0 * const.sigma_sb.cgs / c)
        T    = (3.0 * fields['p'] * edens_scale  / a)**0.25
        T_eV = (const.k_B.cgs * T).to(units.eV)
        return T_eV.value
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
            files += sorted([file for file in inputs])
            break
    
    if not multidir:
        # sort by length of strings now
        files.sort(key=len, reverse=False)
        return files, len(files)
    else:
        [file_dict[key].sort(key=len, reverse=False) for key in file_dict.keys()]
        return file_dict, isDirectory