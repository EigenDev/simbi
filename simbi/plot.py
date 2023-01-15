import argparse 
import sys
import matplotlib.pyplot as plt 
import importlib
from .utility import DEFAULT_SIZE, SMALL_SIZE, get_dimensionality, get_file_list
from pathlib import Path 

derived = ['D', 'momentum', 'energy', 'energy_rst', 'enthalpy', 'temperature', 'T_eV', 'mass', 'chi_dens',
          'mach', 'u1', 'u2']
field_choices = ['rho', 'v1', 'v2', 'v3', 'v', 'p', 'gamma_beta', 'chi'] + derived
lin_fields    = ['chi', 'gamma_beta', 'u1', 'u2', 'u3']

tool_src = Path(__file__).resolve().parent / 'tools'

def main(parser: argparse.ArgumentParser = None, args: argparse.Namespace = None, *_) -> None:
    if args.tex:
        plt.rc('font',   size=DEFAULT_SIZE)          # controls default text sizes
        plt.rc('axes',   titlesize=DEFAULT_SIZE)     # fontsize of the axes title
        plt.rc('axes',   labelsize=DEFAULT_SIZE)     # fontsize of the x and y labels
        plt.rc('xtick',  labelsize=DEFAULT_SIZE)     # fontsize of the tick labels
        plt.rc('ytick',  labelsize=DEFAULT_SIZE)     # fontsize of the tick labels
        plt.rc('legend', fontsize=DEFAULT_SIZE)      # legend fontsize
        plt.rc('figure', titlesize=DEFAULT_SIZE)     # fontsize of the figure title
        
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": "Times New Roman",
                "font.size": DEFAULT_SIZE
            }
        )
    
    sys.path.insert(1, f'{tool_src}')
    file_list, _  = get_file_list(args.files)
    ndim          = get_dimensionality(file_list)
    visual_module = getattr(importlib.import_module(f'{args.kind}{ndim}d'), f'{args.kind}')
    visual_module(parser)
        
if __name__ == '__main__':
    sys.exit(main(*(parse_arguments())))