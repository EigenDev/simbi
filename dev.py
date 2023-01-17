import argparse 
import sys 
import subprocess
import json
import os 
from pathlib import Path 
from typing import Optional

cache_file = "simbi_build_cache.txt"
default = {}
default['gpu_compilation']="disabled"
default['oned_bz']=128
default['twod_bz']=16  
default['thrd_bz']=4 
default['column_major']=False
default['float_precision']=False
default['install_mode']="default"  
default['dev_arch']=86 
default['build_dir']="builddir"

YELLOW='\033[0;33m'
RST='\033[0m' # No Color

flag_overrides = {} 
flag_overrides['float_precision'] = ['--double', '--float']
flag_overrides['gpu_compilation'] = ['gpu-compilation', '--cpu-compilation']
flag_overrides['column_major']    = ['--row-major', '--column-major']
flag_overrides['install_mode']    = ['develop', 'default']
def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    from shutil import which
    return which(name) is not None

def read_from_cache() -> Optional[dict[str, str]]:
    cached_args = Path(cache_file)
    if cached_args.exists():
        with open(cache_file, 'r') as f:
            data = f.read()
        return json.loads(data)
    else:
        with open(cache_file, 'w+'):
            ...
        return None 

def check_minimal_depencies():
    if not is_tool('meson'):
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'meson'], check=True)
    
    if not is_tool('cython'):
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'meson'], check=True)
        
def write_to_cache(args: argparse.Namespace) -> None:
    details = vars(args)
    details['build_dir'] = str(Path(details['build_dir']).resolve())
    details.pop('func')
    with open(cache_file, 'w') as f:
        f.write(json.dumps(details))
    
def get_output(command: str) -> str:
    return subprocess.Popen(command, stdout=subprocess.PIPE, shell=True).stdout.read().decode('utf-8').strip()

def configure(args: argparse.Namespace, 
              cxx: str, reconfigure: str, 
              hdf5_include: str, 
              gpu_include: str) -> list[str]:
    command = f'''CXX={cxx} meson setup {args.build_dir} -Dgpu_compilation={args.gpu_compilation}  
    -Dhdf5_include_dir={hdf5_include} -Dgpu_include_dir={gpu_include} \
    -D1d_block_size={args.oned_bz} -D2d_block_size={args.twod_bz} -D3d_block_size={args.thrd_bz} \
    -Dcolumn_major={args.column_major} -Dfloat_precision={args.float_precision} \
    -Dprofile={args.install_mode} -Dgpu_arch={args.dev_arch} {reconfigure}'''.split()
    return ' '.join(var for var in command)
    
def parse_the_arguments():
    parser = argparse.ArgumentParser('Parser for installing simbi with meson')
    subparsers = parser.add_subparsers(help='sub-commands that install / uninstall the code')
    install_parser = subparsers.add_parser('install', help='install simbi')
    install_parser.set_defaults(func=install_simbi)
    uninstall_parser = subparsers.add_parser('uninstall', help='uninstall simbi')
    uninstall_parser.set_defaults(func=uninstall_simbi)
    install_parser.add_argument(
        '--oned_bz',                      
        help='block size for 1D simulations (only set for gpu compilation)', 
        type=int, 
        default=128
    )
    install_parser.add_argument(
        '--twod_bz',                      
        help='block size for 2D simulations (only set for gpu compilation)', 
        type=int, 
        default=16
    )
    install_parser.add_argument(
        '--thrd_bz',                    
        help='block size for 3D simulations (only set for gpu compilation)', 
        type=int, 
        default=4
    )
    install_parser.add_argument(
        '--dev-arch',                     
        help='SM architecture specification for gpu compilation', 
        type=int, 
        default=86
    )
    install_parser.add_argument(
        '--verbose','-v',                 
        help='flag for verbose compilation output', 
        action='store_true', 
        default=False
    )
    install_parser.add_argument(
        '--configure',                    
        help='flag to only configure the meson build directory without installing', 
        action='store_true',
        default=False
    )
    install_parser.add_argument(
        '--install-mode',                 
        help='install mode (normal or editable)',
        default='default',
        type=str,
        choices=['default', 'develop']
    )
    install_parser.add_argument(
        '--build-dir',
        help='build directory name for meson build',
        type=str,
        default='builddir',
    )
    compile_type = install_parser.add_mutually_exclusive_group()
    compile_type.add_argument('--gpu-compilation',  action='store_const', dest='gpu_compilation', const='enabled')
    compile_type.add_argument('--cpu-compilation',  action='store_const', dest='gpu_compilation', const='disabled')
    precision = install_parser.add_mutually_exclusive_group()
    precision.add_argument('--double',  action='store_const', dest='float_precision', const=False)
    precision.add_argument('--float',   action='store_const', dest='float_precision', const=True)
    major = install_parser.add_mutually_exclusive_group()
    major.add_argument('--row-major',   action='store_const', dest='column_major', const=False)
    major.add_argument('--column-major',action='store_const', dest='column_major', const=True)
    install_parser.set_defaults(float_precision=False, column_major=False, gpu_compilation='disabled')
    return parser, parser.parse_args(args=None if sys.argv[1:] else ['--help'])

def install_simbi(args: argparse.Namespace) -> None:
    simbi_dir = Path().resolve()
    if args.build_dir == 'build':
        raise argparse.ArgumentError(args.builddir, "please choose a different build name other than 'build'")

    # Check if any args passed to the cli exist that would override the cache args
    cli_args = sys.argv[1:]
    if cached_vars := read_from_cache():
        for arg in vars(args):
            if arg not in ['verbose', 'configure', 'func']:
                if getattr(args,arg) == default[arg]:
                    if arg in flag_overrides.keys():
                        if any(x in flag_overrides[arg] for x in cli_args):
                            continue
                    elif arg in cli_args:
                        continue
                    else:
                        setattr(args, arg, cached_vars[arg])
    write_to_cache(args)
    
    simbi_build_dir = args.build_dir 
    configure_only = False
    try:
        subprocess.check_call([
            "meson", 
            "introspect", 
            f"{args.build_dir}", 
            "-i",
            "--targets"], stdout=subprocess.DEVNULL)
        build_configured = True
    except subprocess.CalledProcessError:
        build_configured = False
    
    reconfigure_flag = ''
    if build_configured:
        reconfigure_flag = '--reconfigure'
    
    verbose_flag = '--verbose' if args.verbose else ''
    if 'CXX' not in os.environ: 
        cxx = get_output('echo $(command -v c++)')
        print(f"{YELLOW}WRN{RST}: C++ compiler not set")
        print(f"Using symbolic link {cxx} as default")
    else:
        cxx = os.environ['CXX']
    
    gpu_runtime_dir=''
    if is_tool('nvcc'):
        gpu_runtime_command=r'echo $PATH | sed "s/:/\n/g" | grep "cuda/bin" | sed "s/\/bin//g" |  head -n 1'
    elif is_tool('hipcc'):
        gpu_runtime_command=['hipconfig', '--rocmpath'] 
    ''
    gpu_runtime_dir = get_output(gpu_runtime_command)
    gpu_include=f"{gpu_runtime_dir}/include"
    hdf5_path_command   =r'echo $(command -v h5cc) | sed "s/:/\n/g" | grep "bin/h5cc" | sed "s/\/bin//g;s/\/h5cc//g" |  head -n 1'
    hdf5_path = get_output(hdf5_path_command)
    hdf5_include_command=f'( dirname $(find {hdf5_path} -iname "H5Cpp.h" -type f 2>/dev/null -print -quit ) )'
    hdf5_include = get_output(hdf5_include_command)
    
    check_minimal_depencies()
    config_command = configure(args, cxx, reconfigure_flag, None, None, hdf5_include, gpu_include)
    subprocess.run(config_command, shell=True)
    if not args.configure:
        build_dir=f"{simbi_dir}/build"
        egg_dir  =f"{simbi_dir}/simbi.egg-info"
        install_command = f'cd {args.build_dir} && meson compile {verbose_flag} && meson install' 
        subprocess.run(install_command, check=True, shell=True)
        subprocess.run(f'rm -rf {egg_dir} {build_dir}', check=True, shell=True)
        
def uninstall_simbi(args: argparse.Namespace) -> None:
    if (config_cache := read_from_cache()):
        build_dir = config_cache['build_dir']
    else:
        build_dir = args.build_dir
        
    subprocess.run(['ninja', '-C', f'{build_dir}', 'uninstall'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'simbi'], check=True)
    
def main():
    parser, args = parse_the_arguments()
    args.func(args)
    
if __name__ == '__main__':
    sys.exit(main())