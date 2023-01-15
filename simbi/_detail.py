import argparse 
from typing import Any

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def get_subparser(parser: argparse.ArgumentParser, idx: int) -> Any:
    subparser = [
        subparser 
        for action in parser._actions 
        if isinstance(action, argparse._SubParsersAction) 
        for _, subparser in action.choices.items()
    ]
    return subparser[idx]

def max_thread_count(param: Any) -> int:
    import multiprocessing
    num_threads_available = multiprocessing.cpu_count()
    
    try:
        val = int(param)
    except ValueError:    
        raise argparse.ArgumentTypeError("\nMust be a integer\n")
    
    if val > num_threads_available:
        raise argparse.ArgumentTypeError(f'\nTrying to set thread count greater than available compute core(s) equal to {num_threads_available}\n')
    
    return val

__all__ = ['bcolors', 'get_subparser', 'max_thread_count']