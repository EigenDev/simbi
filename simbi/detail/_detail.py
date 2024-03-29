import argparse 
from typing import Any

class bcolors:
    HEADER        = '\033[95m'
    OKBLUE        = '\033[94m'
    OKCYAN        = '\033[96m'
    OKGREEN       = '\033[92m'
    WARNING       = '\033[93m'
    FAIL          = '\033[91m'
    ENDC          = '\033[0m'
    BOLD          = '\033[1m'
    UNDERLINE     = '\033[4m'
    BLACK         = '\033[0;30m'     
    DARK_GRAY     = '\033[1;30m'
    RED           = '\033[0;31m'     
    LIGHT_RED     = '\033[1;31m'
    GREEN         = '\033[0;32m'     
    LIGHT_GREEN   = '\033[1;32m'
    ORANGE        = '\033[0;33m'     
    YELLOW        = '\033[1;33m'
    BLUE          = '\033[0;34m'     
    LIGHT_BLUE    = '\033[1;34m'
    PURPLE        = '\033[0;35m'     
    LIGHT_PURPLE  = '\033[1;35m'
    CYAN          = '\033[0;36m'     
    LIGHT_CYAN    = '\033[1;36m'
    LIGHT_GRAY    = '\033[0;37m'     
    WHITE         = '\033[1;37m'
    

class ParseKVAction(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, 
                 namespace: argparse.Namespace, 
                 values: Any, 
                 option_string: str | None = None) -> None:
        
        setattr(namespace, self.dest, dict())
        for each in values:
            try:
                key, value = each.split("=")
                getattr(namespace, self.dest)[key] = value
            except ValueError as ex:
                message = "\nTraceback: {}".format(ex)
                message += "\nError on '{}' || It should be 'key=value'".format(each)
                raise argparse.ArgumentError(self, str(message))
            
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

__all__ = ['bcolors', 'get_subparser', 'max_thread_count', 'ParseKVAction']