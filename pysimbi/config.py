from typing import Callable, Union, Tuple
from .free_arg import DynamicArg
class BaseConfig:
    dynamic_args = None 
    @property
    def initial_state(self):
        raise NotImplementedError("Your subclass need to implement the initial_state property")
    
    @property
    def coord_system(self):
        raise NotImplementedError("Your subclass needs to implement the coord_system property")
    
    @property
    def regime(self):
        raise NotImplementedError("Your subclass needs to implement the regime property")
        
    @property
    def dimensions(self):
        raise NotImplementedError("Your subclass needs to implement the dimensions property")
    
    @property
    def geometry(self):
        raise NotImplementedError("Your subclass needs to implement the geometry property")
    
    @property
    def linspace(self):
        return False
    
    @property
    def sources(self):
       return None
    
    @property
    def scalars(self):
        return 0.0 
    
    @property
    def passive_scalars(self):
        return 0.0
    
    @property
    def scale_factor(self) -> Callable:
        return None 
    
    @property
    def scale_factor_derivative(self) -> Callable:
       return None
    
    @property
    def edens_outer(self) -> Callable:
        return None 
    
    @property
    def mom_outer(self) -> Union[Callable, Tuple[Callable]]:
        return None
    
    @property
    def dens_outer(self) -> Callable:
       return None
   
    @property
    def start_time(self):
       return None
   
    @property
    def end_time(self):
       return None
   
    @property
    def use_hllc_solver(self):
       return None
   
    @property
    def boundary_condition(self):
       return None 
   
    @property
    def plm_theta(self):
        return None 
    
    @property
    def data_directory(self):
        return None
    
    @property 
    def dlogt(self):
        return None 
    
    @property
    def args(self):
        return self.dynamic_args
    
    def find_dynamic_args(self):
        """
        Find all derived class member's members defined as DynamicArg class instances 
        """
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        self.dynamic_args = [getattr(self, member) for member in members if isinstance(getattr(self,member), DynamicArg)]
        
    
    def parse_args(self, parser):
        """
        Parse extra problem-specific args from command line
        """
        if not self.dynamic_args:
            self.find_dynamic_args()
            
        for member in self.args:
            if type(member.value) == bool:
                parser.add_argument(
                    f'--{member.name}',
                    help    = member.help,
                    action  = member.action,
                    default = member.value,
                )
            else:
                parser.add_argument(
                    f'--{member.name}',
                    help    = member.help,
                    action  = member.action,
                    type    = member.var_type,
                    choices = member.choices,
                    default = member.value,
                )
                    
        args = parser.parse_args()
        # Update dynamic var attributes to reflect new values passed from cli
        for var in self.dynamic_args:
            if var.name in vars(args):
                var.value = vars(args)[var.name]
        
        self.__init__()

    def print_problem_params(self):
        """
        Read from problem params and print to stdout
        """
        if not self.dynamic_args:
            self.find_dynamic_args()
            
        print("\nProblem Parameters:", flush=True)
        print("="*80, flush=True)
        for member in self.args:
            val = member.value
            if (isinstance(val, float)):
                val = round(val, 3)
            val = str(val)
            print(f"{member.name:.<30} {val:<15} {member.help}", flush = True)
    
    def __del__(self):
        """
        Print problem params on class destruction
        """
        self.print_problem_params()