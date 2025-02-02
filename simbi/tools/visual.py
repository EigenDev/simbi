from .visualization.config import Config
from .visualization.core import BasePlotter
from .visualization.plotters import (
    TemporalPlotter,
    LinePlotter,
    MultidimPlotter,
    HistogramPlotter,
)
from .visualization.utils.formatting import PlotTextStyle
from .utility import get_dimensionality
from typing import Any

def create_plotter(config: Config) -> BasePlotter:
    # Create appropriate plotter with configuration
    if config['plot'].plot_type == "line":
        return LinePlotter(config)
    elif config['plot'].plot_type == "multidim":
        return MultidimPlotter(config)
    elif config['plot'].plot_type == "temporal":
        return TemporalPlotter(config)
    elif config['plot'].plot_type == "histogram":
        return HistogramPlotter(config)
    else:
        raise ValueError(f"plot type {config['plot'].plot_type} not an option")

def visualize(config: dict[str, Any]) -> None:
    """Create visualization based on parser configuration"""
    config['plot'].ndim = get_dimensionality(config['plot'].files)
    
    # if no plot type is specified, default to the dimensionality of the data
    if config['plot'].plot_type is None:
        if config['plot'].ndim == 1 or config['multidim'].slice_along is not None:
            config['plot'].plot_type = "line"
        else:
            config['plot'].plot_type = "multidim"
            
    # initialize plot text and font style from config
    PlotTextStyle(config)
    
    # Parse arguments into configuration groups
    plotter = create_plotter(config)
    
    # Execute visualization
    with plotter:
        if config['plot'].kind == "movie":
            plotter.animate()
        else:
            plotter.plot()
        
        if not config['plot'].save_as:
            plotter.show()
        else:
            plotter.save()
