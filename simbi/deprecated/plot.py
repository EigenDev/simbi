import matplotlib.pyplot as plt
from ..tools import visual
from ..tools.utility import (
    BIGGER_SIZE,
    DEFAULT_SIZE,
    get_dimensionality,
    get_file_list,
)


def main() -> None:
    plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    if args.tex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": "Times New Roman",
                "font.size": BIGGER_SIZE,
                "text.color": args.font_color,
                "axes.labelcolor": args.font_color,
                "xtick.color": args.font_color,
                "ytick.color": args.font_color,
                "axes.edgecolor": args.font_color,
            }
        )

        if args.print:
            plt.rcParams.update(
                {
                    "legend.fontsize": DEFAULT_SIZE,
                }
            )
            
    file_list, _ = get_file_list(args.files)
    ndim = get_dimensionality(file_list)

    visual.visualize(parser, ndim)
