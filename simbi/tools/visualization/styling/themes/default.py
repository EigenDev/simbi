from ..theme import Theme

default_theme = Theme(
    # Text styling
    font_family="serif",
    font_size=12,
    title_size=14,
    label_size=12,
    text_color="black",
    # Line styling
    line_styles=["-", "--", ":", "-."],
    line_width=1.5,
    # Color styling
    color_maps=["viridis"],
    color_cycle=[
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
    # Axis styling
    hide_spines=["top", "right"],
    grid=False,
    # Figure styling
    fig_size=(8, 6),
    dpi=300,
    # Polar styling
    polar_style={
        "grid": False,
        "zero_location": "N",
        "direction": -1,
        "show_ticks": True,
    },
    # LaTeX settings
    use_tex=False,
)
