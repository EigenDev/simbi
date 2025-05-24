from ..theme import Theme

dark_theme = Theme(
    # Text styling
    font_family="sans-serif",
    font_size=12,
    title_size=14,
    label_size=12,
    text_color="white",
    # Line styling
    line_styles=["-", "--", ":", "-."],
    line_width=1.8,
    # Color styling
    color_maps=["plasma"],
    color_cycle=[
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#bcbd22",
        "#17becf",
        "#1f77b4",
        "#7f7f7f",
    ],
    # Axis styling
    hide_spines=[],
    grid=True,
    # Figure styling
    fig_size=(8, 6),
    dpi=100,
    # Background colors
    background_colors={
        "figure": "#1e1e1e",
        "axes": "#1e1e1e",
    },
    # LaTeX settings
    use_tex=False,
)
