from ..theme import ThemeConfig

default_theme = ThemeConfig(
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
    color_map="viridis",
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
