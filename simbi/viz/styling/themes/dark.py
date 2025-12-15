from ..theme import ThemeConfig

dark_theme = ThemeConfig(
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
    color_map="plasma",
    # Axis styling
    hide_spines=[],
    grid=True,
    # Figure styling
    fig_size=(8, 6),
    dpi=300,
    # Background colors
    background_colors={
        "figure": "#1e1e1e",
        "axes": "#1e1e1e",
    },
    # LaTeX settings
    use_tex=False,
)
