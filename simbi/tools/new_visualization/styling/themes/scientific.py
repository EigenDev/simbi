from ..theme import Theme

scientific_theme = Theme(
    # Text styling
    font_family="Times New Roman",
    font_size=10,
    title_size=12,
    label_size=10,
    text_color="black",
    # Line styling
    line_styles=["-", "--", ":", "-."],
    line_width=1.2,
    # Color styling
    color_maps=["viridis"],
    # Axis styling
    hide_spines=["top", "right"],
    grid=False,
    axis_below=True,
    # Figure styling
    fig_size=(6, 4.5),  # Standard figure size for publications (I think)
    dpi=300,
    # # LaTeX settings
    use_tex=True,
)
