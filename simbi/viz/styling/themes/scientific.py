from ..theme import ThemeConfig

scientific_theme = ThemeConfig(
    # text styling
    font_family="Times New Roman",
    font_size=10,
    title_size=12,
    label_size=10,
    text_color="black",
    # line styling
    line_styles=["-", "--", ":", "-."],
    line_width=1.2,
    # color styling
    color_map="viridis",
    # axis styling
    hide_spines=["top", "right"],
    grid=False,
    axis_below=True,
    # figure styling
    fig_size=(6, 4.5),  # standard figure size for publications
    dpi=300,
    # # LaTeX settings
    use_tex=True,
)
