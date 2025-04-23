VALID_PLOT_TYPES = ["line", "multidim", "temporal", "histogram"]

FIELD_ALIASES = {
    "Sigma": "rho",
    "b1": "b1_mean",
    "b2": "b2_mean",
    "b3": "b3_mean",
}

DERIVED = [
    "D",
    "momentum",
    "energy",
    "energy_rst",
    "enthalpy",
    "temperature",
    "T_eV",
    "mass",
    "chi_dens",
    "mach",
    "u1",
    "u2",
    "u3",
    "u",
    "tau-s",
    "ptot",
    "pmag",
    "sigma",
    "enthalpy_density",
    "mdot",
]
FIELD_CHOICES = [
    "rho",
    "v1",
    "v2",
    "v3",
    "v",
    "p",
    "gamma_beta",
    "chi",
    "b1_mean",
    "b2_mean",
    "b3_mean",
    "Sigma",
] + DERIVED

# these fields will be plotted in
# linear scale regardless of the
# log_scale setting
LINEAR_FIELDS = [
    "chi",
    "gamma_beta",
    "u1",
    "u2",
    "u3",
    "u",
    "tau-s",
    "v",
    "v1",
    "v2",
    "v3",
    "sigma",
]

LEGEND_LOCATIONS = [
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]

FONT_SIZES = {f"{x}pt": x for x in range(6, 21)}
