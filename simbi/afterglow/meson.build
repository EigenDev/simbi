# Radiation module sources
rad_src = files(
    'src/rad_units.cpp',
    'src/rad_units.hpp',
    'src/units/units.hpp',
    'pybind11/binding.cpp',
)
# Create the Python extension for the radiation module
py3.extension_module(
    'rad_hydro',
    rad_src,
    include_directories: include_directories('src', numpy_include),
    dependencies: [
        py3.dependency(),
        openmp_dep,
        pybind11_dep,
    ],
    install: true,
    install_dir: py_install_dir,
)
