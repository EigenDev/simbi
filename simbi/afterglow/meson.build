# Radiation module sources
rad_src = files(
    'src/rad_units.cpp',
    'src/rad_units.hpp',
    'src/units/units.hpp'
)
# Create the Python extension for the radiation module
py3.extension_module(
    'rad_hydro',
    ['src/rad_hydro.pyx'] + rad_src,
    include_directories: include_directories('src', incdir_numpy),
    dependencies: [
        py3.dependency(),
        dependency('openmp', language: 'cpp')
    ],
    override_options: ['cython_language=cpp'],
    install: true,
    install_dir: meson.current_source_dir() + '/libs'
)