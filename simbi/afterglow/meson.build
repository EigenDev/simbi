# Radiation module sources
rad_src = files(
  'bindings/binding.cpp',
  'src/photon_event_io.cpp',
  'src/rad.cpp',
)
# Create the Python extension for the radiation module
py3.extension_module(
  'rad_hydro',
  rad_src,
  include_directories: include_directories('src'),
  dependencies: [
    py3.dependency(),
    openmp_dep,
    pybind11_dep,
    hdf5_dep,
  ],
  install: true,
  install_dir: py_install_dir,
)
