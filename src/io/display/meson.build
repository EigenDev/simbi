# =============================================================================
# src/io/display/meson.build
#
# build configuration for terminal display system
# =============================================================================

# display library sources
display_sources = files(
  'renderer.cpp',
  'table.cpp',
  'terminal.cpp',
)

# display library (static, used by simbi_core)
display_lib = static_library(
  'display',
  display_sources,
  include_directories: all_inc,
  install: false,
)

# standalone demo executable (for isolated testing)
display_demo = executable(
  'display_demo',
  'display_demo.cpp',
  include_directories: all_inc,
  link_with: display_lib,
  install: false,
)
