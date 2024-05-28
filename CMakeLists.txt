cmake_minimum_required(VERSION 3.15...3.27)
project(PyNanoInstantMeshes)

add_compile_options(-std=gnu++17)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/native)
include_directories(
  # Eigen
  ${CMAKE_CURRENT_SOURCE_DIR}/native/ext/eigen
  # Intel Thread Building Blocks
  ${CMAKE_CURRENT_SOURCE_DIR}/native/ext/tbb/include
  # Disjoint set data structure
  ${CMAKE_CURRENT_SOURCE_DIR}/native/ext/dset
  # Parallel stable sort
  ${CMAKE_CURRENT_SOURCE_DIR}/native/ext/pss
  # Pseudorandom number generator
  ${CMAKE_CURRENT_SOURCE_DIR}/native/ext/pcg32
  # RPLY mesh I/O library
  ${CMAKE_CURRENT_SOURCE_DIR}/native/ext/rply
  # Half precision type
  ${CMAKE_CURRENT_SOURCE_DIR}/native/ext/half
)

# Nanobind Module
nanobind_add_module(PyNIM NB_STATIC src/process_numpy.cpp)

include_directories(native/im/include)
target_link_libraries(PyNIM PRIVATE InstantMeshesLib tbb_static pthread dl)
# # target_compile_features(PyNIM PRIVATE cxx_std_14)
# set_property(TARGET PyNIM PROPERTY LINKER_LANGUAGE CXX)
# set_property(TARGET PyNIM PROPERTY CXX_STANDARD 14)
