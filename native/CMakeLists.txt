cmake_minimum_required (VERSION 2.8.3)
project(InstantMeshes)

option(INSTANT_MESHES_DEV_MODE "Instant meshes developer mode" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

# Enable folders for projects in Visual Studio
if (CMAKE_GENERATOR MATCHES "Visual Studio")
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

# Sanitize build environment for static build with C++11
if (MSVC)
  add_definitions (/D "_CRT_SECURE_NO_WARNINGS")
  add_definitions (/D "__TBB_NO_IMPLICIT_LINKAGE")

  # Parallel build on MSVC (all targets)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")

  if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:SSE2")

    # Disable Eigen vectorization for Windows 32 bit builds (issues with unaligned access segfaults)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DEIGEN_DONT_ALIGN")
  endif()

  # Static build
  set(CompilerFlags
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
elseif(APPLE)
  # Try to auto-detect a suitable SDK
  # Commented out for now -- causes a too new SDK to be selected on AppVeyor
  #execute_process(COMMAND bash -c "xcodebuild -version -sdk | grep MacOSX | grep Path | head -n 1 | cut -f 2 -d ' '" OUTPUT_VARIABLE CMAKE_OSX_SYSROOT)
  #string(REGEX REPLACE "(\r?\n)+$" "" CMAKE_OSX_SYSROOT "${CMAKE_OSX_SYSROOT}")
  #string(REGEX REPLACE "^.*X([0-9.]*).sdk$" "\\1" CMAKE_OSX_DEPLOYMENT_TARGET "${CMAKE_OSX_SYSROOT}")
endif()

# Add Eigen
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ext/eigen)

# Build TBB
set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/tbb ext_build/tbb)
set_property(TARGET tbb_static tbb_def_files PROPERTY FOLDER "dependencies")

# Compile instant meshes with various compiler warnings turned on
if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gnu-anonymous-struct -Wno-c99-extensions -Wno-nested-anon-types")
  endif()
endif()

# Header file directories of dependency libraries
include_directories(
  # Eigen
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/eigen
  # Intel Thread Building Blocks
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/tbb/include
  # Disjoint set data structure
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/dset
  # Parallel stable sort
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/pss
  # Pseudorandom number generator
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/pcg32
  # RPLY mesh I/O library
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/rply
  # Half precision type
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/half
)

if (INSTANT_MESHES_DEV_MODE)
  add_definitions(-DDEV_MODE)
endif()

add_library(InstantMeshesLib
  im/include/common.h
  im/include/aabb.h
  im/include/meshio.h im/src/meshio.cpp
  im/include/bvh.h im/src/bvh.cpp
  im/include/smoothing.h im/src/smoothing.cpp
  im/include/normal.h im/src/normal.cpp
  im/include/dedge.h im/src/dedge.cpp
  im/include/batch.h im/src/batch.cpp
  im/include/subdivide.h im/src/subdivide.cpp
  im/include/meshstats.h im/src/meshstats.cpp
  im/include/hierarchy.h im/src/hierarchy.cpp
  im/include/adjacency.h im/src/adjacency.cpp
  im/include/serializer.h im/src/serializer.cpp
  im/include/field.h im/src/field.cpp
  im/include/extract.h im/src/extract.cpp
  im/include/reorder.h im/src/reorder.cpp
  im/include/cleanup.h im/src/cleanup.cpp)

target_include_directories(InstantMeshesLib PRIVATE im/include)
